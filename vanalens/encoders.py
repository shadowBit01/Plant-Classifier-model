"""Lane L2 (encoders, the ONLY lane that runs GPU jobs). Tests: tests/test_encoders.py (GPU-gated).

Models (frozen, fp16, fits 6 GB):
  BioCLIP 2: open_clip.create_model_and_transforms(config.BIOCLIP_MODEL) -> (model, _, preprocess_val)
  DINOv3:    transformers AutoImageProcessor / AutoModel(config.DINO_MODEL); use .pooler_output
             (weights are gated: accept the license on Hugging Face, then `huggingface-cli login`;
              fallback while access is pending: facebook/dinov2-large, also 1024-d, so the contract holds)
CLI:
  python -m vanalens.encoders --manifest $VANALENS_CACHE/manifests/dimpsar_gallery.jsonl \
                              --out $VANALENS_CACHE/emb/dimpsar_gallery.npz --batch-size 32
Writes contracts.EmbeddingSet (ids in manifest order, y from Sample.label). Skips unreadable images with a
warning (log their ids). Resume-safe: write shards every 2,000 images, merge at the end.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import time
import warnings
from pathlib import Path

import numpy as np
import open_clip
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModel

from concurrent.futures import Future, ThreadPoolExecutor
import math
from torchvision import transforms as T
from torchvision.transforms import functional as TF

from . import config, contracts

logger = logging.getLogger(__name__)

DINO_FALLBACK_MODEL = "facebook/dinov2-large"

_SAFETENSORS_DTYPE_MAP = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "BOOL": torch.bool,
}


def _renorm_blocks(X: np.ndarray) -> np.ndarray:
    """L2-normalise the BioCLIP and DINO blocks of (N, EMB_DIM) separately in float32."""
    if X.size == 0:
        return np.zeros((0, config.EMB_DIM), dtype=np.float32)
    out = np.empty((X.shape[0], config.EMB_DIM), dtype=np.float32)
    bio = X[:, : config.BIOCLIP_DIM].astype(np.float32, copy=False)
    dino = X[:, config.BIOCLIP_DIM :].astype(np.float32, copy=False)
    bio_norm = np.linalg.norm(bio, axis=1, keepdims=True)
    dino_norm = np.linalg.norm(dino, axis=1, keepdims=True)
    out[:, : config.BIOCLIP_DIM] = bio / np.maximum(bio_norm, 1e-12)
    out[:, config.BIOCLIP_DIM :] = dino / np.maximum(dino_norm, 1e-12)
    return out


def _random_resized_crop_pil(
    rgb: Image.Image,
    rng: np.random.Generator,
    size: int = 224,
    scale: tuple[float, float] = (0.6, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> tuple[Image.Image, bool]:
    """Deterministic RandomResizedCrop(size, scale=(0.6, 1.0)) + RandomHorizontalFlip using numpy Generator."""
    width, height = rgb.size
    area = float(height * width)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))

    crop_box = None
    for _ in range(10):
        target_area = area * float(rng.uniform(scale[0], scale[1]))
        aspect_ratio = math.exp(float(rng.uniform(log_ratio[0], log_ratio[1])))
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        if 0 < w <= width and 0 < h <= height:
            top = int(rng.integers(0, height - h + 1))
            left = int(rng.integers(0, width - w + 1))
            crop_box = (left, top, left + w, top + h)
            break

    if crop_box is None:
        in_ratio = float(width) / float(max(height, 1))
        if in_ratio < ratio[0]:
            w = width
            h = max(1, int(round(w / ratio[0])))
        elif in_ratio > ratio[1]:
            h = height
            w = max(1, int(round(h * ratio[1])))
        else:
            w, h = width, height
        top = max(0, (height - h) // 2)
        left = max(0, (width - w) // 2)
        crop_box = (left, top, left + w, top + h)

    cropped = rgb.crop(crop_box).resize((size, size), Image.Resampling.BICUBIC)
    do_flip = bool(rng.integers(0, 2))
    if do_flip:
        cropped = cropped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return cropped, do_flip


_DINOV2_KEY_REMAP = (
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
)


def _remap_checkpoint_key(key: str, targets: dict[str, torch.Tensor]) -> str | None:
    """Resolve a checkpoint tensor key against model targets, applying DINOv2 attention key remaps if needed."""
    candidates = [key]
    remapped = key
    for old, new in _DINOV2_KEY_REMAP:
        if old in remapped:
            remapped = remapped.replace(old, new)
    if remapped != key:
        candidates.append(remapped)

    for cand in candidates:
        if cand in targets:
            return cand
        if "." in cand:
            stripped = cand.split(".", 1)[1]
            if stripped in targets:
                return stripped
    return None


def _stream_safetensors_into_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device,
    fp16: bool,
) -> dict:
    """Stream tensors from a .safetensors file directly into model parameters on device without full-file mmap.

    Raises RuntimeError if any model parameter or persistent buffer is left unloaded.
    """
    if fp16:
        model.half()
    model.to(device)

    non_persistent: set[str] = set()
    for mod_name, mod in model.named_modules():
        for buf_name in getattr(mod, "_non_persistent_buffers_set", ()):
            non_persistent.add(f"{mod_name}.{buf_name}" if mod_name else buf_name)

    params = dict(model.named_parameters())
    buffers = {k: v for k, v in model.named_buffers() if k not in non_persistent}
    targets = {**params, **buffers}
    loaded_keys: set[str] = set()

    with open(checkpoint_path, "rb") as f, torch.no_grad():
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base_offset = 8 + header_len
        for key, info in header.items():
            if key == "__metadata__":
                continue
            target_key = _remap_checkpoint_key(key, targets)
            if target_key is None:
                continue
            target = targets[target_key]
            start, end = info["data_offsets"]
            f.seek(base_offset + start)
            buf = bytearray(f.read(end - start))
            tensor = torch.frombuffer(buf, dtype=_SAFETENSORS_DTYPE_MAP[info["dtype"]]).reshape(info["shape"])
            target.data.copy_(tensor.reshape(target.shape).to(device=target.device, dtype=target.dtype))
            loaded_keys.add(target_key)

    unloaded = sorted(set(targets) - loaded_keys)
    if unloaded:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} left {len(unloaded)}/{len(targets)} model tensors unloaded "
            f"(first 5: {unloaded[:5]})"
        )
    return {}


class _ImagePathDataset(Dataset):
    """Dataset that loads and RGB-converts images from paths for DataLoader."""

    def __init__(self, paths: list[str], bio_preprocess, dino_processor):
        self.paths = paths
        self.bio_preprocess = bio_preprocess
        self.dino_processor = dino_processor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[idx]) as img:
            rgb = img.convert("RGB")
        bio_tensor = self.bio_preprocess(rgb)
        if callable(self.dino_processor) and not hasattr(self.dino_processor, "image_mean"):
            dino_pixel_values = self.dino_processor(rgb)
        else:
            dino_pixel_values = self.dino_processor(images=rgb, return_tensors="pt")["pixel_values"].squeeze(0)
        return bio_tensor, dino_pixel_values


class Encoder:
    def __init__(
        self,
        device: str = "cuda",
        fp16: bool = True,
        dino_model: str | None = None,
        load_bioclip: bool = True,
    ):
        """Load both models in eval mode; assert output dims == config.BIOCLIP_DIM / DINO_DIM.
        dino_model overrides config.DINO_MODEL (e.g. the dinov2-large fallback)."""
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.fp16 = bool(fp16 and str(self.device).startswith("cuda"))
        self.dtype = torch.float16 if self.fp16 else torch.float32

        if str(self.device).startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        if load_bioclip:
            orig_load_ckpt = open_clip.factory.load_checkpoint
            prev_dtype = torch.get_default_dtype()
            try:
                if self.fp16:
                    torch.set_default_dtype(torch.float16)
                open_clip.factory.load_checkpoint = (
                    lambda m, p, **kw: _stream_safetensors_into_model(m, p, self.device, self.fp16)
                )
                self.bioclip, _, self.preprocess_val = open_clip.create_model_and_transforms(
                    config.BIOCLIP_MODEL,
                    precision="fp16" if self.fp16 else "fp32",
                    device=self.device,
                )
            finally:
                open_clip.factory.load_checkpoint = orig_load_ckpt
                torch.set_default_dtype(prev_dtype)

            if self.fp16 and hasattr(open_clip, "layers") and hasattr(open_clip.layers, "LayerNormFp32"):
                for m in self.bioclip.modules():
                    if isinstance(m, open_clip.layers.LayerNormFp32):
                        m.forward = lambda x, m=m: torch.nn.functional.layer_norm(
                            x, m.normalized_shape, m.weight, m.bias, m.eps
                        )

            self.bioclip.eval()
            self.bioclip.requires_grad_(False)
            self.tokenizer = open_clip.get_tokenizer(config.BIOCLIP_MODEL)

            bioclip_dim = getattr(self.bioclip.visual, "output_dim", getattr(self.bioclip, "embed_dim", None))
            assert bioclip_dim == config.BIOCLIP_DIM, f"expected BIOCLIP_DIM={config.BIOCLIP_DIM}, got {bioclip_dim}"
        else:
            self.bioclip = None
            self.preprocess_val = None
            self.tokenizer = None

        target_dino = dino_model or config.DINO_MODEL
        try:
            self.dino_processor = self._load_processor(target_dino)
            self.dino = self._load_dino(target_dino)
            self.dino_model_name = target_dino
        except Exception as exc:
            if dino_model is not None:
                raise
            warnings.warn(
                f"Could not load {target_dino} ({exc}); falling back to {DINO_FALLBACK_MODEL}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.dino_processor = self._load_processor(DINO_FALLBACK_MODEL)
            self.dino = self._load_dino(DINO_FALLBACK_MODEL)
            self.dino_model_name = DINO_FALLBACK_MODEL

        self.dino.eval()
        self.dino.requires_grad_(False)
        dino_short = "dinov3" if "dinov3" in self.dino_model_name.lower() else self.dino_model_name.split("/")[-1]
        self.encoder_name = f"bioclip2+{dino_short}"

        dino_dim = getattr(self.dino.config, "hidden_size", None)
        assert dino_dim == config.DINO_DIM, f"expected DINO_DIM={config.DINO_DIM}, got {dino_dim}"

        dino_mean = list(getattr(self.dino_processor, "image_mean", [0.485, 0.456, 0.406]))
        dino_std = list(getattr(self.dino_processor, "image_std", [0.229, 0.224, 0.225]))
        self.dino_preprocess = T.Compose(
            [
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=dino_mean, std=dino_std),
            ]
        )
        self._bio_mean = [0.48145466, 0.4578275, 0.40821073]
        self._bio_std = [0.26862954, 0.26130258, 0.27577711]
        self._dino_mean = dino_mean
        self._dino_std = dino_std

    @staticmethod
    def _load_processor(model_name: str):
        try:
            return AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception:
            return AutoImageProcessor.from_pretrained(model_name)

    def _load_dino(self, model_name: str) -> torch.nn.Module:
        prev_dtype = torch.get_default_dtype()
        try:
            if self.fp16:
                torch.set_default_dtype(torch.float16)
            try:
                cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
                ckpt = hf_hub_download(model_name, "model.safetensors", local_files_only=True)
            except Exception:
                cfg = AutoConfig.from_pretrained(model_name)
                ckpt = hf_hub_download(model_name, "model.safetensors")
            model = AutoModel.from_config(cfg, dtype=self.dtype)
            _stream_safetensors_into_model(model, ckpt, self.device, self.fp16)
            return model.eval()
        finally:
            torch.set_default_dtype(prev_dtype)

    def _encode_dino_batch(self, dino_batch: torch.Tensor) -> np.ndarray:
        """Encode a batch of DINO pixel tensors into L2-normalised (B, DINO_DIM) float32 embeddings."""
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            dino_batch = dino_batch.to(self.device, dtype=self.dtype, non_blocking=True)
            dino_out = self.dino(pixel_values=dino_batch)
            dino_feats = dino_out.pooler_output.float()
            dino_feats = torch.nn.functional.normalize(dino_feats, p=2, dim=-1)
            return dino_feats.cpu().numpy().astype(np.float32)

    def _encode_batch_tensors(self, bio_batch: torch.Tensor, dino_batch: torch.Tensor) -> np.ndarray:
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            bio_batch = bio_batch.to(self.device, dtype=self.dtype, non_blocking=True)
            dino_batch = dino_batch.to(self.device, dtype=self.dtype, non_blocking=True)

            bio_feats = self.bioclip.encode_image(bio_batch).float()
            bio_feats = torch.nn.functional.normalize(bio_feats, p=2, dim=-1)

            dino_out = self.dino(pixel_values=dino_batch)
            dino_feats = dino_out.pooler_output.float()
            dino_feats = torch.nn.functional.normalize(dino_feats, p=2, dim=-1)

            combined = torch.cat([bio_feats, dino_feats], dim=1)
            return combined.cpu().numpy().astype(np.float32)

    def _encode_batch_with_tta(
        self,
        bio_batch: torch.Tensor,
        dino_batch: torch.Tensor,
        tta: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (X_out, X_plain) where X_out is block-renormalised mean of orig + hflip when tta=True."""
        X_plain = self._encode_batch_tensors(bio_batch, dino_batch)
        if not tta:
            return X_plain, X_plain
        X_flip = self._encode_batch_tensors(torch.flip(bio_batch, dims=[-1]), torch.flip(dino_batch, dims=[-1]))
        X_tta = _renorm_blocks(0.5 * (X_plain + X_flip))
        return X_tta, X_plain

    def embed_images(self, pil_images: list, tta: bool = True) -> np.ndarray:
        """(N, EMB_DIM) float32, each block L2-normalised (passes contracts.check_embeddings)."""
        if not pil_images:
            return np.zeros((0, config.EMB_DIM), dtype=np.float32)

        batch_size = 64
        chunks: list[np.ndarray] = []
        for start in range(0, len(pil_images), batch_size):
            batch_imgs = [
                img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img
                for img in pil_images[start : start + batch_size]
            ]
            bio_batch = torch.stack([self.preprocess_val(img) for img in batch_imgs])
            dino_batch = torch.stack([self.dino_preprocess(img) for img in batch_imgs])
            x_out, _ = self._encode_batch_with_tta(bio_batch, dino_batch, tta=tta)
            chunks.append(x_out)

        out = np.vstack(chunks).astype(np.float32)
        contracts.check_embeddings(out)
        return out

    def embed_paths(self, paths: list[str], batch_size: int = 64, tta: bool = True) -> np.ndarray:
        """Batched, RGB-convert, tqdm progress, torch.inference_mode, multi-threaded image loading."""
        if not paths:
            return np.zeros((0, config.EMB_DIM), dtype=np.float32)

        num_threads = max(1, int(os.environ.get("VANALENS_NUM_WORKERS", "8")))

        def _load_pair(p: str) -> tuple[torch.Tensor, torch.Tensor]:
            with Image.open(p) as img:
                rgb = img.convert("RGB")
            return self.preprocess_val(rgb), self.dino_preprocess(rgb)

        chunks: list[np.ndarray] = []
        batches = [paths[i : i + batch_size] for i in range(0, len(paths), batch_size)]
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            next_fut: Future | None = pool.submit(lambda b=batches[0]: list(pool.map(_load_pair, b))) if batches else None
            for idx in tqdm(range(len(batches)), desc="Embedding", leave=False):
                assert next_fut is not None
                pairs = next_fut.result()
                if idx + 1 < len(batches):
                    next_b = batches[idx + 1]
                    next_fut = pool.submit(lambda b=next_b: list(pool.map(_load_pair, b)))
                bio_batch = torch.stack([p[0] for p in pairs])
                dino_batch = torch.stack([p[1] for p in pairs])
                x_out, _ = self._encode_batch_with_tta(bio_batch, dino_batch, tta=tta)
                chunks.append(x_out)

        out = np.vstack(chunks).astype(np.float32)
        contracts.check_embeddings(out)
        return out

    def embed_manifest_shard(
        self,
        samples: list[contracts.Sample],
        batch_size: int = 64,
        start_idx: int = 0,
        tta: bool = True,
        aug_views: int = 0,
        seed: int = 0,
    ) -> tuple[list[contracts.Sample], np.ndarray, np.ndarray, np.ndarray | None, int]:
        """Decode, preprocess, and encode a shard of samples in one pass, skipping unreadable images.

        Returns (valid_samples, X, X_plain, X_aug, failed_count).
        """
        if not samples:
            empty_x = np.zeros((0, config.EMB_DIM), dtype=np.float32)
            empty_aug = np.zeros((0, config.EMB_DIM), dtype=np.float32) if aug_views > 0 else None
            return [], empty_x, empty_x, empty_aug, 0

        num_threads = max(1, int(os.environ.get("VANALENS_NUM_WORKERS", "8")))

        def _prepare_sample(item: tuple[int, contracts.Sample]):
            local_i, sample = item
            global_i = start_idx + local_i
            try:
                with Image.open(sample.path) as img:
                    rgb = img.convert("RGB")
                bio_t = self.preprocess_val(rgb)
                dino_t = self.dino_preprocess(rgb)
                if aug_views > 0:
                    rng = np.random.default_rng([seed, global_i])
                    crop_pil, do_flip = _random_resized_crop_pil(rgb, rng, size=224, scale=(0.6, 1.0))
                    crop_t = TF.to_tensor(crop_pil)
                    bio_crop = TF.normalize(crop_t, mean=self._bio_mean, std=self._bio_std)
                    dino_crop = TF.normalize(crop_t, mean=self._dino_mean, std=self._dino_std)
                    alpha = float(rng.uniform(0.6, 0.85))
                    return (sample, bio_t, dino_t, bio_crop, dino_crop, do_flip, alpha, None)
                return (sample, bio_t, dino_t, None, None, False, 0.0, None)
            except (OSError, UnidentifiedImageError, ValueError, SyntaxError, RuntimeError) as exc:
                return (sample, None, None, None, None, False, 0.0, exc)

        indexed = list(enumerate(samples))
        batches = [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]

        valid_samples: list[contracts.Sample] = []
        x_chunks: list[np.ndarray] = []
        x_plain_chunks: list[np.ndarray] = []
        x_aug_chunks: list[np.ndarray] = []
        failed_count = 0

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            next_fut: Future | None = pool.submit(lambda b=batches[0]: list(pool.map(_prepare_sample, b)))
            for b_idx in tqdm(range(len(batches)), desc="Embedding shard", leave=False):
                assert next_fut is not None
                prepared = next_fut.result()
                if b_idx + 1 < len(batches):
                    next_b = batches[b_idx + 1]
                    next_fut = pool.submit(lambda b=next_b: list(pool.map(_prepare_sample, b)))

                good = []
                for row in prepared:
                    sample, bio_t, dino_t, bio_c, dino_c, do_flip, alpha, err = row
                    if err is not None or bio_t is None:
                        failed_count += 1
                        logger.warning("Skipping unreadable image %s (%s): %s", sample.image_id, sample.path, err)
                        warnings.warn(f"Skipping unreadable image {sample.image_id}: {err}", RuntimeWarning, stacklevel=2)
                    else:
                        good.append((sample, bio_t, dino_t, bio_c, dino_c, do_flip, alpha))

                if not good:
                    continue

                valid_samples.extend(g[0] for g in good)
                bio_batch = torch.stack([g[1] for g in good])
                dino_batch = torch.stack([g[2] for g in good])

                X_plain = self._encode_batch_tensors(bio_batch, dino_batch)
                if tta or aug_views > 0:
                    X_flip = self._encode_batch_tensors(
                        torch.flip(bio_batch, dims=[-1]),
                        torch.flip(dino_batch, dims=[-1]),
                    )
                else:
                    X_flip = X_plain

                X_tta = _renorm_blocks(0.5 * (X_plain + X_flip)) if tta else X_plain
                x_chunks.append(X_tta)
                x_plain_chunks.append(X_plain)

                if aug_views > 0:
                    bio_crop_b = torch.stack([g[3] for g in good])
                    dino_crop_b = torch.stack([g[4] for g in good])
                    X_crop1 = self._encode_batch_tensors(bio_crop_b, dino_crop_b)

                    B_cur = len(good)
                    batch_aug = np.empty((B_cur * aug_views, config.EMB_DIM), dtype=np.float32)
                    if aug_views == 1:
                        batch_aug[:] = X_crop1
                    else:
                        flips = np.array([g[5] for g in good], dtype=bool)[:, None]
                        alphas = np.array([g[6] for g in good], dtype=np.float32)[:, None]
                        partner = np.where(flips, X_plain, X_flip)
                        X_crop2 = _renorm_blocks(alphas * X_crop1 + (1.0 - alphas) * partner)
                        batch_aug[0::2] = X_crop1
                        batch_aug[1::2] = X_crop2
                        for extra_v in range(2, aug_views):
                            batch_aug[extra_v::aug_views] = X_crop1
                    x_aug_chunks.append(batch_aug)

        X_out = np.vstack(x_chunks).astype(np.float32) if x_chunks else np.zeros((0, config.EMB_DIM), dtype=np.float32)
        X_plain_out = (
            np.vstack(x_plain_chunks).astype(np.float32)
            if x_plain_chunks
            else np.zeros((0, config.EMB_DIM), dtype=np.float32)
        )
        if X_out.shape[0] > 0:
            contracts.check_embeddings(X_out)
        if aug_views > 0:
            X_aug_out = (
                np.vstack(x_aug_chunks).astype(np.float32)
                if x_aug_chunks
                else np.zeros((0, config.EMB_DIM), dtype=np.float32)
            )
            if X_aug_out.shape[0] > 0:
                contracts.check_embeddings(X_aug_out)
        else:
            X_aug_out = None

        return valid_samples, X_out, X_plain_out, X_aug_out, failed_count

    def recompute_dino_shard(
        self,
        samples: list[contracts.Sample],
        batch_size: int = 32,
        start_idx: int = 0,
        tta: bool = True,
        aug_views: int = 0,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Recompute only the DINO block (DINO_DIM) for a shard of valid samples.

        Returns (dino_out, dino_plain_out, dino_aug_out) with exact flip TTA and seed-0 gallery crop/flip views.
        """
        if not samples:
            empty_d = np.zeros((0, config.DINO_DIM), dtype=np.float32)
            empty_aug = np.zeros((0, config.DINO_DIM), dtype=np.float32) if aug_views > 0 else None
            return empty_d, empty_d, empty_aug

        num_threads = max(1, int(os.environ.get("VANALENS_NUM_WORKERS", "8")))

        def _renorm_dino(D: np.ndarray) -> np.ndarray:
            d32 = D.astype(np.float32, copy=False)
            norms = np.linalg.norm(d32, axis=1, keepdims=True)
            return (d32 / np.maximum(norms, 1e-12)).astype(np.float32)

        def _prepare_dino(item: tuple[int, contracts.Sample]):
            local_i, sample = item
            global_i = start_idx + local_i
            with Image.open(sample.path) as img:
                rgb = img.convert("RGB")
            dino_t = self.dino_preprocess(rgb)
            if aug_views > 0:
                rng = np.random.default_rng([seed, global_i])
                crop_pil, do_flip = _random_resized_crop_pil(rgb, rng, size=224, scale=(0.6, 1.0))
                crop_t = TF.to_tensor(crop_pil)
                dino_crop = TF.normalize(crop_t, mean=self._dino_mean, std=self._dino_std)
                alpha = float(rng.uniform(0.6, 0.85))
                return (dino_t, dino_crop, do_flip, alpha)
            return (dino_t, None, False, 0.0)

        indexed = list(enumerate(samples))
        batches = [indexed[i : i + batch_size] for i in range(0, len(indexed), batch_size)]

        d_chunks: list[np.ndarray] = []
        d_plain_chunks: list[np.ndarray] = []
        d_aug_chunks: list[np.ndarray] = []

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            next_fut: Future | None = pool.submit(lambda b=batches[0]: list(pool.map(_prepare_dino, b)))
            for b_idx in tqdm(range(len(batches)), desc="Recomputing DINO shard", leave=False):
                assert next_fut is not None
                prepared = next_fut.result()
                if b_idx + 1 < len(batches):
                    next_b = batches[b_idx + 1]
                    next_fut = pool.submit(lambda b=next_b: list(pool.map(_prepare_dino, b)))

                dino_batch = torch.stack([g[0] for g in prepared])
                D_plain = self._encode_dino_batch(dino_batch)
                if tta or aug_views > 0:
                    D_flip = self._encode_dino_batch(torch.flip(dino_batch, dims=[-1]))
                else:
                    D_flip = D_plain

                D_tta = _renorm_dino(0.5 * (D_plain + D_flip)) if tta else D_plain
                d_chunks.append(D_tta)
                d_plain_chunks.append(D_plain)

                if aug_views > 0:
                    dino_crop_b = torch.stack([g[1] for g in prepared])
                    D_crop1 = self._encode_dino_batch(dino_crop_b)

                    B_cur = len(prepared)
                    batch_aug = np.empty((B_cur * aug_views, config.DINO_DIM), dtype=np.float32)
                    if aug_views == 1:
                        batch_aug[:] = D_crop1
                    else:
                        flips = np.array([g[2] for g in prepared], dtype=bool)[:, None]
                        alphas = np.array([g[3] for g in prepared], dtype=np.float32)[:, None]
                        partner = np.where(flips, D_plain, D_flip)
                        D_crop2 = _renorm_dino(alphas * D_crop1 + (1.0 - alphas) * partner)
                        batch_aug[0::2] = D_crop1
                        batch_aug[1::2] = D_crop2
                        for extra_v in range(2, aug_views):
                            batch_aug[extra_v::aug_views] = D_crop1
                    d_aug_chunks.append(batch_aug)

        D_out = np.vstack(d_chunks).astype(np.float32)
        D_plain_out = np.vstack(d_plain_chunks).astype(np.float32)
        D_aug_out = np.vstack(d_aug_chunks).astype(np.float32) if aug_views > 0 else None
        return D_out, D_plain_out, D_aug_out

    def encode_text(self, strings: list[str]) -> np.ndarray:
        """BioCLIP 2 text tower -> (N, BIOCLIP_DIM) float32, L2-normalised. Used by traits.build_text_protos."""
        if not strings:
            return np.zeros((0, config.BIOCLIP_DIM), dtype=np.float32)

        batch_size = 64
        chunks: list[np.ndarray] = []
        with torch.inference_mode(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            for start in range(0, len(strings), batch_size):
                tokens = self.tokenizer(strings[start : start + batch_size]).to(self.device)
                feats = self.bioclip.encode_text(tokens).float()
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
                chunks.append(feats.cpu().numpy().astype(np.float32))

        return np.vstack(chunks).astype(np.float32)


def _read_manifest_samples(manifest_path: Path) -> list[contracts.Sample]:
    try:
        from .data import read_manifest

        return read_manifest(manifest_path)
    except Exception:
        samples: list[contracts.Sample] = []
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                samples.append(contracts.Sample(**json.loads(line)))
        return samples


def _filter_readable_samples(samples: list[contracts.Sample]) -> list[contracts.Sample]:
    if not samples:
        return []

    def _check_one(sample: contracts.Sample) -> tuple[contracts.Sample, Exception | None]:
        try:
            with Image.open(sample.path) as img:
                img.verify()
            return sample, None
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
            return sample, exc

    if len(samples) <= 32:
        results = [_check_one(s) for s in samples]
    else:
        num_threads = max(1, int(os.environ.get("VANALENS_NUM_WORKERS", "8")))
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            results = list(pool.map(_check_one, samples))

    valid: list[contracts.Sample] = []
    for sample, exc in results:
        if exc is None:
            valid.append(sample)
        else:
            logger.warning("Skipping unreadable image %s (%s): %s", sample.image_id, sample.path, exc)
            warnings.warn(f"Skipping unreadable image {sample.image_id}: {exc}", RuntimeWarning, stacklevel=2)
    return valid


def _save_npz_atomic(path: Path, **arrays) -> None:
    """Atomically write arrays to path (.npz) so interrupted saves never leave a corrupt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f"{path.stem}.tmp_{os.getpid()}.npz"
    np.savez(tmp_path, **arrays)
    tmp_path.replace(path)


def recompute_dino_for_manifest(
    manifest_path: Path,
    out_path: Path,
    enc: Encoder,
    batch_size: int = 32,
    shard_size: int = 2000,
    tta: bool = True,
    aug_views: int | None = None,
    seed: int = 0,
    max_shards_per_run: int | None = None,
) -> float:
    """Recompute the DINO block for an existing cached .npz (preserving BioCLIP blocks bit-identically when compatible).

    Writes via atomic temp files and sets `encoder` to `enc.encoder_name` (e.g. "bioclip2+dinov2-large").
    Returns throughput in images/second.
    """
    resolved_aug = (2 if manifest_path.stem.endswith("_gallery") else 0) if aug_views is None else int(aug_views)
    samples = _read_manifest_samples(manifest_path)

    # Check if existing cache has matching ids and all required BioCLIP blocks (X, X_plain, and X_aug for galleries).
    can_reuse_bioclip = False
    cached_data: dict[str, np.ndarray] = {}
    if out_path.exists():
        try:
            with np.load(out_path, allow_pickle=False) as d:
                cached_ids = [str(i) for i in d["ids"]]
                manifest_ids = [s.image_id for s in samples]
                has_plain = "X_plain" in d and d["X_plain"].shape == d["X"].shape
                has_aug = (resolved_aug == 0) or (
                    "X_aug" in d
                    and "y_aug" in d
                    and d["X_aug"].shape == (len(samples) * resolved_aug, config.EMB_DIM)
                )
                if cached_ids == manifest_ids and has_plain and has_aug:
                    can_reuse_bioclip = True
                    cached_data = {k: d[k].copy() for k in d.keys()}
        except Exception as exc:
            logger.warning("Could not load existing cache %s for BioCLIP reuse (%s); full re-extract.", out_path, exc)

    t0 = time.perf_counter()
    if not can_reuse_bioclip:
        if enc.bioclip is None:
            raise RuntimeError(f"BioCLIP model required for full re-extraction of {manifest_path.name}")
        logger.info("Full re-extraction (TTA=%s, aug_views=%d) for %s -> %s", tta, resolved_aug, manifest_path.name, out_path.name)
        valid_samples, X_out, X_plain_out, X_aug_out, failed = enc.embed_manifest_shard(
            samples,
            batch_size=batch_size,
            start_idx=0,
            tta=tta,
            aug_views=resolved_aug,
            seed=seed,
        )
        ids_arr = np.array([s.image_id for s in valid_samples])
        has_labels = any(s.label is not None for s in valid_samples)
        y_arr = (
            np.array([s.label if s.label is not None else -1 for s in valid_samples], dtype=np.int64)
            if has_labels
            else np.array([], dtype=np.int64)
        )
        payload: dict[str, object] = {
            "ids": ids_arr,
            "X": X_out.astype(np.float32),
            "y": y_arr,
            "encoder": enc.encoder_name,
            "X_plain": X_plain_out.astype(np.float32),
        }
        if resolved_aug > 0 and X_aug_out is not None:
            payload["X_aug"] = X_aug_out.astype(np.float32)
            payload["y_aug"] = np.repeat(y_arr, resolved_aug).astype(np.int64) if y_arr.size > 0 else np.array([], dtype=np.int64)
        _save_npz_atomic(out_path, **payload)
        elapsed = max(time.perf_counter() - t0, 1e-6)
        rate = len(valid_samples) / elapsed
        logger.info(
            "Saved %d embeddings (full) to %s (X=%s, encoder=%s, %.1f img/s, failed=%d)",
            len(valid_samples),
            out_path.name,
            X_out.shape,
            enc.encoder_name,
            rate,
            failed,
        )
        return rate

    logger.info(
        "Recomputing DINO block only (keeping BioCLIP bit-identical, aug_views=%d) for %s -> %s",
        resolved_aug,
        manifest_path.name,
        out_path.name,
    )
    X_new = cached_data["X"].astype(np.float32, copy=True)
    X_plain_new = cached_data["X_plain"].astype(np.float32, copy=True)
    X_aug_new = cached_data["X_aug"].astype(np.float32, copy=True) if resolved_aug > 0 else None

    num_shards = max(1, (len(samples) + shard_size - 1) // shard_size)
    dino_shard_paths: list[Path] = []
    new_shards_computed = 0
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(len(samples), start_idx + shard_size)
        shard_samples = samples[start_idx:end_idx]
        dino_shard_path = out_path.parent / f"{out_path.stem}.dino_shard_{shard_idx:04d}.npz"
        dino_shard_paths.append(dino_shard_path)

        loaded_from_shard = False
        if dino_shard_path.exists():
            try:
                with np.load(dino_shard_path, allow_pickle=False) as ds:
                    D_shard = ds["D"]
                    D_plain_shard = ds["D_plain"]
                    D_aug_shard = ds["D_aug"] if (resolved_aug > 0 and "D_aug" in ds) else None
                    if D_shard.shape == (len(shard_samples), config.DINO_DIM):
                        loaded_from_shard = True
                        logger.info(
                            "[%s] Resuming DINO shard %d/%d from %s",
                            out_path.name,
                            shard_idx + 1,
                            num_shards,
                            dino_shard_path.name,
                        )
            except Exception:
                loaded_from_shard = False

        if not loaded_from_shard:
            D_shard, D_plain_shard, D_aug_shard = enc.recompute_dino_shard(
                shard_samples,
                batch_size=batch_size,
                start_idx=start_idx,
                tta=tta,
                aug_views=resolved_aug,
                seed=seed,
            )
            shard_save: dict[str, object] = {"D": D_shard, "D_plain": D_plain_shard}
            if resolved_aug > 0 and D_aug_shard is not None:
                shard_save["D_aug"] = D_aug_shard
            _save_npz_atomic(dino_shard_path, **shard_save)
            new_shards_computed += 1

        X_new[start_idx:end_idx, config.BIOCLIP_DIM :] = D_shard
        X_plain_new[start_idx:end_idx, config.BIOCLIP_DIM :] = D_plain_shard
        if resolved_aug > 0 and X_aug_new is not None and D_aug_shard is not None:
            X_aug_new[start_idx * resolved_aug : end_idx * resolved_aug, config.BIOCLIP_DIM :] = D_aug_shard

        elapsed_so_far = max(time.perf_counter() - t0, 1e-6)
        logger.info(
            "[%s] DINO shard %d/%d done (%d/%d images, %.1f img/s)",
            out_path.name,
            shard_idx + 1,
            num_shards,
            end_idx,
            len(samples),
            end_idx / elapsed_so_far,
        )
        if max_shards_per_run is not None and new_shards_computed >= max_shards_per_run and (shard_idx + 1) < num_shards:
            logger.info(
                "[%s] Reached --max-shards-per-run=%d at shard %d/%d; pausing for resume.",
                out_path.name,
                max_shards_per_run,
                shard_idx + 1,
                num_shards,
            )
            return end_idx / elapsed_so_far

    assert np.array_equal(X_new[:, : config.BIOCLIP_DIM], cached_data["X"][:, : config.BIOCLIP_DIM]), "BioCLIP X modified!"
    assert np.array_equal(
        X_plain_new[:, : config.BIOCLIP_DIM], cached_data["X_plain"][:, : config.BIOCLIP_DIM]
    ), "BioCLIP X_plain modified!"
    contracts.check_embeddings(X_new)
    contracts.check_embeddings(X_plain_new)

    payload = {
        "ids": cached_data["ids"],
        "X": X_new,
        "y": cached_data["y"],
        "encoder": enc.encoder_name,
        "X_plain": X_plain_new,
    }
    if resolved_aug > 0 and X_aug_new is not None:
        assert np.array_equal(
            X_aug_new[:, : config.BIOCLIP_DIM], cached_data["X_aug"][:, : config.BIOCLIP_DIM]
        ), "BioCLIP X_aug modified!"
        contracts.check_embeddings(X_aug_new)
        payload["X_aug"] = X_aug_new
        payload["y_aug"] = cached_data["y_aug"]

    _save_npz_atomic(out_path, **payload)
    for dsp in dino_shard_paths:
        if dsp.exists():
            dsp.unlink()
    elapsed = max(time.perf_counter() - t0, 1e-6)
    rate = len(samples) / elapsed
    logger.info(
        "Saved %d embeddings (DINO-recomputed, BioCLIP bit-identical) to %s (X=%s, encoder=%s, %.1f img/s)",
        len(samples),
        out_path.name,
        X_new.shape,
        enc.encoder_name,
        rate,
    )
    return rate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract BioCLIP 2 + DINO embeddings for a manifest.")
    parser.add_argument("--manifest", required=False, type=Path, help="Input manifest .jsonl path")
    parser.add_argument("--out", required=False, type=Path, help="Output EmbeddingSet .npz path")
    parser.add_argument(
        "--recompute-dino",
        action="store_true",
        help="Recompute DINO block across cached splits (or for --manifest/--out), keeping BioCLIP bit-identical",
    )
    parser.add_argument("--detach", action="store_true", help="Detach process into background")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional log file path")
    parser.add_argument("--force", action="store_true", help="Recompute even if encoder field already matches")
    parser.add_argument("--max-shards-per-run", type=int, default=None, help="Max new shards to compute before pausing")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for image encoding")
    parser.add_argument("--shard-size", type=int, default=2000, help="Shard size for resume-safe extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--dino-model", type=str, default=None, help="Optional DINO model override")
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip test-time augmentation")
    parser.add_argument(
        "--aug-views",
        type=int,
        default=None,
        help="Extra augmented views per image saved as X_aug/y_aug (default: 2 for *_gallery, 0 otherwise)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for gallery crop+flip augmentations")
    args = parser.parse_args(argv)

    if args.detach:
        import subprocess
        import sys

        raw_args = list(sys.argv[1:] if argv is None else argv)
        child_args = [a for a in raw_args if a != "--detach"]
        target_log = args.log_file or (config.RUNS_DIR / "logs" / "fix_dino.log")
        target_log.parent.mkdir(parents=True, exist_ok=True)
        log_fh = target_log.open("a", encoding="utf-8")

        cmd = [sys.executable, "-m", "vanalens.encoders", *child_args]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["VANALENS_DETACHED_CHILD"] = "1"
        popen_kwargs: dict = {
            "cwd": str(config.REPO_DIR),
            "stdin": subprocess.DEVNULL,
            "stdout": log_fh,
            "stderr": subprocess.STDOUT,
            "env": env,
        }
        if os.name == "nt":
            detached_flag = getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
            new_group_flag = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
            popen_kwargs["creationflags"] = detached_flag | new_group_flag
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **popen_kwargs)
        print(f"Detached vanalens.encoders (pid={proc.pid})", flush=True)
        return

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_file is not None and not os.environ.get("VANALENS_DETACHED_CHILD"):
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)

    tta = not args.no_tta
    if args.recompute_dino:
        if args.manifest is not None and args.out is not None:
            pairs = [(args.manifest, args.out)]
        else:
            m_dir = config.CACHE_DIR / "manifests"
            e_dir = config.CACHE_DIR / "emb"
            order = [
                "inat_india_gallery",
                "inat_india_cal",
                "inat_india_test",
                "vl600_cal",
                "vl600_test",
                "vl600_gallery",
            ]
            manifests = sorted(m_dir.glob("*.jsonl"), key=lambda p: (order.index(p.stem) if p.stem in order else 99, p.name))
            pairs = [(m, e_dir / f"{m.stem}.npz") for m in manifests]

        pending_pairs = []
        for m_path, o_path in pairs:
            if not args.force and o_path.exists():
                try:
                    with np.load(o_path, allow_pickle=False) as d_chk:
                        if str(d_chk["encoder"]) == "bioclip2+dinov2-large":
                            logger.info("Already updated (%s), skipping %s", str(d_chk["encoder"]), o_path.name)
                            continue
                except Exception:
                    pass
            pending_pairs.append((m_path, o_path))

        needs_bio = False
        for m_path, o_path in pending_pairs:
            req_aug = (2 if m_path.stem.endswith("_gallery") else 0) if args.aug_views is None else int(args.aug_views)
            if not o_path.exists():
                needs_bio = True
                break
            with np.load(o_path, allow_pickle=False) as d_chk:
                if "X_plain" not in d_chk or (req_aug > 0 and "X_aug" not in d_chk):
                    needs_bio = True
                    break

        if pending_pairs:
            enc = Encoder(device=args.device, fp16=True, dino_model=args.dino_model, load_bioclip=needs_bio)
            t_all = time.perf_counter()
            total_imgs = 0
            for m_path, o_path in pending_pairs:
                req_aug = (2 if m_path.stem.endswith("_gallery") else 0) if args.aug_views is None else int(args.aug_views)
                if enc.bioclip is not None and o_path.exists():
                    with np.load(o_path, allow_pickle=False) as d_chk:
                        if "X_plain" in d_chk and (req_aug == 0 or "X_aug" in d_chk):
                            enc.bioclip = None
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                recompute_dino_for_manifest(
                    manifest_path=m_path,
                    out_path=o_path,
                    enc=enc,
                    batch_size=args.batch_size,
                    shard_size=args.shard_size,
                    tta=tta,
                    aug_views=args.aug_views,
                    seed=args.seed,
                    max_shards_per_run=args.max_shards_per_run,
                )
                with np.load(o_path, allow_pickle=False) as d_done:
                    total_imgs += int(d_done["X"].shape[0])
            elapsed_all = max(time.perf_counter() - t_all, 1e-6)
            logger.info("ALL DINO BLOCKS RECOMPUTED: %d images in %.1fs (%.1f img/s)", total_imgs, elapsed_all, total_imgs / elapsed_all)
        else:
            logger.info("ALL DINO BLOCKS RECOMPUTED: all splits already up-to-date")
        return

    if args.manifest is None or args.out is None:
        parser.error("--manifest and --out are required unless --recompute-dino is set")
    aug_views = (2 if args.manifest.stem.endswith("_gallery") else 0) if args.aug_views is None else int(args.aug_views)

    samples = _read_manifest_samples(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    enc: Encoder | None = None
    shard_paths: list[Path] = []
    total_new_images = 0
    total_failed = 0
    t0 = time.perf_counter()

    num_shards = max(1, (len(samples) + args.shard_size - 1) // args.shard_size)
    for shard_idx in range(num_shards):
        start_idx = shard_idx * args.shard_size
        shard_samples = samples[start_idx : start_idx + args.shard_size]
        shard_path = args.out.parent / f"{args.out.stem}.shard_{shard_idx:04d}.npz"
        shard_paths.append(shard_path)

        if shard_path.exists():
            try:
                _ = contracts.EmbeddingSet.load(shard_path)
                with np.load(shard_path, allow_pickle=False) as d_chk:
                    if aug_views > 0 and ("X_aug" not in d_chk or "y_aug" not in d_chk):
                        raise ValueError("Shard missing X_aug/y_aug")
                    if "failed_count" in d_chk and d_chk["failed_count"].size > 0:
                        total_failed += int(d_chk["failed_count"][0])
                logger.info("Resuming: loaded existing shard %s (%d/%d)", shard_path.name, shard_idx + 1, num_shards)
                continue
            except Exception as exc:
                logger.warning("Corrupt or incomplete shard %s (%s); recomputing.", shard_path.name, exc)

        if enc is None:
            enc = Encoder(device=args.device, fp16=True, dino_model=args.dino_model)

        valid_samples, X_shard, X_plain_shard, X_aug_shard, failed_shard = enc.embed_manifest_shard(
            shard_samples,
            batch_size=args.batch_size,
            start_idx=start_idx,
            tta=tta,
            aug_views=aug_views,
            seed=args.seed,
        )
        total_failed += failed_shard

        if valid_samples:
            ids_shard = [s.image_id for s in valid_samples]
            has_labels = any(s.label is not None for s in valid_samples)
            y_shard = (
                np.array([s.label if s.label is not None else -1 for s in valid_samples], dtype=np.int64)
                if has_labels
                else np.array([], dtype=np.int64)
            )
            total_new_images += len(valid_samples)
        else:
            ids_shard = []
            y_shard = np.array([], dtype=np.int64)

        shard_payload: dict[str, object] = {
            "ids": np.array(ids_shard),
            "X": X_shard.astype(np.float32),
            "y": y_shard,
            "encoder": getattr(enc, "encoder_name", "bioclip2+dinov2-large"),
            "X_plain": X_plain_shard.astype(np.float32),
            "failed_count": np.array([failed_shard], dtype=np.int64),
        }
        if aug_views > 0 and X_aug_shard is not None:
            y_aug_shard = np.repeat(y_shard, aug_views).astype(np.int64) if y_shard.size > 0 else np.array([], dtype=np.int64)
            shard_payload["X_aug"] = X_aug_shard.astype(np.float32)
            shard_payload["y_aug"] = y_aug_shard

        _save_npz_atomic(shard_path, **shard_payload)
        elapsed_so_far = max(time.perf_counter() - t0, 1e-6)
        logger.info(
            "Shard %d/%d saved: %s (%d valid, %d failed, %.1f img/s)",
            shard_idx + 1,
            num_shards,
            shard_path.name,
            len(ids_shard),
            failed_shard,
            total_new_images / elapsed_so_far,
        )

    all_ids: list[str] = []
    all_X: list[np.ndarray] = []
    all_X_plain: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_X_aug: list[np.ndarray] = []
    all_y_aug: list[np.ndarray] = []
    any_y = False
    has_all_plain = True
    resolved_encoder_name = getattr(enc, "encoder_name", None)

    for shard_path in shard_paths:
        s_set = contracts.EmbeddingSet.load(shard_path)
        if resolved_encoder_name is None and s_set.encoder:
            resolved_encoder_name = s_set.encoder
        all_ids.extend(s_set.ids)
        if s_set.X.size > 0:
            all_X.append(s_set.X)
        if s_set.y is not None and s_set.y.size > 0:
            any_y = True
            all_y.append(s_set.y)
        with np.load(shard_path, allow_pickle=False) as d_shard:
            if "X_plain" in d_shard and d_shard["X_plain"].size > 0:
                all_X_plain.append(d_shard["X_plain"].astype(np.float32))
            elif s_set.X.size > 0:
                has_all_plain = False
            if aug_views > 0:
                if "X_aug" in d_shard and d_shard["X_aug"].size > 0:
                    all_X_aug.append(d_shard["X_aug"].astype(np.float32))
                if "y_aug" in d_shard and d_shard["y_aug"].size > 0:
                    all_y_aug.append(d_shard["y_aug"].astype(np.int64))

    X_final = np.vstack(all_X).astype(np.float32) if all_X else np.zeros((0, config.EMB_DIM), dtype=np.float32)
    y_final = np.concatenate(all_y).astype(np.int64) if any_y and all_y else np.array([], dtype=np.int64)
    if X_final.shape[0] > 0:
        contracts.check_embeddings(X_final)

    final_payload: dict[str, object] = {
        "ids": np.array(all_ids),
        "X": X_final,
        "y": y_final,
        "encoder": resolved_encoder_name or "bioclip2+dinov2-large",
    }
    if has_all_plain and all_X_plain:
        X_plain_final = np.vstack(all_X_plain).astype(np.float32)
        if X_plain_final.shape == X_final.shape:
            final_payload["X_plain"] = X_plain_final

    aug_info = ""
    if aug_views > 0:
        X_aug_final = (
            np.vstack(all_X_aug).astype(np.float32)
            if all_X_aug
            else np.zeros((0, config.EMB_DIM), dtype=np.float32)
        )
        y_aug_final = (
            np.concatenate(all_y_aug).astype(np.int64)
            if all_y_aug
            else (np.repeat(y_final, aug_views).astype(np.int64) if y_final.size > 0 else np.array([], dtype=np.int64))
        )
        if X_aug_final.shape[0] > 0:
            contracts.check_embeddings(X_aug_final)
        final_payload["X_aug"] = X_aug_final
        final_payload["y_aug"] = y_aug_final
        aug_info = f", X_aug={X_aug_final.shape}"

    _save_npz_atomic(args.out, **final_payload)

    for shard_path in shard_paths:
        if shard_path.exists():
            shard_path.unlink()

    elapsed = max(time.perf_counter() - t0, 1e-6)
    img_per_s = total_new_images / elapsed if total_new_images else len(all_ids) / elapsed
    logger.info(
        "Saved %d embeddings to %s (X=%s%s, %.1f img/s, failed=%d)",
        len(all_ids),
        args.out,
        X_final.shape,
        aug_info,
        img_per_s,
        total_failed,
    )


if __name__ == "__main__":
    main()

