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
    def __init__(self, device: str = "cuda", fp16: bool = True, dino_model: str | None = None):
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract BioCLIP 2 + DINO embeddings for a manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="Input manifest .jsonl path")
    parser.add_argument("--out", required=True, type=Path, help="Output EmbeddingSet .npz path")
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    tta = not args.no_tta
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

