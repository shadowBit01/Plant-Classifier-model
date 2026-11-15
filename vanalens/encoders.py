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


def _stream_safetensors_into_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device,
    fp16: bool,
) -> dict:
    """Stream tensors from a .safetensors file directly into model parameters on device without full-file mmap."""
    if fp16:
        model.half()
    model.to(device)
    targets = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    with open(checkpoint_path, "rb") as f, torch.no_grad():
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        base_offset = 8 + header_len
        for key, info in header.items():
            if key == "__metadata__":
                continue
            target = targets.get(key)
            if target is None and "." in key:
                target = targets.get(key.split(".", 1)[1])
            if target is None:
                continue
            start, end = info["data_offsets"]
            f.seek(base_offset + start)
            buf = bytearray(f.read(end - start))
            tensor = torch.frombuffer(buf, dtype=_SAFETENSORS_DTYPE_MAP[info["dtype"]]).reshape(info["shape"])
            target.data.copy_(tensor.reshape(target.shape).to(device=target.device, dtype=target.dtype))
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

        self.bioclip.eval()
        self.bioclip.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(config.BIOCLIP_MODEL)

        bioclip_dim = getattr(self.bioclip.visual, "output_dim", getattr(self.bioclip, "embed_dim", None))
        assert bioclip_dim == config.BIOCLIP_DIM, f"expected BIOCLIP_DIM={config.BIOCLIP_DIM}, got {bioclip_dim}"

        target_dino = dino_model or config.DINO_MODEL
        try:
            self.dino_processor = AutoImageProcessor.from_pretrained(target_dino)
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
            self.dino_processor = AutoImageProcessor.from_pretrained(DINO_FALLBACK_MODEL)
            self.dino = self._load_dino(DINO_FALLBACK_MODEL)
            self.dino_model_name = DINO_FALLBACK_MODEL

        self.dino.eval()
        self.dino.requires_grad_(False)

        dino_dim = getattr(self.dino.config, "hidden_size", None)
        assert dino_dim == config.DINO_DIM, f"expected DINO_DIM={config.DINO_DIM}, got {dino_dim}"

    def _load_dino(self, model_name: str) -> torch.nn.Module:
        prev_dtype = torch.get_default_dtype()
        try:
            if self.fp16:
                torch.set_default_dtype(torch.float16)
            cfg = AutoConfig.from_pretrained(model_name)
            ckpt = hf_hub_download(model_name, "model.safetensors")
            model = AutoModel.from_config(cfg, dtype=self.dtype)
            _stream_safetensors_into_model(model, ckpt, self.device, self.fp16)
            return model.eval()
        except Exception:
            return AutoModel.from_pretrained(model_name, dtype=self.dtype).to(self.device).eval()
        finally:
            torch.set_default_dtype(prev_dtype)

    def _encode_batch_tensors(self, bio_batch: torch.Tensor, dino_batch: torch.Tensor) -> np.ndarray:
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            bio_batch = bio_batch.to(self.device, dtype=self.dtype, non_blocking=True)
            dino_batch = dino_batch.to(self.device, dtype=self.dtype, non_blocking=True)

            bio_feats = self.bioclip.encode_image(bio_batch).float()
            bio_feats = torch.nn.functional.normalize(bio_feats, p=2, dim=-1)

            dino_out = self.dino(pixel_values=dino_batch)
            dino_feats = dino_out.pooler_output.float()
            dino_feats = torch.nn.functional.normalize(dino_feats, p=2, dim=-1)

            combined = torch.cat([bio_feats, dino_feats], dim=1)
            return combined.cpu().numpy().astype(np.float32)

    def embed_images(self, pil_images: list) -> np.ndarray:
        """(N, EMB_DIM) float32, each block L2-normalised (passes contracts.check_embeddings)."""
        if not pil_images:
            return np.zeros((0, config.EMB_DIM), dtype=np.float32)

        batch_size = 32
        chunks: list[np.ndarray] = []
        for start in range(0, len(pil_images), batch_size):
            batch_imgs = [
                img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img
                for img in pil_images[start : start + batch_size]
            ]
            bio_batch = torch.stack([self.preprocess_val(img) for img in batch_imgs])
            dino_batch = self.dino_processor(images=batch_imgs, return_tensors="pt")["pixel_values"]
            chunks.append(self._encode_batch_tensors(bio_batch, dino_batch))

        out = np.vstack(chunks).astype(np.float32)
        contracts.check_embeddings(out)
        return out

    def embed_paths(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Batched, RGB-convert, tqdm progress, torch.no_grad, pin_memory DataLoader with 4 workers."""
        if not paths:
            return np.zeros((0, config.EMB_DIM), dtype=np.float32)

        dataset = _ImagePathDataset(paths, self.preprocess_val, self.dino_processor)
        pin_memory = str(self.device).startswith("cuda")
        num_workers = int(os.environ.get("VANALENS_NUM_WORKERS", "0" if os.name == "nt" else "4"))
        chunks: list[np.ndarray] = []

        with torch.no_grad():
            try:
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                for bio_batch, dino_batch in tqdm(loader, desc="Embedding", leave=False):
                    chunks.append(self._encode_batch_tensors(bio_batch, dino_batch))
            except (OSError, RuntimeError) as exc:
                if chunks or num_workers == 0:
                    raise
                logger.warning("DataLoader with %d workers failed (%s); falling back to num_workers=0", num_workers, exc)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=pin_memory,
                )
                for bio_batch, dino_batch in tqdm(loader, desc="Embedding", leave=False):
                    chunks.append(self._encode_batch_tensors(bio_batch, dino_batch))

        out = np.vstack(chunks).astype(np.float32)
        contracts.check_embeddings(out)
        return out

    def encode_text(self, strings: list[str]) -> np.ndarray:
        """BioCLIP 2 text tower -> (N, BIOCLIP_DIM) float32, L2-normalised. Used by traits.build_text_protos."""
        if not strings:
            return np.zeros((0, config.BIOCLIP_DIM), dtype=np.float32)

        batch_size = 64
        chunks: list[np.ndarray] = []
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
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
    valid: list[contracts.Sample] = []
    for sample in samples:
        try:
            with Image.open(sample.path) as img:
                img.verify()
            valid.append(sample)
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
            logger.warning("Skipping unreadable image %s (%s): %s", sample.image_id, sample.path, exc)
            warnings.warn(f"Skipping unreadable image {sample.image_id}: {exc}", RuntimeWarning, stacklevel=2)
    return valid


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract BioCLIP 2 + DINO embeddings for a manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="Input manifest .jsonl path")
    parser.add_argument("--out", required=True, type=Path, help="Output EmbeddingSet .npz path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for image encoding")
    parser.add_argument("--shard-size", type=int, default=2000, help="Shard size for resume-safe extraction")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device")
    parser.add_argument("--dino-model", type=str, default=None, help="Optional DINO model override")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    samples = _read_manifest_samples(args.manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    enc: Encoder | None = None
    shard_paths: list[Path] = []
    total_new_images = 0
    t0 = time.perf_counter()

    num_shards = max(1, (len(samples) + args.shard_size - 1) // args.shard_size)
    for shard_idx in range(num_shards):
        shard_samples = samples[shard_idx * args.shard_size : (shard_idx + 1) * args.shard_size]
        shard_path = args.out.parent / f"{args.out.stem}.shard_{shard_idx:04d}.npz"
        shard_paths.append(shard_path)

        if shard_path.exists():
            try:
                _ = contracts.EmbeddingSet.load(shard_path)
                logger.info("Resuming: loaded existing shard %s", shard_path.name)
                continue
            except Exception as exc:
                logger.warning("Corrupt shard %s (%s); recomputing.", shard_path.name, exc)

        if enc is None:
            enc = Encoder(device=args.device, fp16=True, dino_model=args.dino_model)

        valid_samples = _filter_readable_samples(shard_samples)
        if valid_samples:
            X_shard = enc.embed_paths([s.path for s in valid_samples], batch_size=args.batch_size)
            ids_shard = [s.image_id for s in valid_samples]
            has_labels = any(s.label is not None for s in valid_samples)
            y_shard = (
                np.array([s.label if s.label is not None else -1 for s in valid_samples], dtype=np.int64)
                if has_labels
                else None
            )
            total_new_images += len(valid_samples)
        else:
            X_shard = np.zeros((0, config.EMB_DIM), dtype=np.float32)
            ids_shard = []
            y_shard = None

        shard_set = contracts.EmbeddingSet(ids=ids_shard, X=X_shard, y=y_shard)
        shard_set.save(shard_path)

    all_ids: list[str] = []
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    any_y = False

    for shard_path in shard_paths:
        s_set = contracts.EmbeddingSet.load(shard_path)
        all_ids.extend(s_set.ids)
        if s_set.X.size > 0:
            all_X.append(s_set.X)
        if s_set.y is not None and s_set.y.size > 0:
            any_y = True
            all_y.append(s_set.y)

    X_final = np.vstack(all_X).astype(np.float32) if all_X else np.zeros((0, config.EMB_DIM), dtype=np.float32)
    y_final = np.concatenate(all_y).astype(np.int64) if any_y and all_y else None
    if X_final.shape[0] > 0:
        contracts.check_embeddings(X_final)

    final_set = contracts.EmbeddingSet(ids=all_ids, X=X_final, y=y_final)
    final_set.save(args.out)

    for shard_path in shard_paths:
        if shard_path.exists():
            shard_path.unlink()

    elapsed = max(time.perf_counter() - t0, 1e-6)
    img_per_s = total_new_images / elapsed if total_new_images else len(all_ids) / elapsed
    logger.info(
        "Saved %d embeddings to %s (%.1f img/s)",
        len(all_ids),
        args.out,
        img_per_s,
    )


if __name__ == "__main__":
    main()
