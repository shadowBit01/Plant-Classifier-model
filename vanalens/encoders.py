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

import numpy as np


class Encoder:
    def __init__(self, device: str = "cuda", fp16: bool = True, dino_model: str | None = None):
        """Load both models in eval mode; assert output dims == config.BIOCLIP_DIM / DINO_DIM.
        dino_model overrides config.DINO_MODEL (e.g. the dinov2-large fallback)."""
        raise NotImplementedError

    def embed_images(self, pil_images: list) -> np.ndarray:
        """(N, EMB_DIM) float32, each block L2-normalised (passes contracts.check_embeddings)."""
        raise NotImplementedError

    def embed_paths(self, paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Batched, RGB-convert, tqdm progress, torch.no_grad, pin_memory DataLoader with 4 workers."""
        raise NotImplementedError

    def encode_text(self, strings: list[str]) -> np.ndarray:
        """BioCLIP 2 text tower -> (N, BIOCLIP_DIM) float32, L2-normalised. Used by traits.build_text_protos."""
        raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
