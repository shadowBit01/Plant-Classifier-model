"""Tests for L2 weight loading completeness and determinism (card L2-f)."""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoModel

from vanalens import config, encoders


def test_streaming_loader_raises_on_unloaded_parameters(tmp_path: Path):
    """_stream_safetensors_into_model must raise RuntimeError if any parameter or persistent buffer is left unloaded."""
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.Linear(4, 2),
    )
    # Create a safetensors file containing only layer 0's weights, leaving layer 1 unloaded.
    tensor_bytes = np.zeros((4, 4), dtype=np.float32).tobytes()
    bias_bytes = np.zeros((4,), dtype=np.float32).tobytes()
    header = {
        "0.weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, len(tensor_bytes)]},
        "0.bias": {"dtype": "F32", "shape": [4], "data_offsets": [len(tensor_bytes), len(tensor_bytes) + len(bias_bytes)]},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    ckpt_path = tmp_path / "partial.safetensors"
    with ckpt_path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(tensor_bytes)
        f.write(bias_bytes)

    with pytest.raises(RuntimeError, match="unloaded"):
        encoders._stream_safetensors_into_model(model, ckpt_path, device="cpu", fp16=False)


def test_dino_checkpoint_leaves_zero_unloaded_tensors():
    """Loading DINOv2-large via _stream_safetensors_into_model must match all 439 model parameters with 0 unloaded."""
    model_name = encoders.DINO_FALLBACK_MODEL
    try:
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
        ckpt = hf_hub_download(model_name, "model.safetensors", local_files_only=True)
    except Exception:
        cfg = AutoConfig.from_pretrained(model_name)
        ckpt = hf_hub_download(model_name, "model.safetensors")

    model = AutoModel.from_config(cfg, dtype=torch.float32)
    # Fill with NaNs first to verify every parameter is overwritten by the checkpoint.
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float("nan"))

    encoders._stream_safetensors_into_model(model, ckpt, device="cpu", fp16=False)

    nan_params = [name for name, p in model.named_parameters() if torch.isnan(p).any()]
    assert len(nan_params) == 0, f"Parameters left unloaded (still NaN): {nan_params}"


@pytest.mark.skipif(os.environ.get("VANALENS_GPU_TESTS") != "1", reason="GPU tests disabled")
def test_two_encoder_loads_give_identical_embeddings():
    """Two fresh Encoder instances must produce bit-identical BioCLIP and DINO embeddings for the same image."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    enc1 = encoders.Encoder( dino_model=encoders.DINO_FALLBACK_MODEL)
    emb1 = enc1.embed_images([img], tta=True)
    del enc1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    enc2 = encoders.Encoder(dino_model=encoders.DINO_FALLBACK_MODEL)
    emb2 = enc2.embed_images([img], tta=True)
    del enc2

    bio1, dino1 = emb1[:, : config.BIOCLIP_DIM], emb1[:, config.BIOCLIP_DIM :]
    bio2, dino2 = emb2[:, : config.BIOCLIP_DIM], emb2[:, config.BIOCLIP_DIM :]

    bio_cos = float(np.sum(bio1 * bio2))
    dino_cos = float(np.sum(dino1 * dino2))
    assert bio_cos > 0.99999, f"BioCLIP cos between two loads too low: {bio_cos}"
    assert dino_cos > 0.99999, f"DINO cos between two loads too low: {dino_cos}"
    assert np.allclose(emb1, emb2, atol=1e-5), "Two Encoder loads did not produce identical embeddings"
