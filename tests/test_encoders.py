"""Lane L2 tests. GPU-only: run with VANALENS_GPU_TESTS=1 pytest -q tests/test_encoders.py"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from vanalens.config import BIOCLIP_DIM
from vanalens.contracts import check_embeddings

pytestmark = pytest.mark.skipif(os.environ.get("VANALENS_GPU_TESTS") != "1", reason="GPU tests disabled")


@pytest.fixture(scope="module")
def enc():
    from vanalens.encoders import Encoder
    return Encoder()


def test_embed_images(enc):
    imgs = [Image.new("RGB", (300, 200), c) for c in ("green", "white")]
    X = enc.embed_images(imgs)
    check_embeddings(X)
    assert X.shape[0] == 2


def test_embed_paths_matches_images(enc, tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGB", (256, 256), "green").save(p)
    a = enc.embed_paths([str(p)])
    b = enc.embed_images([Image.open(p).convert("RGB")])
    assert np.allclose(a, b, atol=2e-2)


def test_encode_text(enc):
    T = enc.encode_text(["a photo of Ficus religiosa", "a photo of a leaf"])
    assert T.shape == (2, BIOCLIP_DIM) and np.allclose(np.linalg.norm(T, axis=1), 1, atol=1e-3)
