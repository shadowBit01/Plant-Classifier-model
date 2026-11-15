"""Lane L3 (core-ml). Three scorers over frozen embeddings. All return (N, C) probabilities.

Tests: tests/test_core.py
"""
from __future__ import annotations

import faiss
import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import BIOCLIP_DIM
from .contracts import EmbeddingSet

EPS = 1e-6


def _l2norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


class KNNScorer:
    """Cosine kNN over the reference gallery (inner product on L2-normalised X; use faiss.IndexFlatIP).

    predict_proba: per query take top-k neighbours, add exp(sim / temperature) to each neighbour's class,
    add EPS to every class, renormalise. Shape (N, n_classes), column j = class idx j.
    neighbors: the top-n gallery image ids per query (evidence shown in the UI).
    add(): append new gallery items (few-shot "add a species") without rebuilding from scratch.
    """

    def __init__(self, gallery: EmbeddingSet, n_classes: int, k: int = 10, temperature: float = 0.05):
        if gallery.y is None:
            raise ValueError("KNNScorer needs a labelled gallery")
        self.n_classes = n_classes
        self.k = k
        self.temperature = temperature
        self.index = faiss.IndexFlatIP(gallery.X.shape[1])
        self.ids: list[str] = []
        self.y = np.empty(0, dtype=np.int64)
        self.add(gallery)

    def add(self, items: EmbeddingSet) -> None:
        if items.y is None:
            raise ValueError("added gallery items need labels")
        self.index.add(np.ascontiguousarray(items.X, dtype=np.float32))
        self.ids.extend(items.ids)
        self.y = np.concatenate([self.y, np.asarray(items.y, dtype=np.int64)])

    def _search(self, X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        Xq = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float32)
        return self.index.search(Xq, min(k, self.index.ntotal))

    def max_similarity(self, X: np.ndarray) -> np.ndarray:
        """Best gallery cosine per query (input to conformal.is_unknown)."""
        D, _ = self._search(X, 1)
        return D[:, 0].astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        D, I = self._search(X, self.k)
        P = np.full((D.shape[0], self.n_classes), EPS, dtype=np.float64)
        valid = I >= 0
        rows = np.broadcast_to(np.arange(D.shape[0])[:, None], D.shape)[valid]
        np.add.at(P, (rows, self.y[I[valid]]), np.exp(D[valid].astype(np.float64) / self.temperature))
        return P / P.sum(1, keepdims=True)

    def neighbors(self, X: np.ndarray, n: int = 3) -> list[list[str]]:
        _, I = self._search(X, n)
        return [[self.ids[i] for i in row if i >= 0] for row in I]


class ProbeScorer:
    """Multinomial logistic regression on frozen features (sklearn, or a 1-layer torch head on GPU if slow).

    Output columns MUST be class idx 0..n_classes-1 even if some class has no training rows
    (give those columns EPS, then renormalise).
    """

    def __init__(self, n_classes: int, C: float = 1.0):
        self.n_classes = n_classes
        self.C = C
        self.model: LogisticRegression | None = None

    def fit(self, gallery: EmbeddingSet) -> "ProbeScorer":
        if gallery.y is None:
            raise ValueError("ProbeScorer needs a labelled gallery")
        self.model = LogisticRegression(C=self.C, max_iter=1000)
        self.model.fit(np.asarray(gallery.X, dtype=np.float32), np.asarray(gallery.y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("call fit() first")
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        P = np.full((X.shape[0], self.n_classes), EPS, dtype=np.float64)
        P[:, self.model.classes_.astype(np.int64)] = self.model.predict_proba(X)
        return P / P.sum(1, keepdims=True)


class TextScorer:
    """Zero-shot scorer over text prototypes (classification by description).

    protos[c]: (T_c, BIOCLIP_DIM) L2-normalised BioCLIP 2 text embeddings for class c
               (trait prompts + a plain "a photo of <species>" prompt).
    trait_strings[c]: the T_c strings behind protos[c], same order.
    Uses only the BioCLIP block X[:, :BIOCLIP_DIM] (re-normalise it).
    score[n, c] = mean_t cos(x_n, protos[c][t]);  P = softmax(score / temperature).
    explain(x, c, top): the `top` trait strings of class c with highest cosine to x.
    """

    def __init__(self, protos: list[np.ndarray], trait_strings: list[list[str]], temperature: float = 0.01):
        assert len(protos) == len(trait_strings)
        self.protos = [_l2norm(np.asarray(p, dtype=np.float64)) for p in protos]
        self.trait_strings = trait_strings
        self.temperature = temperature

    def _bio(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return _l2norm(X[:, :BIOCLIP_DIM])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        B = self._bio(X)
        S = np.stack([(B @ p.T).mean(1) for p in self.protos], axis=1) / self.temperature
        E = np.exp(S - S.max(1, keepdims=True))
        return E / E.sum(1, keepdims=True)

    def explain(self, x: np.ndarray, c: int, top: int = 3) -> list[str]:
        sims = (self._bio(x) @ self.protos[c].T)[0]
        return [self.trait_strings[c][i] for i in np.argsort(-sims)[:top]]


class HeadScorer:
    """Trained head on frozen features: linear (hidden=0) or one hidden layer MLP (ReLU + dropout).

    Training uses torch (GPU if available, else CPU) with AdamW, label smoothing and optional class-balanced
    loss weights; with a validation split it early-stops on validation top-1 (ties: lower NLL) and keeps the
    best epoch. Inputs are multiplied by `scale` (the blocks are already L2-normalised; per-feature
    standardisation was tried on vl600 and overfits badly: 0.56 vs 0.94 cal top-1). `weight_decay` is an explicit
    L2 penalty sum(W**2) on the weight matrices, optimiser Adam. Inference is pure numpy, so a saved head
    (save/load, .npz without pickle) needs no torch. Output columns are class idx 0..n_classes-1; classes with
    no training rows still get a column (EPS floor, renormalised).
    """

    def __init__(self, n_classes: int, hidden: int = 0, dropout: float = 0.0, weight_decay: float = 1e-4,
                 label_smoothing: float = 0.0, balanced: bool = False, lr: float = 1e-3, epochs: int = 60,
                 batch_size: int = 1024, patience: int = 8, device: str | None = None, seed: int = 0,
                 scale: float = 10.0):
        self.n_classes = n_classes
        self.scale = scale
        self.hidden = hidden
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.balanced = balanced
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.seed = seed
        self.params: dict[str, np.ndarray] | None = None
        self.best_epoch = -1

    def config(self) -> dict:
        return {"hidden": self.hidden, "dropout": self.dropout, "weight_decay": self.weight_decay,
                "label_smoothing": self.label_smoothing, "balanced": self.balanced, "lr": self.lr,
                "epochs": self.epochs, "batch_size": self.batch_size, "patience": self.patience, "scale": self.scale,
                "best_epoch": self.best_epoch}

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None) -> "HeadScorer":
        import torch

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        g = torch.Generator().manual_seed(self.seed)
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        mu = np.zeros(X.shape[1], dtype=np.float32)
        sd = np.full(X.shape[1], 1.0 / self.scale, dtype=np.float32)
        Xt = torch.from_numpy((X - mu) / sd).to(dev)
        yt = torch.from_numpy(y).to(dev)
        D, C = X.shape[1], self.n_classes
        if self.hidden > 0:
            net = torch.nn.Sequential(torch.nn.Dropout(min(self.dropout, 0.2)), torch.nn.Linear(D, self.hidden),
                                      torch.nn.ReLU(), torch.nn.Dropout(self.dropout),
                                      torch.nn.Linear(self.hidden, C))
        else:
            net = torch.nn.Sequential(torch.nn.Linear(D, C))
        net = net.to(dev)
        cw = None
        if self.balanced:
            cnt = np.bincount(y, minlength=C).astype(np.float64)
            w = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / max((cnt > 0).sum(), 1), 0.0)
            cw = torch.tensor(w, dtype=torch.float32, device=dev)
        loss_fn = torch.nn.CrossEntropyLoss(weight=cw, label_smoothing=self.label_smoothing)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        mats = [m.weight for m in net if isinstance(m, torch.nn.Linear)]
        has_val = X_val is not None and y_val is not None
        if has_val:
            Xv = torch.from_numpy((np.asarray(X_val, dtype=np.float32) - mu) / sd).to(dev)
            yv = np.asarray(y_val, dtype=np.int64)
        best_key, best_state, bad = None, None, 0
        n = len(y)
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(n, generator=g).to(dev)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(Xt[idx]), yt[idx])
                if self.weight_decay > 0:
                    loss = loss + self.weight_decay * sum((W ** 2).sum() for W in mats)
                loss.backward()
                opt.step()
            if not has_val:
                continue
            net.eval()
            with torch.no_grad():
                logp = torch.cat([torch.log_softmax(net(Xv[i:i + 8192]), 1) for i in range(0, len(yv), 8192)])
            logp = logp.cpu().numpy()
            key = (-float((logp.argmax(1) == yv).mean()), float(-logp[np.arange(len(yv)), yv].mean()))
            if best_key is None or key < best_key:
                best_key, bad, self.best_epoch = key, 0, ep
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        else:
            self.best_epoch = self.epochs - 1
        sdict = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
        lin = [k[:-len(".weight")] for k in sdict if k.endswith(".weight")]
        self.params = {"mu": mu, "sd": sd.astype(np.float32)}
        for j, k in enumerate(lin):
            self.params[f"W{j}"] = sdict[f"{k}.weight"].T.copy()
            self.params[f"b{j}"] = sdict[f"{k}.bias"].copy()
        return self

    def logits(self, X: np.ndarray) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("call fit() or load() first")
        p = self.params
        H = (np.atleast_2d(np.asarray(X, dtype=np.float32)) - p["mu"]) / p["sd"]
        H = H @ p["W0"] + p["b0"]
        if "W1" in p:
            H = np.maximum(H, 0.0) @ p["W1"] + p["b1"]
        return H.astype(np.float64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = self.logits(X)
        E = np.exp(Z - Z.max(1, keepdims=True))
        P = E / E.sum(1, keepdims=True) + EPS
        return P / P.sum(1, keepdims=True)

    def save(self, path) -> None:
        import json

        if self.params is None:
            raise RuntimeError("nothing to save")
        np.savez(path, config_json=json.dumps(self.config() | {"n_classes": self.n_classes}), **self.params)

    @classmethod
    def load(cls, path) -> "HeadScorer":
        import json

        d = np.load(path, allow_pickle=False)
        cfg = json.loads(str(d["config_json"]))
        best = cfg.pop("best_epoch", -1)
        h = cls(**cfg)
        h.best_epoch = best
        h.params = {k: d[k] for k in d.files if k != "config_json"}
        return h
