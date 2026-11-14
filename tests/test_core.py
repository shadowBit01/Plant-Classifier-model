"""Lane L3 tests: scorers, fusion, conformal, hierarchy."""
from __future__ import annotations

import numpy as np

from vanalens.conformal import coverage, fit_qhat, is_unknown, mean_set_size, predict_sets
from vanalens.contracts import EmbeddingSet, check_proba
from vanalens.fusion import apply_temperature, fit_temperature, fit_weights, fuse, nll
from vanalens.hierarchy import aggregate, backoff
from vanalens.scorers import KNNScorer, ProbeScorer, TextScorer

C = 5


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------- scorers ----------
def test_knn_accuracy_and_shape(clusters):
    knn = KNNScorer(clusters["gallery"], n_classes=C, k=10)
    P = knn.predict_proba(clusters["Xq"])
    check_proba(P, C)
    assert (P.argmax(1) == clusters["yq"]).mean() > 0.95


def test_knn_neighbors_are_gallery_ids(clusters):
    knn = KNNScorer(clusters["gallery"], n_classes=C)
    nb = knn.neighbors(clusters["Xq"][:4], n=3)
    assert len(nb) == 4 and all(len(r) == 3 for r in nb)
    assert all(i in set(clusters["gallery"].ids) for r in nb for i in r)
    assert nb[0][0].startswith("g/0/")


def test_knn_add_species_without_retraining(clusters):
    g = clusters["gallery"]
    keep = g.y != 4
    knn = KNNScorer(EmbeddingSet([i for i, k in zip(g.ids, keep) if k], g.X[keep], g.y[keep]), n_classes=C)
    q4 = clusters["Xq"][clusters["yq"] == 4]
    assert (knn.predict_proba(q4).argmax(1) == 4).mean() == 0.0
    knn.add(EmbeddingSet([i for i, k in zip(g.ids, keep) if not k][:5], g.X[~keep][:5], g.y[~keep][:5]))
    assert (knn.predict_proba(q4).argmax(1) == 4).mean() > 0.9


def test_probe_accuracy_and_missing_class(clusters):
    g = clusters["gallery"]
    P = ProbeScorer(C).fit(g).predict_proba(clusters["Xq"])
    check_proba(P, C)
    assert (P.argmax(1) == clusters["yq"]).mean() > 0.95
    keep = g.y != 4
    P2 = ProbeScorer(C).fit(EmbeddingSet([i for i, k in zip(g.ids, keep) if k], g.X[keep], g.y[keep])
                            ).predict_proba(clusters["Xq"])
    check_proba(P2, C)
    assert P2[:, 4].max() < 0.01


def test_text_scorer_zero_shot(clusters, text_protos):
    protos, strings = text_protos
    ts = TextScorer(protos, strings)
    P = ts.predict_proba(clusters["Xq"])
    check_proba(P, C)
    assert (P.argmax(1) == clusters["yq"]).mean() > 0.9
    ex = ts.explain(clusters["Xq"][0], c=0, top=2)
    assert len(ex) == 2 and set(ex) <= set(strings[0])


# ---------- fusion ----------
def test_fuse_rows_and_identity():
    rng = np.random.default_rng(0)
    P1, P2 = softmax(rng.normal(size=(50, C))), softmax(rng.normal(size=(50, C)))
    F = fuse([P1, P2], [0.5, 0.5])
    check_proba(F, C)
    assert np.allclose(fuse([P1, P2], [1.0, 0.0]), P1, atol=1e-6)


def test_temperature_fixes_overconfidence():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3000, C))
    y = np.array([rng.choice(C, p=row) for row in softmax(logits)])
    P_sharp = softmax(3.0 * logits)
    T = fit_temperature(P_sharp, y)
    assert T > 1.8
    assert nll(apply_temperature(P_sharp, T), y) < nll(P_sharp, y)


def test_fit_weights_prefers_good_scorer():
    rng = np.random.default_rng(0)
    y = rng.integers(0, C, size=500)
    good = softmax(np.eye(C)[y] * 3 + rng.normal(size=(500, C)))
    bad = softmax(rng.normal(size=(500, C)))
    w = fit_weights([good, bad], y)
    assert len(w) == 2 and w[0] > w[1]


# ---------- conformal ----------
def test_conformal_coverage(calibrated):
    P, y = calibrated
    q = fit_qhat(P[:3000], y[:3000], alpha=0.1)
    sets = predict_sets(P[3000:], q)
    assert coverage(sets, y[3000:]) >= 0.87
    assert 1.0 <= mean_set_size(sets) < C


def test_sets_never_empty_and_sorted():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    sets = predict_sets(P, qhat=0.0)
    assert [list(s) for s in sets] == [[0], [2]]
    s = predict_sets(P, qhat=0.75)[0]
    assert list(s) == [0, 1]


def test_is_unknown():
    assert list(is_unknown(np.array([0.9, 0.2]), 0.5)) == [False, True]


# ---------- hierarchy ----------
def test_aggregate_genus(taxa):
    P = np.array([[0.4, 0.35, 0.05, 0.1, 0.1]])
    Pg, names = aggregate(P, taxa, "genus")
    assert names == ["Alpha", "Beta", "Gamma"]
    assert np.allclose(Pg, [[0.75, 0.05, 0.2]])


def test_backoff_levels(taxa):
    assert backoff(np.array([0.7, 0.1, 0.1, 0.05, 0.05]), taxa)[:2] == ("species", "Alpha one")
    r, n, c = backoff(np.array([0.4, 0.35, 0.05, 0.1, 0.1]), taxa)
    assert (r, n) == ("genus", "Alpha") and abs(c - 0.75) < 1e-9
    r, n, c = backoff(np.array([0.3, 0.1, 0.25, 0.2, 0.15]), taxa)
    assert (r, n) == ("family", "F1") and abs(c - 0.65) < 1e-9
    assert backoff(np.array([0.3, 0.1, 0.25, 0.2, 0.15]), taxa, tau=0.9)[:2] == ("unknown", "")
