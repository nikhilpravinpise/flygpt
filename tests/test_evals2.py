"""Tests for the eval battery runners (T-EMB / T-MODEL / T-SEX / T-SHIP).

Synthetic fixtures only - planted cluster structure, fabricated
prediction artifacts with known margins, and hand-computed expectations
make every pass/fail deterministic without the real corpus, checkpoints,
or connectome. Kill flags (K1/K2/K3), NA-with-reason, and the CLAIMS
linter are exercised directly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flygpt.evals import emb_evals, model_evals, sex_evals, ship_checks
from flygpt.schema import bundle_from_edges, write_result

REPO = Path(__file__).resolve().parents[1]
DS_PATH = REPO / "evals" / "dimorphism_set.json"


def _bundle(types, edges):
    """GraphBundle over ``len(types)`` nodes from (pre, post, w, sign) tuples."""
    nodes = pl.DataFrame(
        {"bodyId": np.arange(1, len(types) + 1), "type": list(types)}
    )
    if edges:
        pre, post, w, s = (np.array(list(col)) for col in zip(*edges, strict=True))
    else:
        pre = post = np.array([], dtype=np.uint32)
        w = np.array([], dtype=np.int32)
        s = np.array([], dtype=np.int8)
    return bundle_from_edges(
        nodes, pre.astype(np.uint32), post.astype(np.uint32),
        w.astype(np.int32), s.astype(np.int8),
    )


# ---------------------------------------------------------------------------
# T-EMB-1: kNN type recovery
# ---------------------------------------------------------------------------


def test_t_emb_1_knn_recovers_planted_clusters():
    rng = np.random.default_rng(7)
    types = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    # axis-separated centers (cosine kNN needs angularly distinct blobs)
    centers = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]])
    emb = centers[np.repeat([0, 1, 2], 10)] + rng.normal(scale=0.05, size=(30, 2))
    # structureless-ish graph: the spectral baseline stays near chance
    edges = [(i, (i * 7 + 3) % 30, 5, 1) for i in range(30)]
    edges += [(i, (i + 11) % 30, 5, 1) for i in range(0, 30, 3)]
    art = emb_evals.run_t1_type_recovery(emb, types, bundle=_bundle(types, edges))
    assert art["metrics"]["knn_acc"] == pytest.approx(1.0)
    assert art["metrics"]["majority_rate"] == pytest.approx(1.0 / 3.0)
    assert art["checks"]["knn_acc_ge_2x_majority"] is True
    assert art["checks"]["knn_acc_ge_spectral_plus_margin"] is True
    assert art["pass"] is True


def test_t_emb_1_clustered_graph_sets_the_bar():
    # within-type cliques: the spectral baseline recovers types, so the
    # +0.05 margin over it cannot be cleared by a perfect embedding
    rng = np.random.default_rng(9)
    types = ["A"] * 6 + ["B"] * 6 + ["C"] * 6
    centers = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]])
    emb = centers[np.repeat([0, 1, 2], 6)] + rng.normal(scale=0.02, size=(18, 3))
    edges = [
        (i, j, 5, 1)
        for base in (0, 6, 12)
        for i in range(base, base + 6)
        for j in range(base, base + 6)
        if i != j
    ]
    art = emb_evals.run_t1_type_recovery(emb, types, bundle=_bundle(types, edges))
    assert art["metrics"]["spectral_acc"] >= 0.95
    assert art["checks"]["knn_acc_ge_spectral_plus_margin"] is False
    assert art["pass"] is False


def test_t_emb_1_random_embeddings_fail():
    rng = np.random.default_rng(13)
    types = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
    emb = rng.normal(size=(30, 4))
    edges = [(i, (i + 1) % 30, 5, 1) for i in range(30)]
    art = emb_evals.run_t1_type_recovery(emb, types, bundle=_bundle(types, edges))
    assert art["metrics"]["knn_acc"] < 2.0 / 3.0
    assert art["checks"]["knn_acc_ge_2x_majority"] is False
    assert art["pass"] is False


def test_t_emb_1_no_bundle_is_na_not_silent():
    rng = np.random.default_rng(7)
    types = ["A"] * 10 + ["B"] * 10
    emb = np.vstack(
        [8 * np.array([1.0, 0.0, 0.0]) + rng.normal(scale=0.1, size=(10, 3)),
         8 * np.array([0.0, 1.0, 0.0]) + rng.normal(scale=0.1, size=(10, 3))]
    )
    art = emb_evals.run_t1_type_recovery(emb, types, bundle=None)
    assert art["checks"]["knn_acc_ge_2x_majority"] is True
    assert "NA" in art["checks"]["knn_acc_ge_spectral_plus_margin"]
    assert art["pass"] is None


# ---------------------------------------------------------------------------
# T-EMB-2 / T-EMB-3: linear probes
# ---------------------------------------------------------------------------


def _onehot_emb(labels, scale=4.0, noise=0.05, seed=3):
    rng = np.random.default_rng(seed)
    classes = sorted(set(labels))
    emb = np.zeros((len(labels), len(classes)))
    for i, lab in enumerate(labels):
        emb[i, classes.index(lab)] = scale
    return emb + rng.normal(scale=noise, size=emb.shape)


def test_t_emb_2_nt_probe_separable_passes():
    classes = [f"nt{i}" for i in range(7)]
    labels = [c for c in classes for _ in range(8)]
    emb = _onehot_emb(labels)
    art = emb_evals.run_t2_nt_probe(emb, labels, seed=0)
    # split: 6 train / 2 test per class -> majority rate = 2/14
    assert art["metrics"]["majority_rate"] == pytest.approx(2.0 / 14.0)
    assert art["metrics"]["probe_acc"] >= 0.95
    assert art["checks"]["acc_ge_majority_plus_0.15"] is True
    assert art["pass"] is True
    assert art["metrics"]["n_classes"] == 7


def test_t_emb_2_random_embeddings_fail():
    rng = np.random.default_rng(21)
    classes = [f"nt{i}" for i in range(7)]
    labels = [c for c in classes for _ in range(8)]
    art = emb_evals.run_t2_nt_probe(rng.normal(size=(56, 7)), labels, seed=0)
    assert art["metrics"]["margin"] < 0.15
    assert art["checks"]["acc_ge_majority_plus_0.15"] is False
    assert art["pass"] is False


def test_t_emb_3_dimorphism_probe_passes():
    labels = ["isomorphic"] * 12 + ["dimorphic"] * 10 + ["male-specific"] * 10
    emb = _onehot_emb(labels, noise=0.02, seed=4)
    art = emb_evals.run_t3_dimorphism_probe(emb, labels, seed=0)
    assert art["metrics"]["balanced_acc"] >= 0.9
    assert art["checks"]["balanced_acc_ge_0.50"] is True
    assert art["pass"] is True


def test_t_emb_3_random_embeddings_fail():
    rng = np.random.default_rng(41)
    labels = ["isomorphic"] * 60 + ["dimorphic"] * 60 + ["male-specific"] * 60
    art = emb_evals.run_t3_dimorphism_probe(rng.normal(size=(180, 8)), labels, seed=0)
    # uninformative features -> probe sits at the 1/3 chance level
    assert art["metrics"]["balanced_acc"] < 0.50
    assert art["pass"] is False


def test_balanced_accuracy_math():
    # class recalls 0.5 / 1.0 / 0.5 -> balanced accuracy 2/3
    yt = [0, 0, 1, 1, 2, 2]
    yp = [0, 1, 1, 1, 2, 0]
    assert emb_evals._balanced_accuracy(yt, yp) == pytest.approx(2.0 / 3.0)


# ---------------------------------------------------------------------------
# T-EMB-4: NBLAST agreement (parquet + connectivity proxy)
# ---------------------------------------------------------------------------


def _block_fixture(n_blocks=8, block=25, noise=0.02, seed=5):
    """200 typed nodes in ``n_blocks`` connectivity/embedding blocks."""
    rng = np.random.default_rng(seed)
    n = n_blocks * block
    types = [f"T{b}_{i}" for b in range(n_blocks) for i in range(block)]
    emb = np.zeros((n, n_blocks))
    for b in range(n_blocks):
        emb[b * block:(b + 1) * block, b] = 10.0
    return types, emb + rng.normal(scale=noise, size=emb.shape), n_blocks, block


def _nblast_df(types, n_blocks, block):
    """query T -> top-50 = its 24 blockmates + 26 filler types."""
    q, m, s = [], [], []
    for t in types:
        b = int(t.split("_")[0][1:])
        mates = [f"T{b}_{i}" for i in range(block) if f"T{b}_{i}" != t]
        fillers = [x for x in types if int(x.split("_")[0][1:]) != b][:26]
        for j, mate in enumerate(mates + fillers):
            q.append(t)
            m.append(mate)
            s.append(100.0 - j)
    return pl.DataFrame({"query_id": q, "match_id": m, "score": s})


def test_t_emb_4_nblast_parquet_passes(tmp_path):
    types, emb, nb, blk = _block_fixture()
    nb_path = tmp_path / "nblast.parquet"
    _nblast_df(types, nb, blk).write_parquet(nb_path)
    art = emb_evals.run_t4_nblast_agreement(emb, types, nblast_path=nb_path)
    assert art["metrics"]["nblast_source"] == "nblast_parquet"
    # every type's top-10 embedding neighbors are blockmates inside its top-50
    assert art["metrics"]["observed_fraction"] == pytest.approx(1.0)
    assert art["metrics"]["chance_fraction"] == pytest.approx(50.0 / 199.0)
    assert art["checks"]["observed_ge_3x_chance"] is True
    assert art["pass"] is True


def test_t_emb_4_nblast_parquet_random_fails(tmp_path):
    types, _emb, nb, blk = _block_fixture()
    nb_path = tmp_path / "nblast.parquet"
    _nblast_df(types, nb, blk).write_parquet(nb_path)
    emb = np.random.default_rng(11).normal(size=(200, 8))
    art = emb_evals.run_t4_nblast_agreement(emb, types, nblast_path=nb_path)
    # expected hits ~ 10 * 50/199 ~ 2.5 per type -> fraction ~0.25 << 0.754
    assert art["metrics"]["observed_fraction"] < 3 * art["metrics"]["chance_fraction"]
    assert art["checks"]["observed_ge_3x_chance"] is False
    assert art["pass"] is False


def test_t_emb_4_connectivity_proxy_passes():
    types, emb, nb, blk = _block_fixture()
    n = nb * blk
    labels = types + [""] * nb
    emb_full = np.vstack([emb, np.random.default_rng(2).normal(size=(nb, nb))])
    # each typed node links out to its block's (untyped) hub -> blockmates
    # are exactly each other's undirected 2-hop typed neighborhood
    edges = [(b * blk + i, n + b, 5, 1) for b in range(nb) for i in range(blk)]
    bundle = _bundle(labels, edges)
    assert bundle.n_nodes == n + nb
    art = emb_evals.run_t4_nblast_agreement(
        emb_full, labels, nblast_path=None, bundle=bundle
    )
    assert art["metrics"]["nblast_source"] == "connectivity_2hop_proxy"
    assert art["metrics"]["observed_fraction"] == pytest.approx(1.0)
    assert art["pass"] is True


def test_t_emb_4_proxy_unaligned_embeddings_fail():
    types, _emb, nb, blk = _block_fixture()
    n = nb * blk
    labels = types + [""] * nb
    emb = np.random.default_rng(23).normal(size=(n + nb, nb))
    edges = [(b * blk + i, n + b, 5, 1) for b in range(nb) for i in range(blk)]
    art = emb_evals.run_t4_nblast_agreement(
        emb, labels, bundle=_bundle(labels, edges)
    )
    assert art["metrics"]["nblast_source"] == "connectivity_2hop_proxy"
    assert art["pass"] is False


def test_t_emb_4_missing_everything_is_na():
    types, emb, _nb, _blk = _block_fixture()
    art = emb_evals.run_t4_nblast_agreement(
        emb, types, nblast_path=Path("nonexistent.parquet"), bundle=None
    )
    assert art["pass"] is None
    assert "NA" in art["checks"]["status"]


def test_result_file_written(tmp_path):
    rng = np.random.default_rng(7)
    types = ["A"] * 10 + ["B"] * 10
    emb = np.vstack(
        [8 * np.array([1.0, 0.0, 0.0]) + rng.normal(scale=0.1, size=(10, 3)),
         8 * np.array([0.0, 1.0, 0.0]) + rng.normal(scale=0.1, size=(10, 3))]
    )
    edges = [(i, (i + 1) % 20, 5, 1) for i in range(20)]
    art = emb_evals.run_t1_type_recovery(
        emb, types, bundle=_bundle(types, edges), results_dir=tmp_path
    )
    saved = json.loads((tmp_path / "T-EMB-1.json").read_text())
    assert saved["artifact"] == "T-EMB-1"
    assert saved["pass"] == art["pass"]
    assert saved["schema"] == "results_v1"


# ---------------------------------------------------------------------------
# T-MODEL-1 / T-MODEL-2: fabricated prediction artifacts
# ---------------------------------------------------------------------------


def _make_pred(tmp_path, modes, split="stimulus_holdout", name="pred.npz"):
    """3-seed prediction artifact. ``modes`` = per-seed quality tag."""
    rng = np.random.default_rng(1)
    n_bins, n_types = 60, 3
    yt_parts, yp_parts = [], []
    for mode in modes:
        yt = np.zeros((n_bins, n_types))
        for v in range(n_types):
            yt[(np.arange(n_bins) + v) % 5 == 0, v] = 1.0  # 12/60 = 0.2 rate
        if mode == "perfect":
            yp = np.where(yt > 0, 0.55 + 0.1 * yt, 0.05)
        elif mode == "zero":
            yp = np.full(yt.shape, 0.4)  # binarizes all-zero -> F1 0 (= maj)
        elif mode == "noisy":
            yp = np.where(yt > 0, 0.55, 0.05) + rng.normal(scale=0.02, size=yt.shape)
        else:
            raise ValueError(mode)
        yt_parts.append(yt)
        yp_parts.append(yp)
    n = n_bins * len(modes)
    path = tmp_path / name
    np.savez(
        path,
        y_true=np.vstack(yt_parts),
        y_pred=np.vstack(yp_parts),
        seed=np.repeat(np.arange(len(modes)), n_bins),
        split=np.array([split] * n),
        family=np.array(["famX"] * n),
    )
    return path


def test_t_model_1_pass_clear_margin(tmp_path):
    pred = _make_pred(tmp_path, ["perfect"] * 3)
    art = model_evals.run_t1_heldout_families(pred)
    # majority predicts all-zero at 0.2 event rate -> margin = 1.0 per seed
    assert art["metrics"]["margin_mean"] == pytest.approx(1.0)
    assert art["checks"]["margin_mean_ge_0.10"] is True
    assert art["checks"]["margin_ci_lo_gt_0.05"] is True
    assert art["checks"]["kill_K1_triggered"] is False
    assert art["pass"] is True


def test_t_model_1_kill_k1_when_ci_crosses(tmp_path):
    # margins [1, 0, 0]: mean 0.333 clears +0.10 but the CI dips below +0.05
    pred = _make_pred(tmp_path, ["perfect", "zero", "zero"])
    art = model_evals.run_t1_heldout_families(pred)
    assert art["metrics"]["margin_mean"] == pytest.approx(1.0 / 3.0)
    assert art["checks"]["margin_mean_ge_0.10"] is True
    assert art["checks"]["margin_ci_lo_gt_0.05"] is False
    assert art["checks"]["kill_K1_triggered"] is True
    assert art["pass"] is False


def test_t_model_1_missing_split_is_na(tmp_path):
    pred = _make_pred(tmp_path, ["perfect"], split="train")
    art = model_evals.run_t1_heldout_families(pred)
    assert art["pass"] is None
    assert "NA" in art["checks"]["status"]


def test_t_model_2_pass(tmp_path):
    pred = _make_pred(tmp_path, ["perfect"] * 3, split="celltype_holdout")
    art = model_evals.run_t2_heldout_types(pred)
    assert art["metrics"]["margin_mean"] == pytest.approx(1.0)
    assert art["metrics"]["rate_pearson_macro"] == pytest.approx(1.0)
    assert art["checks"]["margin_mean_ge_0.05"] is True
    assert art["checks"]["rate_pearson_ge_0.20"] is True
    assert art["pass"] is True


def test_t_model_2_uncorrelated_fails(tmp_path):
    pred = _make_pred(tmp_path, ["zero"] * 3, split="celltype_holdout")
    art = model_evals.run_t2_heldout_types(pred)
    # margin 0 and zero-variance predictions (no qualifying type for r)
    assert art["checks"]["margin_mean_ge_0.05"] is False
    assert art["checks"]["rate_pearson_ge_0.20"] is False
    assert art["pass"] is False


# ---------------------------------------------------------------------------
# T-MODEL-3..11: control-comparison score series
# ---------------------------------------------------------------------------


def test_t_model_3_beats_both_controls():
    scores = {"real": [0.8, 0.82, 0.81],
              "rewired": [0.5, 0.52, 0.51],
              "wrong_animal": [0.4, 0.41, 0.39]}
    art = model_evals.run_t3_wrong_connectome(scores)
    assert art["metrics"]["margin_rewired"]["mean"] == pytest.approx(0.3)
    assert art["checks"]["margin_rewired_ge_0.05"] is True
    assert art["checks"]["margin_wrong_animal_ci_separated"] is True
    assert art["checks"]["kill_K2_triggered"] is False
    assert art["pass"] is True


def test_t_model_3_kill_k2():
    scores = {"real": [0.5, 0.5, 0.5],
              "rewired": [0.55, 0.55, 0.55],
              "wrong_animal": [0.1, 0.1, 0.1]}
    art = model_evals.run_t3_wrong_connectome(scores)
    assert art["checks"]["kill_K2_triggered"] is True
    assert art["pass"] is False


def test_t_model_4_permuted_margin():
    # dyadic margins (0.25 / 0.03125) keep float subtraction exact
    assert model_evals.run_t4_permuted_labels(
        {"real": [0.75, 0.75, 0.75], "permuted": [0.5, 0.5, 0.5]}
    )["pass"] is True
    assert model_evals.run_t4_permuted_labels(
        {"real": [0.5, 0.5, 0.5], "permuted": [0.46875, 0.46875, 0.46875]}
    )["pass"] is False


def test_t_model_5_dense_gate_and_upgrade():
    art = model_evals.run_t5_dense_baseline(
        {"masked": [0.70, 0.70, 0.70], "dense": [0.67, 0.67, 0.67]}
    )
    assert art["checks"]["margin_ge_minus_0.01"] is True
    assert art["checks"]["upgrade_C5_plus_0.02"] is True
    assert art["pass"] is True
    # within -0.01 but no upgrade
    art2 = model_evals.run_t5_dense_baseline(
        {"masked": [0.70, 0.70, 0.70], "dense": [0.705, 0.705, 0.705]}
    )
    assert art2["pass"] is True
    assert art2["checks"]["upgrade_C5_plus_0.02"] is False
    # below -0.01 fails the ship gate
    art3 = model_evals.run_t5_dense_baseline(
        {"masked": [0.60, 0.60, 0.60], "dense": [0.65, 0.65, 0.65]}
    )
    assert art3["pass"] is False


def test_t_model_6_linear_baseline_and_k3():
    art = model_evals.run_t6_linear_baseline(
        {"model": [0.5, 0.52, 0.51], "linear": [0.3, 0.31, 0.3]}
    )
    assert art["pass"] is True
    assert art["checks"]["kill_K3_triggered"] is False
    art = model_evals.run_t6_linear_baseline(
        {"model": [0.5, 0.5, 0.5], "linear": [0.5, 0.5, 0.5]}
    )
    assert art["checks"]["kill_K3_triggered"] is True
    assert art["pass"] is False


def test_t_model_7_reservoir_margin():
    assert model_evals.run_t7_reservoir_baseline(
        {"model": [0.6, 0.6, 0.6], "reservoir": [0.54, 0.54, 0.54]}
    )["pass"] is True
    assert model_evals.run_t7_reservoir_baseline(
        {"model": [0.6, 0.6, 0.6], "reservoir": [0.58, 0.58, 0.58]}
    )["pass"] is False


def test_t_model_8_silence_guard():
    sparse = np.zeros((40, 2))
    sparse[::5, 0] = 1.0   # 0.2 event rate
    dense = np.ones((40, 2))
    dense[::4, 0] = 0.0    # 0.75 event rate -> nonzero majority F1
    report = {
        "splits": {
            "family_holdout": {"allzero_f1": 0.0, "majority_f1": 0.42,
                               "metrics": {"event_f1_macro": 0.5}},
            "type_holdout": {"y_true": sparse},
            "dense_slice": {"y_true": dense},
        }
    }
    art = model_evals.run_t8_silence_guard(report)
    assert art["checks"]["allzero_le_majority_all_splits"] is True
    assert art["checks"]["no_accuracy_metric_keys"] is True
    assert art["pass"] is True


def test_t_model_8_invariant_violation_fails():
    report = {"splits": {"fam": {"allzero_f1": 0.6, "majority_f1": 0.5}}}
    art = model_evals.run_t8_silence_guard(report)
    assert art["checks"]["allzero_le_majority_all_splits"] is False
    assert art["pass"] is False


def test_t_model_8_raw_accuracy_key_fails():
    report = {"splits": {"fam": {"allzero_f1": 0.0, "majority_f1": 0.3,
                                 "metrics": {"raw_accuracy": 0.9}}}}
    art = model_evals.run_t8_silence_guard(report)
    assert art["checks"]["no_accuracy_metric_keys"] is False
    assert art["metrics"]["forbidden_metric_keys"]
    assert art["pass"] is False


def test_t_model_9_mask_direction():
    records = [
        {"changed_t1": [1, 2, 3], "postsyn_partners": [1, 2, 3, 4],
         "delta_t": [0.0]},
        {"changed_t1": [5], "postsyn_partners": [5, 6], "max_abs_rel_t": 5e-7},
    ]
    art = model_evals.run_t9_mask_direction(records)
    assert art["metrics"]["frac_changed_t1_postsyn"] == pytest.approx(1.0)
    assert art["pass"] is True


def test_t_model_9_leak_and_same_step_fail():
    leak = [{"changed_t1": [1, 2, 99], "postsyn_partners": [1, 2],
             "delta_t": [0.0]}]
    art = model_evals.run_t9_mask_direction(leak)
    assert art["metrics"]["frac_changed_t1_postsyn"] == pytest.approx(2.0 / 3.0)
    assert art["checks"]["frac_postsyn_ge_0.99"] is False
    assert art["pass"] is False
    same_step = [{"changed_t1": [1], "postsyn_partners": [1], "delta_t": [0.1]}]
    art2 = model_evals.run_t9_mask_direction(same_step)
    assert art2["checks"]["no_same_step_changes"] is False
    assert art2["pass"] is False


def test_t_model_10_gnn_flag():
    art = model_evals.run_t10_gnn_headtohead(
        {"transformer": [0.70, 0.70, 0.70], "gnn": [0.76, 0.76, 0.76]}
    )
    assert art["checks"]["gnn_win_gt_0.02"] is True
    assert art["pass"] is True
    art = model_evals.run_t10_gnn_headtohead(
        {"transformer": [0.70, 0.70, 0.70], "gnn": [0.71, 0.71, 0.71]}
    )
    assert art["checks"]["gnn_win_gt_0.02"] is False


def test_t_model_11_shuffled_margin():
    assert model_evals.run_t11_shuffled_weights(
        {"real": [0.6, 0.6, 0.6], "shuffled": [0.5, 0.5, 0.5]}
    )["pass"] is True
    assert model_evals.run_t11_shuffled_weights(
        {"real": [0.6, 0.6, 0.6], "shuffled": [0.58, 0.58, 0.58]}
    )["pass"] is False


def test_aggregate_evals(tmp_path):
    write_result(tmp_path, "T-FOO-1", {"a": True}, metrics={"m": 1.0}, passed=True)
    write_result(tmp_path, "T-FOO-2", {"a": False}, passed=False)
    write_result(tmp_path, "T-FOO-3", {"a": "NA: x"}, passed=None)
    table = model_evals.aggregate_evals(tmp_path)
    assert table["n_artifacts"] == 3
    assert table["n_pass"] == 1 and table["n_fail"] == 1 and table["n_na"] == 1
    assert table["artifacts"]["T-FOO-1"]["metrics"]["m"] == 1.0


# ---------------------------------------------------------------------------
# T-SEX-1..6
# ---------------------------------------------------------------------------


def test_t_sex_1_dimorphism_set():
    measured = {f"DS-{i}": True for i in range(1, 6)}
    measured["DS-6"] = False
    art = sex_evals.run_t1_dimorphism_set(DS_PATH, measured)
    assert art["metrics"]["fraction_reproduced"] == pytest.approx(5.0 / 6.0)
    assert art["checks"]["fraction_ge_0.80"] is True
    assert art["checks"]["kill_K4_triggered"] is False
    assert art["pass"] is True


def test_t_sex_1_missing_items_count_against():
    art = sex_evals.run_t1_dimorphism_set(DS_PATH, {"DS-1": True})
    assert art["metrics"]["fraction_reproduced"] == pytest.approx(1.0 / 6.0)
    assert art["metrics"]["missing_measurements"] == [f"DS-{i}" for i in range(2, 7)]
    assert art["checks"]["kill_K4_triggered"] is True
    assert art["pass"] is False


def _rates(level, n=40, v=3, amp=1.0, off=0.0):
    t = np.linspace(0, 2 * np.pi, n)
    return level + amp * np.sin(t)[:, None] * np.ones((1, v)) + off


def _circuit(cid, dimorphic, female_level, seeds=(0, 1, 2)):
    return {
        "id": cid,
        "dimorphic": dimorphic,
        "seeds": [
            {"male": _rates(5.0, off=s * 0.01), "female": _rates(female_level)}
            for s in seeds
        ],
    }


def test_t_sex_2_divergence_detects_dimorphic():
    circuits = [
        _circuit("iso1", False, 5.0),
        _circuit("iso2", False, 5.01),
        _circuit("dim1", True, 1.0),
        _circuit("dim2", True, 0.5),
    ]
    art = sex_evals.run_t2_divergence(circuits)
    assert art["metrics"]["fraction_divergent"] == pytest.approx(1.0)
    assert art["metrics"]["per_circuit"]["dim1"]["divergent"] is True
    assert art["checks"]["fraction_divergent_ge_0.50"] is True
    assert art["pass"] is True


def test_t_sex_2_isomorphic_dimorphics_fail():
    circuits = [
        _circuit("iso1", False, 5.0),
        _circuit("iso2", False, 5.01),
        _circuit("dim1", True, 5.0),   # within the null band
        _circuit("dim2", True, 5.01),
    ]
    art = sex_evals.run_t2_divergence(circuits)
    assert art["metrics"]["fraction_divergent"] == pytest.approx(0.0)
    assert art["pass"] is False


def test_t_sex_2_no_null_is_na():
    art = sex_evals.run_t2_divergence([_circuit("dim1", True, 1.0)])
    assert art["pass"] is None
    assert "NA" in art["checks"]["status"]


def test_t_sex_3_confound_report():
    good = {"metrics": {"sex_conditioned": {"f1": 0.5},
                        "specimen_conditioned": {"f1": 0.4}}}
    assert sex_evals.run_t3_confound_report(good)["pass"] is True
    bad = {"metrics": {"sex_conditioned": {"f1": 0.5}}}
    art = sex_evals.run_t3_confound_report(bad)
    assert art["pass"] is False
    assert art["metrics"]["missing_conditioning_blocks"] == ["specimen_conditioned"]


def test_t_sex_4_masking_audit():
    align = [
        {"male_type": "LoVP92", "banc_type": None, "iso_class": "male-specific"},
        {"male_type": "T1", "banc_type": "bT1", "iso_class": "isomorphic"},
        {"male_type": None, "banc_type": "fOnly", "iso_class": "female-specific"},
    ]
    clean = sex_evals.run_t4_masking_audit(
        align, female_types={"bT1", "fOnly"}, male_types={"LoVP92", "T1"}
    )
    assert clean["pass"] is True
    assert clean["metrics"]["n_leaks"] == 0
    leak = sex_evals.run_t4_masking_audit(
        align, female_types={"bT1", "fOnly", "LoVP92"}, male_types={"LoVP92", "T1"}
    )
    assert leak["pass"] is False
    assert leak["metrics"]["leaks"] == [{"type": "LoVP92", "leaked_into": "female"}]


def test_t_sex_5_graph_swap_audit():
    male = [(0, 1), (1, 2), (2, 3), (3, 4)]
    # swapped subgraph = nodes {3,4}: (2,3),(3,4) replaced by BANC edges
    female = [(0, 1), (1, 2), (2, 5), (5, 4)]
    expected = [(0, 1), (1, 2), (2, 5), (5, 4)]
    art = sex_evals.run_t5_graph_swap_audit(
        male, female, expected_female_edges=expected, swapped_nodes={3, 4, 5}
    )
    assert art["checks"]["female_mask_matches_banc"] is True
    assert art["checks"]["diff_confined_to_swapped_subgraph"] is True
    assert art["pass"] is True
    bad = [(0, 1), (1, 2), (2, 5), (0, 9)]  # (0,9): wrong edge, outside swap
    art2 = sex_evals.run_t5_graph_swap_audit(
        male, bad, expected_female_edges=expected, swapped_nodes={3, 4, 5}
    )
    assert art2["checks"]["female_mask_matches_banc"] is False
    assert art2["checks"]["diff_confined_to_swapped_subgraph"] is False
    assert art2["pass"] is False


def test_t_sex_6_isomorphic_agreement():
    art = sex_evals.run_t6_isomorphic_agreement(
        {"circA": True, "circB": True, "circC": False}
    )
    assert art["metrics"]["fraction_agree"] == pytest.approx(2.0 / 3.0)
    assert art["metrics"]["documented_failures"] == ["circC"]
    assert art["pass"] is True
    art = sex_evals.run_t6_isomorphic_agreement(
        {"circA": False, "circB": True, "circC": False}
    )
    assert art["pass"] is False
    records = [
        {"id": "x", "predicted": "up", "reference": "up"},
        {"id": "y", "predicted": "up", "reference": "down"},
        {"id": "z", "predicted": "down", "reference": "down"},
    ]
    art2 = sex_evals.run_t6_isomorphic_agreement(records)
    assert art2["metrics"]["fraction_agree"] == pytest.approx(2.0 / 3.0)
    assert art2["metrics"]["documented_failures"] == ["y"]


# ---------------------------------------------------------------------------
# T-SHIP-2/3/4/5/9/10
# ---------------------------------------------------------------------------

_CLAIMS = """# CLAIMS.md

| ID | Claim | Status | Evidence artifact | Notes |
|---|---|---|---|---|
| C0 | graph built | EVIDENCED | `results/T-OK-1.json` | x |
| C1 | generalizes | UNSUPPORTED | `results/T-MODEL-1.json` | x |
"""


def test_ship2_claims_linter(tmp_path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    write_result(res_dir, "T-OK-1", {"c": True}, passed=True)
    write_result(res_dir, "T-BAD-1", {"c": False}, passed=False)
    write_result(res_dir, "T-R-1", {"c": True}, passed=True)
    write_result(res_dir, "T-R-2", {"c": True}, passed=True)
    text = _CLAIMS + (
        "| C2 | claimed | EVIDENCED | `results/T-MISS-1.json` | x |\n"
        "| C3 | refuted | REFUTED | `results/T-BAD-1.json` | x |\n"
        "| C4 | broken | EVIDENCED | `results/T-BAD-1.json` | x |\n"
        "| C5 | ranged | EVIDENCED | `results/T-R-1.json` .. `results/T-R-2.json` | x |\n"
    )
    (tmp_path / "CLAIMS.md").write_text(text)
    out = ship_checks.claims_evidence_check(tmp_path)
    assert out["pass"] is False
    assert [m["artifact"] for m in out["missing"]] == ["results/T-MISS-1.json"]
    assert [r["artifact"] for r in out["refuted"]] == ["results/T-BAD-1.json"]
    # the expanded range resolves both artifacts for C5
    ranged = [c["artifact"] for c in out["checked"] if c["claim"] == "C5"]
    assert ranged == ["results/T-R-1.json", "results/T-R-2.json"]


def test_ship2_clean_repo_passes(tmp_path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    write_result(res_dir, "T-OK-1", {"c": True}, passed=True)
    (tmp_path / "CLAIMS.md").write_text(_CLAIMS)
    out = ship_checks.claims_evidence_check(tmp_path)
    assert out["pass"] is True
    assert ship_checks.main(["claims", "--repo-root", str(tmp_path),
                             "--results-dir", str(res_dir)]) == 0


def test_ship2_main_returns_1_on_failure(tmp_path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    (tmp_path / "CLAIMS.md").write_text(
        _CLAIMS + "| C9 | gone | EVIDENCED | `results/nope.json` | x |\n"
    )
    assert ship_checks.main(["claims", "--repo-root", str(tmp_path),
                             "--results-dir", str(res_dir)]) == 1


def _write_license_fixture(root, deps, notice_rows):
    (root / "pyproject.toml").write_text(
        "[project]\nname = " '"x"\n' "dependencies = [\n"
        + "".join(f'    "{d}",\n' for d in deps)
        + "]\n"
    )
    (root / "THIRD_PARTY_NOTICES.md").write_text(
        "# N\n\n## Runtime dependencies\n\n"
        "| Package | Version | License |\n|---|---|---|\n"
        + "".join(f"| {r} |\n" for r in notice_rows)
    )


def test_ship3_license_audit_clean(tmp_path):
    _write_license_fixture(tmp_path, ["numpy==2.5.3"], ["numpy | 2.5.3 | BSD-3-Clause"])
    out = ship_checks.license_audit(tmp_path)
    assert out["pass"] is True
    assert out["flagged"] == []


def test_ship3_flags_known_gpl(tmp_path):
    _write_license_fixture(tmp_path, ["numpy", "brian2==1.0"],
                           ["numpy | 2 | BSD-3-Clause", "brian2 | 1 | GPL-3.0"])
    out = ship_checks.license_audit(tmp_path)
    assert out["pass"] is False
    assert any(f["dep"] == "brian2" and "GPL" in f["reason"] for f in out["flagged"])


def test_ship3_flags_forbidden_license_and_missing_row(tmp_path):
    _write_license_fixture(
        tmp_path,
        ["evilpkg", "ncdata", "mystery"],
        ["evilpkg | 1 | LGPL-3.0", "ncdata | 1 | CC BY-NC 4.0"],
    )
    out = ship_checks.license_audit(tmp_path)
    assert out["pass"] is False
    reasons = {f["dep"]: f["reason"] for f in out["flagged"]}
    assert "LGPL" in reasons["evilpkg"]
    assert "CC BY-NC" in reasons["ncdata"]
    assert out["missing_notices"] == ["mystery"]


def test_ship9_secrets_scan(tmp_path):
    (tmp_path / "clean.txt").write_text("nothing to see here\n")
    (tmp_path / "leaky.txt").write_text(
        'hf_token = "hf_abcdefghijklmnopqrstuvwxyz"\n'
        'key = "sk-0123456789abcdefghijklmno"\n'
    )
    (tmp_path / "key.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nMII..\n")
    out = ship_checks.secrets_scan(tmp_path)
    assert out["pass"] is False
    pats = {f["pattern"] for f in out["findings"]}
    assert "hf_token" in pats and "openai_key" in pats and "private_key_block" in pats


def test_ship9_clean_tree_passes(tmp_path):
    (tmp_path / "a.txt").write_text("boring\n")
    (tmp_path / "b.md").write_text("# docs, no secrets\n")
    out = ship_checks.secrets_scan(tmp_path)
    assert out["pass"] is True
    assert out["findings"] == []


def test_ship4_card_claims(tmp_path):
    (tmp_path / "CLAIMS.md").write_text(_CLAIMS)
    card = tmp_path / "card.md"
    card.write_text(
        "Our model reaches 92% event-F1 on held-out families [C0].\n"
        "Limitation: synthetic-only release.\n"
    )
    out = ship_checks.card_claims_check(
        card, claims_path=tmp_path / "CLAIMS.md",
        required_limitations=["synthetic-only release"],
    )
    assert out["pass"] is True
    assert out["cited_claims"] == ["C0"]


def test_ship4_card_violations(tmp_path):
    (tmp_path / "CLAIMS.md").write_text(_CLAIMS)
    card = tmp_path / "card.md"
    card.write_text(
        "We beat every baseline [C1].\n"   # C1 is UNSUPPORTED
        "This is the first connectome GPT ever built.\n"  # uncited claim-ish
        "Numbers look great.\n"
    )
    out = ship_checks.card_claims_check(
        card, claims_path=tmp_path / "CLAIMS.md",
        required_limitations=["simulated data only"],
    )
    assert out["pass"] is False
    assert out["non_evidenced_citations"] == [{"claim": "C1", "status": "UNSUPPORTED"}]
    assert any("first" in s for s in out["uncited_claimish_sentences"])
    assert out["missing_limitations"] == ["simulated data only"]


def test_ship5_dataset_card_markers():
    text = (
        "This corpus is simulated, not recorded. Built from "
        "gs://flyem-male-cns/v1.0/ under CC-BY-4.0. Conventions: "
        "synapse_cutoff=5, dt=20ms.\n"
    )
    out = ship_checks.dataset_card_check(text)
    assert out["pass"] is True
    out2 = ship_checks.dataset_card_check("Some card without the markers.\n")
    assert out2["pass"] is False
    assert "simulated, not recorded" in out2["missing_markers"]


def test_ship10_speed_benchmark(tmp_path):
    def model(n):
        return sum(range(n))

    def sim(n):
        time.sleep(0.0002 * n)

    art = ship_checks.speed_benchmark(model, sim, n_steps=50, device="cpu",
                                      results_dir=tmp_path)
    assert art["metrics"]["speedup_ratio"] > 1.0
    assert art["metrics"]["n_steps"] == 50
    assert art["checks"]["ratio_ge_10_for_C18"] is True
    assert (tmp_path / "T-SHIP-10.json").exists()
