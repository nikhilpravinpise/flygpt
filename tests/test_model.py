"""T-MODEL test battery - connectome transformer + control suite.

Two layers matching tests/test_graph.py:

1. Controls tests (rewire / permuted labels / shuffled weights / dense /
   no-edge / wrong-animal / linear-influence / synthetic data) are pure
   numpy - they run with NO jax installed. ``flygpt.model`` itself
   imports cleanly without jax (lazy framework imports per impl-spec).
2. Model tests (jax/flax) are guarded per-test by the ``jax_mod``
   fixture - ``pytest.importorskip`` at call time, so a jax-less CI env
   skips only the model half of this file.

Synthetic ground truth (conftest ``synthetic_bundle``): nodes 0..5 =
bodies 10..15; edges 0->1 w7 (+1), 0->2 w6 (-1), 2->1 w4 (+1),
1->1 w5 (-1 autapse), 3->1 w6 (sign 0). Node 4 is isolated. Node 0's
postsynaptic partners are exactly {1, 2}.

Plan refs: docs/plan.md T-MODEL-3/4/5/8/9/10/11 + T-SEX-5; EVAL.md 1.7
(silence-robust); docs/impl-spec.md "model.py"/"controls.py".
"""

from __future__ import annotations

import dataclasses
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import flygpt.controls as controls
import flygpt.model as model_mod
from flygpt.schema import bundle_from_edges

REPO = Path(__file__).resolve().parents[1]
WORM_FILE = REPO / "data" / "raw" / "worm" / controls.WORM_LOCAL_NAME

real_worm = pytest.mark.skipif(
    not WORM_FILE.exists(),
    reason="WormWiring SI5 adjacency not downloaded - run controls --worm",
)


# ---------------------------------------------------------------------------
# Shared helpers (numpy only)
# ---------------------------------------------------------------------------


def _degree_seqs(bundle):
    return np.diff(bundle.in_indptr), np.diff(bundle.out_indptr)


def _weight_sign_multiset(bundle):
    pairs = zip(bundle.out_weight.tolist(), bundle.out_sign.tolist(), strict=True)
    return sorted(pairs)


def _edge_set(bundle):
    return set(zip(
        np.repeat(np.arange(bundle.n_nodes), np.diff(bundle.out_indptr)).tolist(),
        bundle.out_indices.tolist(),
        strict=True,
    ))


def _chain_bundle(n=3):
    nodes = pl.DataFrame(
        {
            "bodyId": np.arange(n, dtype=np.int64),
            "type": pl.Series(["t"] * n, dtype=pl.String),
            "superclass": pl.Series(["s"] * n, dtype=pl.String),
        }
    )
    return bundle_from_edges(
        nodes,
        np.asarray([0, 1], dtype=np.uint32),
        np.asarray([1, 2], dtype=np.uint32),
        np.asarray([1, 1], dtype=np.int32),
        np.asarray([1, 1], dtype=np.int8),
    )


def _random_digraph(seed=11, n_nodes=12, n_edges=40):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n_nodes, n_edges * 4)
    post = rng.integers(0, n_nodes, n_edges * 4)
    keep = (pre != post) & (np.arange(pre.size) < n_edges * 3)
    pairs = list(
        dict.fromkeys(zip(pre[keep].tolist(), post[keep].tolist(), strict=True))
    )[:n_edges]
    nodes = pl.DataFrame(
        {
            "bodyId": np.arange(n_nodes, dtype=np.int64),
            "type": pl.Series(["t"] * n_nodes, dtype=pl.String),
            "superclass": pl.Series(["s"] * n_nodes, dtype=pl.String),
        }
    )
    return bundle_from_edges(
        nodes,
        np.asarray([p for p, _ in pairs], dtype=np.uint32),
        np.asarray([q for _, q in pairs], dtype=np.uint32),
        rng.integers(1, 10, len(pairs)).astype(np.int32),
        rng.choice(np.asarray([-1, 1], dtype=np.int8), len(pairs)),
    )


# ---------------------------------------------------------------------------
# Controls - rewiring (T-MODEL-3 primary control)
# ---------------------------------------------------------------------------


def test_rewire_preserves_degrees_and_weights(synthetic_bundle):
    rw = controls.rewire(synthetic_bundle, seed=7)
    in_a, out_a = _degree_seqs(synthetic_bundle)
    in_b, out_b = _degree_seqs(rw)
    assert np.array_equal(in_a, in_b), "in-degree sequence changed"
    assert np.array_equal(out_a, out_b), "out-degree sequence changed"
    assert _weight_sign_multiset(rw) == _weight_sign_multiset(synthetic_bundle)
    assert rw.n_nodes == synthetic_bundle.n_nodes


def test_rewire_random_digraph_changes_edges_deterministically():
    bundle = _random_digraph()
    rw1 = controls.rewire(bundle, seed=3)
    rw2 = controls.rewire(bundle, seed=3)
    in_a, out_a = _degree_seqs(bundle)
    in_b, out_b = _degree_seqs(rw1)
    assert np.array_equal(in_a, in_b) and np.array_equal(out_a, out_b)
    assert _weight_sign_multiset(rw1) == _weight_sign_multiset(bundle)
    assert _edge_set(rw1) != _edge_set(bundle), "40-edge digraph: no swap applied?"
    # determinism: same seed -> identical rewiring
    assert np.array_equal(rw1.out_indices, rw2.out_indices)
    assert np.array_equal(rw1.out_weight, rw2.out_weight)


# ---------------------------------------------------------------------------
# Controls - permuted labels (T-MODEL-4)
# ---------------------------------------------------------------------------


def test_permuted_labels_within_superclass(synthetic_bundle):
    pl_bundle = controls.permuted_labels(synthetic_bundle, seed=5)
    # wiring untouched - every CSR array identical
    for name in ("out_indptr", "out_indices", "out_weight", "out_sign",
                 "in_indptr", "in_indices", "in_weight", "in_sign"):
        assert np.array_equal(getattr(pl_bundle, name), getattr(synthetic_bundle, name))
    orig = synthetic_bundle.type_array()
    new = pl_bundle.type_array()
    # same superclass (all cb_intrinsic) -> same type multiset, nulls kept
    assert Counter(orig) == Counter(new)
    assert new[3] == ""  # untyped node stays untyped
    assert (new != orig).any(), "seeded permutation should reassign labels"
    # determinism
    again = controls.permuted_labels(synthetic_bundle, seed=5)
    assert (again.type_array() == new).all()


def test_permuted_labels_respects_strata():
    nodes = pl.DataFrame(
        {
            "bodyId": np.arange(4, dtype=np.int64),
            "type": pl.Series(["A1", "A2", "B1", "B2"], dtype=pl.String),
            "superclass": pl.Series(["sa", "sa", "sb", "sb"], dtype=pl.String),
        }
    )
    bundle = bundle_from_edges(
        nodes,
        np.asarray([0], dtype=np.uint32),
        np.asarray([1], dtype=np.uint32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1], dtype=np.int8),
    )
    for seed in (1, 2, 3):
        p = controls.permuted_labels(bundle, seed=seed)
        types = p.type_array()
        assert set(types[:2]) == {"A1", "A2"}, "types leaked across superclass"
        assert set(types[2:]) == {"B1", "B2"}, "types leaked across superclass"


# ---------------------------------------------------------------------------
# Controls - shuffled weights (T-MODEL-11), dense, no-edge
# ---------------------------------------------------------------------------


def test_shuffled_weights_topology_kept(synthetic_bundle):
    sw = controls.shuffled_weights(synthetic_bundle, seed=9)
    assert np.array_equal(sw.out_indptr, synthetic_bundle.out_indptr)
    assert np.array_equal(sw.out_indices, synthetic_bundle.out_indices)
    assert _weight_sign_multiset(sw) == _weight_sign_multiset(synthetic_bundle)
    # the assignment changed somewhere (deterministic under the seed)
    assert not np.array_equal(sw.out_weight, synthetic_bundle.out_weight)


def test_no_edge_bundle():
    b = controls.no_edge_bundle(5)
    assert b.n_nodes == 5
    assert b.out_indices.size == 0
    in_deg, out_deg = _degree_seqs(b)
    assert in_deg.sum() == 0 and out_deg.sum() == 0


def test_dense_mask_bundle():
    n = 6
    b = controls.dense_mask_bundle(n, weight=3)
    assert b.n_nodes == n
    in_deg, out_deg = _degree_seqs(b)
    assert (in_deg == n - 1).all() and (out_deg == n - 1).all()
    assert not any(p == q for p, q in _edge_set(b)), "self-loop in dense mask"
    assert (b.out_weight == 3).all()


# ---------------------------------------------------------------------------
# Controls - wrong-animal C. elegans bundle (T-MODEL-3b)
# ---------------------------------------------------------------------------


def _write_minimal_worm_xlsx(path: Path) -> None:
    """Write a tiny but valid .xlsx in the WormWiring SI5 layout.

    Sheets: row 3 = post-synaptic names in columns D+; col C = pre names;
    values = EM section counts. Includes a post-only non-neuron column
    (MUSCLE1) that must be dropped from the node set.
    """
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    names = ["AAA", "BBB", "CCC", "DDD", "MUSCLE1"]
    shared = "".join(f"<si><t>{n}</t></si>" for n in names)
    ss_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<sst xmlns="{ns}">{shared}</sst>'
    )
    # header row 3: D3=AAA E3=BBB F3=CCC G3=DDD H3=MUSCLE1
    hdr = "".join(
        f'<c r="{col}3" t="s"><v>{i}</v></c>' for i, col in enumerate("DEFGH")
    )
    row3 = f'<row r="3">{hdr}</row>'
    # data rows: AAA->BBB w3, AAA->MUSCLE1 w9 (dropped), BBB->CCC w4, CCC/DDD none
    row4 = '<row r="4"><c r="C4" t="s"><v>0</v></c><c r="E4"><v>3</v></c><c r="H4"><v>9</v></c></row>'
    row5 = '<row r="5"><c r="C5" t="s"><v>1</v></c><c r="F5"><v>4</v></c></row>'
    row6 = '<row r="6"><c r="C6" t="s"><v>2</v></c></row>'
    row7 = '<row r="7"><c r="C7" t="s"><v>3</v></c></row>'
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<worksheet xmlns="{ns}"><sheetData>{row3}{row4}{row5}{row6}{row7}</sheetData></worksheet>'
    )
    wb_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="hermaphrodite chemical" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Target="worksheets/sheet1.xml" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/workbook.xml", wb_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        zf.writestr("xl/sharedStrings.xml", ss_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_wrong_animal_csv(tmp_path):
    f = tmp_path / "worm.csv"
    f.write_text("pre,post,weight\nAAA,BBB,3\nBBB,CCC,4\nCCC,AAA,1\n")
    b = controls.wrong_animal_bundle(f)
    assert b.n_nodes == 3
    assert _edge_set(b) == {(0, 1), (1, 2), (2, 0)}
    assert sorted(b.out_weight.tolist()) == [1, 3, 4]
    assert (b.out_sign == 0).all(), "worm signs must be 0 (undetermined)"
    assert b.column("name").tolist() == ["AAA", "BBB", "CCC"]


def test_wrong_animal_xlsx(tmp_path):
    f = tmp_path / "si5.xlsx"
    _write_minimal_worm_xlsx(f)
    b = controls.wrong_animal_bundle(f)
    assert b.n_nodes == 4, "post-only non-neuron (MUSCLE1) must be dropped"
    assert _edge_set(b) == {(0, 1), (1, 2)}  # AAA->BBB, BBB->CCC
    assert sorted(b.out_weight.tolist()) == [3, 4]
    assert (b.out_sign == 0).all()
    assert b.column("name").tolist() == ["AAA", "BBB", "CCC", "DDD"]


@real_worm
def test_wrong_animal_real_file():
    """The downloaded SI5 workbook parses into a sane connectome."""
    b = controls.wrong_animal_bundle(WORM_FILE)
    assert 250 <= b.n_nodes <= 500  # N2U hermaphrodite is ~300 neurons
    assert b.out_indices.size > 1000
    assert (b.out_sign == 0).all()


# ---------------------------------------------------------------------------
# Controls - linear influence proxy (T-MODEL-6)
# ---------------------------------------------------------------------------


def test_linear_influence_chain():
    b = _chain_bundle()
    m = controls.linear_influence(b, hops=3, decay=0.5)
    assert m.shape == (3, 3) and m.dtype == np.float32
    assert m[0, 1] > m[0, 2] > 0, "direct edge must beat the 2-hop path"
    assert m[0, 2] == pytest.approx(0.5)  # single 2-hop path at decay 0.5
    assert m[2, 0] == 0  # no backward influence


def test_linear_influence_synthetic(synthetic_bundle):
    m = controls.linear_influence(synthetic_bundle, hops=3, decay=0.5)
    assert m.shape == (6, 6)
    assert m[0, 1] > 0 and m[0, 2] > 0  # real edges carry influence
    assert m[0, 4] == 0 and m[4, :].sum() == 0  # isolated node unreachable


# ---------------------------------------------------------------------------
# Synthetic dataset (numpy-only - the model's own fallback rollout)
# ---------------------------------------------------------------------------


def test_synthetic_dataset_graph_structured(synthetic_bundle):
    items = list(model_mod.synthetic_dataset(synthetic_bundle, n_trials=6, T=9, vocab_size=6, seed=4))
    assert len(items) == 6
    spikes = np.zeros(6)
    for x, y_events, y_rates in items:
        assert x.shape == (1, 9, 6) and y_events.shape == (1, 9, 6)
        assert x.dtype == np.float32 and y_rates.dtype == np.float32
        assert set(np.unique(y_events)) <= {0.0, 1.0}
        spikes += y_events.sum(axis=(0, 1))
    # graph-structured: isolated node 4 never fires (undriven, no in-edges)
    assert spikes[4] == 0
    assert spikes[1] > 0, "post partner of driven sources should fire"
    # determinism
    again = list(model_mod.synthetic_dataset(synthetic_bundle, 6, 9, 6, seed=4))
    assert np.array_equal(items[0][0], again[0][0])


# ===========================================================================
# jax/flax model tests - every test below requires the `model` extra
# ===========================================================================


@pytest.fixture(scope="module")
def jax_mod():
    pytest.importorskip("jax", reason="model tests need jax[cpu]")
    pytest.importorskip("flax")
    pytest.importorskip("optax")
    import flygpt.model as m

    return m


@pytest.fixture(scope="module")
def tiny_cfg(synthetic_bundle):
    return model_mod.ModelConfig(
        d_model=32,
        n_heads=2,
        d_ff=64,
        n_layers=2,               # one time-axis block + one connectome block
        n_vocab=synthetic_bundle.n_nodes,
        max_in_degree=4,
        max_len=16,
        dropout=0.0,
        batch_trials=4,
        lr=3e-3,
        warmup_steps=0,
        total_steps=20,
        checkpoint_every=10,
        seed=1,
    )


@pytest.fixture(scope="module")
def edge_set(synthetic_bundle, tiny_cfg):
    return model_mod.edge_set_from_bundle(synthetic_bundle, tiny_cfg.max_in_degree)


@pytest.fixture(scope="module")
def built(jax_mod, tiny_cfg, edge_set):
    return jax_mod.build_model(tiny_cfg, edge_set)


@pytest.fixture(scope="module")
def train_data(jax_mod, synthetic_bundle):
    return list(
        jax_mod.synthetic_dataset(synthetic_bundle, n_trials=8, T=9, vocab_size=6, seed=0)
    )


def _x(seed=0, b=2, t=8, v=6):
    return np.random.default_rng(seed).random((b, t, v)).astype(np.float32)


def test_edge_set_contents(synthetic_bundle, edge_set):
    """The mask gather direction itself: row u = presyn partners of u."""
    idx, weight, sign, valid = edge_set
    assert idx.shape == (6, 4)
    # node 1's presyn set = {0, 2, 1, 3} (edges 0->1, 2->1, 1->1, 3->1)
    assert sorted(idx[1][valid[1]].tolist()) == [0, 1, 2, 3]
    # node 2's only presyn partner is 0 (edge 0->2)
    assert idx[2][valid[2]].tolist() == [0]
    # isolated node 4: all padding
    assert not valid[4].any() and (idx[4] == 4).all()
    # weights normalized to (0,1]; sign carried (0->2 edge is inhibitory)
    assert weight[1][valid[1]].max() <= 1.0
    assert sign[2][valid[2]].tolist() == [-1.0]


def test_edge_set_vocab_aggregation(synthetic_bundle, edge_set):
    """node_token maps the mask onto the vocab axis: identity mapping
    reproduces the node-space mask; pooled nodes aggregate weights,
    net-sign, and top-K partners."""
    n = 6
    ident = np.arange(n, dtype=np.uint32)
    ves = model_mod.edge_set_from_bundle(synthetic_bundle, 4, ident)
    idx, weight, sign, valid = edge_set
    vidx, vw, vs, vv = ves
    assert vidx.shape == (n, 4)
    np.testing.assert_array_equal(vidx, idx)
    np.testing.assert_allclose(vw, weight)
    np.testing.assert_array_equal(vs, sign)
    np.testing.assert_array_equal(vv, valid)
    # pool nodes {1,2} into token 0; token 1 = node 0 only; rest -> token 2
    tok = np.array([1, 0, 0, 2, 2, 2], dtype=np.uint32)
    ves2 = model_mod.edge_set_from_bundle(synthetic_bundle, 4, tok)
    vidx2, vw2, vs2, vv2 = ves2
    assert vidx2.shape == (3, 4)
    # token 0 pools post units 1,2: their presyn tokens are
    # {tok(0)=1, tok(2)=0(self), tok(1)=0(self), tok(3)=2} -> partners {0,1,2}
    assert sorted(vidx2[0][vv2[0]].tolist()) == [0, 1, 2]
    # weights sum across pooled edges: edge 0->1 (w5) and 0->2 (w5)
    # both land on pair (1,0); weight is the summed count
    assert vw2[0][vv2[0]].max() <= 1.0
    # token 1 (node 0) has no presyn partners
    assert not vv2[1].any()
    # length validation
    with pytest.raises(ValueError):
        model_mod.edge_set_from_bundle(synthetic_bundle, 4, np.zeros(3, np.uint32))


def test_forward_shapes(jax_mod, built, edge_set):
    m, params = built
    logits, rates = m.apply({"params": params}, _x(), edge_set, deterministic=True)
    assert logits.shape == (2, 8, 6) and rates.shape == (2, 8, 6)


def test_gnn_and_dense_shapes(jax_mod, tiny_cfg, edge_set):
    for arch in ("dense", "gnn"):
        cfg = dataclasses.replace(tiny_cfg, arch=arch)
        m, params = jax_mod.build_model(cfg, edge_set)
        logits, rates = m.apply({"params": params}, _x(), edge_set, deterministic=True)
        assert logits.shape == (2, 8, 6) and rates.shape == (2, 8, 6), arch


def test_t_model_9_mask_direction(jax_mod, built, edge_set):
    """T-MODEL-9 causality transpose check.

    Perturb node 0's input at t0=4. With edges acting pre(t)->post(t+1):
    - every logit at t < t0 is bitwise unchanged (strict past);
    - at t0 only node 0's own logits move (no same-step cross-unit leak);
    - at t0+1 exactly the postsynaptic partners {1,2} move via the mask
      (node 0 itself also moves via its own causal time path - allowed);
    - non-partners {3,4,5} stay unchanged at t0+1.
    """
    m, params = built
    t0 = 4
    x0 = _x(seed=2)
    x1 = x0.copy()
    x1[:, t0, 0] += 3.0
    l0, _ = m.apply({"params": params}, x0, edge_set, deterministic=True)
    l1, _ = m.apply({"params": params}, x1, edge_set, deterministic=True)
    d = np.abs(np.asarray(l1 - l0))
    assert d[:, :t0].max() < 1e-6, "past changed - non-causal leak"
    assert d[:, t0, 0].min() > 1e-3, "perturbed unit's own t0 logit must move"
    assert d[:, t0, 1:].max() < 1e-6, "same-step leak into other units"
    assert d[:, t0 + 1, 1].min() > 1e-4, "postsyn partner 1 unchanged at t0+1"
    assert d[:, t0 + 1, 2].min() > 1e-4, "postsyn partner 2 unchanged at t0+1"
    assert d[:, t0 + 1, 3:].max() < 1e-6, "non-partners moved at t0+1 - wrong direction"


def test_mask_swap_changes_postsynaptic_outputs(jax_mod, built, edge_set):
    """Same params, swapped edge set: outputs must differ on units whose
    gathered neighbors changed (T-MODEL-3's premise + T-SEX-5 machinery)."""
    m, params = built
    es_dense = model_mod.edge_set_from_bundle(controls.dense_mask_bundle(6, weight=5), 4)
    x = _x(seed=1)
    l_masked, _ = m.apply({"params": params}, x, edge_set, deterministic=True)
    l_dense, _ = m.apply({"params": params}, x, es_dense, deterministic=True)
    d = np.abs(np.asarray(l_dense - l_masked))
    # isolated-in-real-graph node 4 gets real neighbors under the dense mask
    assert d[:, 1:, 4].max() > 1e-4
    # t=0 rows unchanged everywhere (no t-1 exists for either mask)
    assert d[:, 0].max() < 1e-6


def test_sex_conditioned_forward_swaps_edge_set(jax_mod, built, edge_set):
    """T-SEX-5: one parameter set, two edge sets -> different outputs."""
    m, params = built
    es_female = model_mod.edge_set_from_bundle(controls.dense_mask_bundle(6, weight=5), 4)
    x = _x(seed=3)
    l_male, _ = jax_mod.sex_conditioned_forward(
        params, m, x, edge_set, es_female, "male"
    )
    l_female, _ = jax_mod.sex_conditioned_forward(
        params, m, x, edge_set, es_female, "female"
    )
    assert not np.allclose(np.asarray(l_male), np.asarray(l_female))
    # male pass must equal the direct masked forward
    l_direct, _ = m.apply({"params": params}, x, edge_set, deterministic=True)
    assert np.allclose(np.asarray(l_male), np.asarray(l_direct))


def test_param_match_dense_vs_masked(jax_mod, tiny_cfg):
    """T-MODEL-5 twin check: dense vs masked within the 2% tolerance."""
    cfg_dense = dataclasses.replace(tiny_cfg, arch="dense")
    cfg_masked = dataclasses.replace(tiny_cfg, arch="connectome")
    res = jax_mod.param_match(cfg_dense, cfg_masked)
    assert res["dense"] > 0 and res["masked"] > 0
    assert res["rel_diff"] <= 0.02 and res["matched"], res


def test_predict_events_probabilities(jax_mod, built, edge_set):
    m, params = built
    probs = jax_mod.predict_events(params, m, _x(), edge_set)
    assert probs.shape == (2, 8, 6)
    assert probs.min() >= 0.0 and probs.max() <= 1.0


def test_checkpoint_roundtrip(jax_mod, built, edge_set, tmp_path):
    m, params = built
    x = _x(seed=4)
    l0, r0 = m.apply({"params": params}, x, edge_set, deterministic=True)
    path = tmp_path / "ckpt_test.pkl"
    jax_mod.save_checkpoint(path, params, model_mod.ModelConfig(d_model=32), step=7)
    ck = jax_mod.load_checkpoint(path)
    assert ck["step"] == 7 and ck["cfg"]["d_model"] == 32
    l1, r1 = m.apply({"params": ck["params"]}, x, edge_set, deterministic=True)
    assert np.allclose(np.asarray(l0), np.asarray(l1))
    assert np.allclose(np.asarray(r0), np.asarray(r1))


def test_silence_weighted_loss(jax_mod):
    """EVAL 1.7 / T-MODEL-8: inactive units weighted below active."""
    y = np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    w = np.asarray(jax_mod.event_weights(y, 0.1))
    assert w[0, 1] == pytest.approx(1.0)
    assert w[0, 0] == pytest.approx(0.1) and w[1, 0] == pytest.approx(0.1)
    assert (w[y == 0] < w[y == 1]).all()


def test_train_smoke_checkpoint_resume(
    jax_mod, tiny_cfg, synthetic_bundle, edge_set, train_data, tmp_path
):
    """20 real optax steps on graph-structured data: finite, decreasing
    loss; checkpoint save + load identical; resume continues the step
    counter (the NORMAL path per plan session rules)."""
    cfg = dataclasses.replace(tiny_cfg, edge_set=edge_set)
    ckpt_dir = jax_mod.train(cfg, train_data, tmp_path)
    latest = ckpt_dir / "ckpt_latest.pkl"
    assert latest.exists()
    assert (ckpt_dir / "ckpt_00000020.pkl").exists()
    ck = jax_mod.load_checkpoint(latest)
    losses = ck["losses"]
    assert ck["step"] == 20 and len(losses) == 20
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}"

    # resume is normal: continue the SAME run to step 30
    cfg30 = dataclasses.replace(cfg, total_steps=30)
    jax_mod.train(cfg30, train_data, tmp_path)
    ck2 = jax_mod.load_checkpoint(latest)
    assert ck2["step"] == 30 and len(ck2["losses"]) == 30
    assert ck2["losses"][:20] == losses, "resumed history must continue, not restart"


def test_train_determinism(
    jax_mod, tiny_cfg, synthetic_bundle, edge_set, train_data, tmp_path
):
    """Same (cfg, seed, data) twice -> identical first-step loss and
    identical final params (CPU bitwise determinism)."""
    cfg = dataclasses.replace(tiny_cfg, total_steps=5, checkpoint_every=5, edge_set=edge_set)
    jax_mod.train(cfg, train_data, tmp_path / "a")
    jax_mod.train(cfg, train_data, tmp_path / "b")
    ca = jax_mod.load_checkpoint(tmp_path / "a" / "checkpoints" / "ckpt_latest.pkl")
    cb = jax_mod.load_checkpoint(tmp_path / "b" / "checkpoints" / "ckpt_latest.pkl")
    assert ca["losses"][0] == cb["losses"][0], "first-step loss not deterministic"
    leaves_a = jax_mod.jax.tree_util.tree_leaves(ca["params"])
    leaves_b = jax_mod.jax.tree_util.tree_leaves(cb["params"])
    assert len(leaves_a) == len(leaves_b)
    for la, lb in zip(leaves_a, leaves_b, strict=True):
        assert np.array_equal(np.asarray(la), np.asarray(lb)), "params diverged"


def test_count_params_positive(jax_mod, built):
    _, params = built
    assert jax_mod.count_params(params) > 0
