"""T-BUILD test battery - graph construction (can't ship a wrong brain).

Two layers:

1. Synthetic-fixture tests (``synthetic_build``) run the real pipeline on a
   hand-computed mini connectome - they fail if dedup, autapse policy,
   annotation filtering, NT sign rules, site dedup, or reconciliation
   regress. These run in CI without the 10 GB dataset.

2. Real-artifact tests verify the committed build outputs under
   ``data/graph/`` + ``provenance/graph_v1.json`` against the canonical
   raw files. They skip when artifacts are absent.

Plan refs: docs/plan.md T-BUILD-1..7 (lines ~148-158).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.feather as feather
import pytest

from flygpt.graph import KEY_TO_FIELD

REPO = Path(__file__).resolve().parents[1]
GRAPH_DIR = REPO / "data" / "graph"
RAW_DIR = REPO / "data" / "raw"
INGEST_MANIFEST = REPO / "provenance" / "malecns_v1.json"
GRAPH_MANIFEST = REPO / "provenance" / "graph_v1.json"
RESULTS_REPORT = REPO / "results" / "graph_build_v1.json"

ANN_FILE = RAW_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"

# Ground truth measured on the canonical file itself (2026-09-22):
# 151,856,684 rows, all (pre,post) unique; weight sums to 311,833,243 =
# syn-partners row count exactly (weight = partnership count).
CANONICAL_WEIGHTS_ROWS = 151_856_684
CANONICAL_WEIGHT_SUM = 311_833_243
CANONICAL_RAW_EDGES_AT_CUTOFF = {"3": 23_014_406, "5": 7_622_864, "10": 2_799_910}
CANONICAL_RAW_AUTAPSES = 123
# Published convention cross-check: ~25.6M connections among traced
# neurons (audit: flybrain baked graph; cephalochromoscope gives the
# precise figure 25,582,938). Tolerance because "traced" conventions vary.
PUBLISHED_TRACED_EDGES = 25_582_938

real_data = pytest.mark.skipif(
    not (GRAPH_DIR / "nodes.parquet").exists()
    or not (GRAPH_DIR / "edges.parquet").exists()
    or not GRAPH_MANIFEST.exists(),
    reason="built graph artifacts absent - run `python -m flygpt.graph` first",
)


# ---------------------------------------------------------------------------
# Synthetic-fixture tests (CI-runnable)
# ---------------------------------------------------------------------------


def _synthetic_edges(build):
    return pl.read_parquet(build["root"] / "graph" / "edges.parquet")


def _edge(df, pre_id, post_id, nodes):
    pre_idx = nodes.filter(pl.col("bodyId") == pre_id)["node_idx"][0]
    post_idx = nodes.filter(pl.col("bodyId") == post_id)["node_idx"][0]
    rows = df.filter((pl.col("pre_idx") == pre_idx) & (pl.col("post_idx") == post_idx))
    assert rows.height == 1, f"edge ({pre_id},{post_id}) not found exactly once"
    return rows.to_dicts()[0]


def test_synthetic_node_universe(synthetic_build):
    nodes = pl.read_parquet(synthetic_build["root"] / "graph" / "nodes.parquet")
    assert nodes.height == 6
    assert nodes["bodyId"].is_unique().all()
    assert sorted(nodes["bodyId"].to_list()) == [10, 11, 12, 13, 14, 15]
    # isolated annotated body 14 retained as a zero-degree node
    row14 = nodes.filter(pl.col("bodyId") == 14).to_dicts()[0]
    assert row14["out_degree"] == 0 and row14["in_degree"] == 0


def test_synthetic_edge_filter_dedup(synthetic_build):
    df = _synthetic_edges(synthetic_build)
    nodes = pl.read_parquet(synthetic_build["root"] / "graph" / "nodes.parquet")
    # 6 annotated rows -> 5 unique edges after the (10,11) merge
    assert df.height == 5
    # unannotated endpoints dropped: no edge references bodies 888/999
    ids = set(nodes["bodyId"].to_list())
    assert 888 not in ids and 999 not in ids
    e = _edge(df, 10, 11, nodes)
    assert e["weight"] == 7  # 6 + 1 merged duplicate
    assert e["derived_weight"] == 7
    counts = synthetic_build["report"]["checks"]
    assert counts["duplicate_pairs_merged"] == 1
    manifest = json.loads(
        (synthetic_build["root"] / "provenance" / "graph_test.json").read_text()
    )
    assert manifest["counts"]["edge_rows_dropped_unannotated"] == 2
    assert manifest["counts"]["weight_sum_dropped"] == 11  # 7 + 4


def test_synthetic_autapse_and_cutoffs(synthetic_build):
    df = _synthetic_edges(synthetic_build)
    nodes = pl.read_parquet(synthetic_build["root"] / "graph" / "nodes.parquet")
    auto = _edge(df, 11, 11, nodes)
    assert auto["weight"] == 5
    manifest = json.loads(
        (synthetic_build["root"] / "provenance" / "graph_test.json").read_text()
    )
    assert manifest["counts"]["autapses_kept"] == 1
    assert manifest["counts"]["edges_at_cutoff"] == {"3": 5, "5": 4, "10": 0}
    # raw-file convention counts include dropped + duplicate rows
    assert manifest["counts"]["raw_edges_at_cutoff"] == {"3": 7, "5": 5, "10": 0}


def test_synthetic_nt_signs(synthetic_build):
    df = _synthetic_edges(synthetic_build)
    nodes = pl.read_parquet(synthetic_build["root"] / "graph" / "nodes.parquet")

    e = _edge(df, 10, 11, nodes)  # 2 inhib / 4 classifiable = 0.5 -> not >0.5
    assert e["nt_sign"] == 1 and e["n_sites_classifiable"] == 4
    assert e["nt_inhibitory_fraction"] == pytest.approx(0.5)

    e = _edge(df, 10, 12, nodes)  # 3/4 inhibitory
    assert e["nt_sign"] == -1 and e["nt_inhibitory_fraction"] == pytest.approx(0.75)

    e = _edge(df, 12, 11, nodes)  # u1 unclassifiable -> denominator excludes it
    assert e["nt_sign"] == 1 and e["n_sites_classifiable"] == 1 and e["n_sites"] == 2

    e = _edge(df, 11, 11, nodes)  # autapse, single site x5 partnerships
    assert e["nt_sign"] == -1 and e["n_sites"] == 1 and e["weight"] == 5

    e = _edge(df, 13, 11, nodes)  # zero classifiable sites -> flagged, no guess
    assert e["nt_sign"] == 0 and bool(e["nt_low_confidence"])


def test_synthetic_neuron_signs_and_disagreement(synthetic_build):
    nodes = pl.read_parquet(synthetic_build["root"] / "graph" / "nodes.parquet")
    by_body = {r["bodyId"]: r for r in nodes.to_dicts()}
    # body 10: 5 inhibitory of 8 classifiable sites -> inhibitory neuron,
    # while its edge to 11 is excitatory -> per-edge/per-neuron divergence.
    assert by_body[10]["nt_sign_neuron"] == -1
    assert by_body[11]["nt_sign_neuron"] == -1
    assert by_body[12]["nt_sign_neuron"] == 1
    assert by_body[13]["nt_sign_neuron"] == 0  # zero classifiable sites
    manifest = json.loads(
        (synthetic_build["root"] / "provenance" / "graph_test.json").read_text()
    )
    assert manifest["nt"]["edge_vs_neuron_sign_disagree"] == 1


def test_synthetic_weight_reconciliation(synthetic_build):
    df = _synthetic_edges(synthetic_build)
    assert (df["weight"] == df["derived_weight"]).all()
    report = synthetic_build["report"]["checks"]["derived_weight_reconciliation"]
    assert report["mismatch_edges"] == 0
    manifest = json.loads(
        (synthetic_build["root"] / "provenance" / "graph_test.json").read_text()
    )
    assert manifest["counts"]["syn_rows_total"] == 30
    assert manifest["counts"]["syn_rows_unannotated_endpoint"] == 1  # (999,888)
    assert manifest["counts"]["syn_rows_annotated_no_weights_entry"] == 1  # (15,10)
    assert manifest["counts"]["syn_rows_site_unmatched"] == 1
    assert manifest["counts"]["unique_site_post_pairs"] == 14


def test_synthetic_csr_structure(synthetic_build):
    graph_dir = synthetic_build["root"] / "graph"
    csr = np.load(graph_dir / "csr_cutoff5.npz")
    nodes = pl.read_parquet(graph_dir / "nodes.parquet")
    n = nodes.height
    assert csr["indptr"].shape == (n + 1,)
    assert int(csr["indptr"][-1]) == csr["indices"].size == 4
    assert (np.diff(csr["indptr"]) >= 0).all()
    assert (csr["indices"] < n).all()
    assert (csr["weight"] >= 5).all()
    # every CSR edge exists in the edge table at this cutoff
    df = _synthetic_edges(synthetic_build).filter(pl.col("weight") >= 5)
    assert df.height == csr["indices"].size
    # CSR row i's edges all have pre == i
    pre_of_edge = np.repeat(np.arange(n), np.diff(csr["indptr"]))
    edge_pre = df.sort(["pre_idx", "post_idx"])["pre_idx"].to_numpy()
    assert (pre_of_edge == edge_pre).all()


# ---------------------------------------------------------------------------
# Real-artifact tests - T-BUILD-1..7 against data/graph + provenance/
# ---------------------------------------------------------------------------


@real_data
def test_t_build_1_node_universe_matches_annotations():
    """T-BUILD-1: node set = body-annotations row count; isolated retained."""
    ann_rows = feather.read_table(ANN_FILE, columns=["bodyId"]).num_rows
    nodes = pl.read_parquet(GRAPH_DIR / "nodes.parquet", columns=["bodyId", "node_idx"])
    assert nodes.height == ann_rows
    assert nodes["bodyId"].is_unique().all()
    assert nodes["node_idx"].to_numpy().tolist() == list(range(ann_rows))


@real_data
def test_t_build_2_edge_counts_by_convention():
    """T-BUILD-2: counts recorded for 3/5/10 and self-consistent with CSR."""
    manifest = json.loads(GRAPH_MANIFEST.read_text())
    edges = pl.read_parquet(GRAPH_DIR / "edges.parquet", columns=["weight"])
    for c in ("3", "5", "10"):
        csr = np.load(GRAPH_DIR / f"csr_cutoff{c}.npz")
        expect = int((edges["weight"] >= int(c)).sum())
        assert manifest["counts"]["edges_at_cutoff"][c] == expect == csr["indices"].size
        # canonical raw-file counts are pinned to the measured ground truth
        assert manifest["counts"]["raw_edges_at_cutoff"][c] == CANONICAL_RAW_EDGES_AT_CUTOFF[c]
    # published ~25.6M traced-neuron connection convention reconciles
    traced = manifest["counts"]["edges_both_traced"]
    assert abs(traced - PUBLISHED_TRACED_EDGES) / PUBLISHED_TRACED_EDGES < 0.05


@real_data
def test_t_build_3_autapse_policy():
    """T-BUILD-3: autapses kept, counted, consistent across artifacts."""
    manifest = json.loads(GRAPH_MANIFEST.read_text())
    edges = pl.read_parquet(GRAPH_DIR / "edges.parquet", columns=["pre_idx", "post_idx"])
    n_auto = int((edges["pre_idx"] == edges["post_idx"]).sum())
    assert manifest["counts"]["autapses_kept"] == n_auto
    assert manifest["conventions"]["autapse_policy"] == "keep"
    # raw canonical file carries 123; the annotated-graph count must be <= it
    assert 0 <= n_auto <= CANONICAL_RAW_AUTAPSES


@real_data
def test_t_build_4_nt_sign_rule_consistency():
    """T-BUILD-4: every stored edge sign follows the documented rule."""
    edges = pl.read_parquet(
        GRAPH_DIR / "edges.parquet",
        columns=["nt_sign", "nt_inhibitory_fraction", "n_sites_classifiable", "nt_low_confidence"],
    )
    sign = edges["nt_sign"].to_numpy()
    frac = edges["nt_inhibitory_fraction"].to_numpy()
    n_class = edges["n_sites_classifiable"].to_numpy()
    low = edges["nt_low_confidence"].to_numpy()

    determined = n_class > 0
    assert (sign[determined] != 0).all()
    assert int((sign[determined] == -1).sum()) == int((frac[determined] > 0.5).sum())
    assert (sign[~determined] == 0).all()
    assert (low == ~determined).all()
    # sanity: inhibitory edge share is in the GABA+Glu plausible band
    share = float((sign[determined] == -1).mean())
    assert 0.15 < share < 0.65


@real_data
@pytest.mark.slow
def test_t_build_4b_nt_sign_spot_recompute():
    """Re-derive NT signs for a random edge sample straight from the raw
    files - the internal-verification check, not just stored consistency.
    Reuses the real classify_sites classifier but recounts site membership
    independently via a filtered polars scan of syn-partners."""
    from flygpt.graph import (
        BuildStats,
        classify_sites,
        load_config,
        resolve_sources,
    )

    rng = np.random.default_rng(20260922)
    manifest = json.loads(INGEST_MANIFEST.read_text())
    cfg = load_config(REPO / "configs" / "graph_v1.yaml")
    sources = resolve_sources(manifest, RAW_DIR)

    nodes = pl.read_parquet(GRAPH_DIR / "nodes.parquet", columns=["bodyId"])
    body_ids = np.ascontiguousarray(nodes["bodyId"].to_numpy().astype(np.int64))
    edges = pl.read_parquet(
        GRAPH_DIR / "edges.parquet",
        columns=["pre_idx", "post_idx", "nt_sign", "n_sites_classifiable"],
    )
    sample = rng.choice(edges.height, size=min(64, edges.height), replace=False)
    pre_col = edges["pre_idx"].to_list()
    post_col = edges["post_idx"].to_list()
    sign_col = edges["nt_sign"].to_list()
    nclass_col = edges["n_sites_classifiable"].to_list()
    targets = {
        (pre_col[int(i)], post_col[int(i)]) for i in sample
    }
    target_bodies = sorted({(int(body_ids[p]), int(body_ids[q])) for p, q in targets})

    sites = classify_sites(sources, body_ids, cfg, BuildStats())

    pre_set = sorted({p for p, _ in target_bodies})
    post_set = sorted({q for _, q in target_bodies})
    # superset filter on each side; exact-pair bucketing happens in recount
    hits = (
        pl.scan_ipc(sources.syn_partners)
        .filter(pl.col("body_pre").is_in(pre_set) & pl.col("body_post").is_in(post_set))
        .select(["x_pre", "y_pre", "z_pre", "body_pre", "body_post"])
        .collect(engine="streaming")
    )
    site_idx = sites.lookup(
        hits["x_pre"].to_numpy().astype(np.int64),
        hits["y_pre"].to_numpy().astype(np.int64),
        hits["z_pre"].to_numpy().astype(np.int64),
    )
    hits = hits.with_columns(pl.Series("site_idx", site_idx))

    recount: dict[tuple[int, int], set[int]] = {}
    for row in hits.iter_rows(named=True):
        if row["site_idx"] >= 0:
            recount.setdefault((row["body_pre"], row["body_post"]), set()).add(
                int(row["site_idx"])
            )

    for i in sample:
        pre_idx, post_idx = pre_col[int(i)], post_col[int(i)]
        pair = (int(body_ids[pre_idx]), int(body_ids[post_idx]))
        site_set = recount.get(pair, set())
        flags = sites.flag[np.fromiter(site_set, dtype=np.int64)] if site_set else []
        n_class = int(np.sum(np.asarray(flags) > 0)) if len(flags) else 0
        n_inhib = int(np.sum(np.asarray(flags) == 1)) if len(flags) else 0
        expected = 0 if n_class == 0 else (-1 if n_inhib / n_class > 0.5 else 1)
        assert sign_col[int(i)] == expected, f"edge {pair} sign mismatch"
        assert nclass_col[int(i)] == n_class


@real_data
def test_t_build_5_checksums_match_manifest():
    """T-BUILD-5: sha256 of every required upstream file matches manifest."""
    manifest = json.loads(INGEST_MANIFEST.read_text())
    recorded = {f["key"]: f for f in manifest["files"]}
    for key in KEY_TO_FIELD:
        entry = recorded[key]
        assert entry["sha256"], f"{key} has no recorded sha256"
        path = RAW_DIR / entry["local_name"]
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
                h.update(block)
        assert h.hexdigest() == entry["sha256"], f"{key} sha256 mismatch"


@real_data
def test_t_build_6_no_duplicate_edges():
    """T-BUILD-6: no repeated (pre, post) pairs in the built edge table."""
    edges = pl.read_parquet(GRAPH_DIR / "edges.parquet", columns=["pre_idx", "post_idx"])
    assert edges.height == edges.select(["pre_idx", "post_idx"]).unique().height


@real_data
def test_t_build_7_type_coverage_reported():
    """T-BUILD-7: typed fraction in manifest equals recomputed; untyped
    neurons are retained as nodes (UNK handling happens at tokenize time
    on this same node table - they must not be silently dropped)."""
    manifest = json.loads(GRAPH_MANIFEST.read_text())
    nodes = pl.read_parquet(GRAPH_DIR / "nodes.parquet", columns=["type"])
    n_typed = int(nodes["type"].is_not_null().sum())
    assert manifest["counts"]["nodes_typed"] == n_typed
    assert manifest["counts"]["nodes"] == nodes.height
    assert nodes["type"].null_count() > 0  # untyped neurons retained, not dropped


@real_data
def test_weight_reconciliation_zero_mismatch():
    """Derived weights (syn-partners row counts) reproduce canonical
    connectome-weights exactly for every annotated edge."""
    report = json.loads(RESULTS_REPORT.read_text())
    recon = report["checks"]["derived_weight_reconciliation"]
    assert recon["mismatch_edges"] == 0
    assert recon["total_abs_delta"] == 0


@real_data
def test_csr_edges_exist_in_table():
    """CSR cutoff-5 edges are exactly the edge-table rows at that cutoff."""
    edges = pl.read_parquet(GRAPH_DIR / "edges.parquet", columns=["weight"])
    nodes = pl.read_parquet(GRAPH_DIR / "nodes.parquet", columns=["bodyId"])
    csr = np.load(GRAPH_DIR / "csr_cutoff5.npz")
    assert csr["indices"].size == int((edges["weight"] >= 5).sum())
    assert int(csr["n_nodes"]) == nodes.height


def test_unique_accum_counts_new_additions_not_total():
    """_UniqueAccum must not re-sort the whole accumulator on every add.

    Regression: a merge used to set pending=merged.size, so once the
    accumulator crossed merge_at it re-triggered a full sort every few
    hundred keys - quadratic on real streams (the actual graph build
    stalled ~28h in _unique1d merges).
    """
    from flygpt.graph import _UniqueAccum

    acc = _UniqueAccum(merge_at=100)
    rng = np.random.default_rng(0)
    # Feed mostly-unique keys in small batches; merge must only fire on
    # NEW-key volume crossing the threshold, not on accumulator size.
    for i in range(50):
        acc.add(rng.integers(0, 1 << 60, size=10).astype(np.uint64)
                + np.uint64(i << 48))
    # 50x10=500 unique keys total; merges happen at pending>=100, i.e.
    # ~5 merges, each sorting only the fresh slice + acc - never the
    # ~500-key accumulator repeatedly.
    res = acc.result()
    assert res.size == np.unique(res).size
    assert acc._pending < 100
    # semantic: result = union of all added keys
    acc2 = _UniqueAccum()
    keys = np.arange(1 << 20, dtype=np.uint64)
    acc2.add(keys)
    acc2.add(keys[:100])          # duplicates collapse
    acc2.add(keys + (1 << 40))    # disjoint range
    out = acc2.result()
    assert out.size == (1 << 20) * 2
