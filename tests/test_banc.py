"""BANC module tests - synthetic female-connectome fixture (T-SEX aligned).

A local BANC-like fixture is authored below as polars/pyarrow tables
written to feather/csv in tmp dirs - no network, no real dataset:

- 13 meta rows (banc_888_id strings "1001".."1013"): aligned types
  (A_R x2, B_L x2, DIM1, C_M via manc fallback, OVR1 via reviewed-match
  override), female-only types (FSP1 labeled female-specific, FUN1, E_L,
  ISO1, REJ1 name-colliding with a male type), one untyped row (1007,
  retained as a graph node, excluded from alignment).
- 16 edgelist rows (string pre/post + count/norm/post_count/pre_count):
  one unannotated endpoint (9999, dropped), one duplicate pair
  (1003->1001 merged 9+2=11), two below-cutoff rows (c4, c3), NO
  autapses (BANC excludes them at build - the build must report 0 and
  still enforce the policy).
- reviewed-matches csv: accepted override (1009 WRONG_M -> OVR_M) and a
  rejection (1010 REJ1 -> unmatched).

Plan refs: docs/plan.md Phase 4, EVAL.md T-SEX-4/5, RISKS.md R9.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.feather as feather
import pytest

from flygpt.banc import (
    BANC_REGISTRY,
    BancBuildStats,
    BancSourcePaths,
    alignment_table,
    build_banc_graph,
    download_banc,
    female_graph_in_male_space,
    female_stimulus_map,
    gcs_url,
    graph_manifest_name,
    load_banc_edges,
    resolve_banc_sources,
    run_align,
    sex_conditioned_vocab,
)
from flygpt.schema import TYPE_TOKEN_BASE, bundle_from_edges, load_graph_bundle

# ---------------------------------------------------------------------------
# Synthetic BANC fixture - expectations computed by hand
# ---------------------------------------------------------------------------

# (banc_888_id, cell_type, malecns_cell_type, manc_cell_type, predicted NT)
META_ROWS = [
    ("1001", "A_R", "A_R", None, "acetylcholine"),
    ("1002", "A_R", "A_R", None, "acetylcholine"),
    ("1003", "B_L", "B_L", None, "gaba"),
    ("1004", "DIM1", "DIM1", None, "glutamate"),
    ("1005", "FSP1", None, None, "acetylcholine"),
    ("1006", "FUN1", None, None, "gaba"),
    ("1007", None, None, None, "acetylcholine"),      # untyped node
    ("1008", "C_M", None, "C_M", "acetylcholine"),   # manc fallback align
    ("1009", "OVR1", "WRONG_M", None, "gaba"),       # reviewed -> OVR_M
    ("1010", "REJ1", "REJ1", None, "acetylcholine"), # reviewed -> rejected
    ("1011", "B_L", "B_L", None, "gaba"),
    ("1012", "E_L", None, None, None),               # no NT -> sign 0
    ("1013", "ISO1", None, None, "acetylcholine"),   # isolated node
]

# (pre, post, count) - string ids; 9999 absent from meta; last row is a
# duplicate of (1003->1001) merged at build.
EDGE_ROWS = [
    ("1001", "1003", 7),
    ("1002", "1003", 5),
    ("1003", "1001", 9),
    ("1001", "1004", 6),
    ("1004", "1005", 8),
    ("1005", "1006", 5),
    ("1006", "1005", 4),   # below read-time cutoff 5
    ("1007", "1003", 6),   # untyped pre
    ("1003", "1007", 6),   # untyped post
    ("1008", "1001", 7),
    ("1009", "1004", 5),
    ("1010", "1005", 6),
    ("1011", "1003", 6),   # B_L -> B_L intra-type at type level
    ("1012", "1001", 3),   # below cutoff AND undetermined-NT pre
    ("1001", "9999", 5),   # post absent from meta -> dropped
    ("1003", "1001", 2),   # duplicate of row 3 -> merged weight 11
]

MATCHES_ROWS = [
    ("1009", "OVR_M", "accepted"),   # curated override wins over WRONG_M
    ("1010", "REJ1", "rejected"),    # curated rejection unmatches
]

# Male-annotation dimorphism labels (shared type-name namespace).
DIMORPHISM = {
    "A_R": "isomorphic",
    "B_L": "isomorphic",
    "C_M": "isomorphic",
    "DIM1": "dimorphic",
    "OVR_M": "isomorphic",
    "MSP1": "male-specific",
    "MUN1": "isomorphic",            # unmatched -> male-only-unmatched
    "FSP1": "female-specific",       # labels the banc-only type
}

MALE_TYPES = ["A_R", "B_L", "C_M", "DIM1", "MSP1", "MUN1", "OVR_M", "REJ1"]

BANC_CFG = {
    "version": "banc_test",
    "dataset": "banc:test",
    "snapshot": "v888",
    "synapse_snapshot": "v2",
    "conventions": {
        "autapse_policy": "excluded_at_build",
        "edge_columns": {"pre": "pre", "post": "post", "weight": "count"},
        "norm_column": "norm",
        "read_time_cutoff": 5,
        "node_universe": "meta rows joined on banc_888_id",
    },
    "alignment": {
        "primary": "malecns_cell_type",
        "secondary": "manc_cell_type",
        "iso_classes": [
            "isomorphic",
            "dimorphic",
            "male-specific",
            "female-specific",
            "male-only-unmatched",
            "female-only-unmatched",
        ],
    },
}


def _meta_df() -> pl.DataFrame:
    ids = [r[0] for r in META_ROWS]
    return pl.DataFrame(
        {
            "banc_888_id": ids,
            "root_888": [int(i) for i in ids],
            "super_class": ["central"] * len(ids),
            "cell_class": ["intrinsic"] * len(ids),
            "cell_sub_class": ["sub"] * len(ids),
            "cell_type": [r[1] for r in META_ROWS],
            "hemilineage": ["h"] * len(ids),
            "region": ["brain"] * len(ids),
            "side": ["L"] * len(ids),
            "flow": ["intrinsic"] * len(ids),
            "proofread": ["TRUE"] * len(ids),
            "position": ["1, 2, 3"] * len(ids),
            "fafb_cell_type": [None] * len(ids),
            "manc_cell_type": [r[3] for r in META_ROWS],
            "malecns_cell_type": [r[2] for r in META_ROWS],
            "neurotransmitter_predicted": [r[4] for r in META_ROWS],
            "neurotransmitter_score": [0.9] * len(ids),
            "neurotransmitter_verified": ["FALSE"] * len(ids),
        }
    )


def _edges_df() -> pl.DataFrame:
    post_total: dict[str, int] = {}
    pre_total: dict[str, int] = {}
    for p, q, c in EDGE_ROWS:
        post_total[q] = post_total.get(q, 0) + c
        pre_total[p] = pre_total.get(p, 0) + c
    return pl.DataFrame(
        {
            "pre": [p for p, _, _ in EDGE_ROWS],
            "post": [q for _, q, _ in EDGE_ROWS],
            "count": [c for _, _, c in EDGE_ROWS],
            "norm": [c / post_total[q] for _, q, c in EDGE_ROWS],
            "post_count": [post_total[q] for _, q, _ in EDGE_ROWS],
            "pre_count": [pre_total[p] for p, _, _ in EDGE_ROWS],
        },
        schema={
            "pre": pl.Utf8,
            "post": pl.Utf8,
            "count": pl.Int32,
            "norm": pl.Float64,
            "post_count": pl.Int32,
            "pre_count": pl.Int32,
        },
    )


def _write_fixture(raw: Path) -> BancSourcePaths:
    feather.write_feather(_meta_df().to_arrow(), raw / "banc_888_meta.feather")
    feather.write_feather(
        _edges_df().to_arrow(), raw / "banc_888_edgelist_simple_v2.feather"
    )
    pl.DataFrame(
        {
            "banc_888_id": [r[0] for r in MATCHES_ROWS],
            "malecns_cell_type": [r[1] for r in MATCHES_ROWS],
            "decision": [r[2] for r in MATCHES_ROWS],
        }
    ).write_csv(raw / "banc_malecns_reviewed_matches.csv")
    return BancSourcePaths(
        meta=raw / "banc_888_meta.feather",
        edgelist=raw / "banc_888_edgelist_simple_v2.feather",
        malecns_matches=raw / "banc_malecns_reviewed_matches.csv",
    )


@pytest.fixture(scope="session")
def banc_sources(tmp_path_factory):
    raw = tmp_path_factory.mktemp("banc_raw")
    return _write_fixture(raw)


@pytest.fixture(scope="session")
def banc_build(tmp_path_factory, banc_sources):
    """Run the real build_banc_graph on the fixture once per session."""
    root = tmp_path_factory.mktemp("banc_build")
    report = build_banc_graph(
        dict(BANC_CFG),
        banc_sources,
        graph_dir=root / "graph",
        prov_dir=root / "provenance",
        results_dir=root / "results",
        input_manifest=None,
        verify_hashes=False,
    )
    return {"root": root, "report": report, "sources": banc_sources}


@pytest.fixture(scope="session")
def banc_bundle(banc_build):
    return load_graph_bundle(banc_build["root"] / "graph", cutoff=5)


@pytest.fixture(scope="session")
def male_bundle():
    """Tiny male-side GraphBundle exposing the male manifest types."""
    nodes = pl.DataFrame(
        {"bodyId": list(range(1, len(MALE_TYPES) + 1)), "type": MALE_TYPES}
    )
    return bundle_from_edges(
        nodes,
        np.array([0, 1], dtype=np.uint32),
        np.array([1, 2], dtype=np.uint32),
        np.array([7, 6], dtype=np.int32),
        np.array([1, -1], dtype=np.int8),
        cutoff=5,
    )


@pytest.fixture(scope="session")
def align(banc_sources):
    meta = pl.from_arrow(feather.read_table(banc_sources.meta))
    matches = pl.read_csv(banc_sources.malecns_matches)
    return alignment_table(meta, matches, DIMORPHISM)


@pytest.fixture(scope="session")
def vocab(male_bundle, banc_bundle, align):
    return sex_conditioned_vocab(male_bundle, banc_bundle, align)


def _edge_row(df: pl.DataFrame, pre_id: str, post_id: str, nodes: pl.DataFrame) -> dict:
    pre_idx = nodes.filter(pl.col("banc_888_id") == pre_id)["node_idx"][0]
    post_idx = nodes.filter(pl.col("banc_888_id") == post_id)["node_idx"][0]
    rows = df.filter(
        (pl.col("pre_idx") == pre_idx) & (pl.col("post_idx") == post_idx)
    )
    assert rows.height == 1, f"edge ({pre_id},{post_id}) not found exactly once"
    return rows.to_dicts()[0]


def _type_edges(bundle) -> dict[tuple[int, int], tuple[int, int]]:
    n = bundle.n_nodes
    pre = np.repeat(np.arange(n), np.diff(bundle.out_indptr))
    return {
        (int(p), int(q)): (int(w), int(s))
        for p, q, w, s in zip(
            pre, bundle.out_indices, bundle.out_weight, bundle.out_sign, strict=True
        )
    }


# ---------------------------------------------------------------------------
# Registry + download (mocked fetch - no network)
# ---------------------------------------------------------------------------


def test_registry_shape():
    keys = {f.key for f in BANC_REGISTRY}
    assert keys == {
        "meta",
        "edgelist_v2",
        "edgelist_v3",
        "nt_prediction",
        "malecns_matches",
        "malecns_nblast",
    }
    required = {f.key for f in BANC_REGISTRY if f.required}
    assert required == {"meta", "edgelist_v2"}
    by_key = {f.key: f for f in BANC_REGISTRY}
    assert by_key["meta"].name.endswith("banc_888_meta.feather")
    assert by_key["edgelist_v2"].name.endswith("edgelist_simple_v2.feather")
    assert by_key["edgelist_v3"].name.endswith("edgelist_simple_v3.feather")
    assert by_key["malecns_matches"].name == "nblast/banc_malecns_reviewed_matches.csv"
    assert by_key["malecns_nblast"].name.startswith("nblast/")


def test_download_banc_mocked(tmp_path):
    payload = {
        f.name: f"{f.key}-fixture-bytes-{i}".encode() * (i + 3)
        for i, f in enumerate(BANC_REGISTRY)
    }
    listing_meta = {
        name: {
            "size": len(data),
            "md5_b64": base64.b64encode(hashlib.md5(data).digest()).decode(),
        }
        for name, data in payload.items()
    }

    def fake_listing(prefix: str):
        return {n: m for n, m in listing_meta.items() if n.startswith(prefix)}

    class _Resp:
        def __init__(self, data: bytes):
            self._data = data
            self.status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            return None

        def iter_content(self, n):
            yield self._data

    def fake_get(url, headers=None, stream=None, timeout=None):
        name = url.split("storage.googleapis.com/", 1)[1].split("/", 1)[1]
        return _Resp(payload[name])

    dest = tmp_path / "raw" / "banc"
    mpath = tmp_path / "provenance" / "banc_v888.json"
    manifest = download_banc(None, dest, mpath, listing=fake_listing, getter=fake_get)

    assert mpath.exists()
    on_disk = json.loads(mpath.read_text())
    assert on_disk == manifest
    assert manifest["dataset"] == "banc:v888"
    conv = manifest["conventions"]
    assert conv["snapshot"] == "v888"
    assert conv["synapse_snapshot"] == "v2"
    assert conv["autapse_policy"] == "excluded_at_build"

    files = {f["key"]: f for f in manifest["files"]}
    for key in ("meta", "edgelist_v2"):
        entry = files[key]
        data = payload[next(f.name for f in BANC_REGISTRY if f.key == key)]
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["bytes"] == len(data)
        assert (dest / entry["local_name"]).read_bytes() == data
    for key in ("edgelist_v3", "nt_prediction", "malecns_matches", "malecns_nblast"):
        assert files[key]["not_downloaded"] is True
        assert files[key]["sha256"] is None
        assert files[key]["bytes"] is not None  # size recorded from listing


def test_list_bucket_prefix_url():
    # only the URL composition is checked here; live listing is exercised
    # implicitly by the mocked download path above.
    for f in BANC_REGISTRY:
        assert gcs_url(f.name).endswith(f.name)


# ---------------------------------------------------------------------------
# Graph build
# ---------------------------------------------------------------------------


def test_banc_node_universe(banc_build):
    nodes = pl.read_parquet(banc_build["root"] / "graph" / "nodes.parquet")
    assert nodes.height == len(META_ROWS) == 13
    assert nodes["banc_888_id"].is_unique().all()
    # sorted by string id -> deterministic node_idx
    assert nodes["banc_888_id"].to_list() == sorted(r[0] for r in META_ROWS)
    assert nodes["node_idx"].to_list() == list(range(13))
    assert set(nodes.columns) >= {
        "super_class",
        "cell_class",
        "cell_sub_class",
        "cell_type",
        "type",
        "region",
        "side",
        "flow",
        "proofread",
        "position",
        "neurotransmitter_predicted",
        "malecns_cell_type",
        "manc_cell_type",
        "nt_sign_neuron",
        "out_degree",
        "in_degree",
    }
    by_id = {r["banc_888_id"]: r for r in nodes.to_dicts()}
    # isolated node 1013 retained as zero-degree (same policy as male build)
    assert by_id["1013"]["out_degree"] == 0
    assert by_id["1013"]["in_degree"] == 0
    # per-neuron sign from predicted NT
    assert by_id["1003"]["nt_sign_neuron"] == -1  # gaba
    assert by_id["1004"]["nt_sign_neuron"] == -1  # glutamate
    assert by_id["1001"]["nt_sign_neuron"] == 1   # acetylcholine
    assert by_id["1012"]["nt_sign_neuron"] == 0   # no prediction
    assert bool(by_id["1012"]["nt_low_confidence_neuron"])


def test_banc_edge_join_dedup_drop(banc_build):
    edges = pl.read_parquet(banc_build["root"] / "graph" / "edges.parquet")
    nodes = pl.read_parquet(banc_build["root"] / "graph" / "nodes.parquet")
    # 16 raw - 1 unannotated endpoint - 1 merged duplicate = 14 edges
    assert edges.height == 14
    # no edge references the absent 9999 id
    ids = set(nodes["banc_888_id"].to_list())
    assert "9999" not in ids
    e = _edge_row(edges, "1003", "1001", nodes)
    assert e["weight"] == 11  # 9 + 2 merged duplicate
    assert e["post_count"] == 21
    report = banc_build["report"]["checks"]
    assert report["edge_rows_dropped_unannotated"] == 1
    assert report["duplicate_pairs_merged"] == 1
    assert report["autapses_found"] == 0
    manifest = json.loads(
        (banc_build["root"] / "provenance" / "banc_graph_test.json").read_text()
    )
    assert manifest["counts"]["weight_sum_dropped"] == 5


def test_banc_autapse_policy_enforced(tmp_path):
    # the fixture ships none (BANC excludes them at build); a residual
    # autapse row must still be dropped AND counted.
    feather.write_feather(
        _meta_df().to_arrow(), tmp_path / "meta.feather"
    )
    rows = _edges_df().to_dicts()
    rows.append(
        {"pre": "1003", "post": "1003", "count": 9, "norm": 0.5,
         "post_count": 24, "pre_count": 17}
    )
    feather.write_feather(
        pl.DataFrame(rows).to_arrow(), tmp_path / "edges.feather"
    )
    stats = BancBuildStats()
    nodes = _meta_df().sort("banc_888_id").with_row_index("node_idx")
    ids = np.ascontiguousarray(nodes["banc_888_id"].to_numpy().astype(str))
    pre, post, weight, *_ = load_banc_edges(
        BancSourcePaths(
            meta=tmp_path / "meta.feather", edgelist=tmp_path / "edges.feather"
        ),
        ids,
        dict(BANC_CFG),
        [5],
        stats,
    )
    assert stats.autapses_found == 1
    assert not (pre == post).any()


def test_banc_csr_cutoff(banc_build):
    graph_dir = banc_build["root"] / "graph"
    csr = np.load(graph_dir / "csr_cutoff5.npz")
    nodes = pl.read_parquet(graph_dir / "nodes.parquet")
    n = nodes.height
    assert csr["indptr"].shape == (n + 1,)
    # 14 table edges minus count-4 (1006->1005) and count-3 (1012->1001)
    assert csr["indices"].size == 12
    assert (csr["weight"] >= 5).all()
    edges = pl.read_parquet(graph_dir / "edges.parquet")
    assert int((edges["weight"] >= 5).sum()) == csr["indices"].size
    manifest = json.loads(
        (banc_build["root"] / "provenance" / "banc_graph_test.json").read_text()
    )
    assert manifest["counts"]["edges_at_cutoff"] == {"5": 12}
    # raw rows include dropped + duplicate rows; c4/c3/dup-c2 are <5
    assert manifest["counts"]["raw_edges_at_cutoff"] == {"5": 13}


def test_banc_per_neuron_sign(banc_build):
    edges = pl.read_parquet(banc_build["root"] / "graph" / "edges.parquet")
    nodes = pl.read_parquet(banc_build["root"] / "graph" / "nodes.parquet")
    # edge sign = pre neuron's predicted-NT sign (sign_source=per_neuron)
    assert _edge_row(edges, "1003", "1001", nodes)["nt_sign"] == -1  # gaba pre
    assert _edge_row(edges, "1004", "1005", nodes)["nt_sign"] == -1  # glut pre
    assert _edge_row(edges, "1001", "1003", nodes)["nt_sign"] == 1   # aCh pre
    assert _edge_row(edges, "1010", "1005", nodes)["nt_sign"] == 1   # aCh pre
    e = _edge_row(edges, "1012", "1001", nodes)
    assert e["nt_sign"] == 0 and bool(e["nt_low_confidence"])
    manifest = json.loads(
        (banc_build["root"] / "provenance" / "banc_graph_test.json").read_text()
    )
    assert manifest["nt"]["edges_inhibitory"] == 6
    assert manifest["nt"]["edges_excitatory"] == 7
    assert manifest["nt"]["edges_low_confidence"] == 1


def test_banc_manifest_conventions(banc_build):
    manifest = json.loads(
        (banc_build["root"] / "provenance" / "banc_graph_test.json").read_text()
    )
    conv = manifest["conventions"]
    assert conv["snapshot"] == "v888"
    assert conv["synapse_snapshot"] == "v2"
    assert conv["autapse_policy"] == "excluded_at_build"
    assert conv["autapses_found"] == 0
    assert conv["sign_source"] == "per_neuron"
    assert conv["inhibitory_classes"] == ["gaba", "glutamate"]
    assert conv["read_time_cutoff"] == 5
    assert manifest["graph_version"] == "banc_test"
    report = json.loads(
        (banc_build["root"] / "results" / "banc_build_test.json").read_text()
    )
    assert report["artifact"] == "banc_build_test"
    assert report["schema"] == "results_v1"


# ---------------------------------------------------------------------------
# Alignment table
# ---------------------------------------------------------------------------


def test_alignment_table_iso_classes(align):
    assert align.columns == ["male_type", "banc_type", "banc_id", "iso_class"]
    # 12 typed banc neurons + 3 male-side rows (MSP1, MUN1, REJ1)
    assert align.height == 15
    counts = dict(align.group_by("iso_class").len().iter_rows())
    assert counts == {
        "isomorphic": 6,          # A_R x2, B_L x2, C_M, OVR1
        "dimorphic": 1,           # DIM1
        "female-specific": 1,     # FSP1
        "female-only-unmatched": 4,  # FUN1, E_L, ISO1, REJ1(banc)
        "male-specific": 1,       # MSP1
        "male-only-unmatched": 2, # MUN1, REJ1(male)
    }
    by_id = {
        r["banc_id"]: r for r in align.to_dicts() if r["banc_id"] is not None
    }
    assert by_id["1008"]["male_type"] == "C_M"     # manc fallback
    assert by_id["1009"]["male_type"] == "OVR_M"   # reviewed override wins
    assert by_id["1009"]["iso_class"] == "isomorphic"
    assert by_id["1010"]["iso_class"] == "female-only-unmatched"  # rejected
    assert by_id["1004"]["iso_class"] == "dimorphic"              # label join
    assert by_id["1005"]["iso_class"] == "female-specific"        # label join
    assert "1007" not in by_id                     # untyped row excluded
    male_only = [r for r in align.to_dicts() if r["banc_id"] is None]
    assert {r["male_type"]: r["iso_class"] for r in male_only} == {
        "MSP1": "male-specific",
        "MUN1": "male-only-unmatched",
        "REJ1": "male-only-unmatched",
    }


def test_alignment_table_deterministic(banc_sources):
    meta = pl.from_arrow(feather.read_table(banc_sources.meta))
    matches = pl.read_csv(banc_sources.malecns_matches)
    a = alignment_table(meta, matches, DIMORPHISM)
    b = alignment_table(meta, matches, DIMORPHISM)
    assert a.equals(b)
    # no matches csv / no dimorphism arg -> still deterministic
    c = alignment_table(meta)
    assert c.equals(alignment_table(meta))


def test_run_align_writes_artifacts(tmp_path, banc_sources):
    build_banc_graph(
        dict(BANC_CFG),
        banc_sources,
        graph_dir=tmp_path / "graph",
        prov_dir=tmp_path / "provenance",
        results_dir=tmp_path / "results",
        input_manifest=None,
        verify_hashes=False,
    )
    report = run_align(
        dict(BANC_CFG), banc_sources, tmp_path / "provenance",
        tmp_path / "evals", male_dimorphism=DIMORPHISM,
    )
    art = json.loads((tmp_path / "evals" / "alignment_v1.json").read_text())
    assert art["artifact"] == "alignment_v1"
    assert art["checks"]["n_rows"] == 15
    assert (tmp_path / "evals" / "alignment_v1.parquet").exists()
    manifest = json.loads(
        (tmp_path / "provenance" / "banc_graph_test.json").read_text()
    )
    assert "alignment" in manifest
    assert manifest["alignment"]["iso_class_counts"]["dimorphic"] == 1
    assert report["checks"]["iso_class_counts"]["male-specific"] == 1


# ---------------------------------------------------------------------------
# Sex-conditioned union vocab
# ---------------------------------------------------------------------------


def test_sex_conditioned_vocab(vocab):
    base = TYPE_TOKEN_BASE
    # 8 male types + 5 female-only appended tokens
    assert vocab["n_types"] == 13
    assert vocab["type_to_id"]["A_R"] == base + 0
    assert vocab["type_to_id"]["REJ1"] == base + 7
    # female block appended after the male block, sorted by banc type name
    for i, key in enumerate(["E_L", "FSP1", "FUN1", "ISO1", "REJ1__banc"]):
        assert vocab["type_to_id"][key] == base + 8 + i
    # aligned banc type shares the male token, even under a different name
    assert vocab["banc_type_to_id"]["OVR1"] == vocab["type_to_id"]["OVR_M"]
    assert vocab["banc_type_to_id"]["A_R"] == vocab["type_to_id"]["A_R"]
    assert vocab["banc_type_to_id"]["FSP1"] == vocab["type_to_id"]["FSP1"]
    sets = vocab["sex_specific_sets"]
    assert set(sets["male_only"]) == {"MSP1", "MUN1", "REJ1"}
    assert set(sets["female_only"]) == {
        "E_L", "FSP1", "FUN1", "ISO1", "REJ1__banc",
    }
    # disjoint by construction even with the REJ1 name collision
    assert not (set(sets["male_only"]) & set(sets["female_only"]))
    assert sets["male_specific_annotated"] == ["MSP1"]
    assert sets["female_specific_annotated"] == ["FSP1"]
    assert vocab["uncovered_banc_types"] == []
    assert vocab["alignment_gaps"] == []


def test_sex_conditioned_vocab_deterministic(male_bundle, banc_bundle, align):
    a = sex_conditioned_vocab(male_bundle, banc_bundle, align)
    b = sex_conditioned_vocab(male_bundle, banc_bundle, align)
    assert a == b


# ---------------------------------------------------------------------------
# Female graph in the shared (male) type space - T-SEX-4/5
# ---------------------------------------------------------------------------


def test_female_graph_in_male_space(vocab, banc_bundle, align):
    fb = female_graph_in_male_space(banc_bundle, align, vocab, vocab["n_types"])
    assert fb.n_nodes == 13
    edges = _type_edges(fb)
    # A_R -> B_L aggregates BOTH A_R neurons: w = 7 + 5 = 12
    assert edges[(0, 1)] == (12, 1)
    # B_L -> A_R (merged 9+2=11), inhibitory pre (gaba)
    assert edges[(1, 0)] == (11, -1)
    # B_L -> B_L intra-type neuron pair -> type-level self-loop
    assert edges[(1, 1)] == (6, -1)
    assert edges[(0, 3)] == (6, 1)    # A_R -> DIM1
    assert edges[(3, 9)] == (8, -1)   # DIM1 -> FSP1 (glutamate pre)
    assert edges[(9, 10)] == (5, 1)   # FSP1 -> FUN1
    assert edges[(2, 0)] == (7, 1)    # C_M -> A_R
    assert edges[(6, 3)] == (5, -1)   # OVR1 (OVR_M token) -> DIM1
    assert edges[(12, 9)] == (6, 1)   # REJ1__banc -> FSP1
    assert len(edges) == 9
    meta = fb.type_space_meta
    assert meta["type_edges"] == 9
    assert meta["neuron_edges_in"] == 12
    # both edges touching the untyped node are dropped, never re-homed
    assert meta["neuron_edges_dropped_untyped_or_unmapped"] == 2
    assert meta["intra_type_self_loops"] == 1
    # 10 kept neuron edges - 9 type edges = 1 merge (the two A_R->B_L pairs)
    assert meta["neuron_pairs_merged_into_type_edges"] == 10 - 9


def test_female_graph_masks_male_only_types(vocab, banc_bundle, align):
    fb = female_graph_in_male_space(banc_bundle, align, vocab, vocab["n_types"])
    nodes = fb.nodes
    # male-only slots exist in the union vocab but are structurally absent
    for j in (4, 5, 7):  # MSP1, MUN1, REJ1 token slots
        assert nodes["type"][j] in {"MSP1", "MUN1", "REJ1"}
        assert bool(nodes["present_female"][j]) is False
        assert nodes["sex_scope"][j] == "male_only"
        assert fb.out_indptr[j + 1] - fb.out_indptr[j] == 0
        assert fb.in_indptr[j + 1] - fb.in_indptr[j] == 0
    # female-only + shared slots are present
    for j in (0, 1, 2, 3, 6, 8, 9, 10, 11, 12):
        assert bool(nodes["present_female"][j]) is True


# ---------------------------------------------------------------------------
# Female stimulus map - homolog re-resolution + coverage reporting
# ---------------------------------------------------------------------------


def test_female_stimulus_map(align):
    spec = {
        "family_id": "optogenetic",
        "program_id": "p1",
        "drives": [
            {"types": ["A_R", "MSP1"], "rate_hz": 100.0, "window": [0.0, 1.0]},
            {"types": ["B_L", "MUN1", "REJ1"], "rate_hz": 50.0,
             "window": [0.0, 0.5]},
            {"subclasses": ["x"], "rate_hz": 10.0, "window": [0.0, 1.0]},
        ],
    }
    out = female_stimulus_map(spec, align)
    d0, d1, d2 = out["drives"]
    # aligned male type resolves to its banc homolog type
    assert d0["types"] == ["A_R"]
    assert d0["male_types"] == ["A_R", "MSP1"]
    assert d0["unmatched_male_types"] == ["MSP1"]
    assert d1["types"] == ["B_L"]
    assert d1["unmatched_male_types"] == ["MUN1", "REJ1"]
    # non-type queries pass through flagged unresolved
    assert d2["female_resolution"] == "passthrough_unresolved"
    cov = out["female_homolog_coverage"]
    assert cov["coverage"] == pytest.approx(0.4)  # 2 of 5 requested
    assert cov["resolved_male_types"] == ["A_R", "B_L"]
    assert cov["missing_male_types"] == ["MSP1", "MUN1", "REJ1"]
    assert cov["drives_unresolvable"] == 1
    # input spec not mutated
    assert spec["drives"][0]["types"] == ["A_R", "MSP1"]


# ---------------------------------------------------------------------------
# Source resolution from a manifest dict
# ---------------------------------------------------------------------------


def test_resolve_banc_sources(banc_sources, tmp_path):
    manifest = {
        "files": [
            {"key": "meta", "local_name": banc_sources.meta.name},
            {"key": "edgelist_v2", "local_name": banc_sources.edgelist.name},
            {"key": "malecns_matches",
             "local_name": banc_sources.malecns_matches.name},
        ]
    }
    raw = banc_sources.meta.parent
    paths = resolve_banc_sources(manifest, raw)
    assert paths.meta == banc_sources.meta
    assert paths.edgelist == banc_sources.edgelist
    assert paths.malecns_matches == banc_sources.malecns_matches
    assert paths.edgelist_v3 is None
    with pytest.raises(FileNotFoundError):
        resolve_banc_sources({"files": []}, raw)


def test_graph_manifest_name():
    assert graph_manifest_name("banc_v1") == "banc_graph_v1.json"
    assert graph_manifest_name("banc_test") == "banc_graph_test.json"
