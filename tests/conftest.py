"""Shared fixtures: a synthetic mini-connectome in the canonical schema.

The fixture exercises every Phase-0 code path - dedup, autapse policy,
unannotated-body filtering, per-edge NT signs, site-level dedup of
polyadic T-bars, unclassifiable/low-confidence handling, derived-weight
reconciliation - with expectations computed by hand below.

Synthetic universe
------------------
Bodies 10..15 are annotated nodes (14 is an isolated zero-degree node).
999/888 are unannotated segments that must never enter the graph.

Edges (canonical weights rows, with one intentional duplicate row):
    (10,11,6) + (10,11,1)  -> merged weight 7
    (10,12,6)              -> inhibitory (3/4 sites GABA/Glu)
    (12,11,4)              -> excitatory (one site unclassifiable)
    (11,11,5)              -> inhibitory autapse (kept)
    (13,11,6)              -> all sites unclassifiable -> sign 0, flagged
    (10,999,7),(888,11,4)  -> dropped (unannotated endpoints)

Edge counts after the annotated filter: >=3 -> 5, >=5 -> 4, >=10 -> 0.
syn-partners rows are authored so derived weights match canonical
exactly, plus one row forming a pair absent from weights (15->10) and
one fully-unannotated row (999->888).
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flygpt.graph import SourcePaths

NT = [
    "nt_acetylcholine_prob",
    "nt_dopamine_prob",
    "nt_gaba_prob",
    "nt_glutamate_prob",
    "nt_histamine_prob",
    "nt_octopamine_prob",
    "nt_serotonin_prob",
]


def _probs(top: str | None, top_p: float) -> list[float]:
    names = [c.removeprefix("nt_").removesuffix("_prob") for c in NT]
    if top is None:
        return [round(1.0 / 7, 4)] * 7  # no majority -> unclassifiable
    rest = round((1.0 - top_p) / 6, 4)
    return [top_p if n == top else rest for n in names]


def _write_feathers(raw):
    ann = pa.table(
        {
            "bodyId": [10, 11, 12, 13, 14, 15],
            "type": ["A_R", "B_L", "A_R", None, "C_M", "D_L"],
            "instance": ["A_R", "B_L", "A_R", None, "C_M", "D_L"],
            "superclass": ["cb_intrinsic"] * 6,
            "class": ["gustatory"] * 6,
            "somaSide": ["R", "L", "R", "L", "M", "L"],
            "rootSide": ["R", "L", "R", "L", "M", "L"],
            "statusLabel": ["Reviewed"] * 6,
            "status": ["Traced", "Traced", "Traced", "Orphan", "Traced", "Traced"],
            "dimorphism": [None] * 6,
        }
    )
    feather.write_feather(ann, raw / "ann.feather")

    nt_rows = dict.fromkeys((10, 11, 12, 13, 14, 15), "acetylcholine")
    body_nt = pa.table(
        {
            "body": list(nt_rows),
            "predicted_nt": list(nt_rows.values()),
            "predicted_nt_confidence": [0.9] * len(nt_rows),
            "celltype_predicted_nt": list(nt_rows.values()),
            "celltype_predicted_nt_confidence": [0.9] * len(nt_rows),
            "consensus_nt": list(nt_rows.values()),
            "ground_truth": [None] * len(nt_rows),
        }
    )
    feather.write_feather(body_nt, raw / "body_nt.feather")

    weights = pa.table(
        {
            "body_pre": [10, 10, 10, 12, 11, 13, 10, 888],
            "body_post": [11, 11, 12, 11, 11, 11, 999, 11],
            "weight": [6, 1, 6, 4, 5, 6, 7, 4],
        }
    )
    feather.write_feather(weights, raw / "weights.feather")

    # T-bars: (x, y, z, body, conf, top_nt, top_p)
    sites = [
        (1, 1, 1, 10, 0.9, "gaba", 0.9),          # s1 inhib
        (2, 2, 2, 10, 0.9, "gaba", 0.8),          # s2 inhib
        (3, 3, 3, 10, 0.9, "acetylcholine", 0.9),  # s3 excit
        (4, 4, 4, 10, 0.9, "acetylcholine", 0.95),  # s4 excit
        (11, 1, 1, 10, 0.9, "gaba", 0.9),         # t1 inhib
        (12, 1, 1, 10, 0.9, "glutamate", 0.9),    # t2 inhib
        (13, 1, 1, 10, 0.9, "gaba", 0.9),         # t3 inhib
        (14, 1, 1, 10, 0.9, "acetylcholine", 0.9),  # t4 excit
        (21, 1, 1, 12, 0.9, "acetylcholine", 0.4),  # u1 top_p<0.5 unclass
        (22, 1, 1, 12, 0.9, "acetylcholine", 0.9),  # u2 excit
        (31, 1, 1, 11, 0.9, "gaba", 0.9),         # v1 inhib (autapse)
        (41, 1, 1, 13, 0.9, None, 0.0),           # w1 no majority
        (42, 1, 1, 13, 0.9, "serotonin", 0.4),    # w2 top_p<0.5
        (43, 1, 1, 13, 0.3, "gaba", 0.9),         # w3 conf<0.5
        (51, 1, 1, 15, 0.9, "acetylcholine", 0.9),  # q1 (dangling pair)
        (61, 1, 1, 999, 0.9, "acetylcholine", 0.9),  # z1 unannotated body
    ]
    tbar = pa.table(
        {
            "x": [s[0] for s in sites],
            "y": [s[1] for s in sites],
            "z": [s[2] for s in sites],
            "body": [s[3] for s in sites],
            "conf": [s[4] for s in sites],
            **{c: [_probs(s[5], s[6])[i] for s in sites] for i, c in enumerate(NT)},
        }
    )
    feather.write_feather(tbar, raw / "tbar.feather")

    # (x_pre, y_pre, z_pre, body_pre, body_post); one row per partnership.
    syn_rows = [
        # edge (10,11) weight 7 - 4 distinct sites, X unmatched row
        (1, 1, 1, 10, 11), (1, 1, 1, 10, 11),
        (2, 2, 2, 10, 11), (2, 2, 2, 10, 11),
        (3, 3, 3, 10, 11), (4, 4, 4, 10, 11),
        (90, 90, 90, 10, 11),  # site absent from tbar file
        # edge (10,12) weight 6 - sites t1,t2,t3,t4 (t3,t4 polyadic x2)
        (11, 1, 1, 10, 12), (12, 1, 1, 10, 12),
        (13, 1, 1, 10, 12), (13, 1, 1, 10, 12),
        (14, 1, 1, 10, 12), (14, 1, 1, 10, 12),
        # edge (12,11) weight 4 - u1 unclassifiable, u2 excitatory
        (21, 1, 1, 12, 11), (21, 1, 1, 12, 11),
        (22, 1, 1, 12, 11), (22, 1, 1, 12, 11),
        # edge (11,11) autapse weight 5 - single inhibitory site x5
        (31, 1, 1, 11, 11), (31, 1, 1, 11, 11), (31, 1, 1, 11, 11),
        (31, 1, 1, 11, 11), (31, 1, 1, 11, 11),
        # edge (13,11) weight 6 - all three sites unclassifiable
        (41, 1, 1, 13, 11), (42, 1, 1, 13, 11), (43, 1, 1, 13, 11),
        (41, 1, 1, 13, 11), (42, 1, 1, 13, 11), (43, 1, 1, 13, 11),
        # partnership forming an edge absent from weights (15->10)
        (51, 1, 1, 15, 10),
        # fully unannotated partnership
        (61, 1, 1, 999, 888),
    ]
    syn = pa.table(
        {
            "x_pre": [r[0] for r in syn_rows],
            "y_pre": [r[1] for r in syn_rows],
            "z_pre": [r[2] for r in syn_rows],
            "body_pre": [r[3] for r in syn_rows],
            "body_post": [r[4] for r in syn_rows],
        }
    )
    feather.write_feather(syn, raw / "syn.feather")

    return SourcePaths(
        annotations=raw / "ann.feather",
        body_nt=raw / "body_nt.feather",
        weights=raw / "weights.feather",
        tbar_nt=raw / "tbar.feather",
        syn_partners=raw / "syn.feather",
    )


SYNTHETIC_CONFIG = {
    "dataset": "synthetic:test",
    "minconf": 0.5,
    "synapse_cutoff": 5,
    "sensitivity_cutoffs": [3, 5, 10],
    "autapse_policy": "keep",
    "dedup_edges": True,
    "keep_isolated_nodes": True,
    "nt_sign_rule": {
        "inhibitory_fraction_threshold": 0.5,
        "inhibitory_classes": ["gaba", "glutamate"],
        "site_conf_min": 0.5,
        "site_nt_prob_min": 0.5,
        "on_low_confidence": "flag",
    },
    "seed_derivation": "hash_trial_config",
    "version": "graph_test",
}


@pytest.fixture(scope="session")
def synthetic_sources(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    return _write_feathers(raw)


@pytest.fixture(scope="session")
def synthetic_build(tmp_path_factory, synthetic_sources):
    """Run the real build_graph on the synthetic fixture once per session."""
    from flygpt.graph import build_graph

    root = tmp_path_factory.mktemp("build")
    cfg = dict(SYNTHETIC_CONFIG)
    report = build_graph(
        cfg,
        synthetic_sources,
        graph_dir=root / "graph",
        provenance_dir=root / "provenance",
        results_dir=root / "results",
        input_manifest=None,
        verify_hashes=False,
    )
    return {"root": root, "report": report, "sources": synthetic_sources}


@pytest.fixture(scope="session")
def synthetic_bundle(synthetic_build):
    """GraphBundle of the synthetic mini-connectome (6 nodes, 5 edges).

    Node ids 10..15 -> node_idx 0..5 (sorted by bodyId). Edges:
    10->11 w7 (+1), 10->12 w6 (-1), 12->11 w4 (+1), 11->11 w5 (-1 autapse),
    13->11 w6 (sign 0). Type labels: 10=A_R, 11=B_L, 12=A_R, 13=None,
    14=C_M, 15=D_L. 14 is isolated (zero degree).
    """
    from flygpt.schema import load_graph_bundle

    return load_graph_bundle(synthetic_build["root"] / "graph", cutoff=3)


@pytest.fixture(scope="session")
def synthetic_nodes_df(synthetic_build):
    """Raw nodes.parquet of the synthetic build (for vocab/index tests)."""
    import polars as pl

    return pl.read_parquet(synthetic_build["root"] / "graph" / "nodes.parquet")
