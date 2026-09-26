"""Shared contracts: seed derivation, graph bundle loading, result envelope.

Every module in ``src/flygpt`` codes against these definitions - they are
the single source of truth for the event schema, the shard schema, the
results JSON envelope, and the hash-derived seed rule (``hash(trial
config)`` - never hand-assigned, plan.md section 6).

Schema version ``schema_v1``: any change here requires a new version tag
recorded in every downstream artifact.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

SCHEMA_VERSION = "schema_v1"

# ---------------------------------------------------------------------------
# Event schema (long format; one row per (trial, step, neuron) spike event)
# ---------------------------------------------------------------------------
EVENT_TRIAL = "trial_id"      # u64 - hash-derived trial identifier
EVENT_STEP = "step"           # u32 - integer sim step inside the trial
EVENT_NODE = "neuron_idx"     # u32 - node index into the graph's node table
EVENT_COLUMNS = (EVENT_TRIAL, EVENT_STEP, EVENT_NODE)

# Trials table: one row per trial (kept next to each shard's events).
TRIAL_COLUMNS = (
    "trial_id",           # u64
    "family_id",          # str  - stimulus family (atomic for splits)
    "program",            # str  - canonical JSON stimulus-program spec
    "seed",               # u64  - hash(program)
    "dt",                 # f64  - seconds per step
    "n_steps",            # u32
    "warmup_steps",       # u32  - dropped steps (cold-start fix)
    "n_events",           # u32
    "n_active_neurons",   # u32
    "n_active_types",     # u32
    "split",              # str  - train | test | eval label
    "degenerate",         # bool - flagged by health metrics (never dropped)
    "slice",              # str  - convention slice tag (e.g. main|cutoff3|dt5|shuffled)
)

# Reserved vocab slots (T-TOK-5): type ids start after this block.
VOCAB_PAD = 0
VOCAB_UNK = 1
VOCAB_SEX_MALE = 2
VOCAB_SEX_FEMALE = 3
VOCAB_RESERVED_BASE = 4  # stimulus-family + behavior tokens follow
N_STIMULUS_SLOTS = 64
N_BEHAVIOR_SLOTS = 32
TYPE_TOKEN_BASE = VOCAB_RESERVED_BASE + N_STIMULUS_SLOTS + N_BEHAVIOR_SLOTS

RESULTS_SCHEMA = "results_v1"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "uncommitted"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NaN-free."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def seed_from_config(obj: Any) -> int:
    """hash(trial config) -> u64 seed. The ONLY sanctioned seed source."""
    digest = hashlib.sha256(canonical_json(obj).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def rng_from_seed(*parts: Any) -> np.random.Generator:
    """Deterministic Generator from arbitrary hashable parts."""
    return np.random.default_rng(seed_from_config(parts) % (2**63))


# ---------------------------------------------------------------------------
# Graph bundle: nodes.parquet + csr_cutoffN.npz with both directions
# ---------------------------------------------------------------------------


@dataclass
class GraphBundle:
    """In-memory connectome graph with adjacency in both directions.

    ``out_`` arrays are CSR keyed by *presynaptic* node (row i = outgoing
    edges of i; ``indices`` are postsynaptic targets). ``in_`` arrays are
    CSR keyed by *postsynaptic* node (row j = incoming edges of j;
    ``indices`` are presynaptic sources). ``weight`` is the synapse count;
    ``sign`` is the per-edge NT sign (+1 excitatory, -1 inhibitory, 0
    undetermined - always paired with nt_low_confidence at build time).
    """

    nodes: pl.DataFrame
    n_nodes: int
    cutoff: int
    out_indptr: np.ndarray
    out_indices: np.ndarray
    out_weight: np.ndarray
    out_sign: np.ndarray
    in_indptr: np.ndarray
    in_indices: np.ndarray
    in_weight: np.ndarray
    in_sign: np.ndarray

    @property
    def body_ids(self) -> np.ndarray:
        return np.ascontiguousarray(self.nodes["bodyId"].to_numpy().astype(np.int64))

    @property
    def signed_weights_out(self) -> np.ndarray:
        """weight * sign per outgoing edge (i8->f32), undetermined edges = 0."""
        return (self.out_weight.astype(np.float32) * self.out_sign.astype(np.float32))

    @property
    def signed_weights_in(self) -> np.ndarray:
        return (self.in_weight.astype(np.float32) * self.in_sign.astype(np.float32))

    def node_of_body(self, body_ids: np.ndarray) -> np.ndarray:
        """bodyId -> node_idx, -1 when absent."""
        ids = self.body_ids
        pos = np.searchsorted(ids, body_ids)
        pos_c = np.clip(pos, 0, ids.size - 1)
        return np.where(ids[pos_c] == body_ids, pos_c, -1)

    def column(self, name: str) -> np.ndarray:
        return self.nodes[name].to_numpy()

    def type_array(self) -> np.ndarray:
        """Type label per node; untyped nodes get ''."""
        col = self.nodes["type"].to_numpy()
        return np.where(col == None, "", col).astype(str)  # noqa: E711

    @property
    def manifest_types(self) -> list[str]:
        """Sorted distinct non-null types - THE manifest denominator."""
        types = sorted({t for t in self.nodes["type"].to_list() if t})
        return types

    def in_neighbors(self, node: int) -> np.ndarray:
        """Presynaptic partner node indices of ``node``."""
        return self.in_indices[self.in_indptr[node]: self.in_indptr[node + 1]]

    def out_neighbors(self, node: int) -> np.ndarray:
        """Postsynaptic partner node indices of ``node``."""
        return self.out_indices[self.out_indptr[node]: self.out_indptr[node + 1]]


def build_incoming_csr(
    out_indptr: np.ndarray,
    out_indices: np.ndarray,
    out_weight: np.ndarray,
    out_sign: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transpose the by-pre CSR into by-post CSR (incoming adjacency)."""
    counts = np.bincount(out_indices, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    order = np.argsort(out_indices, kind="stable")
    src = np.repeat(np.arange(out_indptr.size - 1), np.diff(out_indptr))
    pre_sorted = src[order]
    return indptr, pre_sorted.astype(np.uint32), out_weight[order], out_sign[order]


def load_graph_bundle(graph_dir: Path, cutoff: int = 5) -> GraphBundle:
    """Load nodes.parquet + csr_cutoff{cutoff}.npz into a GraphBundle."""
    nodes = pl.read_parquet(graph_dir / "nodes.parquet")
    csr = np.load(graph_dir / f"csr_cutoff{cutoff}.npz")
    n = int(csr["n_nodes"])
    in_indptr, in_indices, in_weight, in_sign = build_incoming_csr(
        csr["indptr"], csr["indices"], csr["weight"], csr["sign"], n
    )
    return GraphBundle(
        nodes=nodes,
        n_nodes=n,
        cutoff=int(csr["cutoff"]),
        out_indptr=csr["indptr"],
        out_indices=csr["indices"],
        out_weight=csr["weight"],
        out_sign=csr["sign"],
        in_indptr=in_indptr,
        in_indices=in_indices,
        in_weight=in_weight,
        in_sign=in_sign,
    )


def bundle_from_edges(
    nodes: pl.DataFrame,
    pre: np.ndarray,
    post: np.ndarray,
    weight: np.ndarray,
    sign: np.ndarray,
    cutoff: int = 0,
) -> GraphBundle:
    """Build a GraphBundle directly from edge arrays (controls/tests)."""
    n = nodes.height
    key = (pre.astype(np.uint64) << 32) | post.astype(np.uint64)
    order = np.argsort(key, kind="stable")
    pre, post, weight, sign = (
        pre[order], post[order], weight[order], sign[order]
    )
    counts = np.bincount(pre, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    in_indptr, in_indices, in_weight, in_sign = build_incoming_csr(
        indptr, post.astype(np.uint32), weight.astype(np.int32), sign.astype(np.int8), n
    )
    return GraphBundle(
        nodes=nodes,
        n_nodes=n,
        cutoff=cutoff,
        out_indptr=indptr,
        out_indices=post.astype(np.uint32),
        out_weight=weight.astype(np.int32),
        out_sign=sign.astype(np.int8),
        in_indptr=in_indptr,
        in_indices=in_indices,
        in_weight=in_weight,
        in_sign=in_sign,
    )


# ---------------------------------------------------------------------------
# Results envelope - every eval/test artifact in results/ uses this shape
# ---------------------------------------------------------------------------


def write_result(
    results_dir: Path,
    test_id: str,
    checks: dict[str, Any],
    *,
    inputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    passed: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``results/<test_id>.json`` in the shared envelope and return it.

    Envelope: {artifact: test_id, schema, created_utc, code{git_sha,
    generator}, inputs, checks, metrics, pass}.
    """
    artifact = {
        "artifact": test_id,
        "schema": RESULTS_SCHEMA,
        "created_utc": utcnow(),
        "code": {"git_sha": git_sha(), "generator": f"flygpt {test_id}"},
        "inputs": inputs or {},
        "checks": checks,
        "metrics": metrics or {},
        "pass": passed,
    }
    if extra:
        artifact.update(extra)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{test_id}.json").write_text(json.dumps(artifact, indent=2, default=str) + "\n")
    return artifact
