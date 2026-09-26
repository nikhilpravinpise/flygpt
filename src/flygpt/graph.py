"""CSR graph build from the canonical MaleCNS flat connectome.

Implements the Phase-0 graph build per ``docs/plan.md`` and the versioned
conventions in ``configs/graph_v1.yaml``:

- Node universe = every row of ``body-annotations`` (isolated bodies are
  retained as zero-degree nodes so model vocab covers them).
- Edges come from ``connectome-weights`` restricted to pairs where both
  endpoints are annotated nodes. The weights file spans the whole
  segmentation (~87M postsynaptic segments); edges touching unannotated
  bodies are dropped and counted in the manifest, never silently kept.
- Dedup: repeated ``(pre, post)`` pairs are merged (weights summed) and
  counted; the canonical file ships already deduplicated.
- Autapse policy ``keep``: self-loops are retained and counted.
- Per-edge NT sign: an edge is inhibitory iff a strict majority of its
  classifiable presynaptic sites (distinct T-bars of that edge, joined
  from ``syn-partners`` to ``tbar-neurotransmitters`` on the ``(x,y,z)``
  coordinate triple) have top-predicted NT in the inhibitory class set.
  Edges with zero classifiable sites are flagged ``nt_low_confidence``
  with ``nt_sign = 0`` - flagged, never guessed. The Shiu-convention
  per-neuron majority sign is computed per node for cross-check.
- Weight reconciliation: edge weights are independently re-derived as
  syn-partners row counts per ``(pre, post)`` and reconciled against the
  canonical ``weight`` column; deltas land in the manifest and results.

Artifacts (under ``--graph-dir``, gitignored, regenerable):
    nodes.parquet            one row per annotated body
    edges.parquet            one row per edge (all weights, not just cutoff)
    csr_cutoff{3,5,10}.npz   CSR arrays per sensitivity cutoff
    site_flags.npz           per-T-bar classification used for edge signs
Manifests (committed):
    provenance/graph_v1.json   build provenance + convention record
    results/graph_build_v1.json  verification report (CLAIMS.md evidence)

Usage:
    python -m flygpt.graph --config configs/graph_v1.yaml
    python -m flygpt.graph --config configs/graph_v1.yaml --skip-verify
    python -m flygpt.graph --config configs/graph_v1.yaml --topo-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import yaml

NT_PROB_COLUMNS = (
    "nt_acetylcholine_prob",
    "nt_dopamine_prob",
    "nt_gaba_prob",
    "nt_glutamate_prob",
    "nt_histamine_prob",
    "nt_octopamine_prob",
    "nt_serotonin_prob",
)
NT_NAMES = tuple(c.removeprefix("nt_").removesuffix("_prob") for c in NT_PROB_COLUMNS)

# Site classification flags (u8) for each T-bar.
SITE_UNCLASSIFIABLE = 0
SITE_INHIBITORY = 1
SITE_NONINHIBITORY = 2

# Edge NT sign values (i8). Zero means "undetermined" - only ever set for
# edges with zero classifiable sites, always paired with low_confidence.
SIGN_UNDETERMINED = 0
SIGN_EXCITATORY = 1
SIGN_INHIBITORY = -1

# Manifest registry key -> SourcePaths field.
KEY_TO_FIELD = {
    "body_annotations": "annotations",
    "body_neurotransmitters": "body_nt",
    "connectome_weights": "weights",
    "tbar_neurotransmitters": "tbar_nt",
    "syn_partners": "syn_partners",
}
REQUIRED_SOURCE_KEYS = tuple(KEY_TO_FIELD)


@dataclass(frozen=True)
class SourcePaths:
    """Resolved local paths for the five required canonical files."""

    annotations: Path
    body_nt: Path
    weights: Path
    tbar_nt: Path
    syn_partners: Path

    def by_key(self, key: str) -> Path:
        return getattr(self, KEY_TO_FIELD[key])


@dataclass
class BuildStats:
    """Numbers that land in provenance/graph_v1.json + results/*.json."""

    wall_seconds: float = 0.0
    n_nodes: int = 0
    n_nodes_typed: int = 0
    n_nodes_traced_status: int = 0
    edges_raw_rows: int = 0
    edges_both_annotated: int = 0
    edges_both_traced: int = 0
    edge_rows_dropped_unannotated: int = 0
    weight_sum_dropped: int = 0
    duplicate_pairs_merged: int = 0
    autapses_kept: int = 0
    edges_at_cutoff: dict[str, int] = field(default_factory=dict)
    raw_edges_at_cutoff: dict[str, int] = field(default_factory=dict)
    syn_rows_total: int = 0
    syn_rows_edge_found: int = 0
    syn_rows_unannotated_endpoint: int = 0
    syn_rows_annotated_no_weights_entry: int = 0
    syn_rows_site_unmatched: int = 0
    syn_body_pre_mismatch: int = 0
    unique_site_post_pairs: int = 0
    weight_mismatch_edges: int = 0
    weight_mismatch_max_delta: int = 0
    weight_mismatch_total_abs: int = 0
    edges_with_zero_syn_rows: int = 0
    nt_edges_inhibitory: int = 0
    nt_edges_excitatory: int = 0
    nt_edges_low_confidence: int = 0
    nt_edge_sign_vs_neuron_sign_disagree: int = 0
    site_flag_counts: dict[str, int] = field(default_factory=dict)


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "uncommitted"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_sources(manifest: dict, raw_dir: Path) -> SourcePaths:
    """Map manifest keys to local paths for the five required files."""
    by_key = {f["key"]: f for f in manifest["files"]}
    missing = [k for k in REQUIRED_SOURCE_KEYS if k not in by_key or by_key[k].get("local_name") is None]
    if missing:
        raise FileNotFoundError(f"manifest lacks downloaded files for keys: {missing}")
    return SourcePaths(
        **{KEY_TO_FIELD[k]: raw_dir / by_key[k]["local_name"] for k in REQUIRED_SOURCE_KEYS}
    )


def verify_source_hashes(manifest: dict, sources: SourcePaths) -> dict[str, str]:
    """T-BUILD-5: recompute sha256 of every required input; fail closed."""
    recorded = {f["key"]: f["sha256"] for f in manifest["files"]}
    actual = {}
    for key in REQUIRED_SOURCE_KEYS:
        path = sources.by_key(key)
        digest = _sha256_file(path)
        if digest != recorded[key]:
            raise RuntimeError(
                f"sha256 mismatch for {key} ({path.name}): "
                f"{digest} != manifest {recorded[key]}"
            )
        actual[key] = digest
    return actual


def _batches(path: Path, columns: list[str]):
    """Yield record batches of selected columns from a feather/IPC file."""
    reader = ipc.open_file(path)
    names = reader.schema.names
    keep = [c for c in columns if c in names]
    if len(keep) != len(columns):
        raise KeyError(f"{path.name} missing columns {set(columns) - set(keep)}")
    for i in range(reader.num_record_batches):
        yield reader.get_record_batch(i).select(keep)


def _lookup_sorted(sorted_keys: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Return row index into ``sorted_keys`` per key, or -1 when absent."""
    if sorted_keys.size == 0:
        return np.full(keys.shape, -1, dtype=np.int64)
    pos = np.searchsorted(sorted_keys, keys)
    pos_c = np.clip(pos, 0, sorted_keys.size - 1)
    found = sorted_keys[pos_c] == keys
    return np.where(found, pos_c, -1)


def _body_to_node(body_ids_sorted: np.ndarray, bodies: np.ndarray) -> np.ndarray:
    """Map body ids to node indices; -1 for bodies outside the universe."""
    return _lookup_sorted(body_ids_sorted, bodies).astype(np.int32)


class _UniqueAccum:
    """Memory-bounded running np.unique over streamed u64 keys.

    ``_pending`` counts keys added *since the last merge* - a merge
    triggers on new additions, not on accumulator size (setting it to
    ``merged.size`` re-triggers a sort of the whole accumulator every
    few hundred keys once it is large: O(n^2) on real streams).
    """

    def __init__(self, merge_at: int = 512_000_000):
        self._parts: list[np.ndarray] = []
        self._pending = 0
        self._merge_at = merge_at

    def add(self, keys: np.ndarray) -> None:
        if keys.size == 0:
            return
        self._parts.append(np.unique(keys))
        self._pending += keys.size
        if self._pending >= self._merge_at:
            self._merge()

    def _merge(self) -> None:
        merged = np.unique(np.concatenate(self._parts))
        self._parts = [merged]
        self._pending = 0

    def result(self) -> np.ndarray:
        if not self._parts:
            return np.empty(0, dtype=np.uint64)
        self._merge()
        return self._parts[0]


def load_nodes(sources: SourcePaths) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Node universe: all body-annotation rows, sorted by bodyId.

    Returns the node dataframe (one row per node, ``node_idx`` aligned),
    the sorted int64 body_id array used for O(log N) body lookups, and a
    bool array marking rows whose tracing status is ``Traced``.
    """
    ann = pl.from_arrow(feather.read_table(sources.annotations))
    ann = ann.sort("bodyId")
    if not ann["bodyId"].is_unique().all():
        raise ValueError("body-annotations contains duplicate bodyId rows")
    ann = ann.with_row_index("node_idx")
    body_ids = np.ascontiguousarray(ann["bodyId"].to_numpy().astype(np.int64))
    is_traced = (ann["status"] == "Traced").to_numpy().astype(np.bool_)

    nt = pl.from_arrow(
        feather.read_table(
            sources.body_nt,
            columns=[
                "body",
                "predicted_nt",
                "predicted_nt_confidence",
                "celltype_predicted_nt",
                "celltype_predicted_nt_confidence",
                "consensus_nt",
                "ground_truth",
            ],
        )
    )
    if not nt["body"].is_unique().all():
        raise ValueError("body-neurotransmitters contains duplicate body rows")
    nodes = ann.join(nt, left_on="bodyId", right_on="body", how="left")
    if nodes.height != ann.height:
        raise ValueError("NT join changed node count")
    return nodes, body_ids, is_traced


def load_edges(
    sources: SourcePaths,
    body_ids: np.ndarray,
    is_traced: np.ndarray,
    cutoffs: list[int],
    stats: BuildStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load connectome-weights, restrict to annotated pairs, dedup, sort.

    Returns (pre_idx u32, post_idx u32, weight i32, edge_key u64 sorted).
    """
    pres: list[np.ndarray] = []
    posts: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for batch in _batches(sources.weights, ["body_pre", "body_post", "weight"]):
        for name in ("body_pre", "body_post", "weight"):
            if batch.column(name).null_count:
                raise ValueError(f"{sources.weights.name}:{name} has nulls")
        pre_b = batch.column("body_pre").to_numpy()
        post_b = batch.column("body_post").to_numpy()
        w = batch.column("weight").to_numpy()
        stats.edges_raw_rows += len(batch)
        for c in cutoffs:
            stats.raw_edges_at_cutoff[str(c)] = stats.raw_edges_at_cutoff.get(str(c), 0) + int(
                (w >= c).sum()
            )
        pre_idx = _body_to_node(body_ids, pre_b)
        post_idx = _body_to_node(body_ids, post_b)
        ok = (pre_idx >= 0) & (post_idx >= 0)
        stats.edge_rows_dropped_unannotated += int((~ok).sum())
        stats.weight_sum_dropped += int(w[~ok].sum())
        pres.append(pre_idx[ok].astype(np.uint32))
        posts.append(post_idx[ok].astype(np.uint32))
        weights.append(w[ok].astype(np.int32))

    pre = np.concatenate(pres)
    post = np.concatenate(posts)
    weight = np.concatenate(weights)
    stats.edges_both_annotated = pre.size

    key = (pre.astype(np.uint64) << 32) | post.astype(np.uint64)
    order = np.argsort(key, kind="stable")
    key = key[order]
    pre = pre[order]
    post = post[order]
    weight = weight[order]

    # Dedup: runs of equal packed keys are merged with weights summed.
    run_start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if key.size else key
    n_dup_rows = int(key.size - run_start.size)
    if n_dup_rows:
        merged_weight = np.add.reduceat(weight.astype(np.int64), run_start)
        key = key[run_start]
        weight = merged_weight.astype(np.int32)
        pre = (key >> 32).astype(np.uint32)
        post = (key & 0xFFFFFFFF).astype(np.uint32)
        stats.duplicate_pairs_merged = n_dup_rows

    stats.autapses_kept = int((pre == post).sum())
    stats.edges_both_traced = int((is_traced[pre] & is_traced[post]).sum())
    return pre, post, weight, key


@dataclass
class SiteIndex:
    """Sorted-key lookup index over all T-bar presynaptic sites."""

    key_sorted: np.ndarray  # u64 packed (x,y,z), ascending
    flag: np.ndarray  # u8 SITE_*, aligned with key_sorted
    node_idx: np.ndarray  # i32 node index of the site body, -1 if unannotated
    conf: np.ndarray  # f32, aligned
    x_bits: int
    y_bits: int
    z_bits: int
    x_max: int
    y_max: int
    z_max: int

    def pack(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return (
            (x.astype(np.uint64) << (self.y_bits + self.z_bits))
            | (y.astype(np.uint64) << self.z_bits)
            | z.astype(np.uint64)
        )

    def lookup(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Site row per (x,y,z) or -1. Out-of-range coords never match."""
        in_range = (x <= self.x_max) & (y <= self.y_max) & (z <= self.z_max)
        idx = _lookup_sorted(self.key_sorted, self.pack(x, y, z))
        return np.where(in_range, idx, -1)


def classify_sites(
    sources: SourcePaths, body_ids: np.ndarray, cfg: dict[str, Any], stats: BuildStats
) -> SiteIndex:
    """Per-T-bar classification from tbar-neurotransmitters.

    Returns a SiteIndex keyed on the packed (x, y, z) coordinate triple
    (unique per T-bar in the canonical file) using bit widths measured
    from the data.
    """
    rule = cfg["nt_sign_rule"]
    conf_min = float(rule["site_conf_min"])
    prob_min = float(rule["site_nt_prob_min"])
    inhib = {c.strip().lower() for c in rule["inhibitory_classes"]}
    inhib_idx = [i for i, n in enumerate(NT_NAMES) if n in inhib]
    if not inhib_idx:
        raise ValueError(f"inhibitory_classes {inhib} not in {NT_NAMES}")

    flags: list[np.ndarray] = []
    node_idxs: list[np.ndarray] = []
    confs: list[np.ndarray] = []

    cols = ["x", "y", "z", "body", "conf", *NT_PROB_COLUMNS]
    # Collect raw coordinate columns first; pack keys once the bit widths
    # are known from the full pass.
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    for batch in _batches(sources.tbar_nt, cols):
        x = batch.column("x").to_numpy()
        y = batch.column("y").to_numpy()
        z = batch.column("z").to_numpy()
        body = batch.column("body").to_numpy()
        conf = batch.column("conf").to_numpy()
        probs = np.stack([batch.column(c).to_numpy() for c in NT_PROB_COLUMNS], axis=1)

        top = probs.max(axis=1)
        am = probs.argmax(axis=1)
        is_inhib = np.isin(am, inhib_idx)
        classifiable = (conf >= conf_min) & (top >= prob_min)
        flag = np.where(
            ~classifiable,
            SITE_UNCLASSIFIABLE,
            np.where(is_inhib, SITE_INHIBITORY, SITE_NONINHIBITORY),
        ).astype(np.uint8)

        xs.append(x.astype(np.int64))
        ys.append(y.astype(np.int64))
        zs.append(z.astype(np.int64))
        flags.append(flag)
        confs.append(conf.astype(np.float32))
        node_idxs.append(_body_to_node(body_ids, body))

    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    z_all = np.concatenate(zs)
    flag_all = np.concatenate(flags)
    conf_all = np.concatenate(confs)
    node_all = np.concatenate(node_idxs)

    stats.site_flag_counts = {
        "unclassifiable": int((flag_all == SITE_UNCLASSIFIABLE).sum()),
        "inhibitory": int((flag_all == SITE_INHIBITORY).sum()),
        "noninhibitory": int((flag_all == SITE_NONINHIBITORY).sum()),
    }

    x_max, y_max, z_max = int(x_all.max()), int(y_all.max()), int(z_all.max())
    x_bits, y_bits, z_bits = (
        x_max.bit_length(),
        y_max.bit_length(),
        z_max.bit_length(),
    )
    if x_bits + y_bits + z_bits > 64:
        raise ValueError("coordinate triple does not fit in u64 packing")
    site_key = (
        (x_all.astype(np.uint64) << (y_bits + z_bits))
        | (y_all.astype(np.uint64) << z_bits)
        | z_all.astype(np.uint64)
    )
    order = np.argsort(site_key)
    return SiteIndex(
        key_sorted=np.ascontiguousarray(site_key[order]),
        flag=np.ascontiguousarray(flag_all[order]),
        node_idx=np.ascontiguousarray(node_all[order]),
        conf=np.ascontiguousarray(conf_all[order]),
        x_bits=x_bits,
        y_bits=y_bits,
        z_bits=z_bits,
        x_max=x_max,
        y_max=y_max,
        z_max=z_max,
    )


def synapse_pass(
    sources: SourcePaths,
    body_ids: np.ndarray,
    edge_key_sorted: np.ndarray,
    sites: SiteIndex,
    n_edges: int,
    stats: BuildStats,
) -> dict[str, np.ndarray]:
    """Stream syn-partners: derived weights + per-edge site statistics.

    Per edge this produces:
      derived_weight      - row count per (pre, post) pair
      n_sites             - distinct presynaptic sites (T-bars) matched
                            in tbar-neurotransmitters
      n_sites_classifiable, n_sites_inhibitory - NT classification counts
    """
    derived_weight = np.zeros(n_edges, dtype=np.int64)
    site_post = _UniqueAccum()
    cols = ["x_pre", "y_pre", "z_pre", "body_pre", "body_post"]

    n_batches = ipc.open_file(sources.syn_partners).num_record_batches
    t_pass = time.time()
    for bi, batch in enumerate(_batches(sources.syn_partners, cols)):
        if bi % 250 == 0:
            done = stats.syn_rows_total
            rate = done / max(time.time() - t_pass, 1e-6) / 1e6
            try:
                import os

                _rss = os.popen(
                    "powershell -NoProfile -Command "
                    f'"(Get-Process -Id {os.getpid()}).WorkingSet64"'
                ).read().strip()
                _mb = int(_rss or 0) // 1_048_576
            except Exception:
                _mb = -1
            print(
                f"  [syn] batch {bi}/{n_batches} rows={done/1e6:.0f}M "
                f"({rate:.1f}M rows/s) rss={_mb}MB parts={len(site_post._parts)}",
                flush=True,
            )
        xp = batch.column("x_pre").to_numpy().astype(np.int64)
        yp = batch.column("y_pre").to_numpy().astype(np.int64)
        zp = batch.column("z_pre").to_numpy().astype(np.int64)
        body_pre = batch.column("body_pre").to_numpy()
        body_post = batch.column("body_post").to_numpy()
        stats.syn_rows_total += len(batch)

        pre_idx = _body_to_node(body_ids, body_pre)
        post_idx = _body_to_node(body_ids, body_post)
        both = (pre_idx >= 0) & (post_idx >= 0)

        edge_key = (pre_idx.astype(np.uint64) << 32) | post_idx.astype(np.uint64)
        edge_idx = _lookup_sorted(edge_key_sorted, edge_key)
        edge_found = both & (edge_idx >= 0)
        stats.syn_rows_edge_found += int(edge_found.sum())
        stats.syn_rows_unannotated_endpoint += int((~both).sum())
        stats.syn_rows_annotated_no_weights_entry += int(
            (both & ~edge_found).sum()
        )
        if edge_found.any():
            # per-batch bincount bounded by batch size, then scatter-add -
            # avoids allocating an n_edges-wide temp per batch
            cnt = np.bincount(edge_idx[edge_found])
            nz = np.flatnonzero(cnt)
            derived_weight[nz] += cnt[nz]

        site_idx = sites.lookup(xp, yp, zp)
        site_ok = site_idx >= 0
        stats.syn_rows_site_unmatched += int((edge_found & ~site_ok).sum())
        # Sanity: the matched site's body should equal syn body_pre.
        # Only meaningful where both sides resolve to annotated nodes.
        checkable = site_ok & (pre_idx >= 0)
        if checkable.any():
            s = site_idx[checkable]
            resolvable = sites.node_idx[s] >= 0
            stats.syn_body_pre_mismatch += int(
                (sites.node_idx[s[resolvable]] != pre_idx[checkable][resolvable]).sum()
            )
        keep = edge_found & site_ok
        if keep.any():
            pair_key = (site_idx[keep].astype(np.uint64) << 32) | post_idx[keep].astype(np.uint64)
            site_post.add(pair_key)

    pairs = site_post.result()
    stats.unique_site_post_pairs = int(pairs.size)

    n_sites = np.zeros(n_edges, dtype=np.int64)
    n_class = np.zeros(n_edges, dtype=np.int64)
    n_inhib = np.zeros(n_edges, dtype=np.int64)
    if pairs.size:
        site_of_pair = (pairs >> 32).astype(np.int64)
        post_of_pair = (pairs & 0xFFFFFFFF).astype(np.int64)
        pre_of_pair = sites.node_idx[site_of_pair]
        valid = pre_of_pair >= 0
        pair_edge_key = (pre_of_pair[valid].astype(np.uint64) << 32) | post_of_pair[
            valid
        ].astype(np.uint64)
        eidx = _lookup_sorted(edge_key_sorted, pair_edge_key)
        found = eidx >= 0
        eidx = eidx[found]
        fl = sites.flag[site_of_pair[valid][found]]
        n_sites += np.bincount(eidx, minlength=n_edges)
        n_class += np.bincount(eidx[fl > 0], minlength=n_edges)
        n_inhib += np.bincount(eidx[fl == SITE_INHIBITORY], minlength=n_edges)

    return {
        "derived_weight": derived_weight,
        "n_sites": n_sites,
        "n_sites_classifiable": n_class,
        "n_sites_inhibitory": n_inhib,
    }


def edge_signs(site_stats: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    """Apply the configured majority rule to per-edge site counts."""
    thr = float(cfg["nt_sign_rule"]["inhibitory_fraction_threshold"])
    n_class = site_stats["n_sites_classifiable"]
    n_inhib = site_stats["n_sites_inhibitory"]
    fraction = np.where(n_class > 0, n_inhib / np.maximum(n_class, 1), np.nan).astype(np.float32)
    sign = np.where(
        n_class == 0,
        SIGN_UNDETERMINED,
        np.where(fraction > thr, SIGN_INHIBITORY, SIGN_EXCITATORY),
    ).astype(np.int8)
    low_conf = (n_class == 0).astype(np.bool_)
    return {
        "nt_sign": sign,
        "nt_inhibitory_fraction": fraction,
        "nt_low_confidence": low_conf,
    }


def neuron_signs(
    sites: SiteIndex, n_nodes: int, cfg: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Shiu-convention per-neuron sign: majority over ALL of a neuron's
    classifiable T-bars. Stored per node for cross-checking edge signs."""
    thr = float(cfg["nt_sign_rule"]["inhibitory_fraction_threshold"])
    ok = sites.node_idx >= 0
    idx = sites.node_idx[ok].astype(np.int64)
    fl = sites.flag[ok]
    n_class = np.bincount(idx[fl > 0], minlength=n_nodes)
    n_inhib = np.bincount(idx[fl == SITE_INHIBITORY], minlength=n_nodes)
    sign = np.where(
        n_class == 0,
        SIGN_UNDETERMINED,
        np.where(n_inhib > n_class * thr, SIGN_INHIBITORY, SIGN_EXCITATORY),
    ).astype(np.int8)
    return {
        "nt_sign_neuron": sign,
        "nt_sites_classifiable_neuron": n_class.astype(np.int32),
        "nt_sites_inhibitory_neuron": n_inhib.astype(np.int32),
    }


def build_csr(
    pre: np.ndarray, post: np.ndarray, weight: np.ndarray, sign: np.ndarray,
    n_nodes: int, cutoff: int,
) -> dict[str, np.ndarray]:
    """CSR (indptr/indices/weight/sign) for edges with weight >= cutoff.

    Input edges are sorted by (pre, post), so the CSR row ordering falls
    out directly.
    """
    keep = weight >= cutoff
    kpre = pre[keep]
    counts = np.bincount(kpre, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return {
        "indptr": indptr,
        "indices": post[keep].astype(np.uint32),
        "weight": weight[keep].astype(np.int32),
        "sign": sign[keep].astype(np.int8),
        "n_nodes": np.int64(n_nodes),
        "cutoff": np.int64(cutoff),
    }


def _gitignore_safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def build_graph(
    cfg: dict[str, Any],
    sources: SourcePaths,
    graph_dir: Path,
    provenance_dir: Path,
    results_dir: Path,
    input_manifest: dict | None = None,
    verify_hashes: bool = True,
    topo_only: bool = False,
) -> dict[str, Any]:
    """Run the full build. Returns the verification report dict."""
    t0 = time.time()
    stats = BuildStats()
    version = cfg["version"]
    cutoff = int(cfg["synapse_cutoff"])
    cutoffs = sorted({cutoff, *[int(c) for c in cfg["sensitivity_cutoffs"]]})

    hashes = {}
    if verify_hashes and input_manifest is not None:
        hashes = verify_source_hashes(input_manifest, sources)

    print("[1/5] loading node universe ...", flush=True)
    nodes, body_ids, is_traced = load_nodes(sources)
    stats.n_nodes = nodes.height
    stats.n_nodes_typed = int(nodes["type"].is_not_null().sum())
    stats.n_nodes_traced_status = int(is_traced.sum())

    print("[2/5] loading + filtering edges ...", flush=True)
    pre, post, weight, edge_key_sorted = load_edges(
        sources, body_ids, is_traced, cutoffs, stats
    )
    n_edges = pre.size
    graph_dir.mkdir(parents=True, exist_ok=True)

    if topo_only:
        site_stats = {
            "derived_weight": np.full(n_edges, -1, dtype=np.int64),
            "n_sites": np.zeros(n_edges, dtype=np.int64),
            "n_sites_classifiable": np.zeros(n_edges, dtype=np.int64),
            "n_sites_inhibitory": np.zeros(n_edges, dtype=np.int64),
        }
        sign_cols = edge_signs(site_stats, cfg)
        neuron = {
            "nt_sign_neuron": np.zeros(stats.n_nodes, dtype=np.int8),
            "nt_sites_classifiable_neuron": np.zeros(stats.n_nodes, dtype=np.int32),
            "nt_sites_inhibitory_neuron": np.zeros(stats.n_nodes, dtype=np.int32),
        }
    else:
        print("[3/5] classifying presynaptic sites ...", flush=True)
        sites = classify_sites(sources, body_ids, cfg, stats)
        print("[4/5] synapse pass over syn-partners ...", flush=True)
        site_stats = synapse_pass(
            sources, body_ids, edge_key_sorted, sites, n_edges, stats,
        )
        sign_cols = edge_signs(site_stats, cfg)
        neuron = neuron_signs(sites, stats.n_nodes, cfg)
        np.savez_compressed(
            graph_dir / "site_flags.npz",
            site_key=sites.key_sorted,
            site_flag=sites.flag,
            site_node_idx=sites.node_idx,
        )

    # Reconciliation: derived vs canonical weight.
    derived = site_stats["derived_weight"]
    if not topo_only:
        delta = derived - weight.astype(np.int64)
        mismatch = delta != 0
        stats.weight_mismatch_edges = int(mismatch.sum())
        stats.weight_mismatch_max_delta = int(np.abs(delta).max()) if n_edges else 0
        stats.weight_mismatch_total_abs = int(np.abs(delta).sum())
        stats.edges_with_zero_syn_rows = int((derived == 0).sum())
        stats.nt_edges_inhibitory = int((sign_cols["nt_sign"] == SIGN_INHIBITORY).sum())
        stats.nt_edges_excitatory = int((sign_cols["nt_sign"] == SIGN_EXCITATORY).sum())
        stats.nt_edges_low_confidence = int(sign_cols["nt_low_confidence"].sum())
        # Edge sign vs pre-neuron majority sign (both must be determined).
        edge_pre_neuron_sign = neuron["nt_sign_neuron"][pre]
        both_det = (sign_cols["nt_sign"] != 0) & (edge_pre_neuron_sign != 0)
        stats.nt_edge_sign_vs_neuron_sign_disagree = int(
            (sign_cols["nt_sign"][both_det] != edge_pre_neuron_sign[both_det]).sum()
        )

    print("[5/5] writing artifacts ...", flush=True)

    edges_df = pl.DataFrame(
        {
            "pre_idx": pre,
            "post_idx": post,
            "weight": weight,
            "derived_weight": derived.astype(np.int32),
            "n_sites": site_stats["n_sites"].astype(np.int32),
            "n_sites_classifiable": site_stats["n_sites_classifiable"].astype(np.int32),
            "n_sites_inhibitory": site_stats["n_sites_inhibitory"].astype(np.int32),
            "nt_sign": sign_cols["nt_sign"],
            "nt_inhibitory_fraction": sign_cols["nt_inhibitory_fraction"],
            "nt_low_confidence": sign_cols["nt_low_confidence"],
        }
    )
    edges_df.write_parquet(graph_dir / "edges.parquet", compression="zstd")

    out_deg = np.bincount(pre[weight >= cutoff], minlength=stats.n_nodes).astype(np.int32)
    in_deg = np.bincount(post[weight >= cutoff], minlength=stats.n_nodes).astype(np.int32)
    nodes = nodes.with_columns(
        [
            pl.Series("nt_sign_neuron", neuron["nt_sign_neuron"]),
            pl.Series("nt_sites_classifiable_neuron", neuron["nt_sites_classifiable_neuron"]),
            pl.Series("nt_sites_inhibitory_neuron", neuron["nt_sites_inhibitory_neuron"]),
            pl.Series("out_degree", out_deg),
            pl.Series("in_degree", in_deg),
        ]
    )
    nodes.write_parquet(graph_dir / "nodes.parquet", compression="zstd")

    for c in cutoffs:
        csr = build_csr(pre, post, weight, sign_cols["nt_sign"], stats.n_nodes, c)
        stats.edges_at_cutoff[str(c)] = int(csr["indices"].size)
        np.savez_compressed(graph_dir / f"csr_cutoff{c}.npz", **csr)

    stats.wall_seconds = round(time.time() - t0, 2)

    manifest = {
        "graph_version": version,
        "dataset": cfg["dataset"],
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": {"git_sha": _git_sha(), "generator": "flygpt graph.py"},
        "config_sha256": cfg.get("_config_sha256"),
        "conventions": {
            "synapse_cutoff": cutoff,
            "sensitivity_cutoffs": cutoffs,
            "autapse_policy": cfg["autapse_policy"],
            "dedup_edges": cfg["dedup_edges"],
            "keep_isolated_nodes": cfg["keep_isolated_nodes"],
            "nt_sign_rule": cfg["nt_sign_rule"],
            "node_universe": "every body-annotations row; zero-degree nodes retained",
            "edge_universe": "connectome-weights rows with both endpoints annotated",
        },
        "input_sha256": hashes or None,
        "counts": {
            "nodes": stats.n_nodes,
            "nodes_typed": stats.n_nodes_typed,
            "nodes_traced_status": stats.n_nodes_traced_status,
            "edges_raw_rows": stats.edges_raw_rows,
            "edges_both_annotated": stats.edges_both_annotated,
            "edges_both_traced": stats.edges_both_traced,
            "edge_rows_dropped_unannotated": stats.edge_rows_dropped_unannotated,
            "weight_sum_dropped": stats.weight_sum_dropped,
            "duplicate_pairs_merged": stats.duplicate_pairs_merged,
            "autapses_kept": stats.autapses_kept,
            "edges_at_cutoff": stats.edges_at_cutoff,
            "raw_edges_at_cutoff": stats.raw_edges_at_cutoff,
            "syn_rows_total": stats.syn_rows_total,
            "syn_rows_edge_found": stats.syn_rows_edge_found,
            "syn_rows_unannotated_endpoint": stats.syn_rows_unannotated_endpoint,
            "syn_rows_annotated_no_weights_entry": stats.syn_rows_annotated_no_weights_entry,
            "syn_rows_site_unmatched": stats.syn_rows_site_unmatched,
            "unique_site_post_pairs": stats.unique_site_post_pairs,
        },
        "nt": {
            "site_flag_counts": {k: v for k, v in stats.site_flag_counts.items() if not k.startswith("_")},
            "edges_inhibitory": stats.nt_edges_inhibitory,
            "edges_excitatory": stats.nt_edges_excitatory,
            "edges_low_confidence": stats.nt_edges_low_confidence,
            "edge_vs_neuron_sign_disagree": stats.nt_edge_sign_vs_neuron_sign_disagree,
        },
        "reconciliation": {
            "weight_mismatch_edges": stats.weight_mismatch_edges,
            "weight_mismatch_max_delta": stats.weight_mismatch_max_delta,
            "weight_mismatch_total_abs": stats.weight_mismatch_total_abs,
            "edges_with_zero_syn_rows": stats.edges_with_zero_syn_rows,
        },
        "topo_only": topo_only,
    }
    _gitignore_safe_write_json(provenance_dir / f"{version}.json", manifest)

    report = {
        "artifact": "graph_build",
        "graph_version": version,
        "manifest": f"provenance/{version}.json",
        "checks": {
            "node_count_equals_annotation_rows": stats.n_nodes,
            "edges_at_cutoff": stats.edges_at_cutoff,
            "raw_edges_at_cutoff": stats.raw_edges_at_cutoff,
            "edges_both_traced": stats.edges_both_traced,
            "autapses_kept": stats.autapses_kept,
            "duplicate_pairs_merged": stats.duplicate_pairs_merged,
            "derived_weight_reconciliation": {
                "mismatch_edges": stats.weight_mismatch_edges,
                "max_abs_delta": stats.weight_mismatch_max_delta,
                "total_abs_delta": stats.weight_mismatch_total_abs,
                "edges_with_zero_syn_rows": stats.edges_with_zero_syn_rows,
            },
            "nt_sign": {
                "inhibitory_edges": stats.nt_edges_inhibitory,
                "excitatory_edges": stats.nt_edges_excitatory,
                "low_confidence_edges": stats.nt_edges_low_confidence,
                "edge_vs_neuron_sign_disagree": stats.nt_edge_sign_vs_neuron_sign_disagree,
            },
            "typed_fraction": round(stats.n_nodes_typed / max(stats.n_nodes, 1), 6),
        },
        "wall_seconds": stats.wall_seconds,
    }
    results_name = f"graph_build_{version.removeprefix('graph_')}"
    _gitignore_safe_write_json(results_dir / f"{results_name}.json", report)
    print(f"[done] {stats.n_nodes} nodes, {n_edges} edges, "
          f"cutoff {cutoff}: {stats.edges_at_cutoff.get(str(cutoff))} edges, "
          f"{stats.wall_seconds}s", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--graph-dir", type=Path, default=None,
                    help="default: config output.graph_dir")
    ap.add_argument("--provenance-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=Path("provenance/malecns_v1.json"))
    ap.add_argument("--skip-verify", action="store_true",
                    help="dev only: skip input sha256 re-verification")
    ap.add_argument("--topo-only", action="store_true",
                    help="topology only: skip the synapse/NT pass")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    graph_dir = args.graph_dir or Path(cfg["output"]["graph_dir"])
    provenance_dir = args.provenance_dir or Path(cfg["output"]["provenance_dir"])
    results_dir = args.results_dir or Path(cfg["output"]["results_dir"])

    input_manifest = None
    if args.manifest.exists():
        input_manifest = json.loads(args.manifest.read_text())
        sources = resolve_sources(input_manifest, args.raw_dir)
    elif args.skip_verify:
        # Dev path: manifest absent - fall back to canonical filenames.
        sources = SourcePaths(
            annotations=args.raw_dir / "body-annotations-male-cns-v1.0-minconf-0.5.feather",
            body_nt=args.raw_dir / "body-neurotransmitters-male-cns-v1.0.feather",
            weights=args.raw_dir / "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            tbar_nt=args.raw_dir / "tbar-neurotransmitters-male-cns-v1.0.feather",
            syn_partners=args.raw_dir / "syn-partners-male-cns-v1.0-minconf-0.5.feather",
        )
    else:
        raise FileNotFoundError(
            f"{args.manifest} missing - run `python -m flygpt.ingest` first "
            "(or pass --skip-verify for dev builds on unmanifested files)"
        )

    build_graph(
        cfg, sources, graph_dir, provenance_dir, results_dir,
        input_manifest=input_manifest,
        verify_hashes=not args.skip_verify,
        topo_only=args.topo_only,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
