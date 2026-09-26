"""Corpus trial runner: sim -> event lists -> immutable parquet shards.

Artifacts under ``--work-dir`` (schema ``schema_v1``, flygpt.schema):

    events-<shard>.parquet   EVENT_COLUMNS: trial_id u64, step u32,
                             neuron_idx u32 - sorted by
                             (trial_id, step, neuron_idx), zstd
    trials-<shard>.parquet   TRIAL_COLUMNS, one row per trial
    <shard>.manifest.json    {schema, corpus_version, shard_id,
                             code{git_sha}, graph{version,sha256},
                             sim_cfg, backend, gpu_class, dt, cutoff,
                             slice, split, trial_ids, n_trials, n_events,
                             source_manifest, created_utc, shard_sha256}

Shard discipline (EVAL.md section 3, T-CORPUS-5/7): one (slice, split)
pair per shard - convention and dt slices never mix inside a split.
``shard_sha256`` hashes the events-parquet bytes (the shard payload
anchor; T-CORPUS-2/6). Seeds come only from
``schema.seed_from_config`` on the canonical trial spec - never
hand-assigned. A failed trial gets exactly one retry, then a
``deadletter.jsonl`` row {shard_id, spec, error, attempts, utc}; a shard
is written only when every spec has produced a trial row or a
dead-letter row.

Splits (T-CORPUS-4, regenerable from the manifest alone):

- S-STIM:   families are atomic; >=1 held-out family per sensory
            modality + the naturalistic family always held out.
            Perturbation/control families are train-only (``lc_silence``
            is the recorded S-STIM exception; also optogenetic,
            tonic_baseline, babble, silence_test).
- S-TYPE:   seeded stratified 5% of typed cell types; strata =
            region x NT-class, >=1 type per populated stratum.
- S-REGION: >=1 whole somaNeuromere region per superclass-derived
            compartment (optic_lobe / central_brain / vnc). A null
            somaNeuromere is a valid region bucket named "unassigned"
            (the optic-lobe-dominant bucket in MaleCNS v1.0).

Plan refs: docs/plan.md Phase 2 + T-CORPUS-1..7; EVAL.md sections
1.4/1.8/3 + T-SIM-6 (degenerate-trial rule).

Usage:
    python -m flygpt.corpus --config configs/corpus_v1.yaml --graph-dir data/graph --plan
    python -m flygpt.corpus --config configs/corpus_v1.yaml --graph-dir data/graph --run
    python -m flygpt.corpus --config configs/corpus_v1.yaml --graph-dir data/graph --pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from flygpt.schema import (
    EVENT_COLUMNS,
    EVENT_NODE,
    EVENT_STEP,
    EVENT_TRIAL,
    SCHEMA_VERSION,
    TRIAL_COLUMNS,
    GraphBundle,
    canonical_json,
    git_sha,
    load_graph_bundle,
    rng_from_seed,
    seed_from_config,
    sha256_file,
    utcnow,
    write_result,
)

TRAIN = "train"
TEST = "test"

EVENT_SCHEMA = pa.schema(
    [
        pa.field(EVENT_TRIAL, pa.uint64()),
        pa.field(EVENT_STEP, pa.uint32()),
        pa.field(EVENT_NODE, pa.uint32()),
    ]
)

TRIAL_SCHEMA = pa.schema(
    [
        pa.field("trial_id", pa.uint64()),
        pa.field("family_id", pa.string()),
        pa.field("program", pa.string()),
        pa.field("seed", pa.uint64()),
        pa.field("dt", pa.float64()),
        pa.field("n_steps", pa.uint32()),
        pa.field("warmup_steps", pa.uint32()),
        pa.field("n_events", pa.uint32()),
        pa.field("n_active_neurons", pa.uint32()),
        pa.field("n_active_types", pa.uint32()),
        pa.field("split", pa.string()),
        pa.field("degenerate", pa.bool_()),
        pa.field("slice", pa.string()),
    ]
)

MANIFEST_KEYS = (
    "schema",
    "corpus_version",
    "shard_id",
    "code",
    "graph",
    "sim_cfg",
    "backend",
    "gpu_class",
    "dt",
    "cutoff",
    "slice",
    "split",
    "trial_ids",
    "n_trials",
    "n_events",
    "source_manifest",
    "created_utc",
    "shard_sha256",
)

# Sensory modality per stimulus family for S-STIM. Families absent here
# are train-only: perturbation/regime-control families never enter the
# holdout pool (``lc_silence`` is the recorded S-STIM exception -
# benchmark-matched training distribution with trial-disjoint eval).
# Composite families form their own modality ("multisensory").
FAMILY_MODALITY = {
    "olfactory_pulse": "olfactory",
    "gustatory_sugar": "gustatory",
    "gustatory_leg": "gustatory",
    "visual_lc": "visual",
    "mechanosensory_jo": "mechanosensory",
    "mechanosensory_other": "mechanosensory",
    "courtship_cue": "multisensory",
    "naturalistic": "naturalistic",
}

COMPARTMENTS = ("optic_lobe", "central_brain", "vnc")

# Region label used when the region column is absent/null on a node.
REGION_UNASSIGNED = "unassigned"

# T-SIM-6 degenerate-trial thresholds (EVAL.md section 5).
DEGENERATE_TYPE_FRAC_LOW = 0.005
DEGENERATE_TYPE_FRAC_HIGH = 0.60
DEGENERATE_STATIONARITY_R = 0.5
# A drive below this count can never reach the 0.5% active-type floor on
# its own -> sparse perturbation (population criteria waived). On the
# real graph the floor is ~59 driven cells; the rule scales with the
# manifest rather than hardcoding a number per graph.
def _sparse_drive_max(n_types: int) -> int:
    return max(int(DEGENERATE_TYPE_FRAC_LOW * n_types), 1)


# ---------------------------------------------------------------------------
# TrialSpec - the atomic unit of corpus planning
# ---------------------------------------------------------------------------


def _to_jsonable(obj: Any) -> Any:
    """Recursively coerce numpy containers/scalars to JSON-able values."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


@dataclass(frozen=True)
class TrialSpec:
    """One trial: a stimulus program under one slice's conventions.

    ``program`` is the sim-facing spec (drives/silences/duration/dt);
    ``cfg`` is the SimConfig-compatible mapping (dt, weight_mode, cutoff,
    backend, warmup). ``seed`` is derived, never assigned: it is
    ``seed_from_config(program_spec())`` per the global seed rule.
    """

    family_id: str
    program_id: str
    slice: str
    split: str
    program: dict
    cfg: dict

    def program_spec(self) -> dict:
        """Canonical JSON-able identity of this trial."""
        return _to_jsonable(
            {
                "family_id": self.family_id,
                "program_id": self.program_id,
                "slice": self.slice,
                "split": self.split,
                "program": self.program,
                "cfg": self.cfg,
            }
        )

    @property
    def seed(self) -> int:
        """u64 sim seed = hash(canonical program spec)."""
        return seed_from_config(self.program_spec())

    @property
    def trial_id(self) -> int:
        """u64 hash-derived trial identifier (distinct from the seed)."""
        return seed_from_config({"trial_id": self.program_spec()})


# ---------------------------------------------------------------------------
# Lazy seams - sim_backend and stimuli are owned by other components
# ---------------------------------------------------------------------------


def _default_simulate() -> Callable[[GraphBundle, dict, dict], Any]:
    from flygpt.sim_backend import simulate

    return simulate


def _default_library() -> dict[str, Callable]:
    from flygpt.stimuli import LIBRARY

    return LIBRARY


def _default_index(bundle: GraphBundle, stim_cfg: dict | None = None) -> Any:
    from flygpt.stimuli import SensoryIndex

    return SensoryIndex(bundle, stim_cfg)


# ---------------------------------------------------------------------------
# Splits - S-STIM / S-TYPE / S-REGION (EVAL.md section 3, T-CORPUS-4)
# ---------------------------------------------------------------------------


def _family_modality(family_id: str, family_cfg: dict | None) -> str | None:
    """Sensory modality of a family, or None for train-only families."""
    fcfg = family_cfg or {}
    if fcfg.get("modality"):
        return str(fcfg["modality"])
    if family_id in FAMILY_MODALITY:
        return FAMILY_MODALITY[family_id]
    fid = family_id.lower()
    for key, mod in (
        ("olfactory", "olfactory"),
        ("gustatory", "gustatory"),
        ("mechanosensory", "mechanosensory"),
        ("visual", "visual"),
        ("courtship", "multisensory"),
        ("naturalistic", "naturalistic"),
    ):
        if key in fid:
            return mod
    return None


def _node_strings(bundle: GraphBundle, column: str) -> np.ndarray:
    """Object array of str|None for a nodes column (missing -> all None)."""
    if column not in bundle.nodes.columns:
        return np.full(bundle.n_nodes, None, dtype=object)
    return np.array(
        [None if v is None else str(v) for v in bundle.nodes[column].to_list()],
        dtype=object,
    )


def region_labels(bundle: GraphBundle) -> np.ndarray | None:
    """Per-node region labels. ``somaNeuromere`` is THE region field in
    body-annotations; a ``region`` column is accepted for other graphs.
    Returns None when neither column exists (region machinery off)."""
    for col in ("somaNeuromere", "region"):
        if col in bundle.nodes.columns:
            return _node_strings(bundle, col)
    return None


# Internal alias (call sites predate the public name).
_region_labels = region_labels


def _region_display(value: Any) -> str:
    """Display form of a region label; null regions -> 'unassigned'."""
    return str(value) if value else REGION_UNASSIGNED


def _nt_labels(bundle: GraphBundle) -> np.ndarray:
    """Per-node NT-class labels: consensus_nt with documented fallbacks."""
    for col in ("consensus_nt", "predicted_nt", "celltype_predicted_nt"):
        if col in bundle.nodes.columns:
            arr = _node_strings(bundle, col)
            return np.array([v if v else "unclear" for v in arr], dtype=object)
    return np.full(bundle.n_nodes, "unclear", dtype=object)


def _superclass_compartment(superclass: Any) -> str:
    """Map a superclass label to a compartment for S-REGION.

    optic_lobe = ``ol_*`` + ``visual_*`` superclasses (ol_intrinsic,
    ol_sensory, visual_projection, visual_centrifugal, ol_bilateral).
    vnc = ``vnc_*`` + the ascending labels whose somas sit in the cord.
    Everything else (cb_*, descending, ENS, unannotated) lands in
    central_brain. Region -> compartment is decided by majority vote of
    member nodes, so mixed regions are tolerated.
    """
    s = str(superclass or "").lower()
    if s.startswith(("ol_", "visual_")):
        return "optic_lobe"
    if s.startswith(("vnc_", "ascending", "sensory_ascending")):
        return "vnc"
    return "central_brain"


def _stimulus_holdout(
    family_ids: list[str], family_cfgs: dict, seed_tag: str
) -> list[str]:
    """S-STIM: seeded >=1 held-out family per sensory modality.

    Families flagged ``always_held_out`` in the library config (plus the
    naturalistic family by rule) are always in the holdout and count
    toward their modality's >=1. Train-only families (no modality) are
    never candidates.
    """
    fams = sorted(dict.fromkeys(family_ids))
    always = {
        f
        for f in fams
        if (family_cfgs.get(f) or {}).get("always_held_out") or f == "naturalistic"
    }
    holdout = set(always)
    by_mod: dict[str, list[str]] = {}
    for f in fams:
        if f in always:
            continue
        mod = _family_modality(f, family_cfgs.get(f))
        if mod is not None:
            by_mod.setdefault(mod, []).append(f)
    rng = np.random.default_rng(seed_from_config(seed_tag) % (2**63))
    for mod in sorted(by_mod):
        if any(f in holdout for f in by_mod[mod]):
            continue  # modality already satisfied by an always-held-out family
        cand = sorted(by_mod[mod])
        holdout.add(cand[int(rng.integers(len(cand)))])
    return sorted(holdout)


def _celltype_holdout(
    bundle: GraphBundle, fraction: float, seed_tag: str
) -> list[str]:
    """S-TYPE: seeded stratified sample of typed cell types.

    Strata = region x NT-class; each type joins the stratum that holds
    the majority of its member nodes (ties resolve lexicographically via
    np.unique order). Every populated stratum contributes >=1 type, so
    small strata are covered; the per-stratum count is
    ``max(1, round(fraction * stratum_size))``.
    """
    types = bundle.type_array()
    regions = _region_labels(bundle)
    n_nodes = bundle.n_nodes
    region_disp = (
        np.array([_region_display(v) for v in regions], dtype=object)
        if regions is not None
        else np.full(n_nodes, REGION_UNASSIGNED, dtype=object)
    )
    nts = _nt_labels(bundle)
    node_stratum = np.array(
        [f"{r}|{nt}" for r, nt in zip(region_disp, nts, strict=True)], dtype=object
    )
    typed = types != ""
    stratum_of: dict[str, str] = {}
    for t in np.unique(types[typed]):
        mask = typed & (types == t)
        vals, cnts = np.unique(node_stratum[mask], return_counts=True)
        stratum_of[str(t)] = str(vals[int(np.argmax(cnts))])
    by_stratum: dict[str, list[str]] = {}
    for t, s in stratum_of.items():
        by_stratum.setdefault(s, []).append(t)
    rng = np.random.default_rng(seed_from_config(seed_tag) % (2**63))
    held: list[str] = []
    for s in sorted(by_stratum):
        tl = sorted(by_stratum[s])
        k = min(len(tl), max(1, int(round(fraction * len(tl)))))
        idx = np.sort(rng.choice(len(tl), size=k, replace=False))
        held.extend(tl[i] for i in idx)
    return sorted(set(held))


def _region_holdout(
    bundle: GraphBundle, per_compartment: int, seed_tag: str
) -> dict[str, list[str]]:
    """S-REGION: seeded >=1 whole region per compartment.

    Region = somaNeuromere value ("unassigned" for nulls); compartment =
    majority ``_superclass_compartment`` over the region's nodes. A graph
    without a region column yields empty holdouts.
    """
    out: dict[str, list[str]] = {c: [] for c in COMPARTMENTS}
    regions = _region_labels(bundle)
    if regions is None:
        return out
    disp = np.array([_region_display(v) for v in regions], dtype=object)
    comps = np.array(
        [_superclass_compartment(s) for s in _node_strings(bundle, "superclass")],
        dtype=object,
    )
    by_comp: dict[str, set[str]] = {c: set() for c in COMPARTMENTS}
    for r in np.unique(disp):
        mask = disp == r
        vals, cnts = np.unique(comps[mask], return_counts=True)
        by_comp[str(vals[int(np.argmax(cnts))])].add(str(r))
    rng = np.random.default_rng(seed_from_config(seed_tag) % (2**63))
    for c in COMPARTMENTS:
        cand = sorted(by_comp[c])
        k = min(len(cand), int(per_compartment))
        if k:
            idx = np.sort(rng.choice(len(cand), size=k, replace=False))
            out[c] = [cand[i] for i in idx]
    return out


def build_splits(
    corpus_cfg: dict,
    bundle: GraphBundle,
    family_ids: list[str] | None = None,
    library_cfg: dict | None = None,
) -> dict:
    """Compute the corpus ``splits`` block (lands in the manifest).

    The block records rule + seed + membership per axis so membership is
    regenerable from the manifest alone - see ``splits_from_manifest``.
    """
    scfg = corpus_cfg.get("splits", {}) if isinstance(corpus_cfg, dict) else {}
    stim = scfg.get("stimulus_holdout", {})
    cell = scfg.get("celltype_holdout", {})
    reg = scfg.get("region_holdout", {})
    family_cfgs = (library_cfg or {}).get("families", {})
    fams = list(family_ids) if family_ids is not None else list(family_cfgs)

    stim_seed = str(stim.get("seed", "corpus_v1:stimulus_holdout"))
    cell_seed = str(cell.get("seed", "corpus_v1:celltype_holdout"))
    reg_seed = str(reg.get("seed", "corpus_v1:region_holdout"))
    fraction = float(cell.get("fraction", 0.05))
    per_compartment = int(reg.get("per_compartment", 1))

    return {
        "stimulus_holdout": {
            "rule": stim.get(
                "rule",
                ">=1 held-out family per sensory modality + naturalistic always",
            ),
            "seed": stim_seed,
            "families": _stimulus_holdout(fams, family_cfgs, stim_seed),
        },
        "celltype_holdout": {
            "fraction": fraction,
            "strata": list(cell.get("strata", ["region", "nt_class"])),
            "seed": cell_seed,
            "types": _celltype_holdout(bundle, fraction, cell_seed),
        },
        "region_holdout": {
            "per_compartment": per_compartment,
            "seed": reg_seed,
            "regions": _region_holdout(bundle, per_compartment, reg_seed),
        },
    }


def splits_from_manifest(
    manifest: dict,
    bundle: GraphBundle,
    family_ids: list[str] | None = None,
    library_cfg: dict | None = None,
) -> dict:
    """Regenerate the splits block from a corpus manifest (T-CORPUS-4).

    The manifest's ``splits`` block carries the rule + seed fields
    verbatim; rerunning ``build_splits`` on them must reproduce the
    recorded membership exactly.
    """
    return build_splits(
        {"splits": manifest["splits"]},
        bundle,
        family_ids=family_ids,
        library_cfg=library_cfg,
    )


# ---------------------------------------------------------------------------
# Health metrics (T-SIM-6) - delegates to sim_backend when it provides one
# ---------------------------------------------------------------------------


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r tolerant of constant series: identical constants -> 1,
    otherwise undefined -> 0."""
    if a.size == 0:
        return 1.0
    sa, sb = float(a.std()), float(b.std())
    if sa == 0.0 and sb == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _type_frac(nodes_fired: np.ndarray, types: np.ndarray, n_types: int) -> float:
    if n_types == 0 or nodes_fired.size == 0:
        return 0.0
    fired_types = {str(types[i]) for i in np.unique(nodes_fired) if types[i]}
    return len(fired_types) / n_types


def _health_metrics_local(
    events_step: np.ndarray,
    events_node: np.ndarray,
    n_nodes: int,
    n_steps: int,
    types: np.ndarray,
    dt: float | None = None,
    n_driven: int = -1,
) -> dict:
    """Per-trial health: pct active, rate stats, split-half stationarity,
    degenerate flag (T-SIM-6).

    ``types`` is the per-node type array (bundle.type_array()). The
    degenerate rule is: active-type fraction < 0.5%, or > 60% sustained
    (>=60% of manifest types active in BOTH halves), or per-neuron
    split-half stationarity r < 0.5. Stationarity compares per-neuron
    event counts between the two halves - drift/decay across the trial
    drops r below threshold.

    ``n_driven`` applies the T-SIM-6 sparse/silence semantics: an
    explicit ``0`` marks a silence control (degenerate only on a >60%
    blow-up) and drives under ~0.5% of the manifest's types are sparse
    perturbations (degenerate only on a zero event stream or the same
    blow-up). The default ``-1`` means "unspecified" - the conservative
    population rule applies, so a silent trial still flags degenerate
    for callers that do not know the program's drive size.
    """
    events_step = np.asarray(events_step, dtype=np.int64)
    events_node = np.asarray(events_node, dtype=np.int64)
    types = np.asarray(types, dtype=object)
    n_steps = int(n_steps)
    n_events = int(events_node.size)
    counts = (
        np.bincount(events_node, minlength=n_nodes)
        if n_events
        else np.zeros(n_nodes, dtype=np.int64)
    )
    active = counts > 0
    n_active_neurons = int(active.sum())
    n_types = len({str(t) for t in types if t})
    n_active_types = len({str(types[i]) for i in np.flatnonzero(active) if types[i]})
    type_frac = n_active_types / n_types if n_types else 0.0

    dur_s = max(n_steps * float(dt), 1e-9) if dt else float(max(n_steps, 1))
    rates = counts[active] / dur_s

    half = n_steps // 2
    first = events_node[events_step < half]
    second = events_node[events_step >= half]
    c1 = np.bincount(first, minlength=n_nodes) if first.size else np.zeros(n_nodes, np.int64)
    c2 = np.bincount(second, minlength=n_nodes) if second.size else np.zeros(n_nodes, np.int64)
    stationarity = _safe_pearson(c1, c2)
    f1 = _type_frac(first, types, n_types)
    f2 = _type_frac(second, types, n_types)
    if n_driven == 0:
        # Silence control: expected silence is healthy; only a blow-up
        # (impossible without a drive, but checked anyway) is degenerate.
        degenerate = bool(min(f1, f2) > DEGENERATE_TYPE_FRAC_HIGH)
    elif n_driven < _sparse_drive_max(n_types):
        # Sparse perturbation: flagged when it yields no events at all
        # (dead drive) or when the whole brain blows up anyway.
        degenerate = bool(
            n_events == 0 or min(f1, f2) > DEGENERATE_TYPE_FRAC_HIGH
        )
    else:
        degenerate = bool(
            type_frac < DEGENERATE_TYPE_FRAC_LOW
            or min(f1, f2) > DEGENERATE_TYPE_FRAC_HIGH
            or stationarity < DEGENERATE_STATIONARITY_R
        )
    return {
        "n_events": n_events,
        "n_active_neurons": n_active_neurons,
        "n_active_types": n_active_types,
        "pct_active_neurons": n_active_neurons / max(n_nodes, 1),
        "type_frac_active": type_frac,
        "rate_mean_hz": float(rates.mean()) if rates.size else 0.0,
        "rate_median_hz": float(np.median(rates)) if rates.size else 0.0,
        "rate_p05_hz": float(np.percentile(rates, 5)) if rates.size else 0.0,
        "rate_p95_hz": float(np.percentile(rates, 95)) if rates.size else 0.0,
        "stationarity_r": stationarity,
        "degenerate": degenerate,
    }


def health_metrics(
    events_step: np.ndarray,
    events_node: np.ndarray,
    n_nodes: int,
    n_steps: int,
    types: np.ndarray,
    dt: float | None = None,
    n_driven: int = -1,
) -> dict:
    """Health metrics per trial over raw event arrays + the type array.

    Distinct from ``sim_backend.health_metrics`` (which consumes a
    ``TrialResult`` and reports different fields); this is the corpus
    trial-row implementation and the two are not interchangeable.
    """
    return _health_metrics_local(
        events_step, events_node, n_nodes, n_steps, types, dt=dt, n_driven=n_driven
    )


# ---------------------------------------------------------------------------
# Shard writing + validation (T-CORPUS-1/2)
# ---------------------------------------------------------------------------


def write_shard(
    events: dict[str, np.ndarray],
    trial_rows: list[dict],
    manifest: dict,
    work_dir: Path,
) -> dict[str, Path]:
    """Write one shard's three artifacts. Returns the path map.

    ``events``: equal-length arrays under EVENT_COLUMNS keys (sorted
    here). ``manifest``: every MANIFEST_KEYS entry except ``trial_ids``,
    ``n_trials``, ``n_events``, ``created_utc``, ``shard_sha256``, which
    are computed from the payload.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shard_id = manifest["shard_id"]

    tid = np.asarray(events[EVENT_TRIAL], dtype=np.uint64)
    step = np.asarray(events[EVENT_STEP], dtype=np.uint32)
    node = np.asarray(events[EVENT_NODE], dtype=np.uint32)
    order = np.lexsort((node, step, tid))
    events_path = work_dir / f"events-{shard_id}.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {EVENT_TRIAL: tid[order], EVENT_STEP: step[order], EVENT_NODE: node[order]},
            schema=EVENT_SCHEMA,
        ),
        events_path,
        compression="zstd",
    )

    trials_path = work_dir / f"trials-{shard_id}.parquet"
    pq.write_table(
        pa.Table.from_pylist(trial_rows, schema=TRIAL_SCHEMA),
        trials_path,
        compression="zstd",
    )

    full = dict(manifest)
    full["trial_ids"] = sorted(int(r["trial_id"]) for r in trial_rows)
    full["n_trials"] = len(trial_rows)
    full["n_events"] = int(tid.size)
    full["created_utc"] = utcnow()
    full["shard_sha256"] = sha256_file(events_path)
    full["trials_sha256"] = sha256_file(trials_path)
    manifest_path = work_dir / f"{shard_id}.manifest.json"
    manifest_path.write_text(json.dumps(full, indent=2, default=str) + "\n")
    return {"events": events_path, "trials": trials_path, "manifest": manifest_path}


def validate_shard(
    events_path: Path, trials_path: Path, manifest_path: Path
) -> dict:
    """Validate one shard against the declared schema (T-CORPUS-1/2).

    Returns a checks dict; ``ok`` is the conjunction of all checks.
    """
    events_path = Path(events_path)
    trials_path = Path(trials_path)
    manifest_path = Path(manifest_path)
    checks: dict[str, Any] = {}

    ev = pq.read_table(events_path)
    tr = pq.read_table(trials_path)
    man = json.loads(manifest_path.read_text())

    checks["events_columns"] = ev.schema.names == list(EVENT_COLUMNS)
    checks["events_dtypes"] = (
        ev.schema.field(EVENT_TRIAL).type == pa.uint64()
        and ev.schema.field(EVENT_STEP).type == pa.uint32()
        and ev.schema.field(EVENT_NODE).type == pa.uint32()
    )
    tid = ev.column(EVENT_TRIAL).to_numpy()
    step = ev.column(EVENT_STEP).to_numpy()
    node = ev.column(EVENT_NODE).to_numpy()
    order = np.lexsort((node, step, tid))
    checks["events_sorted"] = bool(np.array_equal(order, np.arange(order.size)))

    checks["trials_columns"] = tr.schema.names == list(TRIAL_COLUMNS)
    expected = {f.name: f.type for f in TRIAL_SCHEMA}
    checks["trials_dtypes"] = all(
        tr.schema.field(name).type == expected[name] for name in tr.schema.names
    )

    checks["manifest_keys"] = all(k in man for k in MANIFEST_KEYS)
    trial_ids = sorted(int(v) for v in tr.column("trial_id").to_pylist())
    checks["trial_ids_match"] = (
        sorted(int(v) for v in man.get("trial_ids", [])) == trial_ids
        and set(np.unique(tid).tolist()) <= set(trial_ids)
    )
    checks["n_trials_match"] = man.get("n_trials") == tr.num_rows
    checks["n_events_match"] = man.get("n_events") == ev.num_rows
    checks["shard_sha256"] = man.get("shard_sha256") == sha256_file(events_path)
    checks["trials_sha256"] = man.get("trials_sha256") == sha256_file(trials_path)
    checks["slice_pure"] = bool(
        len(set(tr.column("slice").to_pylist()) | {man.get("slice")}) == 1
        and len(set(tr.column("split").to_pylist()) | {man.get("split")}) == 1
    )
    checks["dt_present"] = isinstance(man.get("dt"), (int, float)) and bool(
        (tr.column("dt").to_numpy() == float(man["dt"])).all()
        if tr.num_rows
        else True
    )
    checks["ok"] = all(v for v in checks.values() if isinstance(v, bool))
    return checks


# ---------------------------------------------------------------------------
# Trial runner (T-CORPUS-1/2/6)
# ---------------------------------------------------------------------------


def _spec_dt(spec: TrialSpec, result: Any = None) -> float:
    """The trial's dt: spec cfg wins, then result config, then program."""
    if "dt" in spec.cfg:
        return float(spec.cfg["dt"])
    cfg = getattr(result, "config", None)
    dt = cfg.get("dt") if isinstance(cfg, dict) else getattr(cfg, "dt", None)
    if dt is not None:
        return float(dt)
    if "dt" in spec.program:
        return float(spec.program["dt"])
    raise KeyError(f"no dt in cfg/program for {spec.program_id}")


def _spec_n_steps(spec: TrialSpec, result: Any = None) -> int:
    n = getattr(result, "n_steps", None)
    if n is not None:
        return int(n)
    duration = float(spec.program.get("duration_s", 1.0))
    return int(round(duration / _spec_dt(spec, result)))


def _try_simulate(
    simulate_fn: Callable[[GraphBundle, dict, dict], Any],
    bundle: GraphBundle,
    spec: TrialSpec,
) -> tuple[Any | None, str | None]:
    """One attempt + exactly one retry. (result, None) or (None, error)."""
    try:
        return simulate_fn(bundle, spec.program, spec.cfg), None
    except Exception as exc:
        first = exc
    try:
        return simulate_fn(bundle, spec.program, spec.cfg), None
    except Exception as exc:
        return None, f"attempt1={first!r} | retry={exc!r}"


def _dead_letter(path: Path, shard_id: str, spec: TrialSpec, error: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "shard_id": shard_id,
        "spec": spec.program_spec(),
        "error": error,
        "attempts": 2,
        "utc": utcnow(),
    }
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=str) + "\n")


def _run_shard(
    bundle: GraphBundle,
    shard_specs: list[TrialSpec],
    simulate_fn: Callable[[GraphBundle, dict, dict], Any],
    type_arr: np.ndarray,
    work_dir: Path,
    shard_id: str,
    slice_name: str,
    split: str,
    *,
    corpus_version: str,
    graph_sha: str,
    graph_version: str,
    backend: str,
    gpu_class: str,
    dead_letter_path: Path,
    source_manifest: str | None,
) -> Path:
    """Run one shard's trials, dead-letter failures, write artifacts."""
    ev_trial: list[np.ndarray] = []
    ev_step: list[np.ndarray] = []
    ev_node: list[np.ndarray] = []
    rows: list[dict] = []
    shuffle_seed: int | None = None
    if any(spec.cfg.get("weight_shuffle") for spec in shard_specs):
        # Shiu's control lifted to corpus level: same topology, (weight,
        # sign) pairs jointly permuted. One deterministic shuffle per
        # shard so regeneration from manifest+seed is bitwise stable
        # (T-CORPUS-6).
        from flygpt.controls import shuffled_weights

        shuffle_seed = int(
            seed_from_config(
                (
                    "corpus:shuffled_weights",
                    corpus_version,
                    slice_name,
                    shard_id,
                    int(bundle.out_indices.size),
                )
            )
        )
        bundle = shuffled_weights(bundle, shuffle_seed)
    for spec in shard_specs:
        result, err = _try_simulate(simulate_fn, bundle, spec)
        if result is None:
            _dead_letter(dead_letter_path, shard_id, spec, err or "unknown")
            continue
        step = np.asarray(result.events_step, dtype=np.uint32)
        node = np.asarray(result.events_node, dtype=np.uint32)
        dt = _spec_dt(spec, result)
        n_steps = _spec_n_steps(spec, result)
        n_driven = sum(
            len(d.get("nodes", ())) for d in spec.program.get("drives", ())
        )
        hm = health_metrics(
            step, node, bundle.n_nodes, n_steps, type_arr, dt=dt, n_driven=n_driven
        )
        tid = spec.trial_id
        ev_trial.append(np.full(step.size, tid, dtype=np.uint64))
        ev_step.append(step)
        ev_node.append(node)
        rows.append(
            {
                "trial_id": tid,
                "family_id": spec.family_id,
                "program": canonical_json(_to_jsonable(spec.program)),
                "seed": spec.seed,
                "dt": float(dt),
                "n_steps": n_steps,
                "warmup_steps": int(spec.cfg.get("warmup_steps", 0) or 0),
                "n_events": int(step.size),
                "n_active_neurons": hm["n_active_neurons"],
                "n_active_types": hm["n_active_types"],
                "split": spec.split,
                "degenerate": bool(hm["degenerate"]),
                "slice": spec.slice,
            }
        )

    events = {
        EVENT_TRIAL: np.concatenate(ev_trial) if ev_trial else np.empty(0, np.uint64),
        EVENT_STEP: np.concatenate(ev_step) if ev_step else np.empty(0, np.uint32),
        EVENT_NODE: np.concatenate(ev_node) if ev_node else np.empty(0, np.uint32),
    }
    manifest = {
        "schema": SCHEMA_VERSION,
        "corpus_version": corpus_version,
        "shard_id": shard_id,
        "code": {"git_sha": git_sha()},
        "graph": {"version": graph_version, "sha256": graph_sha},
        "sim_cfg": _to_jsonable(shard_specs[0].cfg) if shard_specs else {},
        "backend": backend,
        "gpu_class": gpu_class,
        "dt": _spec_dt(shard_specs[0]) if shard_specs else None,
        "cutoff": bundle.cutoff,
        "slice": slice_name,
        "split": split,
        "source_manifest": source_manifest,
    }
    if shuffle_seed is not None:
        manifest["weight_shuffle_seed"] = shuffle_seed
    return write_shard(events, rows, manifest, work_dir)["manifest"]


def run_trials(
    bundle: GraphBundle,
    specs: list[TrialSpec],
    simulate_fn: Callable[[GraphBundle, dict, dict], Any] | None = None,
    work_dir: Path | str = ".",
    shard_prefix: str = "shard",
    trials_per_shard: int = 1000,
    corpus_version: str = "corpus_v1",
    graph_sha: str = "",
    dead_letter_path: Path | str | None = None,
    backend: str = "numpy",
    gpu_class: str = "cpu",
    graph_version: str = "",
    source_manifest: str | None = None,
) -> list[Path]:
    """Run TrialSpecs grouped into (slice, split) shards.

    ``simulate_fn(bundle, program, cfg) -> TrialResult`` is injected for
    testability; the default lazily imports
    ``flygpt.sim_backend.simulate``. Returns the manifest paths written.
    """
    if simulate_fn is None:
        simulate_fn = _default_simulate()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dl_path = Path(dead_letter_path) if dead_letter_path else work_dir / "deadletter.jsonl"
    type_arr = bundle.type_array()

    groups: dict[tuple[str, str], list[TrialSpec]] = {}
    for spec in specs:
        groups.setdefault((spec.slice, spec.split), []).append(spec)

    manifests: list[Path] = []
    for (slice_name, split), gspecs in groups.items():
        for i in range(0, len(gspecs), trials_per_shard):
            shard_specs = gspecs[i : i + trials_per_shard]
            shard_id = f"{shard_prefix}-{slice_name}-{split}-{i // trials_per_shard:04d}"
            manifests.append(
                _run_shard(
                    bundle,
                    shard_specs,
                    simulate_fn,
                    type_arr,
                    work_dir,
                    shard_id,
                    slice_name,
                    split,
                    corpus_version=corpus_version,
                    graph_sha=graph_sha,
                    graph_version=graph_version,
                    backend=backend,
                    gpu_class=gpu_class,
                    dead_letter_path=dl_path,
                    source_manifest=source_manifest,
                )
            )
    return manifests


# ---------------------------------------------------------------------------
# Corpus planning - corpus_v1.yaml slices + stim library -> TrialSpec list
# ---------------------------------------------------------------------------


def _load_yaml_path(path: Any) -> dict:
    if not path:
        return {}
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def _load_sim_cfg(corpus_cfg: dict) -> dict:
    return _load_yaml_path(corpus_cfg.get("sim_config"))


def _program_dict(program: Any) -> dict:
    """Normalize a StimulusProgram-like object or dict to a plain dict."""
    if hasattr(program, "spec") and callable(program.spec):
        return dict(program.spec())
    return dict(program)


def _driven_nodes(program: dict) -> np.ndarray:
    nodes = [
        np.asarray(d.get("nodes", []), dtype=np.int64)
        for d in program.get("drives", [])
    ]
    nodes = [n for n in nodes if n.size]
    return np.concatenate(nodes) if nodes else np.empty(0, np.int64)


def _split_label(
    family_id: str,
    program: dict,
    type_arr: np.ndarray | None,
    regions: np.ndarray | None,
    stim_held: set,
    type_held: set,
    region_held: set,
) -> str:
    """test iff the family is held out or the program's driven targets
    touch a held-out type/region (S-STIM/S-TYPE/S-REGION)."""
    if family_id in stim_held:
        return TEST
    driven = _driven_nodes(program)
    if driven.size:
        if type_arr is not None and type_held:
            driven = driven[driven < type_arr.size]
            if {str(t) for t in type_arr[driven] if t} & type_held:
                return TEST
        if regions is not None and region_held and driven.size:
            driven = driven[driven < regions.size]
            if {_region_display(r) for r in regions[driven]} & region_held:
                return TEST
    return TRAIN


def _slice_sim_cfg(sim_cfg: dict, slice_name: str, slice_cfg: dict) -> dict:
    """SimConfig-compatible mapping for one slice (dt/weight_mode/cutoff
    are the slice's own; weight_shuffle flags the shuffled slice)."""
    model = dict(sim_cfg.get("model", sim_cfg))
    cfg = dict(model)
    cfg["dt"] = float(slice_cfg.get("dt", model.get("dt", 0.020)))
    cfg["weight_mode"] = slice_cfg.get("weight_mode", model.get("weight_mode", "count"))
    cfg["cutoff"] = int(slice_cfg.get("cutoff", sim_cfg.get("cutoff", 5)))
    cfg["weight_shuffle"] = bool(slice_cfg.get("weight_shuffle", False))
    cfg["slice"] = slice_name
    cfg["backend"] = sim_cfg.get("backend", "numpy")
    cfg["gpu_class"] = sim_cfg.get("gpu_class", "cpu")
    return cfg


def plan_corpus(
    cfg: dict,
    index: Any = None,
    *,
    bundle: GraphBundle | None = None,
    library: dict[str, Callable] | None = None,
    library_cfg: dict | None = None,
    sim_cfg: dict | None = None,
    splits: dict | None = None,
    max_trials: int | None = None,
    slices: list[str] | None = None,
) -> list[TrialSpec]:
    """Expand corpus_v1.yaml slices x stim library into the TrialSpec list.

    ``index`` is the SensoryIndex-like resolver (carries ``.bundle`` and
    answers family grounding queries); ``library`` maps family_id ->
    program generator ``(index, rng, **family_cfg) -> [program]`` and
    defaults to the lazily imported ``flygpt.stimuli.LIBRARY``. Slice
    attributes (dt/weight_mode/cutoff) land in each spec's cfg; the
    shuffled slice is flagged via ``weight_shuffle``.
    """
    sim_cfg = sim_cfg if sim_cfg is not None else _load_sim_cfg(cfg)
    library_cfg = library_cfg if library_cfg is not None else _load_yaml_path(cfg.get("stim_library"))
    family_cfgs = library_cfg.get("families", {})
    family_ids = list(family_cfgs)
    bundle = bundle if bundle is not None else getattr(index, "bundle", None)
    splits = (
        splits
        if splits is not None
        else build_splits(cfg, bundle, family_ids, library_cfg)
        if bundle is not None
        else {"stimulus_holdout": {"families": []},
              "celltype_holdout": {"types": []},
              "region_holdout": {"regions": {c: [] for c in COMPARTMENTS}}}
    )
    stim_held = set(splits["stimulus_holdout"]["families"])
    type_held = set(splits["celltype_holdout"]["types"])
    region_held = set()
    for regs in splits["region_holdout"]["regions"].values():
        region_held.update(regs)

    library = library if library is not None else _default_library()
    if index is None and bundle is not None:
        index = _default_index(bundle, library_cfg)
    version = cfg.get("version", "corpus")
    programs: list[tuple[str, dict]] = []
    for fam in family_ids:
        gen = library.get(fam)
        if gen is None:
            continue
        gen_kw = {k: v for k, v in family_cfgs.get(fam, {}).items() if k != "slices"}
        n_prog = int(gen_kw.get("n_programs", 1))
        rng = rng_from_seed("stim", version, fam)
        # one call = one program; `trial` differentiates draws so distinct
        # programs sample distinct drives under the family seed stream.
        for i in range(n_prog):
            pd = _program_dict(gen(index, rng, gen_kw, trial=i))
            pd.setdefault("family_id", fam)
            pd.setdefault("program_id", f"{fam}-{i:04d}")
            programs.append((fam, pd))

    type_arr = bundle.type_array() if bundle is not None else None
    regions = _region_labels(bundle) if bundle is not None else None

    specs: list[TrialSpec] = []
    slice_filter = set(slices) if slices else None
    for slice_name, sc in (cfg.get("slices") or {}).items():
        if slice_filter is not None and slice_name not in slice_filter:
            continue
        # With a slice filter the cap applies per slice so capped runs
        # still exercise every requested slice; unfiltered it stays a
        # global cap for backward compatibility.
        if slice_filter is not None and max_trials is not None:
            sc = dict(sc, n_trials=min(int(sc.get("n_trials", 0)), int(max_trials)))
        eligible = [
            (fam, pd)
            for fam, pd in programs
            if slice_name in family_cfgs.get(fam, {}).get("slices", cfg.get("slices", {}).keys())
        ]
        n_target = int(sc.get("n_trials", 0))
        base, rem = divmod(n_target, max(len(eligible), 1))
        for i, (fam, pd) in enumerate(eligible):
            reps = base + (1 if i < rem else 0)
            for rep in range(reps):
                program = dict(pd)
                program["rep"] = rep
                split = _split_label(
                    fam, program, type_arr, regions, stim_held, type_held, region_held
                )
                specs.append(
                    TrialSpec(
                        fam,
                        str(pd["program_id"]),
                        slice_name,
                        split,
                        program,
                        _slice_sim_cfg(sim_cfg, slice_name, sc),
                    )
                )
                if (
                    max_trials is not None
                    and slice_filter is None
                    and len(specs) >= max_trials
                ):
                    return specs
    return specs


# ---------------------------------------------------------------------------
# Pilot gate (Phase 1, T-SIM-6/7 harness)
# ---------------------------------------------------------------------------


def run_pilot(
    bundle: GraphBundle,
    library: dict[str, Callable],
    cfg: dict,
    work_dir: Path,
    results_dir: Path,
    *,
    index: Any = None,
    simulate_fn: Callable[[GraphBundle, dict, dict], Any] | None = None,
    stim_cfg: dict | None = None,
    sim_cfg: dict | None = None,
) -> dict:
    """Phase-1 pilot gate: N families x M trials + health distribution +
    throughput measurement -> results/pilot_report.json.

    Pilot sizing comes from the stim library's ``pilot`` block
    (n_families x trials_per_family, duration, dt). The report carries
    the T-SIM-6 pilot-gate band (per-family median +- 3xMAD of
    pct_active_neurons and of the event rate), health metrics, and
    throughput (trials/s, events/s, bytes/trial).
    """
    library = library if library is not None else _default_library()
    stim_cfg = stim_cfg or _load_yaml_path(cfg.get("stim_library")) or cfg
    sim_cfg = sim_cfg if sim_cfg is not None else _load_sim_cfg(cfg)
    pcfg = dict(stim_cfg.get("pilot", {}))
    n_families = int(pcfg.get("n_families", 10))
    n_trials = int(pcfg.get("trials_per_family", 100))
    dt = float(pcfg.get("dt", 0.020))
    duration = float(pcfg.get("duration_s", 1.0))
    # The pilot's 10 families are the grounded stimulus families in config
    # order - regime controls (babble / tonic_baseline / silence_test)
    # are sim-harness utilities, not pilot material (alphabetical slicing
    # would pull babble in and drop visual_lc).
    regime_controls = {"babble", "tonic_baseline", "silence_test"}
    ordered = [
        f for f in (stim_cfg.get("families") or {})
        if f not in regime_controls and f in library
    ]
    fams = ordered[:n_families] or sorted(
        f for f in library if f not in regime_controls
    )[:n_families]
    if index is None:
        index = _default_index(bundle, stim_cfg)

    specs: list[TrialSpec] = []
    for fam in fams:
        fam_cfg = dict(stim_cfg.get("families", {}).get(fam, {}))
        fam_cfg.pop("slices", None)
        rng = rng_from_seed("pilot", cfg.get("version", "corpus"), fam)
        progs = [_program_dict(library[fam](index, rng, fam_cfg))]
        if not progs:
            continue
        pd = progs[0]
        pd.setdefault("duration_s", duration)
        pd.setdefault("dt", dt)
        pid = str(pd.get("program_id", fam))
        for rep in range(n_trials):
            program = dict(pd)
            program["rep"] = rep
            specs.append(
                TrialSpec(
                    fam,
                    pid,
                    "pilot",
                    TRAIN,
                    program,
                    _slice_sim_cfg(sim_cfg, "pilot", {"dt": dt}),
                )
            )

    work_dir = Path(work_dir)
    # Graph provenance for every pilot shard: the build manifest carries
    # the graph_version and its own sha256 fingerprints the built state.
    graph_sha, graph_version = "", ""
    gman = Path("provenance/graph_v1.json")
    if gman.exists():
        try:
            gdoc = json.loads(gman.read_text())
            graph_version = str(gdoc.get("graph_version", ""))
            graph_sha = hashlib.sha256(gman.read_bytes()).hexdigest()
        except Exception:
            pass

    # Pilot-scope corpus manifest: the declared coverage contract lives
    # next to the shards (corpus.manifest.json) so T-CORPUS-3's driven-
    # type denominator is this corpus's own planned set, not the whole
    # brain manifest - a 10-family pilot cannot cover all types.
    driven_types = _driven_type_set(bundle, specs)
    (work_dir / "pilot").mkdir(parents=True, exist_ok=True)
    _write_corpus_manifest(
        work_dir / "pilot",
        cfg_version=str(cfg.get("version", "corpus_v1")),
        driven_types=driven_types,
        splits={"pilot": TRAIN},
        slices=["pilot"],
        backend=str(sim_cfg.get("backend", "numpy")),
        gpu_class=str(sim_cfg.get("gpu_class", "cpu")),
        graph_version=graph_version,
        graph_sha=graph_sha,
    )

    t0 = time.time()
    manifests = run_trials(
        bundle,
        specs,
        simulate_fn,
        work_dir / "pilot",
        shard_prefix="pilot",
        trials_per_shard=max(n_trials, 1),
        corpus_version=cfg.get("version", "corpus_v1"),
        backend=str(sim_cfg.get("backend", "numpy")),
        gpu_class=str(sim_cfg.get("gpu_class", "cpu")),
        graph_sha=graph_sha,
        graph_version=graph_version,
        source_manifest="pilot",
    )
    wall = time.time() - t0

    rows: list[dict] = []
    n_events = 0
    total_bytes = 0
    for mp in manifests:
        man = json.loads(mp.read_text())
        rows.extend(pq.read_table(mp.parent / f"trials-{man['shard_id']}.parquet").to_pylist())
        n_events += int(man["n_events"])
        total_bytes += (
            mp.parent.joinpath(f"events-{man['shard_id']}.parquet").stat().st_size
            + mp.parent.joinpath(f"trials-{man['shard_id']}.parquet").stat().st_size
        )
    n_done = len(rows)
    deg = [bool(r["degenerate"]) for r in rows]
    deg_frac = float(np.mean(deg)) if deg else 0.0

    band: dict[str, dict] = {}
    for fam in sorted({r["family_id"] for r in rows}):
        fr = [r for r in rows if r["family_id"] == fam]
        pct = np.array([r["n_active_neurons"] / max(bundle.n_nodes, 1) for r in fr])
        ev_rate = np.array(
            [r["n_events"] / max(r["n_steps"] * r["dt"], 1e-9) for r in fr]
        )
        band[fam] = {
            "pct_active_median": float(np.median(pct)),
            "pct_active_mad": float(np.median(np.abs(pct - np.median(pct)))),
            "event_rate_median": float(np.median(ev_rate)),
            "event_rate_mad": float(np.median(np.abs(ev_rate - np.median(ev_rate)))),
        }

    metrics = {
        "n_trials": n_done,
        "n_events": n_events,
        "wall_seconds": round(wall, 3),
        "trials_per_s": round(n_done / max(wall, 1e-9), 3),
        "events_per_s": round(n_events / max(wall, 1e-9), 3),
        "bytes_per_trial": round(total_bytes / max(n_done, 1), 1),
        "degenerate_fraction": deg_frac,
    }
    checks = {
        "all_trials_completed": n_done == len(specs),
        "degenerate_fraction_le_5pct": deg_frac <= 0.05,
        "n_families": len(fams),
        "n_shards": len(manifests),
    }
    return write_result(
        Path(results_dir),
        "pilot_report",
        checks,
        inputs={"pilot": pcfg, "corpus_version": cfg.get("version")},
        metrics=metrics,
        passed=bool(checks["all_trials_completed"] and checks["degenerate_fraction_le_5pct"]),
        extra={"test_id": "pilot_report", "sane_band": band},
    )


def _driven_type_set(bundle: GraphBundle, specs: list[TrialSpec]) -> list[str]:
    """Union of cell types this trial plan drives (T-CORPUS-3 scope)."""
    types = np.asarray(bundle.type_array(), dtype=object)
    seen: set[str] = set()
    for s in specs:
        for d in (s.program or {}).get("drives", ()):
            idx = np.asarray(d.get("nodes", ()), dtype=np.int64)
            idx = idx[(idx >= 0) & (idx < types.size)]
            for t in types[idx]:
                if t:
                    seen.add(str(t))
    return sorted(seen)


def _write_corpus_manifest(
    corpus_dir: Path,
    *,
    cfg_version: str,
    driven_types: list[str],
    splits: dict,
    slices,
    backend: str,
    gpu_class: str,
    graph_version: str = "",
    graph_sha: str = "",
) -> Path:
    """In-dir corpus manifest - the declared coverage scope for T-CORPUS-3.

    ``types`` is the union of cell types the trial plan drives - the
    corpus's own coverage contract, so a scope-limited corpus (pilot,
    subset run) is judged against what it declared rather than the full
    brain manifest.
    """
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": SCHEMA_VERSION,
        "corpus_version": cfg_version,
        "created_utc": utcnow(),
        "code": {"git_sha": git_sha()},
        "types": list(driven_types),
        "n_types": len(driven_types),
        "splits": splits,
        "slices": slices,
        "backend": backend,
        "gpu_class": gpu_class,
        "graph": {"version": graph_version, "sha256": graph_sha},
    }
    path = corpus_dir / "corpus.manifest.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_corpus(
    graph_dir: Path,
    corpus_config: Path,
    stim_config: Path | None = None,
    work_dir: Path | str = "data/corpus",
    slices: list[str] | None = None,
    backend: str = "numpy",
    gpu_class: str = "cpu",
    max_shards: int | None = None,
    results_dir: Path | str = "results",
    shard_prefix: str = "shard",
    max_trials: int | None = None,
) -> list[Path]:
    """Driver-facing corpus generation entry (notebooks/phase2, Modal).

    Plans the full trial set (optionally restricted to ``slices``),
    writes the corpus manifest, runs shards, and returns the written
    manifest paths - capped at ``max_shards`` for checkpoint-resume
    sessions. ``gpu_class`` is REQUIRED for real shards: manifests record
    the actual device class, never a guess.
    """
    cfg = yaml.safe_load(Path(corpus_config).read_text())
    cfg["_config_path"] = str(corpus_config)
    main_slice = (cfg.get("slices") or {}).get("main", {})
    bundle = load_graph_bundle(Path(graph_dir), cutoff=int(main_slice.get("cutoff", 5)))
    library_cfg = _load_yaml_path(stim_config or cfg.get("stim_library"))
    specs = plan_corpus(cfg, bundle=bundle, max_trials=max_trials, slices=slices)
    splits = build_splits(cfg, bundle, library_cfg=library_cfg)

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(
        cfg.get("output", {}).get("manifest", "provenance/corpus_v1.json")
    )
    graph_nodes = Path(graph_dir) / "nodes.parquet"
    corpus_manifest = {
        "schema": SCHEMA_VERSION,
        "corpus_version": cfg.get("version"),
        "created_utc": utcnow(),
        "code": {"git_sha": git_sha()},
        "config": {"path": str(corpus_config), "sha256": sha256_file(Path(corpus_config))},
        "graph": {
            "dir": str(graph_dir),
            "cutoff": bundle.cutoff,
            "nodes_sha256": sha256_file(graph_nodes) if graph_nodes.exists() else "",
        },
        "splits": splits,
        "slices": slices or cfg.get("slices"),
        "backend": backend,
        "gpu_class": gpu_class,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(corpus_manifest, indent=2, default=str) + "\n")

    # In-dir manifest declares the plan's driven-type scope (T-CORPUS-3
    # denominator) next to the shards.
    _write_corpus_manifest(
        work_dir,
        cfg_version=str(cfg.get("version", "corpus_v1")),
        driven_types=_driven_type_set(bundle, specs),
        splits=splits,
        slices=slices or list(cfg.get("slices") or {}),
        backend=backend,
        gpu_class=gpu_class,
        graph_version="graph_v1",
        graph_sha=corpus_manifest["graph"]["nodes_sha256"],
    )

    tps = int(cfg.get("output", {}).get("trials_per_shard", 1000))
    if max_shards is not None:
        # checkpoint-resume: cap the plan, not the shard size
        specs = specs[: max_shards * tps]

    # Multi-cutoff corpora: each slice's spec.cfg carries its own cutoff;
    # group the plan by it and simulate each group on its own CSR so a
    # cutoff-3 slice never sees the cutoff-5 graph.
    groups: dict[int, list[TrialSpec]] = {}
    for s in specs:
        groups.setdefault(int((s.cfg or {}).get("cutoff", bundle.cutoff)), []).append(s)
    bundles = {bundle.cutoff: bundle}
    manifests: list[Path] = []
    for cutoff, gspecs in sorted(groups.items()):
        if cutoff not in bundles:
            bundles[cutoff] = load_graph_bundle(Path(graph_dir), cutoff=cutoff)
        manifests.extend(
            run_trials(
                bundles[cutoff],
                gspecs,
                None,
                work_dir,
                shard_prefix=shard_prefix,
                trials_per_shard=tps,
                corpus_version=str(cfg.get("version", "corpus_v1")),
                graph_sha=corpus_manifest["graph"]["nodes_sha256"],
                dead_letter_path=cfg.get("output", {}).get("dead_letter"),
                backend=backend,
                gpu_class=gpu_class,
                graph_version="graph_v1",
                source_manifest=str(manifest_path),
            )
        )
    corpus_manifest["shards"] = [str(p) for p in manifests]
    manifest_path.write_text(json.dumps(corpus_manifest, indent=2, default=str) + "\n")
    return manifests


def _print_plan(specs: list[TrialSpec], splits: dict) -> None:
    by_slice: dict[str, int] = {}
    by_split: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for s in specs:
        by_slice[s.slice] = by_slice.get(s.slice, 0) + 1
        by_split[s.split] = by_split.get(s.split, 0) + 1
        by_family[s.family_id] = by_family.get(s.family_id, 0) + 1
    print(f"[plan] {len(specs)} trials")
    for name, counts in (("slice", by_slice), ("split", by_split), ("family", by_family)):
        summary = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"[plan] {name}: {summary}")
    print(f"[plan] splits: {json.dumps(splits, indent=2, default=str)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FlyGPT corpus runner")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--graph-dir", type=Path, default=Path("data/graph"))
    ap.add_argument("--work-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--shard-prefix", default="shard")
    ap.add_argument("--max-trials", type=int, default=None)
    ap.add_argument("--plan", action="store_true", help="print the trial plan")
    ap.add_argument("--run", action="store_true", help="run shards")
    ap.add_argument("--pilot", action="store_true", help="run the Phase-1 pilot")
    args = ap.parse_args(argv)
    if not (args.plan or args.run or args.pilot):
        args.plan = True

    cfg = yaml.safe_load(args.config.read_text())
    cfg["_config_path"] = str(args.config)
    main_slice = (cfg.get("slices") or {}).get("main", {})
    cutoff = int(main_slice.get("cutoff", 5))
    bundle = load_graph_bundle(args.graph_dir, cutoff=cutoff)
    work_dir = args.work_dir or Path(cfg.get("output", {}).get("corpus_dir", "data/corpus"))

    if args.pilot:
        report = run_pilot(
            bundle, _default_library(), cfg, work_dir, args.results_dir
        )
        print(json.dumps(report["metrics"], indent=2))
        return 0

    specs = plan_corpus(cfg, bundle=bundle, max_trials=args.max_trials)
    splits = build_splits(cfg, bundle, library_cfg=_load_yaml_path(cfg.get("stim_library")))
    if args.plan:
        _print_plan(specs, splits)
    if args.run:
        manifests = run_corpus(
            args.graph_dir,
            args.config,
            work_dir=work_dir,
            backend="numpy",
            gpu_class="cpu",
            shard_prefix=args.shard_prefix,
            max_trials=args.max_trials,
            results_dir=args.results_dir,
        )
        print(f"[run] {len(manifests)} shards -> {work_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
