"""Stimulus program library + coverage tracker (T-STIM-1..5).

Implements the ``stimuli.py`` contract of ``docs/impl-spec.md`` against the
shared schema in ``src/flygpt/schema.py`` and the pre-registered thresholds
in ``EVAL.md`` (T-STIM table + section 1.8 coverage), driven by the family
definitions pinned in ``configs/stim_library_v1.yaml``:

- ``Drive`` / ``StimulusProgram``: immutable stimulus programs. ``spec()``
  is the canonical JSON-able form and ``seed()`` is always
  ``seed_from_config(spec())`` - seeds are never hand-assigned.
- ``SensoryIndex``: resolves grounding queries (type / class / subclass /
  superclass / side / neuromere / entry-nerve / type-regex / DANDI panel /
  typed) onto the ``nodes`` table -> sorted u32 ``node_idx`` arrays
  (T-STIM-1 grounding, T-STIM-2 laterality).
- ``LIBRARY``: family_id -> generator registry covering every family in the
  config. Every program records its grounding query in ``meta`` so T-STIM-1
  can re-resolve it independently against the same nodes table.
- ``naturalistic_envelope``: bounded non-negative OU multiplier in [0, 1],
  sampled at the program dt, deterministic from (program seed, envelope_id)
  (T-STIM-4).
- ``CoverageTracker``: per-manifest-type driven / robustly-driven counts
  per the EVAL 1.8 definitions (T-STIM-3, T-CORPUS-3 denominator).
- ``program_to_sim_spec``: the ``{"drives", "silences", "duration_s",
  "dt"}`` dict consumed by ``sim_backend.simulate``. Silenced populations
  go into ``silences`` so the sim drops their spikes pre-propagation and
  never records them (T-STIM-5); ``lc_silence`` additionally drives a
  reference ORN population so the perturbation has downstream context.

Usage:
    python -m flygpt.stimuli --config configs/stim_library_v1.yaml --list
    python -m flygpt.stimuli --config configs/stim_library_v1.yaml \
        --graph-dir data/graph --demo
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flygpt.schema import (
    GraphBundle,
    load_graph_bundle,
    rng_from_seed,
    seed_from_config,
)

STIM_SPEC_SCHEMA = "stim_spec_v1"
ENVELOPE_RNG_TAG = "stimuli:naturalistic_envelope:v1"

DEFAULT_DT = 0.020                        # seconds per step (pilot slice)
DEFAULT_DURATION_S = 1.0
POST_DRIVE_TAIL_S = 0.5                   # observation tail after a window
DEFAULT_WINDOW_S = (0.2, 1.0)
DEFAULT_RATE_HZ = (40.0, 100.0, 200.0)    # Shiu-protocol band
CONTEXT_REFERENCE_QUERY = {"type_regex": "^ORN_"}

# Bounded OU envelope parameters (T-STIM-4): mean-reverting to OU_MEAN,
# clipped into [0, 1]; a dimensionless multiplier on rate_hz.
OU_THETA = 5.0
OU_SIGMA = 0.6
OU_MEAN = 0.5

# EVAL 1.8 coverage thresholds.
DRIVEN_MIN_TRIALS = 1
ROBUST_MIN_SPIKES = 5
ROBUST_MIN_TRIALS = 10

# Side vocabulary. "bilateral" = both hemispheres plus unpaired midline
# cells (e.g. pharyngeal sensilla belong to a bilateral gustatory drive).
SIDE_SETS = {
    "L": frozenset({"L"}),
    "R": frozenset({"R"}),
    "M": frozenset({"M"}),
    "bilateral": frozenset({"L", "R", "M"}),
}

# Sensory families whose grounded pools feed the naturalistic family.
NATURALISTIC_SOURCE_FAMILIES = (
    "olfactory_pulse",
    "gustatory_sugar",
    "gustatory_leg",
    "visual_lc",
    "mechanosensory_jo",
    "mechanosensory_other",
    "courtship_cue",
)

_RESOLVE_KEYS = (
    "types",
    "classes",
    "subclasses",
    "superclasses",
    "sides",
    "neuromeres",
    "entry_nerves",
    "type_regex",
    "dandi_panel",
    "typed",
)

_QUERY_COLUMN = {
    "types": "type",
    "classes": "class",
    "subclasses": "subclass",
    "superclasses": "superclass",
    "neuromeres": "somaNeuromere",
    "entry_nerves": "entryNerve",
}

_NODE_COLUMNS = (
    "type",
    "instance",
    "class",
    "subclass",
    "superclass",
    "somaSide",
    "rootSide",
    "somaNeuromere",
    "entryNerve",
)


# ---------------------------------------------------------------------------
# Program dataclasses + canonical spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Drive:
    """One Poisson/envelope drive on a node set over ``window`` seconds."""

    nodes: np.ndarray                        # u32 node indices
    rate_hz: float                           # f32 drive rate
    window: tuple[float, float]              # seconds, [t0, t1)
    envelope: str = "poisson"                # poisson|naturalistic|tonic|step
    envelope_id: str = ""                    # deterministic envelope key
    # Recorded naturalistic envelope (T-STIM-4): the program carries the
    # sampled OU trace so artifacts state exactly what was driven.
    envelope_values: np.ndarray | None = None
    envelope_dt: float | None = None
    envelope_max: float | None = None

    def __post_init__(self) -> None:
        nodes = np.unique(np.asarray(self.nodes, dtype=np.uint32))
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "rate_hz", float(np.float32(self.rate_hz)))
        t0, t1 = float(self.window[0]), float(self.window[1])
        if not 0.0 <= t0 < t1:
            raise ValueError(f"drive window must satisfy 0 <= t0 < t1, got {(t0, t1)}")
        object.__setattr__(self, "window", (t0, t1))
        if self.envelope_values is not None:
            object.__setattr__(
                self, "envelope_values", np.asarray(self.envelope_values, dtype=np.float32)
            )


def _drive_dict(d: Drive) -> dict[str, Any]:
    out = {
        "nodes": [int(n) for n in d.nodes],
        "rate_hz": float(d.rate_hz),
        "window": [float(d.window[0]), float(d.window[1])],
        "envelope": d.envelope,
        "envelope_id": d.envelope_id,
    }
    if d.envelope_values is not None:
        out["envelope_values"] = [float(v) for v in d.envelope_values]
        out["envelope_dt"] = float(d.envelope_dt) if d.envelope_dt is not None else None
        if d.envelope_max is not None:
            out["envelope_max"] = float(d.envelope_max)
    return out


def _silence_dict(s: tuple[np.ndarray, tuple[float, float]]) -> dict[str, Any]:
    nodes, window = s
    t0, t1 = float(window[0]), float(window[1])
    if not 0.0 <= t0 < t1:
        raise ValueError(f"silence window must satisfy 0 <= t0 < t1, got {(t0, t1)}")
    return {
        "nodes": [int(n) for n in np.unique(np.asarray(nodes, dtype=np.uint32))],
        "window": [t0, t1],
    }


def _norm_silences(
    silences: tuple[tuple[np.ndarray, tuple[float, float]], ...],
) -> tuple[tuple[np.ndarray, tuple[float, float]], ...]:
    out = []
    for nodes, window in silences:
        t0, t1 = float(window[0]), float(window[1])
        if not 0.0 <= t0 < t1:
            raise ValueError(f"silence window must satisfy 0 <= t0 < t1, got {(t0, t1)}")
        out.append((np.unique(np.asarray(nodes, dtype=np.uint32)), (t0, t1)))
    return tuple(out)


@dataclass(frozen=True)
class StimulusProgram:
    """A fully-grounded stimulus: drives + silences + provenance meta."""

    family_id: str
    program_id: str
    drives: tuple[Drive, ...]
    silences: tuple[tuple[np.ndarray, tuple[float, float]], ...]
    duration_s: float
    dt: float
    meta: dict[str, Any]                     # JSON-able; hashed for the seed

    def spec(self) -> dict[str, Any]:
        """Canonical JSON-able spec: nodes sorted, all values plain types."""
        return {
            "schema": STIM_SPEC_SCHEMA,
            "family_id": self.family_id,
            "program_id": self.program_id,
            "duration_s": float(self.duration_s),
            "dt": float(self.dt),
            "drives": [_drive_dict(d) for d in self.drives],
            "silences": [_silence_dict(s) for s in self.silences],
            "meta": self.meta,
        }

    def seed(self) -> int:
        """hash(spec) - the ONLY sanctioned seed source (plan.md section 6)."""
        return seed_from_config(self.spec())


# ---------------------------------------------------------------------------
# SensoryIndex: grounding queries -> node_idx arrays
# ---------------------------------------------------------------------------


def _str_column(df: Any, name: str, n: int) -> np.ndarray:
    """nodes[name] as a str array; missing column or null -> ''."""
    if name not in df.columns:
        return np.full(n, "", dtype=object)
    col = df[name].to_numpy()
    return np.where(col == None, "", col).astype(str)  # noqa: E711


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _side_set(sides: Any) -> frozenset[str]:
    labels = [sides] if isinstance(sides, str) else list(sides)
    out: set[str] = set()
    for lab in labels:
        if lab not in SIDE_SETS:
            raise ValueError(f"unknown side label {lab!r}; expected one of {sorted(SIDE_SETS)}")
        out |= SIDE_SETS[lab]
    return frozenset(out)


class SensoryIndex:
    """Resolves grounding queries -> node_idx arrays via the nodes table.

    Criteria AND together. ``sides`` uses ``somaSide`` falling back to
    ``rootSide``; ``"bilateral"`` matches L, R and midline M. ``dandi_panel``
    takes a driver name, a list of driver names, or ``True`` (the whole
    23-driver DANDI LC panel from the stim config). ``typed=True`` selects
    every non-null-typed node. ``resolve()`` with no criteria returns all
    node indices (the whole-network convention used by tonic sweeps).
    """

    def __init__(self, bundle: GraphBundle, stim_cfg: dict[str, Any] | None = None):
        self.bundle = bundle
        self.stim_cfg = stim_cfg
        self.n_nodes = int(bundle.n_nodes)
        self.dandi_panel = dict((stim_cfg or {}).get("dandi_lc_panel") or {})
        self.config_version = (stim_cfg or {}).get("version", "unversioned")
        self.config_sha256 = (stim_cfg or {}).get("_config_sha256")
        self._cols = {
            name: _str_column(bundle.nodes, name, self.n_nodes) for name in _NODE_COLUMNS
        }
        self.type_of = self._cols["type"]
        soma, root = self._cols["somaSide"], self._cols["rootSide"]
        self.effective_side = np.where(soma == "", root, soma)
        self.manifest_types = list(bundle.manifest_types)

    def _col(self, name: str) -> np.ndarray:
        return self._cols[name]

    def _panel_types(self, sel: Any) -> list[str]:
        if not self.dandi_panel:
            raise ValueError(
                "dandi_panel resolution requires a SensoryIndex built with the stim config"
            )
        if sel is True:
            return sorted({t for types in self.dandi_panel.values() for t in types})
        drivers = [sel] if isinstance(sel, str) else list(sel)
        unknown = [d for d in drivers if d not in self.dandi_panel]
        if unknown:
            raise ValueError(
                f"unknown DANDI driver(s) {unknown}; panel has {sorted(self.dandi_panel)}"
            )
        return sorted({t for d in drivers for t in self.dandi_panel[d]})

    def resolve(
        self,
        *,
        types: Any = None,
        classes: Any = None,
        subclasses: Any = None,
        superclasses: Any = None,
        sides: Any = None,
        neuromeres: Any = None,
        entry_nerves: Any = None,
        type_regex: str | None = None,
        dandi_panel: Any = None,
        typed: bool | None = None,
    ) -> np.ndarray:
        """AND-combined grounding query -> sorted u32 node_idx array."""
        mask = np.ones(self.n_nodes, dtype=np.bool_)
        for key, value in (
            ("types", types),
            ("classes", classes),
            ("subclasses", subclasses),
            ("superclasses", superclasses),
            ("neuromeres", neuromeres),
            ("entry_nerves", entry_nerves),
        ):
            if value is not None:
                mask &= np.isin(self._col(_QUERY_COLUMN[key]), _as_str_list(value))
        if type_regex is not None:
            rx = re.compile(type_regex)
            type_col = self._col("type")
            mask &= np.array([rx.match(t) is not None for t in type_col], dtype=np.bool_)
        if sides is not None:
            mask &= np.isin(self.effective_side, list(_side_set(sides)))
        if dandi_panel is not None:
            mask &= np.isin(self._col("type"), self._panel_types(dandi_panel))
        if typed is not None:
            is_typed = self._col("type") != ""
            mask &= is_typed if typed else ~is_typed
        return np.flatnonzero(mask).astype(np.uint32)

    def types_present(self, nodes: np.ndarray) -> list[str]:
        """Sorted distinct non-null types among ``nodes``."""
        return sorted({self.type_of[int(n)] for n in nodes if self.type_of[int(n)]})


# ---------------------------------------------------------------------------
# Naturalistic envelope (T-STIM-4)
# ---------------------------------------------------------------------------


def naturalistic_envelope(
    seed: int,
    envelope_id: str,
    n_steps: int,
    dt: float,
    theta: float = OU_THETA,
    sigma: float = OU_SIGMA,
) -> np.ndarray:
    """Bounded non-negative OU rate multiplier in [0, 1], one sample per step.

    ``x += theta*(OU_MEAN - x)*dt + sigma*sqrt(dt)*N(0,1)``, clipped to
    [0, 1]. Deterministic from ``(seed, envelope_id)`` via the shared
    ``rng_from_seed`` rule; same inputs reproduce the trace bitwise.
    """
    rng = rng_from_seed(ENVELOPE_RNG_TAG, int(seed), str(envelope_id))
    n = int(n_steps)
    out = np.empty(n, dtype=np.float32)
    decay = float(theta) * float(dt)
    scale = float(sigma) * float(np.sqrt(float(dt)))
    x = OU_MEAN
    for i in range(n):
        x += decay * (OU_MEAN - x) + scale * float(rng.standard_normal())
        out[i] = min(1.0, max(0.0, x))
    return out


# ---------------------------------------------------------------------------
# Family generators
# ---------------------------------------------------------------------------


def _choose(rng: np.random.Generator, seq: Any) -> Any:
    return seq[int(rng.integers(0, len(seq)))]


def _pick_side(rng: np.random.Generator) -> str:
    return "L" if rng.random() < 0.5 else "R"


def _window(cfg: dict[str, Any]) -> tuple[float, float]:
    w = cfg.get("window_s", DEFAULT_WINDOW_S)
    return (float(w[0]), float(w[1]))


def _duration(cfg: dict[str, Any], window: tuple[float, float]) -> float:
    return float(cfg.get("duration_s", window[1] + POST_DRIVE_TAIL_S))


def _dt(cfg: dict[str, Any]) -> float:
    return float(cfg.get("dt", DEFAULT_DT))


def _trial(kw: dict[str, Any]) -> int:
    return int(kw.get("trial", 0))


def _resolve_kwargs(query: dict[str, Any]) -> dict[str, Any]:
    return {k: query[k] for k in _RESOLVE_KEYS if k in query}


def _entry_queries(entry: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    if "query" in entry:
        out.append(dict(entry["query"]))
    out.extend(dict(q) for q in entry.get("query_multi", []))
    return out


def _sensory_drive(
    index: SensoryIndex,
    rng: np.random.Generator,
    query: dict[str, Any],
    side_policy: str,
    *,
    side: str | None = None,
    pick_type: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Ground one drive: resolve the query, optionally narrow to one type.

    Returns (nodes, meta_entry) where meta_entry records the broad query
    and the exact ``resolve_kwargs`` used, so T-STIM-1 can re-resolve the
    drive independently.
    """
    if side is None:
        side = _pick_side(rng) if side_policy == "lateralized" else "bilateral"
    meta: dict[str, Any] = {
        "query": dict(query),
        "side_policy": side_policy,
        "side": side,
        "chosen_type": None,
    }
    base = _resolve_kwargs(query)
    if side_policy in ("lateralized", "bilateral"):
        base["sides"] = side
    if pick_type:
        # Pick among types present on the requested side so a sided pick can
        # never yield an empty target set when the sided pool is non-empty.
        candidates = index.types_present(index.resolve(**base))
        chosen = _choose(rng, candidates) if candidates else None
        kwargs: dict[str, Any] = {"types": [chosen] if chosen else []}
        if "sides" in base:
            kwargs["sides"] = base["sides"]
        meta["chosen_type"] = chosen
    else:
        kwargs = base
    nodes = index.resolve(**kwargs)
    meta["resolve_kwargs"] = kwargs
    return nodes, meta


def _build_program(
    index: SensoryIndex,
    family_id: str,
    drives: tuple[Drive, ...],
    silences: tuple[tuple[np.ndarray, tuple[float, float]], ...],
    duration_s: float,
    dt: float,
    meta: dict[str, Any],
) -> StimulusProgram:
    """Assemble a StimulusProgram; program_id is a content-derived tag."""
    full_meta = {
        "config_version": index.config_version,
        "config_sha256": index.config_sha256,
        **meta,
    }
    norm_silences = _norm_silences(silences)
    body = {
        "family_id": family_id,
        "drives": [_drive_dict(d) for d in drives],
        "silences": [_silence_dict(s) for s in norm_silences],
        "duration_s": float(duration_s),
        "dt": float(dt),
        "meta": full_meta,
    }
    program_id = f"{family_id}_{seed_from_config(body) % (16**8):08x}"
    return StimulusProgram(
        family_id=family_id,
        program_id=program_id,
        drives=tuple(drives),
        silences=norm_silences,
        duration_s=float(duration_s),
        dt=float(dt),
        meta=full_meta,
    )


def _pulse_program(
    family_id: str,
    index: SensoryIndex,
    rng: np.random.Generator,
    cfg: dict[str, Any],
    *,
    pick_type: bool,
    **kw: Any,
) -> StimulusProgram:
    """Single-drive sensory pulse shared by the olfactory/gustatory/LC/
    mechanosensory families: resolve the config query, pick a rate and a
    window from the config lists, lateralize per the side policy."""
    trial = _trial(kw)
    side_policy = cfg.get("side_policy", "bilateral")
    side = kw.get("side") or (_pick_side(rng) if side_policy == "lateralized" else "bilateral")
    nodes, dmeta = _sensory_drive(
        index, rng, dict(cfg.get("query", {})), side_policy, side=side, pick_type=pick_type
    )
    window = _window(cfg)
    rate = _choose(rng, cfg.get("rate_hz", DEFAULT_RATE_HZ))
    drives = (Drive(nodes=nodes, rate_hz=rate, window=window),)
    meta = {"trial": trial, "drives": [dmeta], "silences": []}
    return _build_program(index, family_id, drives, (), _duration(cfg, window), _dt(cfg), meta)


def _gen_olfactory_pulse(index, rng, cfg, **kw):
    return _pulse_program("olfactory_pulse", index, rng, cfg, pick_type=True, **kw)


def _gen_gustatory_sugar(index, rng, cfg, **kw):
    return _pulse_program("gustatory_sugar", index, rng, cfg, pick_type=False, **kw)


def _gen_gustatory_leg(index, rng, cfg, **kw):
    return _pulse_program("gustatory_leg", index, rng, cfg, pick_type=False, **kw)


def _gen_visual_lc(index, rng, cfg, **kw):
    return _pulse_program("visual_lc", index, rng, cfg, pick_type=True, **kw)


def _gen_mechanosensory_jo(index, rng, cfg, **kw):
    return _pulse_program("mechanosensory_jo", index, rng, cfg, pick_type=False, **kw)


def _gen_mechanosensory_other(index, rng, cfg, **kw):
    return _pulse_program("mechanosensory_other", index, rng, cfg, pick_type=False, **kw)


def _gen_courtship(index, rng, cfg, **kw) -> StimulusProgram:
    """Composite courtship cue: one visual LC type + one pheromone ORN type."""
    trial = _trial(kw)
    side_policy = cfg.get("side_policy", "lateralized")
    side = kw.get("side") or (_pick_side(rng) if side_policy == "lateralized" else "bilateral")
    window = _window(cfg)
    drives, dmetas = [], []
    for q in cfg.get("query_multi", []):
        nodes, m = _sensory_drive(
            index, rng, dict(q), side_policy, side=side, pick_type=True
        )
        rate = _choose(rng, cfg.get("rate_hz", DEFAULT_RATE_HZ))
        drives.append(Drive(nodes=nodes, rate_hz=rate, window=window))
        dmetas.append(m)
    meta = {"trial": trial, "drives": dmetas, "silences": []}
    return _build_program(
        index, "courtship_cue", tuple(drives), (), _duration(cfg, window), _dt(cfg), meta
    )


def _gen_lc_silence(index, rng, cfg, **kw) -> StimulusProgram:
    """DANDI-matched LC silencing: silence one driver's type set while a
    reference ORN drive provides downstream context (T-STIM-5)."""
    trial = _trial(kw)
    if not index.dandi_panel:
        raise ValueError("lc_silence requires dandi_lc_panel in the stim config")
    drivers = sorted(index.dandi_panel)
    driver = kw.get("variant") or kw.get("driver") or _choose(rng, drivers)
    if driver not in index.dandi_panel:
        raise ValueError(f"unknown DANDI driver {driver!r}; panel has {drivers}")
    s_kwargs = {"dandi_panel": driver}
    silenced = index.resolve(**s_kwargs)
    sw = cfg.get("silence_window_s", DEFAULT_WINDOW_S)
    window = (float(sw[0]), float(sw[1]))
    side = kw.get("side") or _pick_side(rng)
    ref_nodes, ref_meta = _sensory_drive(
        index, rng, dict(CONTEXT_REFERENCE_QUERY), "lateralized", side=side, pick_type=True
    )
    rate = _choose(rng, cfg.get("rate_hz", DEFAULT_RATE_HZ))
    drives = (Drive(nodes=ref_nodes, rate_hz=rate, window=window),)
    meta = {
        "trial": trial,
        "driver": driver,
        "panel_types": list(index.dandi_panel[driver]),
        "context": "reference_orn_drive",
        "drives": [ref_meta],
        "silences": [
            {
                "resolve_kwargs": s_kwargs,
                "driver": driver,
                "panel_types": list(index.dandi_panel[driver]),
            }
        ],
    }
    return _build_program(
        index, "lc_silence", drives, ((silenced, window),), _duration(cfg, window), _dt(cfg), meta
    )


def _gen_optogenetic(index, rng, cfg, **kw) -> StimulusProgram:
    """Arbitrary type activation/silencing over the typed universe."""
    trial = _trial(kw)
    query = dict(cfg.get("query", {"typed": True}))
    candidates = index.types_present(index.resolve(**_resolve_kwargs(query)))
    chosen = _choose(rng, candidates) if candidates else None
    mode = kw.get("mode") or ("silence" if rng.random() < 0.5 else "activate")
    kwargs = {"types": [chosen] if chosen else []}
    nodes = index.resolve(**kwargs)
    window = _window(cfg)
    entry = {
        "resolve_kwargs": kwargs,
        "chosen_type": chosen,
        "mode": mode,
        "query": query,
    }
    if mode == "silence":
        drives: tuple[Drive, ...] = ()
        silences: tuple = ((nodes, window),)
        dmeta, smeta = [], [entry]
    else:
        rate = _choose(rng, cfg.get("rate_hz", DEFAULT_RATE_HZ))
        drives = (Drive(nodes=nodes, rate_hz=rate, window=window),)
        silences = ()
        dmeta, smeta = [entry], []
    meta = {"trial": trial, "mode": mode, "drives": dmeta, "silences": smeta}
    return _build_program(
        index, "optogenetic", drives, silences, _duration(cfg, window), _dt(cfg), meta
    )


def _gen_tonic_baseline(index, rng, cfg, **kw) -> StimulusProgram:
    """Whole-network tonic Poisson sweep (untyped nodes included)."""
    trial = _trial(kw)
    rate = float(_choose(rng, cfg.get("tonic_rate_hz", [1.0])))
    duration = float(cfg.get("duration_s", DEFAULT_DURATION_S))
    nodes = np.arange(index.n_nodes, dtype=np.uint32)
    drives = (Drive(nodes=nodes, rate_hz=rate, window=(0.0, duration), envelope="tonic"),)
    meta = {
        "trial": trial,
        "tonic_rate_hz": rate,
        "drives": [{"resolve_kwargs": {}, "scope": "all_nodes", "envelope": "tonic"}],
        "silences": [],
    }
    return _build_program(index, "tonic_baseline", drives, (), duration, _dt(cfg), meta)


def _naturalistic_pool(index: SensoryIndex) -> np.ndarray:
    """Union of nodes resolved by the grounded sensory families."""
    fams = (index.stim_cfg or {}).get("families", {})
    parts = []
    for fid in NATURALISTIC_SOURCE_FAMILIES:
        entry = fams.get(fid)
        if not entry:
            continue
        for q in _entry_queries(entry):
            parts.append(index.resolve(**_resolve_kwargs(q)))
    if not parts:
        return index.resolve(typed=True)
    return np.unique(np.concatenate(parts))


def _gen_naturalistic(index, rng, cfg, **kw) -> StimulusProgram:
    """OU-envelope drive on one type drawn from the sensory-family union."""
    trial = _trial(kw)
    candidates = index.types_present(_naturalistic_pool(index))
    chosen = _choose(rng, candidates) if candidates else None
    kwargs = {"types": [chosen] if chosen else []}
    nodes = index.resolve(**kwargs)
    window = _window(cfg)
    rate = _choose(rng, cfg.get("rate_hz", DEFAULT_RATE_HZ))
    envelope_id = f"nat_{chosen or 'none'}_t{trial:04d}"
    dt = _dt(cfg)
    # Record the sampled OU trace in the program (T-STIM-4): the drive
    # carries exactly what the simulator will play back - auditable, and
    # the sim needn't re-derive it from the trial seed.
    env_steps = int(round((window[1] - window[0]) / dt))
    env_vals = naturalistic_envelope(
        int(seed_from_config({"env": envelope_id})),
        envelope_id,
        env_steps,
        dt,
    )
    drives = (
        Drive(
            nodes=nodes,
            rate_hz=rate,
            window=window,
            envelope="naturalistic",
            envelope_id=envelope_id,
            envelope_values=env_vals,
            envelope_dt=dt,
            envelope_max=1.0,
        ),
    )
    meta = {
        "trial": trial,
        "pool": "sensory_family_union",
        "pool_n_types": len(candidates),
        "drives": [
            {
                "resolve_kwargs": kwargs,
                "chosen_type": chosen,
                "envelope": "naturalistic",
                "envelope_id": envelope_id,
                "envelope_values": [float(v) for v in env_vals],
                "envelope_dt": dt,
                "envelope_max": 1.0,
            }
        ],
        "silences": [],
    }
    return _build_program(
        index, "naturalistic", drives, (), _duration(cfg, window), _dt(cfg), meta
    )


def _gen_babble(index, rng, cfg, **kw) -> StimulusProgram:
    """Random seeded neuron subset (babble control - not query-grounded)."""
    trial = _trial(kw)
    k = min(int(cfg.get("neurons_per_program", 500)), index.n_nodes)
    sel = np.sort(rng.choice(index.n_nodes, size=k, replace=False)).astype(np.uint32)
    window = _window(cfg)
    rate = _choose(rng, cfg.get("rate_hz", DEFAULT_RATE_HZ))
    drives = (Drive(nodes=sel, rate_hz=rate, window=window),)
    meta = {
        "trial": trial,
        "grounded": False,
        "neurons_per_program": k,
        "drives": [
            {"resolve_kwargs": None, "sampling": "uniform_without_replacement"}
        ],
        "silences": [],
    }
    return _build_program(index, "babble", drives, (), _duration(cfg, window), _dt(cfg), meta)


def _gen_silence_test(index, rng, cfg, **kw) -> StimulusProgram:
    """Unstimulated control program (T-SIM-5 harness): no drives, no silences."""
    trial = _trial(kw)
    duration = float(cfg.get("duration_s", DEFAULT_DURATION_S))
    meta = {"trial": trial, "unstimulated": True, "drives": [], "silences": []}
    return _build_program(index, "silence_test", (), (), duration, _dt(cfg), meta)


LIBRARY: dict[str, Any] = {
    "olfactory_pulse": _gen_olfactory_pulse,
    "gustatory_sugar": _gen_gustatory_sugar,
    "gustatory_leg": _gen_gustatory_leg,
    "visual_lc": _gen_visual_lc,
    "mechanosensory_jo": _gen_mechanosensory_jo,
    "mechanosensory_other": _gen_mechanosensory_other,
    "courtship_cue": _gen_courtship,
    "lc_silence": _gen_lc_silence,
    "optogenetic": _gen_optogenetic,
    "tonic_baseline": _gen_tonic_baseline,
    "naturalistic": _gen_naturalistic,
    "babble": _gen_babble,
    "silence_test": _gen_silence_test,
}


# ---------------------------------------------------------------------------
# Config + dispatch
# ---------------------------------------------------------------------------


def load_stim_config(path: Path) -> dict[str, Any]:
    """Read the stim-library YAML; records path + sha256 for provenance."""
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "families" not in cfg:
        raise KeyError(f"{path} has no 'families' block")
    return cfg


def build_library(stim_cfg: dict[str, Any]) -> dict[str, Any]:
    """family_id -> generator for every family present in the config."""
    fams = stim_cfg["families"]
    missing = [fid for fid in fams if fid not in LIBRARY]
    if missing:
        raise KeyError(f"stim config families without generators: {missing}")
    return {fid: LIBRARY[fid] for fid in fams}


def generate(
    family_id: str,
    index: SensoryIndex,
    rng: np.random.Generator,
    stim_cfg: dict[str, Any],
    **kw: Any,
) -> StimulusProgram:
    """Dispatch one program: LIBRARY[family_id](index, rng, family_cfg)."""
    fams = stim_cfg.get("families", {})
    if family_id not in fams:
        raise KeyError(f"unknown stimulus family {family_id!r}; config has {sorted(fams)}")
    if family_id not in LIBRARY:
        raise KeyError(f"stimulus family {family_id!r} has no generator in LIBRARY")
    return LIBRARY[family_id](index, rng, fams[family_id], **kw)


# ---------------------------------------------------------------------------
# CoverageTracker (T-STIM-3, EVAL 1.8)
# ---------------------------------------------------------------------------


class CoverageTracker:
    """Accumulates per-manifest-type driven counts across trials.

    A type is *driven* iff it emits >= 1 spike in >= 1 trial, *robustly
    driven* iff it emits >= 5 spikes across >= 10 distinct trials
    (EVAL 1.8). ``fired_nodes`` is one trial's per-spike node array (repeats
    count); a TrialResult-like object with ``events_node`` also works.
    """

    def __init__(self, bundle: GraphBundle):
        self._types = list(bundle.manifest_types)
        self._type_of = bundle.type_array()
        self._stats = {t: {"spikes": 0, "trials": 0} for t in self._types}
        self.n_trials = 0
        self.program_ids: set[str] = set()

    def record(self, program: StimulusProgram | None, fired_nodes: Any) -> None:
        nodes = getattr(fired_nodes, "events_node", fired_nodes)
        nodes = np.asarray(nodes, dtype=np.int64).ravel()
        nodes = nodes[(nodes >= 0) & (nodes < self._type_of.size)]
        self.n_trials += 1
        if program is not None:
            self.program_ids.add(str(getattr(program, "program_id", program)))
        if nodes.size == 0:
            return
        spikes = np.bincount(nodes, minlength=self._type_of.size)
        for i in np.flatnonzero(spikes):
            t = self._type_of[i]
            if not t or t not in self._stats:
                continue
            self._stats[t]["spikes"] += int(spikes[i])
            self._stats[t]["trials"] += 1

    def report(self) -> dict[str, Any]:
        driven = sorted(t for t, s in self._stats.items() if s["trials"] >= DRIVEN_MIN_TRIALS)
        robust = sorted(
            t
            for t, s in self._stats.items()
            if s["spikes"] >= ROBUST_MIN_SPIKES and s["trials"] >= ROBUST_MIN_TRIALS
        )
        total = len(self._types)
        return {
            "driven": driven,
            "robustly_driven": robust,
            "driven_fraction": (len(driven) / total) if total else 0.0,
            "robust_fraction": (len(robust) / total) if total else 0.0,
            "total_types": total,
        }


# ---------------------------------------------------------------------------
# Sim-facing spec
# ---------------------------------------------------------------------------


def program_to_sim_spec(program: StimulusProgram) -> dict[str, Any]:
    """Program -> the dict consumed by ``sim_backend.simulate``.

    Shape: ``{"drives": [...], "silences": [...], "duration_s", "dt"}`` plus
    provenance extras. Naturalistic drives carry a precomputed
    ``envelope_trace`` (f32, one multiplier per step, T-STIM-4).
    """
    n_steps = int(round(program.duration_s / program.dt))
    drives = []
    for d in program.drives:
        entry: dict[str, Any] = {
            "nodes": np.asarray(d.nodes, dtype=np.uint32),
            "rate_hz": float(d.rate_hz),
            "window": (float(d.window[0]), float(d.window[1])),
            "envelope": d.envelope,
            "envelope_id": d.envelope_id,
        }
        if d.envelope == "naturalistic":
            entry["envelope_trace"] = naturalistic_envelope(
                program.seed(), d.envelope_id, n_steps, program.dt
            )
        drives.append(entry)
    silences = [
        {
            "nodes": np.unique(np.asarray(nodes, dtype=np.uint32)),
            "window": (float(w[0]), float(w[1])),
        }
        for nodes, w in program.silences
    ]
    spec: dict[str, Any] = {
        "drives": drives,
        "silences": silences,
        "duration_s": float(program.duration_s),
        "dt": float(program.dt),
        "n_steps": n_steps,
        "family_id": program.family_id,
        "program_id": program.program_id,
        "seed": program.seed(),
    }
    # Provenance: corpus shards record this spec in trials.program; the
    # grounding queries live in meta so T-STIM-1 can re-resolve them.
    if program.meta:
        spec["meta"] = program.meta
    return spec


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/stim_library_v1.yaml"))
    ap.add_argument("--graph-dir", type=Path, default=None)
    ap.add_argument("--cutoff", type=int, default=5)
    ap.add_argument("--list", action="store_true", help="print family ids and exit")
    ap.add_argument(
        "--demo", action="store_true", help="resolve one program per family on the graph"
    )
    args = ap.parse_args(argv)

    cfg = load_stim_config(args.config)
    families = list(cfg["families"])
    if args.list or not args.demo:
        for fid in families:
            print(fid)
    if not args.demo:
        return 0

    if args.graph_dir is None:
        ap.error("--demo requires --graph-dir")
    bundle = load_graph_bundle(args.graph_dir, cutoff=args.cutoff)
    index = SensoryIndex(bundle, cfg)
    print(f"[demo] {bundle.n_nodes} nodes, {len(index.manifest_types)} manifest types")
    for fid in families:
        rng = rng_from_seed("stimuli-demo", fid)
        program = generate(fid, index, rng, cfg)
        n_drive = int(sum(d.nodes.size for d in program.drives))
        n_sil = int(sum(s[0].size for s in program.silences))
        print(
            f"{fid:24s} {program.program_id} drives={len(program.drives)} "
            f"drive_nodes={n_drive} silenced_nodes={n_sil}"
        )
    panel = cfg.get("dandi_lc_panel", {})
    if panel:
        print(f"[demo] dandi_lc_panel: {len(panel)} drivers")
        for driver in sorted(panel):
            rng = rng_from_seed("stimuli-demo", "lc_silence", driver)
            program = generate("lc_silence", index, rng, cfg, variant=driver)
            n_sil = int(sum(s[0].size for s in program.silences))
            print(f"  {driver:10s} silenced_nodes={n_sil} types={panel[driver]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
