"""DN/MN -> behavior head; song decoder (C19, T-REAL-3).

The readout populations and decoding rules for every behavior head are
fixed in ``configs/decoder_spec_v1.yaml`` - THE written decoder spec,
committed before any behavior-head training (plan.md Phase 3 hard
requirement; docs/impl-spec.md "behavior.py" section). Wing MNs decode
song, leg MNs decode locomotion alternation, DNp01 decodes the escape
command, etc. This module resolves each spec head against the nodes
table of a ``GraphBundle`` and decodes event lists into bout rows.

Schemas
-------
Spec heads carry::

    readout: {superclasses, classes, subclasses, types, neuromeres,
              sides, entry_nerves, exit_nerves, type_regex}
    decode:  {rule, dt?, window_s?, min_bout_s?, bout_gap_s?,
              min_rate_hz?, min_events?, min_switches?}
    sex:     male_only | female_only | both

Readout fields AND across keys and OR within a key's value list; the
column mapping is ``types->type``, ``classes->class``,
``subclasses->subclass``, ``superclasses->superclass``,
``neuromeres->somaNeuromere``, ``sides->somaSide``,
``entry_nerves->entryNerve``, ``exit_nerves->exitNerve`` (the same node
columns the stimulus grounding queries use, per stim_library_v1.yaml).

Decode output rows::

    {behavior, t0_s, t1_s,
     evidence: {nodes: [node_idx...], rate_hz, n_events},
     masked: bool, masked_reason: str|None}

Sex masking (T-SEX-4 shared rule): a ``male_only`` head emits a masked
row under ``sex="female"`` and vice versa. Heads resolving to zero
nodes are reported in ``coverage_report()`` and also emit a masked row
under decode - coverage is recorded, never silently empty.

``SongDecoder`` wraps the two ``song_*`` heads at the courtship-slice
dt=5ms: pulse events are segmented per MN by the spec ``bout_gap_s``
(10ms), pulses closer than ``bout_merge_ipi_s`` merge into pulse-mode
bouts, and IPI is the onset-to-onset interval between pulses.
``compare_ethogram`` scores the ethogram statistics against
``evals/song_reference.json`` (SongExplorer ground-truth ranges,
EVAL.md T-REAL-3: >=4/5 statistics inside range, KS reported per stat).

Usage:
    python -m flygpt.behavior --graph-dir data/graph --resolve
    python -m flygpt.behavior --graph-dir data/graph --demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flygpt.schema import GraphBundle, load_graph_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = REPO_ROOT / "configs" / "decoder_spec_v1.yaml"
DEFAULT_REFERENCE_PATH = REPO_ROOT / "evals" / "song_reference.json"

SONG_DT_S = 0.005  # courtship-slice resolution (EVAL.md 1.1)

# Spec readout keys -> nodes.parquet columns. The same field set the
# SensoryIndex grounding queries use (configs/stim_library_v1.yaml).
QUERY_COLUMNS = {
    "types": "type",
    "classes": "class",
    "subclasses": "subclass",
    "superclasses": "superclass",
    "neuromeres": "somaNeuromere",
    "sides": "somaSide",
    "root_sides": "rootSide",
    "entry_nerves": "entryNerve",
    "exit_nerves": "exitNerve",
}
REGEX_FIELDS = {"type_regex": "type"}

SEX_VALUES = ("male", "female")
SEX_MASK = {"male_only": "female", "female_only": "male"}


def load_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the decoder spec yaml; defaults to configs/decoder_spec_v1.yaml."""
    path = path or DEFAULT_SPEC_PATH
    return yaml.safe_load(Path(path).read_text())


def load_song_reference(path: Path | None = None) -> dict[str, Any]:
    """Load evals/song_reference.json (SongExplorer frozen ranges)."""
    import json

    path = path or DEFAULT_REFERENCE_PATH
    return json.loads(Path(path).read_text())


def _column_or_none(nodes, name: str) -> np.ndarray | None:
    if name not in nodes.columns:
        return None
    col = nodes[name].to_numpy()
    # normalize nulls to '' so isin/matching treats them as non-matching
    return np.where(col == None, "", col).astype(str)  # noqa: E711


def _as_step_node_arrays(
    events_step: Any, events_node: Any, n_steps: int | None, dt: float | None
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Accept either raw arrays or a TrialResult-like object.

    ``decode``/``ethogram`` are pinned on ``(events_step, events_node,
    n_steps, dt)`` but sim trials arrive as ``TrialResult`` objects with
    matching attribute names - unwrap them so callers can pass either.
    """
    if hasattr(events_step, "events_step") and hasattr(events_step, "events_node"):
        result = events_step
        events_step = result.events_step
        events_node = result.events_node
        n_steps = result.n_steps if n_steps is None else n_steps
        dt = float(result.config.dt) if dt is None else dt
    steps = np.asarray(events_step, dtype=np.int64)
    nodes = np.asarray(events_node, dtype=np.int64)
    if n_steps is None or dt is None:
        raise ValueError("n_steps and dt are required when passing raw event arrays")
    return steps, nodes, int(n_steps), float(dt)


def _node_event_steps(events_step: np.ndarray, events_node: np.ndarray) -> dict[int, np.ndarray]:
    """node_idx -> sorted unique event steps (single pass, stable sort)."""
    order = np.lexsort((events_step, events_node))
    node_sorted = events_node[order]
    step_sorted = events_step[order]
    bounds = np.flatnonzero(np.r_[True, node_sorted[1:] != node_sorted[:-1]])
    out: dict[int, np.ndarray] = {}
    starts = np.r_[bounds, node_sorted.size]
    for i in range(bounds.size):
        seg = step_sorted[starts[i]: starts[i + 1]]
        out[int(node_sorted[bounds[i]])] = np.unique(seg)
    return out


def _population_counts(
    events_step: np.ndarray, events_node: np.ndarray, nodes: np.ndarray, n_steps: int
) -> np.ndarray:
    """Per-step event count restricted to the readout population."""
    counts = np.zeros(n_steps, dtype=np.float64)
    if events_step.size == 0 or nodes.size == 0:
        return counts
    sel = np.isin(events_node, nodes)
    np.add.at(counts, events_step[sel], 1.0)
    return counts


def _windowed_rate(counts: np.ndarray, window_steps: int, dt: float) -> np.ndarray:
    """Centered-boxcar rate in Hz; edge windows normalize by covered time."""
    n = counts.size
    if n == 0:
        return counts
    w = max(1, int(window_steps))
    half = w // 2
    csum = np.r_[0.0, np.cumsum(counts)]
    idx = np.arange(n)
    lo = np.clip(idx - half, 0, n)
    hi = np.clip(idx - half + w, 0, n)
    covered = hi - lo
    return (csum[hi] - csum[lo]) / np.clip(covered * dt, 1e-12, None)


def _steps_to_bout_rows(
    behavior: str,
    runs: list[tuple[int, int, np.ndarray]],
    events_step: np.ndarray,
    events_node: np.ndarray,
    dt: float,
    min_bout_s: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert (first_step, last_step, contributing_nodes) runs to rows."""
    rows = []
    for first, last, run_nodes in sorted(runs, key=lambda r: r[0]):
        t0 = first * dt
        t1 = (last + 1) * dt
        if t1 - t0 < min_bout_s - 1e-12:
            continue
        mask = (events_step >= first) & (events_step <= last)
        if run_nodes.size:
            mask &= np.isin(events_node, run_nodes)
        n_events = int(mask.sum())
        rows.append(
            {
                "behavior": behavior,
                "t0_s": t0,
                "t1_s": t1,
                "evidence": {
                    "nodes": sorted(int(n) for n in run_nodes),
                    "rate_hz": n_events / max(t1 - t0, dt),
                    "n_events": n_events,
                },
                "masked": False,
                "masked_reason": None,
            }
        )
    return rows


def _merge_overlapping(
    runs: list[tuple[int, int, int]],
) -> list[tuple[int, int, np.ndarray]]:
    """Merge overlapping/adjacent (first, last, node) runs; union nodes."""
    if not runs:
        return []
    runs = sorted(runs)
    merged: list[list[Any]] = [[runs[0][0], runs[0][1], {runs[0][2]}]]
    for first, last, node in runs[1:]:
        cur = merged[-1]
        if first <= cur[1] + 1:
            cur[1] = max(cur[1], last)
            cur[2].add(node)
        else:
            merged.append([first, last, {node}])
    return [(a, b, np.fromiter(sorted(s), dtype=np.int64)) for a, b, s in merged]


class BehaviorDecoder:
    """Resolves decoder-spec heads to node populations and decodes events.

    ``spec_cfg`` is the parsed ``decoder_spec_v1.yaml`` dict (or a Path
    to it, or None for the default config). Resolution coverage per head
    is recorded at construction: ``coverage_report()`` lists which heads
    resolved to zero nodes so an empty readout is always reported.
    """

    def __init__(self, bundle: GraphBundle, spec_cfg: dict | Path | None = None):
        if spec_cfg is None:
            spec_cfg = load_spec()
        elif isinstance(spec_cfg, (str, Path)):
            spec_cfg = load_spec(Path(spec_cfg))
        self.bundle = bundle
        self.spec = spec_cfg
        self.heads: dict[str, dict[str, Any]] = dict(spec_cfg.get("heads", {}))
        self.spec_version = spec_cfg.get("version", "decoder_v1")
        self.resolution: dict[str, dict[str, Any]] = {
            name: self._resolve_head(name, head) for name, head in self.heads.items()
        }

    # ------------------------------------------------------------------
    # Readout resolution
    # ------------------------------------------------------------------

    def _resolve_head(self, name: str, head: dict[str, Any]) -> dict[str, Any]:
        readout = head.get("readout", {}) or {}
        table = self.bundle.nodes
        mask = np.ones(self.bundle.n_nodes, dtype=bool)
        missing: list[str] = []
        applied: dict[str, Any] = {}
        for key, colname in QUERY_COLUMNS.items():
            if key not in readout:
                continue
            values = [str(v) for v in readout[key]]
            applied[key] = values
            col = _column_or_none(table, colname)
            if col is None:
                missing.append(colname)
                mask &= False
                continue
            mask &= np.isin(col, values)
        for key, colname in REGEX_FIELDS.items():
            if key not in readout:
                continue
            pattern = str(readout[key])
            applied[key] = pattern
            col = _column_or_none(table, colname)
            if col is None:
                missing.append(colname)
                mask &= False
                continue
            import re

            rx = re.compile(pattern)
            mask &= np.array([bool(rx.search(v)) for v in col])
        node_idx = np.flatnonzero(mask).astype(np.uint32)
        return {
            "node_idx": node_idx,
            "n_nodes": int(node_idx.size),
            "resolved": bool(node_idx.size),
            "query": applied,
            "missing_columns": missing,
            "rule": (head.get("decode", {}) or {}).get("rule", ""),
            "sex": head.get("sex", "both"),
            "decode": dict(head.get("decode", {}) or {}),
            "description": head.get("description", ""),
        }

    def readout_map(self) -> dict[str, list[int]]:
        """head -> resolved node_idx list (tokenize/vocab + head wiring)."""
        return {
            name: [int(i) for i in res["node_idx"]]
            for name, res in self.resolution.items()
        }

    def coverage_report(self) -> dict[str, Any]:
        """Per-head resolution coverage; zero-node heads are reported."""
        per_head = {
            name: {
                "n_nodes": res["n_nodes"],
                "resolved": res["resolved"],
                "rule": res["rule"],
                "sex": res["sex"],
                "missing_columns": res["missing_columns"],
                "query": res["query"],
            }
            for name, res in self.resolution.items()
        }
        return {
            "spec_version": self.spec_version,
            "n_heads": len(self.heads),
            "resolved_heads": [n for n, r in self.resolution.items() if r["resolved"]],
            "unresolved_heads": [n for n, r in self.resolution.items() if not r["resolved"]],
            "per_head": per_head,
        }

    # ------------------------------------------------------------------
    # Decode rules (per configs/decoder_spec_v1.yaml decode.rule)
    # ------------------------------------------------------------------

    def _decode_pulse_train(
        self, name: str, nodes: np.ndarray, steps: np.ndarray, ev_node: np.ndarray,
        n_steps: int, dt: float, params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """per-MN firing runs with inter-event gaps < bout_gap_s, merged."""
        bout_gap_s = float(params.get("bout_gap_s", 2 * dt))
        min_events = int(params.get("min_events", 1))
        by_node = _node_event_steps(steps, ev_node)
        runs: list[tuple[int, int, int]] = []
        for node in nodes:
            ev = by_node.get(int(node))
            if ev is None or ev.size == 0:
                continue
            gaps = np.diff(ev) * dt
            split_at = np.flatnonzero(gaps >= bout_gap_s - 1e-12) + 1
            for run in np.split(ev, split_at):
                if run.size < min_events:
                    continue
                runs.append((int(run[0]), int(run[-1]), int(node)))
        merged = _merge_overlapping(runs)
        return _steps_to_bout_rows(name, merged, steps, ev_node, dt)

    def _decode_sustained_drive(
        self, name: str, nodes: np.ndarray, steps: np.ndarray, ev_node: np.ndarray,
        n_steps: int, dt: float, params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """uninterrupted population drive >= min_rate_hz for >= window.

        A bout is a maximal cluster of population activity where no
        inter-event gap reaches ``window_s`` (the drive never pauses
        longer than the window); it qualifies when its mean rate is
        >= ``min_rate_hz`` and its span >= ``min_bout_s`` (default:
        ``window_s``). Isolated spikes form dt-span clusters and are
        rejected by the duration rule - "sustained" is measured, not
        inferred from a single supra-threshold sample.
        """
        window_s = float(params.get("window_s") or params.get("min_bout_s") or 0.1)
        min_rate = float(params.get("min_rate_hz", 10.0))
        min_bout_s = float(params.get("min_bout_s") or params.get("window_s") or 0.0)
        if ev_node.size == 0 or nodes.size == 0:
            return []
        sel = np.isin(ev_node, nodes)
        ev = np.unique(steps[sel])
        if ev.size == 0:
            return []
        counts = np.zeros(n_steps, dtype=np.float64)
        np.add.at(counts, steps[sel], 1.0)
        gaps = np.diff(ev) * dt
        split_at = np.flatnonzero(gaps >= window_s - 1e-12) + 1
        runs: list[tuple[int, int, np.ndarray]] = []
        for cluster in np.split(ev, split_at):
            first, last = int(cluster[0]), int(cluster[-1])
            span = (last - first + 1) * dt
            n_events = int(counts[first: last + 1].sum())
            if span < min_bout_s - 1e-12 or n_events / span < min_rate - 1e-12:
                continue
            runs.append((first, last, nodes))
        return _steps_to_bout_rows(name, runs, steps, ev_node, dt, min_bout_s)

    def _decode_alternating_rate(
        self, name: str, nodes: np.ndarray, steps: np.ndarray, ev_node: np.ndarray,
        n_steps: int, dt: float, params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """anti-correlated subpopulation alternation -> locomotion bouts.

        Subpopulations are the L/R soma sides when the resolved set is
        lateralized, else the first/second half of the sorted node list.
        A window alternates when the per-step dominance sign (which
        half drives) flips at least ``min_switches`` times while both
        halves' windowed rates stay above zero and the combined rate
        clears ``min_rate_hz``.
        """
        window_s = float(params.get("window_s", 0.2))
        min_rate = float(params.get("min_rate_hz", 5.0))
        min_switches = int(params.get("min_switches", 2))
        a, b = self._split_subpopulations(nodes)
        w = max(1, int(round(window_s / dt)))
        counts_a = _population_counts(steps, ev_node, a, n_steps)
        counts_b = _population_counts(steps, ev_node, b, n_steps)
        rate_a = _windowed_rate(counts_a, w, dt)
        rate_b = _windowed_rate(counts_b, w, dt)
        # per-step dominance: which half drives right now. Anti-phase
        # groups flip this sign continuously; windowed rates would
        # average the alternation away.
        sign = np.sign(counts_a - counts_b)
        sign[sign == 0] = np.nan
        # forward-fill ties so momentary equality does not fake a switch
        for i in range(1, sign.size):
            if np.isnan(sign[i]):
                sign[i] = sign[i - 1]
        sign = np.nan_to_num(sign)
        flips = np.r_[0.0, (sign[1:] * sign[:-1] < 0).astype(np.float64)]
        switch_count = np.convolve(flips, np.ones(w), "same")
        active = (
            (rate_a + rate_b >= min_rate - 1e-12)
            & (rate_a > 0)
            & (rate_b > 0)
            & (switch_count >= min_switches)
        )
        runs = self._active_runs(active, nodes)
        return _steps_to_bout_rows(name, runs, steps, ev_node, dt)

    def _decode_threshold_events(
        self, name: str, nodes: np.ndarray, steps: np.ndarray, ev_node: np.ndarray,
        n_steps: int, dt: float, params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """population rate above min_rate_hz (instantaneous or windowed)."""
        min_rate = float(params.get("min_rate_hz", 20.0))
        window_s = params.get("window_s")
        w = max(1, int(round(float(window_s) / dt))) if window_s else 1
        min_bout_s = float(params.get("min_bout_s", 0.0))
        counts = _population_counts(steps, ev_node, nodes, n_steps)
        active = _windowed_rate(counts, w, dt) >= min_rate - 1e-12
        runs = self._active_runs(active, nodes)
        return _steps_to_bout_rows(name, runs, steps, ev_node, dt, min_bout_s)

    def _decode_burst_pattern(
        self, name: str, nodes: np.ndarray, steps: np.ndarray, ev_node: np.ndarray,
        n_steps: int, dt: float, params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """population event clusters: gaps < bout_gap_s, >= min_events."""
        window_s = float(params.get("window_s", 0.0))
        bout_gap_s = float(params.get("bout_gap_s", max(3 * dt, 0.25 * window_s)))
        min_events = int(params.get("min_events", 3))
        min_bout_s = float(params.get("min_bout_s", 0.0))
        if ev_node.size == 0 or nodes.size == 0:
            return []
        ev = np.unique(steps[np.isin(ev_node, nodes)])
        if ev.size == 0:
            return []
        gaps = np.diff(ev) * dt
        split_at = np.flatnonzero(gaps >= bout_gap_s - 1e-12) + 1
        runs = [
            (int(run[0]), int(run[-1]), nodes)
            for run in np.split(ev, split_at)
            if run.size >= min_events
        ]
        return _steps_to_bout_rows(name, runs, steps, ev_node, dt, min_bout_s)

    _RULES = {
        "pulse_train": _decode_pulse_train,
        "sustained_drive": _decode_sustained_drive,
        "alternating_rate": _decode_alternating_rate,
        "threshold_events": _decode_threshold_events,
        "burst_pattern": _decode_burst_pattern,
    }

    @staticmethod
    def _active_runs(active: np.ndarray, nodes: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
        """Contiguous True runs of an active mask -> (first, last, nodes)."""
        edges = np.flatnonzero(np.diff(np.r_[False, active, False].astype(np.int8)))
        return [
            (int(edges[i]), int(edges[i + 1] - 1), nodes)
            for i in range(0, edges.size, 2)
        ]

    def _split_subpopulations(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """L/R split when the readout set is lateralized, else halves."""
        sides = _column_or_none(self.bundle.nodes, "somaSide")
        if sides is not None:
            side_sel = sides[nodes]
            left = nodes[side_sel == "L"]
            right = nodes[side_sel == "R"]
            if left.size and right.size and left.size + right.size >= nodes.size // 2:
                return left, right
        half = nodes.size // 2 or 1
        return nodes[:half], nodes[half:]

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(
        self,
        events_step: Any,
        events_node: Any = None,
        n_steps: int | None = None,
        dt: float | None = None,
        sex: str = "male",
    ) -> list[dict[str, Any]]:
        """Decode all spec heads -> bout rows (schema in module docstring).

        Male-only heads emit a masked row under ``sex="female"`` and
        vice versa (T-SEX-4). Zero-resolution heads emit a masked row
        and are listed in ``coverage_report().unresolved_heads``.
        """
        if sex not in SEX_VALUES:
            raise ValueError(f"sex must be one of {SEX_VALUES}, got {sex!r}")
        steps, ev_node, n_steps, dt = _as_step_node_arrays(
            events_step, events_node, n_steps, dt
        )
        rows: list[dict[str, Any]] = []
        for name, res in self.resolution.items():
            if SEX_MASK.get(res["sex"]) == sex:
                rows.append(self._masked_row(name, "sex"))
                continue
            if not res["resolved"]:
                rows.append(self._masked_row(name, "unresolved_readout"))
                continue
            params = dict(res["decode"])
            rule = res["rule"]
            fn = self._RULES.get(rule)
            if fn is None:
                rows.append(self._masked_row(name, f"unknown_rule:{rule}"))
                continue
            rows.extend(
                fn(self, name, res["node_idx"], steps, ev_node, n_steps, dt, params)
            )
        rows.sort(
            key=lambda r: (float("inf") if r["t0_s"] is None else r["t0_s"], r["behavior"])
        )
        return rows

    @staticmethod
    def _masked_row(name: str, reason: str) -> dict[str, Any]:
        return {
            "behavior": name,
            "t0_s": None,
            "t1_s": None,
            "evidence": {"nodes": [], "rate_hz": 0.0, "n_events": 0},
            "masked": True,
            "masked_reason": reason,
        }


# ---------------------------------------------------------------------------
# Song decoder: pulse/sine ethogram at the courtship-slice dt (T-REAL-3)
# ---------------------------------------------------------------------------


class SongDecoder:
    """Wraps the ``song_pulse``/``song_sine`` heads at dt=5ms.

    Two-tier segmentation: per-MN spike trains first collapse into
    *pulses* via the spec ``bout_gap_s`` (10ms), then pulses with
    onset-to-onset gaps < ``bout_merge_ipi_s`` merge into pulse-mode
    bouts. The IPI series is the onset interval between consecutive
    pulses - the statistic the SongExplorer reference ranges pin.
    """

    PULSE_HEAD = "song_pulse"
    SINE_HEAD = "song_sine"

    def __init__(self, bundle: GraphBundle, spec_cfg: dict | Path | None = None):
        self.decoder = BehaviorDecoder(bundle, spec_cfg)
        self.bundle = bundle
        for head in (self.PULSE_HEAD, self.SINE_HEAD):
            if head not in self.decoder.heads:
                raise KeyError(f"decoder spec lacks required head {head!r}")

    def ethogram(
        self,
        events_step: Any,
        events_node: Any = None,
        n_steps: int | None = None,
        dt: float = SONG_DT_S,
        sex: str = "male",
        context: list[tuple[str, float, float]] | None = None,
        bout_merge_ipi_s: float = 0.150,
    ) -> dict[str, Any]:
        """Segment events into a song ethogram statistics dict.

        ``context`` is an optional list of ``(label, t0_s, t1_s)``
        segments (e.g. near/far female distance) used for the
        context-dependence direction check: ``direction_consistent``
        iff the pulse-mode share of bout time is higher in "near"
        segments than in "far" segments (Roemschied 2023 direction).
        """
        steps, ev_node, n_steps, dt = _as_step_node_arrays(
            events_step, events_node, n_steps, dt
        )
        res_pulse = self.decoder.resolution[self.PULSE_HEAD]
        res_sine = self.decoder.resolution[self.SINE_HEAD]
        masked = SEX_MASK.get(res_pulse["sex"]) == sex or not (
            res_pulse["resolved"] and res_sine["resolved"]
        )
        if masked:
            return {
                "dt_s": dt,
                "n_steps": n_steps,
                "masked": True,
                "pulses": [],
                "pulse_bouts": [],
                "sine_bouts": [],
                "ipi_ms": [],
                "bout_durations_s": [],
                "pulse_time_s": 0.0,
                "sine_time_s": 0.0,
                "pulse_sine_ratio": 0.0,
                "ipi_median_ms": None,
                "ipi_cv": None,
                "bout_duration_median_s": None,
                "context_switch_log": [],
                "context_dependence": None,
            }

        pulse_rows = self.decoder.decode(steps, ev_node, n_steps, dt, sex=sex)
        pulses = [r for r in pulse_rows if r["behavior"] == self.PULSE_HEAD]
        sine_bouts = [r for r in pulse_rows if r["behavior"] == self.SINE_HEAD]
        pulse_bouts = self._merge_pulse_bouts(pulses, bout_merge_ipi_s, dt)

        onsets = np.array([p["t0_s"] for p in pulses])
        bout_ids = np.zeros(onsets.size, dtype=int)
        for i, bout in enumerate(pulse_bouts):
            in_bout = (onsets >= bout["t0_s"] - 1e-12) & (onsets < bout["t1_s"] + 1e-12)
            bout_ids[in_bout] = i
        ipi_ms = [
            (onsets[i] - onsets[i - 1]) * 1e3
            for i in range(1, onsets.size)
            if bout_ids[i] == bout_ids[i - 1]
        ]

        durations = [b["t1_s"] - b["t0_s"] for b in pulse_bouts + sine_bouts]
        pulse_time = float(sum(b["t1_s"] - b["t0_s"] for b in pulse_bouts))
        sine_time = float(sum(b["t1_s"] - b["t0_s"] for b in sine_bouts))
        total = pulse_time + sine_time
        ipi_arr = np.asarray(ipi_ms)
        stats = {
            "dt_s": dt,
            "n_steps": n_steps,
            "masked": False,
            "pulses": pulses,
            "pulse_bouts": pulse_bouts,
            "sine_bouts": sine_bouts,
            "ipi_ms": ipi_ms,
            "bout_durations_s": durations,
            "pulse_time_s": pulse_time,
            "sine_time_s": sine_time,
            "pulse_sine_ratio": (pulse_time / total) if total > 0 else 0.0,
            "ipi_median_ms": float(np.median(ipi_arr)) if ipi_arr.size else None,
            "ipi_cv": (
                float(ipi_arr.std() / ipi_arr.mean())
                if ipi_arr.size and ipi_arr.mean() > 0
                else None
            ),
            "bout_duration_median_s": (
                float(np.median(durations)) if durations else None
            ),
            "context_switch_log": self._context_switches(pulse_bouts, sine_bouts),
            "context_dependence": self._context_direction(pulse_bouts, sine_bouts, context),
        }
        return stats

    @staticmethod
    def _merge_pulse_bouts(
        pulses: list[dict[str, Any]], merge_gap_s: float, dt: float
    ) -> list[dict[str, Any]]:
        """Group pulses with onset gaps < merge gap into pulse-mode bouts."""
        if not pulses:
            return []
        pulses = sorted(pulses, key=lambda r: r["t0_s"])
        bouts: list[dict[str, Any]] = []
        cur = dict(pulses[0])
        for p in pulses[1:]:
            if p["t0_s"] - cur["t1_s"] < merge_gap_s:
                cur["t1_s"] = max(cur["t1_s"], p["t1_s"])
                cur["evidence"] = {
                    "nodes": sorted(set(cur["evidence"]["nodes"]) | set(p["evidence"]["nodes"])),
                    "rate_hz": 0.0,
                    "n_events": cur["evidence"]["n_events"] + p["evidence"]["n_events"],
                }
            else:
                bouts.append(cur)
                cur = dict(p)
        bouts.append(cur)
        for b in bouts:
            span = max(b["t1_s"] - b["t0_s"], dt)
            b["evidence"]["rate_hz"] = b["evidence"]["n_events"] / span
        return bouts

    @staticmethod
    def _context_switches(
        pulse_bouts: list[dict[str, Any]], sine_bouts: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Chronological pulse<->sine mode transitions."""
        seq = sorted(
            [(b["t0_s"], "pulse") for b in pulse_bouts]
            + [(b["t0_s"], "sine") for b in sine_bouts]
        )
        log = []
        for (_, m_prev), (t, m) in zip(seq, seq[1:], strict=False):
            if m != m_prev:
                log.append({"t_s": t, "from_mode": m_prev, "to_mode": m})
        return log

    @staticmethod
    def _context_direction(
        pulse_bouts: list[dict[str, Any]],
        sine_bouts: list[dict[str, Any]],
        context: list[tuple[str, float, float]] | None,
    ) -> str | None:
        """near -> more pulse share than far; None without context."""
        if not context:
            return None
        share = {}
        for label, t0, t1 in context:
            pt = sum(
                max(0.0, min(b["t1_s"], t1) - max(b["t0_s"], t0)) for b in pulse_bouts
            )
            st = sum(
                max(0.0, min(b["t1_s"], t1) - max(b["t0_s"], t0)) for b in sine_bouts
            )
            share[label] = pt / (pt + st) if (pt + st) > 0 else 0.0
        near = share.get("near")
        far = share.get("far")
        if near is None or far is None:
            return None
        return "direction_consistent" if near > far else "direction_reversed"

    @staticmethod
    def compare_ethogram(stats: dict[str, Any], reference: dict | Path | None) -> dict[str, Any]:
        """Score an ethogram dict against the frozen reference ranges."""
        return compare_ethogram(stats, reference)


# ---------------------------------------------------------------------------
# Ethogram comparison vs evals/song_reference.json (T-REAL-3)
# ---------------------------------------------------------------------------

_STAT_MEASURED = {
    "ipi_median_ms": ("ipi_median_ms", "expected_range_ms", "ipi_ms"),
    "pulse_sine_ratio": ("pulse_sine_ratio", "expected_range", None),
    "bout_duration_s": ("bout_duration_median_s", "expected_range_s", "bout_durations_s"),
    "ipi_cv": ("ipi_cv", "expected_range", None),
    "context_dependence": ("context_dependence", "expected", None),
}


def _ks_to_uniform(sample: np.ndarray, lo: float, hi: float) -> float:
    """KS distance of the empirical CDF vs uniform on [lo, hi].

    The frozen reference stores ranges, not raw SongExplorer samples, so
    the documented KS surrogate compares each measured sample's ECDF to
    the uniform distribution over the stat's expected range (EVAL.md
    T-REAL-3 "KS statistics reported per stat").
    """
    x = np.sort(np.asarray(sample, dtype=np.float64))
    n = x.size
    if n == 0 or hi <= lo:
        return 0.0
    u = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    e_hi = np.arange(1, n + 1) / n
    e_lo = np.arange(0, n) / n
    return float(np.max(np.maximum(e_hi - u, u - e_lo)))


def compare_ethogram(stats: dict[str, Any], reference: dict | Path | None) -> dict[str, Any]:
    """Score an ethogram dict against the frozen reference ranges.

    Returns per-stat inside/outside + measured value + KS surrogate, the
    inside count, and ``passed`` per T-REAL-3 (>=4/5 inside range).
    ``reference`` is the parsed song_reference.json dict or its path.
    """
    if reference is None:
        reference = load_song_reference()
    elif isinstance(reference, (str, Path)):
        reference = load_song_reference(Path(reference))
    per_stat: dict[str, Any] = {}
    n_inside = 0
    entries = {s["id"]: s for s in reference.get("statistics", [])}
    for stat_id, (measured_key, range_key, sample_key) in _STAT_MEASURED.items():
        entry = entries.get(stat_id, {})
        expected = entry.get(range_key)
        value = stats.get(measured_key)
        record: dict[str, Any] = {
            "value": value,
            "expected": expected,
            "inside": False,
            "ks": None,
        }
        if stat_id == "context_dependence":
            record["inside"] = value is not None and value == expected
        elif isinstance(expected, (list, tuple)) and len(expected) == 2 and value is not None:
            lo, hi = float(expected[0]), float(expected[1])
            record["inside"] = bool(lo <= float(value) <= hi)
            sample = stats.get(sample_key) if sample_key else None
            if sample:
                record["ks"] = _ks_to_uniform(np.asarray(sample), lo, hi)
        if record["inside"]:
            n_inside += 1
        per_stat[stat_id] = record
    n_total = len(entries) or len(_STAT_MEASURED)
    return {
        "panel": reference.get("panel", "song_reference_v1"),
        "n_inside": n_inside,
        "n_total": n_total,
        "statistics": per_stat,
        "passed": n_inside >= 4,
        "pass_rule": ">=4/5 statistics inside expected ranges (EVAL.md T-REAL-3)",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _demo_events(decoder: BehaviorDecoder) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Deterministic synthetic event set exercising every decode rule."""
    dt = 0.005
    n_steps = 800
    steps: list[int] = []
    nodes: list[int] = []
    for res in decoder.resolution.values():
        pop = res["node_idx"]
        if not pop.size:
            continue
        rule = res["rule"]
        if rule == "pulse_train":
            for onset in range(100, 700, 37):  # ~37ms IPI pulse train
                node = int(pop[onset % pop.size])
                steps += [onset, onset + 1]
                nodes += [node, node]
        elif rule in ("sustained_drive", "burst_pattern"):
            for s in range(200, 400):
                node = int(pop[s % pop.size])
                steps.append(s)
                nodes.append(node)
        elif rule == "alternating_rate":
            a, b = decoder._split_subpopulations(pop)
            for s in range(100, 500, 2):
                steps.append(s)
                nodes.append(int(a[s % a.size] if a.size else pop[0]))
                steps.append(s + 1)
                nodes.append(int(b[s % b.size] if b.size else pop[-1]))
        elif rule == "threshold_events":
            for s in range(600, 650):
                steps.append(s)
                nodes.append(int(pop[s % pop.size]))
    arr_s = np.asarray(steps, dtype=np.int64)
    arr_n = np.asarray(nodes, dtype=np.int64)
    order = np.lexsort((arr_n, arr_s))
    return arr_s[order], arr_n[order], n_steps, dt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph-dir", type=Path, default=REPO_ROOT / "data" / "graph")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    ap.add_argument("--cutoff", type=int, default=5)
    ap.add_argument("--sex", choices=SEX_VALUES, default="male")
    ap.add_argument("--resolve", action="store_true", help="print per-head populations")
    ap.add_argument("--demo", action="store_true", help="decode a synthetic event set")
    args = ap.parse_args(argv)

    if not (args.graph_dir / "nodes.parquet").exists():
        print(f"nodes.parquet absent under {args.graph_dir} - run the graph build first")
        return 1
    bundle = load_graph_bundle(args.graph_dir, cutoff=args.cutoff)
    decoder = BehaviorDecoder(bundle, load_spec(args.spec))
    report = decoder.coverage_report()
    if args.resolve or not args.demo:
        print(f"decoder spec {report['spec_version']}: {report['n_heads']} heads")
        for name, h in report["per_head"].items():
            tag = "OK" if h["resolved"] else "ZERO-NODES"
            print(
                f"  {name:12s} [{tag:10s}] n={h['n_nodes']:<6d} rule={h['rule']:<18s}"
                f" sex={h['sex']} missing_cols={h['missing_columns']}"
            )
        if report["unresolved_heads"]:
            print(f"  unresolved: {report['unresolved_heads']}")
    if args.demo:
        steps, nodes, n_steps, dt = _demo_events(decoder)
        rows = decoder.decode(steps, nodes, n_steps, dt, sex=args.sex)
        print(f"demo decode: {steps.size} events -> {len(rows)} rows (sex={args.sex})")
        for r in rows[:20]:
            if r["masked"]:
                print(f"  {r['behavior']:12s} MASKED ({r['masked_reason']})")
            else:
                e = r["evidence"]
                print(
                    f"  {r['behavior']:12s} {r['t0_s']:.3f}-{r['t1_s']:.3f}s "
                    f"n={e['n_events']} rate={e['rate_hz']:.0f}Hz nodes={e['nodes'][:6]}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
