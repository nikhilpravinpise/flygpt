"""T-STIM-1..5 - stimulus correctness runners (plan.md T-STIM table).

Stimulus programs are accepted as dicts or StimulusProgram-like objects
with fields: ``family_id``, ``program_id``, ``drives`` (list of
{nodes, rate_hz, window, envelope, envelope_id, query}),
``silences`` (list of {nodes, window}), ``duration_s``, ``dt`` and
``meta`` (JSON-able; these checks read ``meta.drive_query`` /
``meta.queries``, ``meta.side``/``meta.side_policy``, and
``meta.envelope_*`` for naturalistic programs). ``index`` is any
query-resolver exposing ``resolve(types=, classes=, subclasses=,
superclasses=, sides=, neuromeres=, entry_nerves=)`` - the real
``flygpt.stimuli.SensoryIndex`` lazily, or a test double.

Every runner tolerates a missing optional dep via a documented skip
(``checks.skipped`` + ``pass: null``), never a silent pass, and writes
``results/T-STIM-*.json`` via ``flygpt.schema.write_result``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from flygpt.evals import common
from flygpt.evals.corpus_checks import _corpus_events
from flygpt.evals.sim_checks import (
    _call_sim,
    _cfg_get,
    _cfg_with,
    _events_of,
    _resolve_query,
    _skipped,
)
from flygpt.schema import rng_from_seed, seed_from_config, write_result

# ---------------------------------------------------------------------------
# Program normalization helpers (dict or attribute access)
# ---------------------------------------------------------------------------


def _prog_get(prog: Any, key: str, default: Any = None) -> Any:
    if isinstance(prog, dict):
        return prog.get(key, default)
    return getattr(prog, key, default)


def _prog_meta(prog: Any) -> dict:
    return dict(_prog_get(prog, "meta", {}) or {})


def _prog_drives(prog: Any) -> list[dict]:
    return [dict(d) for d in (_prog_get(prog, "drives", []) or [])]


def _prog_id(prog: Any) -> str:
    return str(_prog_get(prog, "program_id", _prog_get(prog, "family_id", "?")))


def _drive_queries(prog: Any) -> list[tuple[int, dict | None, bool]]:
    """(drive_index, resolve_spec, exact) triples.

    ``resolve_spec`` re-feeds the index resolver: a drive's own
    ``resolve_kwargs`` (the narrowed reproduction a sampled pick must
    match exactly) wins, then per-drive ``query`` /
    meta.drive_queries[i] / meta.drives[i].query / meta.drive_query
    (a broad query claiming the whole drive - stored nodes must equal
    its resolution, so subset-sampling without a recorded
    resolve_kwargs is flagged as the provenance gap it is).
    """
    drives = _prog_drives(prog)
    meta = _prog_meta(prog)
    per_drive = meta.get("drive_queries") or []
    drive_metas = meta.get("drives") or []
    out = []
    for i, d in enumerate(drives):
        exact = d.get("resolve_kwargs")
        if exact is None and i < len(drive_metas) and isinstance(drive_metas[i], dict):
            exact = drive_metas[i].get("resolve_kwargs")
        if exact is not None:
            out.append((i, dict(exact), True))
            continue
        q = d.get("query")
        if q is None and i < len(per_drive):
            q = per_drive[i]
        if (
            q is None
            and i < len(drive_metas)
            and isinstance(drive_metas[i], dict)
        ):
            q = drive_metas[i].get("query")
        if q is None:
            q = meta.get("drive_query")
        out.append((i, q, False))
    if not drives and meta.get("drive_query"):
        out.append((-1, meta["drive_query"], False))
    return out


def _lazy_index(bundle):
    try:
        from flygpt.stimuli import SensoryIndex, load_stim_config

        stim_yaml = Path("configs/stim_library_v1.yaml")
        scfg = load_stim_config(stim_yaml) if stim_yaml.exists() else None
        return SensoryIndex(bundle, scfg)
    except Exception:
        return None


def _first(*vals):
    """First non-None value - numpy-safe (no ``or`` on arrays)."""
    for v in vals:
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# T-STIM-1: grounding re-resolution (dangling refs = 0)
# ---------------------------------------------------------------------------


def run_t_stim_1(library, bundle, results_dir: Path, *, index=None) -> dict:
    """T-STIM-1: every program's recorded grounding query re-resolves.

    For each program the stored drive-node set is recomputed through the
    sensory index from ``meta``'s recorded query. A dangling reference
    is a stored node outside the graph (>= n_nodes) or a query that no
    longer resolves to the same non-empty set. Dangling refs = 0 and
    re-resolution mismatches = 0 to pass.
    """
    test_id = "T-STIM-1"
    if index is None:
        index = _lazy_index(bundle)
    inputs = {"n_programs": len(library) if library else 0}
    if index is None:
        return _skipped(
            results_dir, test_id, "flygpt.stimuli.SensoryIndex unavailable", inputs=inputs
        )
    per_program = []
    n_dangling = 0
    n_mismatch = 0
    n_unresolvable = 0
    n_no_query = 0
    for prog in library or []:
        pid = _prog_id(prog)
        queries = _drive_queries(prog)
        per_drive = _prog_drives(prog)
        stored = sorted({int(n) for d in per_drive for n in (d.get("nodes") or [])})
        stored_per_drive = [
            sorted({int(n) for n in (d.get("nodes") or [])}) for d in per_drive
        ]
        resolved_exact: set[int] = set()
        resolved_broad: set[int] = set()
        problems = []
        saw_query = False
        for di, q, exact in queries:
            if not q:
                problems.append(f"drive {di}: no recorded query")
                continue
            saw_query = True
            try:
                nodes = {int(x) for x in _resolve_query(index, q, bundle, None)}
            except Exception as exc:
                problems.append(f"drive {di}: query error {exc}")
                continue
            if exact:
                resolved_exact.update(nodes)
                if 0 <= di < len(stored_per_drive) and sorted(nodes) != stored_per_drive[di]:
                    problems.append(
                        f"drive {di}: resolve_kwargs re-resolution mismatch: "
                        f"stored {len(stored_per_drive[di])} nodes vs resolved {len(nodes)}"
                    )
                    n_mismatch += 1
            else:
                resolved_broad.update(nodes)
        if not saw_query and stored:
            n_no_query += 1
        out_of_graph = [n for n in stored if n < 0 or n >= bundle.n_nodes]
        if out_of_graph:
            problems.append(f"dangling node refs: {out_of_graph}")
            n_dangling += len(out_of_graph)
        if saw_query and not (resolved_exact or resolved_broad):
            problems.append("queries resolve to zero nodes")
            n_unresolvable += 1
        if resolved_broad:
            outside = [n for n in stored if n not in resolved_broad]
            if outside or sorted(resolved_broad) != stored:
                problems.append(
                    "re-resolution mismatch: broad query resolves "
                    f"{len(resolved_broad)} nodes vs stored {len(stored)}"
                    + (f" ({len(outside)} stored outside)" if outside else "")
                )
                n_mismatch += 1
        per_program.append({"program_id": pid, "ok": not problems, "problems": problems})
    checks = {
        "n_programs": len(per_program),
        "n_dangling_refs": n_dangling,
        "n_unresolvable": n_unresolvable,
        "n_resolution_mismatches": n_mismatch,
        "n_programs_missing_query": n_no_query,
        "per_program": per_program,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"n_dangling_refs": n_dangling},
        passed=bool(
            n_dangling == 0 and n_mismatch == 0 and n_unresolvable == 0 and n_no_query == 0
        ),
    )


# ---------------------------------------------------------------------------
# T-STIM-2: laterality audit
# ---------------------------------------------------------------------------


def run_t_stim_2(library, bundle, results_dir: Path) -> dict:
    """T-STIM-2: lateralized programs drive the flagged hemisphere only.

    A program is lateralized when meta.side in {L, R} (or
    meta.side_policy == "lateralized" with an explicit side). Every
    driven node's somaSide (rootSide fallback) must equal the declared
    side. Bilateral programs are recorded, not audited.
    """
    test_id = "T-STIM-2"
    col = "somaSide" if "somaSide" in bundle.nodes.columns else (
        "rootSide" if "rootSide" in bundle.nodes.columns else None
    )
    inputs = {"side_column": col, "n_programs": len(library) if library else 0}
    if col is None:
        return _skipped(results_dir, test_id, "no somaSide/rootSide column on nodes", inputs=inputs)
    sides = np.asarray(bundle.nodes[col].to_list()).astype(str)
    per_program = []
    n_bad = 0
    n_checked = 0
    for prog in library or []:
        meta = _prog_meta(prog)
        side = meta.get("side") or meta.get("hemisphere")
        lateralized = side in ("L", "R") or meta.get("side_policy") == "lateralized"
        if not lateralized:
            per_program.append({"program_id": _prog_id(prog), "lateralized": False, "ok": True})
            continue
        n_checked += 1
        if side not in ("L", "R"):
            n_bad += 1
            per_program.append(
                {
                    "program_id": _prog_id(prog),
                    "lateralized": True,
                    "ok": False,
                    "problems": [f"lateralized program lacks side flag (meta.side={side!r})"],
                }
            )
            continue
        nodes = sorted({int(n) for d in _prog_drives(prog) for n in (d.get("nodes") or [])})
        wrong = [n for n in nodes if n < len(sides) and sides[n] != side]
        out_of_range = [n for n in nodes if n >= len(sides)]
        problems = []
        if out_of_range:
            problems.append(f"driven nodes out of graph range: {out_of_range}")
        if wrong:
            problems.append(f"{len(wrong)} driven nodes on wrong side: {wrong[:8]}")
        n_bad += int(bool(problems))
        per_program.append(
            {
                "program_id": _prog_id(prog),
                "lateralized": True,
                "side": side,
                "n_driven": len(nodes),
                "n_wrong_side": len(wrong),
                "ok": not problems,
                "problems": problems,
            }
        )
    checks = {
        "n_programs": len(per_program),
        "n_lateralized_checked": n_checked,
        "n_violations": n_bad,
        "per_program": per_program,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"n_violations": n_bad},
        passed=bool(n_bad == 0),
    )


# ---------------------------------------------------------------------------
# T-STIM-3: coverage-tracker recount equality
# ---------------------------------------------------------------------------


def run_t_stim_3(
    results_dir: Path,
    *,
    tracker_report: dict | None = None,
    tracker=None,
    corpus_dir: Path | None = None,
    node_type=None,
    manifest_types: list[str] | None = None,
) -> dict:
    """T-STIM-3: tracker output == independent recount, exactly.

    The recount recomputes driven/robust per EVAL 1.8 straight from
    shard event columns (``corpus_dir``). ``tracker_report`` (or a
    ``tracker.report()``) is a per-type dict of {driven, robust}; any
    mismatch on any manifest type fails.
    """
    test_id = "T-STIM-3"
    if tracker_report is None and tracker is not None:
        tracker_report = tracker.report()
    inputs = {"corpus_dir": str(corpus_dir) if corpus_dir else None}
    if tracker_report is None:
        return _skipped(results_dir, test_id, "no tracker report supplied", inputs=inputs)
    if corpus_dir is None or node_type is None or manifest_types is None:
        return _skipped(
            results_dir,
            test_id,
            "recount requires corpus_dir + node_type + manifest_types",
            inputs=inputs,
        )

    node_type = np.asarray(node_type).astype(str)
    ev_trial, ev_node = _corpus_events(Path(corpus_dir))
    recount = common.coverage_by_type(ev_trial, ev_node, node_type, manifest_types)["per_type"]
    mismatches = []
    for t in manifest_types:
        tracked = tracker_report.get(t) or {}
        counted = recount.get(t) or {"driven": False, "robust": False}
        for field in ("driven", "robust"):
            if bool(tracked.get(field)) != bool(counted.get(field)):
                mismatches.append(
                    {"type": t, "field": field, "tracker": tracked.get(field), "recount": counted.get(field)}
                )
    checks = {"n_types": len(manifest_types), "n_mismatches": len(mismatches), "mismatches": mismatches}
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"n_mismatches": len(mismatches)},
        passed=bool(len(mismatches) == 0),
    )


# ---------------------------------------------------------------------------
# T-STIM-4: naturalistic envelope bounds + dt
# ---------------------------------------------------------------------------


def run_t_stim_4(library, results_dir: Path) -> dict:
    """T-STIM-4: envelope samples within [0, stated max] at declared dt.

    Naturalistic programs expose their sampled envelope via
    meta.envelope_values (per drive: envelope_values too) plus
    meta.envelope_dt and meta.envelope_max. Checks: non-negative, <=
    stated max, envelope_dt == the program dt, and
    len(values) == round(window_len / envelope_dt).
    """
    test_id = "T-STIM-4"
    per_program = []
    n_bad = 0
    n_checked = 0
    for prog in library or []:
        meta = _prog_meta(prog)
        dt = float(_prog_get(prog, "dt", meta.get("dt") or 0.02))
        problems = []
        saw_envelope = False
        for i, d in enumerate(_prog_drives(prog)):
            values = _first(d.get("envelope_values"), meta.get("envelope_values"))
            env_dt = _first(d.get("envelope_dt"), meta.get("envelope_dt"), dt)
            env_max = _first(d.get("envelope_max"), meta.get("envelope_max"))
            envelope = _first(d.get("envelope"), meta.get("envelope"))
            if values is None and envelope not in ("naturalistic",):
                continue
            saw_envelope = True
            if values is None:
                problems.append(f"drive {i}: naturalistic envelope lacks recorded samples")
                continue
            vals = np.asarray(values, dtype=np.float64)
            w0, w1 = (
                _first(
                    d.get("window_s"),
                    d.get("window"),
                    [0.0, float(_prog_get(prog, "duration_s", 0.0))],
                )
            )[:2]
            if vals.size and float(vals.min()) < 0.0:
                problems.append(f"drive {i}: envelope below 0 (min={float(vals.min())})")
            if env_max is not None and vals.size and float(vals.max()) > float(env_max):
                problems.append(
                    f"drive {i}: envelope exceeds stated max {env_max} (max={float(vals.max())})"
                )
            if abs(float(env_dt) - dt) > 1e-12:
                problems.append(f"drive {i}: envelope_dt {env_dt} != program dt {dt}")
            expected_len = int(round((float(w1) - float(w0)) / float(env_dt))) if env_dt else 0
            if expected_len > 0 and vals.size != expected_len:
                problems.append(
                    f"drive {i}: envelope length {vals.size} != window/dt {expected_len}"
                )
        if saw_envelope:
            n_checked += 1
            n_bad += int(bool(problems))
            per_program.append({"program_id": _prog_id(prog), "ok": not problems, "problems": problems})
    checks = {"n_envelope_programs": n_checked, "n_violations": n_bad, "per_program": per_program}
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs={"n_programs": len(library) if library else 0},
        metrics={"n_violations": n_bad},
        passed=bool(n_bad == 0),
    )


# ---------------------------------------------------------------------------
# T-STIM-5: silencing semantics (verified on sim outputs)
# ---------------------------------------------------------------------------


def run_t_stim_5(
    bundle,
    results_dir: Path,
    *,
    simulate_fn: Callable | None = None,
    cfg: Any = None,
    silence_nodes: np.ndarray | None = None,
    window_s: tuple[float, float] | None = None,
    n_steps: int = 50,
    rate_hz: float = 300.0,
    seed: int | None = None,
) -> dict:
    """T-STIM-5: silenced nodes -> 0 spikes AND 0 outgoing effect.

    Two arms, both with the silence applied: (A) no drive, (B) the
    silenced nodes driven at ``rate_hz`` over the silence window. Exact
    requirements: (a) no recorded events on silenced nodes inside the
    window in arm B; (b) every non-silenced node's in-window event list
    is identical between A and B - a sim that drops the record but
    still propagates phantom spikes fails here.
    """
    test_id = "T-STIM-5"
    dt = float(_cfg_get(cfg, "dt", 0.02) if cfg is not None else 0.02)
    if seed is None:
        seed = seed_from_config({"test": test_id, "n_steps": n_steps})
    if silence_nodes is None:
        rng = rng_from_seed(test_id, "silence_nodes", bundle.n_nodes)
        candidates = [i for i in range(bundle.n_nodes) if bundle.out_neighbors(i).size]
        if not candidates:
            return _skipped(results_dir, test_id, "graph has no outgoing edges to silence")
        silence_nodes = np.array(sorted(rng.choice(candidates, size=min(2, len(candidates)), replace=False)))
    silence_nodes = np.asarray(silence_nodes, dtype=np.int64)
    if window_s is None:
        t0 = round(0.4 * n_steps * dt, 9)
        window_s = (t0, round(0.8 * n_steps * dt, 9))
    inputs = {
        "silence_nodes": [int(i) for i in silence_nodes],
        "window_s": list(window_s),
        "n_steps": n_steps,
        "rate_hz": rate_hz,
        "seed": seed,
    }
    if simulate_fn is None:
        try:
            from flygpt.sim_backend import simulate as simulate_fn
        except Exception:
            simulate_fn = None
    if simulate_fn is None:
        return _skipped(
            results_dir, test_id, "flygpt.sim_backend.simulate unavailable", inputs=inputs
        )

    s0, s1 = int(window_s[0] / dt), min(int(np.ceil(window_s[1] / dt)), n_steps)
    silence = [{"nodes": [int(i) for i in silence_nodes], "window": list(window_s)}]
    base_prog = {
        "family_id": "t_stim_5_silence",
        "drives": [],
        "silences": silence,
        "n_steps": n_steps,
        "dt": dt,
    }
    sil_prog = dict(base_prog)
    sil_prog["drives"] = [
        {"nodes": [int(i) for i in silence_nodes], "rate_hz": rate_hz, "window": list(window_s)}
    ]
    # Both arms must share an identical background for the phantom/missing
    # comparison to be exact - force tonic/noise off so spec-seeded streams
    # cannot inject arm differences unrelated to the silenced drive.
    cfg_clean = _cfg_with(
        cfg, tonic_rate_hz=0.0, tonic_amount=0.0, noise_hz=0.0, noise_amp=0.0
    )
    base = _events_of(_call_sim(simulate_fn, bundle, base_prog, cfg_clean, seed))
    sil = _events_of(_call_sim(simulate_fn, bundle, sil_prog, cfg_clean, seed))
    sil_set = {int(i) for i in silence_nodes}

    in_window_base = (base[0] >= s0) & (base[0] < s1)
    in_window_sil = (sil[0] >= s0) & (sil[0] < s1)
    leaked_spikes = sorted({int(n) for n in sil[1][in_window_sil]} & sil_set)
    base_pairs = {
        (t, n)
        for t, n in zip(
            base[0][in_window_base].tolist(), base[1][in_window_base].tolist(), strict=True
        )
        if n not in sil_set
    }
    sil_pairs = {
        (t, n)
        for t, n in zip(
            sil[0][in_window_sil].tolist(), sil[1][in_window_sil].tolist(), strict=True
        )
        if n not in sil_set
    }
    phantom = sorted(sil_pairs - base_pairs)
    missing = sorted(base_pairs - sil_pairs)
    checks = {
        "window_steps": [s0, s1],
        "n_spikes_on_silenced_in_window": len(leaked_spikes),
        "leaked_silenced_nodes": leaked_spikes,
        "n_phantom_events_elsewhere": len(phantom),
        "n_missing_events_elsewhere": len(missing),
        "phantom_events": phantom[:16],
        "missing_events": missing[:16],
        "zero_spikes_ok": len(leaked_spikes) == 0,
        "zero_effect_ok": len(phantom) == 0 and len(missing) == 0,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={
            "n_spikes_on_silenced_in_window": len(leaked_spikes),
            "n_phantom_events_elsewhere": len(phantom),
        },
        passed=bool(
            len(leaked_spikes) == 0 and len(phantom) == 0 and len(missing) == 0
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_library(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, list) else raw.get("programs", [])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="T-STIM-1..5 stimulus checks")
    parser.add_argument("--library", type=Path, default=Path("data/stim/library_v1.json"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graph"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    bundle = None
    if (args.graph_dir / "nodes.parquet").exists():
        from flygpt.schema import load_graph_bundle

        bundle = load_graph_bundle(args.graph_dir)
    library = _load_library(args.library) if args.library.exists() else None
    if bundle is None or library is None:
        reason = f"missing {'bundle' if bundle is None else 'library'} ({args.library})"
        for tid in (f"T-STIM-{i}" for i in (1, 2, 4, 5)):
            _skipped(args.results_dir, tid, reason)
    else:
        run_t_stim_1(library, bundle, args.results_dir)
        run_t_stim_2(library, bundle, args.results_dir)
        run_t_stim_4(library, args.results_dir)
        run_t_stim_5(bundle, args.results_dir)
    run_t_stim_3(args.results_dir, corpus_dir=args.corpus_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
