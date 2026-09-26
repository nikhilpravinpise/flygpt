"""Sex-conditioning evals T-SEX-1..6 (EVAL.md section 5, T-SEX table).

Runners over injected measured outcomes - every ``run_*`` works on any
scale because the measurements arrive as artifacts/arguments:

- T-SEX-1  dimorphism set: ``evals/dimorphism_set.json`` items resolved
  to ``{item_id: measured_bool}``; pass at >= 80% reproduced, kill
  condition K4 flagged below 50% (EVAL section 6).
- T-SEX-2  sex-divergence per EVAL 1.9: ``d_i = |r_male,i - r_female,i| /
  (0.5*(sd_male,i + sd_female,i) + eps)`` per type, circuit divergence =
  mean over types. A dimorphic circuit is divergent iff its divergence
  exceeds the 95th percentile of the same metric over matched
  isomorphic-control circuits AND the male-vs-female direction is
  consistent across >= 3 seeds. Pass: >= 50% of dimorphic circuits
  divergent.
- T-SEX-3  confound report: every artifact must carry BOTH a
  ``sex_conditioned`` and a ``specimen_conditioned`` block - reporting
  requirement, missing = fail.
- T-SEX-4  masking audit: cross-sex type leaks must be exactly 0 -
  male-specific types absent under female conditioning, female-specific
  absent under male. Input: alignment table (male_type, banc_type,
  iso_class) + the two conditioning type sets.
- T-SEX-5  graph-swap audit: the female-conditioned edge set must equal
  the expected BANC mask exactly, and its difference vs the male mask
  must be confined to the swapped subgraph.
- T-SEX-6  isomorphic-circuit agreement >= 2/3 between the female corpus
  and male-corpus/FlyWire findings; failures are listed verbatim for the
  model card.

``pass: null`` plus a recorded reason marks NA, never a silent pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from flygpt.schema import write_result

EPS = 1e-6  # EVAL 1.9
MIN_SEEDS = 3
DS_PASS_FRACTION = 0.80
DS_KILL_FRACTION = 0.50
DIVERGENT_FRACTION_MIN = 0.50
ISO_AGREEMENT_MIN = 2.0 / 3.0


def _emit(
    results_dir: Path | None,
    test_id: str,
    checks: dict[str, Any],
    *,
    inputs: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    passed: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    if results_dir is not None:
        return write_result(
            results_dir, test_id, checks,
            inputs=inputs, metrics=metrics, passed=passed, extra=extra,
        )
    art = {
        "artifact": test_id,
        "inputs": inputs or {},
        "checks": checks,
        "metrics": metrics or {},
        "pass": passed,
    }
    if extra:
        art.update(extra)
    return art


# ---------------------------------------------------------------------------
# T-SEX-1 / T-SEX-2 / T-SEX-3
# ---------------------------------------------------------------------------


def run_t1_dimorphism_set(
    ds_path: Path,
    measured: dict[str, bool],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-SEX-1: >= 80% of the frozen DS items reproduced.

    Items absent from ``measured`` count as NOT reproduced (conservative)
    and are listed under ``missing_measurements``.
    """
    panel = json.loads(Path(ds_path).read_text())
    items = panel.get("items", [])
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {"n_items": len(items)}
    if not items:
        checks["status"] = f"NA: {ds_path} has no items"
        return _emit(results_dir, "T-SEX-1", checks, metrics=metrics, passed=None)

    per_item: dict[str, bool] = {}
    missing: list[str] = []
    for item in items:
        iid = item["id"]
        if iid in measured:
            per_item[iid] = bool(measured[iid])
        else:
            per_item[iid] = False
            missing.append(iid)
    frac = sum(per_item.values()) / len(items)
    metrics.update(
        {
            "fraction_reproduced": frac,
            "n_reproduced": int(sum(per_item.values())),
            "per_item": per_item,
            "missing_measurements": missing,
        }
    )
    checks["fraction_ge_0.80"] = bool(frac >= DS_PASS_FRACTION)
    checks["kill_K4_triggered"] = bool(frac < DS_KILL_FRACTION)
    return _emit(
        results_dir,
        "T-SEX-1",
        checks,
        inputs={"dimorphism_set": str(ds_path), "panel": panel.get("panel")},
        metrics=metrics,
        passed=checks["fraction_ge_0.80"],
    )


def _circuit_seed_stats(seed_rec: dict[str, Any], eps: float) -> tuple[float, int]:
    """One seed of one circuit -> (divergence, direction) per EVAL 1.9."""
    male = np.asarray(seed_rec["male"], dtype=np.float64)
    female = np.asarray(seed_rec["female"], dtype=np.float64)
    if male.ndim == 1:
        male = male[:, None]
        female = female[:, None]
    rm, rf = male.mean(axis=0), female.mean(axis=0)
    sm, sf = male.std(axis=0), female.std(axis=0)
    d_i = np.abs(rm - rf) / (0.5 * (sm + sf) + eps)
    direction = float(np.mean(rm - rf))
    return float(d_i.mean()), int(np.sign(direction))


def run_t2_divergence(
    circuits: list[dict[str, Any]],
    *,
    eps: float = EPS,
    min_seeds: int = MIN_SEEDS,
    results_dir: Path | None = None,
) -> dict:
    """T-SEX-2: dimorphic divergence vs the isomorphic null (EVAL 1.9).

    ``circuits``: ``[{"id": str, "dimorphic": bool, "seeds":
    [{"male": (T,V) rates, "female": (T,V) rates}, ...]}, ...]``.
    """
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {"per_circuit": {}}
    iso_null: list[float] = []
    dimorphic: list[dict[str, Any]] = []
    for c in circuits:
        seeds = c.get("seeds", [])
        divs, dirs = [], []
        for rec in seeds:
            d, sign = _circuit_seed_stats(rec, eps)
            divs.append(d)
            dirs.append(sign)
        divergence = float(np.mean(divs)) if divs else 0.0
        consistent = (
            len(dirs) >= min_seeds and len(set(dirs)) == 1 and dirs[0] != 0
        )
        entry = {
            "divergence": divergence,
            "n_seeds": len(dirs),
            "direction_consistent": bool(consistent),
        }
        metrics["per_circuit"][c["id"]] = entry
        if c.get("dimorphic"):
            dimorphic.append({"id": c["id"], **entry})
        else:
            iso_null.append(divergence)

    n_dim = len(dimorphic)
    if not iso_null or not n_dim:
        reason = []
        if not iso_null:
            reason.append("no isomorphic control circuits")
        if not n_dim:
            reason.append("no dimorphic circuits")
        checks["status"] = f"NA: {', '.join(reason)}"
        return _emit(
            results_dir, "T-SEX-2", checks,
            metrics=metrics, passed=None,
        )

    p95 = float(np.percentile(np.asarray(iso_null), 95))
    metrics["iso_null_p95"] = p95
    n_divergent = 0
    for d in dimorphic:
        divergent = bool(d["divergence"] > p95 and d["direction_consistent"])
        metrics["per_circuit"][d["id"]]["divergent"] = divergent
        n_divergent += int(divergent)
    frac = n_divergent / n_dim
    metrics["n_dimorphic"] = n_dim
    metrics["n_divergent"] = n_divergent
    metrics["fraction_divergent"] = frac
    checks["fraction_divergent_ge_0.50"] = bool(frac >= DIVERGENT_FRACTION_MIN)
    checks["min_seeds_met"] = bool(
        all(d["n_seeds"] >= min_seeds for d in dimorphic)
    )
    return _emit(
        results_dir,
        "T-SEX-2",
        checks,
        metrics=metrics,
        passed=bool(all(checks.values())),
    )


def _has_conditioning(artifact: dict, key: str) -> bool:
    for src in (
        artifact.get("metrics"),
        artifact.get("conditionings"),
        artifact,
    ):
        if isinstance(src, dict) and key in src:
            return True
    return False


def run_t3_confound_report(
    artifact: Any,
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-SEX-3: specimen-conditioned AND sex-conditioned both reported.

    Reporting requirement only (no numeric bar); a missing block fails.
    ``artifact`` accepts a dict or a path to a results JSON.
    """
    if isinstance(artifact, (str, Path)):
        artifact = json.loads(Path(artifact).read_text())
    checks: dict[str, Any] = {
        "sex_conditioned_reported": _has_conditioning(artifact, "sex_conditioned"),
        "specimen_conditioned_reported": _has_conditioning(artifact, "specimen_conditioned"),
    }
    missing = [k for k, v in {
        "sex_conditioned": checks["sex_conditioned_reported"],
        "specimen_conditioned": checks["specimen_conditioned_reported"],
    }.items() if not v]
    metrics = {"missing_conditioning_blocks": missing}
    return _emit(
        results_dir,
        "T-SEX-3",
        checks,
        metrics=metrics,
        passed=bool(all(checks.values())),
    )


# ---------------------------------------------------------------------------
# T-SEX-4 / T-SEX-5 / T-SEX-6
# ---------------------------------------------------------------------------


def _align_rows(align: Any) -> list[dict]:
    """Alignment table -> row dicts (male_type, banc_type, iso_class)."""
    if isinstance(align, (str, Path)):
        p = Path(align)
        import polars as pl

        df = pl.read_parquet(p) if p.suffix == ".parquet" else pl.read_csv(p)
        return df.to_dicts()
    if hasattr(align, "to_dicts"):  # polars DataFrame
        return align.to_dicts()
    return [dict(r) for r in align]


def run_t4_masking_audit(
    align: Any,
    female_types: Any,
    male_types: Any,
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-SEX-4: cross-sex type leaks = 0, exact.

    ``female_types``/``male_types`` are the type sets exposed under each
    conditioning. A male-specific ``male_type`` inside the female set (or
    a female-specific ``banc_type`` inside the male set) is a leak.
    """
    fset = {str(t) for t in female_types}
    mset = {str(t) for t in male_types}
    leaks: list[dict[str, str]] = []
    n_specific = 0
    for row in _align_rows(align):
        cls = row.get("iso_class")
        mt = row.get("male_type")
        bt = row.get("banc_type")
        if cls == "male-specific":
            n_specific += 1
            if mt is not None and str(mt) in fset:
                leaks.append({"type": str(mt), "leaked_into": "female"})
        elif cls == "female-specific":
            n_specific += 1
            if bt is not None and str(bt) in mset:
                leaks.append({"type": str(bt), "leaked_into": "male"})
    checks = {"cross_sex_leaks_zero": not leaks}
    metrics = {"n_leaks": len(leaks), "leaks": leaks,
               "n_sex_specific_rows": n_specific}
    return _emit(
        results_dir, "T-SEX-4", checks, metrics=metrics,
        passed=bool(all(checks.values())),
    )


def _edge_set(edges: Any) -> set[tuple[int, int]]:
    arr = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    return {(int(a), int(b)) for a, b in arr}


def run_t5_graph_swap_audit(
    male_edges: Any,
    female_edges: Any,
    *,
    expected_female_edges: Any | None = None,
    swapped_nodes: Any | None = None,
    results_dir: Path | None = None,
) -> dict:
    """T-SEX-5: the female-conditioned mask IS the BANC mask.

    Checks (whichever inputs are supplied):
      ``female_edges == expected_female_edges`` exactly (the BANC-derived
      mask, 100%), and the male/female symmetric difference is confined
      to edges incident on ``swapped_nodes``.
    """
    mset = _edge_set(male_edges)
    fset = _edge_set(female_edges)
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {
        "n_male_edges": len(mset),
        "n_female_edges": len(fset),
    }
    if expected_female_edges is not None:
        eset = _edge_set(expected_female_edges)
        missing = sorted(eset - fset)
        unexpected = sorted(fset - eset)
        metrics["missing_expected_edges"] = [list(e) for e in missing[:100]]
        metrics["unexpected_edges"] = [list(e) for e in unexpected[:100]]
        checks["female_mask_matches_banc"] = not missing and not unexpected
    if swapped_nodes is not None:
        sn = {int(v) for v in swapped_nodes}
        outside = sorted(e for e in (mset ^ fset) if e[0] not in sn and e[1] not in sn)
        metrics["diff_outside_swapped"] = [list(e) for e in outside[:100]]
        checks["diff_confined_to_swapped_subgraph"] = not outside
    if not checks:
        checks["status"] = "NA: provide expected_female_edges and/or swapped_nodes"
        passed = None
    else:
        passed = bool(all(checks.values()))
    return _emit(results_dir, "T-SEX-5", checks, metrics=metrics, passed=passed)


def run_t6_isomorphic_agreement(
    agreements: Any,
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-SEX-6: >= 2/3 isomorphic-circuit agreement with male/FlyWire.

    ``agreements`` accepts ``{circuit_id: bool}`` or a list of
    ``{"id", "predicted", "reference"}`` records (agree = values equal).
    """
    resolved: dict[str, bool] = {}
    if isinstance(agreements, dict):
        resolved = {str(k): bool(v) for k, v in agreements.items()}
    else:
        for rec in agreements:
            resolved[str(rec["id"])] = bool(rec.get("predicted") == rec.get("reference"))
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {"n_circuits": len(resolved)}
    if not resolved:
        checks["status"] = "NA: no isomorphic-circuit comparisons supplied"
        return _emit(results_dir, "T-SEX-6", checks, metrics=metrics, passed=None)
    failures = sorted(cid for cid, ok in resolved.items() if not ok)
    frac = sum(resolved.values()) / len(resolved)
    metrics["fraction_agree"] = frac
    metrics["documented_failures"] = failures  # verbatim for the model card
    checks["fraction_agree_ge_2_3"] = bool(frac >= ISO_AGREEMENT_MIN - 1e-9)
    return _emit(
        results_dir,
        "T-SEX-6",
        checks,
        metrics=metrics,
        passed=checks["fraction_agree_ge_2_3"],
    )


# ---------------------------------------------------------------------------
# Dispatcher entry points (flygpt.evals.RUNNERS names). Required inputs
# arrive via kwargs; absent inputs record a documented NA - never a crash.
# ---------------------------------------------------------------------------


def _na(results_dir: Path, test_id: str, reason: str,
        inputs: dict[str, Any] | None = None) -> dict:
    return _emit(
        results_dir, test_id, {"status": f"NA: {reason}"},
        inputs=inputs, passed=None,
    )


def run_t_sex_1(results_dir: Path, *, ds_path: Path | None = None,
                measured: Any = None, **_ignored: Any) -> dict:
    """Dispatcher entry for T-SEX-1 (see ``run_t1_dimorphism_set``)."""
    path = Path(ds_path) if ds_path is not None else Path("evals/dimorphism_set.json")
    if measured is None:
        return _na(results_dir, "T-SEX-1", "no measured DS outcomes injected",
                   inputs={"dimorphism_set": str(path)})
    return run_t1_dimorphism_set(path, measured, results_dir=results_dir)


def run_t_sex_2(results_dir: Path, *, circuits: Any = None,
                **_ignored: Any) -> dict:
    """Dispatcher entry for T-SEX-2 (see ``run_t2_divergence``)."""
    if circuits is None:
        return _na(results_dir, "T-SEX-2", "no circuit rate artifacts injected")
    return run_t2_divergence(circuits, results_dir=results_dir)


def run_t_sex_3(results_dir: Path, *, artifact: Any = None,
                **_ignored: Any) -> dict:
    """Dispatcher entry for T-SEX-3 (see ``run_t3_confound_report``)."""
    if artifact is None:
        return _na(results_dir, "T-SEX-3", "no conditioning artifact injected")
    return run_t3_confound_report(artifact, results_dir=results_dir)


def run_t_sex_4(results_dir: Path, *, align: Any = None,
                female_types: Any = None, male_types: Any = None,
                **_ignored: Any) -> dict:
    """Dispatcher entry for T-SEX-4 (see ``run_t4_masking_audit``)."""
    if align is None or female_types is None or male_types is None:
        return _na(results_dir, "T-SEX-4",
                   "no alignment table / conditioning type sets injected")
    return run_t4_masking_audit(align, female_types, male_types,
                                results_dir=results_dir)


def run_t_sex_5(results_dir: Path, *, male_edges: Any = None,
                female_edges: Any = None, expected_female_edges: Any = None,
                swapped_nodes: Any = None, **_ignored: Any) -> dict:
    """Dispatcher entry for T-SEX-5 (see ``run_t5_graph_swap_audit``)."""
    if male_edges is None or female_edges is None:
        return _na(results_dir, "T-SEX-5", "no conditioning edge masks injected")
    return run_t5_graph_swap_audit(
        male_edges, female_edges,
        expected_female_edges=expected_female_edges,
        swapped_nodes=swapped_nodes,
        results_dir=results_dir,
    )


def run_t_sex_6(results_dir: Path, *, agreements: Any = None,
                **_ignored: Any) -> dict:
    """Dispatcher entry for T-SEX-6 (see ``run_t6_isomorphic_agreement``)."""
    if agreements is None:
        return _na(results_dir, "T-SEX-6",
                   "no isomorphic-circuit agreements injected")
    return run_t6_isomorphic_agreement(agreements, results_dir=results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dimorphism-set", type=Path,
                    default=Path("evals/dimorphism_set.json"))
    ap.add_argument("--measured", type=Path, default=None,
                    help='JSON {"DS-1": true, ...} for T-SEX-1')
    ap.add_argument("--circuits", type=Path, default=None,
                    help="JSON circuit list for T-SEX-2")
    ap.add_argument("--artifact", type=Path, default=None,
                    help="results JSON audited by T-SEX-3")
    ap.add_argument("--align", type=Path, default=None,
                    help="alignment table (csv/parquet) for T-SEX-4")
    ap.add_argument("--female-types", type=str, default="",
                    help="comma-separated types under female conditioning")
    ap.add_argument("--male-types", type=str, default="",
                    help="comma-separated types under male conditioning")
    ap.add_argument("--agreements", type=Path, default=None,
                    help="JSON agreements for T-SEX-6")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    ran = 0
    if args.measured is not None:
        measured = json.loads(args.measured.read_text())
        art = run_t1_dimorphism_set(args.dimorphism_set, measured,
                                    results_dir=args.results_dir)
        print(f"T-SEX-1: pass={art['pass']}")
        ran += 1
    if args.circuits is not None:
        circuits = json.loads(args.circuits.read_text())
        circuits = [
            {**c, "seeds": [
                {"male": np.asarray(s["male"]), "female": np.asarray(s["female"])}
                for s in c["seeds"]
            ]}
            for c in circuits
        ]
        art = run_t2_divergence(circuits, results_dir=args.results_dir)
        print(f"T-SEX-2: pass={art['pass']}")
        ran += 1
    if args.artifact is not None:
        art = run_t3_confound_report(args.artifact, results_dir=args.results_dir)
        print(f"T-SEX-3: pass={art['pass']}")
        ran += 1
    if args.align is not None:
        art = run_t4_masking_audit(
            args.align,
            {t for t in args.female_types.split(",") if t},
            {t for t in args.male_types.split(",") if t},
            results_dir=args.results_dir,
        )
        print(f"T-SEX-4: pass={art['pass']}")
        ran += 1
    if args.agreements is not None:
        art = run_t6_isomorphic_agreement(
            json.loads(args.agreements.read_text()),
            results_dir=args.results_dir,
        )
        print(f"T-SEX-6: pass={art['pass']}")
        ran += 1
    if not ran:
        print("nothing to run - pass --measured/--circuits/--artifact at minimum")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
