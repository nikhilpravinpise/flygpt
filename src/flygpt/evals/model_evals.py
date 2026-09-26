"""Model-eval runners T-MODEL-1..11 (EVAL.md section 5, T-MODEL table).

These orchestrate over trained checkpoints, prediction artifacts, and
control-run scores - every ``run_*`` accepts injected artifact paths or
in-memory arrays so the same code runs on synthetic fixtures and on the
full corpus. Nothing here trains a model; these consume artifacts.

Prediction artifacts (T-MODEL-1/2) are ``.npz`` or ``.parquet`` or dicts
with per-bin arrays ``y_true`` (event counts) and ``y_pred`` (event
probabilities) of shape ``(n_bins, n_types)`` plus per-bin meta columns
``split`` and ``seed`` (``family`` optional). Events binarize at the
fixed 0.5 threshold (EVAL.md 1.1); F1 macros use the 1.2 inclusion rule;
the per-type marginal ("majority") baseline and the all-zero baseline of
1.7 are recomputed from ``y_true`` on the same split rows.

Control comparisons (T-MODEL-3..7, 10, 11) take per-seed metric series
``{"real": [...], "rewired": [...], ...}`` of macro event-F1 values.
Margins are paired per seed when series have equal length, else a Welch
margin CI is used. Kill flags follow EVAL.md section 6: K1 (T-MODEL-1),
K2 (T-MODEL-3, primary = rewired control), K3 (T-MODEL-6).

Metrics come from ``flygpt.evals.common`` when importable (the shared
implementation); identical local fallbacks keep the module runnable
standalone. ``pass: null`` plus a recorded reason marks NA, never a
silent pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from flygpt.schema import write_result

EVENT_THRESHOLD = 0.5  # EVAL.md 1.1 - fixed, never tuned
FORBIDDEN_METRIC_KEYS = {"accuracy", "raw_accuracy", "acc", "top1_accuracy"}


# ---------------------------------------------------------------------------
# Shared-metric access: flygpt.evals.common with identical local fallbacks
# ---------------------------------------------------------------------------


def _common():
    """Return ``flygpt.evals.common`` when it exists, else ``None``."""
    try:
        from flygpt.evals import common
    except ImportError:
        return None
    return common


def _binarize(x: np.ndarray) -> np.ndarray:
    """EVAL 1.1: event iff value >= 0.5 (counts and probs share this)."""
    return (np.asarray(x) >= EVENT_THRESHOLD).astype(np.uint8)


def _event_f1(y_true_b: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """EVAL 1.2 event-F1: per type over concatenated bins.

    A type is included iff it has >= 1 true OR >= 1 predicted event.
    """
    yt = np.asarray(y_true_b)
    yp = np.asarray(y_pred_b)
    if yt.ndim == 1:
        yt = yt[:, None]
        yp = yp[:, None]
    tp = ((yt == 1) & (yp == 1)).sum(axis=0)
    fp = ((yt == 0) & (yp == 1)).sum(axis=0)
    fn = ((yt == 1) & (yp == 0)).sum(axis=0)
    included = (yt.sum(axis=0) + yp.sum(axis=0)) > 0
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    per_type = [float(v) for v, inc in zip(f1, included, strict=True) if inc]
    macro = float(np.mean(per_type)) if per_type else 0.0
    tp_t = tp.sum()
    micro = float(2 * tp_t / (2 * tp_t + fp.sum() + fn.sum())) if (2 * tp_t + fp.sum() + fn.sum()) else 0.0
    return {
        "macro": macro,
        "micro": micro,
        "per_type": per_type,
        "n_included": int(included.sum()),
    }


def _majority_f1(y_true_b: np.ndarray) -> dict:
    """Per-type marginal-rate ("majority") baseline F1 (EVAL 1.7)."""
    yt = np.asarray(y_true_b)
    if yt.ndim == 1:
        yt = yt[:, None]
    maj = (yt.mean(axis=0) >= EVENT_THRESHOLD).astype(np.uint8)
    yp = np.broadcast_to(maj, yt.shape).copy()
    return _event_f1(yt, yp)


def _allzero_f1(y_true_b: np.ndarray) -> dict:
    yt = np.asarray(y_true_b)
    return _event_f1(yt, np.zeros_like(yt))


def _rate_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """EVAL 1.3 macro Pearson r; types need variance on both sides."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.ndim == 1:
        yt = yt[:, None]
        yp = yp[:, None]
    rs: list[float] = []
    n_excl = 0
    for i in range(yt.shape[1]):
        a, b = yt[:, i], yp[:, i]
        if a.std() == 0.0 or b.std() == 0.0:
            n_excl += 1
            continue
        rs.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "macro": float(np.mean(rs)) if rs else None,
        "n_qualifying": len(rs),
        "n_excluded": n_excl,
        "per_type": rs,
    }


def _ci_mean(x: Any) -> dict:
    """Student-t 95% CI of the mean (n-1). Degenerate for n<2."""
    from scipy.stats import t as t_dist

    arr = np.asarray(list(x), dtype=np.float64).ravel()
    n = arr.size
    mean = float(arr.mean()) if n else None
    if n < 2:
        return {"mean": mean, "lo": mean, "hi": mean, "n": int(n)}
    se = float(arr.std(ddof=1) / np.sqrt(n))
    tc = float(t_dist.ppf(0.975, n - 1))
    return {"mean": mean, "lo": mean - tc * se, "hi": mean + tc * se, "n": int(n)}


def _f1_macro(com, yt_b, yp_b) -> float:
    res = com.event_f1(yt_b, yp_b) if com is not None else _event_f1(yt_b, yp_b)
    return float(res["macro"])


def _maj_macro(com, yt_b) -> float:
    """Majority-baseline macro event-F1 on the eval split's own marginals."""
    if com is not None:
        return float(com.majority_baseline(yt_b)["event_f1"]["macro"])
    return float(_majority_f1(yt_b)["macro"])


def _allzero_macro(com, yt_b) -> float:
    if com is not None:
        return float(com.allzero_baseline(yt_b)["event_f1"]["macro"])
    return float(_allzero_f1(yt_b)["macro"])


def _pearson_macro(com, yt, yp):
    """Macro rate Pearson r; None when no type qualifies (zero variance)."""
    res = com.rate_pearson(yt, yp) if com is not None else _rate_pearson(yt, yp)
    m = res.get("macro")
    if m is None or not np.isfinite(float(m)):
        return None
    return float(m)


def _ci_of(com, x) -> dict:
    """Normalize a ci_mean result (dict or (mean, lo, hi)) to a dict."""
    res = (
        com.ci_mean(np.asarray(list(x), dtype=np.float64))
        if com is not None
        else _ci_mean(x)
    )
    if isinstance(res, dict):
        return {
            "mean": res.get("mean"),
            "lo": res.get("lo", res.get("lower")),
            "hi": res.get("hi", res.get("upper")),
            "n": res.get("n", len(list(x))),
        }
    return {"mean": float(res[0]), "lo": float(res[1]), "hi": float(res[2]), "n": len(list(x))}


def _margin_ci(com, a: Any, b: Any) -> dict:
    """CI of mean(a) - mean(b): paired per-seed when lengths match."""
    from scipy.stats import t as t_dist

    xa = np.asarray(list(a), dtype=np.float64).ravel()
    xb = np.asarray(list(b), dtype=np.float64).ravel()
    if xa.size == xb.size:
        return _ci_of(com, xa - xb)
    mean = float(xa.mean() - xb.mean())
    va, vb = float(xa.var(ddof=1)), float(xb.var(ddof=1))
    se = float(np.sqrt(va / xa.size + vb / xb.size))
    if se == 0.0:
        return {"mean": mean, "lo": mean, "hi": mean, "n": xa.size + xb.size}
    df = (va / xa.size + vb / xb.size) ** 2 / (
        (va / xa.size) ** 2 / (xa.size - 1) + (vb / xb.size) ** 2 / (xb.size - 1)
    )
    tc = float(t_dist.ppf(0.975, df))
    return {"mean": mean, "lo": mean - tc * se, "hi": mean + tc * se, "n": int(xa.size + xb.size)}


# ---------------------------------------------------------------------------
# Prediction artifact loading
# ---------------------------------------------------------------------------


def _load_predictions(src: Any) -> dict[str, np.ndarray]:
    """Accept dict / .npz / .parquet prediction artifacts -> arrays."""
    if isinstance(src, dict):
        return {k: np.asarray(v) for k, v in src.items()}
    p = Path(src)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        return {k: z[k] for k in z.files}
    if p.suffix == ".json":
        return {k: np.asarray(v) for k, v in json.loads(p.read_text()).items()}
    if p.suffix == ".parquet":
        return _parquet_to_matrices(p)
    raise ValueError(f"unsupported predictions artifact: {p}")


def _parquet_to_matrices(path: Path) -> dict[str, np.ndarray]:
    """Long-format parquet -> canonical (n_bins, n_types) matrices.

    Expected columns: ``seed, split, row, unit, y_true, y_pred`` plus
    optional ``family`` (broadcast per seed/split group).
    """
    df = pl.read_parquet(path)
    need = {"seed", "split", "row", "unit", "y_true", "y_pred"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    yt_parts, yp_parts = [], []
    seeds, splits, fams = [], [], []
    for key, grp in df.group_by(["seed", "split"], maintain_order=True):
        seed_v, split_v = key[0], key[1]
        rows = np.sort(grp["row"].unique().to_numpy())
        units = np.sort(grp["unit"].unique().to_numpy())
        rmap = {r: i for i, r in enumerate(rows)}
        umap = {u: i for i, u in enumerate(units)}
        yt = np.zeros((rows.size, units.size))
        yp = np.zeros((rows.size, units.size))
        for r, u, a, b in grp.select(["row", "unit", "y_true", "y_pred"]).iter_rows():
            yt[rmap[r], umap[u]] = a
            yp[rmap[r], umap[u]] = b
        yt_parts.append(yt)
        yp_parts.append(yp)
        fam = grp["family"][0] if "family" in grp.columns else ""
        n_rows = rows.size
        seeds.extend([seed_v] * n_rows)
        splits.extend([split_v] * n_rows)
        fams.extend([fam] * n_rows)
    return {
        "y_true": np.vstack(yt_parts),
        "y_pred": np.vstack(yp_parts),
        "seed": np.asarray(seeds),
        "split": np.asarray(splits, dtype=object),
        "family": np.asarray(fams, dtype=object),
    }


def _split_margins(
    com, data: dict[str, np.ndarray], split_name: str
) -> tuple[list[float], list[float], dict]:
    """Per-seed (model F1 - majority F1) margins and Pearson r on a split."""
    yt_all = np.asarray(data["y_true"])
    yp_all = np.asarray(data["y_pred"])
    split = np.asarray(data.get("split", np.full(yt_all.shape[0], split_name)), dtype=object)
    seed = np.asarray(data.get("seed", np.zeros(yt_all.shape[0])), dtype=object)
    mask = np.array([s == split_name for s in split])
    if not mask.any():
        return [], [], {"reason": f"no rows with split == {split_name!r}"}
    yt, yp = yt_all[mask], yp_all[mask]
    seeds = np.unique(seed[mask])
    margins, rs = [], []
    for s in seeds:
        sm = np.array([v == s for v in seed[mask]])
        ytb, ypb = _binarize(yt[sm]), _binarize(yp[sm])
        margins.append(float(_f1_macro(com, ytb, ypb) - _maj_macro(com, ytb)))
        r = _pearson_macro(com, yt[sm], yp[sm])
        if r is not None:
            rs.append(r)
    return margins, rs, {"n_seeds": int(seeds.size), "n_bins": int(yt.shape[0])}


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
# T-MODEL-1 / T-MODEL-2: held-out stimulus families / cell types
# ---------------------------------------------------------------------------


def run_t1_heldout_families(
    pred: Any,
    *,
    split_name: str = "stimulus_holdout",
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-1: macro event-F1 >= majority + 0.10, CI lo > +0.05.

    A 95% CI lower bound <= +0.05 raises kill condition K1 (lookup table).
    """
    com = _common()
    data = _load_predictions(pred)
    margins, _rs, meta = _split_margins(com, data, split_name)
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if not margins:
        checks["status"] = f"NA: {meta['reason']}"
        passed = None
    else:
        ci = _ci_of(com, margins)
        metrics = {
            "margin_mean": ci["mean"],
            "margin_ci_lo": ci["lo"],
            "margin_ci_hi": ci["hi"],
            "per_seed_margins": margins,
            **meta,
        }
        checks["margin_mean_ge_0.10"] = bool(ci["mean"] >= 0.10)
        checks["margin_ci_lo_gt_0.05"] = bool(ci["lo"] > 0.05)
        checks["kill_K1_triggered"] = bool(ci["lo"] <= 0.05)
        passed = bool(checks["margin_mean_ge_0.10"] and checks["margin_ci_lo_gt_0.05"])
    return _emit(
        results_dir, "T-MODEL-1", checks,
        inputs={"split": split_name}, metrics=metrics, passed=passed,
    )


def run_t2_heldout_types(
    pred: Any,
    *,
    split_name: str = "celltype_holdout",
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-2: margin >= +0.05 AND macro rate Pearson >= 0.20."""
    com = _common()
    data = _load_predictions(pred)
    margins, rs, meta = _split_margins(com, data, split_name)
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if not margins:
        checks["status"] = f"NA: {meta['reason']}"
        passed = None
    else:
        ci = _ci_of(com, margins)
        r_macro = float(np.mean(rs)) if rs else None
        metrics = {
            "margin_mean": ci["mean"],
            "margin_ci_lo": ci["lo"],
            "margin_ci_hi": ci["hi"],
            "rate_pearson_macro": r_macro,
            "per_seed_margins": margins,
            **meta,
        }
        checks["margin_mean_ge_0.05"] = bool(ci["mean"] >= 0.05)
        checks["rate_pearson_ge_0.20"] = bool(r_macro is not None and r_macro >= 0.20)
        passed = bool(all(checks.values()))
    return _emit(
        results_dir, "T-MODEL-2", checks,
        inputs={"split": split_name}, metrics=metrics, passed=passed,
    )


# ---------------------------------------------------------------------------
# T-MODEL-3..7, 10, 11: control-comparison runners over per-seed scores
# ---------------------------------------------------------------------------


def _need_series(scores: dict, keys: tuple[str, ...]) -> str | None:
    missing = [k for k in keys if k not in scores or len(scores[k]) == 0]
    return f"missing/empty score series: {missing}" if missing else None


def run_t3_wrong_connectome(
    scores: dict[str, Any],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-3: real mask beats rewired AND wrong-animal controls.

    Each margin must be >= 0.05 macro event-F1 with a CI separated from
    0. K2 fires when real - rewired (primary control) has CI lo <= 0.
    """
    com = _common()
    reason = _need_series(scores, ("real", "rewired", "wrong_animal"))
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if reason:
        checks["status"] = f"NA: {reason}"
        passed = None
    else:
        m_rw = _margin_ci(com, scores["real"], scores["rewired"])
        m_wa = _margin_ci(com, scores["real"], scores["wrong_animal"])
        metrics = {
            "margin_rewired": m_rw,
            "margin_wrong_animal": m_wa,
            "real_scores": list(scores["real"]),
            "rewired_scores": list(scores["rewired"]),
            "wrong_animal_scores": list(scores["wrong_animal"]),
        }
        checks["margin_rewired_ge_0.05"] = bool(m_rw["mean"] >= 0.05)
        checks["margin_rewired_ci_separated"] = bool(m_rw["lo"] > 0.0)
        checks["margin_wrong_animal_ge_0.05"] = bool(m_wa["mean"] >= 0.05)
        checks["margin_wrong_animal_ci_separated"] = bool(m_wa["lo"] > 0.0)
        checks["kill_K2_triggered"] = bool(m_rw["lo"] <= 0.0)
        passed = bool(all(v for k, v in checks.items() if not k.startswith("kill")))
    return _emit(results_dir, "T-MODEL-3", checks, metrics=metrics, passed=passed)


def run_t4_permuted_labels(
    scores: dict[str, Any],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-4: real node identities >= permuted labels + 0.05."""
    com = _common()
    reason = _need_series(scores, ("real", "permuted"))
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if reason:
        checks["status"] = f"NA: {reason}"
        passed = None
    else:
        m = _margin_ci(com, scores["real"], scores["permuted"])
        metrics = {"margin": m, "real_scores": list(scores["real"]),
                   "permuted_scores": list(scores["permuted"])}
        checks["margin_ge_0.05"] = bool(m["mean"] >= 0.05)
        passed = bool(all(checks.values()))
    return _emit(results_dir, "T-MODEL-4", checks, metrics=metrics, passed=passed)


def run_t5_dense_baseline(
    scores: dict[str, Any],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-5: masked >= dense - 0.01 (ship gate); +0.02 CI-separated
    upgrades the architecture claim (C5)."""
    com = _common()
    reason = _need_series(scores, ("masked", "dense"))
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if reason:
        checks["status"] = f"NA: {reason}"
        passed = None
    else:
        m = _margin_ci(com, scores["masked"], scores["dense"])
        metrics = {"margin": m, "masked_scores": list(scores["masked"]),
                   "dense_scores": list(scores["dense"])}
        checks["margin_ge_minus_0.01"] = bool(m["mean"] >= -0.01)
        checks["upgrade_C5_plus_0.02"] = bool(m["mean"] >= 0.02 and m["lo"] > 0.0)
        passed = checks["margin_ge_minus_0.01"]
    return _emit(results_dir, "T-MODEL-5", checks, metrics=metrics, passed=passed)


def run_t6_linear_baseline(
    scores: dict[str, Any],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-6: model >= BANC linear-influence predictor + 0.05.

    Margin <= +0.01 raises kill condition K3 ("linear suffices").
    """
    com = _common()
    reason = _need_series(scores, ("model", "linear"))
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if reason:
        checks["status"] = f"NA: {reason}"
        passed = None
    else:
        m = _margin_ci(com, scores["model"], scores["linear"])
        metrics = {"margin": m, "model_scores": list(scores["model"]),
                   "linear_scores": list(scores["linear"])}
        checks["margin_ge_0.05"] = bool(m["mean"] >= 0.05)
        checks["kill_K3_triggered"] = bool(m["mean"] <= 0.01)
        passed = checks["margin_ge_0.05"]
    return _emit(results_dir, "T-MODEL-6", checks, metrics=metrics, passed=passed)


def run_t7_reservoir_baseline(
    scores: dict[str, Any],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-7: model >= flybrain Trace/Readout reservoir + 0.05."""
    com = _common()
    reason = _need_series(scores, ("model", "reservoir"))
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if reason:
        checks["status"] = f"NA: {reason}"
        passed = None
    else:
        m = _margin_ci(com, scores["model"], scores["reservoir"])
        metrics = {"margin": m, "model_scores": list(scores["model"]),
                   "reservoir_scores": list(scores["reservoir"])}
        checks["margin_ge_0.05"] = bool(m["mean"] >= 0.05)
        passed = checks["margin_ge_0.05"]
    return _emit(results_dir, "T-MODEL-7", checks, metrics=metrics, passed=passed)


# ---------------------------------------------------------------------------
# T-MODEL-8 / 9 / 10 / 11
# ---------------------------------------------------------------------------


def _scan_forbidden_keys(obj: Any, path: str = "") -> list[str]:
    """Find metric keys violating EVAL 1.7 (raw accuracy is never reported)."""
    hits: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in FORBIDDEN_METRIC_KEYS:
                hits.append(f"{path}{k}")
            hits.extend(_scan_forbidden_keys(v, f"{path}{k}."))
    return hits


def run_t8_silence_guard(
    report: dict[str, Any],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-8: the silence-robustness sanity invariant (EVAL 1.7).

    ``report`` maps split names to either ``{"y_true": counts}`` (the two
    baselines are recomputed) or precomputed ``{"allzero_f1": x,
    "majority_f1": y}``. On every split the all-zero predictor must score
    <= the majority baseline; a violation means the metric pipeline is
    broken and the eval is void. Any ``accuracy``-style metric key in the
    report is a 1.7 violation.
    """
    com = _common()
    splits = report.get("splits", report)
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {"per_split": {}}
    if not isinstance(splits, dict) or not splits:
        checks["status"] = "NA: report has no splits"
        passed = None
    else:
        violations: list[str] = []
        for name, entry in splits.items():
            if not isinstance(entry, dict):
                violations.append(f"{name}: unparseable entry")
                continue
            if "y_true" in entry:
                yt = _binarize(np.asarray(entry["y_true"]))
                az = _allzero_macro(com, yt)
                mj = _maj_macro(com, yt)
            else:
                az = entry.get("allzero_f1")
                mj = entry.get("majority_f1")
            metrics["per_split"][name] = {"allzero_f1": az, "majority_f1": mj}
            if az is None or mj is None:
                violations.append(f"{name}: baseline values missing")
            else:
                # silence_guard convention: NaN compares as -inf, so
                # NaN-vs-finite is a violation, NaN-vs-NaN holds.
                az_v = float(az) if np.isfinite(float(az)) else -np.inf
                mj_v = float(mj) if np.isfinite(float(mj)) else -np.inf
                if az_v > mj_v + 1e-9:
                    violations.append(f"{name}: allzero {az} > majority {mj}")
        forbidden = _scan_forbidden_keys(report)
        checks["allzero_le_majority_all_splits"] = not violations
        checks["no_accuracy_metric_keys"] = not forbidden
        metrics["violations"] = violations
        metrics["forbidden_metric_keys"] = forbidden
        passed = bool(all(checks.values()))
    return _emit(results_dir, "T-MODEL-8", checks, metrics=metrics, passed=passed)


def run_t9_mask_direction(
    records: list[dict[str, Any]],
    *,
    rel_eps: float = 1e-6,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-9: mask direction on single-neuron interventions.

    Each record carries ``changed_t1`` (unit ids changed at t+1),
    ``postsyn_partners`` (the intervened neuron's postsynaptic set), and
    same-step changes as ``delta_t`` (relative magnitudes) or
    ``max_abs_rel_t``. >= 99% of changed t+1 units must be postsynaptic
    partners and all same-step changes must be <= rel_eps.
    """
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {"n_interventions": len(records)}
    if not records:
        checks["status"] = "NA: no intervention records"
        passed = None
    else:
        inside = 0
        total = 0
        t_viol = 0
        outside_units: list[int] = []
        for rec in records:
            changed = [int(u) for u in rec.get("changed_t1", [])]
            post = {int(u) for u in rec.get("postsyn_partners", [])}
            inside += sum(1 for u in changed if u in post)
            outside_units.extend(u for u in changed if u not in post)
            total += len(changed)
            deltas = rec.get("delta_t")
            if deltas is not None:
                arr = np.atleast_1d(np.asarray(deltas, dtype=np.float64))
                mx = float(np.max(np.abs(arr))) if arr.size else 0.0
            else:
                mx = float(rec.get("max_abs_rel_t", 0.0))
            t_viol += int(mx > rel_eps)
        frac = inside / total if total else 1.0
        metrics.update(
            {
                "frac_changed_t1_postsyn": frac,
                "n_changed_t1": total,
                "n_outside_postsyn": total - inside,
                "outside_postsyn_units": outside_units[:100],
                "n_same_step_violations": t_viol,
                "rel_eps": rel_eps,
            }
        )
        checks["frac_postsyn_ge_0.99"] = bool(frac >= 0.99)
        checks["no_same_step_changes"] = bool(t_viol == 0)
        passed = bool(all(checks.values()))
    return _emit(results_dir, "T-MODEL-9", checks, metrics=metrics, passed=passed)


def run_t10_gnn_headtohead(
    scores: dict[str, Any],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-10: transformer vs GNN on identical splits, head-to-head.

    A reporting check - both numbers are recorded. If the GNN wins by
    > 0.02 macro event-F1 the artifact flags ``gnn_win_gt_0.02`` (ship
    the GNN and adjust the story, per EVAL).
    """
    com = _common()
    reason = _need_series(scores, ("transformer", "gnn"))
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if reason:
        checks["status"] = f"NA: {reason}"
        passed = None
    else:
        tr = _ci_of(com, scores["transformer"])
        gn = _ci_of(com, scores["gnn"])
        m = _margin_ci(com, scores["gnn"], scores["transformer"])
        metrics = {"transformer": tr, "gnn": gn, "margin_gnn_minus_transformer": m}
        checks["reported_both"] = True
        checks["gnn_win_gt_0.02"] = bool(m["mean"] > 0.02)
        passed = True
    return _emit(results_dir, "T-MODEL-10", checks, metrics=metrics, passed=passed)


def run_t11_shuffled_weights(
    scores: dict[str, Any],
    *,
    results_dir: Path | None = None,
) -> dict:
    """T-MODEL-11: real-corpus model >= shuffled-slice model + 0.05."""
    com = _common()
    reason = _need_series(scores, ("real", "shuffled"))
    checks: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if reason:
        checks["status"] = f"NA: {reason}"
        passed = None
    else:
        m = _margin_ci(com, scores["real"], scores["shuffled"])
        metrics = {"margin": m, "real_scores": list(scores["real"]),
                   "shuffled_scores": list(scores["shuffled"])}
        checks["margin_ge_0.05"] = bool(m["mean"] >= 0.05)
        passed = checks["margin_ge_0.05"]
    return _emit(results_dir, "T-MODEL-11", checks, metrics=metrics, passed=passed)


# ---------------------------------------------------------------------------
# Aggregation: results/*.json -> CLAIMS evidence table shape
# ---------------------------------------------------------------------------


def aggregate_evals(results_dir: Path) -> dict[str, Any]:
    """Scan ``results/*.json`` into the evidence-table shape T-SHIP-2 and
    CLAIMS.md consume: one entry per artifact with its pass verdict and
    headline metrics."""
    results_dir = Path(results_dir)
    artifacts: dict[str, Any] = {}
    n_pass = n_fail = n_na = 0
    for path in sorted(results_dir.glob("*.json")):
        try:
            art = json.loads(path.read_text())
        except Exception as exc:  # a malformed artifact is a finding, not a skip
            artifacts[path.stem] = {"error": f"unreadable: {exc}"}
            n_fail += 1
            continue
        verdict = art.get("pass")
        artifacts[art.get("artifact", path.stem)] = {
            "file": path.name,
            "schema": art.get("schema"),
            "created_utc": art.get("created_utc"),
            "pass": verdict,
            "checks": art.get("checks", {}),
            "metrics": art.get("metrics", {}),
        }
        if verdict is True:
            n_pass += 1
        elif verdict is False:
            n_fail += 1
        else:
            n_na += 1
    return {
        "results_dir": str(results_dir),
        "n_artifacts": len(artifacts),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_na": n_na,
        "artifacts": artifacts,
    }


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


def run_t_model_1(results_dir: Path, *, pred: Any = None,
                  split_name: str = "stimulus_holdout",
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-1 (see ``run_t1_heldout_families``)."""
    if pred is None:
        return _na(results_dir, "T-MODEL-1", "no predictions artifact injected")
    return run_t1_heldout_families(pred, split_name=split_name,
                                   results_dir=results_dir)


def run_t_model_2(results_dir: Path, *, pred: Any = None,
                  split_name: str = "celltype_holdout",
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-2 (see ``run_t2_heldout_types``)."""
    if pred is None:
        return _na(results_dir, "T-MODEL-2", "no predictions artifact injected")
    return run_t2_heldout_types(pred, split_name=split_name,
                                results_dir=results_dir)


def run_t_model_3(results_dir: Path, *, scores: Any = None,
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-3 (see ``run_t3_wrong_connectome``)."""
    if scores is None:
        return _na(results_dir, "T-MODEL-3", "no control score series injected")
    return run_t3_wrong_connectome(scores, results_dir=results_dir)


def run_t_model_4(results_dir: Path, *, scores: Any = None,
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-4 (see ``run_t4_permuted_labels``)."""
    if scores is None:
        return _na(results_dir, "T-MODEL-4", "no control score series injected")
    return run_t4_permuted_labels(scores, results_dir=results_dir)


def run_t_model_5(results_dir: Path, *, scores: Any = None,
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-5 (see ``run_t5_dense_baseline``)."""
    if scores is None:
        return _na(results_dir, "T-MODEL-5", "no dense-baseline score series injected")
    return run_t5_dense_baseline(scores, results_dir=results_dir)


def run_t_model_6(results_dir: Path, *, scores: Any = None,
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-6 (see ``run_t6_linear_baseline``)."""
    if scores is None:
        return _na(results_dir, "T-MODEL-6", "no linear-baseline score series injected")
    return run_t6_linear_baseline(scores, results_dir=results_dir)


def run_t_model_7(results_dir: Path, *, scores: Any = None,
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-7 (see ``run_t7_reservoir_baseline``)."""
    if scores is None:
        return _na(results_dir, "T-MODEL-7", "no reservoir-baseline score series injected")
    return run_t7_reservoir_baseline(scores, results_dir=results_dir)


def run_t_model_8(results_dir: Path, *, report: Any = None,
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-8 (see ``run_t8_silence_guard``)."""
    if report is None:
        return _na(results_dir, "T-MODEL-8", "no split report injected")
    return run_t8_silence_guard(report, results_dir=results_dir)


def run_t_model_9(results_dir: Path, *, records: Any = None,
                  **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-9 (see ``run_t9_mask_direction``)."""
    if records is None:
        return _na(results_dir, "T-MODEL-9", "no intervention records injected")
    return run_t9_mask_direction(records, results_dir=results_dir)


def run_t_model_10(results_dir: Path, *, scores: Any = None,
                   **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-10 (see ``run_t10_gnn_headtohead``)."""
    if scores is None:
        return _na(results_dir, "T-MODEL-10", "no head-to-head score series injected")
    return run_t10_gnn_headtohead(scores, results_dir=results_dir)


def run_t_model_11(results_dir: Path, *, scores: Any = None,
                   **_ignored: Any) -> dict:
    """Dispatcher entry for T-MODEL-11 (see ``run_t11_shuffled_weights``)."""
    if scores is None:
        return _na(results_dir, "T-MODEL-11", "no shuffled-corpus score series injected")
    return run_t11_shuffled_weights(scores, results_dir=results_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", type=Path, default=None,
                    help="predictions artifact (.npz/.parquet) for T-MODEL-1/2")
    ap.add_argument("--stim-split", default="stimulus_holdout")
    ap.add_argument("--type-split", default="celltype_holdout")
    ap.add_argument("--metrics-json", type=Path, default=None,
                    help='JSON {"T-MODEL-3": {"real": [...], "rewired": [...], ...}}')
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args(argv)

    if args.aggregate_only:
        table = aggregate_evals(args.results_dir)
        print(json.dumps({"n_artifacts": table["n_artifacts"],
                          "n_pass": table["n_pass"], "n_fail": table["n_fail"]}, indent=2))
        return 0

    ran = 0
    if args.pred is not None:
        for art in (
            run_t1_heldout_families(args.pred, split_name=args.stim_split,
                                    results_dir=args.results_dir),
            run_t2_heldout_types(args.pred, split_name=args.type_split,
                                 results_dir=args.results_dir),
        ):
            print(f"{art['artifact']}: pass={art['pass']}")
            ran += 1

    if args.metrics_json is not None:
        suites = json.loads(args.metrics_json.read_text())
        runners = {
            "T-MODEL-3": run_t3_wrong_connectome,
            "T-MODEL-4": run_t4_permuted_labels,
            "T-MODEL-5": run_t5_dense_baseline,
            "T-MODEL-6": run_t6_linear_baseline,
            "T-MODEL-7": run_t7_reservoir_baseline,
            "T-MODEL-10": run_t10_gnn_headtohead,
            "T-MODEL-11": run_t11_shuffled_weights,
        }
        for tid, scores in suites.items():
            fn = runners.get(tid)
            if fn is None:
                print(f"skip {tid}: no runner")
                continue
            art = fn(scores, results_dir=args.results_dir)
            print(f"{tid}: pass={art['pass']}")
            ran += 1

    if not ran:
        print("nothing to run - pass --pred and/or --metrics-json")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
