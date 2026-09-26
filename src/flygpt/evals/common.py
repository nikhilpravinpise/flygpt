"""Metric library for the eval battery - EVAL.md section 1, verbatim.

Pure-numpy implementations of the pre-registered definitions so two
implementers handed the same eval split produce the same number:

- 1.1 ``binarize_events`` / ``rate_from_events`` - fixed 0.5 threshold,
  Hz rates from raw (step, node) event lists.
- 1.2 ``event_f1`` - per-type / macro / micro with the inclusion rule
  (a type counts iff >= 1 true OR >= 1 predicted event in the split).
- 1.3 ``rate_pearson`` / ``rate_spearman`` - per-type correlation with
  nonzero-variance qualification; ``n_nonqualifying`` reported.
- 1.4 ``spike_agreement`` - event / +-1-step / rate-Pearson on raw event
  lists at native sim resolution (T-SIM-1/2, T-CORPUS-6).
- 1.5 ``response_sign`` / ``noise_floor`` - the Shiu-protocol response
  rule (T-SIM-7, T-REAL-1).
- 1.7 ``majority_baseline`` / ``allzero_baseline`` / ``silence_guard`` -
  silence-robust baselines and the eval-void invariant (T-MODEL-8).
- 1.8 ``coverage_by_type`` - driven / robustly-driven counting vs the
  manifest denominator (T-CORPUS-3, T-STIM-3).
- Section 2 ``ci_mean`` (Student-t, n-1), ``bh_fdr``, plus the
  ``binomial_test`` / ``bootstrap_sign_test`` significance helpers.

No pipeline imports: everything here is dependency-light by contract.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# EVAL 1.1 - the predicted-event threshold is fixed for all evals.
EVENT_THRESHOLD = 0.5

# Response labels for the 1.5 sign rule.
RESPONSE_INCREASE = "increase"
RESPONSE_DECREASE = "decrease"
RESPONSE_NO_EFFECT = "no-effect"
RESPONSE_AMBIGUOUS = "ambiguous"  # CI spans zero but |estimate| >= floor


# ---------------------------------------------------------------------------
# 1.1 binarization + rates
# ---------------------------------------------------------------------------


def binarize_events(probs, thr: float = EVENT_THRESHOLD) -> np.ndarray:
    """Fixed-threshold event binarization (EVAL 1.1) - never tuned."""
    return np.asarray(probs) >= thr


def _as_2d(a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"expected (T,V) array, got shape {arr.shape}")
    return arr


def _to_events(a: np.ndarray) -> np.ndarray:
    """Coerce an event/count/probability array to bool events.

    Integer and bool arrays are truth-masked (``> 0`` = the 1.1 count
    rule); float arrays are thresholded at the fixed 0.5 so probability
    predictions obey 1.1 while 0.0/1.0 binarized floats round-trip.
    """
    arr = np.asarray(a)
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return binarize_events(arr)
    return arr > 0


def events_matrix(
    events_step: np.ndarray,
    events_node: np.ndarray,
    n_steps: int,
    n_nodes: int,
) -> np.ndarray:
    """(step, node) event lists -> (T,V) bool active matrix."""
    mat = np.zeros((int(n_steps), int(n_nodes)), dtype=bool)
    steps = np.asarray(events_step, dtype=np.int64).ravel()
    nodes = np.asarray(events_node, dtype=np.int64).ravel()
    if steps.size:
        mat[steps, nodes] = True
    return mat


def rate_from_events(
    events_step: np.ndarray,
    events_node: np.ndarray,
    n_steps: int,
    dt: float,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Per-node firing rates in Hz from a raw (step, node) event list."""
    nodes = np.asarray(events_node, dtype=np.int64).ravel()
    if n_nodes is None:
        n_nodes = int(nodes.max()) + 1 if nodes.size else 0
    counts = np.bincount(nodes, minlength=int(n_nodes))
    span = float(n_steps) * float(dt)
    if span <= 0:
        raise ValueError("n_steps * dt must be positive")
    return counts / span


# ---------------------------------------------------------------------------
# 1.2 event-F1
# ---------------------------------------------------------------------------


def event_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Event-F1 per EVAL 1.2 on (T,V) arrays.

    Returns dict(macro, micro, per_type, included_types, n_included,
    n_types). A type is included iff it has >= 1 true event OR >= 1
    predicted event; types silent in both are excluded so all-zero
    output is never rewarded. per_type keys are column indices as str
    (JSON-safe); callers map indices to type names.
    """
    t = _as_2d(_to_events(y_true))
    p = _as_2d(_to_events(y_pred))
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {p.shape}")

    tp = (t & p).sum(axis=0).astype(np.float64)
    fp = (~t & p).sum(axis=0).astype(np.float64)
    fn = (t & ~p).sum(axis=0).astype(np.float64)
    denom = 2.0 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2.0 * tp / denom, np.nan)

    included = t.any(axis=0) | p.any(axis=0)  # 1.2 inclusion rule
    inc_idx = np.flatnonzero(included)
    per_type = {str(int(i)): float(f1[i]) for i in inc_idx}
    macro = float(np.mean(f1[inc_idx])) if inc_idx.size else float("nan")

    tp_t, fp_t, fn_t = float(tp.sum()), float(fp.sum()), float(fn.sum())
    micro = (
        2.0 * tp_t / (2.0 * tp_t + fp_t + fn_t)
        if (2.0 * tp_t + fp_t + fn_t) > 0
        else float("nan")
    )
    return {
        "macro": macro,
        "micro": float(micro),
        "per_type": per_type,
        "included_types": [int(i) for i in inc_idx],
        "n_included": int(inc_idx.size),
        "n_types": int(t.shape[1]),
    }


# ---------------------------------------------------------------------------
# 1.3 rate correlations
# ---------------------------------------------------------------------------


def _rate_corr(r_true: np.ndarray, r_pred: np.ndarray, method: str) -> dict:
    t = _as_2d(np.asarray(r_true, dtype=np.float64))
    p = _as_2d(np.asarray(r_pred, dtype=np.float64))
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch {t.shape} vs {p.shape}")

    per_type: dict[str, float] = {}
    n_nonqualifying = 0
    for i in range(t.shape[1]):
        a, b = t[:, i], p[:, i]
        if method == "spearman":
            a, b = stats.rankdata(a), stats.rankdata(b)
        if a.std() == 0.0 or b.std() == 0.0:  # 1.3 qualification rule
            n_nonqualifying += 1
            continue
        per_type[str(i)] = float(np.corrcoef(a, b)[0, 1])
    vals = np.asarray(list(per_type.values()))
    return {
        "macro": float(vals.mean()) if vals.size else float("nan"),
        "per_type": per_type,
        "n_qualifying": int(vals.size),
        "n_nonqualifying": int(n_nonqualifying),
    }


def rate_pearson(r_true: np.ndarray, r_pred: np.ndarray) -> dict:
    """Per-type Pearson r over eval bins (EVAL 1.3).

    A type qualifies iff BOTH sequences have nonzero variance; the count
    of non-qualifying types is reported alongside macro-r.
    """
    return _rate_corr(r_true, r_pred, "pearson")


def rate_spearman(r_true: np.ndarray, r_pred: np.ndarray) -> dict:
    """Per-type Spearman rho (Pearson on ranks) per EVAL 1.3."""
    return _rate_corr(r_true, r_pred, "spearman")


# ---------------------------------------------------------------------------
# 1.4 spike-time agreement (T-SIM-1/2, T-CORPUS-6)
# ---------------------------------------------------------------------------


def _events_to_matrix(
    events,
    n_nodes: int,
    n_steps: int | None,
) -> np.ndarray:
    """Accept a (T,V) matrix, a (steps, nodes) tuple, or (N,2) pairs.

    Disambiguation: tuples are always (steps, nodes) lists; bool arrays
    are matrices; non-bool arrays with exactly 2 columns are (step,
    node) pair lists unless ``n_nodes == 2 and n_steps == n_rows`` makes
    the matrix reading forced. For n_nodes == 2 pass the tuple form.
    """
    if (
        isinstance(events, (tuple, list))
        and len(events) == 2
        and np.asarray(events[0]).ndim <= 1
    ):
        steps = np.asarray(events[0], dtype=np.int64).ravel()
        nodes = np.asarray(events[1], dtype=np.int64).ravel()
        if n_steps is None:
            n_steps = int(steps.max()) + 1 if steps.size else 0
        return events_matrix(steps, nodes, n_steps, n_nodes)
    arr = np.asarray(events)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype != bool and not (
        n_nodes == 2 and n_steps is not None and arr.shape[0] == n_steps
    ):
        steps = arr[:, 0].astype(np.int64)
        if n_steps is None:
            n_steps = int(steps.max()) + 1 if steps.size else 0
        return events_matrix(steps, arr[:, 1].astype(np.int64), n_steps, n_nodes)
    if arr.ndim == 2 and arr.shape[1] == n_nodes:
        if n_steps is not None and arr.shape[0] != n_steps:
            raise ValueError(f"matrix rows {arr.shape[0]} != n_steps {n_steps}")
        return _to_events(arr)
    raise ValueError(f"cannot interpret events of shape {arr.shape} (n_nodes={n_nodes})")


def _adjacent_any(mat: np.ndarray) -> np.ndarray:
    """(T,V) -> (T,V): True where mat has an event within +-1 step."""
    padded = np.zeros((mat.shape[0] + 2, mat.shape[1]), dtype=bool)
    padded[1:-1] = mat
    return padded[:-2] | padded[1:-1] | padded[2:]


def spike_agreement(events_a, events_b, n_nodes: int, n_steps: int | None = None) -> dict:
    """Spike-time agreement rate between two runs (EVAL 1.4).

    Inputs: (T,V) bool matrices, (steps, nodes) tuple pairs, or (N,2)
    arrays - all at native sim resolution, before binning. Returns
    dict(event_agreement, pm1_step_agreement, rate_pearson,
    rate_pearson_qualified, n_steps, n_events_a, n_events_b).
    """
    a = _events_to_matrix(events_a, n_nodes, n_steps)
    b = _events_to_matrix(events_b, n_nodes, n_steps)
    if a.shape != b.shape:
        raise ValueError(f"event matrix shape mismatch {a.shape} vs {b.shape}")

    same = a == b
    event_agreement = float(same.mean()) if same.size else 1.0

    # +-1-step: an active/inactive mismatch counts as matched when the
    # other run fires the same neuron in an adjacent step.
    a_near = _adjacent_any(a)
    b_near = _adjacent_any(b)
    matched = same | (a & ~b & b_near) | (~a & b & a_near)
    pm1 = float(matched.mean()) if matched.size else 1.0

    ca = a.sum(axis=0).astype(np.float64)
    cb = b.sum(axis=0).astype(np.float64)
    qualified = bool(ca.std() > 0.0 and cb.std() > 0.0)
    r = float(np.corrcoef(ca, cb)[0, 1]) if qualified else float("nan")
    return {
        "event_agreement": event_agreement,
        "pm1_step_agreement": pm1,
        "rate_pearson": r,
        "rate_pearson_qualified": qualified,
        "n_steps": int(a.shape[0]),
        "n_events_a": int(a.sum()),
        "n_events_b": int(b.sum()),
    }


def events_bitwise_equal(events_a, events_b) -> bool:
    """Exact equality of sorted (step, node) event lists (CPU regime)."""
    sa, na = _event_pairs(events_a)
    sb, nb = _event_pairs(events_b)
    return bool(sa.shape == sb.shape and (sa == sb).all() and (na == nb).all())


def _event_pairs(events) -> tuple[np.ndarray, np.ndarray]:
    """Normalize any accepted event form to sorted (steps, nodes)."""
    if (
        isinstance(events, (tuple, list))
        and len(events) == 2
        and np.asarray(events[0]).ndim <= 1
    ):
        steps, nodes = np.asarray(events[0]).ravel(), np.asarray(events[1]).ravel()
        order = np.lexsort((nodes, steps))
        return steps[order], nodes[order]
    arr = np.asarray(events)
    if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype != bool:
        order = np.lexsort((arr[:, 1], arr[:, 0]))
        return arr[order, 0].astype(np.int64), arr[order, 1].astype(np.int64)
    if arr.ndim == 2:  # (T,V) matrix
        steps, nodes = np.nonzero(_to_events(arr))
        order = np.lexsort((nodes, steps))
        return steps[order], nodes[order]
    raise ValueError(f"cannot interpret events of shape {arr.shape}")


# ---------------------------------------------------------------------------
# 1.5 Shiu response rule + noise floor
# ---------------------------------------------------------------------------


def noise_floor(rates_baseline, pct: float = 95.0) -> float:
    """95th-pct (default) of |rate change| over unstimulated control trials."""
    vals = np.abs(np.asarray(rates_baseline, dtype=np.float64).ravel())
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, pct)) if vals.size else float("nan")


def response_sign(deltas, floor: float) -> str:
    """Predicted response sign per EVAL 1.5.

    ``deltas`` are per-trial rate changes vs the experiment's own
    baseline window. increase/decrease require the 95% CI to exclude
    zero; no-effect requires the CI to include zero AND the absolute
    point estimate below ``floor`` (the experiment noise floor of 1.5).
    A CI including zero with |estimate| >= floor is "ambiguous" -
    untestable, reported as its own label rather than silently binned.
    """
    mean, lo, hi = ci_mean(np.asarray(deltas, dtype=np.float64))
    if not np.isfinite(mean):
        return RESPONSE_AMBIGUOUS
    if lo > 0.0:
        return RESPONSE_INCREASE
    if hi < 0.0:
        return RESPONSE_DECREASE
    if abs(mean) <= floor:
        return RESPONSE_NO_EFFECT
    return RESPONSE_AMBIGUOUS


# ---------------------------------------------------------------------------
# 1.7 silence-robust baselines + the sanity invariant
# ---------------------------------------------------------------------------


def majority_baseline(y_true: np.ndarray, dt: float = 1.0) -> dict:
    """Per-type marginal-rate predictor (EVAL 1.7).

    Fit on the eval split's own marginals: predicted event probability
    for type ``i`` in every bin is the type's marginal event rate; the
    event prediction applies the fixed 0.5 threshold (predict event iff
    marginal rate >= 0.5). The rate prediction is the type's constant
    marginal count rate in Hz (``mean(c_i)/dt``).

    Returns dict(event_pred, rate_pred, event_f1, rate_pearson) - the
    predictor arrays plus its metrics evaluated on this split.
    """
    y = _as_2d(_to_events(y_true))
    p = y.mean(axis=0)  # (V,) marginal event rate
    e_hat = np.broadcast_to(binarize_events(p), y.shape)
    counts = _as_2d(np.asarray(y_true, dtype=np.float64))
    r_true = counts / float(dt)
    r_hat = np.broadcast_to(counts.mean(axis=0) / float(dt), y.shape)
    return {
        "event_pred": np.ascontiguousarray(e_hat),
        "rate_pred": np.ascontiguousarray(r_hat),
        "event_f1": event_f1(y, e_hat),
        "rate_pearson": rate_pearson(r_true, r_hat),
    }


def allzero_baseline(y_true: np.ndarray, dt: float = 1.0) -> dict:
    """All-zero predictor metrics (EVAL 1.7)."""
    y = _as_2d(_to_events(y_true))
    e_hat = np.zeros_like(y)
    r_true = _as_2d(np.asarray(y_true, dtype=np.float64)) / float(dt)
    r_hat = np.zeros_like(y, dtype=np.float64)
    return {
        "event_pred": e_hat,
        "rate_pred": r_hat,
        "event_f1": event_f1(y, e_hat),
        "rate_pearson": rate_pearson(r_true, r_hat),
    }


def silence_guard(metrics_zero: dict, metrics_majority: dict, tol: float = 1e-12) -> dict:
    """EVAL 1.7 sanity invariant (T-MODEL-8).

    The all-zero predictor must score <= the majority baseline on every
    compared metric on every split; a violation voids the eval. Compares
    macro + micro event-F1 (the silence-robust headline metrics). NaN on
    both sides counts as holding (empty split); NaN vs a finite number
    is a violation - fail closed.
    """
    checks = {}
    ok = True
    for key in ("macro", "micro"):
        z = float(metrics_zero["event_f1"][key])
        m = float(metrics_majority["event_f1"][key])
        z_v = -np.inf if np.isnan(z) else z
        m_v = -np.inf if np.isnan(m) else m
        held = bool(z_v <= m_v + tol)
        checks[key] = {"allzero": z, "majority": m, "held": held}
        ok = ok and held
    return {"ok": ok, "void": not ok, "checks": checks}


# ---------------------------------------------------------------------------
# 1.8 coverage counting (T-CORPUS-3, T-STIM-3)
# ---------------------------------------------------------------------------

DRIVEN_MIN_EVENTS = 1          # >= 1 spike in >= 1 trial
ROBUST_MIN_EVENTS = 5          # >= 5 spikes total
ROBUST_MIN_TRIALS = 10         # across >= 10 distinct trials


def coverage_by_type(
    events_trial: np.ndarray,
    events_node: np.ndarray,
    node_type: np.ndarray,
    manifest_types: list[str],
) -> dict:
    """Driven / robustly-driven type counts per EVAL 1.8.

    ``events_trial``/``events_node`` are the concatenated corpus event
    columns; ``node_type`` maps node_idx -> type label ("" for untyped);
    ``manifest_types`` is the denominator from the corpus manifest -
    never hardcoded. Untyped nodes' events are counted but map to no
    manifest type.
    """
    node_type = np.asarray(node_type).astype(str)
    ev_trial = np.asarray(events_trial, dtype=np.int64)
    ev_node = np.asarray(events_node, dtype=np.int64)

    # Integer-code path: a per-type string scan over the event stream is
    # O(n_types x n_events) on an object array (~30 GB at corpus scale) -
    # integer codes reduce it to one index + two uniques / bincounts.
    uniq_types, node_code = np.unique(node_type, return_inverse=True)
    ev_code = node_code[ev_node]
    n_codes = uniq_types.size
    events_per_code = np.bincount(ev_code, minlength=n_codes)
    if ev_code.size:
        # distinct trials per type via unique (trial, type) pairs
        pair = ev_trial * np.int64(n_codes) + ev_code
        upair = np.unique(pair)
        trials_per_code = np.bincount(upair % n_codes, minlength=n_codes)
    else:
        trials_per_code = np.zeros(n_codes, dtype=np.int64)
    n_events_of = dict(zip(uniq_types.tolist(), events_per_code.tolist(), strict=True))
    n_trials_of = dict(zip(uniq_types.tolist(), trials_per_code.tolist(), strict=True))

    report: dict[str, dict] = {}
    for t in manifest_types:
        n_events = int(n_events_of.get(t, 0))
        n_trials = int(n_trials_of.get(t, 0))
        report[t] = {
            "n_events": n_events,
            "n_trials": n_trials,
            "driven": n_events >= DRIVEN_MIN_EVENTS and n_trials >= 1,
            "robust": n_events >= ROBUST_MIN_EVENTS and n_trials >= ROBUST_MIN_TRIALS,
        }
    n_types = len(manifest_types)
    n_driven = sum(1 for v in report.values() if v["driven"])
    n_robust = sum(1 for v in report.values() if v["robust"])
    return {
        "per_type": report,
        "n_types": n_types,
        "n_driven": n_driven,
        "n_robust": n_robust,
        "driven_fraction": n_driven / n_types if n_types else float("nan"),
        "robust_fraction": n_robust / n_types if n_types else float("nan"),
    }


# ---------------------------------------------------------------------------
# Section 2 statistics: CIs, FDR, significance helpers
# ---------------------------------------------------------------------------


def ci_mean(vals, level: float = 0.95) -> tuple[float, float, float]:
    """Student-t CI of the mean (EVAL section 2: n-1 dof).

    Returns (mean, lo, hi). n=1 collapses to the point estimate (no
    sampling distribution exists); empty input -> NaN triple.
    """
    v = np.asarray(vals, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n == 0:
        return (float("nan"),) * 3
    mean = float(v.mean())
    if n == 1:
        return (mean, mean, mean)
    half = float(stats.t.ppf(0.5 + level / 2.0, n - 1)) * float(v.std(ddof=1)) / np.sqrt(n)
    return (mean, mean - half, mean + half)


def bh_fdr(pvals, q: float = 0.05) -> dict:
    """Benjamini-Hochberg FDR (EVAL section 2).

    Returns dict(reject, p_adj, q). ``reject`` is the BH rejection mask;
    ``p_adj`` the monotone step-up adjusted p values.
    """
    p = np.asarray(pvals, dtype=np.float64).ravel()
    m = p.size
    if m == 0:
        return {"reject": np.zeros(0, dtype=bool), "p_adj": np.zeros(0), "q": q}
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    thresholds = q * (np.arange(m) + 1) / m
    below = ranked <= thresholds
    if below.any():
        k_max = int(np.flatnonzero(below).max())
        cutoff = ranked[k_max]
        reject = p <= cutoff
    else:
        reject = np.zeros(m, dtype=bool)
    # monotone adjusted p: min_{j >= i} p_(j) * m / j, clipped to 1
    adj_ranked = np.minimum.accumulate((ranked * m / (np.arange(m) + 1))[::-1])[::-1]
    p_adj = np.empty(m)
    p_adj[order] = np.clip(adj_ranked, 0.0, 1.0)
    return {"reject": reject, "p_adj": p_adj, "q": q}


def binomial_test(k: int, n: int, p: float = 0.5) -> dict:
    """Two-sided exact binomial test (EVAL 1.6 per-type significance)."""
    res = stats.binomtest(int(k), int(n), float(p), alternative="two-sided")
    return {"k": int(k), "n": int(n), "p_null": float(p), "p": float(res.pvalue)}


def bootstrap_sign_test(deltas, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Bootstrap sign test on per-trial/seed deltas.

    Resamples the deltas with replacement and reports the two-sided
    fraction of bootstrap means on the wrong side of zero. Deterministic
    under ``seed`` (hash-derived at the call site, never hand-picked).
    """
    v = np.asarray(deltas, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"p": float("nan"), "mean": float("nan"), "n": 0, "n_boot": int(n_boot)}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    means = v[idx].mean(axis=1)
    p_side = min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {
        "p": min(1.0, 2.0 * p_side),
        "mean": float(v.mean()),
        "n": int(v.size),
        "n_boot": int(n_boot),
    }
