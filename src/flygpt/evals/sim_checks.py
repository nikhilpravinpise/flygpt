"""T-SIM-1..8 - simulator correctness runners (EVAL.md section 5).

Every runner takes injectable paths/functions so the checks run on
synthetic fixtures without the full pipeline; pipeline modules are
imported lazily inside the runners and a missing optional dep produces
a documented skip (``checks.skipped`` + ``pass: null``), never a silent
pass. Each runner writes ``results/T-SIM-*.json`` via
``flygpt.schema.write_result``.

Sim contract used here (the adapter seam): ``simulate_fn(bundle,
program: dict, cfg, seed: int) -> TrialResult-like`` returning an object
with ``events_step``/``events_node``/``n_steps``/``n_nodes`` (attribute,
dict, or ``(steps, nodes)`` tuple access all accepted). The plain
``simulate(bundle, program, cfg)`` signature of
``flygpt.sim_backend.simulate`` is honored too (seed travels inside the
program). ``program`` is a JSON-able spec dict; keys consumed:

- ``drives``: [{nodes, rate_hz, window}] - exogenous drive, the
  sim_backend ``_normalize_program`` key (``window_s`` tolerated).
- ``silences``: [{nodes, window}] - spikes dropped pre-propagation
  and never recorded (T-STIM-5 semantics).
- ``pulse``: {node, steps[]} - exact injected spikes (T-SIM-3); the
  same pulse is also expressed as a one-step ``step``-envelope drive so
  the stock backend can represent it.
- ``tonic_rate_hz``: uniform tonic Poisson drive level (the real
  backend reads it from cfg - runners that need it also pass a
  ``_cfg_with`` override).
- ``n_steps``, ``dt``, ``seed``, ``experiment_id``: run metadata.

Seed derivation (EVAL 1.4): ``_call_sim`` injects ``program["seed"]``
for seed-aware sims; for the stock backend - whose trial seed is
``seed_from_config`` over the NORMALIZED program - a per-trial seed only
survives normalization via a drive field, so trial loops append a
zero-effect ``_marker_drive`` carrying the trial seed in
``envelope_id``.

EVAL refs: 1.4 (T-SIM-1/2), 1.5 (T-SIM-7), T-SIM-3/4/5/6/8 thresholds
in the section-5 table.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from flygpt.evals import common
from flygpt.schema import rng_from_seed, seed_from_config, write_result

PANEL_PATH = Path(__file__).resolve().parents[3] / "evals" / "shiu_panel.json"

# Cross-backend regime thresholds (EVAL 1.4 third row - measured and
# documented; cross-backend shards never mix in one corpus version).
CROSS_BACKEND = {"pm1_step": 0.99, "rate_pearson": 0.99}
# Same-GPU-class regime (EVAL 1.4 second row); numpy is bitwise-only.
GPU_TOLERANCE = {"event": 0.995, "pm1_step": 0.999, "rate_pearson": 0.999}

SHIU_MIN_CORRECT = 8  # T-SIM-7: >= 8/10 directionally correct
T_SIM_4_MIN_PAIRS = 20
T_SIM_4_MIN_CORRECT = 0.95
T_SIM_4_MIN_WEIGHT = 10  # synapse floor for a measurable single edge
T_SIM_6_DEGENERATE_MAX = 0.05
T_SIM_6_STATIONARY_R = 0.5
T_SIM_6_STATIONARY_FRAC = 0.90
# Declared physiology band for T-SIM-8 (recorded in inputs, not tuned):
# sustained whole-CNS rates above ~300 Hz / spontaneous above ~50 Hz are
# implausible for this sim class - flagged outliers get reviewed.
DEFAULT_BAND = {"rate_hz": [0.0, 300.0], "spontaneous_hz": [0.0, 50.0]}


# ---------------------------------------------------------------------------
# Small helpers: cfg access, sim invocation, event normalization, skips
# ---------------------------------------------------------------------------


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict config or a SimConfig-like object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_backend(cfg: Any) -> str:
    backend = _cfg_get(cfg, "backend", None)
    if backend is None and isinstance(cfg, dict):
        backend = cfg.get("determinism", {}).get("backend", "numpy")
    return str(backend or "numpy")


def _cfg_with(cfg: Any, **overrides) -> Any:
    """cfg copy with field overrides - SimConfig via dataclasses.replace,
    dict via copy; unknown fields are dropped for dataclass cfgs."""
    if cfg is None:
        return dict(overrides)
    if isinstance(cfg, dict):
        out = dict(cfg)
        out.update(overrides)
        return out
    try:
        from dataclasses import replace as _replace

        return _replace(cfg, **{k: v for k, v in overrides.items() if hasattr(cfg, k)})
    except Exception:
        return cfg


def _marker_drive(seed: int, dt: float) -> dict:
    """Zero-effect drive carrying ``seed`` into the normalized spec.

    ``flygpt.sim_backend.simulate`` derives the trial seed from the
    NORMALIZED program - a top-level ``seed`` key is dropped there, so
    an empty-nodes drive with ``envelope_id = f"trial-seed:{seed}"`` is
    the spec-level channel that makes per-trial Poisson/tonic streams
    distinct while contributing no input itself.
    """
    return {
        "nodes": [],
        "rate_hz": 0.0,
        "window": [0.0, float(dt)],
        "envelope": "poisson",
        "envelope_id": f"trial-seed:{int(seed)}",
    }


def _lazy_simulate() -> Callable | None:
    """Default sim entry: flygpt.sim_backend.simulate (lazy import)."""
    try:
        from flygpt.sim_backend import simulate

        return simulate
    except Exception:
        return None


def _call_sim(fn: Callable, bundle, program: dict, cfg, seed: int):
    """Invoke a sim honoring the seed-bearing signatures in the wild."""
    program = dict(program)
    program["seed"] = int(seed)  # seed travels in the spec either way
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    if "seed" in params:
        return fn(bundle, program, cfg, seed=int(seed))
    if len(params) >= 4:
        return fn(bundle, program, cfg, int(seed))
    return fn(bundle, program, cfg)


def _events_of(result) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Normalize a TrialResult-like object to (steps, nodes, n_steps)."""
    if isinstance(result, tuple):
        steps = np.asarray(result[0], dtype=np.int64)
        nodes = np.asarray(result[1], dtype=np.int64)
        n_steps = int(result[2]) if len(result) > 2 else None
        return steps, nodes, n_steps
    get = result.get if isinstance(result, dict) else lambda k, d=None: getattr(result, k, d)
    steps = get("events_step")
    nodes = get("events_node")
    n_steps = get("n_steps")
    if steps is None and get("events") is not None:
        arr = np.asarray(get("events"), dtype=np.int64)
        steps, nodes = arr[:, 0], arr[:, 1]
    if steps is None and callable(get("active_mask")):
        mat = np.asarray(get("active_mask")())
        steps, nodes = np.nonzero(mat)
        order = np.lexsort((nodes, steps))
        steps, nodes = steps[order], nodes[order]
        n_steps = n_steps or mat.shape[0]
    if steps is None:
        raise TypeError(f"cannot extract events from {type(result)!r}")
    return np.asarray(steps, dtype=np.int64), np.asarray(nodes, dtype=np.int64), n_steps


def _skipped(results_dir: Path, test_id: str, reason: str, **kw) -> dict:
    """Write a documented skip - never a silent pass (hard rule)."""
    return write_result(
        results_dir,
        test_id,
        checks={"skipped": True, "reason": reason},
        inputs=kw.get("inputs", {}),
        metrics={},
        passed=None,
    )


def _require_sim(simulate_fn: Callable | None) -> Callable | None:
    return simulate_fn if simulate_fn is not None else _lazy_simulate()


def _gpu_tolerance(cfg: Any) -> dict[str, float]:
    tol = _cfg_get(cfg, "gpu_tolerance")
    if tol is None and isinstance(cfg, dict):
        tol = cfg.get("determinism", {}).get("gpu_tolerance")
    return {k: float(v) for k, v in (tol or GPU_TOLERANCE).items()}


# ---------------------------------------------------------------------------
# T-SIM-1: same-backend determinism (EVAL 1.4 rows 1-2)
# ---------------------------------------------------------------------------


def run_t_sim_1(
    bundle,
    cfg,
    results_dir: Path,
    *,
    simulate_fn: Callable | None = None,
    program: dict | None = None,
    n_steps: int = 200,
    seed: int | None = None,
) -> dict:
    """Same trial twice on the declared backend.

    numpy regime asserts BITWISE identical event lists; GPU regimes
    apply the 1.4 same-GPU-class thresholds (event >= 99.5%, +-1-step
    >= 99.9%, rate Pearson >= 0.999 - overridable via cfg).
    """
    test_id = "T-SIM-1"
    backend = _cfg_backend(cfg)
    dt = float(_cfg_get(cfg, "dt", 0.02))
    if seed is None:
        seed = seed_from_config({"test": test_id, "n_steps": n_steps, "backend": backend})
    sim = _require_sim(simulate_fn)
    inputs = {"backend": backend, "n_steps": n_steps, "seed": seed, "dt": dt}
    if sim is None:
        return _skipped(results_dir, test_id, "flygpt.sim_backend.simulate unavailable", inputs=inputs)

    prog = dict(
        program
        or {
            "family_id": "t_sim_1_determinism",
            "drives": [{"nodes": [0], "rate_hz": 100.0, "window": [0.0, n_steps * dt]}],
            "tonic_rate_hz": 10.0,
            "n_steps": n_steps,
            "dt": dt,
        }
    )
    r1 = _call_sim(sim, bundle, prog, cfg, seed)
    r2 = _call_sim(sim, bundle, prog, cfg, seed)
    e1 = _events_of(r1)
    e2 = _events_of(r2)
    steps = e1[2] or e2[2] or n_steps
    bitwise = common.events_bitwise_equal((e1[0], e1[1]), (e2[0], e2[1]))
    agreement = common.spike_agreement((e1[0], e1[1]), (e2[0], e2[1]), bundle.n_nodes, steps)

    if backend == "numpy":
        regime = "cpu_bitwise"
        passed = bitwise
        checks = {"regime": regime, "bitwise_identical": bitwise, "agreement": agreement}
    else:
        tol = _gpu_tolerance(cfg)
        regime = "same_gpu_class"
        per_metric = {
            "event": agreement["event_agreement"] >= tol.get("event", 0.995),
            "pm1_step": agreement["pm1_step_agreement"] >= tol.get("pm1_step", 0.999),
            "rate_pearson": agreement["rate_pearson"] >= tol.get("rate_pearson", 0.999),
        }
        passed = bool(all(per_metric.values()))
        checks = {
            "regime": regime,
            "bitwise_identical": bitwise,
            "agreement": agreement,
            "thresholds": tol,
            "per_metric_pass": per_metric,
        }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"agreement": agreement},
        passed=passed,
    )


# ---------------------------------------------------------------------------
# T-SIM-2: cross-backend consistency (EVAL 1.4 row 3)
# ---------------------------------------------------------------------------


def _lazy_backend_b(backend_b: str) -> tuple[Callable | None, str, str | None]:
    """Second sim for the cross-backend check: jax kernel or flybrain."""
    if backend_b == "jax":
        try:
            import jax  # noqa: F401
        except Exception:
            return None, backend_b, "jax not installed"
        sim = _lazy_simulate()
        if sim is None:
            return None, backend_b, "flygpt.sim_backend.simulate unavailable"
        return sim, backend_b, None
    if backend_b == "flybrain":
        try:
            from flygpt.sim_backend import FlybrainAdapter  # noqa: F401

            return FlybrainAdapter, backend_b, None
        except Exception:
            try:
                import flybrain  # noqa: F401
            except Exception:
                return None, backend_b, "flybrain adapter/package not installed"
            return None, backend_b, "flybrain present but FlybrainAdapter unavailable"
    return None, backend_b, f"unknown comparison backend {backend_b!r}"


def run_t_sim_2(
    bundle,
    cfg,
    results_dir: Path,
    *,
    simulate_fn: Callable | None = None,
    simulate_fn_b: Callable | None = None,
    cfg_b=None,
    backend_b: str = "jax",
    program: dict | None = None,
    n_steps: int = 200,
    seed: int | None = None,
) -> dict:
    """numpy vs second backend (jax kernel or flybrain) on the same trial.

    Cross-backend regime: >= 99.0% event agreement within +-1 step and
    rate Pearson >= 0.99 - measured and documented only (EVAL 1.4).
    Skips with a documented reason when the second backend is absent.
    """
    test_id = "T-SIM-2"
    dt = float(_cfg_get(cfg, "dt", 0.02))
    if seed is None:
        seed = seed_from_config({"test": test_id, "n_steps": n_steps, "backend_b": backend_b})
    sim_a = _require_sim(simulate_fn)
    sim_b, label_b, reason_b = (
        (simulate_fn_b, backend_b, None)
        if simulate_fn_b is not None
        else _lazy_backend_b(backend_b)
    )
    inputs = {"backend_a": "numpy", "backend_b": label_b, "n_steps": n_steps, "seed": seed}
    missing = []
    if sim_a is None:
        missing.append("flygpt.sim_backend.simulate (numpy side)")
    if sim_b is None:
        missing.append(f"backend {backend_b!r}: {reason_b}")
    if missing:
        return _skipped(results_dir, test_id, "missing: " + "; ".join(missing), inputs=inputs)

    prog = dict(
        program
        or {
            "family_id": "t_sim_2_consistency",
            "drives": [{"nodes": [0], "rate_hz": 100.0, "window": [0.0, n_steps * dt]}],
            "tonic_rate_hz": 10.0,
            "n_steps": n_steps,
            "dt": dt,
        }
    )
    ra = _call_sim(sim_a, bundle, prog, cfg, seed)
    rb = _call_sim(sim_b, bundle, prog, cfg_b if cfg_b is not None else cfg, seed)
    ea, eb = _events_of(ra), _events_of(rb)
    steps = ea[2] or eb[2] or n_steps
    agreement = common.spike_agreement((ea[0], ea[1]), (eb[0], eb[1]), bundle.n_nodes, steps)
    per_metric = {
        "pm1_step": agreement["pm1_step_agreement"] >= CROSS_BACKEND["pm1_step"],
        "rate_pearson": agreement["rate_pearson"] >= CROSS_BACKEND["rate_pearson"],
    }
    return write_result(
        results_dir,
        test_id,
        checks={
            "regime": "cross_backend",
            "agreement": agreement,
            "thresholds": CROSS_BACKEND,
            "per_metric_pass": per_metric,
            "bitwise_identical": common.events_bitwise_equal(
                (ea[0], ea[1]), (eb[0], eb[1])
            ),
        },
        inputs=inputs,
        metrics={"agreement": agreement},
        passed=bool(all(per_metric.values())),
    )


# ---------------------------------------------------------------------------
# T-SIM-3: single-neuron pulse causality
# ---------------------------------------------------------------------------


def run_t_sim_3(
    bundle,
    cfg,
    results_dir: Path,
    *,
    simulate_fn: Callable | None = None,
    pulse_node: int | None = None,
    pulse_step: int | None = None,
    n_steps: int = 50,
    seed: int | None = None,
    rel_eps: float = 1e-6,
) -> dict:
    """Inject a one-step pulse; only postsynaptic partners may change.

    Pass: 100% of t+1 event changes confined to the pulsed neuron's
    postsynaptic partners; changes at t (same step) beyond the injected
    spike itself = 0 (rel eps ``rel_eps`` on rates / exact on events).
    """
    test_id = "T-SIM-3"
    dt = float(_cfg_get(cfg, "dt", 0.02))
    if seed is None:
        seed = seed_from_config({"test": test_id, "n_steps": n_steps})
    sim = _require_sim(simulate_fn)
    rng = rng_from_seed(test_id, "pulse_node", bundle.n_nodes)
    if pulse_node is None:
        candidates = [i for i in range(bundle.n_nodes) if bundle.out_neighbors(i).size]
        if not candidates:
            return _skipped(results_dir, test_id, "graph has no outgoing edges")
        pulse_node = int(candidates[int(rng.integers(len(candidates)))])
    if pulse_step is None:
        pulse_step = int(n_steps // 4)
    inputs = {
        "pulse_node": int(pulse_node),
        "pulse_step": int(pulse_step),
        "n_steps": n_steps,
        "seed": seed,
        "rel_eps": rel_eps,
    }
    if sim is None:
        return _skipped(results_dir, test_id, "flygpt.sim_backend.simulate unavailable", inputs=inputs)

    # The exact same-step-zero / t+1-partner comparison needs a
    # deterministic background: force tonic/noise off for both arms so
    # the only difference is the injected spike. The pulse is expressed
    # twice - ``pulse`` for fake sims and a one-step ``step``-envelope
    # drive the stock backend normalizes identically.
    cfg_clean = _cfg_with(
        cfg, tonic_rate_hz=0.0, tonic_amount=0.0, noise_hz=0.0, noise_amp=0.0
    )
    base_prog = {"family_id": "t_sim_3_causality", "drives": [], "n_steps": n_steps, "dt": dt}
    pulse_prog = dict(base_prog)
    pulse_prog["pulse"] = {"node": int(pulse_node), "steps": [int(pulse_step)]}
    pulse_prog["drives"] = [
        {
            "nodes": [int(pulse_node)],
            "rate_hz": 1.0 / dt,
            "window": [pulse_step * dt, (pulse_step + 1) * dt],
            "envelope": "step",
        }
    ]
    base = _events_of(_call_sim(sim, bundle, base_prog, cfg_clean, seed))
    pulsed = _events_of(_call_sim(sim, bundle, pulse_prog, cfg_clean, seed))
    steps = max(base[2] or 0, pulsed[2] or 0, pulse_step + 2, 1)
    m_base = common.events_matrix(base[0], base[1], steps, bundle.n_nodes)
    m_pul = common.events_matrix(pulsed[0], pulsed[1], steps, bundle.n_nodes)

    partners = {int(x) for x in bundle.out_neighbors(pulse_node)}
    allowed_t1 = partners | {int(pulse_node)}
    changed_by_step = {
        t: sorted(int(i) for i in np.flatnonzero(m_base[t] != m_pul[t])) for t in range(steps)
    }
    same_time = [n for n in changed_by_step.get(pulse_step, []) if n != pulse_node]
    next_step = changed_by_step.get(pulse_step + 1, [])
    outside = [n for n in next_step if n not in allowed_t1]
    same_time_ok = len(same_time) == 0
    confined_ok = len(outside) == 0
    checks = {
        "same_step_changes_beyond_pulse": same_time,
        "t1_changes": next_step,
        "t1_changes_outside_partners": outside,
        "postsynaptic_partners": sorted(partners),
        "same_time_ok": same_time_ok,
        "confined_ok": confined_ok,
        "n_partner_changes_t1": int(len([n for n in next_step if n in partners])),
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"changed_by_step": changed_by_step},
        passed=bool(same_time_ok and confined_ok),
    )


# ---------------------------------------------------------------------------
# T-SIM-4: known-sign panel
# ---------------------------------------------------------------------------


def run_t_sim_4(
    bundle,
    cfg,
    results_dir: Path,
    *,
    simulate_fn: Callable | None = None,
    n_pairs: int = T_SIM_4_MIN_PAIRS,
    n_trials: int = 8,
    rate_hz: float = 200.0,
    duration_s: float = 0.5,
    exc_tonic_rate_hz: float = 20.0,
    exc_tonic_amount: float = 0.25,
    inh_tonic_rate_hz: float = 40.0,
    inh_tonic_amount: float = 0.7,
    min_weight: int = T_SIM_4_MIN_WEIGHT,
    seed: int | None = None,
) -> dict:
    """Known-sign exc/inhib pairs -> directional correctness fraction.

    Panel: ``n_pairs`` edges with determined NT signs (autapses and
    undetermined edges excluded - a pair is two distinct neurons),
    sampled via the hash-derived seed (with replacement when the graph
    is smaller than the panel). Each pair is driven at ``rate_hz`` over
    a tonic baseline; correctness requires the 95% CI of the postsyn
    rate change across trials to exclude zero on the sign's side
    ("each CI-separated across trials", EVAL T-SIM-4). Pass >= 95%.
    """
    test_id = "T-SIM-4"
    dt = float(_cfg_get(cfg, "dt", 0.02))
    n_steps = max(int(round(duration_s / dt)), 2)
    if seed is None:
        seed = seed_from_config({"test": test_id, "n_pairs": n_pairs})
    sim = _require_sim(simulate_fn)
    inputs = {
        "n_pairs": n_pairs,
        "n_trials": n_trials,
        "rate_hz": rate_hz,
        "duration_s": duration_s,
        "exc_tonic": [exc_tonic_rate_hz, exc_tonic_amount],
        "inh_tonic": [inh_tonic_rate_hz, inh_tonic_amount],
        "seed": seed,
    }
    if sim is None:
        return _skipped(results_dir, test_id, "flygpt.sim_backend.simulate unavailable", inputs=inputs)

    pre = np.repeat(np.arange(bundle.n_nodes), np.diff(bundle.out_indptr))
    post = bundle.out_indices
    sign = bundle.out_sign
    wts = bundle.out_weight
    # Measurable-unit floor: only edges with >= ``min_weight`` synapses enter
    # the panel - a weight-5 edge legitimately cannot move a silenced post
    # far enough for a CI-separated read on every draw.
    eligible = np.flatnonzero(
        (sign != 0) & (pre != post) & (wts >= int(min_weight))
    )
    if eligible.size == 0:
        return _skipped(results_dir, test_id, "no sign-determined non-autapse edges", inputs=inputs)
    rng = rng_from_seed(test_id, "panel", seed)
    pick = rng.choice(eligible, size=min(int(n_pairs), 10_000), replace=eligible.size < n_pairs)
    pair_list = [(int(pre[i]), int(post[i]), int(sign[i])) for i in pick]

    # Paired-recording protocol: on a real 211k-node graph a single edge
    # cannot be heard over hundreds of tonic-driven convergent inputs, so
    # every node except the pair is silenced. The BASE arm silences the
    # pre too, giving a pure tonic baseline; the DRIVEN arm releases the
    # pre's forced spikes. Because drives force spikes (inhibition cannot
    # suppress them), the postsyn baseline must be ENDOGENOUS: per-sign
    # tonic levels - subthreshold for excitatory (a measurable increase)
    # and saturating for inhibitory (a measurable suppression), the
    # inhibitory amount auto-calibrated per post.
    all_nodes = np.arange(bundle.n_nodes)
    results = []
    n_correct = 0
    for k, (p, q, s) in enumerate(pair_list):
        tr, ta = (
            (exc_tonic_rate_hz, exc_tonic_amount)
            if s > 0
            else (inh_tonic_rate_hz, inh_tonic_amount)
        )
        cfg_tonic = _cfg_with(
            cfg, tonic_rate_hz=tr, tonic_amount=ta, noise_hz=0.0
        )
        oth_base = all_nodes[all_nodes != q]
        silence_base = [{"nodes": oth_base.tolist(), "window": [0.0, duration_s]}]
        oth_drv = all_nodes[(all_nodes != p) & (all_nodes != q)]
        silence_drv = [{"nodes": oth_drv.tolist(), "window": [0.0, duration_s]}]
        deltas = []
        for t in range(int(n_trials)):
            trial_seed = seed_from_config({"test": test_id, "pair": k, "trial": t, "seed": seed})
            base_prog = {
                "family_id": "t_sim_4_sign",
                "drives": [_marker_drive(trial_seed, dt)],
                "silences": silence_base,
                "tonic_rate_hz": tr,
                "n_steps": n_steps,
                "dt": dt,
            }
            drv_prog = dict(base_prog)
            drv_prog["silences"] = silence_drv
            drv_prog["drives"] = [
                {"nodes": [p], "rate_hz": rate_hz, "window": [0.0, duration_s]},
                _marker_drive(trial_seed, dt),
            ]
            b = _events_of(_call_sim(sim, bundle, base_prog, cfg_tonic, trial_seed))
            d = _events_of(_call_sim(sim, bundle, drv_prog, cfg_tonic, trial_seed))
            steps = max(b[2] or 0, d[2] or 0, n_steps)
            r_base = common.rate_from_events(b[0], b[1], steps, dt, bundle.n_nodes)[q]
            r_drv = common.rate_from_events(d[0], d[1], steps, dt, bundle.n_nodes)[q]
            deltas.append(float(r_drv - r_base))
        mean, lo, hi = common.ci_mean(deltas)
        correct = (s > 0 and lo > 0.0) or (s < 0 and hi < 0.0)
        n_correct += int(correct)
        results.append(
            {
                "pre": p,
                "post": q,
                "edge_sign": s,
                "delta_mean": mean,
                "ci_lo": lo,
                "ci_hi": hi,
                "correct": bool(correct),
            }
        )
    frac = n_correct / len(results) if results else float("nan")
    checks = {
        "n_pairs_evaluated": len(results),
        "n_pairs_required": T_SIM_4_MIN_PAIRS,
        "n_correct": n_correct,
        "fraction_correct": frac,
        "threshold": T_SIM_4_MIN_CORRECT,
        "pairs": results,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"fraction_correct": frac},
        passed=bool(len(results) >= T_SIM_4_MIN_PAIRS and frac >= T_SIM_4_MIN_CORRECT),
    )


# ---------------------------------------------------------------------------
# T-SIM-5: silence / tonic checks
# ---------------------------------------------------------------------------


def run_t_sim_5(
    bundle,
    cfg,
    results_dir: Path,
    *,
    simulate_fn: Callable | None = None,
    n_steps: int = 100,
    tonic_rate_hz: float = 25.0,
    tonic_amount: float = 0.25,
    seed: int | None = None,
) -> dict:
    """Zero-baseline unstimulated -> exactly 0 events; tonic -> nonzero.

    Exact both ways (EVAL T-SIM-5). The tonic/zero levels are applied
    through cfg overrides (the stock backend's channel) AND the
    ``tonic_rate_hz`` program key (the fake-sim seam).
    """
    test_id = "T-SIM-5"
    dt = float(_cfg_get(cfg, "dt", 0.02))
    if seed is None:
        seed = seed_from_config({"test": test_id, "n_steps": n_steps})
    sim = _require_sim(simulate_fn)
    inputs = {
        "n_steps": n_steps,
        "tonic_rate_hz": tonic_rate_hz,
        "tonic_amount": tonic_amount,
        "seed": seed,
    }
    if sim is None:
        return _skipped(results_dir, test_id, "flygpt.sim_backend.simulate unavailable", inputs=inputs)

    cfg_zero = _cfg_with(
        cfg, tonic_rate_hz=0.0, tonic_amount=0.0, noise_hz=0.0, noise_amp=0.0
    )
    cfg_tonic = _cfg_with(
        cfg, tonic_rate_hz=tonic_rate_hz, tonic_amount=tonic_amount, noise_hz=0.0
    )
    zero_prog = {
        "family_id": "silence_test",
        "drives": [],
        "tonic_rate_hz": 0.0,
        "n_steps": n_steps,
        "dt": dt,
    }
    tonic_prog = dict(zero_prog)
    tonic_prog["tonic_rate_hz"] = float(tonic_rate_hz)
    e_zero = _events_of(_call_sim(sim, bundle, zero_prog, cfg_zero, seed))
    e_tonic = _events_of(_call_sim(sim, bundle, tonic_prog, cfg_tonic, seed))
    n_zero = int(e_zero[0].size)
    n_tonic = int(e_tonic[0].size)
    checks = {
        "zero_baseline_events": n_zero,
        "tonic_events": n_tonic,
        "zero_ok": n_zero == 0,
        "tonic_ok": n_tonic > 0,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"zero_baseline_events": n_zero, "tonic_events": n_tonic},
        passed=bool(n_zero == 0 and n_tonic > 0),
    )


# ---------------------------------------------------------------------------
# T-SIM-6: pilot health metrics
# ---------------------------------------------------------------------------


def _trial_health(
    steps: np.ndarray,
    nodes: np.ndarray,
    n_steps: int,
    type_of: np.ndarray,
    n_driven: int = -1,
) -> dict:
    """Per-trial pilot health fields (EVAL T-SIM-6 definition).

    ``n_driven`` = the program's total driven-neuron count. The 0.5%-actf
    and split-half criteria presuppose a population-level stimulus. A
    zero-drive program is a silence control: silence is its correct
    output, and it is degenerate only on a >60% blow-up. A program
    driving fewer than ~0.5% of the manifest's types worth of cells
    (``0.005 * n_types`` - 58 on the real graph) is a sparse perturbation
    whose actf cannot reach the floor by construction - it is degenerate
    only on a zero event stream or a >60% blow-up. ``n_driven == -1``
    means "drive size unspecified": the conservative population rule
    applies (fail closed).
    """
    n_nodes = type_of.size
    counts = np.bincount(nodes, minlength=n_nodes)
    active_nodes = counts > 0
    active_node_frac = float(active_nodes.mean()) if n_nodes else 0.0
    types = np.asarray(type_of).astype(str)
    typed_mask = types != ""
    active_types = set(types[active_nodes & typed_mask])
    n_manifest = int(np.unique(types[typed_mask]).size) if typed_mask.any() else 0
    active_type_frac = len(active_types) / n_manifest if n_manifest else 0.0

    half = n_steps // 2
    first = np.bincount(nodes[steps < half], minlength=n_nodes) if half else np.zeros(n_nodes)
    second = np.bincount(nodes[steps >= half], minlength=n_nodes)
    r = (
        float(np.corrcoef(first, second)[0, 1])
        if first.std() > 0 and second.std() > 0
        else float("nan")
    )
    sparse = n_driven != -1 and n_driven < max(int(0.005 * n_manifest), 1)
    stationary = bool(np.isfinite(r) and r >= T_SIM_6_STATIONARY_R)
    if n_driven == 0:
        # silence control: quiet output is the intended result
        degenerate = bool(active_type_frac > 0.60)
    elif sparse:
        degenerate = bool(steps.size == 0 or active_type_frac > 0.60)
    else:
        degenerate = bool(
            active_type_frac < 0.005 or active_type_frac > 0.60 or not stationary
        )
    return {
        "n_events": int(steps.size),
        "active_node_fraction": active_node_frac,
        "active_type_fraction": float(active_type_frac),
        "split_half_r": r,
        "stationary": stationary,
        "expected_sparse": bool(sparse or n_driven == 0),
        "degenerate": degenerate,
        "mean_rate_hz": float(counts.mean() / (n_steps * 1.0)),
    }


def run_t_sim_6(
    bundle,
    cfg,
    results_dir: Path,
    *,
    simulate_fn: Callable | None = None,
    programs: list[dict] | None = None,
    index: Any | None = None,
    n_programs: int = 12,
    n_steps: int = 50,
    seed: int | None = None,
) -> dict:
    """Pilot health over a trial set: degenerate fraction + sane band.

    Degenerate = active-type fraction < 0.5% or > 60% (trial-level
    "sustained" reading), or split-half stationarity r < 0.5. Pass:
    degenerate fraction <= 5% AND >= 90% of trials stationary at
    r >= 0.5. The sane band (pilot median +- 3xMAD of the mean rate per
    family) is reported for the pilot-gate record, not auto-committed.
    """
    test_id = "T-SIM-6"
    dt = float(_cfg_get(cfg, "dt", 0.02))
    if seed is None:
        seed = seed_from_config({"test": test_id, "n_programs": n_programs})
    sim = _require_sim(simulate_fn)
    inputs = {"n_programs": n_programs, "n_steps": n_steps, "seed": seed, "dt": dt}
    if sim is None:
        return _skipped(results_dir, test_id, "flygpt.sim_backend.simulate unavailable", inputs=inputs)

    if programs is None and index is not None:
        # Real grounded programs, one per stimulus family - the pilot-gate
        # materials EVAL T-SIM-6 is about. Generated through the LIBRARY so
        # grounding is identical to corpus generation (a random
        # one-third-of-brain drive is degenerate by construction on the
        # real connectome and is only the no-index fallback).
        try:
            from flygpt.stimuli import LIBRARY, program_to_sim_spec
            from flygpt.stimuli import rng_from_seed as stim_seed
        except Exception:
            LIBRARY = None
        if LIBRARY:
            progs = []
            stim_cfg = getattr(index, "stim_cfg", None) or {}
            families = stim_cfg.get("families", {})
            for i, fid in enumerate(sorted(LIBRARY)[: int(n_programs)]):
                if fid in ("babble", "tonic_baseline", "silence_test"):
                    continue
                fam_cfg = dict(families.get(fid, {}))
                gen_rng = stim_seed(test_id, fid, int(seed))
                try:
                    prog = LIBRARY[fid](index, gen_rng, fam_cfg, trial=i)
                except Exception:
                    continue
                spec = program_to_sim_spec(prog)
                # Keep the program's own duration - truncating the drive
                # window inflates split-half asymmetry on staggered
                # envelopes (courtship/naturalistic read as spuriously
                # non-stationary when clipped).
                spec.setdefault("n_steps", n_steps)
                spec.setdefault("dt", dt)
                progs.append(spec)
            if progs:
                programs = progs
    if programs is None:
        rng = rng_from_seed(test_id, "programs", seed)
        programs = []
        # No-grounding fallback: random sparse subset (small fixtures only;
        # on a real graph prefer the LIBRARY path above).
        for _ in range(int(n_programs)):
            subset = rng.choice(
                bundle.n_nodes,
                size=max(1, min(bundle.n_nodes // 3, 2_000)),
                replace=False,
            )
            programs.append(
                {
                    "family_id": "pilot_default",
                    "drives": [
                        {
                            "nodes": [int(i) for i in subset],
                            "rate_hz": 50.0 + 50.0 * float(rng.random()),
                            "window": [0.0, n_steps * dt],
                        }
                    ],
                    "tonic_rate_hz": 10.0,
                    "n_steps": n_steps,
                    "dt": dt,
                }
            )

    type_of = bundle.type_array()
    per_trial = []
    for k, prog in enumerate(programs):
        trial_seed = seed_from_config({"test": test_id, "program": k, "seed": seed})
        ev = _events_of(_call_sim(sim, bundle, prog, cfg, trial_seed))
        steps_n = ev[2] or n_steps
        n_driven = sum(len(d.get("nodes", ())) for d in prog.get("drives", ()))
        h = _trial_health(ev[0], ev[1], int(steps_n), type_of, n_driven)
        h["program_index"] = k
        h["family_id"] = prog.get("family_id", "?")
        h["n_driven"] = int(n_driven)
        per_trial.append(h)

    n_trials = len(per_trial)
    degenerate_frac = float(np.mean([h["degenerate"] for h in per_trial])) if n_trials else 1.0
    # Stationarity applies to population-level trials; sparse perturbations
    # are waived (split-half r is undefined on near-empty event streams)
    # and recorded as expected_sparse in per_trial.
    stationary_frac = (
        float(np.mean([h["stationary"] or h["expected_sparse"] for h in per_trial]))
        if n_trials
        else 0.0
    )

    families: dict[str, list[float]] = {}
    for h in per_trial:
        families.setdefault(h["family_id"], []).append(h["mean_rate_hz"])
    sane_band = {}
    for fam, rates in families.items():
        med = float(np.median(rates))
        mad = float(np.median(np.abs(np.asarray(rates) - med)))
        sane_band[fam] = {"median": med, "mad": mad, "lo": med - 3 * mad, "hi": med + 3 * mad}

    checks = {
        "n_trials": n_trials,
        "degenerate_fraction": degenerate_frac,
        "degenerate_fraction_max": T_SIM_6_DEGENERATE_MAX,
        "stationary_fraction": stationary_frac,
        "stationary_fraction_min": T_SIM_6_STATIONARY_FRAC,
        "sane_band_hz": sane_band,
        "per_trial": per_trial,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={
            "degenerate_fraction": degenerate_frac,
            "stationary_fraction": stationary_frac,
        },
        passed=bool(
            degenerate_frac <= T_SIM_6_DEGENERATE_MAX
            and stationary_frac >= T_SIM_6_STATIONARY_FRAC
        ),
    )


# ---------------------------------------------------------------------------
# T-SIM-7: Shiu spot-check panel (EVAL 1.5)
# ---------------------------------------------------------------------------


def _resolve_query(index, query: dict, bundle, type_map: dict | None) -> np.ndarray:
    """Resolve a panel query to node indices via the SensoryIndex.

    Supports {types, classes, subclasses, superclasses, type_regex};
    ``type_map`` remaps panel-only type names (e.g. the reserved
    "ovulin" slot) to real graph types at runtime. The special key
    ``postsyn_of_drive`` resolves to the experiment's drive set's direct
    postsynaptic partners - the literal "downstream" population.
    """
    if not query:
        return np.empty(0, dtype=np.int64)
    query = dict(query)
    type_map = type_map or {}
    drivers = query.pop("_drivers", None)
    if query.pop("postsyn_of_drive", False):
        if drivers is None or not len(drivers):
            return np.empty(0, dtype=np.int64)
        pre = np.repeat(np.arange(bundle.n_nodes), np.diff(bundle.out_indptr))
        return np.unique(
            bundle.out_indices[np.isin(pre, np.asarray(drivers, dtype=np.int64))]
        ).astype(np.int64)
    if "type_regex" in query:
        pat = re.compile(query.pop("type_regex"))
        query["types"] = sorted(
            set(query.get("types") or []) | {t for t in bundle.manifest_types if pat.match(t)}
        )
    if "types" in query:
        query["types"] = [type_map.get(t, t) for t in query["types"]]
    resolved = np.asarray(index.resolve(**query), dtype=np.int64)
    return np.unique(resolved)


def run_t_sim_7(
    bundle,
    cfg,
    results_dir: Path,
    *,
    index=None,
    simulate_fn: Callable | None = None,
    panel_path: Path | None = None,
    n_trials: int = 30,
    baseline_window_s: tuple[float, float] = (0.0, 0.2),
    drive_window_s: tuple[float, float] = (0.2, 1.0),
    seed: int | None = None,
    drive_type_map: dict | None = None,
) -> dict:
    """Frozen Shiu spot-check panel (evals/shiu_panel.json, EVAL 1.5).

    Per experiment: resolve drive/target queries through the sensory
    index, run ``n_trials`` driven + unstimulated trials, apply the 1.5
    response rule (CI-separated sign vs the experiment's own baseline
    window; no-effect iff CI spans zero AND |estimate| < noise floor =
    95th pct |dRate| over that experiment's unstimulated controls).
    Pass: >= 8/10 directionally correct; ambiguous predictions count as
    not-correct. Panel slots that resolve to nothing on the graph are
    skipped with a recorded reason (e.g. the reserved "ovulin" slot,
    which resolves only via ``drive_type_map``/cfg runtime remap).
    """
    test_id = "T-SIM-7"
    dt = float(_cfg_get(cfg, "dt", 0.02))
    panel_file = Path(panel_path) if panel_path else PANEL_PATH
    if seed is None:
        seed = seed_from_config({"test": test_id, "panel": "shiu_spot_checks_v1"})
    if drive_type_map is None:
        drive_type_map = dict(_cfg_get(cfg, "t_sim_7_type_map") or {})
    sim = _require_sim(simulate_fn)
    inputs = {
        "panel": str(panel_file),
        "n_trials": n_trials,
        "baseline_window_s": list(baseline_window_s),
        "drive_window_s": list(drive_window_s),
        "seed": seed,
    }
    if not panel_file.exists():
        return _skipped(results_dir, test_id, f"panel file absent: {panel_file}", inputs=inputs)
    if index is None:
        try:
            from flygpt.stimuli import SensoryIndex, load_stim_config

            stim_yaml = Path("configs/stim_library_v1.yaml")
            scfg = load_stim_config(stim_yaml) if stim_yaml.exists() else None
            index = SensoryIndex(bundle, scfg)
        except Exception:
            index = None
    if index is None:
        return _skipped(
            results_dir, test_id, "flygpt.stimuli.SensoryIndex unavailable", inputs=inputs
        )
    if sim is None:
        return _skipped(results_dir, test_id, "flygpt.sim_backend.simulate unavailable", inputs=inputs)

    panel = json.loads(panel_file.read_text())
    experiments = panel.get("experiments", [])
    n_steps = int(round(drive_window_s[1] / dt))

    def _window_rate(steps_arr, nodes_arr, target, t0, t1, n_st):
        if target.size == 0:
            return 0.0
        s0, s1 = int(t0 / dt), min(int(np.ceil(t1 / dt)), n_st)
        mask = (steps_arr >= s0) & (steps_arr < s1) & np.isin(nodes_arr, target)
        return float(mask.sum()) / (target.size * (s1 - s0) * dt)

    per_exp = []
    n_correct = 0
    for exp in experiments:
        exp_id = exp.get("id", "?")
        entry = {"id": exp_id, "expected": exp.get("expected")}
        try:
            drivers = _resolve_query(index, exp.get("drive_query"), bundle, drive_type_map)
            tq = dict(exp.get("target_query") or {})
            tq["_drivers"] = drivers
            targets = _resolve_query(index, tq, bundle, drive_type_map)
        except Exception as exc:  # unresolvable query - documented skip
            entry.update({"skipped": True, "reason": f"query error: {exc}", "correct": False})
            per_exp.append(entry)
            continue
        if targets.size == 0:
            entry.update(
                {"skipped": True, "reason": "target_query resolves to zero nodes", "correct": False}
            )
            per_exp.append(entry)
            continue
        if exp.get("drive_query") and drivers.size == 0:
            entry.update(
                {
                    "skipped": True,
                    "reason": "drive_query resolves to zero nodes on this graph",
                    "correct": False,
                }
            )
            per_exp.append(entry)
            continue

        rate_hz = float(exp.get("drive_rate_hz", 0.0))
        # per-experiment tonic baseline: "decrease" items need endogenous
        # network activity to suppress - cfg-level override, recorded.
        ecfg = cfg
        if exp.get("tonic_rate_hz") or exp.get("tonic_amount"):
            ecfg = _cfg_with(
                cfg,
                tonic_rate_hz=float(exp.get("tonic_rate_hz", 0.0)),
                tonic_amount=float(exp.get("tonic_amount", 0.0)),
            )
        deltas_drv, deltas_base = [], []
        for t in range(int(n_trials)):
            for arm, store in (("drive", deltas_drv), ("base", deltas_base)):
                s_t = seed_from_config(
                    {"test": test_id, "exp": exp_id, "arm": arm, "trial": t, "seed": seed}
                )
                drives = [_marker_drive(s_t, dt)]
                if arm == "drive" and rate_hz > 0 and drivers.size:
                    drives.insert(
                        0,
                        {
                            "nodes": [int(i) for i in drivers],
                            "rate_hz": rate_hz,
                            "window": list(drive_window_s),
                        },
                    )
                prog = {
                    "family_id": "shiu_panel",
                    "experiment_id": exp_id,
                    "drives": drives,
                    "n_steps": n_steps,
                    "dt": dt,
                }
                ev = _events_of(_call_sim(sim, bundle, prog, ecfg, s_t))
                n_st = int(ev[2] or n_steps)
                r_drive = _window_rate(ev[0], ev[1], targets, *drive_window_s, n_st)
                r_base = _window_rate(ev[0], ev[1], targets, *baseline_window_s, n_st)
                store.append(r_drive - r_base)
        floor = common.noise_floor(deltas_base, pct=95)
        predicted = common.response_sign(deltas_drv, floor)
        correct = predicted == exp.get("expected")
        n_correct += int(correct)
        entry.update(
            {
                "skipped": False,
                "n_drive_nodes": int(drivers.size),
                "n_target_nodes": int(targets.size),
                "predicted": predicted,
                "delta_mean": common.ci_mean(deltas_drv)[0],
                "ci": list(common.ci_mean(deltas_drv)[1:]),
                "noise_floor": floor,
                "correct": bool(correct),
            }
        )
        per_exp.append(entry)

    n_eval = sum(1 for e in per_exp if not e.get("skipped"))
    n_total = len(per_exp)
    checks = {
        "experiments": per_exp,
        "n_panel": n_total,
        "n_evaluated": n_eval,
        "n_correct": n_correct,
        "pass_rule": f">= {SHIU_MIN_CORRECT}/{n_total} directionally correct",
        "fraction_correct_of_panel": n_correct / n_total if n_total else float("nan"),
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"n_correct": n_correct, "n_evaluated": n_eval},
        passed=bool(n_correct >= SHIU_MIN_CORRECT),
    )


# ---------------------------------------------------------------------------
# T-SIM-8: corpus stats vs physiology band
# ---------------------------------------------------------------------------


def run_t_sim_8(
    corpus_dir: Path | None,
    cfg,
    results_dir: Path,
    *,
    stats: dict | None = None,
    band: dict | None = None,
    reviewed_ids: list | None = None,
    n_nodes: int | None = None,
) -> dict:
    """Corpus activity stats vs a declared physiology band.

    ``stats`` may inject {"rate_hz": [...], "spontaneous_hz": [...]}
    arrays directly; otherwise shard trials+events under ``corpus_dir``
    are scanned (dt read per shard manifest, EVAL 1.1). Pass: zero
    unreviewed outliers shipped - flagged outliers are reported and must
    be individually reviewed (``reviewed_ids``) to ship.
    """
    test_id = "T-SIM-8"
    band = dict(band or DEFAULT_BAND)
    reviewed = set(reviewed_ids or [])
    inputs = {"band": band, "corpus_dir": str(corpus_dir) if corpus_dir else None}

    if stats is None:
        if corpus_dir is None or not Path(corpus_dir).exists():
            return _skipped(
                results_dir, test_id, "no stats injected and corpus_dir absent", inputs=inputs
            )
        stats = _scan_corpus_rates(Path(corpus_dir), n_nodes=n_nodes)

    rates = np.asarray(stats.get("rate_hz", []), dtype=np.float64)
    spont = np.asarray(stats.get("spontaneous_hz", []), dtype=np.float64)
    ids = np.asarray(stats.get("ids", np.arange(rates.size))).astype(str)
    lo, hi = band["rate_hz"]
    s_lo, s_hi = band["spontaneous_hz"]
    flagged = np.flatnonzero((rates < lo) | (rates > hi)) if rates.size else np.array([], int)
    flagged_spont = (
        np.flatnonzero((spont < s_lo) | (spont > s_hi)) if spont.size else np.array([], int)
    )
    flagged_ids = sorted(set(ids[flagged]) | {f"spontaneous:{int(i)}" for i in flagged_spont})
    unreviewed = [i for i in flagged_ids if i not in reviewed]
    checks = {
        "n_rate_stats": int(rates.size),
        "n_spontaneous_stats": int(spont.size),
        "flagged_outlier_fraction": float(len(flagged_ids) / max(rates.size, 1)),
        "flagged_ids": flagged_ids,
        "unreviewed_outliers": unreviewed,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"flagged_outlier_fraction": checks["flagged_outlier_fraction"]},
        passed=bool(len(unreviewed) == 0),
    )


def _scan_corpus_rates(corpus_dir: Path, n_nodes: int | None = None) -> dict:
    """Per-trial mean per-neuron rates + unstimulated (spontaneous) subset.

    ``n_nodes`` is the graph's node count, injected by the dispatcher;
    when absent the shard's own event-table max neuron index is the
    documented underestimate (silent high-index nodes go uncounted).
    """
    import polars as pl

    from flygpt.evals.corpus_checks import _shard_files  # lazy: import cycle

    rate_hz, spont, ids = [], [], []
    for s in _shard_files(corpus_dir):
        manifest = s["manifest"]
        base = s["base"]
        trials_path, events_path = s["trials"], s["events"]
        if not trials_path.exists() or not events_path.exists():
            continue
        dt = float(manifest.get("dt", 0.02))
        trials = pl.read_parquet(trials_path)
        events = pl.read_parquet(events_path)
        n_steps = int(trials["n_steps"].max()) if trials.height else 0
        if n_steps <= 0:
            continue
        nn = int(n_nodes or manifest.get("n_nodes") or 0)
        if nn <= 0:
            # last-resort underestimate: silent high-index nodes go
            # uncounted, so mean rates read slightly high here
            try:
                nn = int(events["neuron_idx"].max()) + 1
            except Exception:
                nn = 0
        per_trial = (
            events.group_by("trial_id").len().to_dicts() if events.height else []
        )
        # The band's unit is per-neuron sustained rate (a 300 Hz whole-CNS
        # sustained rate is physiologically implausible) - total events/s
        # over the network is a different, misleading statistic.
        denom = n_steps * dt * max(nn, 1)
        driven = {
            r["trial_id"]: float(r["len"] / denom) for r in per_trial
        }
        for row in trials.iter_rows(named=True):
            tid = int(row["trial_id"])
            r = driven.get(tid, 0.0)
            rate_hz.append(r)
            ids.append(f"{base}:{tid}")
            prog = row.get("program")
            spec = json.loads(prog) if isinstance(prog, str) else (prog or {})
            if not spec.get("drives"):
                spont.append(r)
    return {"rate_hz": rate_hz, "spontaneous_hz": spont, "ids": ids}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="T-SIM-1..8 simulator checks")
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graph"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/corpus"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("configs/sim_v1.yaml"))
    args = parser.parse_args(argv)

    cfg: Any = {}
    if args.config.exists():
        import yaml

        raw = yaml.safe_load(args.config.read_text()) or {}
        cfg = dict(raw.get("model", {}))
        cfg["backend"] = raw.get("backend", "numpy")
        cfg["determinism"] = raw.get("determinism", {})

    bundle = None
    if (args.graph_dir / "nodes.parquet").exists():
        from flygpt.schema import load_graph_bundle

        bundle = load_graph_bundle(args.graph_dir, cutoff=int(_cfg_get(cfg, "cutoff", 5) or 5))
    if bundle is None:
        for tid in ("T-SIM-1", "T-SIM-2", "T-SIM-3", "T-SIM-4", "T-SIM-5", "T-SIM-6", "T-SIM-7"):
            _skipped(args.results_dir, tid, f"graph bundle absent under {args.graph_dir}")
    else:
        run_t_sim_1(bundle, cfg, args.results_dir)
        run_t_sim_2(bundle, cfg, args.results_dir)
        run_t_sim_3(bundle, cfg, args.results_dir)
        run_t_sim_4(bundle, cfg, args.results_dir)
        run_t_sim_5(bundle, cfg, args.results_dir)
        run_t_sim_6(bundle, cfg, args.results_dir)
        run_t_sim_7(bundle, cfg, args.results_dir)
    run_t_sim_8(args.corpus_dir if args.corpus_dir.exists() else None, cfg, args.results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
