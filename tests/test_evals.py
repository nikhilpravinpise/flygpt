"""T-EVAL battery tests - metric math + check runners on synthetic fixtures.

Two layers, mirroring tests/test_graph.py:

1. Metric-library tests (evals/common.py): hand-computed expectations for
   every EVAL.md section-1 definition - event-F1 inclusion rules, rate
   correlation qualification, 1.4 agreement regimes, CI/FDR math, the
   silence guard, and the 1.5 response rule.

2. Runner tests: every T-SIM/T-CORPUS/T-TOK/T-STIM check executes against
   injected fakes (a deterministic discrete event sim, a mock sensory
   index, hand-built parquet shards) - no pipeline deps beyond the eval
   modules themselves. Pass AND fail verdicts are asserted; the tests die
   if a check passes something broken.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from scipy import stats as _stats

from flygpt.evals import common
from flygpt.evals import corpus_checks as cc
from flygpt.evals import sim_checks as sc
from flygpt.evals import stim_checks as st
from flygpt.evals import tok_checks as tk
from flygpt.schema import (
    EVENT_COLUMNS,
    TRIAL_COLUMNS,
    TYPE_TOKEN_BASE,
    VOCAB_UNK,
    rng_from_seed,
    write_result,
)

CFG = {"dt": 0.02, "backend": "numpy"}
PROP_GAIN = 20.0


# ---------------------------------------------------------------------------
# Deterministic fake sim honoring the sim_checks program contract
# ---------------------------------------------------------------------------


def _fake_run(bundle, program, seed, *, prop_gain=PROP_GAIN, honor_silence=True,
              intrinsic_hz=0.0, same_step=False, deterministic=True, extra_rate_fn=None):
    """Minimal discrete event sim used to drive the check runners.

    Semantics (deliberately LIF-flavored, but tiny):
    - per-(step, node) uniform draws come from a Generator seeded by
      ``seed`` alone, so two programs sharing a seed see the identical
      draw at each (t, j) - baseline arms are exactly comparable.
    - rate(j, t) = tonic_rate_hz + drive rates in-window + intrinsic_hz
      + propagated boost; fire Bernoulli(rate * dt).
    - propagation: each fired pre contributes sign * weight * prop_gain
      Hz to each postsyn target at t+1 (or, for the buggy ``same_step``
      variant, at t - the causality violation T-SIM-3 must catch).
    - program["scripted"] = {node: [steps]} forces fires (health/corpus
      regeneration tests); program["pulse"] injects exact spikes.
    - silences drop the node's fires from the record AND the propagating
      set in-window (honor_silence); the buggy variant still propagates
      them - the phantom-effect T-STIM-5 must catch.
    """
    dt = float(program.get("dt", 0.02))
    n_steps = int(program.get("n_steps", 50))
    n = bundle.n_nodes
    tonic = float(program.get("tonic_rate_hz") or 0.0) + intrinsic_hz
    drives = [dict(d) for d in (program.get("drives") or [])]
    silences = [dict(s) for s in (program.get("silences") or [])]
    pulse = program.get("pulse") or {}
    scripted = {
        int(k): {int(x) for x in v}
        for k, v in (program.get("scripted") or {}).items()
    }

    if deterministic:
        u = rng_from_seed("fakesim", int(seed)).random((n_steps, n))
    else:
        u = np.random.default_rng().random((n_steps, n))

    def _window(d):
        w = d.get("window") if d.get("window") is not None else d.get("window_s")
        return float(w[0]), float(w[1])

    def in_silence(node, t):
        tt = t * dt
        for s in silences:
            w0, w1 = _window(s)
            if w0 <= tt < w1 and node in s["nodes"]:
                return True
        return False

    def boost_of(fired_set):
        nxt = np.zeros(n)
        for i in fired_set:
            for e in range(bundle.out_indptr[i], bundle.out_indptr[i + 1]):
                j = int(bundle.out_indices[e])
                nxt[j] += (
                    float(bundle.out_sign[e]) * float(bundle.out_weight[e]) * prop_gain
                )
        return nxt

    boost_now = np.zeros(n)
    es, en = [], []
    for t in range(n_steps):
        rate = np.full(n, tonic) + boost_now
        for d in drives:
            w0, w1 = _window(d)
            if w0 <= t * dt < w1:
                for nd in d["nodes"]:
                    rate[int(nd)] += float(d["rate_hz"])
        if extra_rate_fn is not None:
            rate = rate + extra_rate_fn(t, dt, program)
        p = np.clip(rate * dt, 0.0, 1.0)
        fired = set(np.flatnonzero(u[t] < p).tolist())
        for nd, steps in scripted.items():
            if t in steps:
                fired.add(nd)
        if pulse.get("node") is not None and t in set(pulse.get("steps", [])):
            fired.add(int(pulse["node"]))
        if same_step and fired:
            # buggy variant: propagation lands in the SAME step
            nxt_same = boost_of(fired)
            p2 = np.clip((rate + nxt_same) * dt, 0.0, 1.0)
            fired |= set(np.flatnonzero((u[t] < p2) & (u[t] >= p)).tolist())
        sil_now = {j for j in fired if in_silence(j, t)}
        recorded = fired - sil_now
        propagating = recorded if honor_silence else set(fired)
        for j in sorted(recorded):
            es.append(t)
            en.append(j)
        boost_now = boost_of(propagating)
    return {
        "events_step": np.asarray(es, dtype=np.int64),
        "events_node": np.asarray(en, dtype=np.int64),
        "n_steps": n_steps,
        "n_nodes": n,
    }


def make_sim(**kw):
    def sim(bundle, program, cfg, seed=0):
        return _fake_run(bundle, program, seed, **kw)

    return sim


def make_shiu_sim(target_map, effects, base_hz=25.0):
    """Fake sim for T-SIM-7: planted per-experiment target rate effects.

    ``effects[exp_id]`` Hz is added to that experiment's target nodes
    during the drive window, in the driven arm only (drives non-empty).
    ``base_hz`` gives targets intrinsic activity so decreases and the
    no-effect noise floor are measurable.
    """

    def sim(bundle, program, cfg, seed=0):
        exp = program.get("experiment_id")
        targets = np.asarray(target_map.get(exp, []), dtype=np.int64)
        d0 = (program.get("drives") or [{}])[0]
        window = d0.get("window") or d0.get("window_s") or [0.0, 0.0]

        def extra(t, dt, prog):
            r = np.zeros(bundle.n_nodes)
            if targets.size:
                r[targets] += base_hz
                if prog.get("drives") and window[0] <= t * dt < window[1]:
                    r[targets] += float(effects.get(exp, 0.0))
            return r

        return _fake_run(bundle, program, seed, extra_rate_fn=extra)

    return sim


class MockIndex:
    """Query resolver stand-in for SensoryIndex: table of {key: {value: [nodes]}}."""

    def __init__(self, table):
        self.table = table

    def resolve(self, *, types=None, classes=None, subclasses=None, superclasses=None,
                sides=None, neuromeres=None, entry_nerves=None):
        out: list[int] = []
        for kw, vals in (
            ("types", types), ("classes", classes), ("subclasses", subclasses),
            ("superclasses", superclasses), ("sides", sides),
            ("neuromeres", neuromeres), ("entry_nerves", entry_nerves),
        ):
            for v in vals or []:
                out += list(self.table.get(kw, {}).get(v, []))
        return np.array(sorted(set(out)), dtype=np.int64)


# ---------------------------------------------------------------------------
# Corpus fabrication helpers (schema_v1 shard layout)
# ---------------------------------------------------------------------------


def _trial_row(trial_id, family="fam_a", program=None, seed=0, dt=0.02, n_steps=25,
               split="train", slice_="main", degenerate=False, n_events=0,
               n_active_neurons=0, n_active_types=0):
    return {
        "trial_id": int(trial_id),
        "family_id": family,
        "program": json.dumps(program or {}),
        "seed": int(seed),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "warmup_steps": 0,
        "n_events": int(n_events),
        "n_active_neurons": int(n_active_neurons),
        "n_active_types": int(n_active_types),
        "split": split,
        "degenerate": bool(degenerate),
        "slice": slice_,
    }


_TRIAL_DTYPES = {
    "trial_id": pl.UInt64, "family_id": pl.String, "program": pl.String,
    "seed": pl.UInt64, "dt": pl.Float64, "n_steps": pl.UInt32,
    "warmup_steps": pl.UInt32, "n_events": pl.UInt32,
    "n_active_neurons": pl.UInt32, "n_active_types": pl.UInt32,
    "split": pl.String, "degenerate": pl.Boolean, "slice": pl.String,
}


def _write_shard(corpus_dir, shard_id, trial_rows, event_rows, *, sort_events=True,
                 manifest_overrides=None, bad_sha=False):
    """Write a schema_v1 shard triple in the corpus.write_shard layout:
    ``events-<sid>.parquet`` + ``trials-<sid>.parquet`` + ``<sid>.manifest.json``
    (one slice/split pair per shard, real TRIAL_SCHEMA dtypes)."""
    from flygpt.schema import sha256_file

    ev = pl.DataFrame(
        {
            "trial_id": [int(r[0]) for r in event_rows],
            "step": [int(r[1]) for r in event_rows],
            "neuron_idx": [int(r[2]) for r in event_rows],
        },
        schema={"trial_id": pl.UInt64, "step": pl.UInt32, "neuron_idx": pl.UInt32},
    )
    if sort_events:
        ev = ev.sort(list(EVENT_COLUMNS))
    tr = pl.DataFrame(trial_rows).cast(_TRIAL_DTYPES) if trial_rows else pl.DataFrame(
        {c: [] for c in TRIAL_COLUMNS}
    ).cast(_TRIAL_DTYPES)
    ev_path = corpus_dir / f"events-{shard_id}.parquet"
    tr_path = corpus_dir / f"trials-{shard_id}.parquet"
    ev.write_parquet(ev_path)
    tr.write_parquet(tr_path)
    ev_sha = sha256_file(ev_path)
    tr_sha = sha256_file(tr_path)
    slice_ = trial_rows[0]["slice"] if trial_rows else "main"
    split = trial_rows[0]["split"] if trial_rows else "train"
    dt = trial_rows[0]["dt"] if trial_rows else 0.02
    manifest = {
        "schema": "schema_v1",
        "corpus_version": "corpus_test",
        "shard_id": shard_id,
        "code": {"git_sha": "deadbeefcafe"},
        "graph": {"version": "graph_test", "sha256": "ab" * 32},
        "files": {
            "events": {"name": ev_path.name, "sha256": ev_sha},
            "trials": {"name": tr_path.name, "sha256": tr_sha},
        },
        "sim_cfg": {"dt": dt},
        "backend": "numpy",
        "gpu_class": "cpu",
        "dt": dt,
        "cutoff": 5,
        "slice": slice_,
        "split": split,
        "trial_ids": sorted(r["trial_id"] for r in trial_rows),
        "n_trials": len(trial_rows),
        "n_events": len(event_rows),
        "source_manifest": "corpus.manifest.json",
        "created_utc": "2026-01-01T00:00:00Z",
        "shard_sha256": "00" * 32 if bad_sha else ev_sha,
    }
    if bad_sha:
        manifest["files"]["events"]["sha256"] = "00" * 32
    if manifest_overrides:
        manifest.update(manifest_overrides)
    (corpus_dir / f"{shard_id}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


NODE_TYPE = np.array(["A_R", "B_L", "A_R", "", "C_M", "D_L"])
NODE_REGION = np.array(["R1", "R1", "R1", "R1", "R2", "R1"])
MANIFEST_TYPES = ["A_R", "B_L", "C_M", "D_L"]


def _write_corpus_manifest(corpus_dir, **kw):
    manifest = {
        "schema": "schema_v1",
        "corpus_version": "corpus_test",
        "types": MANIFEST_TYPES,
        "node_type": NODE_TYPE.tolist(),
        "node_region": NODE_REGION.tolist(),
        "splits": kw.get("splits", {}),
    }
    manifest.update({k: v for k, v in kw.items() if k != "splits"})
    (corpus_dir / "corpus.manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ===========================================================================
# 1. Metric-library tests (evals/common.py, EVAL.md section 1)
# ===========================================================================


def test_event_f1_hand_computed():
    y_true = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 0, 0]])
    y_pred = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
    r = common.event_f1(y_true, y_pred)
    # type0: TP1 FN1 -> 2/3; type1: silent both -> excluded;
    # type2: FN1 FP1 -> 0; type3: silent both -> excluded
    assert r["n_included"] == 2 and r["included_types"] == [0, 2]
    assert r["per_type"] == {"0": pytest.approx(2 / 3), "2": 0.0}
    assert r["macro"] == pytest.approx(1 / 3)
    assert r["micro"] == pytest.approx(0.4)  # TP1 FP1 FN2 pooled


def test_event_f1_inclusion_rule_and_prob_threshold():
    y_true = np.zeros((4, 3), dtype=int)
    y_pred = np.array([[0.6, 0.4, 0.0], [0, 0, 0], [0.5, 0, 0.9], [0, 0, 0]])
    r = common.event_f1(y_true, y_pred)
    # 0.5 threshold fixed: pred events on types 0 (t0,t2) and 2 (t2)
    assert r["included_types"] == [0, 2]
    assert r["per_type"]["0"] == 0.0 and r["per_type"]["2"] == 0.0
    assert r["macro"] == 0.0
    # all-silent split: nothing included, macro is NaN, never rewarded
    empty = common.event_f1(np.zeros((3, 2), dtype=int), np.zeros((3, 2), dtype=int))
    assert empty["n_included"] == 0 and np.isnan(empty["macro"]) and np.isnan(empty["micro"])


def test_rate_pearson_zero_variance_exclusion():
    rng = np.random.default_rng(0)
    t = rng.random((50, 3))
    p = t.copy()
    p[:, 1] = 0.5  # zero-variance predicted column -> nonqualifying
    r = common.rate_pearson(t, p)
    assert r["n_qualifying"] == 2 and r["n_nonqualifying"] == 1
    assert set(r["per_type"]) == {"0", "2"}
    assert r["macro"] == pytest.approx(1.0)
    # all nonqualifying -> macro NaN, counts reported
    r2 = common.rate_pearson(np.ones((5, 2)), np.ones((5, 2)))
    assert r2["n_nonqualifying"] == 2 and np.isnan(r2["macro"])


def test_rate_spearman_monotone():
    t = np.tile(np.arange(20, dtype=np.float64), (2, 1)).T
    p = t ** 2  # strictly monotone -> rho 1
    r = common.rate_spearman(t, p)
    assert r["n_qualifying"] == 2 and r["macro"] == pytest.approx(1.0)


def test_spike_agreement_exact_and_pm1():
    n = 3
    a = np.zeros((4, n), dtype=bool)
    b = np.zeros((4, n), dtype=bool)
    a[0, 0] = a[1, 1] = True
    b[0, 0] = b[2, 1] = True  # n1 shifted by one step -> +-1 matched
    r = common.spike_agreement(a, b, n)
    assert r["event_agreement"] == pytest.approx(10 / 12)
    assert r["pm1_step_agreement"] == pytest.approx(1.0)
    assert r["rate_pearson"] == pytest.approx(1.0)
    # two-step shift is outside the +-1 window
    b2 = np.zeros((5, n), dtype=bool)
    a2 = np.zeros((5, n), dtype=bool)
    a2[1, 1] = True
    b2[3, 1] = True
    r2 = common.spike_agreement(a2, b2, n)
    assert r2["pm1_step_agreement"] == pytest.approx(13 / 15)
    # identical event lists -> everything 1.0
    r3 = common.spike_agreement(a, a, n)
    assert r3["event_agreement"] == 1.0 and r3["rate_pearson"] == pytest.approx(1.0)
    # tuple event-list input form
    r4 = common.spike_agreement((np.array([0, 1]), np.array([0, 1])),
                                (np.array([0, 1]), np.array([0, 1])), n)
    assert r4["event_agreement"] == 1.0
    # zero-variance counts -> unqualified NaN, not a fake 1.0
    r5 = common.spike_agreement(np.zeros((4, n), bool), np.zeros((4, n), bool), n)
    assert np.isnan(r5["rate_pearson"]) and not r5["rate_pearson_qualified"]


def test_events_bitwise_equal():
    a = (np.array([0, 2, 1]), np.array([1, 0, 1]))
    b = (np.array([0, 1, 2]), np.array([1, 1, 0]))  # same set, sorted differs
    assert common.events_bitwise_equal(a, b)
    c = (np.array([0, 1, 3]), np.array([1, 1, 0]))
    assert not common.events_bitwise_equal(a, c)


def test_ci_mean_known_n():
    mean, lo, hi = common.ci_mean([1, 2, 3, 4, 5])
    assert mean == pytest.approx(3.0)
    half = float(_stats.t.ppf(0.975, 4)) * np.std([1, 2, 3, 4, 5], ddof=1) / np.sqrt(5)
    assert lo == pytest.approx(3.0 - half) and hi == pytest.approx(3.0 + half)
    assert common.ci_mean([7.0]) == (7.0, 7.0, 7.0)
    assert all(np.isnan(v) for v in common.ci_mean([]))


def test_bh_fdr_known_plist():
    r = common.bh_fdr([0.001, 0.01, 0.04, 0.2], q=0.05)
    assert r["reject"].tolist() == [True, True, False, False]
    assert r["p_adj"].tolist() == pytest.approx([0.004, 0.02, 0.04 * 4 / 3, 0.2])
    r2 = common.bh_fdr([0.9, 0.95], q=0.05)
    assert not r2["reject"].any()


def test_baselines_and_silence_guard():
    # type0 events in 4/6 bins (majority fires), type1 silent, type2 rare
    y = np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 0]])
    maj = common.majority_baseline(y)
    zero = common.allzero_baseline(y)
    assert maj["event_pred"][:, 0].all()  # marginal 4/6 >= 0.5 -> always predicts
    assert not maj["event_pred"][:, 1].any() and not maj["event_pred"][:, 2].any()
    assert maj["event_f1"]["macro"] > 0.0
    assert zero["event_f1"]["macro"] == 0.0
    guard = common.silence_guard(zero, maj)
    assert guard["ok"] and not guard["void"]
    # violation: a broken metric where all-zero beats majority -> eval void
    fake_zero = {"event_f1": {"macro": 0.9, "micro": 0.9}}
    fake_maj = {"event_f1": {"macro": 0.3, "micro": 0.3}}
    bad = common.silence_guard(fake_zero, fake_maj)
    assert not bad["ok"] and bad["void"]


def test_response_sign_rule():
    assert common.response_sign([10, 12, 11, 13, 10], floor=1.0) == "increase"
    assert common.response_sign([-10, -12, -11, -13, -10], floor=1.0) == "decrease"
    near = [0.01, -0.02, 0.03, -0.01, 0.0]
    assert common.response_sign(near, floor=5.0) == "no-effect"
    # CI spans zero but |estimate| >= floor -> ambiguous, not no-effect
    wide = [50, -40, 60, -30, 45, -35]
    assert common.response_sign(wide, floor=1.0) == "ambiguous"
    assert common.noise_floor([1, 2, 3, 4, 100], pct=95) == pytest.approx(
        np.percentile([1, 2, 3, 4, 100], 95)
    )


def test_coverage_by_type_eval18():
    node_type = np.array(["A", "B", "", "C"])
    ev_trial = np.array([0] * 5 + [1] * 5 + list(range(12)) + [0] * 3)
    ev_node = np.array([0] * 5 + [0] * 5 + [1] * 12 + [2] * 3)
    cov = common.coverage_by_type(ev_trial, ev_node, node_type, ["A", "B", "C"])
    # A: 10 events in trials 0,1 -> driven, not robust (<10 trials)
    # B: 12 events across 12 trials -> driven + robust; C: 0 -> neither
    assert cov["n_driven"] == 2 and cov["n_robust"] == 1
    assert cov["per_type"]["A"]["driven"] and not cov["per_type"]["A"]["robust"]
    assert cov["per_type"]["B"]["robust"]
    assert cov["driven_fraction"] == pytest.approx(2 / 3)


def test_rate_from_events_and_binarize():
    r = common.rate_from_events(np.array([0, 0, 5]), np.array([1, 1, 2]), 10, 0.02, n_nodes=4)
    assert r.tolist() == pytest.approx([0, 10, 5, 0])
    assert common.binarize_events(np.array([0.49, 0.5, 0.9])).tolist() == [False, True, True]


def test_binomial_and_bootstrap_helpers():
    r = common.binomial_test(29, 30, 0.5)
    assert r["p"] < 1e-5
    b = common.bootstrap_sign_test([1.0, 2.0, 1.5, 1.2, 0.8, 2.2], seed=123)
    assert 0.0 <= b["p"] <= 1.0 and b["mean"] == pytest.approx(np.mean([1, 2, 1.5, 1.2, 0.8, 2.2]))


def test_write_result_envelope(tmp_path):
    write_result(tmp_path, "T-X", checks={"a": 1}, inputs={"i": 2},
                       metrics={"m": 3}, passed=True)
    on_disk = json.loads((tmp_path / "T-X.json").read_text())
    for key in ("artifact", "schema", "created_utc", "code", "inputs", "checks", "metrics", "pass"):
        assert key in on_disk
    assert on_disk["artifact"] == "T-X" and on_disk["pass"] is True
    assert on_disk["code"]["git_sha"] and on_disk["schema"] == "results_v1"


# ===========================================================================
# 2. sim_checks runners on the synthetic bundle + fake sim
# ===========================================================================


def test_t_sim_1_numpy_bitwise(synthetic_bundle, tmp_path):
    sim = make_sim()
    r = sc.run_t_sim_1(synthetic_bundle, CFG, tmp_path, simulate_fn=sim, n_steps=30)
    assert r["pass"] is True
    assert r["checks"]["regime"] == "cpu_bitwise" and r["checks"]["bitwise_identical"]


def test_t_sim_1_nondeterministic_fails(synthetic_bundle, tmp_path):
    sim = make_sim(deterministic=False)
    r = sc.run_t_sim_1(synthetic_bundle, CFG, tmp_path, simulate_fn=sim, n_steps=30)
    assert r["pass"] is False and not r["checks"]["bitwise_identical"]


def test_t_sim_2_cross_backend(synthetic_bundle, tmp_path):
    sim = make_sim()
    r = sc.run_t_sim_2(synthetic_bundle, CFG, tmp_path,
                       simulate_fn=sim, simulate_fn_b=sim, n_steps=30)
    # identical fake on both sides -> trivially within tolerance
    assert r["pass"] is True and r["checks"]["regime"] == "cross_backend"


def test_t_sim_3_pulse_confined(synthetic_bundle, tmp_path):
    sim = make_sim()
    r = sc.run_t_sim_3(synthetic_bundle, CFG, tmp_path, simulate_fn=sim,
                       pulse_node=0, pulse_step=10, n_steps=40)
    assert r["pass"] is True
    assert r["checks"]["same_step_changes_beyond_pulse"] == []
    assert set(r["checks"]["t1_changes"]) <= {0, 1, 2}  # out(0) = {1, 2}


def test_t_sim_3_catches_same_step_leak(synthetic_bundle, tmp_path):
    sim = make_sim(same_step=True)
    r = sc.run_t_sim_3(synthetic_bundle, CFG, tmp_path, simulate_fn=sim,
                       pulse_node=0, pulse_step=10, n_steps=40)
    assert r["pass"] is False
    assert r["checks"]["same_step_changes_beyond_pulse"] != []


def test_t_sim_4_known_sign_panel(synthetic_bundle, tmp_path):
    sim = make_sim()
    r = sc.run_t_sim_4(synthetic_bundle, CFG, tmp_path, simulate_fn=sim,
                       n_pairs=20, n_trials=8, rate_hz=300.0, duration_s=1.0,
                       exc_tonic_rate_hz=10.0, inh_tonic_rate_hz=10.0,
                       min_weight=1)
    assert r["checks"]["n_pairs_evaluated"] == 20  # resampled from 3 eligible edges
    assert r["pass"] is True and r["checks"]["fraction_correct"] == 1.0
    signs = {(p["pre"], p["post"]): p["edge_sign"] for p in r["checks"]["pairs"]}
    assert signs.get((0, 2)) == -1  # inhibitory edge preserved in panel


def test_t_sim_5_silence_and_tonic(synthetic_bundle, tmp_path):
    r = sc.run_t_sim_5(synthetic_bundle, CFG, tmp_path, simulate_fn=make_sim(), n_steps=40)
    assert r["pass"] is True
    bad = sc.run_t_sim_5(synthetic_bundle, CFG, tmp_path,
                         simulate_fn=make_sim(intrinsic_hz=5.0), n_steps=40)
    assert bad["pass"] is False and bad["checks"]["zero_baseline_events"] > 0


def test_t_sim_6_health_metrics(synthetic_bundle, tmp_path):
    # scripted: only nodes 0 (A_R) + 4 (C_M) fire, periodically ->
    # active-type fraction 0.5, split-half r = 1, never degenerate.
    # prop_gain=0 keeps the scripted cells from recruiting partners.
    progs = [
        {"family_id": "pilot_default", "drives": [], "n_steps": 50, "dt": 0.02,
         "scripted": {"0": list(range(0, 50, 3)), "4": list(range(0, 50, 4))}}
        for _ in range(12)
    ]
    r = sc.run_t_sim_6(synthetic_bundle, CFG, tmp_path,
                       simulate_fn=make_sim(prop_gain=0.0), programs=progs)
    assert r["pass"] is True
    assert r["checks"]["degenerate_fraction"] == 0.0
    assert r["checks"]["stationary_fraction"] == 1.0
    assert "pilot_default" in r["checks"]["sane_band_hz"]
    # a fully-active trial is degenerate (>60% of types) and sinks the run
    bad_prog = progs + [{"family_id": "pilot_default", "drives": [], "n_steps": 50,
                         "dt": 0.02,
                         "scripted": {str(n): list(range(50)) for n in (0, 1, 4)}}]
    bad = sc.run_t_sim_6(synthetic_bundle, CFG, tmp_path,
                         simulate_fn=make_sim(prop_gain=0.0), programs=bad_prog)
    assert bad["checks"]["degenerate_fraction"] > 0


def _write_shiu_panel(path, effects_map):
    exps = []
    names = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10"]
    targets = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
    for e, t in zip(names, targets, strict=True):
        expected = ("increase" if effects_map[e] > 0 else
                    "decrease" if effects_map[e] < 0 else "no-effect")
        exps.append({
            "id": e,
            "drive_query": {"types": ["ovulin"] if e == "e10" else ["DRV_A"]},
            "drive_rate_hz": 150,
            "target_query": {"types": [t]},
            "expected": expected,
        })
    (path / "panel.json").write_text(json.dumps({"experiments": exps}))
    return path / "panel.json"


def test_t_sim_7_panel_counting(synthetic_bundle, tmp_path):
    type_table = {
        "DRV_A": [0], "A_R": [0, 2],
        "T1": [3], "T2": [4], "T3": [5], "T4": [3, 4],
        "T5": [4], "T6": [5], "T7": [3], "T8": [4], "T9": [5], "T10": [3],
    }
    index = MockIndex({"types": type_table})
    effects = {"e1": 60, "e2": 60, "e3": 60, "e4": 60,
               "e5": -40, "e6": -40, "e7": -40, "e8": 0, "e9": 0}
    target_map = {"e1": [3], "e2": [4], "e3": [5], "e4": [3, 4],
                  "e5": [4], "e6": [5], "e7": [3], "e8": [4], "e9": [5]}
    sim = make_shiu_sim(target_map, effects)
    panel = _write_shiu_panel(tmp_path, {**effects, "e10": 60})
    r = sc.run_t_sim_7(synthetic_bundle, CFG, tmp_path, index=index, simulate_fn=sim,
                       panel_path=panel, n_trials=30)
    by_id = {e["id"]: e for e in r["checks"]["experiments"]}
    assert by_id["e10"]["skipped"] and "zero nodes" in by_id["e10"]["reason"]
    for e in ("e1", "e2", "e3", "e4"):
        assert by_id[e]["predicted"] == "increase" and by_id[e]["correct"]
    for e in ("e5", "e6", "e7"):
        assert by_id[e]["predicted"] == "decrease" and by_id[e]["correct"]
    for e in ("e8", "e9"):
        assert by_id[e]["predicted"] == "no-effect" and by_id[e]["correct"]
    assert r["checks"]["n_evaluated"] == 9 and r["checks"]["n_correct"] == 9
    assert r["pass"] is True  # >= 8/10
    # runtime type map resolves the reserved slot; the slot then counts
    r2 = sc.run_t_sim_7(synthetic_bundle, CFG, tmp_path, index=index, simulate_fn=sim,
                        panel_path=panel, n_trials=30, drive_type_map={"ovulin": "A_R"})
    by_id2 = {e["id"]: e for e in r2["checks"]["experiments"]}
    assert not by_id2["e10"]["skipped"] and r2["checks"]["n_evaluated"] == 10


def test_t_sim_8_physiology_band(tmp_path):
    stats = {"rate_hz": [10.0, 25.0, 40.0, 350.0],
             "spontaneous_hz": [5.0, 8.0],
             "ids": ["a", "b", "c", "d"]}
    r = sc.run_t_sim_8(None, CFG, tmp_path, stats=stats)
    assert r["pass"] is False and r["checks"]["unreviewed_outliers"] == ["d"]
    ok = sc.run_t_sim_8(None, CFG, tmp_path, stats=stats, reviewed_ids=["d"])
    assert ok["pass"] is True


# ===========================================================================
# 3. corpus_checks runners on fabricated shards
# ===========================================================================


def _mini_corpus(tmp_path, *, mixed_dt=False, missing_slice=False, unsorted=False,
                 bad_sha=False, splits=None, train_drive_nodes=(0,)):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    trials, events = [], []
    for tid in range(10):
        trials.append(_trial_row(
            tid, family="fam_a" if tid < 5 else "fam_b",
            program={"drives": [{"nodes": list(train_drive_nodes), "rate_hz": 100,
                                 "window_s": [0, 0.5]}], "n_steps": 25, "dt": 0.02},
            seed=tid, dt=0.02 if not mixed_dt or tid < 5 else 0.005,
            split="train" if tid < 5 else "test",
            slice_="" if missing_slice and tid == 0 else "main",
            n_events=24, n_active_neurons=4, n_active_types=4))
        events += [(tid, s, nd) for nd in (0, 1, 4, 5) for s in range(6)]
    _write_shard(corpus_dir, "shard_00000", trials, events,
                 sort_events=not unsorted, bad_sha=bad_sha)
    _write_corpus_manifest(corpus_dir, splits=splits or {})
    return corpus_dir


def test_t_corpus_1_schema(tmp_path):
    corpus_dir = _mini_corpus(tmp_path)
    r = cc.run_t_corpus_1(corpus_dir, tmp_path)
    assert r["pass"] is True and r["checks"]["n_shards"] == 1


def test_t_corpus_1_unsorted_fails(tmp_path):
    corpus_dir = _mini_corpus(tmp_path, unsorted=True)
    # rewrite events unsorted by shuffling rows
    ev_path = corpus_dir / "events-shard_00000.parquet"
    ev = pl.read_parquet(ev_path).sample(fraction=1.0, seed=7, shuffle=True)
    ev.write_parquet(ev_path)
    r = cc.run_t_corpus_1(corpus_dir, tmp_path)
    assert r["pass"] is False
    assert any("sorted" in p for v in r["checks"]["invalid"].values() for p in v["problems"])


def test_t_corpus_2_provenance(tmp_path):
    corpus_dir = _mini_corpus(tmp_path)
    assert cc.run_t_corpus_2(corpus_dir, tmp_path)["pass"] is True
    corpus_dir2 = _mini_corpus(tmp_path / "b", bad_sha=True)
    r = cc.run_t_corpus_2(corpus_dir2, tmp_path / "res2")
    assert r["pass"] is False
    assert "sha256 mismatch" in json.dumps(r["checks"]["per_shard"])


def test_t_corpus_3_coverage(tmp_path):
    corpus_dir = _mini_corpus(tmp_path)
    r = cc.run_t_corpus_3(corpus_dir, tmp_path)
    assert r["pass"] is True
    assert r["metrics"]["driven_fraction"] == 1.0 and r["metrics"]["robust_fraction"] == 1.0
    # starved corpus: only A_R driven -> floors not met
    d2 = tmp_path / "c2"
    d2.mkdir()
    trials = [_trial_row(t, n_events=6) for t in range(10)]
    events = [(t, s, 0) for t in range(10) for s in range(6)]
    _write_shard(d2, "shard_00000", trials, events)
    _write_corpus_manifest(d2)
    r2 = cc.run_t_corpus_3(d2, tmp_path / "r2")
    assert r2["pass"] is False and r2["metrics"]["driven_fraction"] == 0.25


def test_t_corpus_3_scope_precedence(tmp_path):
    """The corpus manifest's own ``types`` beats an injected brain manifest:
    a scoped corpus is judged against what it declared it would drive."""
    corpus_dir = _mini_corpus(tmp_path)
    # every node is covered by the mini corpus, but pretend the corpus
    # declared only a 2-type scope
    _write_corpus_manifest(corpus_dir, types=["A_R", "B_L"], splits={})
    r = cc.run_t_corpus_3(corpus_dir, tmp_path, manifest_types=list(MANIFEST_TYPES))
    assert r["checks"]["n_types"] == 2
    assert r["metrics"]["driven_fraction"] == 1.0
    # and a scope including an undriven type fails the floor honestly
    _write_corpus_manifest(corpus_dir, types=["A_R", "NOPE"], splits={})
    r2 = cc.run_t_corpus_3(corpus_dir, tmp_path / "r2", manifest_types=list(MANIFEST_TYPES))
    assert r2["metrics"]["driven_fraction"] == 0.5 and r2["pass"] is False


def test_t_corpus_4_pilot_scope_no_axis_holdouts(tmp_path):
    """A scope-limited corpus (pilot: splits={"pilot":"train"}) declares no
    axis holdouts - regeneration is skipped, axis hygiene still checked."""
    corpus_dir = _mini_corpus(tmp_path, splits={"pilot": "train"})
    regen = lambda manifest: {  # noqa: E731 - regeneration would produce phantom axes
        "stimulus_holdout": ["fam_b"], "celltype_holdout": ["D_L"],
        "region_holdout": ["R2"],
    }
    r = cc.run_t_corpus_4(corpus_dir, tmp_path, split_fn=regen)
    assert r["checks"]["regeneration"]["skipped"] == (
        "no axis holdouts declared in manifest"
    )
    assert r["checks"]["problems"] == []
    assert r["pass"] is True


def test_t_corpus_4_split_hygiene(tmp_path):
    splits = {
        "stimulus_holdout": ["fam_b"],
        "celltype_holdout": ["D_L"],
        "region_holdout": ["R2"],
    }
    corpus_dir = _mini_corpus(tmp_path, splits=splits, train_drive_nodes=(0,))
    regen = lambda manifest: manifest["splits"]  # noqa: E731
    r = cc.run_t_corpus_4(corpus_dir, tmp_path, split_fn=regen)
    assert r["pass"] is True, r["checks"]["problems"]
    # leaky train trial drives the held-out type's node (5 = D_L)
    corpus_dir2 = _mini_corpus(tmp_path / "b", splits=splits, train_drive_nodes=(5,))
    r2 = cc.run_t_corpus_4(corpus_dir2, tmp_path / "r2", split_fn=regen)
    assert r2["pass"] is False
    assert r2["checks"]["axes"]["cell_types"]["holdout_driven_in_train"] == ["D_L"]
    # region leak via node 4 (R2)
    corpus_dir3 = _mini_corpus(tmp_path / "c", splits=splits, train_drive_nodes=(4,))
    r3 = cc.run_t_corpus_4(corpus_dir3, tmp_path / "r3", split_fn=regen)
    assert r3["checks"]["axes"]["regions"]["holdout_driven_in_train"] == ["R2"]
    # regeneration mismatch fails closed
    bad_regen = lambda manifest: {"stimulus_holdout": [], "celltype_holdout": [], "region_holdout": []}  # noqa: E731
    r4 = cc.run_t_corpus_4(corpus_dir, tmp_path / "r4", split_fn=bad_regen)
    assert r4["pass"] is False


def test_t_corpus_4_family_atomicity(tmp_path):
    corpus_dir = _mini_corpus(tmp_path)  # fam_a train, fam_b test - atomic
    r = cc.run_t_corpus_4(corpus_dir, tmp_path)
    assert r["pass"] is True


def test_t_corpus_5_slice_labels(tmp_path):
    corpus_dir = _mini_corpus(tmp_path)
    assert cc.run_t_corpus_5(corpus_dir, tmp_path)["pass"] is True
    corpus_dir2 = _mini_corpus(tmp_path / "b", missing_slice=True)
    r2 = cc.run_t_corpus_5(corpus_dir2, tmp_path / "r2")
    assert r2["pass"] is False
    # mixed slices within a split
    d3 = tmp_path / "c3"
    d3.mkdir()
    trials = [_trial_row(t, split="train", slice_="main" if t < 5 else "cutoff3")
              for t in range(10)]
    _write_shard(d3, "shard_00000", trials, [])
    _write_corpus_manifest(d3)
    r3 = cc.run_t_corpus_5(d3, tmp_path / "r3")
    assert r3["pass"] is False and "train" in r3["checks"]["mixed_splits"]


def test_t_corpus_7_dt_integrity(tmp_path):
    corpus_dir = _mini_corpus(tmp_path)
    assert cc.run_t_corpus_7(corpus_dir, tmp_path)["pass"] is True
    corpus_dir2 = _mini_corpus(tmp_path / "b", mixed_dt=True)
    r2 = cc.run_t_corpus_7(corpus_dir2, tmp_path / "r2")
    assert r2["pass"] is False and "train" in r2["checks"]["mixed_dt_splits"]


def test_t_corpus_6_regeneration(synthetic_bundle, tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    trials, events = [], []
    for tid in range(3):
        prog = {"scripted": {"0": [0, 2, 4], "1": [1, 3]}, "n_steps": 10, "dt": 0.02}
        trials.append(_trial_row(100 + tid, program=prog, seed=tid, n_steps=10))
        # stored events generated with the same fake the runner will use
        out = _fake_run(synthetic_bundle, prog, tid)
        events += [(100 + tid, s, n) for s, n in zip(out["events_step"], out["events_node"], strict=True)]
    _write_shard(corpus_dir, "shard_00000", trials, events)
    r = cc.run_t_corpus_6(corpus_dir, tmp_path, bundle=synthetic_bundle,
                          simulate_fn=make_sim(), cfg=CFG)
    assert r["pass"] is True and r["checks"]["bitwise_identical"]
    # tamper with stored events -> not bitwise -> fail (numpy regime)
    ev_path = corpus_dir / "events-shard_00000.parquet"
    ev = pl.read_parquet(ev_path)
    ev = ev.with_columns(pl.col("step").add(1).alias("step")).sort(EVENT_COLUMNS)
    ev.write_parquet(ev_path)
    r2 = cc.run_t_corpus_6(corpus_dir, tmp_path / "r2", bundle=synthetic_bundle,
                           simulate_fn=make_sim(), cfg=CFG)
    assert r2["pass"] is False and not r2["checks"]["bitwise_identical"]


# ===========================================================================
# 4. tok_checks on a synthetic vocab
# ===========================================================================


class _Vocab:
    def __init__(self, type_to_id, instance_to_id=None, stimulus_to_id=None,
                 behavior_to_id=None, size=None):
        self.type_to_id = dict(type_to_id)
        self.instance_to_id = dict(instance_to_id or {})
        self.stimulus_to_id = dict(stimulus_to_id or {})
        self.behavior_to_id = dict(behavior_to_id or {})
        self._size = size

    def size(self):
        if self._size is not None:
            return self._size
        ids = ([*self.type_to_id.values(), *self.instance_to_id.values(),
                *self.stimulus_to_id.values(), *self.behavior_to_id.values()])
        return max(ids) + 1 if ids else 0


def _vocab(**kw):
    type_ids = {t: TYPE_TOKEN_BASE + i for i, t in enumerate(MANIFEST_TYPES)}
    inst = kw.get("instance_to_id", {5: TYPE_TOKEN_BASE + len(MANIFEST_TYPES)})
    stim = kw.get("stimulus_to_id", {"olfactory_pulse": 4})
    beh = kw.get("behavior_to_id", {"song": 68})
    return _Vocab(type_ids, inst, stim, beh, size=kw.get("size"))


def _e2c(steps, nodes, node_token, vocab, n_steps=None):
    n_steps = n_steps or (int(steps.max()) + 1 if len(steps) else 0)
    counts = np.zeros((n_steps, vocab.size()), dtype=np.uint32)
    np.add.at(counts, (np.asarray(steps), np.asarray(node_token)[np.asarray(nodes)]), 1)
    return counts


def _c2e(counts, vocab):
    out = []
    for t, tok in zip(*np.nonzero(counts), strict=True):
        out.append((int(t), int(tok), int(counts[t, tok])))
    return out


def _e2c_off_by_one(steps, nodes, node_token, vocab, n_steps=None):
    return _e2c(np.asarray(steps) + 1, nodes, node_token, vocab, n_steps)


def test_t_tok_1_roundtrip(tmp_path):
    vocab = _vocab()
    node_token = np.array([100, 101, 100, VOCAB_UNK, 102, 104])
    events = (np.array([0, 0, 0, 3, 3, 7]), np.array([0, 0, 1, 2, 4, 5]))
    r = tk.run_t_tok_1(vocab, tmp_path, events=events, node_token=node_token,
                       n_steps=10, events_to_counts_fn=_e2c, counts_to_events_fn=_c2e)
    assert r["pass"] is True and r["checks"]["n_mismatched"] == 0
    bad = tk.run_t_tok_1(vocab, tmp_path, events=events, node_token=node_token,
                         n_steps=10, events_to_counts_fn=_e2c_off_by_one,
                         counts_to_events_fn=_c2e)
    assert bad["pass"] is False and bad["checks"]["n_mismatched"] > 0


def test_t_tok_2_collisions(tmp_path):
    assert tk.run_t_tok_2(_vocab(), tmp_path)["pass"] is True
    colliding = _vocab(instance_to_id={5: TYPE_TOKEN_BASE})  # overlaps type A_R id
    r = tk.run_t_tok_2(colliding, tmp_path / "r2")
    assert r["pass"] is False and r["checks"]["n_collisions"] >= 1
    out_of_range = _vocab(stimulus_to_id={"olfactory_pulse": 200})
    assert tk.run_t_tok_2(out_of_range, tmp_path / "r3")["pass"] is False


def test_t_tok_3_unk_routing(tmp_path):
    vocab = _vocab()
    node_token = np.array([100, 101, 100, VOCAB_UNK, 102, 104])
    untyped = np.array([3])
    r = tk.run_t_tok_3(vocab, tmp_path, node_token=node_token, untyped_idx=untyped)
    assert r["pass"] is True and r["checks"]["unk_fraction"] == 1.0
    merged = np.array([100, 101, 100, 101, 102, 104])  # node 3 merged into B_L
    bad = tk.run_t_tok_3(vocab, tmp_path / "r2", node_token=merged, untyped_idx=untyped)
    assert bad["pass"] is False and bad["checks"]["n_dropped_or_merged"] == 1


def test_t_tok_4_bin_boundary(tmp_path):
    r = tk.run_t_tok_4(tmp_path, dt=0.02)
    assert r["pass"] is True and r["checks"]["n_misassigned"] == 0
    def bad_fn(t, t0, dt):
        return np.round((np.asarray(t) - t0) / dt).astype(int)

    bad = tk.run_t_tok_4(tmp_path / "r2", dt=0.02, bin_fn=bad_fn)
    assert bad["pass"] is False


def test_t_tok_5_vocab_size(tmp_path):
    vocab = _vocab()
    r = tk.run_t_tok_5(vocab, tmp_path, manifest_types=MANIFEST_TYPES)
    assert r["pass"] is True
    assert r["checks"]["expected_size"] == TYPE_TOKEN_BASE + 4 + 1
    too_big = _vocab(size=999)
    assert tk.run_t_tok_5(too_big, tmp_path / "r2", manifest_types=MANIFEST_TYPES)["pass"] is False
    missing_type = _Vocab({"A_R": 100, "B_L": 101, "C_M": 102}, {}, {}, {}, size=TYPE_TOKEN_BASE + 3)
    r3 = tk.run_t_tok_5(missing_type, tmp_path / "r3", manifest_types=MANIFEST_TYPES)
    assert r3["pass"] is False and not r3["checks"]["type_map_covers_manifest"]


# ===========================================================================
# 5. stim_checks on synthetic programs + fake sim
# ===========================================================================


def test_t_stim_1_grounding(synthetic_bundle, tmp_path):
    index = MockIndex({"types": {"A_R": [0, 2], "B_L": [1]}})
    lib = [
        {"program_id": "p1", "dt": 0.02,
         "drives": [{"nodes": [0, 2], "rate_hz": 100, "window_s": [0, 0.5]}],
         "meta": {"drive_query": {"types": ["A_R"]}}},
        {"program_id": "p2", "dt": 0.02,
         "drives": [{"nodes": [1], "rate_hz": 100, "window_s": [0, 0.5]}],
         "meta": {"drive_query": {"types": ["B_L"]}}},
    ]
    r = st.run_t_stim_1(lib, synthetic_bundle, tmp_path, index=index)
    assert r["pass"] is True and r["checks"]["n_dangling_refs"] == 0
    bad_lib = lib + [{"program_id": "p3", "dt": 0.02,
                      "drives": [{"nodes": [99], "rate_hz": 100, "window_s": [0, 0.5]}],
                      "meta": {"drive_query": {"types": ["B_L"]}}}]
    r2 = st.run_t_stim_1(bad_lib, synthetic_bundle, tmp_path / "r2", index=index)
    assert r2["pass"] is False and r2["checks"]["n_dangling_refs"] == 1
    mismatch_lib = [{"program_id": "p4", "dt": 0.02,
                     "drives": [{"nodes": [0], "rate_hz": 100, "window_s": [0, 0.5]}],
                     "meta": {"drive_query": {"types": ["A_R"]}}}]
    r3 = st.run_t_stim_1(mismatch_lib, synthetic_bundle, tmp_path / "r3", index=index)
    assert r3["pass"] is False and r3["checks"]["n_resolution_mismatches"] == 1


def test_t_stim_2_laterality(synthetic_bundle, tmp_path):
    # synthetic sides (somaSide): nodes [R, L, R, L, M, L]
    lib_ok = [{"program_id": "l1", "dt": 0.02,
               "drives": [{"nodes": [1, 5], "rate_hz": 100, "window_s": [0, 0.5]}],
               "meta": {"side": "L"}}]
    r = st.run_t_stim_2(lib_ok, synthetic_bundle, tmp_path)
    assert r["pass"] is True and r["checks"]["n_lateralized_checked"] == 1
    lib_bad = [{"program_id": "l2", "dt": 0.02,
                "drives": [{"nodes": [0, 1], "rate_hz": 100, "window_s": [0, 0.5]}],
                "meta": {"side": "L"}}]  # node 0 is R-side
    r2 = st.run_t_stim_2(lib_bad, synthetic_bundle, tmp_path / "r2")
    assert r2["pass"] is False and r2["checks"]["n_violations"] == 1


def test_t_stim_3_tracker_recount(tmp_path):
    corpus_dir = _mini_corpus(tmp_path)
    cov = common.coverage_by_type(*cc._corpus_events(corpus_dir), NODE_TYPE, MANIFEST_TYPES)
    tracker = {t: {"driven": v["driven"], "robust": v["robust"]}
               for t, v in cov["per_type"].items()}
    r = st.run_t_stim_3(tmp_path, tracker_report=tracker, corpus_dir=corpus_dir,
                        node_type=NODE_TYPE, manifest_types=MANIFEST_TYPES)
    assert r["pass"] is True and r["checks"]["n_mismatches"] == 0
    tracker["D_L"]["robust"] = False
    r2 = st.run_t_stim_3(tmp_path / "r2", tracker_report=tracker, corpus_dir=corpus_dir,
                         node_type=NODE_TYPE, manifest_types=MANIFEST_TYPES)
    assert r2["pass"] is False and r2["checks"]["n_mismatches"] == 1


def test_t_stim_4_envelope_bounds(tmp_path):
    dt, w = 0.02, [0.0, 0.2]
    good = [{"program_id": "env1", "dt": dt, "duration_s": 1.0,
             "drives": [{"nodes": [1], "rate_hz": 100, "window_s": w,
                         "envelope": "naturalistic",
                         "envelope_values": list(np.linspace(0.0, 5.0, 10)),
                         "envelope_dt": dt, "envelope_max": 5.0}],
             "meta": {}}]
    r = st.run_t_stim_4(good, tmp_path)
    assert r["pass"] is True and r["checks"]["n_envelope_programs"] == 1
    bad = [{"program_id": "env2", "dt": dt, "duration_s": 1.0,
            "drives": [{"nodes": [1], "rate_hz": 100, "window_s": w,
                        "envelope": "naturalistic",
                        "envelope_values": [-0.5, 6.0, 3.0],
                        "envelope_dt": dt, "envelope_max": 5.0}],
            "meta": {}}]
    r2 = st.run_t_stim_4(bad, tmp_path / "r2")
    assert r2["pass"] is False
    problems = json.dumps(r2["checks"]["per_program"])
    assert "below 0" in problems and "exceeds" in problems


def test_t_stim_5_silencing_semantics(synthetic_bundle, tmp_path):
    r = st.run_t_stim_5(synthetic_bundle, tmp_path, simulate_fn=make_sim(),
                      silence_nodes=np.array([0]), window_s=(0.4, 0.8),
                      n_steps=50, rate_hz=300.0)
    assert r["pass"] is True
    assert r["checks"]["zero_spikes_ok"] and r["checks"]["zero_effect_ok"]
    # buggy sim propagates silenced spikes -> phantom partner events
    leaky = make_sim(honor_silence=False)
    bad = st.run_t_stim_5(synthetic_bundle, tmp_path / "r2", simulate_fn=leaky,
                        silence_nodes=np.array([0]), window_s=(0.4, 0.8),
                        n_steps=50, rate_hz=300.0)
    assert bad["pass"] is False and bad["checks"]["n_phantom_events_elsewhere"] > 0
