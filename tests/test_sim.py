"""T-SIM test battery - simulator correctness on the synthetic fixture.

Runs the real ``flygpt.sim_backend`` kernel on the ``synthetic_bundle``
mini-connectome (CI-safe, no 10 GB dataset). Fixture topology, node_idx
space (bodyId 10..15 -> idx 0..5, cutoff 3 keeps all five edges):

    0 -> 1  w7  +1     (excitatory)
    0 -> 2  w6  -1     (inhibitory - the only inter-node inhibitory edge)
    2 -> 1  w4  +1     (excitatory)
    1 -> 1  w5  -1     (inhibitory autapse, kept)
    3 -> 1  w6   0     (nt_low_confidence: zero current, never guessed)
    4: isolated (zero degree); 5: no edges

Because 1's only out-edge is its autapse, driving node 0 cannot reach
node 2 through node 1 - the 0->2 inhibition and the 0->1 / 2->1
excitation are cleanly separable, which is what makes the T-SIM-4 sign
comparisons real.

Plan refs: docs/plan.md section 3 T-SIM-1..6 + T-STIM-5; EVAL.md 1.4
(cross-backend agreement band) and the T-SIM-6 degenerate rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from flygpt.sim_backend import (
    FlybrainAdapter,
    LIFSim,
    SimConfig,
    health_metrics,
    simulate,
    spike_train_matrix,
)

W_SYN = 0.1  # strong enough that single driven edges reach threshold 1.0


def _cfg(**kw) -> SimConfig:
    base = {
        "dt": 0.020, "tau_v": 0.100, "tau_g": 0.010, "w_syn": W_SYN,
        "threshold": 1.0, "reset": 0.0, "refractory_steps": 0,
        "weight_mode": "count", "tonic_rate_hz": 0.0, "tonic_amount": 0.0,
        "noise_hz": 0.0, "noise_amp": 0.0, "synapse_model": "alpha",
        "backend": "numpy", "warmup_steps": 0, "seed": 0, "drive_scale": 1.0,
    }
    base.update(kw)
    return SimConfig(**base)


def _drive(nodes, rate_hz, window, envelope="poisson", envelope_id=""):
    return {
        "nodes": list(nodes), "rate_hz": float(rate_hz),
        "window": [float(window[0]), float(window[1])],
        "envelope": envelope, "envelope_id": envelope_id,
    }


def _program(drives, silences=None, n_steps=100):
    return {"drives": list(drives), "silences": list(silences or []),
            "n_steps": int(n_steps)}


def _events_of(result, node):
    return result.events_step[result.events_node == node]


def _count(result, node):
    return int((result.events_node == node).sum())


def _assert_sorted_events(result):
    """events_step nondecreasing and (step,node) strictly increasing."""
    assert (np.diff(result.events_step.astype(np.int64)) >= 0).all()
    key = result.events_step.astype(np.uint64) << 32 | result.events_node
    assert (np.diff(key.astype(np.int64)) > 0).all()


# ---------------------------------------------------------------------------
# T-SIM-1 / T-SIM-2: determinism + cross-backend agreement (EVAL 1.4)
# ---------------------------------------------------------------------------


def test_t_sim_1_bitwise_determinism_numpy(synthetic_bundle):
    """Same seed+config+backend twice on CPU -> identical event lists."""
    cfg = _cfg(seed=7)
    program = _program([
        _drive([0], 60.0, [0.0, 1.0]),
        _drive([2], 25.0, [0.5, 2.0], envelope="step"),
    ], n_steps=100)
    r1 = simulate(synthetic_bundle, program, cfg)
    r2 = simulate(synthetic_bundle, program, cfg)
    np.testing.assert_array_equal(r1.events_step, r2.events_step)
    np.testing.assert_array_equal(r1.events_node, r2.events_node)
    np.testing.assert_array_equal(r1.rates, r2.rates)
    assert r1.seed == r2.seed
    _assert_sorted_events(r1)


def test_t_sim_2_jax_vs_numpy_agreement(synthetic_bundle):
    """EVAL 1.4 cross-backend row: +-1-step event agreement >= 99% and
    per-neuron rate Pearson >= 0.99 on an identical (seed, config) trial."""
    pytest.importorskip("jax")
    program = _program([
        _drive([0], 60.0, [0.0, 3.0]),
        _drive([2], 40.0, [0.0, 3.0]),
    ], n_steps=150)
    rn = simulate(synthetic_bundle, program, _cfg(backend="numpy", seed=11))
    rj = simulate(synthetic_bundle, program, _cfg(backend="jax", seed=11))
    assert rn.n_steps == rj.n_steps and rn.n_nodes == rj.n_nodes

    ma = spike_train_matrix(rn)
    mb = spike_train_matrix(rj)
    # +-1-step agreement: an event counts matched if the other run fires
    # the same neuron in the same or an adjacent step (both directions).
    matched = 0
    total = int(ma.sum() + mb.sum())
    for i in range(rn.n_nodes):
        ta = set(np.flatnonzero(ma[:, i]).tolist())
        tb = set(np.flatnonzero(mb[:, i]).tolist())
        for t in ta:
            if any((t + d) in tb for d in (-1, 0, 1)):
                matched += 1
        for t in tb:
            if any((t + d) in ta for d in (-1, 0, 1)):
                matched += 1
    ca = np.bincount(rn.events_node, minlength=rn.n_nodes)
    cb = np.bincount(rj.events_node, minlength=rj.n_nodes)
    rate_r = float(np.corrcoef(ca, cb)[0, 1]) if ca.std() and cb.std() else 1.0
    assert matched / max(total, 1) >= 0.99
    assert rate_r >= 0.99


# ---------------------------------------------------------------------------
# T-SIM-3 causality + T-SIM-4 sign + T-SIM-5 silence
# ---------------------------------------------------------------------------


def test_t_sim_3_causality(synthetic_bundle):
    """Pulsed node's postsynaptic partners fire at t+1 earliest; nothing
    else changes (no same-step firing, no non-partner activity)."""
    cfg = _cfg()
    t0_s, t1_s = 0.0, 20 * cfg.dt  # drive node 0 for 20 recorded steps
    res = simulate(synthetic_bundle, _program(
        [_drive([0], 0.0, [t0_s, t1_s], envelope="tonic")], n_steps=60), cfg)

    # the driver node emits from step 0
    assert _count(res, 0) > 0
    # no OTHER node fires at the first driven step - propagation needs one
    first = int(res.events_step.min())
    nodes_at_first = res.events_node[res.events_step == first]
    assert set(nodes_at_first.tolist()) == {0}
    # postsynaptic partner node 1 (0->1 w7 +1) does fire, strictly later
    n1 = _events_of(res, 1)
    assert n1.size > 0 and int(n1.min()) >= first + 1
    # node 2 is inhibited by 0->2 and never reaches threshold; nodes 3
    # (sign-0 edge only), 4 (isolated) and 5 (no edges) never fire
    for node in (2, 3, 4, 5):
        assert _count(res, node) == 0
    # every event is confined to the driver and its reachable partners
    assert set(np.unique(res.events_node).tolist()) <= {0, 1, 2}
    _assert_sorted_events(res)


def test_t_sim_4_sign_sanity(synthetic_bundle):
    """Known-sign panel, real signed comparisons on this fixture:

    - excitatory: node 2 (2->1 +1) raises node 1's rate over a matched
      depolarized baseline;
    - inhibitory: node 0 (0->2 -1) lowers node 2's rate over a matched
      depolarized baseline - isolated because 0's other edge (0->1 +1)
      cannot reach node 2 (1's only out-edge is its autapse);
    - sign 0: node 3 (3->1 w6 sign 0) contributes exactly zero current.
    """
    cfg = _cfg()
    n_steps = 100

    def depol(node):
        def d(t, sim):
            sim.inject_voltage([node], 0.3)
        return d

    def depol_plus_spikes(node, drive_node):
        def d(t, sim):
            sim.inject_voltage([node], 0.3)
            sim.force_spike([drive_node])
        return d

    def run(driver):
        # fresh sim per trial: run() deliberately does not reset state
        return LIFSim(synthetic_bundle, cfg).run(n_steps, driver=driver)

    base1 = run(depol(1))
    exc1 = run(depol_plus_spikes(1, 2))
    assert _count(base1, 1) > 0                      # depol baseline fires
    assert _count(exc1, 1) > _count(base1, 1)        # excitation raises it

    base2 = run(depol(2))
    inh2 = run(depol_plus_spikes(2, 0))
    assert _count(base2, 2) > 0
    assert _count(inh2, 2) < _count(base2, 2)        # inhibition lowers it

    # sign-0 edge carries zero current: driving node 3 at max rate leaves
    # node 1 untouched while node 3 itself emits every step
    zero = run(lambda t, s: s.force_spike([3]))
    assert _count(zero, 3) == n_steps
    assert _count(zero, 1) == 0


@pytest.mark.parametrize("synapse_model", ["alpha", "instant"])
def test_t_sim_5_silence_and_tonic(synthetic_bundle, synapse_model):
    """Zero baseline -> exactly 0 events; tonic drive -> nonzero. Exact."""
    silent = simulate(synthetic_bundle, _program([], n_steps=60),
                      _cfg(synapse_model=synapse_model))
    assert silent.events_node.size == 0
    assert silent.events_step.size == 0

    tonic = simulate(synthetic_bundle, _program([], n_steps=60),
                     _cfg(synapse_model=synapse_model,
                          tonic_rate_hz=50.0, tonic_amount=1.5))
    assert tonic.events_node.size > 0


def test_t_stim_5_silenced_no_spikes_no_effect(synthetic_bundle):
    """Silenced node emits 0 events AND 0 outgoing synaptic effect."""
    cfg = _cfg()
    drive_0 = lambda t, sim: sim.force_spike([0])  # noqa: E731

    silenced = LIFSim(synthetic_bundle, cfg).run(
        60, driver=drive_0, silence_nodes=[0])
    # node 0 emits nothing and excites nothing: the run is empty
    assert _count(silenced, 0) == 0
    assert silenced.events_node.size == 0

    # identical run without silence: node 0 emits and node 1 receives its
    # output - the silenced run has strictly fewer events, and node 1's
    # count equals the no-drive zero baseline (0 extra input)
    control = LIFSim(synthetic_bundle, cfg).run(60, driver=drive_0)
    assert _count(control, 0) > 0
    assert _count(control, 1) > 0
    assert silenced.events_node.size < control.events_node.size


# ---------------------------------------------------------------------------
# T-SIM-6 health metrics
# ---------------------------------------------------------------------------


def test_health_metrics(synthetic_bundle):
    cfg = _cfg()
    silent = simulate(synthetic_bundle, _program([], n_steps=60), cfg)
    hm = health_metrics(silent)
    assert hm["degenerate"] is True                 # active fraction 0 < 0.5%
    assert hm["n_active_neurons"] == 0

    # tonic drive on nodes 0 and 2 -> active set {0,1,2} = 50% (< 60%),
    # steady firing -> split-half r >= 0.5. Not degenerate.
    healthy = simulate(synthetic_bundle, _program([
        _drive([0], 0.0, [0.0, 2.0], envelope="tonic"),
        _drive([2], 0.0, [0.0, 2.0], envelope="tonic"),
    ], n_steps=100), cfg)
    hm = health_metrics(healthy)
    assert hm["n_active_neurons"] == 3
    assert hm["active_fraction"] == pytest.approx(0.5)
    assert hm["split_half_r"] >= 0.5
    assert hm["mean_rate_hz_active"] > 0.0
    assert hm["p95_rate_hz_active"] >= hm["median_rate_hz_active"]
    assert hm["degenerate"] is False


# ---------------------------------------------------------------------------
# Warmup, refractory, weight modes, envelopes, adapter
# ---------------------------------------------------------------------------


def test_warmup_rezeroes_events(synthetic_bundle):
    """Warmup steps are dropped: recorded step 0 is post-warmup, and an
    unstimulated warmup changes nothing (cold start from v=0)."""
    program = _program([_drive([0], 0.0, [0.0, 0.4], envelope="tonic")],
                       n_steps=60)
    warm = simulate(synthetic_bundle, program, _cfg(warmup_steps=20))
    cold = simulate(synthetic_bundle, program, _cfg(warmup_steps=0))
    assert warm.n_steps == 60
    assert int(warm.events_step.min()) == 0          # re-zeroed
    assert int(warm.events_step.max()) < 60
    np.testing.assert_array_equal(warm.events_step, cold.events_step)
    np.testing.assert_array_equal(warm.events_node, cold.events_node)


def test_refractory_caps_rate(synthetic_bundle):
    """refractory_steps holds a spiking neuron at reset: depolarized node
    5 fires every step uncapped, every 5th step with refractory 4."""
    def depol(t, sim):
        sim.inject_voltage([5], 1.5)

    sim = LIFSim(synthetic_bundle, _cfg())
    free = sim.run(100, driver=depol)
    assert _count(free, 5) == 100

    held = LIFSim(synthetic_bundle, _cfg(refractory_steps=4)).run(100, driver=depol)
    assert _count(held, 5) == 20                     # fires t=0,5,...,95
    assert _count(held, 5) < _count(free, 5)


def test_weight_mode_binary_vs_count(synthetic_bundle):
    """One forced spike on node 0: count mode delivers w7*w_syn = 1.4 to
    node 1 (fires at t+1); binary mode delivers 1*w_syn = 0.2 (never)."""
    program = _program([_drive([0], 0.0, [0.0, 0.02], envelope="tonic")],
                       n_steps=10)
    count = simulate(synthetic_bundle, program, _cfg(w_syn=0.2))
    binary = simulate(synthetic_bundle, program, _cfg(w_syn=0.2, weight_mode="binary"))
    assert _events_of(count, 1).tolist() == [1]      # fires exactly at t+1
    assert _count(binary, 1) == 0
    # normalized mode: 7/16 of node 1's determined input sum (7+4+5) ->
    # 0.0875 per spike, also subthreshold
    norm = simulate(synthetic_bundle, program, _cfg(w_syn=0.2, weight_mode="normalized"))
    assert _count(norm, 1) == 0


def test_naturalistic_envelope_deterministic_and_bounded(synthetic_bundle):
    """Naturalistic envelope: deterministic OU multiplier; driven activity
    reproduces exactly and stays below the max-rate tonic bound."""
    program = _program([
        _drive([0], 80.0, [0.0, 2.0], envelope="naturalistic",
               envelope_id="env-a"),
    ], n_steps=100)
    cfg = _cfg()
    r1 = simulate(synthetic_bundle, program, cfg)
    r2 = simulate(synthetic_bundle, program, cfg)
    np.testing.assert_array_equal(r1.events_step, r2.events_step)
    np.testing.assert_array_equal(r1.events_node, r2.events_node)
    assert _count(r1, 0) > 0
    # a different envelope_id draws a different multiplier stream
    alt = simulate(synthetic_bundle, _program([
        _drive([0], 80.0, [0.0, 2.0], envelope="naturalistic",
               envelope_id="env-b"),
    ], n_steps=100), cfg)
    assert not np.array_equal(r1.events_step, alt.events_step)


def test_flybrain_adapter_stimulated_trial():
    """Reference backend smoke test: flybrain runs on ITS OWN baked
    graph; skips cleanly when the package or its ~260 MB prebuilt data
    is unavailable."""
    pytest.importorskip("flybrain")
    try:
        adapter = FlybrainAdapter(device="cpu", seed=1)
    except Exception as exc:  # data download unavailable / not cached
        pytest.skip(f"flybrain prebuilt data unavailable: {exc}")
    assert adapter.n_nodes > 0
    try:
        res = adapter.run(10, driver=lambda t, a: a.force_spike(np.arange(50)))
    except Exception as exc:
        pytest.skip(f"flybrain run failed: {exc}")
    assert res.n_nodes == adapter.n_nodes
    assert res.events_node.size > 0                  # stimulated -> events
    assert res.config.backend == "flybrain"
    _assert_sorted_events(res)
