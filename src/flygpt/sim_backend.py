"""LIF simulator backends + flybrain reference adapter (T-SIM-1..8, T-STIM-5).

Implements the Phase-1 simulator contract of ``docs/impl-spec.md`` under the
versioned recipe in ``configs/sim_v1.yaml`` (tag ``sim_v1``): a Shiu-recipe
leaky integrate-and-fire network with alpha-synapse conductance in exact
bi-exponential discrete form, driven by the per-edge NT signs and
synapse-count weights of ``schema.GraphBundle``.

Discrete update (the corpus contract - pinned in impl-spec + sim_v1.yaml;
any change requires a new version tag):

    g[t+1] = g[t]*exp(-dt/tau_g) + sum_{pre fired at t} w_signed(pre->post)
    v[t+1] = v[t]*exp(-dt/tau_v) + drive_scale*g[t+1] + tonic_v + noise_v
    fired  = v[t+1] >= threshold ; v[fired] = reset
    w_signed = sign * weight * w_syn          (weight_mode = "count")

Two update laws are versioned side by side:

- ``synapse_model="alpha"`` (default): the two-ODE form above - the
  conductance trace ``g`` convolved into the membrane trace ``v`` yields
  the two-exponent ("alpha"/bi-exponential) impulse response.
- ``synapse_model="instant"``: flybrain's fly64 single-ODE form
  ``v <- v*exp(-dt/tau_v) + gain*sum_spikes w + tonic + noise`` with
  ``w_syn`` playing the gain role; kept for cross-checks only.

Weight modes (``weight_mode``): ``count`` = raw synapse count;
``binary`` = 1 per edge; ``normalized`` = each postsynaptic neuron's
incoming weight sum divided to 1 over determined-sign edges (the flybrain
per-post-input convention, used by the sensitivity slices). ``sign == 0``
edges (nt_low_confidence at graph build) carry zero current in every mode
- flagged, never guessed (T-BUILD-4).

Determinism (EVAL 1.4, T-SIM-1): every stochastic input - tonic kicks,
sparse noise kicks, Poisson drives, naturalistic envelopes - is drawn from
``schema.rng_from_seed`` generators rooted at
``seed_from_config({sim, program})``; the numpy backend is bitwise
reproducible run-to-run on CPU. jax is imported lazily (the module runs
without it) and consumes the same host-side exogenous streams, so only the
recurrent f32 arithmetic differs from the numpy backend.

Stimulation arrives as exogenous spikes on target nodes (Shiu protocol:
Poisson drive on an identified cell type inside a time window). A forced
spike emits an event at that step and propagates exactly like an
endogenous one, so postsynaptic partners fire at t+1 earliest (T-SIM-3).
Poisson bin occupancy uses ``p = 1 - exp(-rate_hz*dt)`` - the exact
discretization of a Poisson spike train into binary step bins, correct
for rate_hz*dt > 1. ``tonic_rate_hz``/``tonic_amount`` and
``noise_hz``/``noise_amp`` are two independent per-neuron Poisson
voltage-kick channels (kick size = amount/amp, per-step occupancy
1 - exp(-rate*dt)); both run during warmup. ``refractory_steps`` holds a
neuron's voltage at ``reset`` for that many steps after an emitted spike
(flybrain convention); forced spikes bypass refractory but refresh it.
Silenced nodes are dropped BEFORE propagation and never recorded
(T-STIM-5: 0 spikes and 0 outgoing synaptic effect); their internal
voltage is left free-running (output clamp semantics).

``warmup_steps`` run stimulus-free (cold-start fix - drives and silences
live on the recorded timeline, which re-zeroes at step 0); tonic/noise
kicks DO run during warmup so basal state equilibrates.

``FlybrainAdapter`` wraps the pinned reference package on ITS OWN baked
MaleCNS graph for the consistency eval only - never a corpus backend.
Audited convention divergences vs this kernel (recorded in
configs/sim_v1.yaml ``flybrain_reference``, adopted there not here):
per-NEURON consensus_nt sign incl. histamine as inhibitory, per-post
input normalization, fly64 single-ODE update, params dt=0.020
tau_v=0.100 gain=3.0 tonic=0.14 noise_hz=1.2 noise_amp=0.22 (tonic is a
constant DC term in flybrain, not a Poisson channel).

Usage:
    python -m flygpt.sim_backend --graph-dir data/graph \
        --config configs/sim_v1.yaml --steps 200 \
        --drive '{"nodes":[0],"rate_hz":50,"window":[0,1.0]}'

Plan refs: docs/plan.md section 3 T-SIM table; EVAL.md 1.4-1.5 + T-SIM
thresholds; docs/impl-spec.md ``sim_backend.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
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

SIM_VERSION = "sim_v1"
WEIGHT_MODES = ("count", "binary", "normalized")
SYNAPSE_MODELS = ("alpha", "instant")
BACKENDS = ("numpy", "jax", "flybrain")
ENVELOPES = ("poisson", "naturalistic", "tonic", "step")

# Dense (n x n) propagation is used below this node count; above it the
# CSR scatter path keeps memory O(edges-of-fired).
_DENSE_MAX_NODES = 8192

# Naturalistic envelope: bounded non-negative OU rate multiplier sampled
# at dt and clipped to [0,1] (T-STIM-4). Fixed constants - part of the
# sim_v1 contract, not free knobs.
_OU_MU = 0.6        # mean reversion level
_OU_THETA = 3.0     # mean reversion rate (1/s)
_OU_SIGMA = 0.35    # diffusion scale (per sqrt(s))


@dataclass(frozen=True)
class SimConfig:
    """Versioned simulator knobs - mirrors ``configs/sim_v1.yaml model:``.

    ``seed`` only salts the ``seed_from_config`` trial derivation; trial
    seeds are never hand-assigned (plan.md section 6).
    """

    dt: float = 0.020                 # seconds per step
    tau_v: float = 0.100              # membrane tau, seconds
    tau_g: float = 0.010              # synaptic conductance tau
    w_syn: float = 0.01               # THE free param (per-synapse scale)
    threshold: float = 1.0
    reset: float = 0.0
    refractory_steps: int = 0
    weight_mode: str = "count"        # count|binary|normalized (slices)
    tonic_rate_hz: float = 0.0        # global tonic Poisson drive
    tonic_amount: float = 0.0         # per-event tonic voltage kick
    noise_hz: float = 0.0
    noise_amp: float = 0.0
    synapse_model: str = "alpha"      # alpha|instant (fly64 form)
    backend: str = "numpy"            # numpy|jax|flybrain
    warmup_steps: int = 0
    seed: int = 0
    drive_scale: float = 1.0          # g->v coupling constant


@dataclass
class TrialResult:
    """One simulated trial: sparse spike events plus provenance.

    ``events_step``/``events_node`` are u32 arrays sorted by (step,
    neuron); steps are recorded-timeline indices - post-warmup and
    re-zeroed. ``rates`` is the per-node mean rate (Hz, f32) over the
    recorded span.
    """

    events_step: np.ndarray       # u32, sorted by (step, neuron)
    events_node: np.ndarray       # u32
    n_steps: int
    n_nodes: int
    rates: np.ndarray             # f32 per-node rate over recorded span
    config: SimConfig
    seed: int
    program: dict                 # echoed stimulus spec
    meta: dict                    # backend/device/wall seconds

    def events_per_step(self) -> list[np.ndarray]:
        """Node-index array per recorded step (empty steps -> empty arrays)."""
        counts = np.bincount(self.events_step, minlength=self.n_steps)
        return list(np.split(self.events_node, np.cumsum(counts)[:-1]))

    def active_mask(self) -> np.ndarray:
        """bool (n_nodes,) - True for nodes that emitted >= 1 event."""
        return np.bincount(self.events_node, minlength=self.n_nodes) > 0

    def n_active(self) -> int:
        return int(self.active_mask().sum())


def spike_train_matrix(result: TrialResult) -> np.ndarray:
    """Dense (n_steps, n_nodes) bool spike matrix for a TrialResult."""
    mat = np.zeros((result.n_steps, result.n_nodes), dtype=np.bool_)
    mat[result.events_step, result.events_node] = True
    return mat


def health_metrics(result: TrialResult) -> dict[str, Any]:
    """Per-trial health stats (T-SIM-6 / EVAL degenerate rule).

    Reports active-neuron fraction, active-neuron rate stats, and
    split-half stationarity (Pearson r between first- and second-half
    per-neuron event counts). A trial is ``degenerate`` when the active
    fraction is < 0.5% or > 60%, or split-half r < 0.5 - flagged, never
    dropped. Undefined r (zero-variance half, e.g. all-silent trials)
    counts as 0.0.
    """
    n = result.n_nodes
    counts = np.bincount(result.events_node, minlength=n).astype(np.float64)
    active = counts > 0
    frac = float(active.mean()) if n else 0.0
    rates_active = result.rates[active]

    mat = spike_train_matrix(result)
    half = result.n_steps // 2
    c1 = mat[:half].sum(axis=0).astype(np.float64)
    c2 = mat[half:].sum(axis=0).astype(np.float64)
    if half < 1 or c1.std() == 0.0 or c2.std() == 0.0:
        split_r = 0.0
    else:
        split_r = float(np.corrcoef(c1, c2)[0, 1])
        if not np.isfinite(split_r):
            split_r = 0.0

    degenerate = frac < 0.005 or frac > 0.6 or split_r < 0.5
    return {
        "pct_active_neurons": 100.0 * frac,
        "active_fraction": frac,
        "n_active_neurons": int(active.sum()),
        "mean_rate_hz_active": float(rates_active.mean()) if rates_active.size else 0.0,
        "median_rate_hz_active": float(np.median(rates_active)) if rates_active.size else 0.0,
        "p95_rate_hz_active": float(np.percentile(rates_active, 95)) if rates_active.size else 0.0,
        "split_half_r": split_r,
        "degenerate": bool(degenerate),
    }


class LIFSim:
    """Connectome LIF over a GraphBundle (numpy and jax backends).

    ``backend="numpy"`` is bitwise-deterministic on CPU (T-SIM-1).
    ``backend="jax"`` lazy-imports jax.numpy, jits the step function, and
    uses the same host-side exogenous streams; FP reduction order may
    differ, so jax-vs-numpy agreement is graded on the EVAL 1.4
    cross-backend band, not bitwise.
    """

    def __init__(self, bundle: GraphBundle, cfg: SimConfig):
        if cfg.weight_mode not in WEIGHT_MODES:
            raise ValueError(f"weight_mode {cfg.weight_mode!r} not in {WEIGHT_MODES}")
        if cfg.synapse_model not in SYNAPSE_MODELS:
            raise ValueError(f"synapse_model {cfg.synapse_model!r} not in {SYNAPSE_MODELS}")
        if cfg.backend not in ("numpy", "jax"):
            raise ValueError(
                f"LIFSim backend {cfg.backend!r} not in ('numpy','jax') - "
                "'flybrain' is the FlybrainAdapter path (its own baked graph)"
            )
        if cfg.dt <= 0 or cfg.tau_v <= 0 or cfg.tau_g <= 0:
            raise ValueError("dt/tau_v/tau_g must be positive seconds")
        self.bundle = bundle
        self.cfg = cfg
        self.n = bundle.n_nodes
        self._dg = np.float32(np.exp(-cfg.dt / cfg.tau_g))
        self._dv = np.float32(np.exp(-cfg.dt / cfg.tau_v))
        self._w_edge = self._effective_weights()          # f32, out-CSR order
        self._pre_of_edge = np.repeat(
            np.arange(self.n, dtype=np.int64), np.diff(bundle.out_indptr)
        )
        # Dense W[pre,post] for small graphs; CSR scatter otherwise.
        self._dense = None
        if self.n <= _DENSE_MAX_NODES:
            w = np.zeros((self.n, self.n), dtype=np.float32)
            w[self._pre_of_edge, self.bundle.out_indices] = self._w_edge
            self._dense = w
        self._jax = None
        if cfg.backend == "jax":
            self._jax = _JaxStep(self)
        self.reset(cfg.seed)

    # -- construction --------------------------------------------------

    def _effective_weights(self) -> np.ndarray:
        """sign * mode-weight per out-CSR edge (w_syn applied per step).

        ``normalized``: per-post divide by the incoming weight sum over
        determined-sign edges (flybrain per-post convention). sign==0
        edges contribute 0 to numerator and denominator alike.
        """
        sign = self.bundle.out_sign.astype(np.float32)
        weight = self.bundle.out_weight.astype(np.float32)
        if self.cfg.weight_mode == "binary":
            return np.ascontiguousarray(sign)
        if self.cfg.weight_mode == "normalized":
            determined = self.bundle.out_sign != 0
            denom = np.bincount(
                self.bundle.out_indices,
                weights=weight * determined.astype(np.float32),
                minlength=self.n,
            ).astype(np.float32)
            denom[denom == 0.0] = 1.0   # unreachable posts: edge weight is 0 anyway
            return np.ascontiguousarray(sign * weight / denom[self.bundle.out_indices])
        return np.ascontiguousarray(sign * weight)  # count

    # -- state ----------------------------------------------------------

    def reset(self, seed: int) -> None:
        """Zero all state and restart the deterministic rng streams."""
        self._seed = int(seed)
        self.v = np.zeros(self.n, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self._cool = np.zeros(self.n, dtype=np.int32)   # refractory countdown
        self._fired_prev = np.zeros(self.n, dtype=np.bool_)
        self._pend_force = np.zeros(self.n, dtype=np.bool_)
        self._pend_inject = np.zeros(self.n, dtype=np.float32)
        self._pend_sil = np.zeros(self.n, dtype=np.bool_)
        self._step_index = 0
        self._rng_tonic = rng_from_seed(self._seed, "tonic")
        self._rng_noise = rng_from_seed(self._seed, "noise")
        if self._jax is not None:
            self._jax.reset()

    # -- driver-facing stimulus API (called inside driver(step, sim)) ----

    def force_spike(self, nodes) -> None:
        """Emit an exogenous spike on ``nodes`` at the current step.

        The spike is recorded as an event and propagates next step like
        an endogenous one (unless the node is silenced this step).
        """
        self._pend_force[np.asarray(nodes, dtype=np.int64)] = True

    def inject_voltage(self, nodes, amount: float) -> None:
        """Add ``amount`` to these nodes' voltage at the current step
        (before the threshold test, so it can trigger a same-step spike)."""
        self._pend_inject[np.asarray(nodes, dtype=np.int64)] += np.float32(amount)

    def silence_for_step(self, nodes) -> None:
        """Drop these nodes' spikes at the current step only - windowed
        silences (T-STIM-5) are built on this by simulate()."""
        self._pend_sil[np.asarray(nodes, dtype=np.int64)] = True

    # -- dynamics --------------------------------------------------------

    def _kicks(self) -> np.ndarray:
        """Tonic + noise Poisson voltage kicks for the current step."""
        cfg = self.cfg
        kick = np.zeros(self.n, dtype=np.float32)
        if cfg.tonic_rate_hz > 0.0 and cfg.tonic_amount != 0.0:
            p = np.float32(1.0 - np.exp(-cfg.tonic_rate_hz * cfg.dt))
            kick += np.where(self._rng_tonic.random(self.n) < p,
                             np.float32(cfg.tonic_amount), np.float32(0.0))
        if cfg.noise_hz > 0.0 and cfg.noise_amp != 0.0:
            p = np.float32(1.0 - np.exp(-cfg.noise_hz * cfg.dt))
            kick += np.where(self._rng_noise.random(self.n) < p,
                             np.float32(cfg.noise_amp), np.float32(0.0))
        return kick

    def _propagate(self, fired: np.ndarray) -> np.ndarray:
        """sum_{pre fired} w_signed(pre->post) (without the w_syn scale)."""
        if self._dense is not None:
            return fired.astype(np.float32) @ self._dense
        idx = np.flatnonzero(fired)
        if idx.size == 0:
            return np.zeros(self.n, dtype=np.float32)
        starts = self.bundle.out_indptr[idx]
        counts = self.bundle.out_indptr[idx + 1] - starts
        total = int(counts.sum())
        if total == 0:
            return np.zeros(self.n, dtype=np.float32)
        seg = np.repeat(np.arange(idx.size), counts)
        within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        edges = starts[seg] + within
        return np.bincount(
            self.bundle.out_indices[edges],
            weights=self._w_edge[edges],
            minlength=self.n,
        ).astype(np.float32)

    def _numpy_step(self, kick: np.ndarray, force: np.ndarray,
                    sil: np.ndarray) -> np.ndarray:
        """One sim_v1 update; returns emitted spikes (post-silence)."""
        cfg = self.cfg
        current = self._propagate(self._fired_prev) * np.float32(cfg.w_syn)
        if cfg.synapse_model == "alpha":
            self.g = self.g * self._dg + current
            self.v = self.v * self._dv + np.float32(cfg.drive_scale) * self.g + kick
        else:  # instant (fly64): w_syn plays the gain role, no g trace
            self.v = self.v * self._dv + current + kick
        self.v[self._cool > 0] = np.float32(cfg.reset)          # refractory hold
        fired = (self.v >= np.float32(cfg.threshold)) | force
        emitted = fired & ~sil                                 # T-STIM-5
        self.v[emitted] = np.float32(cfg.reset)
        self._cool[self._cool > 0] -= 1
        self._cool[emitted] = cfg.refractory_steps
        self._fired_prev = emitted
        return emitted

    def run(self, n_steps: int, driver: Callable | None = None,
            silence_nodes=None) -> TrialResult:
        """Run ``n_steps`` recorded steps (plus ``cfg.warmup_steps`` dropped).

        ``driver(step:int, sim)`` is invoked at the TOP of every absolute
        step (warmup included, raw 0-based index) and may call
        ``force_spike`` / ``inject_voltage`` / ``silence_for_step``.
        ``silence_nodes`` is a static node list silenced for the whole
        run; windowed silences go through the driver. Recorded events are
        re-zeroed so recorded step 0 follows the last warmup step.
        """
        cfg = self.cfg
        n_steps = int(n_steps)
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")
        total = int(cfg.warmup_steps) + n_steps
        static_sil = np.zeros(self.n, dtype=np.bool_)
        if silence_nodes is not None:
            static_sil[np.asarray(list(silence_nodes), dtype=np.int64)] = True

        t_start = time.time()
        ev_steps: list[np.ndarray] = []
        ev_nodes: list[np.ndarray] = []
        for t in range(total):
            self._pend_force.fill(False)
            self._pend_inject.fill(0.0)
            self._pend_sil.fill(False)
            self._step_index = t
            if driver is not None:
                driver(t, self)
            kick = self._kicks() + self._pend_inject
            sil = static_sil | self._pend_sil
            if self._jax is not None:
                emitted = self._jax.step(kick, self._pend_force, sil)
                if driver is not None:
                    # keep host-side state mirrors fresh for drivers that
                    # inspect sim.v/g between steps
                    self.v = np.asarray(self._jax.v, dtype=np.float32)
                    self.g = np.asarray(self._jax.g, dtype=np.float32)
                    self._cool = np.asarray(self._jax.cool, dtype=np.int32)
                self._fired_prev = emitted
            else:
                emitted = self._numpy_step(kick, self._pend_force, sil)
            if t >= cfg.warmup_steps and emitted.any():
                idx = np.flatnonzero(emitted)
                ev_steps.append(np.full(idx.size, t - cfg.warmup_steps, dtype=np.uint32))
                ev_nodes.append(idx.astype(np.uint32))

        if ev_nodes:
            events_step = np.concatenate(ev_steps)
            events_node = np.concatenate(ev_nodes)
        else:
            events_step = np.empty(0, dtype=np.uint32)
            events_node = np.empty(0, dtype=np.uint32)
        counts = np.bincount(events_node, minlength=self.n)
        rates = (counts / max(n_steps * cfg.dt, 1e-12)).astype(np.float32)
        return TrialResult(
            events_step=events_step,
            events_node=events_node,
            n_steps=n_steps,
            n_nodes=self.n,
            rates=rates,
            config=cfg,
            seed=self._seed,
            program={},
            meta={
                "version": SIM_VERSION,
                "backend": cfg.backend,
                "device": "cpu",
                "wall_seconds": round(time.time() - t_start, 6),
                "warmup_steps": int(cfg.warmup_steps),
                "n_steps_total": total,
            },
        )


class _JaxStep:
    """jit-compiled sim_v1 step for the jax backend (lazy-imported).

    Mirrors ``LIFSim._numpy_step`` line for line; per-step exogenous
    inputs arrive as host-computed numpy arrays so both backends see
    identical drive/tonic/noise streams.
    """

    def __init__(self, sim: LIFSim):
        import jax
        import jax.numpy as jnp

        cfg = sim.cfg
        n = sim.n
        pre = jnp.asarray(sim._pre_of_edge.astype(np.int32))
        post = jnp.asarray(sim.bundle.out_indices.astype(np.int32))
        w = jnp.asarray(sim._w_edge)
        dg = np.float32(sim._dg)
        dv = np.float32(sim._dv)
        w_syn = np.float32(cfg.w_syn)
        drive_scale = np.float32(cfg.drive_scale)
        thr = np.float32(cfg.threshold)
        reset_v = np.float32(cfg.reset)
        refr = int(cfg.refractory_steps)
        alpha = cfg.synapse_model == "alpha"

        @jax.jit
        def step_fn(v, g, cool, fired_prev, kick, force, sil):
            cur = jnp.zeros((n,), jnp.float32).at[post].add(
                w * fired_prev[pre].astype(jnp.float32)
            ) * w_syn
            if alpha:
                g = g * dg + cur
                v = v * dv + drive_scale * g + kick
            else:
                v = v * dv + cur + kick
            v = jnp.where(cool > 0, reset_v, v)
            fired = (v >= thr) | force
            emitted = fired & ~sil
            v = jnp.where(emitted, reset_v, v)
            cool = jnp.where(cool > 0, cool - 1, cool)
            cool = jnp.where(emitted, refr, cool)
            return v, g, cool, emitted

        self._jnp = jnp
        self._step_fn = step_fn
        self._n = n
        self.reset()

    def reset(self) -> None:
        jnp = self._jnp
        self.v = jnp.zeros((self._n,), jnp.float32)
        self.g = jnp.zeros((self._n,), jnp.float32)
        self.cool = jnp.zeros((self._n,), jnp.int32)
        self._fired_prev = jnp.zeros((self._n,), jnp.bool_)

    def step(self, kick: np.ndarray, force: np.ndarray, sil: np.ndarray) -> np.ndarray:
        jnp = self._jnp
        self.v, self.g, self.cool, emitted = self._step_fn(
            self.v, self.g, self.cool, self._fired_prev,
            jnp.asarray(kick), jnp.asarray(force), jnp.asarray(sil),
        )
        self._fired_prev = emitted
        return np.asarray(emitted)


class FlybrainAdapter:
    """Reference backend: ``flybrain==0.1.0`` on ITS OWN baked graph.

    Used by the consistency eval only - never a corpus backend (the
    baked graph is not our GraphBundle; its conventions are audited and
    deliberately not adopted). Documented divergences vs this module's
    kernel (configs/sim_v1.yaml ``flybrain_reference``):

    - sign: per-NEURON consensus_nt (incl. histamine as inhibitory) vs
      our per-EDGE T-bar majority sign;
    - weights: per-post input normalization vs our raw counts
      (``weight_mode="normalized"`` approximates it);
    - update: fly64 single-ODE ``v <- decay*v + gain*W@spikes + tonic +
      noise`` vs our alpha two-ODE form (``synapse_model="instant"``
      approximates it);
    - params: dt=0.020 tau_v=0.100 gain=3.0 tonic=0.14 noise_hz=1.2
      noise_amp=0.22 (flybrain tonic is a constant DC term, ours is a
      Poisson kick channel).

    The constructor downloads flybrain's prebuilt brain (~260 MB, once)
    if absent; callers should treat any construction failure as
    "reference backend unavailable" and skip.
    """

    def __init__(self, data: Path | str | None = None, device: str = "cpu",
                 seed: int = 64):
        import flybrain  # lazy - optional dependency

        self._fb = flybrain.FlyBrain(data=data, seed=seed, device=device)
        self._seed = int(seed)
        self._inject: dict[int, float] = {}

    @property
    def n_nodes(self) -> int:
        return int(self._fb.n)

    def cells(self, types: list[str], side: str | None = None) -> np.ndarray:
        """flybrain type/superclass lookup -> baked-graph indices."""
        return self._fb.cells(types, side)

    def force_spike(self, nodes) -> None:
        """Queue a guaranteed spike this step (inject >= threshold 1.0)."""
        for i in np.asarray(nodes, dtype=np.int64).tolist():
            self._inject[int(i)] = max(1.0, self._inject.get(int(i), 0.0))

    def inject_voltage(self, nodes, amount: float) -> None:
        for i in np.asarray(nodes, dtype=np.int64).tolist():
            self._inject[int(i)] = self._inject.get(int(i), 0.0) + float(amount)

    def run(self, n_steps: int, driver: Callable | None = None,
            silence_nodes=None) -> TrialResult:
        """Advance flybrain ``n_steps`` steps, collecting a TrialResult.

        ``driver(step, adapter)`` may call ``force_spike`` /
        ``inject_voltage``. ``silence_nodes`` are dropped before the next
        step's propagation and before recording (T-STIM-5 semantics on
        the reference backend, as close as its API allows: their voltage
        reset inside flybrain.step is not undone).
        """
        sil = np.asarray(list(silence_nodes or []), dtype=np.int64)
        ev_steps: list[np.ndarray] = []
        ev_nodes: list[np.ndarray] = []
        t_start = time.time()
        for t in range(int(n_steps)):
            self._inject = {}
            if driver is not None:
                driver(t, self)
            inject: list[tuple[np.ndarray, np.ndarray]] = []
            if self._inject:
                inject.append((
                    np.fromiter(self._inject.keys(), dtype=np.int64),
                    np.fromiter(self._inject.values(), dtype=np.float32),
                ))
            fired = np.asarray(self._fb.step(inject=inject), dtype=np.int64)
            if sil.size:
                keep = ~np.isin(fired, sil)
                fired = fired[keep]
                # rewrite flybrain's carried spike set so silenced spikes
                # contribute no synaptic input next step
                self._fb.fired = self._fb.xp.asarray(fired)
            if fired.size:
                ev_steps.append(np.full(fired.size, t, dtype=np.uint32))
                ev_nodes.append(fired.astype(np.uint32))
        if ev_nodes:
            events_step = np.concatenate(ev_steps)
            events_node = np.concatenate(ev_nodes)
        else:
            events_step = np.empty(0, dtype=np.uint32)
            events_node = np.empty(0, dtype=np.uint32)
        counts = np.bincount(events_node, minlength=self.n_nodes)
        rates = (counts / max(int(n_steps) * self._fb.dt, 1e-12)).astype(np.float32)
        return TrialResult(
            events_step=events_step,
            events_node=events_node,
            n_steps=int(n_steps),
            n_nodes=self.n_nodes,
            rates=rates,
            config=SimConfig(
                dt=self._fb.dt,
                tau_v=self._fb.tau,
                w_syn=self._fb.gain,
                synapse_model="instant",
                backend="flybrain",
                noise_hz=self._fb.noise_hz,
                noise_amp=self._fb.noise_amp,
                seed=self._seed,
            ),
            seed=self._seed,
            program={},
            meta={
                "version": "flybrain-0.1.0",
                "backend": "flybrain",
                "device": self._fb.device,
                "tonic_dc": float(self._fb.tonic),
                "conventions": {
                    "sign": "per-neuron consensus_nt incl. histamine",
                    "weights": "per-post input normalization",
                    "update": "fly64 single-ODE",
                },
                "wall_seconds": round(time.time() - t_start, 6),
            },
        )


# ---------------------------------------------------------------------------
# simulate(): program dict -> driver -> TrialResult
# ---------------------------------------------------------------------------

@dataclass
class _DrivePlan:
    nodes: np.ndarray          # u32 node indices
    s0: int                    # absolute start step (warmup included)
    s1: int                    # absolute end step (exclusive)
    envelope: str
    rate_hz: float
    rng: np.random.Generator | None   # poisson / naturalistic stream
    p_poisson: float           # 1 - exp(-rate*dt), envelope=poisson
    env: np.ndarray | None     # naturalistic rate multipliers [0,1]
    period: int                # step envelope: deterministic spacing
    phases: np.ndarray | None  # per-node phase offsets, step envelope


def _ou_envelope(seed: int, envelope_id: str, n_steps: int, dt: float) -> np.ndarray:
    """Deterministic bounded non-negative OU rate multiplier in [0,1].

    One shared envelope per (seed, envelope_id) - a drive's nodes all see
    the same multiplier (a population-level naturalistic signal).
    """
    rng = rng_from_seed(seed, "envelope", envelope_id)
    env = np.empty(max(n_steps, 0), dtype=np.float32)
    x = _OU_MU
    for k in range(n_steps):
        x = x + _OU_THETA * (_OU_MU - x) * dt + _OU_SIGMA * np.sqrt(dt) * rng.standard_normal()
        env[k] = np.float32(np.clip(x, 0.0, 1.0))
    return env


def _normalize_program(program: dict, cfg: SimConfig) -> dict:
    """Validate + normalize a stimulus-program dict into a canonical spec."""
    spec = {"drives": [], "silences": []}
    for d in program.get("drives", []):
        envelope = d.get("envelope", "poisson")
        if envelope not in ENVELOPES:
            raise ValueError(f"envelope {envelope!r} not in {ENVELOPES}")
        t0, t1 = (float(x) for x in d["window"])
        if t1 <= t0:
            raise ValueError(f"drive window [{t0},{t1}] empty")
        drive_spec = {
            "nodes": [int(x) for x in d["nodes"]],
            "rate_hz": float(d.get("rate_hz", 0.0)),
            "window": [t0, t1],
            "envelope": envelope,
            "envelope_id": str(d.get("envelope_id", "")),
        }
        # Recorded envelopes (T-STIM-4): the program may carry the exact
        # sampled trace - play it back rather than re-deriving it.
        if d.get("envelope_values") is not None:
            drive_spec["envelope_values"] = [
                float(v) for v in d["envelope_values"]
            ]
            if d.get("envelope_dt") is not None:
                drive_spec["envelope_dt"] = float(d["envelope_dt"])
        spec["drives"].append(drive_spec)
    for s in program.get("silences", []):
        t0, t1 = (float(x) for x in s["window"])
        spec["silences"].append({
            "nodes": [int(x) for x in s["nodes"]],
            "window": [t0, t1],
        })
    if "n_steps" in program:
        spec["n_steps"] = int(program["n_steps"])
    elif "duration_s" in program:
        spec["n_steps"] = int(round(float(program["duration_s"]) / cfg.dt))
    else:
        raise ValueError("program needs 'n_steps' or 'duration_s'")
    return spec


def _program_driver(spec: dict, cfg: SimConfig, seed: int) -> Callable:
    """Compile a normalized program into a driver callable.

    Drive/silence windows are on the RECORDED timeline; the +warmup
    offset maps them onto the absolute step index the driver sees.
    """
    warm = int(cfg.warmup_steps)
    drives: list[_DrivePlan] = []
    for i, d in enumerate(spec["drives"]):
        s0 = warm + int(round(d["window"][0] / cfg.dt))
        s1 = warm + int(round(d["window"][1] / cfg.dt))
        if s1 <= s0 or not d["nodes"]:
            continue
        env = d["envelope"]
        rng = rng_from_seed(seed, "drive", i) if env in ("poisson", "naturalistic") else None
        env_vec = None
        if env == "naturalistic":
            recorded = d.get("envelope_values")
            if recorded is not None:
                env_vec = np.asarray(recorded, dtype=np.float32)
                if env_vec.size != s1 - s0:
                    raise ValueError(
                        f"recorded envelope_values len {env_vec.size} != "
                        f"drive window steps {s1 - s0}"
                    )
                if env_vec.size and (env_vec.min() < 0.0 or env_vec.max() > 1.0):
                    raise ValueError("envelope_values outside [0,1]")
            else:
                env_vec = _ou_envelope(
                    seed, d["envelope_id"] or f"drive{i}", s1 - s0, cfg.dt
                )
        period = 0
        phases = None
        if env == "step":
            period = max(1, int(round(1.0 / max(d["rate_hz"] * cfg.dt, 1e-12))))
            phases = np.arange(len(d["nodes"])) % period
        drives.append(_DrivePlan(
            nodes=np.asarray(d["nodes"], dtype=np.uint32),
            s0=s0, s1=s1, envelope=env, rate_hz=float(d["rate_hz"]), rng=rng,
            p_poisson=float(1.0 - np.exp(-d["rate_hz"] * cfg.dt)),
            env=env_vec, period=period, phases=phases,
        ))
    silences = [
        (warm + int(round(s["window"][0] / cfg.dt)),
         warm + int(round(s["window"][1] / cfg.dt)),
         np.asarray(s["nodes"], dtype=np.int64))
        for s in spec["silences"]
    ]

    def driver(t: int, sim: LIFSim) -> None:
        for dr in drives:
            if not (dr.s0 <= t < dr.s1):
                continue
            k = t - dr.s0
            n = dr.nodes.size
            if dr.envelope == "poisson":
                mask = dr.rng.random(n) < dr.p_poisson
            elif dr.envelope == "tonic":
                mask = np.ones(n, dtype=np.bool_)
            elif dr.envelope == "step":
                mask = (k % dr.period) == dr.phases
            else:  # naturalistic: rate_hz scaled by the OU multiplier
                p = 1.0 - np.exp(-dr.rate_hz * cfg.dt * float(dr.env[k]))
                mask = dr.rng.random(n) < np.float32(p)
            if mask.any():
                sim.force_spike(dr.nodes[mask])
        for a, b, nodes in silences:
            if a <= t < b:
                sim.silence_for_step(nodes)

    return driver


def simulate(
    bundle: GraphBundle, program: dict, cfg: SimConfig | dict
) -> TrialResult:
    """Run one stimulus program on the bundle and return its TrialResult.

    ``program`` keys: ``drives`` (list of {nodes, rate_hz, window,
    envelope, envelope_id}) and ``silences`` (list of {nodes, window});
    ``n_steps`` or ``duration_s`` sets the recorded span. The trial seed
    is ``seed_from_config({sim: cfg, program: spec})`` - deterministic
    given (config, program), never hand-assigned.

    ``cfg`` accepts the SimConfig-compatible mapping the corpus serializes
    into shards (slice keys like ``cutoff``/``weight_shuffle``/``slice``
    are dropped - they live at bundle level, not in the LIF).
    """
    if isinstance(cfg, dict):
        model = dict(cfg.get("model", cfg))
        fields = set(SimConfig.__dataclass_fields__)
        cfg = SimConfig(**{k: v for k, v in model.items() if k in fields})
    if cfg.backend == "flybrain":
        raise ValueError(
            "backend='flybrain' runs FlybrainAdapter on its own baked graph - "
            "program node indices do not translate; call FlybrainAdapter directly"
        )
    spec = _normalize_program(program, cfg)
    # Trial identity excludes `backend` - an execution detail, not trial
    # semantics. EVAL 1.4 requires identical (seed, config) trials to be
    # reproducible ACROSS backends, so backend must not perturb the seed.
    sim_spec = {k: v for k, v in asdict(cfg).items() if k != "backend"}
    seed = seed_from_config({"sim": sim_spec, "program": spec})
    all_nodes = (
        [x for d in spec["drives"] for x in d["nodes"]]
        + [x for s in spec["silences"] for x in s["nodes"]]
    )
    bad = [x for x in all_nodes if x < 0 or x >= bundle.n_nodes]
    if bad:
        raise ValueError(
            f"program references node indices outside the graph "
            f"(n={bundle.n_nodes}): {sorted(set(bad))}"
        )
    sim = LIFSim(bundle, cfg)
    sim.reset(seed)
    driver = _program_driver(spec, cfg, seed)
    result = sim.run(spec["n_steps"], driver=driver)
    result.program = spec
    result.seed = seed
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cfg_from_yaml(path: Path, backend: str | None, seed: int | None) -> SimConfig:
    raw = yaml.safe_load(path.read_text()) if path.exists() else {}
    model = dict((raw or {}).get("model", {}))
    model.pop("w_syn_sweep", None)  # sweep metadata, not a knob
    fields = set(SimConfig.__dataclass_fields__)
    return SimConfig(
        backend=backend or (raw or {}).get("backend", "numpy"),
        seed=int(seed) if seed is not None else 0,
        **{k: v for k, v in model.items() if k in fields},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/sim_v1.yaml"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--drive", type=str, default=None,
                    help="JSON drive spec (object or list): "
                         '{nodes:[..], rate_hz:50, window:[t0,t1], envelope:..}')
    ap.add_argument("--backend", choices=list(BACKENDS), default=None)
    ap.add_argument("--cutoff", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = _cfg_from_yaml(args.config, args.backend, args.seed)
    drives: list[dict] = []
    if args.drive:
        parsed = json.loads(args.drive)
        drives = parsed if isinstance(parsed, list) else [parsed]
    program = {"drives": drives, "silences": [], "n_steps": args.steps}

    if cfg.backend == "flybrain":
        adapter = FlybrainAdapter()
        # program node indices are flybrain baked-graph indices here
        plans = [
            (int(round(d["window"][0] / cfg.dt)), int(round(d["window"][1] / cfg.dt)),
             np.asarray(d["nodes"], dtype=np.int64))
            for d in drives
        ]

        def fb_driver(t: int, ad: FlybrainAdapter) -> None:
            for a, b, nodes in plans:
                if a <= t < b:
                    ad.force_spike(nodes)

        res = adapter.run(args.steps, driver=fb_driver if plans else None)
    else:
        bundle = load_graph_bundle(args.graph_dir, cutoff=args.cutoff)
        res = simulate(bundle, program, cfg)

    print(
        f"[sim] backend={res.meta['backend']} nodes={res.n_nodes} "
        f"steps={res.n_steps} events={res.events_node.size} "
        f"active={res.n_active()}/{res.n_nodes} "
        f"wall={res.meta['wall_seconds']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
