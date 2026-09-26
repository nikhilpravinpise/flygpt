"""Dual-axis connectome transformer + control-suite trainer (T-MODEL-*).

Implements docs/impl-spec.md "model.py" under docs/plan.md Phase 3 and
the EVAL.md section-5 thresholds (T-MODEL-1..11, T-SEX-5, silence-robust
loss per EVAL 1.7). The model predicts, for every unit (vocab/type slot)
and every time bin, an event logit and a rate estimate, under a
connectome-derived attention mask.

Layer alternation (``n_layers`` blocks, ``mask_convention =
pre_t_to_post_t1`` per configs/model_v1.yaml):

- even layers: causal attention along T per unit - V is folded into the
  batch ((B*V, T, d) self-attention with a lower-triangular mask), so
  each unit integrates its own past independently;
- odd layers: connectome-masked cross-time attention - unit u at step t
  attends to the representations of its PRESYNAPTIC partners at t-1.
  Partners arrive gathered via an ``EdgeSet`` input ((V, K) index +
  weight + sign + valid arrays built from the bundle's by-POST ``in_*``
  CSR, padded to ``cfg.max_in_degree`` with self-padding). Attention
  score: ``q_u . k_v / sqrt(d_head) + edge_gain_h * log(w_eps + w_uv)``
  with a learnable per-head sign gate on the value channel. Because the
  gather uses ``in_*`` (the transpose of ``out_*``), edges act
  pre(t) -> post(t+1) - the transpose-checked convention of T-MODEL-9.

The edge set is a MODULE INPUT, never a parameter: one parameter set
runs under any mask, which is the whole T-SEX-5 sex-conditioning
mechanism and the T-MODEL-3/4 control wiring.

jax/flax/optax are the model framework (plan Phase 3: Kaggle TPU drip).
They are imported lazily - ``import flygpt.model`` works without them;
anything that builds or trains a module raises a clear ImportError.
"""

from __future__ import annotations

import argparse
import dataclasses
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import yaml

from flygpt.schema import N_STIMULUS_SLOTS, GraphBundle, rng_from_seed, seed_from_config

try:  # model framework - optional on CPU-only data installs
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax

    _JAX_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # pragma: no cover - exercised on jax-less installs
    jax = None
    jnp = None
    nn = None
    optax = None
    _JAX_IMPORT_ERROR = exc

# When jax/flax are absent the nn.Module subclasses below degrade to
# plain classes whose methods re-raise the import error via
# _require_jax() - module import stays clean (impl-spec global rules).
_LinenModule = nn.Module if nn is not None else object
_compact = nn.compact if nn is not None else (lambda fn: fn)

_CHECKPOINT_FORMAT = "flygpt_ckpt_v1"
_CKPT_LATEST = "ckpt_latest.pkl"


def _require_jax() -> None:
    """Clear failure when the model framework is not installed."""
    if _JAX_IMPORT_ERROR is not None:
        raise ImportError(
            "flygpt.model requires jax, flax and optax - install the "
            "`model` extra (pip install 'flygpt[model]')"
        ) from _JAX_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Configuration (per impl-spec + configs/model_v1.yaml)
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Model + training config; mirrors configs/model_v1.yaml sections.

    ``edge_set`` is a runtime attachment, not a hyperparameter: the mask
    rides on cfg so train() needs no extra argument, and T-SEX-5 swaps it
    at call time. It is excluded from ``spec()`` (seed/checkpoint
    material) and from YAML loading.
    """

    # architecture
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 12              # alternates time-axis / connectome blocks
    n_vocab: int = 0                # = vocab.size() - the unit axis V
    max_in_degree: int = 64         # gathered presyn partner cap (recorded)
    max_len: int = 512              # learned positional table length
    dropout: float = 0.1
    w_eps: float = 1e-3             # edge-bias log offset (w_eps + w)
    use_stim: bool = False          # stimulus-family context embedding slot
    arch: str = "connectome"        # connectome | dense | gnn (T-MODEL-5/10)
    # training
    param_target: int = 100_000_000
    batch_trials: int = 8
    lr: float = 3e-4
    warmup_steps: int = 2000
    total_steps: int = 200_000
    checkpoint_every: int = 5000
    seeds: tuple[int, ...] = (1, 2, 3)
    seed: int = 0                   # active run seed (one of ``seeds``)
    silence_weight: float = 0.1     # inactive-unit loss weight (T-MODEL-8)
    event_weight: float = 1.0
    rate_weight: float = 0.1
    # runtime-only attachment - never serialized
    edge_set: Any = field(default=None, repr=False, compare=False)

    def spec(self) -> dict[str, Any]:
        """JSON-able spec for seed derivation + checkpoint manifests."""
        d = dataclasses.asdict(self)
        d.pop("edge_set", None)
        d["seeds"] = list(self.seeds)
        return d

    def init_seed(self) -> int:
        """Deterministic PRNG seed from the config + run seed."""
        return seed_from_config({"model_init": self.spec(), "seed": self.seed}) % (2**31)


def load_model_config(path: Path, dev: bool = False) -> ModelConfig:
    """Load a model_v1-shaped YAML config; ``dev`` applies dev overrides."""
    raw = yaml.safe_load(Path(path).read_text())
    arch = raw.get("architecture", {})
    train_cfg = raw.get("training", {})
    loss = train_cfg.get("loss", {})
    merged = {
        "d_model": arch.get("d_model", 512),
        "n_heads": arch.get("n_heads", 8),
        "d_ff": arch.get("d_ff", 2048),
        "n_layers": arch.get("n_layers", 12),
        "n_vocab": arch.get("n_vocab", 0),
        "max_in_degree": arch.get("max_in_degree", 64),
        "dropout": arch.get("dropout", 0.1),
        "param_target": train_cfg.get("param_target", 100_000_000),
        "batch_trials": train_cfg.get("batch_trials", 8),
        "lr": train_cfg.get("lr", 3e-4),
        "warmup_steps": train_cfg.get("warmup_steps", 2000),
        "total_steps": train_cfg.get("total_steps", 200_000),
        "checkpoint_every": train_cfg.get("checkpoint_every", 5000),
        "seeds": tuple(train_cfg.get("seeds", (1, 2, 3))),
        "silence_weight": loss.get("silence_weight", 0.1),
        "event_weight": loss.get("event_weight", 1.0),
        "rate_weight": loss.get("rate_weight", 0.1),
    }
    if dev:
        merged.update(raw.get("dev", {}))
    return ModelConfig(**merged)


# ---------------------------------------------------------------------------
# EdgeSet - the connectome mask as a module input (T-SEX-5 machinery)
# ---------------------------------------------------------------------------


class EdgeSet(NamedTuple):
    """Gathered presynaptic neighbor lists, one (V, K) row per post unit.

    ``presyn_idx[u]`` holds up to K presynaptic partner node indices of u
    (top-K by synapse count when the in-degree exceeds K - the cap is
    recorded, never silently exceeded); padding slots repeat u itself so
    gathers stay in range while ``valid`` marks them out. ``edge_weight``
    is the unsigned synapse count normalized to (0, 1] by the bundle max;
    ``edge_sign`` is the NT sign in {-1, 0, +1} (0 = undetermined).
    """

    presyn_idx: np.ndarray    # (V, K) i32
    edge_weight: np.ndarray   # (V, K) f32, unsigned, (0,1]-normalized
    edge_sign: np.ndarray     # (V, K) f32 in {-1, 0, +1}
    valid: np.ndarray         # (V, K) bool - False on padding


def edge_set_from_bundle(bundle: GraphBundle, max_in_degree: int) -> EdgeSet:
    """Build the (V, K) EdgeSet from a bundle's by-POST ``in_*`` CSR.

    The gather direction is the T-MODEL-9 contract: row u lists the
    PRESYNAPTIC partners of u (``in_indices``), so an edge v->u lets post
    unit u read pre unit v at the previous step. ``bundle.in_*`` is the
    exact transpose of ``out_*`` (schema.build_incoming_csr), which is
    what "transpose-checked by construction" means here.
    """
    n, k = bundle.n_nodes, int(max_in_degree)
    idx = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], (n, k)).copy()
    weight = np.zeros((n, k), dtype=np.float32)
    sign = np.zeros((n, k), dtype=np.float32)
    valid = np.zeros((n, k), dtype=bool)
    w_max = float(bundle.in_weight.max()) if bundle.in_weight.size else 1.0
    w_max = max(w_max, 1.0)
    for u in range(n):
        lo, hi = int(bundle.in_indptr[u]), int(bundle.in_indptr[u + 1])
        deg = hi - lo
        take = min(deg, k)
        sel = (
            # top-K by synapse count (stable: lowest indices win ties)
            np.argsort(-bundle.in_weight[lo:hi], kind="stable")[:take]
            if take < deg
            else np.arange(deg)
        )
        idx[u, :take] = bundle.in_indices[lo:hi][sel].astype(np.int32)
        weight[u, :take] = bundle.in_weight[lo:hi][sel].astype(np.float32) / w_max
        sign[u, :take] = bundle.in_sign[lo:hi][sel].astype(np.float32)
        valid[u, :take] = True
    return EdgeSet(idx, weight, sign, valid)


def empty_edge_set(n_vocab: int, max_in_degree: int) -> EdgeSet:
    """All-padding EdgeSet (no real edges) - shape scaffold for init/dev."""
    n, k = int(n_vocab), int(max_in_degree)
    return EdgeSet(
        np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], (n, k)).copy(),
        np.zeros((n, k), dtype=np.float32),
        np.zeros((n, k), dtype=np.float32),
        np.zeros((n, k), dtype=bool),
    )


def pad_edge_set(edge_set: EdgeSet, n_vocab: int) -> EdgeSet:
    """Rehome an EdgeSet built for a smaller node table onto this V axis.

    Used by the wrong-animal control: the C. elegans mask keeps its own
    wiring on the first rows (self-gathers of its nodes) and the rest of
    the fly vocab's rows get empty self-padding - same input space,
    wrong-animal edges.
    """
    v, k = edge_set.presyn_idx.shape
    n = int(n_vocab)
    if n <= v:
        return edge_set
    idx = np.broadcast_to(
        np.arange(n, dtype=np.int32)[:, None], (n, k)
    ).copy()
    idx[:v] = edge_set.presyn_idx
    weight = np.zeros((n, k), dtype=np.float32)
    weight[:v] = edge_set.edge_weight
    sign = np.zeros((n, k), dtype=np.float32)
    sign[:v] = edge_set.edge_sign
    valid = np.zeros((n, k), dtype=bool)
    valid[:v] = edge_set.valid
    return EdgeSet(idx, weight, sign, valid)


# ---------------------------------------------------------------------------
# Model modules (flax.linen) - defined on a guarded base so import works
# without jax installed
# ---------------------------------------------------------------------------


class _Embedder(_LinenModule):
    """Count-vector -> d_model embedding keeping the unit axis.

    Each unit slot u carries its event count scalar c_{t,u}; a shared
    linear 1->d projection embeds it per position, plus a learned time
    positional table and a per-unit identity embedding (the spec's "type
    embedding bias"). Optional stimulus-family context adds one learned
    slot embedding broadcast over all positions."""

    d_model: int
    n_vocab: int
    max_len: int
    dropout: float
    use_stim: bool = False

    @_compact
    def __call__(self, x: jax.Array, deterministic: bool, stim_ids=None) -> jax.Array:
        _require_jax()
        b, t, v = x.shape
        if v != self.n_vocab:
            raise ValueError(f"input V={v} != n_vocab={self.n_vocab}")
        if t > self.max_len:
            raise ValueError(f"input T={t} exceeds max_len={self.max_len}")
        h = nn.Dense(self.d_model, name="in_proj")(x[..., None])
        pos = self.param(
            "pos_emb", nn.initializers.normal(0.02), (self.max_len, self.d_model)
        )
        unit = self.param(
            "unit_emb", nn.initializers.normal(0.02), (self.n_vocab, self.d_model)
        )
        h = h + pos[None, :t, None, :] + unit[None, None, :, :]
        if self.use_stim:
            emb = nn.Embed(N_STIMULUS_SLOTS, self.d_model, name="stim_emb")
            ids = jnp.zeros((b,), jnp.int32) if stim_ids is None else stim_ids
            h = h + emb(ids)[:, None, None, :]
        return nn.Dropout(self.dropout)(h, deterministic=deterministic)


class _Ffn(_LinenModule):
    """Position-wise GELU feed-forward (shared by every block)."""

    d_ff: int
    d_model: int
    dropout: float

    @_compact
    def __call__(self, h: jax.Array, deterministic: bool) -> jax.Array:
        _require_jax()
        z = nn.Dense(self.d_ff, name="ff_in")(h)
        z = jax.nn.gelu(z)
        z = nn.Dropout(self.dropout)(z, deterministic=deterministic)
        return nn.Dense(self.d_model, name="ff_out")(z)


class _TimeAxisBlock(_LinenModule):
    """Even layer: causal attention along T per unit (V folded to batch)."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout: float

    @_compact
    def __call__(self, h: jax.Array, deterministic: bool) -> jax.Array:
        _require_jax()
        b, t, v, d = h.shape
        z = nn.LayerNorm(name="ln1")(h)
        z = z.transpose(0, 2, 1, 3).reshape(b * v, t, d)
        causal = jnp.tril(jnp.ones((t, t), dtype=bool))[None, None]
        z = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=d,
            out_features=d,
            dropout_rate=self.dropout,
            name="time_attn",
        )(z, mask=causal, deterministic=deterministic)
        z = z.reshape(b, v, t, d).transpose(0, 2, 1, 3)
        h = h + z
        z = nn.LayerNorm(name="ln2")(h)
        return h + _Ffn(self.d_ff, self.d_model, self.dropout, name="ffn")(
            z, deterministic
        )


class _ConnectomeBlock(_LinenModule):
    """Odd layer: connectome-masked cross-time attention.

    Query = post unit u at step t (t >= 1); keys/values = representations
    of u's presynaptic partners at t-1, gathered through the EdgeSet.
    score = q.k / sqrt(d_head) + edge_gain_h * log(w_eps + w_uv); invalid
    (padding) slots are masked to -inf. The value channel is modulated by
    a learnable per-head sign gate ``(1 + sign_gain_h * sign_uv)`` so
    excitatory/inhibitory edges can contribute opposite-signed values.
    Post units at t = 0 have no t-1 and pass through on the residual;
    units with no valid in-edges get a zeroed attention output.
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout: float
    w_eps: float

    @_compact
    def __call__(self, h: jax.Array, edge_set: EdgeSet, deterministic: bool) -> jax.Array:
        _require_jax()
        b, t, v, d = h.shape
        n_heads, d_head = self.n_heads, d // self.n_heads
        presyn, e_weight, e_sign, e_valid = edge_set
        k_nei = presyn.shape[1]

        z = nn.LayerNorm(name="ln1")(h)
        q = nn.Dense(d, name="q")(z[:, 1:]).reshape(b, t - 1, v, n_heads, d_head)
        kk = nn.Dense(d, name="k")(z[:, :-1])
        vv = nn.Dense(d, name="v")(z[:, :-1])
        # gather presyn partner reprs: (B, T-1, V, K, d) -> heads
        kn = kk[:, :, presyn].reshape(b, t - 1, v, k_nei, n_heads, d_head)
        vn = vv[:, :, presyn].reshape(b, t - 1, v, k_nei, n_heads, d_head)

        gain = self.param("edge_gain", nn.initializers.ones, (n_heads,))
        sgain = self.param("sign_gain", nn.initializers.zeros, (n_heads,))
        # per-head scaled log-weight bias: score[...,k,h] += gain[h]*log(w_eps + w_uk)
        bias = (
            gain[None, None, None, None, :]
            * jnp.log(self.w_eps + e_weight)[None, None, :, :, None]
        )
        score = (
            jnp.einsum("btvhd,btvkhd->btvkh", q, kn) / jnp.sqrt(jnp.float32(d_head))
            + bias
        )
        score = jnp.where(e_valid[None, None, :, :, None], score, jnp.float32(-1e30))
        prob = jax.nn.softmax(score, axis=-2)                  # over K
        vn = vn * (1.0 + sgain[None, None, None, None, :, None]
                   * e_sign[None, None, :, :, None, None])
        out = jnp.einsum("btvkh,btvkhd->btvhd", prob, vn)
        out = out * e_valid.any(axis=-1)[None, None, :, None, None]  # isolated units
        out = nn.Dense(d, name="out")(out.reshape(b, t - 1, v, d))
        h = h.at[:, 1:].add(out)
        z = nn.LayerNorm(name="ln2")(h)
        return h + _Ffn(self.d_ff, self.d_model, self.dropout, name="ffn")(
            z, deterministic
        )


class _UnitAxisBlock(_LinenModule):
    """Odd layer of the dense twin: full attention on the unit axis V.

    Same (B,T,V,d) interface, same projection shapes as _ConnectomeBlock
    but with no mask - every unit attends to every other unit at the same
    step. This is the parameter-matched dense baseline of T-MODEL-5.
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout: float

    @_compact
    def __call__(self, h: jax.Array, deterministic: bool) -> jax.Array:
        _require_jax()
        b, t, v, d = h.shape
        z = nn.LayerNorm(name="ln1")(h)
        z = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=d,
            out_features=d,
            dropout_rate=self.dropout,
            name="unit_attn",
        )(z.reshape(b * t, v, d), deterministic=deterministic)
        h = h + z.reshape(b, t, v, d)
        z = nn.LayerNorm(name="ln2")(h)
        return h + _Ffn(self.d_ff, self.d_model, self.dropout, name="ffn")(
            z, deterministic
        )


class _GnnHop(_LinenModule):
    """One message-passing hop: mean+max aggregate over the same in-edge
    lists the connectome layer gathers, then an MLP update. Aggregation
    is same-step (spatial) - the GNN propagates instantaneously, a
    documented difference from the transformer's t-1 delay."""

    d_model: int
    d_ff: int
    dropout: float

    @_compact
    def __call__(self, h: jax.Array, edge_set: EdgeSet, deterministic: bool) -> jax.Array:
        _require_jax()
        presyn, e_weight, e_sign, e_valid = edge_set
        nb = h[:, :, presyn]                                   # (B,T,V,K,d)
        w = jnp.where(e_valid, e_weight * e_sign, 0.0)         # signed channel
        denom = jnp.maximum(jnp.abs(w).sum(axis=-1), 1e-6)     # (V,)
        mean = jnp.einsum("btvkd,vk->btvd", nb, w) / denom[None, None, :, None]
        mx = jnp.where(e_valid[None, None, :, :, None], nb, -1e30).max(axis=3)
        mx = mx * e_valid.any(axis=-1)[None, None, :, None]
        agg = jnp.concatenate([mean, mx], axis=-1)             # (B,T,V,2d)
        z = jnp.concatenate([h, agg], axis=-1)                 # (B,T,V,3d)
        z = nn.Dense(self.d_ff, name="up_in")(z)
        z = jax.nn.gelu(z)
        z = nn.Dropout(self.dropout)(z, deterministic=deterministic)
        z = nn.Dense(self.d_model, name="up_out")(z)
        return h + z


class _Readout(_LinenModule):
    """Final LayerNorm + two shared per-unit scalar heads: (B,T,V,d) ->
    (B,T,V) event logits and (B,T,V) rate prediction."""

    n_vocab: int

    @_compact
    def __call__(self, h: jax.Array) -> tuple[jax.Array, jax.Array]:
        _require_jax()
        z = nn.LayerNorm(name="ln_out")(h)
        # shared projection + per-unit bias: units carry their own
        # baseline firing-rate prior (unit identity already lives in h
        # via unit_emb; the bias is the cheap additive version)
        e_bias = self.param("event_bias", nn.initializers.zeros, (self.n_vocab,))
        r_bias = self.param("rate_bias", nn.initializers.zeros, (self.n_vocab,))
        return (
            nn.Dense(1, name="event_head")(z)[..., 0] + e_bias[None, None, :],
            nn.Dense(1, name="rate_head")(z)[..., 0] + r_bias[None, None, :],
        )


class ConnectomeTransformer(_LinenModule):
    """Tokens (B,T,V) count vectors -> (event logits, rates) each (B,T,V).

    Even layers = causal time attention per unit; odd layers =
    connectome-masked attention pre(t) -> post(t+1) via the EdgeSet
    argument. The mask is a module INPUT (T-SEX-5 swaps it per call).
    """

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 12
    n_vocab: int = 0
    max_len: int = 512
    dropout: float = 0.1
    w_eps: float = 1e-3
    use_stim: bool = False

    @_compact
    def __call__(
        self, x: jax.Array, edge_set: EdgeSet, deterministic: bool = True, stim_ids=None
    ) -> tuple[jax.Array, jax.Array]:
        _require_jax()
        if edge_set.presyn_idx.shape[0] != x.shape[2]:
            raise ValueError(
                f"edge set V={edge_set.presyn_idx.shape[0]} != input V={x.shape[2]}"
            )
        h = _Embedder(self.d_model, self.n_vocab, self.max_len, self.dropout,
                      self.use_stim, name="embed")(x, deterministic, stim_ids)
        for i in range(self.n_layers):
            if i % 2 == 0:
                h = _TimeAxisBlock(self.d_model, self.n_heads, self.d_ff,
                                   self.dropout, name=f"time_{i}")(h, deterministic)
            else:
                h = _ConnectomeBlock(self.d_model, self.n_heads, self.d_ff,
                                     self.dropout, self.w_eps,
                                     name=f"conn_{i}")(h, edge_set, deterministic)
        return _Readout(self.n_vocab, name="readout")(h)


class DenseTransformer(_LinenModule):
    """Parameter-matched no-mask twin (T-MODEL-5): same interface, same
    even-layer time attention, but odd layers run full unit-axis
    attention instead of the gathered connectome mask. ``edge_set`` is
    accepted for signature parity and ignored."""

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 12
    n_vocab: int = 0
    max_len: int = 512
    dropout: float = 0.1
    w_eps: float = 1e-3
    use_stim: bool = False

    @_compact
    def __call__(
        self, x: jax.Array, edge_set: EdgeSet | None = None,
        deterministic: bool = True, stim_ids=None,
    ) -> tuple[jax.Array, jax.Array]:
        _require_jax()
        h = _Embedder(self.d_model, self.n_vocab, self.max_len, self.dropout,
                      self.use_stim, name="embed")(x, deterministic, stim_ids)
        for i in range(self.n_layers):
            if i % 2 == 0:
                h = _TimeAxisBlock(self.d_model, self.n_heads, self.d_ff,
                                   self.dropout, name=f"time_{i}")(h, deterministic)
            else:
                h = _UnitAxisBlock(self.d_model, self.n_heads, self.d_ff,
                                   self.dropout, name=f"unit_{i}")(h, deterministic)
        return _Readout(self.n_vocab, name="readout")(h)


class GNNBaseline(_LinenModule):
    """Message-passing baseline (T-MODEL-10): mean+max aggregation over
    the same in-edge lists, MLP update, ``n_layers`` hops, same
    input/output signature and same embedder/readout as the transformer."""

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 12
    n_vocab: int = 0
    max_len: int = 512
    dropout: float = 0.1
    w_eps: float = 1e-3
    use_stim: bool = False

    @_compact
    def __call__(
        self, x: jax.Array, edge_set: EdgeSet, deterministic: bool = True, stim_ids=None
    ) -> tuple[jax.Array, jax.Array]:
        _require_jax()
        if edge_set.presyn_idx.shape[0] != x.shape[2]:
            raise ValueError(
                f"edge set V={edge_set.presyn_idx.shape[0]} != input V={x.shape[2]}"
            )
        h = _Embedder(self.d_model, self.n_vocab, self.max_len, self.dropout,
                      self.use_stim, name="embed")(x, deterministic, stim_ids)
        for i in range(self.n_layers):
            h = _GnnHop(self.d_model, self.d_ff, self.dropout, name=f"hop_{i}")(
                h, edge_set, deterministic
            )
        return _Readout(self.n_vocab, name="readout")(h)


_ARCH_CLASSES = {
    "connectome": ConnectomeTransformer,
    "dense": DenseTransformer,
    "gnn": GNNBaseline,
}


def _module_from_config(cfg: ModelConfig):
    """Instantiate the nn.Module for cfg.arch."""
    return _ARCH_CLASSES[cfg.arch](
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        n_layers=cfg.n_layers,
        n_vocab=cfg.n_vocab,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
        w_eps=cfg.w_eps,
        use_stim=cfg.use_stim,
    )


# ---------------------------------------------------------------------------
# Build / param accounting
# ---------------------------------------------------------------------------


def count_params(params: Any) -> int:
    """Total scalar parameter count of a params pytree."""
    _require_jax()
    return int(sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(params)))


def build_model(cfg: ModelConfig, mask_arrays: EdgeSet | None = None):
    """(module, params): init the cfg.arch module under a derived seed.

    ``mask_arrays`` is the EdgeSet used at init (its values are masked
    inputs, not parameters - only shapes matter here). Falls back to
    ``cfg.edge_set``, then to an all-padding scaffold of the right shape.
    """
    _require_jax()
    if cfg.d_model % cfg.n_heads:
        raise ValueError(f"d_model={cfg.d_model} not divisible by n_heads={cfg.n_heads}")
    if cfg.n_vocab < 1:
        raise ValueError("n_vocab must be set (vocab.size()) before build_model")
    model = _module_from_config(cfg)
    edge_set = mask_arrays if mask_arrays is not None else cfg.edge_set
    if edge_set is None:
        edge_set = empty_edge_set(cfg.n_vocab, cfg.max_in_degree)
    key = jax.random.PRNGKey(cfg.init_seed())
    x0 = jnp.zeros((1, 2, cfg.n_vocab), jnp.float32)
    variables = model.init(key, x0, edge_set, deterministic=True)
    return model, variables["params"]


def param_match(cfg_dense: ModelConfig, cfg_masked: ModelConfig) -> dict[str, Any]:
    """T-MODEL-5 helper: parameter counts of dense vs masked twins.

    Returns {"dense", "masked", "rel_diff", "matched"}; ``matched`` holds
    when |dense - masked| / max <= 2% - the pre-registered twin tolerance.
    """
    _, p_dense = build_model(cfg_dense)
    _, p_masked = build_model(cfg_masked)
    n_dense, n_masked = count_params(p_dense), count_params(p_masked)
    rel = abs(n_dense - n_masked) / max(n_dense, n_masked, 1)
    return {"dense": n_dense, "masked": n_masked, "rel_diff": rel, "matched": rel <= 0.02}


# ---------------------------------------------------------------------------
# Loss - silence-robust event BCE + rate term (T-MODEL-8, EVAL 1.7)
# ---------------------------------------------------------------------------


def event_weights(y_events: Any, silence_weight: float) -> Any:
    """Per-element BCE weight: 1.0 on active units, ``silence_weight`` on
    inactive ones - the silence-robust weighting (a model must not win by
    predicting all-zero)."""
    return 1.0 - (1.0 - silence_weight) * (1.0 - y_events)


def compute_loss(params: Any, model: Any, edge_set: Any, batch: Any, cfg: ModelConfig,
                 dropout_key: Any = None):
    """Weighted event BCE + rate_weight * MSE on rates.

    loss = event_weight * sum(w * BCE) / sum(w) + rate_weight * mean(MSE)
    with w = event_weights(y_events, cfg.silence_weight). Returns
    (total, aux-dict) - aux carries the unweighted terms for logging.
    """
    _require_jax()
    x, y_events, y_rates = batch
    apply_kwargs: dict[str, Any] = {"deterministic": cfg.dropout <= 0 or dropout_key is None}
    if dropout_key is not None and cfg.dropout > 0:
        apply_kwargs["rngs"] = {"dropout": dropout_key}
    logits, rates = model.apply({"params": params}, x, edge_set, **apply_kwargs)
    bce = optax.sigmoid_binary_cross_entropy(logits, y_events)
    w = event_weights(y_events, cfg.silence_weight)
    event_term = (w * bce).sum() / jnp.maximum(w.sum(), 1e-6)
    rate_term = jnp.mean((rates - y_rates) ** 2)
    total = cfg.event_weight * event_term + cfg.rate_weight * rate_term
    return total, {"event": event_term, "rate": rate_term, "total": total}


# ---------------------------------------------------------------------------
# Checkpoints - pickle pytrees, no orbax dependency (resume is NORMAL)
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    params: Any,
    cfg: ModelConfig,
    step: int,
    opt_state: Any = None,
    losses: list[float] | None = None,
) -> Path:
    """Pickle {params, opt_state, cfg, step, losses} to ``path``.

    Params are stored via flax.serialization.to_state_dict (plain nested
    dicts of host arrays) so checkpoints don't depend on jax internals.
    """
    _require_jax()
    from flax import serialization

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _CHECKPOINT_FORMAT,
        "step": int(step),
        "params": jax.device_get(serialization.to_state_dict(params)),
        "opt_state": jax.device_get(opt_state) if opt_state is not None else None,
        "cfg": cfg.spec(),
        "losses": list(losses or []),
    }
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Inverse of save_checkpoint: {params, opt_state, cfg, step, losses}."""
    _require_jax()
    with Path(path).open("rb") as fh:
        payload = pickle.load(fh)
    if payload.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {_CHECKPOINT_FORMAT} checkpoint")
    return payload


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    cfg: ModelConfig,
    dataset: Any,
    work_dir: Path,
    resume_from: Path | None = None,
) -> Path:
    """optax Adam trainer. Returns the checkpoint dir.

    Dataset contract: a re-iterable of aligned-target batches
    ``(X_counts (B,T,V) f32, y_events (B,T,V) 0/1, y_rates (B,T,V) f32)``.
    For next-step training the producer shifts targets by one step (see
    ``synthetic_dataset``). The iterable must yield at least
    ``cfg.total_steps`` batches across epochs - it is re-iterated on
    exhaustion, so pass a list/factory, not a one-shot generator.

    Checkpoints land in ``work_dir/checkpoints`` every
    ``cfg.checkpoint_every`` steps (numbered + ``ckpt_latest.pkl``).
    Resume-from-checkpoint is the NORMAL path (plan session rules): pass
    ``resume_from`` or leave ``ckpt_latest.pkl`` in place - training
    continues at the recorded step with restored optimizer state.
    """
    _require_jax()
    edge_set = cfg.edge_set
    if edge_set is None and cfg.arch != "dense":
        raise ValueError("cfg.edge_set is required for arch=" + cfg.arch)
    ckpt_dir = Path(work_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model, params = build_model(cfg, edge_set)
    schedule = (
        optax.linear_schedule(0.0, cfg.lr, cfg.warmup_steps)
        if cfg.warmup_steps > 0
        else cfg.lr
    )
    tx = optax.adam(schedule)
    opt_state = tx.init(params)

    start_step, losses = 0, []
    resume_path = Path(resume_from) if resume_from is not None else ckpt_dir / _CKPT_LATEST
    if resume_path.exists():
        ckpt = load_checkpoint(resume_path)
        params = ckpt["params"]
        opt_state = ckpt["opt_state"] if ckpt["opt_state"] is not None else opt_state
        start_step = int(ckpt["step"])
        losses = list(ckpt["losses"])
        print(f"[resume] {resume_path} at step {start_step}", flush=True)

    es = EdgeSet(*(jnp.asarray(a) for a in edge_set)) if edge_set is not None else None
    base_key = jax.random.PRNGKey(cfg.init_seed())

    def step_fn(params, opt_state, batch, dkey):
        (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(
            params, model, es, batch, cfg, dkey
        )
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    step_jit = jax.jit(step_fn)
    iterator = iter(dataset)
    step = start_step
    while step < cfg.total_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataset)
            batch = next(iterator)
        step += 1
        batch = tuple(jnp.asarray(a, jnp.float32) for a in batch)
        dkey = jax.random.fold_in(base_key, step)
        params, opt_state, loss, aux = step_jit(params, opt_state, batch, dkey)
        losses.append(float(loss))
        if step % 50 == 0 or step == start_step + 1:
            print(
                f"[train] step {step}/{cfg.total_steps} loss={float(loss):.4f} "
                f"event={float(aux['event']):.4f} rate={float(aux['rate']):.4f}",
                flush=True,
            )
        if step % cfg.checkpoint_every == 0 or step == cfg.total_steps:
            for target in (ckpt_dir / f"ckpt_{step:08d}.pkl", ckpt_dir / _CKPT_LATEST):
                save_checkpoint(target, params, cfg, step, opt_state, losses)
    return ckpt_dir


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def predict_events(params: Any, model: Any, X: Any, mask: EdgeSet) -> np.ndarray:
    """(B,T,V) event probabilities = sigmoid(event logits) under ``mask``."""
    _require_jax()
    logits, _ = model.apply(
        {"params": params}, jnp.asarray(X, jnp.float32), mask, deterministic=True
    )
    return np.asarray(jax.nn.sigmoid(logits))


def sex_conditioned_forward(
    params: Any,
    model: Any,
    inputs: Any,
    mask_male: EdgeSet,
    mask_female: EdgeSet,
    sex: str,
) -> tuple[Any, Any]:
    """One parameter set under the male or female edge set (T-SEX-5).

    Sex conditioning swaps the EdgeSet - there are no sex-specific
    parameters; a male-specific type absent from the female mask simply
    contributes nothing to female-conditioned gathers (T-SEX-4's leak
    check runs on these outputs).
    """
    _require_jax()
    edge_set = {"male": mask_male, "female": mask_female}[sex]
    return model.apply(
        {"params": params}, jnp.asarray(inputs, jnp.float32), edge_set, deterministic=True
    )


# ---------------------------------------------------------------------------
# Synthetic dataset - real weakly-structured data on the bundle
# ---------------------------------------------------------------------------


def _threshold_rollout(
    bundle: GraphBundle, n_trials: int, n_steps: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Instant-synapse threshold-model rollouts on the real in-edge lists.

    Fallback for ``flygpt.sim_backend.LIFSim`` when the simulator API
    isn't available in-process: the same equations as the spec's
    discrete update in the ``instant`` synapse limit (drive arrives as
    voltage kicks on chosen source nodes). Returns ``(spikes (n,T,V)
    f32, memb (n,T,V) f32 pre-reset)``. Data is graph-structured, not
    noise: only nodes with real out-edges are driven, and activity
    propagates along signed, weight-normalized edges.
    """
    v_size = bundle.n_nodes
    w = np.zeros((v_size, v_size), dtype=np.float32)  # W[post, pre]
    for post in range(v_size):
        lo, hi = int(bundle.in_indptr[post]), int(bundle.in_indptr[post + 1])
        w[post, bundle.in_indices[lo:hi]] = (
            bundle.in_weight[lo:hi] * bundle.in_sign[lo:hi]
        ).astype(np.float32)
    row = np.abs(w).sum(axis=1, keepdims=True)
    w = w / np.maximum(row, 1.0)

    drivers = np.flatnonzero(np.diff(bundle.out_indptr) > 0)
    if drivers.size == 0:
        drivers = np.arange(v_size)
    decay, theta, gain, drive_p = 0.65, 0.5, 1.1, 0.5

    spikes = np.zeros((n_trials, n_steps, v_size), dtype=np.float32)
    memb = np.zeros((n_trials, n_steps, v_size), dtype=np.float32)
    for trial in range(n_trials):
        drv = drivers[rng.choice(drivers.size, size=int(rng.integers(1, 3)), replace=False)]
        v = np.zeros(v_size, dtype=np.float32)
        refractory = np.zeros(v_size, dtype=bool)
        for t in range(n_steps):
            drive = gain * (rng.random(drv.size) < drive_p)
            kick = np.zeros(v_size, dtype=np.float32)
            kick[drv] = drive
            v = decay * v + w @ (spikes[trial, t - 1] if t else 0.0) + kick
            v[refractory] = 0.0
            memb[trial, t] = v
            fired = v >= theta
            spikes[trial, t] = fired
            refractory = fired
    return spikes, memb


def synthetic_dataset(
    bundle: GraphBundle, n_trials: int, T: int, vocab_size: int, seed: int = 0
):
    """Generator of REAL weakly-structured training batches on the bundle.

    Tries ``flygpt.sim_backend.LIFSim`` lazily first (the numpy reference
    path per impl-spec); while that module is unavailable it falls back
    to the documented threshold-model rollout in ``_threshold_rollout`` -
    either way the data propagates along the real graph, so tests that
    train genuinely exercise mask usefulness.

    Yields ``n_trials`` items, each ``(X_counts (1,T,V) f32, y_events
    (1,T,V) 0/1, y_rates (1,T,V) f32)`` where ``y_*[t]`` is the step-t+1
    target of ``X[t]`` (next-step training per plan Phase 3). ``y_rates``
    is the pre-reset membrane trace - the continuous rate proxy.
    Deterministic given ``(bundle, n_trials, T, vocab_size, seed)``.
    """
    rng = rng_from_seed(
        ("synthetic_dataset", int(n_trials), int(T), int(vocab_size), int(seed))
    )
    n_steps = T + 1
    roll = None
    try:
        from flygpt import sim_backend as _sb  # lazy: optional Phase-1 module
    except ImportError:  # pragma: no cover
        _sb = None
    if getattr(_sb, "LIFSim", None) is not None:
        try:
            roll = _lifsim_rollout(_sb, bundle, n_trials, n_steps, rng)
        except (AttributeError, NotImplementedError):
            roll = None  # API not final - documented fallback below
    if roll is None:
        roll = _threshold_rollout(bundle, n_trials, n_steps, rng)
    spikes, memb = roll
    for trial in range(n_trials):
        x = spikes[trial, :T][None].astype(np.float32)
        y_events = spikes[trial, 1:][None].astype(np.float32)
        y_rates = memb[trial, 1:][None].astype(np.float32)
        yield x, y_events, y_rates


def _lifsim_rollout(_sb, bundle, n_trials, n_steps, rng):
    """Bridge to flygpt.sim_backend.LIFSim when it lands (Phase 1).

    Uses the documented API surface: ``LIFSim(bundle, SimConfig)``,
    ``reset(seed)``, ``run(n_steps, driver=...)`` returning a TrialResult
    with ``events_step/events_node`` and ``rates``. The driver injects
    voltage on the chosen driver nodes via ``sim.v`` when the attribute
    exists - any other surface raises and the caller falls back to the
    threshold rollout.
    """
    sim = _sb.LIFSim(bundle, _sb.SimConfig(backend="numpy", synapse_model="instant"))
    drivers = np.flatnonzero(np.diff(bundle.out_indptr) > 0)
    spikes = np.zeros((n_trials, n_steps, bundle.n_nodes), dtype=np.float32)
    memb = np.zeros_like(spikes)
    for trial in range(n_trials):
        sim.reset(int(rng.integers(0, 2**31 - 1)))
        drv = drivers[rng.choice(drivers.size, size=int(rng.integers(1, 3)), replace=False)]

        def driver(step, sim, _drv=drv, _rng=rng):
            if not hasattr(sim, "v"):
                raise AttributeError("sim driver API not final")
            sim.v[_drv] += (_rng.random(_drv.size) < 0.5).astype(np.float32)

        result = sim.run(n_steps, driver=driver)
        per_step = result.events_per_step()
        for t in range(min(n_steps, len(per_step))):
            spikes[trial, t, np.asarray(per_step[t], dtype=np.int64)] = 1.0
        memb[trial] = np.asarray(result.rates, dtype=np.float32)[None, :]
    return spikes, memb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _npz_dataset(data_dir: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load a shard dir of batch_*.npz files (x, y_events, y_rates)."""
    batches = []
    for f in sorted(Path(data_dir).glob("batch_*.npz")):
        z = np.load(f)
        batches.append((z["x"], z["y_events"], z["y_rates"]))
    if not batches:
        raise FileNotFoundError(f"no batch_*.npz files under {data_dir}")
    return batches


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=str, required=True,
                    help="dataset dir of batch_*.npz files, or "
                         "'synthetic:<graph_dir>:<cutoff>' for dev rollouts")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=None, help="override total_steps")
    ap.add_argument("--dev", action="store_true", help="apply the config's dev overrides")
    ap.add_argument("--resume-from", type=Path, default=None)
    args = ap.parse_args(argv)

    cfg = load_model_config(args.config, dev=args.dev)
    if args.steps is not None:
        cfg.total_steps = args.steps

    if args.data.startswith("synthetic"):
        from flygpt.schema import load_graph_bundle

        _, graph_dir, cutoff = (args.data.split(":") + ["", "5"])[:3]
        bundle = load_graph_bundle(Path(graph_dir), int(cutoff))
        cfg.n_vocab = bundle.n_nodes
        cfg.edge_set = edge_set_from_bundle(bundle, cfg.max_in_degree)
        dataset = list(
            synthetic_dataset(bundle, n_trials=cfg.batch_trials * 4, T=9,
                              vocab_size=cfg.n_vocab, seed=cfg.seed)
        )
    else:
        dataset = _npz_dataset(Path(args.data))
        sample = dataset[0][0]
        cfg.n_vocab = int(sample.shape[-1])
        if cfg.edge_set is None and cfg.arch != "dense":
            raise ValueError(
                "npz datasets need cfg.edge_set - build it from the graph "
                "and attach it (or use synthetic:<graph_dir>:<cutoff>)"
            )

    train(cfg, dataset, args.work_dir, resume_from=args.resume_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
