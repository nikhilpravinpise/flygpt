"""Event lists -> population vectors/patches (T-TOK-*).

Hybrid tokenizer for the FlyGPT corpus (schema ``schema_v1``), implementing
the contract in ``docs/impl-spec.md`` and the integrity checks in
``docs/plan.md`` section 3 / ``EVAL.md`` (T-TOK-1..5).

Vocab layout (constants from ``flygpt.schema``)::

    0                            PAD
    1                            UNK   (untyped nodes - T-TOK-3)
    2, 3                         SEX_MALE, SEX_FEMALE
    4 .. 67                      stimulus-family slots (N_STIMULUS_SLOTS=64)
    68 .. 99                     behavior-head slots (N_BEHAVIOR_SLOTS=32)
    100 ..                       manifest cell-type tokens, sorted types
                                 mapped to TYPE_TOKEN_BASE+i
    TYPE_TOKEN_BASE+n_types ..   per-node instance tokens for designated
                                 circuits - one contiguous block directly
                                 after the type tokens (T-TOK-2)

Every range is pairwise disjoint. ``Vocab.size()`` equals
``TYPE_TOKEN_BASE + n_types + n_instances``, i.e. the reserved block +
manifest types + instance slots exactly, which is also "last id + 1"
because each block is contiguous (T-TOK-5).

Per-node token rule (``node_to_token``): a node emits its instance token
when it belongs to a designated instance circuit (config
``instance_circuits`` ``superclasses``/``types`` lists resolved against
the nodes table) AND carries a type; otherwise its type token. Untyped
nodes (type null/"") always route to VOCAB_UNK - never dropped and never
merged into a real type (T-TOK-3). Nodes whose type is absent from the
vocab (e.g. a vocab built on a different bundle) also route to UNK.

Binning (T-TOK-4): bin index = ``floor((t - t0)/dt)``. Events already
carrying integer sim steps (schema column ``step``) are bin indices
directly and need no ``dt``; float event times require ``dt`` (plus an
optional ``t0``) and are floored through the literal formula. The floor
is taken on the float ratio exactly as specified - corpus events carry
integer steps, so binary representation error at exact ``dt`` multiples
never occurs on the real path.

Round-trip (T-TOK-1): events -> counts -> events reproduces (bin, token)
exactly - the representation is a pure count per (bin, token), so the
quantization eps declared in ``configs/tokenizer_v1.yaml`` is 0.

Usage:
    python -m flygpt.tokenize --config configs/tokenizer_v1.yaml --build
    python -m flygpt.tokenize --config configs/tokenizer_v1.yaml --print
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flygpt.schema import (
    N_BEHAVIOR_SLOTS,
    N_STIMULUS_SLOTS,
    SCHEMA_VERSION,
    TYPE_TOKEN_BASE,
    VOCAB_PAD,
    VOCAB_RESERVED_BASE,
    VOCAB_SEX_FEMALE,
    VOCAB_SEX_MALE,
    VOCAB_UNK,
    GraphBundle,
    git_sha,
    load_graph_bundle,
    sha256_file,
    utcnow,
)

STIMULUS_BASE = VOCAB_RESERVED_BASE
BEHAVIOR_BASE = VOCAB_RESERVED_BASE + N_STIMULUS_SLOTS


def _reserved_record() -> dict[str, int]:
    """The fixed reserved-block layout recorded in every vocab artifact."""
    return {
        "pad": VOCAB_PAD,
        "unk": VOCAB_UNK,
        "sex_male": VOCAB_SEX_MALE,
        "sex_female": VOCAB_SEX_FEMALE,
        "stimulus_base": STIMULUS_BASE,
        "stimulus_slots": N_STIMULUS_SLOTS,
        "behavior_base": BEHAVIOR_BASE,
        "behavior_slots": N_BEHAVIOR_SLOTS,
        "type_token_base": TYPE_TOKEN_BASE,
    }


@dataclass
class Vocab:
    """Hybrid vocab: manifest cell types + designated per-node instances.

    ``type_to_id`` maps each sorted manifest type to ``TYPE_TOKEN_BASE+i``.
    ``instance_to_id`` maps a designated instance-circuit node index into
    the contiguous block starting at ``TYPE_TOKEN_BASE + n_types``.
    ``stimulus_to_id`` maps a stimulus family_id into the reserved slots
    at ``VOCAB_RESERVED_BASE``; ``behavior_to_id`` maps a behavior head
    name into the reserved slots right after them.
    """

    type_to_id: dict[str, int]
    instance_to_id: dict[int, int]
    stimulus_to_id: dict[str, int]
    behavior_to_id: dict[str, int]
    version: str = "tokenizer_v1"

    def size(self) -> int:
        """One past the last token id = reserved + types + instances."""
        return TYPE_TOKEN_BASE + len(self.type_to_id) + len(self.instance_to_id)

    def validate(self) -> None:
        """Check the T-TOK-2/T-TOK-5 layout invariants; raise on violation."""
        n_types = len(self.type_to_id)
        type_ids = sorted(self.type_to_id.values())
        if type_ids != list(range(TYPE_TOKEN_BASE, TYPE_TOKEN_BASE + n_types)):
            raise ValueError("type ids are not the contiguous block at TYPE_TOKEN_BASE")
        inst_ids = sorted(self.instance_to_id.values())
        inst_base = TYPE_TOKEN_BASE + n_types
        if inst_ids != list(range(inst_base, inst_base + len(inst_ids))):
            raise ValueError("instance ids are not the contiguous block after type tokens")
        for ids, lo, hi, label in (
            (self.stimulus_to_id, STIMULUS_BASE, BEHAVIOR_BASE, "stimulus"),
            (self.behavior_to_id, BEHAVIOR_BASE, TYPE_TOKEN_BASE, "behavior"),
        ):
            vals = list(ids.values())
            if len(vals) != len(set(vals)) or any(not lo <= v < hi for v in vals):
                raise ValueError(f"{label} token ids collide or escape [{lo}, {hi})")
        all_ids = type_ids + inst_ids + list(self.stimulus_to_id.values())
        all_ids += list(self.behavior_to_id.values())
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("token id collision across vocab ranges (T-TOK-2)")

    def register_stimulus(self, family_id: str) -> int:
        """Assign ``family_id`` the next free stimulus slot (idempotent)."""
        if family_id in self.stimulus_to_id:
            return self.stimulus_to_id[family_id]
        tok = max(self.stimulus_to_id.values(), default=STIMULUS_BASE - 1) + 1
        if tok >= BEHAVIOR_BASE:
            raise ValueError(f"stimulus slots exhausted ({N_STIMULUS_SLOTS})")
        self.stimulus_to_id[family_id] = tok
        return tok

    def register_behavior(self, head: str) -> int:
        """Assign ``head`` the next free behavior slot (idempotent)."""
        if head in self.behavior_to_id:
            return self.behavior_to_id[head]
        tok = max(self.behavior_to_id.values(), default=BEHAVIOR_BASE - 1) + 1
        if tok >= TYPE_TOKEN_BASE:
            raise ValueError(f"behavior slots exhausted ({N_BEHAVIOR_SLOTS})")
        self.behavior_to_id[head] = tok
        return tok

    def save(self, path: Path) -> None:
        """Write the vocab JSON: version + counts + reserved-block record
        + the four token maps (T-TOK-5 record kept inside the artifact)."""
        path = Path(path)
        payload = {
            "version": self.version,
            "schema": SCHEMA_VERSION,
            "type_count": len(self.type_to_id),
            "instance_count": len(self.instance_to_id),
            "reserved": _reserved_record(),
            "type_to_id": {t: self.type_to_id[t] for t in sorted(self.type_to_id)},
            "instance_to_id": {
                str(k): self.instance_to_id[k] for k in sorted(self.instance_to_id)
            },
            "stimulus_to_id": {k: self.stimulus_to_id[k] for k in sorted(self.stimulus_to_id)},
            "behavior_to_id": {k: self.behavior_to_id[k] for k in sorted(self.behavior_to_id)},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Vocab:
        """Read a vocab JSON written by ``save``; fails closed on a
        reserved-block or count mismatch (schema/vocab drift, T-TOK-5)."""
        data = json.loads(Path(path).read_text())
        if data.get("reserved") != _reserved_record():
            raise ValueError(
                "vocab reserved block does not match schema constants - rebuild the vocab"
            )
        vocab = cls(
            type_to_id={str(k): int(v) for k, v in data["type_to_id"].items()},
            instance_to_id={int(k): int(v) for k, v in data["instance_to_id"].items()},
            stimulus_to_id={str(k): int(v) for k, v in data["stimulus_to_id"].items()},
            behavior_to_id={str(k): int(v) for k, v in data["behavior_to_id"].items()},
            version=str(data.get("version", "tokenizer_v1")),
        )
        if int(data.get("type_count", -1)) != len(vocab.type_to_id):
            raise ValueError("vocab type_count does not match the type_to_id map")
        vocab.validate()
        return vocab


def _slot_map(names: list[str], base: int, n_slots: int, label: str) -> dict[str, int]:
    """Sorted unique names -> contiguous ids starting at ``base``."""
    uniq = sorted({str(s) for s in names})
    if len(uniq) > n_slots:
        raise ValueError(
            f"{label} list has {len(uniq)} entries - only {n_slots} slots reserved"
        )
    return {name: base + i for i, name in enumerate(uniq)}


def _instance_nodes(bundle: GraphBundle, spec: dict[str, Any]) -> np.ndarray:
    """Sorted node indices designated as instance-circuit members.

    A node matches when its ``superclass`` is in ``spec["superclasses"]``
    or its ``type`` is in ``spec["types"]``, AND it carries a type -
    untyped nodes never get instance tokens (they route to UNK, T-TOK-3).
    """
    supers = sorted({str(s) for s in spec.get("superclasses") or []})
    types = sorted({str(t) for t in spec.get("types") or []})
    if not supers and not types:
        return np.empty(0, dtype=np.int64)
    type_arr = bundle.type_array()
    match = np.zeros(bundle.n_nodes, dtype=np.bool_)
    if types:
        match |= np.isin(type_arr, np.array(types))
    if supers:
        if "superclass" not in bundle.nodes.columns:
            raise KeyError(
                "instance_circuits lists superclasses but the nodes table "
                "has no 'superclass' column"
            )
        sc = bundle.nodes["superclass"].fill_null("").to_numpy().astype(str)
        match |= np.isin(sc, np.array(supers))
    return np.flatnonzero(match & (type_arr != ""))


def build_vocab(bundle: GraphBundle, cfg: dict[str, Any]) -> Vocab:
    """Build the hybrid vocab from a graph bundle + tokenizer config.

    Config keys consumed (all optional except where noted):
      ``version``                    - recorded on the vocab + manifest
      ``instance_circuits``          - {superclasses: [...], types: [...]}
      ``stimulus_families``          - seeded stimulus-slot names
      ``behavior_heads``             - seeded behavior-slot names

    Deterministic: types are the sorted manifest types, instance ids are
    assigned in node-index order, and slot names are sorted - so repeated
    calls on the same (bundle, cfg) produce identical maps.
    """
    type_to_id = {
        t: TYPE_TOKEN_BASE + i for i, t in enumerate(bundle.manifest_types)
    }
    inst_nodes = _instance_nodes(bundle, cfg.get("instance_circuits") or {})
    inst_base = TYPE_TOKEN_BASE + len(type_to_id)
    vocab = Vocab(
        type_to_id=type_to_id,
        instance_to_id={int(n): inst_base + i for i, n in enumerate(inst_nodes)},
        stimulus_to_id=_slot_map(
            cfg.get("stimulus_families") or [], STIMULUS_BASE, N_STIMULUS_SLOTS, "stimulus"
        ),
        behavior_to_id=_slot_map(
            cfg.get("behavior_heads") or [], BEHAVIOR_BASE, N_BEHAVIOR_SLOTS, "behavior"
        ),
        version=str(cfg.get("version", "tokenizer_v1")),
    )
    vocab.validate()
    return vocab


def node_to_token(bundle: GraphBundle, vocab: Vocab) -> np.ndarray:
    """Per-node token ids (u32, length ``bundle.n_nodes``).

    Token rule: instance token when the node is in ``instance_to_id`` and
    carries a type, else its type token, else VOCAB_UNK (T-TOK-3). No node
    is ever dropped - the output covers the full node table.
    """
    type_arr = bundle.type_array()
    tokens = np.full(bundle.n_nodes, VOCAB_UNK, dtype=np.uint32)
    keys = np.array(sorted(vocab.type_to_id))
    if keys.size:
        vals = np.array([vocab.type_to_id[t] for t in keys], dtype=np.uint32)
        pos = np.clip(np.searchsorted(keys, type_arr), 0, keys.size - 1)
        found = keys[pos] == type_arr
        tokens[found] = vals[pos[found]]
    if vocab.instance_to_id:
        idx = np.fromiter(vocab.instance_to_id, dtype=np.int64, count=len(vocab.instance_to_id))
        if idx.size and (idx.min() < 0 or idx.max() >= bundle.n_nodes):
            raise ValueError("instance_to_id node index outside the node table")
        vals = np.array([vocab.instance_to_id[int(i)] for i in idx], dtype=np.uint32)
        ok = type_arr[idx] != ""  # instance slots only ever cover typed nodes
        tokens[idx[ok]] = vals[ok]
    return tokens


def events_to_counts(
    events_node: np.ndarray,
    events_step: np.ndarray,
    node_token: np.ndarray,
    vocab: Vocab,
    n_steps: int | None = None,
    *,
    dt: float | None = None,
    t0: float = 0.0,
) -> np.ndarray:
    """Bin an event list into the (T, V) population count matrix (u32).

    ``events_node``/``events_step`` are parallel arrays, one entry per
    spike event. Integer ``events_step`` values are bin indices directly
    (the schema ``step`` convention - for integer-step events the floor
    index is the step itself); floating-point values are event times and
    are binned as ``floor((t - t0)/dt)``, so ``dt`` is required for them
    (T-TOK-4). ``n_steps`` fixes T; events binned outside ``[0, n_steps)``
    fall outside the recorded window and are dropped. Node indices
    outside ``node_token`` are a hard error - fail closed, never guess.
    """
    node_token = np.asarray(node_token)
    ev_node = np.asarray(events_node, dtype=np.int64)
    ev_step = np.asarray(events_step)
    if ev_node.shape != ev_step.shape:
        raise ValueError("events_node and events_step must have identical shapes")
    if np.issubdtype(ev_step.dtype, np.floating):
        if dt is None:
            raise ValueError(
                "float event times need dt for the floor((t - t0)/dt) bin index"
            )
        bins = np.floor((ev_step.astype(np.float64) - t0) / float(dt)).astype(np.int64)
    else:
        bins = ev_step.astype(np.int64)
    if ev_node.size and (ev_node.min() < 0 or ev_node.max() >= node_token.shape[0]):
        raise ValueError("events_node index outside the node table")
    toks = node_token[ev_node].astype(np.int64)
    n_bins = (int(bins.max()) + 1 if bins.size else 0) if n_steps is None else int(n_steps)
    counts = np.zeros((n_bins, vocab.size()), dtype=np.uint32)
    if bins.size:
        keep = (bins >= 0) & (bins < n_bins)
        np.add.at(counts, (bins[keep], toks[keep]), 1)
    return counts


def counts_to_events(counts: np.ndarray, vocab: Vocab) -> list[tuple[int, int, int]]:
    """Invert ``events_to_counts``: [(bin, token, count)] sorted by (bin, token).

    Expanding each tuple into ``count`` copies of (bin, token) reproduces
    the original event multiset exactly (T-TOK-1).
    """
    c = np.asarray(counts)
    if c.ndim != 2:
        raise ValueError("counts must be a (T, V) matrix")
    if c.shape[1] > vocab.size():
        raise ValueError("counts has more token columns than the vocab")
    bins, toks = np.nonzero(c)
    order = np.lexsort((toks, bins))
    return [
        (int(bins[i]), int(toks[i]), int(c[bins[i], toks[i]])) for i in order
    ]


def vocab_manifest(
    bundle: GraphBundle, vocab: Vocab, graph_dir: Path | None = None
) -> dict[str, Any]:
    """Manifest record for T-TOK-5: sizes, ranges, and the graph pin.

    ``graph_dir`` (when it holds ``nodes.parquet``) contributes a sha256
    of the node table the vocab was built from.
    """
    nodes_sha = None
    nodes_path = Path(graph_dir) / "nodes.parquet" if graph_dir is not None else None
    if nodes_path is not None and nodes_path.exists():
        nodes_sha = sha256_file(nodes_path)
    return {
        "artifact": "vocab_manifest",
        "schema": SCHEMA_VERSION,
        "version": vocab.version,
        "created_utc": utcnow(),
        "code": {"git_sha": git_sha(), "generator": "flygpt tokenize.py"},
        "type_count": len(vocab.type_to_id),
        "instance_count": len(vocab.instance_to_id),
        "stimulus_count": len(vocab.stimulus_to_id),
        "behavior_count": len(vocab.behavior_to_id),
        "reserved": _reserved_record(),
        "instance_token_base": TYPE_TOKEN_BASE + len(vocab.type_to_id),
        "size": vocab.size(),
        "size_rule": "TYPE_TOKEN_BASE + n_types + n_instances (last id + 1)",
        "graph": {
            "n_nodes": bundle.n_nodes,
            "cutoff": bundle.cutoff,
            "nodes_sha256": nodes_sha,
        },
    }


def _load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def iter_corpus_batches(
    corpus_dir: Path,
    vocab: Vocab,
    node_token: np.ndarray,
    *,
    T: int = 9,
    batch_trials: int = 8,
    seed: int = 0,
    splits: tuple[str, ...] = ("train",),
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stream model-training batches from corpus shards.

    Each trial's event list is binned to a (n_steps, V) count matrix
    (``events_to_counts``); sliding (T+1)-step windows yield next-step
    pairs: ``x = counts[t : t+T]`` f32, ``y_events = counts[t+1 : t+T+1]
    > 0``, ``y_rates = log1p`` of the same (the continuous rate proxy -
    corpus counts carry no membrane trace). Trials are filtered to the
    recorded ``splits`` so held-out data never leaks into training.
    Window order is shuffled under ``seed`` for SGD; yields
    ``(x (B,T,V) f32, y_events, y_rates)``.
    """
    import polars as pl

    cd = Path(corpus_dir)
    rng = np.random.default_rng(int(seed))

    def _windows():
        """Per trial: (T+1, V) count windows, streamed shard by shard."""
        for tp in sorted(cd.glob("trials-*.parquet")):
            sid = tp.name[len("trials-") : -len(".parquet")]
            ep = cd / f"events-{sid}.parquet"
            if not ep.exists():
                continue
            trials = pl.read_parquet(
                tp, columns=["trial_id", "n_steps", "dt", "split"]
            )
            keep_ids = set(
                trials.filter(pl.col("split").is_in(list(splits)))["trial_id"].to_list()
            )
            n_steps_of = {
                int(r["trial_id"]): int(r["n_steps"])
                for r in trials.iter_rows(named=True)
            }
            events = pl.read_parquet(ep)
            for part in events.partition_by("trial_id", maintain_order=True):
                tid = int(part["trial_id"][0])
                if tid not in keep_ids:
                    continue
                counts = events_to_counts(
                    part["neuron_idx"].to_numpy(), part["step"].to_numpy(),
                    node_token, vocab, n_steps=n_steps_of[tid],
                )
                n_win = counts.shape[0] - T
                for t0 in range(n_win):
                    yield counts[t0 : t0 + T + 1].astype(np.float32)

    # Bounded shuffle: a full global window list is ~V * (T+1) * 4B per
    # trial-step - terabytes at corpus scale. Reservoir-style buffer
    # shuffles locally instead: fill, emit a random slot per incoming
    # window, drain at the end. Statistically biased vs a global shuffle
    # (windows stay near their shard) - the corpus manifest's split
    # hygiene does not depend on emission order.
    buf_size = 2048
    buf: list[np.ndarray] = []
    batch_x, batch_e, batch_r = [], [], []

    def _emit(w):
        nonlocal batch_x, batch_e, batch_r
        batch_x.append(w[:T])
        nxt = w[1 : T + 1]
        batch_e.append((nxt > 0).astype(np.float32))
        batch_r.append(np.log1p(nxt))
        if len(batch_x) == batch_trials:
            bx, be, br = batch_x, batch_e, batch_r
            batch_x, batch_e, batch_r = [], [], []
            return (
                np.stack(bx).astype(np.float32),
                np.stack(be).astype(np.float32),
                np.stack(br).astype(np.float32),
            )
        return None

    for w in _windows():
        if len(buf) < buf_size:
            buf.append(w)
            continue
        j = int(rng.integers(0, buf_size))
        out = _emit(buf[j])
        buf[j] = w
        if out is not None:
            yield out
    while buf:
        j = int(rng.integers(0, len(buf)))
        out = _emit(buf.pop(j))
        if out is not None:
            yield out
    if batch_x:
        yield (
            np.stack(batch_x).astype(np.float32),
            np.stack(batch_e).astype(np.float32),
            np.stack(batch_r).astype(np.float32),
        )


def write_corpus_batches(
    corpus_dir: Path,
    out_dir: Path,
    vocab: Vocab,
    node_token: np.ndarray,
    *,
    T: int = 9,
    batch_trials: int = 8,
    seed: int = 0,
    splits: tuple[str, ...] = ("train",),
) -> list[Path]:
    """Materialize ``iter_corpus_batches`` as ``batch_*.npz`` under
    ``out_dir`` for the model CLI's ``--data`` path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (x, e, r) in enumerate(
        iter_corpus_batches(
            corpus_dir, vocab, node_token, T=T, batch_trials=batch_trials,
            seed=seed, splits=splits,
        )
    ):
        p = out_dir / f"batch_{i:05d}.npz"
        np.savez_compressed(p, x=x, y_events=e, y_rates=r)
        written.append(p)
    return written


def _print_sizes(vocab: Vocab) -> None:
    inst_base = TYPE_TOKEN_BASE + len(vocab.type_to_id)
    print(f"version          {vocab.version}")
    print(f"type tokens      {len(vocab.type_to_id)}  (base {TYPE_TOKEN_BASE})")
    print(f"instance tokens  {len(vocab.instance_to_id)}  (base {inst_base})")
    print(
        f"stimulus slots   {len(vocab.stimulus_to_id)}/{N_STIMULUS_SLOTS} used"
        f"  (base {STIMULUS_BASE})"
    )
    print(
        f"behavior slots   {len(vocab.behavior_to_id)}/{N_BEHAVIOR_SLOTS} used"
        f"  (base {BEHAVIOR_BASE})"
    )
    print(f"vocab size       {vocab.size()}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--graph-dir", type=Path, default=Path("data/graph"))
    ap.add_argument(
        "--cutoff",
        type=int,
        default=5,
        help="CSR cutoff for loading the bundle (vocab uses the node table only)",
    )
    ap.add_argument(
        "--build",
        action="store_true",
        help="build the vocab and write it + its manifest to cfg output.vocab_manifest",
    )
    ap.add_argument("--print", dest="print_vocab", action="store_true", help="print vocab sizes")
    args = ap.parse_args(argv)
    if not (args.build or args.print_vocab):
        ap.error("one of --build / --print is required")

    cfg = _load_config(args.config)
    vocab_path = Path(cfg["output"]["vocab_manifest"])
    manifest_path = vocab_path.with_name(vocab_path.stem + ".manifest.json")

    vocab = None
    if args.build:
        bundle = load_graph_bundle(args.graph_dir, cutoff=args.cutoff)
        vocab = build_vocab(bundle, cfg)
        manifest = vocab_manifest(bundle, vocab, graph_dir=args.graph_dir)
        manifest["vocab_file"] = vocab_path.name
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(f"[wrote] vocab -> {vocab_path}", flush=True)
        print(f"[wrote] manifest -> {manifest_path}", flush=True)

    if args.print_vocab:
        if vocab is None:
            if vocab_path.exists():
                vocab = Vocab.load(vocab_path)
            else:
                bundle = load_graph_bundle(args.graph_dir, cutoff=args.cutoff)
                vocab = build_vocab(bundle, cfg)
        _print_sizes(vocab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
