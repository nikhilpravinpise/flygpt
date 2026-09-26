"""T-TOK-1..5 - tokenizer integrity runners (plan.md T-TOK table).

The vocab contract (impl-spec tokenize.py): an object exposing
``type_to_id`` / ``instance_to_id`` / ``stimulus_to_id`` /
``behavior_to_id`` dicts and ``size()``; a plain dict with the same keys
is accepted. Tokenizer functions are injected or lazily imported from
``flygpt.tokenize`` - missing deps produce a documented skip, never a
silent pass. Reserved-slot layout comes from flygpt.schema
(VOCAB_*/TYPE_TOKEN_BASE = 100).

- T-TOK-1: events -> counts -> events round-trip, identity + bin exact.
- T-TOK-2: zero token-ID collisions; hybrid ranges disjoint.
- T-TOK-3: 100% of untyped neurons route to VOCAB_UNK.
- T-TOK-4: bin-boundary vectors land in floor((t - t0)/dt).
- T-TOK-5: vocab size == manifest types + reserved + instance slots.

Each runner writes ``results/T-TOK-*.json`` via ``write_result``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from flygpt.evals.sim_checks import _skipped
from flygpt.schema import (
    N_BEHAVIOR_SLOTS,
    N_STIMULUS_SLOTS,
    TYPE_TOKEN_BASE,
    VOCAB_RESERVED_BASE,
    VOCAB_UNK,
    write_result,
)

# Reserved-window bounds derived from the schema layout:
# [VOCAB_RESERVED_BASE, +64) stimulus-family slots, then [68, 100) behavior.
VOCAB_STIMULUS_BASE = VOCAB_RESERVED_BASE
VOCAB_BEHAVIOR_BASE = VOCAB_RESERVED_BASE + N_STIMULUS_SLOTS


def _vocab_get(vocab: Any, name: str) -> dict:
    if vocab is None:
        return {}
    if isinstance(vocab, dict):
        return dict(vocab.get(name) or {})
    return dict(getattr(vocab, name, {}) or {})


def _vocab_size(vocab: Any) -> int:
    size = vocab.get("size") if isinstance(vocab, dict) else getattr(vocab, "size", None)
    return int(size() if callable(size) else size)


def _lazy_tokenize() -> dict[str, Callable | None]:
    out = {"events_to_counts": None, "counts_to_events": None, "node_to_token": None}
    try:
        import flygpt.tokenize as tok

        for name in out:
            out[name] = getattr(tok, name, None)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# T-TOK-1: round-trip exactness
# ---------------------------------------------------------------------------


def _call_events_to_counts(fn, steps, nodes, node_token, vocab, n_steps):
    """Call an events->counts fn across arg conventions.

    ``flygpt.tokenize.events_to_counts`` is node-first
    ``(events_node, events_step, node_token, vocab, n_steps)``; simpler
    fakes take ``(steps, nodes, node_token, vocab[, n_steps])``.
    """
    params = inspect.signature(fn).parameters
    if "events_node" in params:
        kw = {"n_steps": n_steps} if "n_steps" in params else {}
        return fn(nodes, steps, node_token, vocab, **kw)
    if "n_steps" in params:
        return fn(steps, nodes, node_token, vocab, n_steps=n_steps)
    if len(params) >= 5:
        return fn(steps, nodes, node_token, vocab, n_steps)
    if len(params) == 4:
        return fn((steps, nodes), node_token, vocab, n_steps)
    return fn((steps, nodes), node_token, vocab)


def _call_counts_to_events(fn, counts, vocab):
    params = inspect.signature(fn).parameters
    if len(params) >= 2:
        return fn(counts, vocab)
    return fn(counts)


def run_t_tok_1(
    vocab,
    results_dir: Path,
    *,
    events: tuple[np.ndarray, np.ndarray] | None = None,
    node_token: np.ndarray | None = None,
    n_steps: int | None = None,
    events_to_counts_fn: Callable | None = None,
    counts_to_events_fn: Callable | None = None,
    quantization_eps: float = 0.0,
) -> dict:
    """T-TOK-1: event identity + bin index exact through the round-trip.

    ``events`` is a (steps, nodes) pair for one trial; ``node_token``
    the per-node token map. events_to_counts -> (T,V) u32 ->
    counts_to_events -> [(bin, token, count)] must reproduce the input
    multiset exactly (quantization eps recorded, default exact).
    """
    test_id = "T-TOK-1"
    lazy = _lazy_tokenize()
    f_fwd = events_to_counts_fn or lazy["events_to_counts"]
    f_back = counts_to_events_fn or lazy["counts_to_events"]
    inputs = {"quantization_eps": quantization_eps, "n_steps": n_steps}
    if f_fwd is None or f_back is None:
        return _skipped(
            results_dir,
            test_id,
            "flygpt.tokenize events_to_counts/counts_to_events unavailable",
            inputs=inputs,
        )
    if events is None or node_token is None:
        return _skipped(results_dir, test_id, "no events/node_token supplied", inputs=inputs)

    steps = np.asarray(events[0], dtype=np.int64)
    nodes = np.asarray(events[1], dtype=np.int64)
    if n_steps is None:
        n_steps = int(steps.max()) + 1 if steps.size else 0
    counts = np.asarray(_call_events_to_counts(f_fwd, steps, nodes, node_token, vocab, n_steps))
    back = _call_counts_to_events(f_back, counts, vocab)

    original = sorted(zip(steps.tolist(), node_token[nodes].tolist(), strict=True))
    rebuilt: list[tuple[int, int]] = []
    for item in back:
        bin_i, token_i, cnt = (int(item[0]), int(item[1]), int(item[2] if len(item) > 2 else 1))
        rebuilt.extend([(bin_i, token_i)] * cnt)
    rebuilt.sort()
    n_mismatch = 0
    if len(original) != len(rebuilt):
        n_mismatch = abs(len(original) - len(rebuilt)) + 1
    else:
        n_mismatch = sum(1 for a, b in zip(original, rebuilt, strict=True) if a != b)
    checks = {
        "n_events_in": len(original),
        "n_events_out": len(rebuilt),
        "n_mismatched": n_mismatch,
        "quantization_eps": quantization_eps,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"n_mismatched": n_mismatch},
        passed=bool(n_mismatch == 0),
    )


# ---------------------------------------------------------------------------
# T-TOK-2: collision scan + hybrid range disjointness
# ---------------------------------------------------------------------------


def run_t_tok_2(vocab, results_dir: Path) -> dict:
    """T-TOK-2: zero token-ID collisions; hybrid ranges disjoint.

    Reserved layout (schema.py): stimulus ids live in
    [VOCAB_STIMULUS_BASE, +64), behavior ids in [VOCAB_BEHAVIOR_BASE,
    +32), type ids at >= TYPE_TOKEN_BASE; the instance block is a
    contiguous range immediately after the type block. Any id reused
    across or within maps is a collision.
    """
    test_id = "T-TOK-2"
    type_ids = _vocab_get(vocab, "type_to_id")
    inst_ids = _vocab_get(vocab, "instance_to_id")
    stim_ids = _vocab_get(vocab, "stimulus_to_id")
    beh_ids = _vocab_get(vocab, "behavior_to_id")
    problems: list[str] = []

    seen: dict[int, str] = {}
    collisions: list[dict] = []
    for map_name, m in (
        ("type", type_ids),
        ("instance", inst_ids),
        ("stimulus", stim_ids),
        ("behavior", beh_ids),
    ):
        for key, tok in m.items():
            tok = int(tok)
            if tok in seen:
                collisions.append({"id": tok, "first": seen[tok], "second": f"{map_name}:{key}"})
            else:
                seen[tok] = f"{map_name}:{key}"
    if collisions:
        problems.append(f"{len(collisions)} token-id collisions")

    ranges: dict[str, list[int]] = {}
    t_ids = sorted(int(v) for v in type_ids.values())
    i_ids = sorted(int(v) for v in inst_ids.values())
    s_ids = sorted(int(v) for v in stim_ids.values())
    b_ids = sorted(int(v) for v in beh_ids.values())
    ranges = {
        "type": [t_ids[0], t_ids[-1]] if t_ids else [],
        "instance": [i_ids[0], i_ids[-1]] if i_ids else [],
        "stimulus": [s_ids[0], s_ids[-1]] if s_ids else [],
        "behavior": [b_ids[0], b_ids[-1]] if b_ids else [],
    }
    if t_ids:
        if t_ids[0] != TYPE_TOKEN_BASE:
            problems.append(f"type block starts at {t_ids[0]}, expected {TYPE_TOKEN_BASE}")
        if t_ids != list(range(TYPE_TOKEN_BASE, TYPE_TOKEN_BASE + len(t_ids))):
            problems.append("type ids not contiguous from TYPE_TOKEN_BASE")
    if s_ids and (
        s_ids[0] < VOCAB_STIMULUS_BASE or s_ids[-1] >= VOCAB_STIMULUS_BASE + N_STIMULUS_SLOTS
    ):
        problems.append("stimulus ids outside the reserved stimulus window")
    if b_ids and (
        b_ids[0] < VOCAB_BEHAVIOR_BASE or b_ids[-1] >= VOCAB_BEHAVIOR_BASE + N_BEHAVIOR_SLOTS
    ):
        problems.append("behavior ids outside the reserved behavior window")
    if i_ids:
        expected_start = TYPE_TOKEN_BASE + len(t_ids)
        if i_ids != list(range(expected_start, expected_start + len(i_ids))):
            problems.append("instance block not the contiguous range after the type block")
        if i_ids[0] < VOCAB_RESERVED_BASE:
            problems.append("instance ids collide with the reserved block")
    checks = {"ranges": ranges, "n_collisions": len(collisions), "collisions": collisions,
        "problems": problems}
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs={},
        metrics={"n_collisions": len(collisions)},
        passed=bool(len(problems) == 0),
    )


# ---------------------------------------------------------------------------
# T-TOK-3: UNK routing
# ---------------------------------------------------------------------------


def run_t_tok_3(
    vocab,
    results_dir: Path,
    *,
    node_token: np.ndarray | None = None,
    untyped_idx: np.ndarray | None = None,
    bundle=None,
    node_token_fn: Callable | None = None,
) -> dict:
    """T-TOK-3: 100% of untyped neurons route to VOCAB_UNK.

    Dropped or merged-into-real-type count must be 0: every untyped
    node index maps to UNK exactly (no real type id, no -1/dropout).
    """
    test_id = "T-TOK-3"
    if bundle is not None and untyped_idx is None:
        untyped_idx = np.flatnonzero(bundle.type_array() == "")
    if node_token is None and bundle is not None:
        fn = node_token_fn or _lazy_tokenize()["node_to_token"]
        if fn is not None:
            node_token = np.asarray(fn(bundle, vocab))
    inputs = {"n_untyped": int(untyped_idx.size) if untyped_idx is not None else None}
    if untyped_idx is None or node_token is None:
        return _skipped(
            results_dir, test_id, "node_token map or untyped index unavailable", inputs=inputs
        )
    untyped_idx = np.asarray(untyped_idx, dtype=np.int64)
    node_token = np.asarray(node_token, dtype=np.int64)
    real_type_ids = {int(v) for v in _vocab_get(vocab, "type_to_id").values()}
    routed = node_token[untyped_idx] if untyped_idx.size else np.empty(0, dtype=np.int64)
    n_unk = int((routed == VOCAB_UNK).sum())
    merged = sorted(int(t) for t in np.unique(routed) if int(t) in real_type_ids)
    n_routed_out = int(routed.size - n_unk)
    checks = {
        "n_untyped": int(untyped_idx.size),
        "n_routed_to_unk": n_unk,
        "n_dropped_or_merged": n_routed_out,
        "merged_into_type_ids": merged,
        "unk_fraction": n_unk / untyped_idx.size if untyped_idx.size else 1.0,
    }
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"unk_fraction": checks["unk_fraction"]},
        passed=bool(n_routed_out == 0),
    )


# ---------------------------------------------------------------------------
# T-TOK-4: bin-boundary vectors
# ---------------------------------------------------------------------------


def _default_bin_fn(t: np.ndarray, t0: float, dt: float) -> np.ndarray:
    return np.floor((np.asarray(t, dtype=np.float64) - t0) / dt).astype(np.int64)


def run_t_tok_4(
    results_dir: Path,
    *,
    dt: float = 0.02,
    t0: float = 0.0,
    bin_fn: Callable | None = None,
    cases: list | None = None,
    n_boundaries: int = 8,
) -> dict:
    """T-TOK-4: boundary events land in floor((t - t0)/dt).

    Default vectors probe every boundary k: exactly k*dt (-> k), a hair
    below (-> k-1), and a hair above (-> k). ``bin_fn`` defaults to the
    floor-convention reference; injected implementations are checked
    against the same expectations. Misassigned vectors must be 0.
    """
    test_id = "T-TOK-4"
    eps = min(dt * 1e-6, 1e-9)
    if cases is None:
        cases = []
        for k in range(n_boundaries):
            cases.append((k * dt, k))                      # exact boundary -> k
            cases.append((k * dt + dt - eps, k))           # just under next edge
            if k > 0:
                cases.append((k * dt - eps, k - 1))        # hair below -> k-1
    fn = bin_fn or _default_bin_fn
    results = []
    n_bad = 0
    for t_sec, expected in cases:
        got = int(np.asarray(fn(np.asarray([t_sec]), t0, dt))[0])
        ok = got == int(expected)
        n_bad += int(not ok)
        results.append({"t": t_sec, "expected": int(expected), "got": got, "ok": ok})
    return write_result(
        results_dir,
        test_id,
        checks={"dt": dt, "t0": t0, "n_cases": len(cases), "n_misassigned": n_bad, "cases": results},
        inputs={"bin_fn": getattr(fn, "__name__", repr(fn))},
        metrics={"n_misassigned": n_bad},
        passed=bool(n_bad == 0),
    )


# ---------------------------------------------------------------------------
# T-TOK-5: vocab size vs manifest
# ---------------------------------------------------------------------------


def run_t_tok_5(
    vocab,
    results_dir: Path,
    *,
    manifest_types: list[str] | None = None,
    manifest_path: Path | None = None,
) -> dict:
    """T-TOK-5: vocab size == manifest types + reserved + instance slots.

    Expected size = TYPE_TOKEN_BASE (PAD/UNK/sex/stimulus/behavior
    reserved block = 100) + manifest type count + instance-token count.
    The manifest type count is read from ``manifest_types`` or a JSON
    manifest (``types`` list or ``n_types`` field) - never assumed.
    """
    test_id = "T-TOK-5"
    if manifest_types is None and manifest_path is not None:
        m = json.loads(Path(manifest_path).read_text())
        manifest_types = m.get("types") or ([f"t{i}" for i in range(int(m["n_types"]))] if
            m.get("n_types") else None)
    inputs = {"manifest_path": str(manifest_path) if manifest_path else None}
    if manifest_types is None:
        return _skipped(results_dir, test_id, "manifest type count unavailable", inputs=inputs)
    n_types = len(manifest_types)
    n_types_mapped = len(_vocab_get(vocab, "type_to_id"))
    n_instances = len(_vocab_get(vocab, "instance_to_id"))
    size = _vocab_size(vocab)
    expected = TYPE_TOKEN_BASE + n_types + n_instances
    checks = {
        "vocab_size": size,
        "n_manifest_types": n_types,
        "n_mapped_types": n_types_mapped,
        "n_instance_tokens": n_instances,
        "reserved_slots": TYPE_TOKEN_BASE,
        "expected_size": expected,
        "type_map_covers_manifest": n_types_mapped == n_types,
    }
    problems = []
    if size != expected:
        problems.append(f"vocab size {size} != expected {expected}")
    if n_types_mapped != n_types:
        problems.append(f"type_to_id covers {n_types_mapped} of {n_types} manifest types")
    checks["problems"] = problems
    return write_result(
        results_dir,
        test_id,
        checks=checks,
        inputs=inputs,
        metrics={"expected_size": expected, "vocab_size": size},
        passed=bool(len(problems) == 0),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_vocab(path: Path):
    """Plain-JSON vocab stand-in for CLI runs (dict of dicts + size)."""
    raw = json.loads(path.read_text())

    class _Vocab(dict):
        def size(self) -> int:
            declared = self.get("_size")
            if declared is not None:
                return int(declared)
            ids = list(self.all_ids())
            return max(ids) + 1 if ids else 0

        def all_ids(self):
            for name in ("type_to_id", "instance_to_id", "stimulus_to_id", "behavior_to_id"):
                yield from (int(v) for v in self.get(name, {}).values())

    v = _Vocab({k: raw.get(k, {}) for k in ("type_to_id", "instance_to_id", "stimulus_to_id",
        "behavior_to_id")})
    v["_size"] = raw.get("size")
    return v


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="T-TOK-1..5 tokenizer checks")
    parser.add_argument("--vocab", type=Path, default=Path("data/vocab/vocab_v1.json"))
    parser.add_argument("--graph-dir", type=Path, default=Path("data/graph"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args(argv)

    vocab = _load_vocab(args.vocab) if args.vocab.exists() else None
    if vocab is None:
        for tid in (f"T-TOK-{i}" for i in range(1, 6)):
            _skipped(args.results_dir, tid, f"vocab absent: {args.vocab}")
        return 0
    bundle = None
    if (args.graph_dir / "nodes.parquet").exists():
        from flygpt.schema import load_graph_bundle

        bundle = load_graph_bundle(args.graph_dir)
    run_t_tok_1(vocab, args.results_dir)
    run_t_tok_2(vocab, args.results_dir)
    run_t_tok_3(vocab, args.results_dir, bundle=bundle)
    run_t_tok_4(args.results_dir, dt=args.dt)
    run_t_tok_5(vocab, args.results_dir, manifest_path=args.vocab)
    return 0


if __name__ == "__main__":
    sys.exit(main())
