"""T-TOK test battery - tokenizer integrity (the silent-corruption risk).

Runs on the synthetic 6-node fixture only (CI-safe). The synthetic node
table maps bodies 10..15 -> node_idx 0..5 with types
[A_R, B_L, A_R, None, C_M, D_L] - node 3 is untyped and must route to
VOCAB_UNK; distinct manifest types are A_R, B_L, C_M, D_L.

Covers T-TOK-1..5 (round-trip exactness, no token collisions, UNK
routing, bin-edge floor convention, vocab-vs-manifest size) plus
instance-circuit designation, reserved slot ranges, determinism across
build/save/load, and the CLI. Plan refs: docs/plan.md T-TOK-1..5
(lines ~185-193), EVAL.md T-TOK table.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from itertools import combinations

import numpy as np
import polars as pl
import pytest
import yaml

from flygpt.schema import (
    N_BEHAVIOR_SLOTS,
    N_STIMULUS_SLOTS,
    TYPE_TOKEN_BASE,
    VOCAB_RESERVED_BASE,
    VOCAB_UNK,
)
from flygpt.tokenize import (
    Vocab,
    build_vocab,
    counts_to_events,
    events_to_counts,
    main,
    node_to_token,
    vocab_manifest,
)

# Layout contract pinned literally: reserved block ends at TYPE_TOKEN_BASE.
TYPE_IDS = {"A_R": 100, "B_L": 101, "C_M": 102, "D_L": 103}


def _cfg(**over):
    cfg = {
        "version": "tokenizer_test",
        "instance_circuits": {"superclasses": [], "types": []},
        "stimulus_families": [],
        "behavior_heads": [],
    }
    cfg.update(over)
    return cfg


def _designate(bundle, updates: dict[int, str]):
    """Bundle copy with per-node ``superclass`` overrides applied."""
    sc = bundle.nodes["superclass"].to_list()
    for node, sup in updates.items():
        sc[node] = sup
    nodes = bundle.nodes.with_columns(pl.Series("superclass", sc))
    return replace(bundle, nodes=nodes)


def _all_ids(vocab: Vocab) -> list[int]:
    return (
        list(vocab.type_to_id.values())
        + list(vocab.instance_to_id.values())
        + list(vocab.stimulus_to_id.values())
        + list(vocab.behavior_to_id.values())
    )


# ---------------------------------------------------------------------------
# Vocab layout + per-node tokens
# ---------------------------------------------------------------------------


def test_reserved_layout_constants():
    """The reserved block ends exactly at TYPE_TOKEN_BASE (schema contract)."""
    assert VOCAB_RESERVED_BASE + N_STIMULUS_SLOTS + N_BEHAVIOR_SLOTS == TYPE_TOKEN_BASE == 100


def test_vocab_layout_and_node_tokens(synthetic_bundle):
    vocab = build_vocab(synthetic_bundle, _cfg())
    assert vocab.type_to_id == TYPE_IDS  # sorted manifest types -> base+i
    assert vocab.instance_to_id == {}
    assert vocab.size() == 104
    tokens = node_to_token(synthetic_bundle, vocab)
    assert tokens.dtype == np.uint32
    assert tokens.tolist() == [100, 101, 100, VOCAB_UNK, 102, 103]


def test_instance_circuit_superclass_designation(synthetic_bundle):
    """A node whose superclass is designated gets an instance token, not
    its type token; the instance block sits right after the type tokens."""
    bundle = _designate(synthetic_bundle, {3: "vnc_motor", 4: "vnc_motor", 5: "vnc_motor"})
    vocab = build_vocab(
        bundle, _cfg(instance_circuits={"superclasses": ["vnc_motor"], "types": []})
    )
    # nodes 4,5 typed+designated -> instance ids 104,105; node 3 untyped -> UNK
    assert vocab.instance_to_id == {4: 104, 5: 105}
    tokens = node_to_token(bundle, vocab)
    assert tokens.tolist() == [100, 101, 100, VOCAB_UNK, 104, 105]
    assert tokens[4] != vocab.type_to_id["C_M"]
    assert vocab.size() == 106


def test_instance_circuit_types_designation(synthetic_bundle):
    """Designation via the ``types`` list hits the same per-node rule."""
    vocab = build_vocab(
        synthetic_bundle,
        _cfg(instance_circuits={"superclasses": [], "types": ["C_M", "D_L"]}),
    )
    assert vocab.instance_to_id == {4: 104, 5: 105}
    tokens = node_to_token(synthetic_bundle, vocab)
    assert tokens[4] == 104 and tokens[5] == 105


# ---------------------------------------------------------------------------
# T-TOK-1: events -> counts -> events round-trip is exact
# ---------------------------------------------------------------------------


def test_t_tok_1_round_trip_exact(synthetic_bundle):
    vocab = build_vocab(synthetic_bundle, _cfg())
    tok = node_to_token(synthetic_bundle, vocab)
    ev_node = np.array([0, 1, 2, 3, 4, 5, 0, 0, 1], dtype=np.uint32)
    ev_step = np.array([0, 1, 1, 2, 0, 3, 0, 0, 1], dtype=np.uint32)

    counts = events_to_counts(ev_node, ev_step, tok, vocab)
    assert counts.dtype == np.uint32
    assert counts.shape == (4, vocab.size())
    assert int(counts.sum()) == ev_node.size

    back = counts_to_events(counts, vocab)
    got: Counter = Counter()
    for b, t, c in back:
        assert c >= 1
        got[(b, t)] += c
    expected = Counter(zip(ev_step.tolist(), tok[ev_node].tolist(), strict=True))
    assert got == expected  # (bin, token) multiset identical - event identity + bin exact

    # re-materializing the matrix from the event list is also exact
    rebuilt = np.zeros_like(counts)
    for b, t, c in back:
        rebuilt[b, t] = c
    assert np.array_equal(counts, rebuilt)


def test_t_tok_1_round_trip_through_instances(synthetic_bundle):
    """Instance-token events round-trip on their instance ids, distinct
    from same-type non-instance events."""
    bundle = _designate(synthetic_bundle, {4: "vnc_motor"})
    vocab = build_vocab(bundle, _cfg(instance_circuits={"superclasses": ["vnc_motor"], "types": []}))
    tok = node_to_token(bundle, vocab)
    ev_node = np.array([4, 0, 2], dtype=np.uint32)  # 4 instance, 0/2 share type A_R
    ev_step = np.array([0, 0, 0], dtype=np.uint32)
    counts = events_to_counts(ev_node, ev_step, tok, vocab)
    assert counts[0, tok[4]] == 1
    assert counts[0, vocab.type_to_id["A_R"]] == 2
    back = counts_to_events(counts, vocab)
    assert back == [(0, vocab.type_to_id["A_R"], 2), (0, tok[4], 1)]


def test_events_out_of_range_and_windowing(synthetic_bundle):
    vocab = build_vocab(synthetic_bundle, _cfg())
    tok = node_to_token(synthetic_bundle, vocab)
    with pytest.raises(ValueError, match="outside the node table"):
        events_to_counts(np.array([99]), np.array([0]), tok, vocab)
    with pytest.raises(ValueError, match="identical shapes"):
        events_to_counts(np.array([0, 1]), np.array([0]), tok, vocab)
    # n_steps windows the matrix; later events are outside the window
    counts = events_to_counts(
        np.array([0, 0, 0]), np.array([0, 1, 5]), tok, vocab, n_steps=2
    )
    assert counts.shape == (2, vocab.size())
    assert int(counts.sum()) == 2


# ---------------------------------------------------------------------------
# T-TOK-2: no collisions across the hybrid ranges
# ---------------------------------------------------------------------------


def test_t_tok_2_no_collisions_disjoint_ranges(synthetic_bundle):
    bundle = _designate(synthetic_bundle, {4: "vnc_motor"})
    vocab = build_vocab(
        bundle,
        _cfg(
            instance_circuits={"superclasses": ["vnc_motor"], "types": ["D_L"]},
            stimulus_families=["olfactory_pulse", "gustatory", "babble"],
            behavior_heads=["song", "walk"],
        ),
    )
    assert vocab.instance_to_id == {4: 104, 5: 105}  # superclass + type hits
    ranges = {
        "type": (set(vocab.type_to_id.values()), 100, 104),
        "instance": (set(vocab.instance_to_id.values()), 104, 106),
        "stimulus": (set(vocab.stimulus_to_id.values()), VOCAB_RESERVED_BASE, 68),
        "behavior": (set(vocab.behavior_to_id.values()), 68, 100),
    }
    for ids, lo, hi in ranges.values():
        assert all(lo <= i < hi for i in ids)
    for (a, *_), (b, *_) in combinations(ranges.values(), 2):
        assert a.isdisjoint(b)
    assert len(_all_ids(vocab)) == len(set(_all_ids(vocab)))
    vocab.validate()  # must not raise


def test_t_tok_2_full_reserved_blocks(synthetic_bundle):
    """With every reserved slot assigned, every slot still owns a unique
    id; the only un-named ids are PAD/UNK/SEX_M/SEX_F."""
    bundle = _designate(synthetic_bundle, dict.fromkeys((0, 1, 2, 4, 5), "vnc_motor"))
    vocab = build_vocab(
        bundle,
        _cfg(
            instance_circuits={"superclasses": ["vnc_motor"], "types": []},
            stimulus_families=[f"fam{i}" for i in range(N_STIMULUS_SLOTS)],
            behavior_heads=[f"head{i}" for i in range(N_BEHAVIOR_SLOTS)],
        ),
    )
    ids = _all_ids(vocab)
    assert len(ids) == len(set(ids))
    assert vocab.size() == TYPE_TOKEN_BASE + 4 + 5
    # named tokens = size minus the 4 un-named reserved ids (PAD, UNK, sex)
    assert len(set(ids)) == vocab.size() - 4
    vocab.validate()


# ---------------------------------------------------------------------------
# T-TOK-3: untyped -> UNK, never dropped, never merged
# ---------------------------------------------------------------------------


def test_t_tok_3_untyped_routes_to_unk(synthetic_bundle):
    vocab = build_vocab(synthetic_bundle, _cfg())
    assert "" not in vocab.type_to_id
    assert "None" not in vocab.type_to_id
    tokens = node_to_token(synthetic_bundle, vocab)
    assert tokens.shape[0] == synthetic_bundle.n_nodes  # nothing dropped
    assert int(tokens[3]) == VOCAB_UNK
    # an event on the untyped node lands at the UNK column, still counted
    counts = events_to_counts(np.array([3]), np.array([0]), tokens, vocab)
    assert int(counts[0, VOCAB_UNK]) == 1


def test_t_tok_3_designated_untyped_stays_unk(synthetic_bundle):
    """Instance designation cannot pull an untyped node out of UNK."""
    bundle = _designate(synthetic_bundle, {3: "vnc_motor"})
    vocab = build_vocab(
        bundle, _cfg(instance_circuits={"superclasses": ["vnc_motor"], "types": []})
    )
    assert vocab.instance_to_id == {}
    assert int(node_to_token(bundle, vocab)[3]) == VOCAB_UNK


# ---------------------------------------------------------------------------
# T-TOK-4: bin index = floor((t - t0)/dt)
# ---------------------------------------------------------------------------


def test_t_tok_4_integer_steps_are_bins(synthetic_bundle):
    vocab = build_vocab(synthetic_bundle, _cfg())
    tok = node_to_token(synthetic_bundle, vocab)
    steps = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32)
    counts = events_to_counts(np.zeros(6, dtype=np.uint32), steps, tok, vocab)
    assert counts.shape[0] == 6
    for s in steps:
        assert int(counts[s, tok[0]]) == 1
        assert int(counts[s].sum()) == 1


def test_t_tok_4_float_times_floor_edges(synthetic_bundle):
    """Boundary events land forward in the floor bin; just-below values
    land in the previous bin. 0.02/0.04/0.08 are exact power-of-two
    multiples of dt=0.02, so the floor is exercised on the boundary
    itself rather than on representation error."""
    vocab = build_vocab(synthetic_bundle, _cfg())
    tok = node_to_token(synthetic_bundle, vocab)
    dt = 0.020
    node = np.zeros(8, dtype=np.uint32)
    times = np.array(
        [0.0, 0.02, 0.02 - 1e-6, 0.04, 0.04 - 1e-9, 0.08, 0.04 + 1e-9, 0.02 + 1e-9]
    )
    expect_bins = [0, 1, 0, 2, 1, 4, 2, 1]
    counts = events_to_counts(node, times, tok, vocab, dt=dt)
    per_bin = counts[:, tok[0]].astype(np.int64)
    assert per_bin.tolist() == np.bincount(expect_bins, minlength=per_bin.size).tolist()
    for i, b in enumerate(expect_bins):
        # each event sits exactly where floor((t-t0)/dt) puts it
        assert int(np.floor(times[i] / dt)) == b


def test_t_tok_4_t0_offset_and_dt_required(synthetic_bundle):
    vocab = build_vocab(synthetic_bundle, _cfg())
    tok = node_to_token(synthetic_bundle, vocab)
    node = np.array([0, 0, 0], dtype=np.uint32)
    counts = events_to_counts(
        node, np.array([1.0, 1.019999, 1.02]), tok, vocab, dt=0.02, t0=1.0
    )
    col = counts[:, tok[0]]
    assert col.tolist() == [2, 1]
    with pytest.raises(ValueError, match="need dt"):
        events_to_counts(node, np.array([0.5, 1.0, 1.5]), tok, vocab)


# ---------------------------------------------------------------------------
# T-TOK-5: vocab size == types + reserved + instances; manifest records it
# ---------------------------------------------------------------------------


def test_t_tok_5_size_and_manifest(synthetic_bundle):
    bundle = _designate(synthetic_bundle, {4: "vnc_motor", 5: "vnc_motor"})
    vocab = build_vocab(
        bundle,
        _cfg(
            instance_circuits={"superclasses": ["vnc_motor"], "types": []},
            stimulus_families=["babble", "olfactory_pulse"],
            behavior_heads=["song"],
        ),
    )
    n_types, n_inst = len(vocab.type_to_id), len(vocab.instance_to_id)
    reserved = VOCAB_RESERVED_BASE + N_STIMULUS_SLOTS + N_BEHAVIOR_SLOTS
    assert vocab.size() == reserved + n_types + n_inst == 106

    man = vocab_manifest(bundle, vocab, graph_dir=None)
    assert man["size"] == vocab.size()
    assert man["type_count"] == n_types
    assert man["instance_count"] == n_inst
    assert man["stimulus_count"] == 2 and man["behavior_count"] == 1
    assert man["reserved"]["stimulus_slots"] == N_STIMULUS_SLOTS
    assert man["reserved"]["behavior_slots"] == N_BEHAVIOR_SLOTS
    assert man["reserved"]["type_token_base"] == TYPE_TOKEN_BASE
    assert man["instance_token_base"] == TYPE_TOKEN_BASE + n_types
    assert man["graph"]["n_nodes"] == synthetic_bundle.n_nodes


# ---------------------------------------------------------------------------
# Determinism + serialization
# ---------------------------------------------------------------------------


def _assert_same_vocab(a: Vocab, b: Vocab) -> None:
    assert a.version == b.version
    assert a.type_to_id == b.type_to_id
    assert a.instance_to_id == b.instance_to_id
    assert a.stimulus_to_id == b.stimulus_to_id
    assert a.behavior_to_id == b.behavior_to_id
    assert a.size() == b.size()


def test_determinism_repeated_builds(synthetic_bundle):
    cfg = _cfg(
        instance_circuits={"superclasses": ["cb_intrinsic"], "types": []},
        stimulus_families=["b", "a", "c"],
        behavior_heads=["walk", "song"],
    )
    v1 = build_vocab(synthetic_bundle, cfg)
    v2 = build_vocab(synthetic_bundle, cfg)
    _assert_same_vocab(v1, v2)
    # input list order does not matter - assignment is sorted
    cfg_shuffled = _cfg(
        instance_circuits={"superclasses": ["cb_intrinsic"], "types": []},
        stimulus_families=["c", "a", "b"],
        behavior_heads=["song", "walk"],
    )
    _assert_same_vocab(v1, build_vocab(synthetic_bundle, cfg_shuffled))


def test_save_load_round_trip_and_byte_determinism(synthetic_bundle, tmp_path):
    bundle = _designate(synthetic_bundle, {4: "vnc_motor"})
    vocab = build_vocab(
        bundle,
        _cfg(
            instance_circuits={"superclasses": ["vnc_motor"], "types": []},
            stimulus_families=["babble"],
            behavior_heads=["song"],
        ),
    )
    p1, p2 = tmp_path / "v1.json", tmp_path / "v2.json"
    vocab.save(p1)
    vocab.save(p2)
    assert p1.read_bytes() == p2.read_bytes()  # deterministic serialization
    loaded = Vocab.load(p1)
    _assert_same_vocab(vocab, loaded)
    # JSON round-trip keeps int node keys in instance_to_id
    assert all(isinstance(k, int) for k in loaded.instance_to_id)
    reloaded_path = tmp_path / "v3.json"
    loaded.save(reloaded_path)
    assert reloaded_path.read_bytes() == p1.read_bytes()


def test_load_rejects_reserved_drift(synthetic_bundle, tmp_path):
    vocab = build_vocab(synthetic_bundle, _cfg())
    path = tmp_path / "v.json"
    vocab.save(path)
    data = json.loads(path.read_text())
    data["reserved"]["stimulus_slots"] = 8  # simulate a schema change
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="reserved block"):
        Vocab.load(path)


# ---------------------------------------------------------------------------
# Reserved stimulus/behavior slots
# ---------------------------------------------------------------------------


def test_stimulus_behavior_slots_reserved_ranges(synthetic_bundle):
    vocab = build_vocab(
        synthetic_bundle,
        _cfg(stimulus_families=["zeta", "alpha"], behavior_heads=["song"]),
    )
    assert vocab.stimulus_to_id == {"alpha": VOCAB_RESERVED_BASE, "zeta": VOCAB_RESERVED_BASE + 1}
    assert vocab.behavior_to_id == {"song": VOCAB_RESERVED_BASE + N_STIMULUS_SLOTS}
    # registers extend deterministically within the declared ranges
    assert vocab.register_stimulus("beta") == VOCAB_RESERVED_BASE + 2
    assert vocab.register_stimulus("beta") == VOCAB_RESERVED_BASE + 2  # idempotent
    assert vocab.register_behavior("walk") == VOCAB_RESERVED_BASE + N_STIMULUS_SLOTS + 1
    assert vocab.size() == 104  # reserved slot use does not change size()


def test_reserved_slot_overflow_fails(synthetic_bundle):
    with pytest.raises(ValueError, match="stimulus"):
        build_vocab(
            synthetic_bundle,
            _cfg(stimulus_families=[f"f{i}" for i in range(N_STIMULUS_SLOTS + 1)]),
        )
    vocab = build_vocab(
        synthetic_bundle, _cfg(behavior_heads=[f"h{i}" for i in range(N_BEHAVIOR_SLOTS)])
    )
    with pytest.raises(ValueError, match="behavior slots exhausted"):
        vocab.register_behavior("one_more")


# ---------------------------------------------------------------------------
# CLI end-to-end on the synthetic build
# ---------------------------------------------------------------------------


def test_cli_build_then_print(synthetic_build, tmp_path, capsys):
    graph_dir = synthetic_build["root"] / "graph"
    vocab_path = tmp_path / "vocab" / "vocab_test.json"
    cfg_path = tmp_path / "tok.yaml"
    cfg_path.write_text(
        yaml.dump(
            {
                "version": "tokenizer_test",
                "instance_circuits": {"superclasses": ["cb_intrinsic"], "types": []},
                "binning": {"convention": "floor", "quantization_eps": 0.0},
                "output": {"vocab_manifest": str(vocab_path)},
            }
        )
    )
    rc = main(["--config", str(cfg_path), "--graph-dir", str(graph_dir), "--build"])
    assert rc == 0
    vocab = Vocab.load(vocab_path)
    assert vocab.type_to_id == TYPE_IDS
    # cb_intrinsic covers all 5 typed nodes -> 5 instance tokens
    assert len(vocab.instance_to_id) == 5
    assert vocab.size() == 109
    manifest = json.loads(
        (tmp_path / "vocab" / "vocab_test.manifest.json").read_text()
    )
    assert manifest["size"] == 109
    assert manifest["type_count"] == 4
    assert manifest["instance_count"] == 5
    assert manifest["graph"]["nodes_sha256"]
    assert manifest["vocab_file"] == vocab_path.name

    rc = main(["--config", str(cfg_path), "--graph-dir", str(graph_dir), "--print"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vocab size" in out and "109" in out
