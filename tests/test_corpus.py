"""T-CORPUS test battery - corpus integrity (docs/plan.md T-CORPUS-1..7).

Runs the real ``flygpt.corpus`` runner on ``synthetic_bundle`` with an
injected deterministic ``simulate_fn`` whose event lists are
hand-computable, so every expectation below is derived rather than
sampled. No sim backend or stimuli module required (both are injected).

Plan refs: docs/plan.md T-CORPUS-1..7; EVAL.md sections 1.4/1.8/3 +
T-SIM-6 (degenerate rule).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from flygpt import corpus
from flygpt.corpus import (
    FAMILY_MODALITY,
    REGION_UNASSIGNED,
    TrialSpec,
    build_splits,
    plan_corpus,
    run_trials,
    splits_from_manifest,
    validate_shard,
)
from flygpt.schema import (
    EVENT_COLUMNS,
    TRIAL_COLUMNS,
    bundle_from_edges,
    canonical_json,
    git_sha,
    seed_from_config,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[1]
CORPUS_CFG = yaml.safe_load((REPO / "configs" / "corpus_v1.yaml").read_text())
LIBRARY_CFG = yaml.safe_load((REPO / "configs" / "stim_library_v1.yaml").read_text())

MAIN_DT = 0.020
COURTSHIP_DT = 0.005


# ---------------------------------------------------------------------------
# Fake deterministic backend (hand-computable event lists)
# ---------------------------------------------------------------------------


@dataclass
class _FakeResult:
    """TrialResult-shaped return value (mirrors the sim_backend contract)."""

    events_step: np.ndarray
    events_node: np.ndarray
    n_steps: int
    n_nodes: int
    rates: np.ndarray
    config: dict
    seed: int
    program: dict
    meta: dict


def _fake_simulate(bundle, program, cfg):
    """Deterministic stand-in for sim_backend.simulate.

    Program fields used:
      drives[0].nodes   neurons that fire (``events_per_node`` evenly
                        spaced events each, default 20)
      mode              "silent" -> 0 events; "burst" -> all events in the
                        first half (stationarity r = 0)
      duration_s        trial length; n_steps = duration_s / cfg["dt"]
    """
    seed = seed_from_config({"program": program, "cfg": cfg})
    dt = float(cfg["dt"])
    n_steps = int(round(float(program.get("duration_s", 1.0)) / dt))
    mode = program.get("mode", "healthy")
    nodes = np.asarray(
        program.get("drives", [{}])[0].get("nodes", []), dtype=np.int64
    )
    if mode == "silent" or nodes.size == 0:
        step = np.empty(0, np.uint32)
        node = np.empty(0, np.uint32)
    else:
        per = int(program.get("events_per_node", 20))
        hi = max(n_steps // 2 - 1, 0) if mode == "burst" else n_steps - 1
        steps = np.unique(np.linspace(0, hi, per).astype(np.int64))
        node = np.repeat(nodes, steps.size)
        step = np.tile(steps, nodes.size)
        order = np.lexsort((node, step))  # (step, node) sort per contract
        step = step[order].astype(np.uint32)
        node = node[order].astype(np.uint32)
    dur = max(n_steps * dt, 1e-9)
    counts = np.bincount(node.astype(np.int64), minlength=bundle.n_nodes)
    return _FakeResult(
        events_step=step,
        events_node=node,
        n_steps=n_steps,
        n_nodes=bundle.n_nodes,
        rates=(counts / dur).astype(np.float32),
        config=dict(cfg),
        seed=seed,
        program=program,
        meta={"backend": "fake"},
    )


def _flaky_simulate(fail_once_for=frozenset(), always_fail_for=frozenset()):
    """Fake sim wrapper that raises on chosen program_ids; counts calls."""
    calls: dict[str, int] = {}

    def fn(bundle, program, cfg):
        pid = str(program.get("program_id"))
        calls[pid] = calls.get(pid, 0) + 1
        if pid in always_fail_for or (pid in fail_once_for and calls[pid] == 1):
            raise RuntimeError(f"injected failure for {pid}")
        return _fake_simulate(bundle, program, cfg)

    fn.calls = calls
    return fn


def _spec(
    rep=0,
    *,
    slice_name="main",
    split="train",
    dt=MAIN_DT,
    nodes=(0, 2),
    mode=None,
    fam="fake_family",
    pid=None,
    duration=1.0,
):
    program = {
        "family_id": fam,
        "program_id": pid or f"{fam}-p0",
        "drives": [
            {
                "nodes": list(nodes),
                "rate_hz": 100.0,
                "window_s": [0.0, duration],
                "envelope": "poisson",
            }
        ],
        "silences": [],
        "duration_s": duration,
        "rep": rep,
    }
    if mode:
        program["mode"] = mode
    cfg = {
        "dt": dt,
        "weight_mode": "count",
        "cutoff": 5,
        "warmup_steps": 10,
        "backend": "fake",
        "slice": slice_name,
    }
    return TrialSpec(fam, program["program_id"], slice_name, split, program, cfg)


def _shard_paths(manifest_path: Path):
    man = json.loads(manifest_path.read_text())
    sid = man["shard_id"]
    return (
        manifest_path.parent / f"events-{sid}.parquet",
        manifest_path.parent / f"trials-{sid}.parquet",
        manifest_path,
        man,
    )


# ---------------------------------------------------------------------------
# T-CORPUS-1: shard schema
# ---------------------------------------------------------------------------


def test_t_corpus_1_shard_schema(synthetic_bundle, tmp_path):
    specs = [_spec(rep=r) for r in range(5)]
    manifests = run_trials(
        synthetic_bundle,
        specs,
        _fake_simulate,
        tmp_path,
        "t1",
        trials_per_shard=2,
    )
    assert len(manifests) == 3  # ceil(5/2), one (slice, split) group
    for mp in manifests:
        ev_p, tr_p, _, man = _shard_paths(mp)
        res = validate_shard(ev_p, tr_p, mp)
        assert res["ok"], res

        ev = pq.read_table(ev_p)
        assert ev.schema.names == list(EVENT_COLUMNS)
        assert ev.schema.field("trial_id").type == pa.uint64()
        assert ev.schema.field("step").type == pa.uint32()
        assert ev.schema.field("neuron_idx").type == pa.uint32()
        # events sorted by (trial_id, step, neuron_idx)
        order = np.lexsort(
            (
                ev.column("neuron_idx").to_numpy(),
                ev.column("step").to_numpy(),
                ev.column("trial_id").to_numpy(),
            )
        )
        assert np.array_equal(order, np.arange(ev.num_rows))

        tr = pl.read_parquet(tr_p)
        assert tr.columns == list(TRIAL_COLUMNS)
        for key in (
            "schema",
            "corpus_version",
            "shard_id",
            "code",
            "graph",
            "sim_cfg",
            "backend",
            "gpu_class",
            "dt",
            "cutoff",
            "slice",
            "split",
            "trial_ids",
            "n_trials",
            "n_events",
            "source_manifest",
            "created_utc",
            "shard_sha256",
        ):
            assert key in man
        assert man["schema"] == "schema_v1"
        assert man["n_trials"] == tr.height
        assert man["n_events"] == ev.num_rows


# ---------------------------------------------------------------------------
# T-CORPUS-2: provenance
# ---------------------------------------------------------------------------


def test_t_corpus_2_provenance(synthetic_bundle, tmp_path):
    specs = [_spec(rep=r) for r in range(3)]
    manifests = run_trials(
        synthetic_bundle,
        specs,
        _fake_simulate,
        tmp_path,
        "t2",
        trials_per_shard=10,
        corpus_version="corpus_test",
        graph_sha="f" * 64,
        graph_version="graph_test",
        source_manifest="provenance/corpus_test.json",
    )
    ev_p, tr_p, mp, man = _shard_paths(manifests[0])
    assert man["code"]["git_sha"] == git_sha()
    assert man["graph"] == {"version": "graph_test", "sha256": "f" * 64}
    assert man["source_manifest"] == "provenance/corpus_test.json"
    assert man["corpus_version"] == "corpus_test"
    assert man["shard_sha256"] == sha256_file(ev_p)
    # tamper with the recorded hash -> validation must fail closed
    man["shard_sha256"] = "0" * 64
    mp.write_text(json.dumps(man, indent=2) + "\n")
    res = validate_shard(ev_p, tr_p, mp)
    assert res["shard_sha256"] is False
    assert res["ok"] is False


# ---------------------------------------------------------------------------
# T-CORPUS-3: coverage/health machinery recorded per trial
# ---------------------------------------------------------------------------


def test_t_corpus_3_health_and_coverage_fields(synthetic_bundle, tmp_path):
    # node types: 0=A_R, 1=B_L, 2=A_R, 3=None, 4=C_M, 5=D_L (4 manifest types)
    specs = [
        _spec(rep=0, pid="h-ok", nodes=(0, 2)),          # 1 type /4 = 25%
        _spec(rep=1, pid="h-mid", nodes=(0, 2, 4)),      # 2 types = 50%
        _spec(rep=2, pid="h-silent", mode="silent"),     # 0 events
        _spec(rep=3, pid="h-hot", nodes=(0, 1, 2, 4)),   # 3 types = 75% > 60%
        _spec(rep=4, pid="h-burst", mode="burst"),       # stationarity r = 0
    ]
    manifests = run_trials(
        synthetic_bundle, specs, _fake_simulate, tmp_path, "t3", trials_per_shard=10
    )
    _, tr_p, _, _ = _shard_paths(manifests[0])
    rows = {r["program"]: r for r in pl.read_parquet(tr_p).to_dicts()}
    by_pid = {json.loads(p)["program_id"]: r for p, r in rows.items()}

    ok = by_pid["h-ok"]
    assert ok["n_events"] == 40          # 2 nodes x 20 events
    assert ok["n_active_neurons"] == 2
    assert ok["n_active_types"] == 1     # nodes 0 and 2 are both A_R
    assert ok["degenerate"] is False
    assert ok["dt"] == pytest.approx(MAIN_DT)
    assert ok["n_steps"] == 50           # 1.0 s / 20 ms
    assert ok["warmup_steps"] == 10

    mid = by_pid["h-mid"]
    assert mid["n_active_neurons"] == 3
    assert mid["n_active_types"] == 2    # A_R + C_M
    assert mid["degenerate"] is False    # 50% < 60%, stationary

    silent = by_pid["h-silent"]
    assert silent["n_events"] == 0
    assert silent["n_active_neurons"] == 0
    assert silent["degenerate"] is True  # active-type fraction < 0.5%

    hot = by_pid["h-hot"]
    assert hot["n_active_types"] == 3
    assert hot["degenerate"] is True     # 75% sustained > 60%

    burst = by_pid["h-burst"]
    assert burst["n_events"] == 40
    assert burst["degenerate"] is True   # split-half stationarity r < 0.5


# ---------------------------------------------------------------------------
# T-CORPUS-4: split hygiene - regenerable, zero overlap, rule sizes
# ---------------------------------------------------------------------------


def _holdout_bundle():
    """Mini bundle exercising S-TYPE strata + S-REGION compartments.

    Regions (somaNeuromere): OL (optic_lobe), CG + GNG (central_brain),
    T1/A1/T2 (vnc), plus one null-region node. Types span
    region x NT-class strata; nodes 107/111 are untyped.
    """
    nodes = pl.DataFrame(
        {
            "bodyId": list(range(100, 112)),
            "type": ["L1", "L1", "L2", "L3", "C1", "C2", "C3", None, "V1", "V2", "V3", None],
            "superclass": [
                "ol_intrinsic", "ol_intrinsic", "ol_intrinsic",
                "visual_projection",
                "cb_intrinsic", "cb_intrinsic", "cb_intrinsic", "cb_intrinsic",
                "vnc_intrinsic", "vnc_motor", "vnc_intrinsic", None,
            ],
            "somaNeuromere": ["OL", "OL", "OL", "OL", "CG", "CG", "GNG", "CG", "T1", "A1", "T2", None],
            "consensus_nt": [
                "acetylcholine", "acetylcholine", "gaba", "acetylcholine",
                "acetylcholine", "gaba", "glutamate", "acetylcholine",
                "acetylcholine", "gaba", "acetylcholine", None,
            ],
        }
    )
    pre = np.array([100, 104, 108], dtype=np.int64) - 100
    post = np.array([104, 108, 100], dtype=np.int64) - 100
    weight = np.array([5, 5, 5], dtype=np.int32)
    sign = np.array([1, -1, 1], dtype=np.int8)
    return bundle_from_edges(nodes, pre, post, weight, sign)


def test_t_corpus_4_split_hygiene():
    bundle = _holdout_bundle()
    fams = list(LIBRARY_CFG["families"])
    s1 = build_splits(CORPUS_CFG, bundle, fams, LIBRARY_CFG)
    s2 = build_splits(CORPUS_CFG, bundle, fams, LIBRARY_CFG)
    assert s1 == s2  # deterministic across calls

    # T-CORPUS-4 core: regenerate from the manifest alone -> identical
    manifest = {"splits": s1}
    assert splits_from_manifest(manifest, bundle, fams, LIBRARY_CFG) == s1

    # S-STIM: zero overlap, naturalistic always held, >=1 per modality
    held_f = set(s1["stimulus_holdout"]["families"])
    train_f = set(fams) - held_f
    assert held_f and not (held_f & train_f)
    assert "naturalistic" in held_f
    assert "lc_silence" not in held_f  # recorded train-eligible exception
    for mod in ("olfactory", "gustatory", "visual", "mechanosensory"):
        assert any(FAMILY_MODALITY.get(f) == mod for f in held_f), mod
    assert all(FAMILY_MODALITY.get(f) is not None or f == "naturalistic" for f in held_f)

    # S-TYPE: zero overlap; exactly 1 per populated stratum (8 strata x 1)
    held_t = set(s1["celltype_holdout"]["types"])
    all_t = set(bundle.manifest_types)
    assert held_t <= all_t
    assert not (held_t & (all_t - held_t))
    assert len(held_t) == 8
    assert s1["celltype_holdout"]["seed"] == "corpus_v1:celltype_holdout"

    # S-REGION: >=1 region per populated compartment; disjoint from train
    held_r = s1["region_holdout"]["regions"]
    assert set(held_r) == {"optic_lobe", "central_brain", "vnc"}
    for _comp, regs in held_r.items():
        assert len(regs) == CORPUS_CFG["splits"]["region_holdout"]["per_compartment"]
        assert regs and all(isinstance(r, str) for r in regs)
    held_flat = {r for regs in held_r.values() for r in regs}
    all_regions = {"OL", "CG", "GNG", "T1", "A1", "T2", REGION_UNASSIGNED}
    assert held_flat <= all_regions
    assert not (held_flat & (all_regions - held_flat))
    # optic lobe holds exactly the OL region (its only bucket)
    assert held_r["optic_lobe"] == ["OL"]
    # vnc pick comes from the vnc-majority regions
    assert held_r["vnc"][0] in {"T1", "T2", "A1"}
    assert held_r["central_brain"][0] in {"CG", "GNG", REGION_UNASSIGNED}


def test_t_corpus_4_splits_on_synthetic_bundle(synthetic_bundle):
    """Graceful degradation: no somaNeuromere column -> empty region
    holdout; single region x NT-class stratum still holds out >=1 type."""
    fams = list(LIBRARY_CFG["families"])
    s = build_splits(CORPUS_CFG, synthetic_bundle, fams, LIBRARY_CFG)
    assert s["region_holdout"]["regions"] == {
        "optic_lobe": [],
        "central_brain": [],
        "vnc": [],
    }
    held_t = s["celltype_holdout"]["types"]
    assert len(held_t) == 1  # one stratum (unassigned|acetylcholine), 4 types
    assert held_t[0] in {"A_R", "B_L", "C_M", "D_L"}
    assert splits_from_manifest({"splits": s}, synthetic_bundle, fams, LIBRARY_CFG) == s


# ---------------------------------------------------------------------------
# T-CORPUS-5/7: slice + dt integrity
# ---------------------------------------------------------------------------


def test_t_corpus_5_7_slice_dt_integrity(synthetic_bundle, tmp_path):
    specs = [_spec(rep=r, slice_name="main", dt=MAIN_DT) for r in range(4)]
    specs += [
        _spec(rep=r, slice_name="courtship", dt=COURTSHIP_DT, duration=0.1)
        for r in range(4)
    ]
    manifests = run_trials(
        synthetic_bundle, specs, _fake_simulate, tmp_path, "t57", trials_per_shard=10
    )
    assert len(manifests) == 2
    by_slice = {}
    for mp in manifests:
        ev_p, tr_p, _, man = _shard_paths(mp)
        by_slice[man["slice"]] = (man, pl.read_parquet(tr_p))
        res = validate_shard(ev_p, tr_p, mp)
        assert res["ok"], res
    # disjoint slice tags, dt recorded per shard and on every trial row
    assert set(by_slice) == {"main", "courtship"}
    assert by_slice["main"][0]["dt"] == pytest.approx(MAIN_DT)
    assert by_slice["courtship"][0]["dt"] == pytest.approx(COURTSHIP_DT)
    main_dt = by_slice["main"][1]["dt"].unique().to_list()
    court_dt = by_slice["courtship"][1]["dt"].unique().to_list()
    assert main_dt == [pytest.approx(MAIN_DT)]
    assert court_dt == [pytest.approx(COURTSHIP_DT)]
    assert by_slice["main"][1]["slice"].unique().to_list() == ["main"]
    assert by_slice["courtship"][1]["slice"].unique().to_list() == ["courtship"]
    # dt governs step count: 0.1 s at 5 ms -> 20 steps; 1.0 s at 20 ms -> 50
    assert by_slice["courtship"][1]["n_steps"].unique().to_list() == [20]
    assert by_slice["main"][1]["n_steps"].unique().to_list() == [50]


# ---------------------------------------------------------------------------
# T-CORPUS-6: regeneration determinism
# ---------------------------------------------------------------------------


def test_t_corpus_6_regeneration_identical_bytes(synthetic_bundle, tmp_path):
    specs = [_spec(rep=r) for r in range(6)]
    a = run_trials(
        synthetic_bundle, specs, _fake_simulate, tmp_path / "a", "t6",
        trials_per_shard=4, corpus_version="corpus_test", graph_sha="g" * 64,
    )
    b = run_trials(
        synthetic_bundle, specs, _fake_simulate, tmp_path / "b", "t6",
        trials_per_shard=4, corpus_version="corpus_test", graph_sha="g" * 64,
    )
    assert len(a) == len(b) == 2
    for pa_, pb_ in zip(a, b, strict=True):
        ev_a, tr_a, _, man_a = _shard_paths(pa_)
        ev_b, tr_b, _, man_b = _shard_paths(pb_)
        assert ev_a.read_bytes() == ev_b.read_bytes()  # identical event bytes
        assert pq.read_table(ev_a).equals(pq.read_table(ev_b))
        assert man_a["shard_sha256"] == man_b["shard_sha256"]
        assert man_a["trial_ids"] == man_b["trial_ids"]
        assert tr_a.read_bytes() == tr_b.read_bytes()


# ---------------------------------------------------------------------------
# Dead-letter: failure -> one retry -> jsonl row -> shard still completes
# ---------------------------------------------------------------------------


def test_deadletter_and_retry(synthetic_bundle, tmp_path):
    specs = [
        _spec(rep=0, pid="ok-0"),
        _spec(rep=1, pid="bad-1"),
        _spec(rep=2, pid="ok-2"),
    ]
    sim = _flaky_simulate(always_fail_for={"bad-1"})
    dl = tmp_path / "dl.jsonl"
    manifests = run_trials(
        synthetic_bundle, specs, sim, tmp_path, "dl", trials_per_shard=10,
        dead_letter_path=dl,
    )
    assert sim.calls["bad-1"] == 2  # original attempt + exactly one retry
    rows = [json.loads(line) for line in dl.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["spec"]["program_id"] == "bad-1"
    assert "error" in rows[0] and "utc" in rows[0]
    assert rows[0]["attempts"] == 2
    # shard completes with the surviving trial set
    _, tr_p, _, man = _shard_paths(manifests[0])
    tr = pl.read_parquet(tr_p)
    assert tr.height == 2
    assert man["n_trials"] == 2
    pids = {json.loads(p)["program_id"] for p in tr["program"].to_list()}
    assert pids == {"ok-0", "ok-2"}

    # a transient failure is retried and produces no dead-letter row
    sim2 = _flaky_simulate(fail_once_for={"bad-1"})
    dl2 = tmp_path / "b" / "deadletter.jsonl"
    manifests2 = run_trials(
        synthetic_bundle, specs, sim2, tmp_path / "b", "dl", trials_per_shard=10
    )
    assert sim2.calls["bad-1"] == 2
    _, tr2_p, _, man2 = _shard_paths(manifests2[0])
    assert pl.read_parquet(tr2_p).height == 3
    assert man2["n_trials"] == 3
    assert not dl2.exists()


# ---------------------------------------------------------------------------
# Seeds: unique per spec and derived from the canonical program spec
# ---------------------------------------------------------------------------


def test_trial_seeds_unique_and_derived(synthetic_bundle):
    specs = [_spec(rep=r) for r in range(12)]
    specs += [_spec(rep=0, slice_name="courtship", dt=COURTSHIP_DT)]
    seeds = [s.seed for s in specs]
    assert len(set(seeds)) == len(specs)
    for s in specs:
        assert s.seed == seed_from_config(s.program_spec())
        assert s.trial_id == seed_from_config({"trial_id": s.program_spec()})
        assert 0 <= s.seed < 2**64
        assert s.trial_id != s.seed


# ---------------------------------------------------------------------------
# plan_corpus: slices keep dt/weight_mode/cutoff; splits drive split labels
# ---------------------------------------------------------------------------


def _plan_library():
    def gen_a(index, rng, cfg, **kw):
        return {
            "family_id": "fam_a",
            "program_id": f"fam_a-{kw.get('trial', 0)}",
            "drives": [{"nodes": [0, 2], "rate_hz": 100.0}],
            "silences": [],
            "duration_s": 1.0,
        }

    def gen_b(index, rng, cfg, **kw):
        return {
            "family_id": "fam_b",
            "program_id": f"fam_b-{kw.get('trial', 0)}",
            "drives": [{"nodes": [1], "rate_hz": 40.0}],
            "silences": [],
            "duration_s": 0.5,
        }

    return {"fam_a": gen_a, "fam_b": gen_b}


def test_plan_corpus_slices_and_splits(synthetic_bundle):
    cfg = {
        "version": "corpus_test",
        "slices": {
            "main": {"dt": MAIN_DT, "weight_mode": "count", "cutoff": 5, "n_trials": 6},
            "courtship": {"dt": COURTSHIP_DT, "weight_mode": "count", "cutoff": 5, "n_trials": 4},
            "shuffled": {"dt": MAIN_DT, "weight_mode": "count", "cutoff": 5,
                         "n_trials": 2, "weight_shuffle": True},
        },
        "splits": CORPUS_CFG["splits"],
    }
    lib_cfg = {
        "families": {
            "fam_a": {"modality": "olfactory"},   # sole olfactory family -> held out
            "fam_b": {},                           # no modality -> train-only
        }
    }
    index = SimpleNamespace(bundle=synthetic_bundle)
    specs = plan_corpus(
        cfg,
        index,
        library=_plan_library(),
        library_cfg=lib_cfg,
        sim_cfg={"model": {"dt": MAIN_DT, "warmup_steps": 10}, "backend": "fake"},
    )
    # 6 + 4 + 2 trials distributed evenly across the 2 programs
    assert len(specs) == 12
    by_slice = {}
    for s in specs:
        by_slice.setdefault(s.slice, []).append(s)
    assert {k: len(v) for k, v in by_slice.items()} == {
        "main": 6,
        "courtship": 4,
        "shuffled": 2,
    }
    for s in by_slice["main"]:
        assert s.cfg["dt"] == pytest.approx(MAIN_DT)
        assert s.cfg["weight_mode"] == "count"
        assert s.cfg["cutoff"] == 5
        assert s.cfg["weight_shuffle"] is False
        assert s.cfg["warmup_steps"] == 10
    for s in by_slice["courtship"]:
        assert s.cfg["dt"] == pytest.approx(COURTSHIP_DT)
    for s in by_slice["shuffled"]:
        assert s.cfg["weight_shuffle"] is True
    # held-out family -> test; train-only family -> train
    assert {s.split for s in specs if s.family_id == "fam_a"} == {"test"}
    assert {s.split for s in specs if s.family_id == "fam_b"} == {"train"}
    # seeds still unique across the whole plan
    assert len({s.seed for s in specs}) == len(specs)
    # program column is canonical JSON (regenerable program spec)
    assert json.loads(canonical_json(specs[0].program))["rep"] == 0


def test_health_metrics_degenerate_rules(synthetic_bundle):
    """Direct check of the T-SIM-6 rule corners."""
    types = synthetic_bundle.type_array()
    n = synthetic_bundle.n_nodes
    # all-silent: type fraction 0 -> degenerate, stationarity defined as 1
    hm = corpus.health_metrics(
        np.empty(0, np.uint32), np.empty(0, np.uint32), n, 50, types, dt=0.02
    )
    assert hm["degenerate"] is True
    assert hm["stationarity_r"] == 1.0
    assert hm["n_active_neurons"] == 0
    # identical halves on one typed neuron -> stationary, not degenerate
    step = np.array([0, 5, 30, 40], dtype=np.uint32)
    node = np.zeros(4, dtype=np.uint32)
    hm = corpus.health_metrics(step, node, n, 50, types, dt=0.02)
    assert hm["n_active_types"] == 1
    assert hm["type_frac_active"] == pytest.approx(0.25)
    assert hm["degenerate"] is False
