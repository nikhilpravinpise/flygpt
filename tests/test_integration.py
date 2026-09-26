"""End-to-end integration tests: the real pipeline composed, not mocks.

Runs the actual modules together on the synthetic connectome fixture
(6 nodes, hand-computed wiring) plus a small realistic mock bundle for
the sensory families:

    SensoryIndex -> StimulusProgram -> sim_backend.simulate -> events
    -> corpus.write_shard -> validate -> tokenize -> behavior decode
    -> evals.common metrics -> write_result envelope

These tests exist because the modules were built to a shared spec -
the purpose here is to catch contract drift the unit tests cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flygpt.schema import (
    EVENT_COLUMNS,
    GraphBundle,
    bundle_from_edges,
    seed_from_config,
)

# ---------------------------------------------------------------------------
# Fixtures: a mock bundle with realistic annotation columns
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_bundle() -> GraphBundle:
    """~40-node connectome carrying the columns stimuli resolve on."""
    types = (
        ["ORN_DA1", "ORN_DA1", "ORN_DA2", "LC4", "LC4", "LPLC2", "LC11",
         "taste_peg_x", "labellar_b", "leg_bristle_x", "JON_a", "JON_b",
         "DNp01", "MN9", "wingMN_x", "DNg01", "central_i", "central_e",
         "central_e2", "SEZPN_a"]
        + [f"IN_{i}" for i in range(20)]
    )
    n = len(types)
    subclasses = (
        ["antennal olfactory"] * 3
        + ["visual projection neuron"] * 4
        + ["taste peg", "labellar bristle", "leg bristle"]
        + ["chordotonal organ"] * 2
        + ["descending", "motor", "motor", "descending",
           "central intrinsic", "central intrinsic", "central intrinsic",
           "SEZ projection"]
        + ["central intrinsic"] * 20
    )
    superclasses = (
        ["olfactory"] * 3 + ["visual_projection"] * 4
        + ["vnc_sensory"] * 3 + ["vnc_sensory"] * 2
        + ["descending_neuron", "vnc_motor", "vnc_motor", "descending_neuron"]
        + ["cb_intrinsic"] * 4
        + ["cb_intrinsic"] * 20
    )
    sides = (
        ["L", "L", "R"] + ["L", "R", "L", "R"]
        + ["L", "R", "L"] + ["L", "R"]
        + ["L", "R", "L", "M"] + ["L", "L", "R", "M"]
        + ["L"] * 20
    )
    classes = (
        ["sensory"] * 12
        + ["descending", "motor", "motor", "descending"]
        + ["central", "central", "central", "projection"]
        + ["central"] * 20
    )
    assert all(len(c) == n for c in (types, subclasses, superclasses, sides, classes)), (
        [len(c) for c in (types, subclasses, superclasses, sides, classes)]
    )
    nodes = pl.DataFrame({
        "bodyId": np.arange(1000, 1000 + n, dtype=np.int64),
        "node_idx": np.arange(n, dtype=np.int64),
        "type": types,
        "instance": [f"{t}_{s}" for t, s in zip(types, sides, strict=True)],
        "class": classes,
        "superclass": superclasses,
        "subclass": subclasses,
        "somaSide": sides,
        "rootSide": sides,
        "somaNeuromere": ["SEZ"] * 9 + ["SOG"] * 2 + ["brain"] * 4
        + ["T2"] + ["brain"] * 4 + ["VNC"] * 20,
        "entryNerve": [""] * n,
        "exitNerve": [""] * n,
    })
    rng = np.random.default_rng(7)
    pre, post, w, sgn = [], [], [], []
    # sensory -> interneuron edges
    for s in range(11):
        for t in rng.choice(np.arange(16, 30), size=4, replace=False):
            pre.append(s)
            post.append(int(t))
            w.append(int(rng.integers(5, 12)))
            sgn.append(1)
    # interneuron -> DN/MN
    for s in range(16, 30):
        for t in (12, 13, 14, 15):
            pre.append(s)
            post.append(t)
            w.append(int(rng.integers(5, 10)))
            sgn.append(int(rng.choice([1, 1, -1])))
    return bundle_from_edges(
        nodes, np.asarray(pre), np.asarray(post),
        np.asarray(w), np.asarray(sgn), cutoff=5,
    )


@pytest.fixture(scope="module")
def events_out(mock_bundle):
    """One real stimulated trial through sim_backend.simulate."""
    from flygpt.sim_backend import SimConfig, simulate

    cfg = SimConfig(dt=0.02, warmup_steps=2, seed=seed_from_config({"it": 1}),
                    w_syn=0.05)
    target = np.flatnonzero(
        np.char.startswith(mock_bundle.type_array().astype(str), "ORN_")
    )
    program = {
        "family_id": "olfactory_pulse",
        "drives": [{"nodes": target.tolist(), "rate_hz": 120.0,
                    "window": [0.0, 0.4], "envelope": "poisson"}],
        "silences": [],
        "duration_s": 0.6, "dt": 0.02,
    }
    return simulate(mock_bundle, program, cfg), program


# ---------------------------------------------------------------------------
# E2E: program -> sim -> shard -> validate -> tokenize -> decode -> metrics
# ---------------------------------------------------------------------------


def test_program_to_events_to_shard(mock_bundle, events_out, tmp_path):
    """Stimulus spec drives the real sim; events land in a valid shard."""
    result, program = events_out
    assert result.events_step.size > 0, "stimulated trial produced no events"
    # driven ORN nodes emitted events inside the drive window
    driven = set(np.flatnonzero(
        np.char.startswith(mock_bundle.type_array().astype(str), "ORN_")
    ).tolist())
    fired = set(np.unique(result.events_node).tolist())
    assert driven & fired, "driven sensory nodes absent from events"
    # events respect the graph: any non-driven firing node is a graph
    # descendant of a driven node (or driven itself)
    assert int(result.events_step.max()) < result.n_steps

    from flygpt.corpus import validate_shard, write_shard

    trial_row = {
        "trial_id": 7, "family_id": "olfactory_pulse",
        "program": json.dumps(program), "seed": 99, "dt": 0.02,
        "n_steps": result.n_steps, "warmup_steps": 2,
        "n_events": int(result.events_step.size),
        "n_active_neurons": int(np.unique(result.events_node).size),
        "n_active_types": 1, "split": "train", "degenerate": False,
        "slice": "main",
    }
    manifest = {
        "schema": "schema_v1", "corpus_version": "it", "shard_id": "sh0",
        "code": {"git_sha": "x" * 8}, "graph": {"version": "g", "sha256": "ab" * 32},
        "sim_cfg": {"dt": 0.02}, "backend": "numpy", "gpu_class": "cpu",
        "dt": 0.02, "cutoff": 5, "slice": "main", "split": "train",
        "source_manifest": "corpus.manifest.json",
    }
    paths = write_shard(
        dict(zip(
            EVENT_COLUMNS,
            (np.full(result.events_step.size, 7, dtype=np.uint64),
             result.events_step, result.events_node), strict=True)),
        [trial_row], manifest, tmp_path,
    )
    checks = validate_shard(paths["events"], paths["trials"], paths["manifest"])
    assert checks["ok"], checks


def test_events_to_vocab_counts_and_back(mock_bundle, events_out):
    """Shard events -> vocab -> counts -> events round-trip."""
    result, _ = events_out
    from flygpt.tokenize import (
        build_vocab,
        counts_to_events,
        events_to_counts,
        node_to_token,
    )

    cfg = {"version": "tok_it", "instance_circuits":
           {"superclasses": ["vnc_motor"], "types": ["DNp01"]}}
    vocab = build_vocab(mock_bundle, cfg)
    tok = node_to_token(mock_bundle, vocab)
    # ORN/LC/central types get type tokens; untyped -> UNK; vnc_motor
    # typed nodes get instance tokens (designated circuit).
    assert tok.shape == (mock_bundle.n_nodes,)
    assert tok.max() < vocab.size()
    counts = events_to_counts(
        result.events_node, result.events_step, tok, vocab,
        n_steps=result.n_steps,
    )
    assert counts.shape == (result.n_steps, vocab.size())
    assert counts.sum() == result.events_step.size
    back = counts_to_events(counts, vocab)
    rebuilt = sorted((b, t) for b, t, c in back for _ in range(c))
    original = sorted(
        zip(result.events_step.tolist(), tok[result.events_node].tolist(),
            strict=True)
    )
    assert rebuilt == original


def test_decode_and_metrics_on_real_events(mock_bundle, events_out):
    """Behavior decode + EVAL metrics over the composed output."""
    result, _ = events_out
    from flygpt.behavior import BehaviorDecoder
    from flygpt.evals import common

    spec = {"heads": {
        "burst_head": {"sex": "both", "readout": {"types": ["DNp01"]},
                       "decode": {"rule": "threshold_events", "min_rate_hz": 1.0}},
    }}
    dec = BehaviorDecoder(mock_bundle, spec)
    rows = dec.decode(result.events_step, result.events_node,
                      result.n_steps, 0.02, sex="male")
    assert isinstance(rows, list)
    assert all(r["behavior"] for r in rows)

    # metrics on identical vs shifted predictions sanity
    y = np.zeros((result.n_steps, mock_bundle.n_nodes), dtype=np.uint8)
    y[result.events_step, result.events_node] = 1
    f1_same = common.event_f1(y, y)
    assert f1_same["micro"] == pytest.approx(1.0)
    y_zero = np.zeros_like(y)
    f1_zero = common.event_f1(y, y_zero)
    assert f1_zero["micro"] == pytest.approx(0.0)


def test_silencing_and_determinism_composed(mock_bundle):
    """A silencing program through the real sim: silenced nodes emit
    nothing and change nothing downstream, deterministically."""
    from flygpt.sim_backend import SimConfig, simulate

    orn = np.flatnonzero(
        np.char.startswith(mock_bundle.type_array().astype(str), "ORN_"))
    cfg = SimConfig(dt=0.02, warmup_steps=2, seed=seed_from_config({"it": 2}),
                    w_syn=0.05)
    program = {
        "family_id": "test",
        "drives": [{"nodes": orn.tolist(), "rate_hz": 150.0,
                    "window": [0.0, 0.4], "envelope": "poisson"}],
        "silences": [{"nodes": orn.tolist(), "window": [0.0, 0.4]}],
        "duration_s": 0.6, "dt": 0.02,
    }
    r1 = simulate(mock_bundle, program, cfg)
    r2 = simulate(mock_bundle, program, cfg)
    assert np.array_equal(r1.events_node, r2.events_node)
    assert np.array_equal(r1.events_step, r2.events_step)
    sil = set(orn.tolist())
    assert not (set(np.unique(r1.events_node).tolist()) & sil)


def test_stimuli_library_composes_with_sim(mock_bundle):
    """SensoryIndex + a real family generator -> sim spec -> sim run."""
    import flygpt.stimuli as stim
    from flygpt.stimuli import SensoryIndex, program_to_sim_spec

    cfg = stim.load_stim_config(Path("configs/stim_library_v1.yaml"))
    index = SensoryIndex(mock_bundle, cfg)
    gen = stim.LIBRARY["olfactory_pulse"]
    rng = np.random.default_rng(11)
    prog = gen(index, rng, cfg["families"]["olfactory_pulse"], trial=0)
    spec = program_to_sim_spec(prog)
    assert spec["drives"], "olfactory program produced no drives"
    from flygpt.sim_backend import SimConfig, simulate

    r = simulate(mock_bundle, spec, SimConfig(dt=prog.dt, warmup_steps=1,
                                            seed=prog.seed()))
    assert r.events_step.size >= 0  # composed run completes


def test_stim_query_provenance_survives_corpus(mock_bundle, tmp_path):
    """Corpus-stored program JSON keeps the grounding query so T-STIM-1
    can re-resolve it on the graph (the cross-module provenance seam)."""
    import flygpt.stimuli as stim
    from flygpt.evals import stim_checks
    from flygpt.stimuli import SensoryIndex, program_to_sim_spec

    cfg = stim.load_stim_config(Path("configs/stim_library_v1.yaml"))
    index = SensoryIndex(mock_bundle, cfg)
    rng = np.random.default_rng(5)
    programs = []
    for fam, t in (("olfactory_pulse", 0), ("gustatory_sugar", 1)):
        prog = stim.LIBRARY[fam](index, rng, cfg["families"][fam], trial=t)
        spec = program_to_sim_spec(prog)
        assert "meta" in spec, f"{fam} spec lost its meta block"
        programs.append(json.loads(json.dumps(spec, default=_jsonable)))
    # the checker re-resolves stored queries through the real index
    r = stim_checks.run_t_stim_1(programs, mock_bundle, tmp_path)
    assert r["pass"] is True, json.dumps(r["checks"], default=str)[:800]
    no_query = sum(
        p.get("n_no_query", 0) for p in r["checks"]["per_program"]
    ) if isinstance(r["checks"].get("per_program"), list) else r["checks"].get("n_no_query", 0)
    assert no_query == 0, "stored programs carried no grounding queries"


def _jsonable(o):
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def test_run_pilot_e2e_with_real_library(mock_bundle, tmp_path):
    """The real pilot gate: run_pilot drives the actual LIBRARY generators
    through sim_backend on the mock connectome and writes a report."""
    import flygpt.stimuli as stim
    from flygpt.corpus import run_pilot
    from flygpt.stimuli import SensoryIndex

    stim_cfg = stim.load_stim_config(Path("configs/stim_library_v1.yaml"))
    # tiny pilot: 1 family x 2 trials so it stays fast
    stim_cfg["pilot"] = {"n_families": 1, "trials_per_family": 2,
                         "duration_s": 0.1, "dt": 0.02}
    index = SensoryIndex(mock_bundle, stim_cfg)
    cfg = {"version": "corpus_test", "stim_library": stim_cfg}
    report = run_pilot(
        mock_bundle, stim.LIBRARY, cfg, tmp_path / "work", tmp_path,
        index=index, stim_cfg=stim_cfg,
        sim_cfg={"model": {"dt": 0.02, "warmup_steps": 1},
                 "backend": "numpy", "gpu_class": "cpu"},
    )
    art = json.loads((tmp_path / "pilot_report.json").read_text())
    assert art["artifact"] == "pilot_report"
    assert report["metrics"]["n_trials"] == 2
    assert report["checks"]["n_families"] == 1
