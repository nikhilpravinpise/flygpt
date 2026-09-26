"""T-STIM test battery - stimulus library correctness.

Two fixture levels:

1. ``synthetic_bundle`` (conftest, 6 nodes) for wiring-level checks of the
   resolver, spec()/seed() and sim-spec conversion.
2. ``mock_bundle`` (this file, 41 nodes): a hand-built nodes table with
   realistic ORN/GRN/JON/LC/motor/untyped populations carrying the columns
   the grounding queries read (type, instance, class, superclass, subclass,
   somaSide, rootSide, somaNeuromere, entryNerve). node_idx = row position.

Every family is exercised on the real ``configs/stim_library_v1.yaml``.
Resolver expectations are recomputed independently from the raw rows - not
through SensoryIndex - so a broken filter fails the test.

Plan refs: docs/plan.md T-STIM-1..5; EVAL.md section 1.8 + T-STIM table.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flygpt.schema import (
    bundle_from_edges,
    canonical_json,
    rng_from_seed,
    seed_from_config,
)
from flygpt.stimuli import (
    LIBRARY,
    CoverageTracker,
    SensoryIndex,
    StimulusProgram,
    generate,
    load_stim_config,
    naturalistic_envelope,
    program_to_sim_spec,
)

REPO = Path(__file__).resolve().parents[1]
STIM_CONFIG = REPO / "configs" / "stim_library_v1.yaml"

# (bodyId, type, class, superclass, subclass, somaSide, rootSide, somaNeuromere, entryNerve)
MOCK_ROWS = [
    # --- olfactory receptor neurons (AL, antennal nerve) ---
    (1001, "ORN_DA1", "olfactory", "sensory", "antennal receptor neuron", "L", "L", "AL", "AN"),
    (1002, "ORN_DA1", "olfactory", "sensory", "antennal receptor neuron", "R", "R", "AL", "AN"),
    (1003, "ORN_DA1", "olfactory", "sensory", "antennal receptor neuron", "L", "L", "AL", "AN"),
    (1004, "ORN_DA1", "olfactory", "sensory", "antennal receptor neuron", "R", "R", "AL", "AN"),
    (1005, "ORN_VA2", "olfactory", "sensory", "antennal receptor neuron", "L", "L", "AL", "AN"),
    (1006, "ORN_VA2", "olfactory", "sensory", "antennal receptor neuron", "R", "R", "AL", "AN"),
    (1007, "ORN_VA3", "olfactory", "sensory", "antennal receptor neuron", "L", "L", "AL", "AN"),
    # somaSide blank -> rootSide fallback must classify this node as L
    (1008, "ORN_DA1", "olfactory", "sensory", "antennal receptor neuron", "", "L", "AL", "AN"),
    # --- gustatory receptor neurons ---
    (1009, "GRN_lab1", "gustatory", "sensory", "labellar bristle", "L", "L", "SEZ", "LbN"),
    (1010, "GRN_lab1", "gustatory", "sensory", "labellar bristle", "R", "R", "SEZ", "LbN"),
    (1011, "GRN_peg1", "gustatory", "sensory", "taste peg", "L", "L", "SEZ", "LbN"),
    # unpaired midline pharyngeal sensillum - belongs to bilateral drives
    (1012, "GRN_phg1", "gustatory", "sensory", "pharyngeal sensillum", "M", "M", "SEZ", ""),
    (1013, "GRN_leg1", "gustatory", "sensory", "leg bristle", "L", "L", "T1", "LegN"),
    (1014, "GRN_leg1", "gustatory", "sensory", "leg bristle", "R", "R", "T1", "LegN"),
    (1015, "GRN_wing1", "gustatory", "sensory", "wing bristle", "L", "L", "T2", "WingN"),
    # --- mechanosensory (chordotonal = Johnston's organ, AMMC) ---
    (1016, "JON_A", "mechanosensory", "sensory", "chordotonal organ", "L", "L", "AMMC", "AN"),
    (1017, "JON_A", "mechanosensory", "sensory", "chordotonal organ", "R", "R", "AMMC", "AN"),
    (1018, "JON_B", "mechanosensory", "sensory", "chordotonal organ", "L", "L", "AMMC", "AN"),
    (1019, "MCH_wind", "mechanosensory", "sensory", "wind_gravity", "L", "L", "AMMC", "AN"),
    (1020, "MCH_bristle", "mechanosensory", "sensory", "mechanosensory bristle", "R", "R", "T1", "LegN"),
    (1021, "MCH_camp", "mechanosensory", "sensory", "campaniform sensilla", "L", "L", "T1", "LegN"),
    (1022, "MCH_groom", "mechanosensory", "sensory", "grooming", "L", "L", "AMMC", "AN"),
    # --- visual projection neurons (optic lobe; DANDI LC panel types) ---
    (1023, "LC4", "visual", "visual_projection", "visual projection neuron", "L", "L", "LOP", ""),
    (1024, "LC4", "visual", "visual_projection", "visual projection neuron", "R", "R", "LOP", ""),
    (1025, "LC6", "visual", "visual_projection", "visual projection neuron", "L", "L", "LOP", ""),
    (1026, "LPLC2", "visual", "visual_projection", "visual projection neuron", "L", "L", "LOP", ""),
    (1027, "LPLC2", "visual", "visual_projection", "visual projection neuron", "R", "R", "LOP", ""),
    (1028, "LC10a", "visual", "visual_projection", "visual projection neuron", "L", "L", "LOP", ""),
    (1029, "LC10b", "visual", "visual_projection", "visual projection neuron", "R", "R", "LOP", ""),
    (1030, "LC10d", "visual", "visual_projection", "visual projection neuron", "R", "R", "LOP", ""),
    (1031, "LC11", "visual", "visual_projection", "visual projection neuron", "L", "L", "LOP", ""),
    (1032, "LC12", "visual", "visual_projection", "visual projection neuron", "R", "R", "LOP", ""),
    (1033, "LC17", "visual", "visual_projection", "visual projection neuron", "L", "L", "LOP", ""),
    (1034, "LC31a", "visual", "visual_projection", "visual projection neuron", "L", "L", "LOP", ""),
    # --- non-sensory + untyped nodes ---
    (1035, "DNp01", "descending", "descending", "descending neuron", "R", "R", "SEZ", ""),
    (1036, "MN_leg1", "motor", "motor", "leg motor neuron", "L", "L", "T1", ""),
    (1037, "FB_in1", "central", "central", "central brain neuron", "L", "L", "CB", ""),
    (1038, "PPL1", "central", "central", "dopaminergic neuron", "R", "R", "CB", ""),
    (1039, None, "central", "central", "central brain neuron", "L", "L", "CB", ""),
    (1040, None, None, None, None, "R", "R", None, None),
    (1041, None, "", "", "", "M", "M", "", ""),
]

_MOCK_EDGES = [  # (pre_idx, post_idx, weight, sign) - deterministic small graph
    (0, 36), (2, 36), (8, 36), (15, 20), (16, 20),
    (22, 34), (23, 34), (24, 34), (27, 34), (34, 35),
    (35, 36), (36, 37), (25, 34), (33, 34),
]

ROW_KEYS = (
    "bodyId", "type", "class", "superclass", "subclass",
    "somaSide", "rootSide", "somaNeuromere", "entryNerve",
)


def _mock_df() -> pl.DataFrame:
    cols = {k: [r[i] for r in MOCK_ROWS] for i, k in enumerate(ROW_KEYS)}
    cols["node_idx"] = list(range(len(MOCK_ROWS)))
    cols["instance"] = [
        (f"{r[1]}_{r[5] or r[6]}" if r[1] else None) for r in MOCK_ROWS
    ]
    return pl.DataFrame(cols)


def _mock_bundle():
    nodes = _mock_df()
    pre = np.array([e[0] for e in _MOCK_EDGES], dtype=np.uint32)
    post = np.array([e[1] for e in _MOCK_EDGES], dtype=np.uint32)
    weight = np.array([6] * len(_MOCK_EDGES), dtype=np.int32)
    sign = np.array([1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1], dtype=np.int8)
    return bundle_from_edges(nodes, pre, post, weight, sign, cutoff=3)


@pytest.fixture(scope="session")
def mock_bundle():
    return _mock_bundle()


@pytest.fixture(scope="session")
def stim_cfg():
    return load_stim_config(STIM_CONFIG)


@pytest.fixture(scope="session")
def index(mock_bundle, stim_cfg):
    return SensoryIndex(mock_bundle, stim_cfg)


@pytest.fixture(scope="session")
def rows():
    """Independent view of the node table: list of dicts keyed by column."""
    return _mock_df().to_dicts()


def _eff_side(row) -> str:
    return (row["somaSide"] or "") or (row["rootSide"] or "")


def _as_list(value):
    return list(value) if isinstance(value, (list, tuple, set)) else [value]


def _side_labels(sides) -> set:
    """Independent copy of the documented side semantics (bilateral=L+R+M)."""
    table = {"L": {"L"}, "R": {"R"}, "M": {"M"}, "bilateral": {"L", "R", "M"}}
    out: set = set()
    for lab in _as_list(sides):
        out |= table[lab]
    return out


def _node_matches(row, query) -> bool:
    """Independent grounding-query semantics over one node-table row."""
    for key, want in query.items():
        if key in ("types", "classes", "subclasses", "superclasses",
                   "neuromeres", "entry_nerves"):
            col = {
                "types": "type", "classes": "class", "subclasses": "subclass",
                "superclasses": "superclass", "neuromeres": "somaNeuromere",
                "entry_nerves": "entryNerve",
            }[key]
            if (row[col] or "") not in [str(w) for w in _as_list(want)]:
                return False
        elif key == "type_regex":
            if re.match(want, row["type"] or "") is None:
                return False
        elif key == "sides":
            if _eff_side(row) not in _side_labels(want):
                return False
        elif key == "dandi_panel":
            panel = _panel()
            if want is True:
                want_types = {t for v in panel.values() for t in v}
            else:
                want_types = {t for d in _as_list(want) for t in panel[d]}
            if row["type"] not in want_types:
                return False
        elif key == "typed":
            if bool(row["type"]) != bool(want):
                return False
        else:
            raise AssertionError(f"unhandled query key {key!r}")
    return True


def _expected_nodes(rows, query) -> set[int]:
    return {i for i, r in enumerate(rows) if _node_matches(r, query)}


@pytest.fixture(scope="session")
def panel(stim_cfg):
    return stim_cfg["dandi_lc_panel"]


_PANEL_CACHE: dict | None = None


def _panel() -> dict:
    """Lazy load of the DANDI LC panel for query checks outside fixtures."""
    global _PANEL_CACHE
    if _PANEL_CACHE is None:
        _PANEL_CACHE = load_stim_config(STIM_CONFIG)["dandi_lc_panel"]
    return _PANEL_CACHE


FAMILIES = [
    "olfactory_pulse", "gustatory_sugar", "gustatory_leg", "visual_lc",
    "mechanosensory_jo", "mechanosensory_other", "courtship_cue",
    "lc_silence", "optogenetic", "tonic_baseline", "naturalistic",
    "babble", "silence_test",
]


# ---------------------------------------------------------------------------
# SensoryIndex resolver paths
# ---------------------------------------------------------------------------


def test_resolver_each_criterion(index, rows):
    cases = [
        {"types": ["LC4", "LC6"]},
        {"classes": ["olfactory"]},
        {"classes": ["olfactory", "gustatory"]},
        {"subclasses": ["chordotonal organ"]},
        {"subclasses": ["labellar bristle", "taste peg", "pharyngeal sensillum"]},
        {"superclasses": ["sensory"]},
        {"superclasses": ["visual_projection"]},
        {"neuromeres": ["AMMC"]},
        {"entry_nerves": ["AN"]},
        {"type_regex": "^ORN_"},
        {"type_regex": "^(LC|LPLC|LLPC|LPC|LCNO)"},
        {"typed": True},
        {"typed": False},
        {"types": ["ORN_DA1"], "sides": "L"},
        {"type_regex": "^ORN_", "sides": "R"},
        {"sides": "bilateral", "classes": ["gustatory"]},
        {},
    ]
    for query in cases:
        got = set(index.resolve(**query).tolist())
        assert got == _expected_nodes(rows, query), f"query {query} mismatch"


def test_resolver_side_fallback_and_labels(index, rows):
    # node 7 (bodyId 1008) has somaSide="" rootSide="L" -> resolves as L
    assert 7 in set(index.resolve(sides="L").tolist())
    assert 7 not in set(index.resolve(sides="R").tolist())
    for side in ("L", "R", "M", "bilateral"):
        got = index.resolve(sides=side)
        expected = {i for i, r in enumerate(rows) if _eff_side(r) in _side_labels(side)}
        assert set(got.tolist()) == expected
    # bilateral includes the midline pharyngeal node 11
    assert 11 in set(index.resolve(sides="bilateral").tolist())
    assert 11 not in set(index.resolve(sides=["L", "R"]).tolist())
    with pytest.raises(ValueError):
        index.resolve(sides="X")


def test_resolver_missing_columns_tolerated(synthetic_bundle):
    """The conftest table lacks subclass/neuromere/nerve cols -> empty, no error."""
    index = SensoryIndex(synthetic_bundle)
    assert index.resolve(subclasses=["anything"]).size == 0
    assert index.resolve(neuromeres=["AL"]).size == 0
    assert index.resolve(entry_nerves=["AN"]).size == 0
    assert index.resolve().tolist() == list(range(6))
    assert index.resolve(types=["A_R"]).tolist() == [0, 2]
    assert index.resolve(type_regex="^A_").tolist() == [0, 2]
    assert index.resolve(sides="L").tolist() == [1, 3, 5]
    assert index.resolve(sides="R").tolist() == [0, 2]
    assert index.resolve(sides="M").tolist() == [4]
    assert index.resolve(typed=True).tolist() == [0, 1, 2, 4, 5]
    assert index.resolve(typed=False).tolist() == [3]
    assert index.resolve(classes=["gustatory"]).tolist() == [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        index.resolve(dandi_panel="LC4")  # no stim config bound


def test_resolver_dandi_panel(index, rows, panel):
    assert len(panel) == 23
    for driver, types in panel.items():
        got = index.resolve(dandi_panel=driver)
        assert set(got.tolist()) == _expected_nodes(rows, {"dandi_panel": driver})
        assert {index.type_of[int(n)] for n in got} <= set(types)
    # union of drivers and the whole panel
    got = index.resolve(dandi_panel=["LC4", "LC6"])
    assert set(got.tolist()) == _expected_nodes(rows, {"dandi_panel": ["LC4", "LC6"]})
    got_all = index.resolve(dandi_panel=True)
    all_types = {t for v in panel.values() for t in v}
    assert set(got_all.tolist()) == {i for i, r in enumerate(rows) if r["type"] in all_types}
    with pytest.raises(ValueError):
        index.resolve(dandi_panel="LC999")


# ---------------------------------------------------------------------------
# T-STIM-1 grounding: programs resolve to real annotated populations
# ---------------------------------------------------------------------------


def test_t_stim_1_programs_grounded(index, stim_cfg, rows):
    drivers = sorted(stim_cfg["dandi_lc_panel"])
    for fid in stim_cfg["families"]:
        for trial in range(2):
            kw = {"trial": trial}
            if fid == "lc_silence":
                kw["variant"] = drivers[trial]
            program = generate(fid, index, rng_from_seed("t-stim-1", fid, trial), stim_cfg, **kw)
            assert isinstance(program, StimulusProgram)
            assert len(program.meta["drives"]) == len(program.drives)
            assert len(program.meta["silences"]) == len(program.silences)
            for drive, dm in zip(program.drives, program.meta["drives"], strict=True):
                kwargs = dm["resolve_kwargs"]
                if kwargs is None:
                    # babble: seeded subset, not a grounding query
                    assert dm.get("sampling") or program.family_id == "babble"
                    continue
                expected = index.resolve(**kwargs)
                assert set(expected.tolist()) == set(drive.nodes.tolist()), (
                    fid, kwargs,
                )
                if dm.get("query"):
                    for n in drive.nodes:
                        assert _node_matches(rows[int(n)], dm["query"]), (
                            fid, int(n), dm["query"],
                        )
                if dm.get("chosen_type"):
                    assert {index.type_of[int(n)] for n in drive.nodes} == {dm["chosen_type"]}
            for s_i, (snodes, _win) in enumerate(program.silences):
                sm = program.meta["silences"][s_i]
                expected = index.resolve(**sm["resolve_kwargs"])
                assert set(expected.tolist()) == set(np.unique(snodes).tolist())


def test_t_stim_1_sensory_families_nonempty(index, stim_cfg):
    for fid in (
        "olfactory_pulse", "gustatory_sugar", "gustatory_leg", "visual_lc",
        "mechanosensory_jo", "mechanosensory_other", "courtship_cue", "naturalistic",
    ):
        program = generate(fid, index, rng_from_seed("nonempty", fid), stim_cfg, trial=0)
        total = sum(d.nodes.size for d in program.drives)
        assert total > 0, f"{fid} produced empty target set on the mock graph"


def test_t_stim_1_family_specific_targets(index, stim_cfg, rows):
    # olfactory: single ORN type, lateralized
    p = generate("olfactory_pulse", index, rng_from_seed("f1"), stim_cfg, trial=0, side="L")
    types = {index.type_of[int(n)] for n in p.drives[0].nodes}
    assert len(types) == 1 and next(iter(types)).startswith("ORN_")
    # visual_lc: single type matching the LC regex
    p = generate("visual_lc", index, rng_from_seed("f2"), stim_cfg, trial=0)
    types = {index.type_of[int(n)] for n in p.drives[0].nodes}
    assert len(types) == 1
    assert re.match("^(LC|LPLC|LLPC|LPC|LCNO)", next(iter(types)))
    # gustatory_sugar: bilateral labellar+taste-peg+pharyngeal incl midline 11
    p = generate("gustatory_sugar", index, rng_from_seed("f3"), stim_cfg, trial=0)
    got = set(p.drives[0].nodes.tolist())
    assert got == _expected_nodes(
        rows,
        {"subclasses": ["labellar bristle", "taste peg", "pharyngeal sensillum"],
         "sides": "bilateral"},
    )
    assert 11 in got  # midline pharyngeal included in a bilateral drive
    # courtship: two drives - one listed LC type + one ORN type
    p = generate("courtship_cue", index, rng_from_seed("f4"), stim_cfg, trial=0)
    assert len(p.drives) == 2
    lc_types = {"LC10a", "LC10d", "LC11", "LC12", "LC17", "LPLC2"}
    assert {index.type_of[int(n)] for n in p.drives[0].nodes} <= lc_types
    orn_types = {index.type_of[int(n)] for n in p.drives[1].nodes}
    assert len(orn_types) == 1 and next(iter(orn_types)).startswith("ORN_")
    # optogenetic: arbitrary manifest type, activate or silence
    p = generate("optogenetic", index, rng_from_seed("f5"), stim_cfg, trial=0, mode="activate")
    types = {index.type_of[int(n)] for n in p.drives[0].nodes}
    assert len(types) == 1 and p.meta["mode"] == "activate"


# ---------------------------------------------------------------------------
# T-STIM-2 laterality
# ---------------------------------------------------------------------------


def test_t_stim_2_lateralized_drives(index, stim_cfg, rows):
    for fid in ("olfactory_pulse", "gustatory_leg", "visual_lc",
                "mechanosensory_jo", "mechanosensory_other", "courtship_cue"):
        for side in ("L", "R"):
            program = generate(
                fid, index, rng_from_seed("side", fid, side), stim_cfg, side=side
            )
            for drive in program.drives:
                assert drive.nodes.size > 0
                for n in drive.nodes:
                    assert _eff_side(rows[int(n)]) == side, (
                        fid, side, int(n), rows[int(n)]["type"],
                    )
            for dm in program.meta["drives"]:
                assert dm["side"] == side


def test_t_stim_2_bilateral_includes_midline(index, stim_cfg):
    program = generate("gustatory_sugar", index, rng_from_seed("bil"), stim_cfg, trial=0)
    nodes = set(program.drives[0].nodes.tolist())
    assert program.meta["drives"][0]["side"] == "bilateral"
    assert 11 in nodes          # M pharyngeal
    assert {8, 9, 10} <= nodes  # L+R labellar + taste peg


# ---------------------------------------------------------------------------
# T-STIM-3 coverage tracker == independent recount
# ---------------------------------------------------------------------------


def _independent_recount(logged, type_of, manifest_types):
    spikes: dict[str, int] = {}
    trials: dict[str, int] = {}
    for arr in logged:
        counts: dict[int, int] = {}
        for n in arr:
            counts[int(n)] = counts.get(int(n), 0) + 1
        hit_types = set()
        for n, c in counts.items():
            t = type_of[n]
            if t:
                spikes[t] = spikes.get(t, 0) + c
                hit_types.add(t)
        for t in hit_types:
            trials[t] = trials.get(t, 0) + 1
    driven = sorted(t for t in manifest_types if trials.get(t, 0) >= 1)
    robust = sorted(
        t for t in manifest_types if spikes.get(t, 0) >= 5 and trials.get(t, 0) >= 10
    )
    return {
        "driven": driven,
        "robustly_driven": robust,
        "driven_fraction": len(driven) / len(manifest_types),
        "robust_fraction": len(robust) / len(manifest_types),
        "total_types": len(manifest_types),
    }


def test_t_stim_3_coverage_tracker(mock_bundle, index, stim_cfg):
    tracker = CoverageTracker(mock_bundle)
    program = generate("olfactory_pulse", index, rng_from_seed("cov"), stim_cfg, trial=0)
    type_of = mock_bundle.type_array()
    manifest = mock_bundle.manifest_types
    logged = []
    # 12 trials: ORN_DA1 fires 2 spikes every trial (robust); LC4 fires 7
    # spikes but only in 3 trials (driven, not robust); untyped node 38 fires
    # and must be ignored.
    for t in range(12):
        fired = [0, 2, 38]
        if t < 3:
            fired += [22] * 7
        fired = np.array(fired, dtype=np.uint32)
        logged.append(fired)
        tracker.record(program, fired)
    expected = _independent_recount(logged, type_of, manifest)
    report = tracker.report()
    assert report == expected
    assert report["driven"] == ["LC4", "ORN_DA1"]
    assert report["robustly_driven"] == ["ORN_DA1"]
    assert report["total_types"] == len(manifest)
    # TrialResult-like objects also accepted via events_node
    class FakeResult:
        events_node = np.array([22, 22, 22], dtype=np.uint32)
    tracker2 = CoverageTracker(mock_bundle)
    tracker2.record(None, FakeResult())
    tracker2.record(None, FakeResult())
    assert tracker2.report()["driven"] == ["LC4"]


def test_t_stim_3_tracker_on_synthetic(synthetic_bundle):
    tracker = CoverageTracker(synthetic_bundle)
    tracker.record(None, np.array([0, 2, 2, 3], dtype=np.uint32))
    report = tracker.report()
    assert report["driven"] == ["A_R"]          # node 3 untyped -> ignored
    assert report["total_types"] == 4
    assert report["driven_fraction"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# T-STIM-4 naturalistic envelope
# ---------------------------------------------------------------------------


def test_t_stim_4_envelope_bounds_and_determinism():
    n_steps, dt = 75, 0.02
    e1 = naturalistic_envelope(12345, "envA", n_steps, dt)
    assert e1.shape == (n_steps,) and e1.dtype == np.float32
    assert (e1 >= 0.0).all() and (e1 <= 1.0).all()
    assert np.array_equal(e1, naturalistic_envelope(12345, "envA", n_steps, dt))
    e2 = naturalistic_envelope(12345, "envB", n_steps, dt)
    e3 = naturalistic_envelope(999, "envA", n_steps, dt)
    assert not np.array_equal(e1, e2)      # different envelope_id
    assert not np.array_equal(e1, e3)      # different program seed
    # varying trace: an OU process must actually move
    assert e1.std() > 0.0


def test_t_stim_4_naturalistic_program_envelope(index, stim_cfg):
    p = generate("naturalistic", index, rng_from_seed("nat"), stim_cfg, trial=0)
    drive = p.drives[0]
    assert drive.envelope == "naturalistic" and drive.envelope_id
    spec = program_to_sim_spec(p)
    trace = spec["drives"][0]["envelope_trace"]
    n_steps = int(round(p.duration_s / p.dt))
    assert trace.shape == (n_steps,)          # sampled at the program dt
    assert (trace >= 0.0).all() and (trace <= 1.0).all()
    # deterministic regeneration of the same program -> same trace
    p_again = generate("naturalistic", index, rng_from_seed("nat"), stim_cfg, trial=0)
    assert p_again.spec() == p.spec()
    trace_again = program_to_sim_spec(p_again)["drives"][0]["envelope_trace"]
    assert np.array_equal(trace, trace_again)
    # a different trial carries a different envelope id
    p2 = generate("naturalistic", index, rng_from_seed("nat"), stim_cfg, trial=1)
    assert p2.drives[0].envelope_id != drive.envelope_id


# ---------------------------------------------------------------------------
# Silencing semantics + sim spec (T-STIM-5 harness side)
# ---------------------------------------------------------------------------


def test_lc_silence_program_semantics(index, stim_cfg):
    program = generate(
        "lc_silence", index, rng_from_seed("sil"), stim_cfg, variant="LC4", trial=0
    )
    assert program.meta["driver"] == "LC4"
    (snodes, swin), = program.silences
    expected = index.resolve(dandi_panel="LC4")
    assert set(np.unique(snodes).tolist()) == set(expected.tolist()) == {22, 23}
    t0, t1 = swin
    assert 0.0 <= t0 < t1 <= program.duration_s
    # reference drive present and disjoint from the silenced population
    assert len(program.drives) == 1
    driven = program.drives[0].nodes
    assert driven.size > 0
    assert np.intersect1d(snodes, driven).size == 0
    sim = program_to_sim_spec(program)
    assert {"drives", "silences", "duration_s", "dt"} <= set(sim)
    assert sim["drives"][0]["nodes"].tolist() == program.drives[0].nodes.tolist()
    assert sim["silences"][0]["nodes"].tolist() == [22, 23]
    assert sim["silences"][0]["window"] == (t0, t1)
    assert sim["duration_s"] == program.duration_s and sim["dt"] == program.dt


def test_lc_silence_all_23_drivers(index, stim_cfg, rows, panel):
    programs = []
    for i, driver in enumerate(sorted(panel)):
        program = generate(
            "lc_silence", index, rng_from_seed("panel", driver), stim_cfg,
            variant=driver, trial=i,
        )
        assert program.meta["driver"] == driver
        assert program.meta["panel_types"] == list(panel[driver])
        (snodes, _), = program.silences
        expected = {j for j, r in enumerate(rows) if r["type"] in set(panel[driver])}
        assert set(np.unique(snodes).tolist()) == expected
        programs.append(program)
    assert len({p.program_id for p in programs}) == 23
    # drivers whose types exist in the mock must produce non-empty silences
    nonempty = {d for d in panel if index.resolve(dandi_panel=d).size}
    expected_nonempty = {
        d for d in panel if _expected_nodes(rows, {"dandi_panel": d})
    }
    assert nonempty == expected_nonempty
    assert {
        "LC4", "LC6", "LPLC2", "LC10a", "LC10ad", "LC10bc", "LC10d",
        "LC11", "LC12", "LC17", "LC31",
    } <= nonempty
    with pytest.raises(ValueError):
        generate("lc_silence", index, rng_from_seed("x"), stim_cfg, variant="LC999")


def test_optogenetic_silence_mode(index, stim_cfg):
    program = generate(
        "optogenetic", index, rng_from_seed("opsil"), stim_cfg, mode="silence"
    )
    assert program.meta["mode"] == "silence"
    assert len(program.drives) == 0 and len(program.silences) == 1
    (snodes, swin), = program.silences
    assert snodes.size > 0 and 0.0 <= swin[0] < swin[1] <= program.duration_s


# ---------------------------------------------------------------------------
# Determinism + seed rule
# ---------------------------------------------------------------------------


def _all_programs(index, stim_cfg, trials=4):
    drivers = sorted(stim_cfg["dandi_lc_panel"])
    out = []
    for fid in stim_cfg["families"]:
        for trial in range(trials):
            kw = {"trial": trial}
            if fid == "lc_silence":
                kw["variant"] = drivers[trial % len(drivers)]
            out.append(
                generate(fid, index, rng_from_seed("det", fid, trial), stim_cfg, **kw)
            )
    return out


def test_determinism_same_inputs_identical(index, stim_cfg):
    for fid in stim_cfg["families"]:
        kw = {"trial": 3}
        if fid == "lc_silence":
            kw["variant"] = "LC4"
        a = generate(fid, index, rng_from_seed("rep"), stim_cfg, **kw)
        b = generate(fid, index, rng_from_seed("rep"), stim_cfg, **kw)
        assert a.spec() == b.spec(), fid
        assert a.seed() == b.seed()
        assert a.program_id == b.program_id


def test_seed_rule_and_jsonable_spec(index, stim_cfg):
    for program in _all_programs(index, stim_cfg, trials=1):
        spec = program.spec()
        assert program.seed() == seed_from_config(spec)
        assert json.loads(canonical_json(spec)) == spec   # JSON-able round trip
        assert canonical_json(spec) == canonical_json(spec)
        # nodes sorted lists in the canonical spec
        for d in spec["drives"]:
            assert d["nodes"] == sorted(d["nodes"])
            assert all(isinstance(n, int) for n in d["nodes"])
        for s in spec["silences"]:
            assert s["nodes"] == sorted(s["nodes"])


def test_no_seed_collisions_across_programs(index, stim_cfg):
    programs = _all_programs(index, stim_cfg, trials=4)  # 52 programs
    seeds = [p.seed() for p in programs]
    ids = [p.program_id for p in programs]
    assert len(set(seeds)) == len(seeds)
    assert len(set(ids)) == len(ids)
    # different trials -> different seeds even for the same family
    fid = "silence_test"   # content-free family: only meta.trial differs
    seeds = {
        generate(fid, index, rng_from_seed("t"), stim_cfg, trial=t).seed()
        for t in range(6)
    }
    assert len(seeds) == 6


def test_library_covers_config(stim_cfg):
    assert set(LIBRARY) >= set(stim_cfg["families"])
    assert set(FAMILIES) == set(stim_cfg["families"])


# ---------------------------------------------------------------------------
# Sim-spec round trip + wiring on the synthetic bundle
# ---------------------------------------------------------------------------


def test_program_to_sim_spec_shape(index, stim_cfg):
    for fid in ("olfactory_pulse", "tonic_baseline", "babble", "silence_test",
                "naturalistic", "optogenetic"):
        program = generate(fid, index, rng_from_seed("ss", fid), stim_cfg)
        sim = program_to_sim_spec(program)
        assert {"drives", "silences", "duration_s", "dt"} <= set(sim)
        assert len(sim["drives"]) == len(program.drives)
        assert len(sim["silences"]) == len(program.silences)
        for d, sd in zip(program.drives, sim["drives"], strict=True):
            assert sd["nodes"].tolist() == d.nodes.tolist()
            assert sd["rate_hz"] == pytest.approx(float(d.rate_hz))
            assert tuple(sd["window"]) == tuple(d.window)
            assert sd["envelope"] == d.envelope


def test_wiring_on_synthetic_bundle(synthetic_bundle, stim_cfg):
    index = SensoryIndex(synthetic_bundle, stim_cfg)
    # tonic drives the whole network incl untyped node 3
    p = generate("tonic_baseline", index, rng_from_seed("w1"), stim_cfg)
    assert p.drives[0].nodes.tolist() == [0, 1, 2, 3, 4, 5]
    assert p.drives[0].envelope == "tonic"
    assert p.seed() == seed_from_config(p.spec())
    # silence_test: unstimulated control
    p = generate("silence_test", index, rng_from_seed("w2"), stim_cfg, trial=1)
    assert p.drives == () and p.silences == ()
    sim = program_to_sim_spec(p)
    assert sim["drives"] == [] and sim["silences"] == []
    # optogenetic on the 6-node graph still grounds to a real type
    p = generate("optogenetic", index, rng_from_seed("w3"), stim_cfg, mode="activate")
    chosen = p.meta["drives"][0]["chosen_type"]
    assert chosen in {"A_R", "B_L", "C_M", "D_L"}
    assert {synthetic_bundle.type_array()[int(n)] for n in p.drives[0].nodes} == {chosen}
    # sensory families find no targets here but still produce valid specs
    p = generate("olfactory_pulse", index, rng_from_seed("w4"), stim_cfg)
    assert p.drives[0].nodes.size == 0
    json.dumps(p.spec())
