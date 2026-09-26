"""T-SHIP test battery - release integrity on the cards + Space side.

Owns the Phase-6 surfaces in this file's scope:

- ``space/build_canned.py``: canned rollouts validate against the
  ``canned_v1`` schema, decode to the intended motor program, are
  deterministic, and carry complete provenance; the manifest validates
  and its file hashes resolve.
- ``space/index.html``: the static demo's offline rules (T-SHIP-7) -
  references ``canned/manifest.json``, contains no external http(s)
  script/css links, and consumes exactly the schema the generator emits.
- ``src/flygpt/cards.py``: the model-card claims table is a strict subset
  of CLAIMS.md EVIDENCED rows (T-SHIP-4), the limitation list is present
  verbatim, the dataset card carries "simulated, not recorded" + full
  provenance + convention markers (T-SHIP-5), and
  ``check_card_claims`` catches an out-of-subset claim.
- T-SHIP-10 machinery is exercised with a fake model_fn/sim_fn (the real
  bench lives in ``flygpt.evals.ship_checks``).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPACE_DIR = REPO / "space"
INDEX_HTML = SPACE_DIR / "index.html"


def _load_build_canned():
    """Import space/build_canned.py as a module (space/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "build_canned", SPACE_DIR / "build_canned.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_canned"] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load_build_canned()

from flygpt import cards  # noqa: E402

CLAIMS_FIXTURE = """# CLAIMS.md - fixture register (subset of statuses exercised)

| ID | Claim | Status | Evidence artifact | Notes |
|---|---|---|---|---|
| C0 | Graph built with sha256 provenance | EVIDENCED | `results/graph_build_v1.json` | fixture |
| C1 | Held-out families above baseline | UNSUPPORTED | `results/T-MODEL-1.json` | - |
| C2 | Mask beats rewired + wrong-animal | REFUTED | `results/T-MODEL-3.json` | refuted |
| C3 | Embeddings recover held-out types | EVIDENCED | `results/T-EMB-1.json` | fixture |
| C4 | Real identities beat permuted | DOWNGRADED | `results/T-MODEL-4.json` | downgraded |
"""

RESULT_FIXTURE = {
    "artifact": "T-MODEL-3",
    "schema": "results_v1",
    "created_utc": "2026-09-30T00:00:00Z",
    "code": {"git_sha": "abc123", "generator": "flygpt T-MODEL-3"},
    "inputs": {},
    "checks": {"beats_rewired": True, "beats_wrong_animal": True},
    "metrics": {"macro_event_f1": 0.42, "margin_vs_rewired": 0.11},
    "pass": True,
}

MANIFEST_FIXTURE = {
    "corpus_version": "corpus_v1",
    "source_manifest": "provenance/malecns_v1.json",
    "code": {"git_sha": "deadbeef", "generator": "flygpt corpus"},
    "graph": {"version": "graph_v1", "sha256": "0" * 64},
    "sim_cfg": {"dt": 0.02, "w_syn": 0.02},
    "backend": "numpy",
    "gpu_class": "cpu",
    "dt": 0.02,
    "cutoff": 5,
    "slice": "main",
    "split": "train",
    "created_utc": "2026-09-30T00:00:00Z",
}


@pytest.fixture(scope="module")
def canned_set(tmp_path_factory):
    """Run the REAL canned-rollout generation on the embedded fixture."""
    out = tmp_path_factory.mktemp("canned")
    spec_graph = bc.demo_graph_spec()
    generated = [
        bc.generate(stim, spec_graph, "embedded_fixture", out)
        for stim in sorted(bc.STIMULI)
    ]
    manifest = bc.write_manifest(generated, spec_graph, "embedded_fixture", out)
    return {"dir": out, "objs": {c["stimulus"]: c for c in generated},
            "manifest": manifest, "spec": spec_graph}


@pytest.fixture()
def cards_env(tmp_path):
    claims = tmp_path / "CLAIMS.md"
    claims.write_text(CLAIMS_FIXTURE)
    results = tmp_path / "results"
    results.mkdir()
    (results / "T-MODEL-3.json").write_text(json.dumps(RESULT_FIXTURE))
    return {"claims": claims, "results": results}


# ---------------------------------------------------------------------------
# build_canned: canned schema, provenance, determinism, decode
# ---------------------------------------------------------------------------


def test_canned_schema_all_stimuli(canned_set):
    for stim_id, obj in canned_set["objs"].items():
        errs = bc.validate_canned(obj)
        assert errs == [], f"{stim_id}: {errs}"
        for key in ("stimulus", "n_nodes_small", "window_ms", "events",
                        "readout", "provenance"):
            assert key in obj
        prov = obj["provenance"]
        for key in ("code_sha", "graph_sha", "sim_cfg", "seed",
                        "generated_utc", "backend"):
            assert prov[key], f"{stim_id} provenance.{key} empty"


def test_canned_events_sorted_and_on_grid(canned_set):
    for stim_id, obj in canned_set["objs"].items():
        events = obj["events"]
        assert events == sorted(events, key=lambda e: (e[0], e[1])), stim_id
        dt = obj["dt_ms"]
        for t_ms, node in events:
            step = t_ms / dt
            assert abs(step - round(step)) < 1e-6
            assert 0 <= node < obj["n_nodes_small"]


def test_canned_decode_matches_intended_program(canned_set):
    """Wiring + drive produce the demo's intended motor program per button."""
    for stim_id, obj in canned_set["objs"].items():
        assert obj["readout"]["decoded"] == bc.STIMULI[stim_id]["expected"], stim_id
        # driven population actually spiked (real sim output, not labels)
        driven = set(obj["population"]["driven_nodes"])
        assert any(e[1] in driven for e in obj["events"]), stim_id


def test_canned_determinism(canned_set, tmp_path):
    a = bc.generate("sugar", canned_set["spec"], "embedded_fixture", tmp_path)
    b = bc.generate("sugar", canned_set["spec"], "embedded_fixture", tmp_path)
    assert a["events"] == b["events"]
    assert a["readout"] == b["readout"]
    assert a["provenance"]["seed"] == b["provenance"]["seed"]


def test_canned_manifest_validates(canned_set):
    errs = bc.validate_manifest(canned_set["manifest"], canned_set["dir"])
    assert errs == [], errs
    for entry in canned_set["manifest"]["stimuli"]:
        assert (canned_set["dir"] / entry["file"]).exists()
        assert entry["decoded"] == bc.STIMULI[entry["stimulus"]]["expected"]


def test_validate_canned_catches_corruption(canned_set):
    obj = json.loads(json.dumps(canned_set["objs"]["sugar"]))  # deep copy
    assert bc.validate_canned(obj) == []
    # unsorted events
    bad = dict(obj, events=list(reversed(obj["events"])))
    assert bc.validate_canned(bad)
    # missing required key
    bad = {k: v for k, v in obj.items() if k != "provenance"}
    assert bc.validate_canned(bad)
    # node index out of range
    bad = dict(obj, events=[[0.0, obj["n_nodes_small"] + 5]])
    assert bc.validate_canned(bad)
    # decode label inconsistent
    bad = json.loads(json.dumps(obj))
    bad["readout"]["decoded"] = "nonexistent"
    assert bc.validate_canned(bad)


def test_ethogram_stats_in_reference_ranges(canned_set):
    """The courtship canned rollout produces a SongExplorer-shaped ethogram."""
    etho = canned_set["objs"]["female_cue"]["ethogram"]
    st = etho["stats"]
    assert etho["pulse_bouts"], "no pulse bouts segmented"
    assert etho["sine_bouts"], "no sine bouts segmented"
    assert 30.0 <= st["ipi_median_ms"] <= 45.0
    assert 0.05 <= st["ipi_cv"] <= 0.5
    assert 0.2 <= st["bout_duration_s"] <= 2.0
    assert 0.3 <= st["pulse_sine_ratio"] <= 0.9
    assert st["context_dependence"] == "direction_consistent"
    # the static reference table rides along for the comparison panel
    ids = {s["id"] for s in etho["reference"]["statistics"]}
    assert {"ipi_median_ms", "pulse_sine_ratio", "bout_duration_s",
            "ipi_cv", "context_dependence"} <= ids


# ---------------------------------------------------------------------------
# index.html: static-asset offline rules (T-SHIP-7)
# ---------------------------------------------------------------------------


def test_index_html_offline_asset_rules():
    assert INDEX_HTML.exists()
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "canned/manifest.json" in html
    # no external http(s) script/css/fetch links - offline from file://
    assert not re.search(r"""(?:src|href)\s*=\s*["']https?://""", html)
    assert not re.search(r"""<script[^>]+src\s*=""", html)
    assert not re.search(r"""<link[^>]+href\s*=\s*["']https?://""", html)
    assert not re.search(r"""fetch\(\s*["']https?://""", html)
    # the honesty marker travels into the demo chrome
    assert "simulated, not recorded" in html


def test_index_html_consumes_canned_schema(canned_set):
    """Every canned file parses as JSON and exposes exactly the fields the
    page consumes - the python-side check of the JS contract."""
    consumed = ("stimulus", "title", "button", "n_nodes_small", "nodes",
                "edges", "dt_ms", "window_ms", "n_steps", "events",
                "population", "readout", "provenance")
    for path in canned_set["dir"].glob("*.json"):
        if path.name == "manifest.json":
            continue
        obj = json.loads(path.read_text())   # what fetch() hands the page
        for key in consumed:
            assert key in obj, f"{path.name} missing {key}"
        assert bc.validate_canned(obj) == []
        ro = obj["readout"]
        assert ro["decoded_label"]
        assert ro["decoded"] in ro["programs"]
        pop = obj["population"]
        assert pop["driven_nodes"] and pop["description"]
        if obj["stimulus"] == "female_cue":
            assert obj["ethogram"]["stats"]
            assert obj["ethogram"]["reference"]["statistics"]


# ---------------------------------------------------------------------------
# cards: T-SHIP-4 claims subset + verbatim limitations, T-SHIP-5 markers
# ---------------------------------------------------------------------------


def test_model_card_claims_subset_evidenced_only(cards_env):
    card = cards.model_card(cards_env["claims"], cards_env["results"])
    ids = cards.card_claim_ids(card)
    assert set(ids) == {"C0", "C3"}          # EVIDENCED rows only
    assert "C2" not in ids                    # REFUTED never appears
    assert "C1" not in ids                    # UNSUPPORTED never appears
    assert "C4" not in ids                    # DOWNGRADED never appears
    report = cards.check_card_claims(card, CLAIMS_FIXTURE)
    assert report["passed"], report
    assert report["out_of_subset"] == []


def test_model_card_limitations_verbatim(cards_env):
    card = cards.model_card(cards_env["claims"], cards_env["results"])
    for lim in cards.LIMITATIONS:
        assert lim in card
    assert cards.SIMULATED_NOT_RECORDED in card
    report = cards.check_card_claims(card, CLAIMS_FIXTURE)
    assert report["limitations_missing"] == []


def test_model_card_control_table(cards_env):
    card = cards.model_card(cards_env["claims"], cards_env["results"])
    # wired from results/*.json: present artifact shows verdict + metrics,
    # absent artifacts show pending - the controls are the product
    assert "T-MODEL-3" in card and "PASS" in card
    assert "rewired connectome mask" in card
    assert "pending" in card                  # T-MODEL-4/5/... not run yet
    assert "macro_event_f1" in card


def test_check_card_claims_flags_out_of_subset(cards_env):
    card = cards.model_card(cards_env["claims"], cards_env["results"])
    # an editor adds an UNSUPPORTED claim id - linter must catch it
    doctored = card + "\n| C1 | unsupported claim text |\n"
    report = cards.check_card_claims(doctored, CLAIMS_FIXTURE)
    assert not report["passed"]
    assert report["out_of_subset"] == ["C1"]
    assert report["out_of_subset_status"]["C1"] == "UNSUPPORTED"
    # a claim id not in the register at all is also caught
    report = cards.check_card_claims(card + "\n| C99 | invented |\n", CLAIMS_FIXTURE)
    assert "C99" in report["out_of_subset"]
    # a card missing a limitation fails the verbatim check
    report = cards.check_card_claims("claims: C0 only\n", CLAIMS_FIXTURE)
    assert not report["passed"]
    assert report["limitations_missing"]


def test_dataset_card_markers_and_provenance():
    card = cards.dataset_card(dict(MANIFEST_FIXTURE))
    assert cards.SIMULATED_NOT_RECORDED in card
    for marker in cards.CONVENTION_MARKERS:
        assert marker in card
    assert "deadbeef" in card                       # provenance chain fields
    assert "corpus_v1" in card
    assert "0" * 16 in card                          # graph sha256 shown
    report = cards.check_dataset_card(card)
    assert report["passed"], report
    # doctored card: drop the disclaimer + a marker -> fails
    bad = card.replace(cards.SIMULATED_NOT_RECORDED, "recorded data")
    report = cards.check_dataset_card(bad)
    assert not report["passed"]
    assert not report["disclaimer_present"]


def test_dataset_card_accepts_path(tmp_path):
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(MANIFEST_FIXTURE))
    assert cards.SIMULATED_NOT_RECORDED in cards.dataset_card(p)


# ---------------------------------------------------------------------------
# T-SHIP-10: speed-benchmark machinery (fake fns; real bench in ship_checks)
# ---------------------------------------------------------------------------


def test_t_ship_10_speed_benchmark_with_fake_fns():
    ship = pytest.importorskip(
        "flygpt.evals.ship_checks",
        reason="ship_checks lands with another agent",
    )
    if not hasattr(ship, "speed_benchmark"):
        pytest.skip("ship_checks.speed_benchmark not present")

    def fake_model_fn(n: int) -> None:          # fast: ~n trivial ops
        x = 0
        for i in range(n):
            x += i % 7
        return x

    def fake_sim_fn(n: int) -> None:            # ~100x heavier per step
        x = 0
        for i in range(n):
            for j in range(100):
                x += (i + j) % 7
        return x

    art = ship.speed_benchmark(
        fake_model_fn, fake_sim_fn, 200, device="cpu", results_dir=None
    )
    assert art["artifact"] == "T-SHIP-10"
    m = art["metrics"]
    assert m["model_steps_per_sec"] > m["sim_steps_per_sec"]
    assert m["speedup_ratio"] > 1.0
    assert "ratio_ge_10_for_C18" in art["checks"]


def test_t_ship_1_entry_points_and_deps(tmp_path):
    """T-SHIP-1: dist metadata + console scripts resolve; deps import."""
    ship = pytest.importorskip("flygpt.evals.ship_checks")
    art = ship.run_t_ship_1(tmp_path, repo_root=Path("."))
    assert art["artifact"] == "T-SHIP-1"
    c = art["checks"]
    assert c["dist_installed"] is True
    assert c["all_entry_points_ok"] is True, c["entry_points_resolve"]
    assert c["all_deps_import"] is True, c["deps_importable"]
    # fresh-venv half is reported, never faked
    assert art.get("fresh_env_verified") is None


def test_t_ship_6_reproduction_missing_artifacts(tmp_path):
    """T-SHIP-6 fails closed when claims cite absent results artifacts."""
    ship = pytest.importorskip("flygpt.evals.ship_checks")
    (tmp_path / "CLAIMS.md").write_text(
        "| C1 | claim | EVIDENCED | `results/never_exists.json` |\n"
        "repro: `python -m flygpt.evals --only T-MODEL-1`\n"
    )
    art = ship.run_t_ship_6(tmp_path / "r", repo_root=tmp_path)
    assert art["pass"] is False
    assert art["checks"]["claim_artifacts_missing"]


def test_t_ship_6_reproduction_with_artifact(tmp_path):
    """T-SHIP-6 passes when cited artifacts exist."""
    ship = pytest.importorskip("flygpt.evals.ship_checks")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "T-MODEL-1.json").write_text("{}")
    (tmp_path / "CLAIMS.md").write_text(
        "| C1 | claim | EVIDENCED | `results/T-MODEL-1.json` |\n"
        "repro: `python -m flygpt.evals --only T-MODEL-1`\n"
    )
    art = ship.run_t_ship_6(tmp_path / "r", repo_root=tmp_path)
    assert art["pass"] is True


def test_t_ship_7_space_offline_structure(tmp_path):
    """T-SHIP-7 audits space/ for live-dep violations."""
    ship = pytest.importorskip("flygpt.evals.ship_checks")
    art = ship.run_t_ship_7(tmp_path, repo_root=Path("."))
    c = art["checks"]
    assert c["space_dir_exists"] and c["index_present"]
    assert c["no_remote_fetches"], c.get("remote_refs")
    assert c["n_canned"] >= 5
    assert c["all_declared_have_canned"], c["manifest_stimuli"]
    assert c["provenance_declared"]


def test_t_ship_8_cpu_forward(tmp_path):
    """T-SHIP-8 CPU leg: a real masked-model forward on CPU, finite logits."""
    pytest.importorskip("jax")
    ship = pytest.importorskip("flygpt.evals.ship_checks")
    art = ship.run_t_ship_8(tmp_path)
    c = art["checks"]
    assert c["cpu_forward_finite"] is True, c.get("_forward_err")
    assert "jax_devices" in c
    # CUDA leg reported per host capability, never faked
    assert c["cuda_leg"] in ("present", "absent-cpu-only")
