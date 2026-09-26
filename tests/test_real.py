"""T-REAL + behavior-decoder test battery (synthetic fixtures only).

Coverage:

- ``BehaviorDecoder`` on a synthetic_bundle-EXTENDED nodes table (adds
  vnc_motor superclass rows over somaNeuromere T1/T2/T3/A1, a cb_motor
  MN9 pair, and a DNp01 so the real ``configs/decoder_spec_v1.yaml``
  heads resolve): pulse-train bout timing, sustained-drive, alternating
  locomotion on anti-phase groups, threshold/burst rules, sex masking
  (T-SEX-4), and zero-resolution heads reported rather than silent.
- ``SongDecoder`` on a planted 5ms event stream (IPI ~35ms): ethogram
  stats land inside evals/song_reference.json ranges; a broken stream
  (IPI 100ms) is flagged by compare_ethogram.
- T-REAL-2: GCaMP6f kernel shape on a delta event; Spearman on
  matching vs shuffled predictions.
- ``DandiClient``: sub-<driver>-fly<n> path filtering over a recorded
  path fixture incl. all-23 discovery; download_subset over a mocked
  fetcher/downloader (no network) writing provenance + license.
- T-REAL-1: silencing_predictions with a planted-direction model_fn.

evals/common is another agent's module: when it is absent a pinned-API
stub is installed into sys.modules so these tests exercise the exact
call surface real_evals relies on (rate_spearman, ci_mean, bh_fdr,
binarize_events).
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flygpt.behavior import BehaviorDecoder, SongDecoder, compare_ethogram, load_spec
from flygpt.evals.real_evals import (
    DandiClient,
    calcium_comparison,
    gcamp_kernel,
    load_lc_panel,
    silencing_predictions,
    song_eval,
)
from flygpt.schema import bundle_from_edges

REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# evals/common pinned-API stub (used only while the real module is absent)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def common_stub():
    """Install a pinned-API flygpt.evals.common stub if the real one is missing.

    The stub mirrors common.py's actual signatures (checked against the
    landed module): rate_spearman/bh_fdr/ci_mean/binarize_events/
    response_sign/noise_floor/binomial_test.
    """
    try:
        import flygpt.evals.common  # noqa: F401

        yield
        return
    except ImportError:
        pass
    from scipy import stats as sstats

    mod = types.ModuleType("flygpt.evals.common")

    def rate_spearman(r_true, r_pred):
        t = np.asarray(r_true, dtype=np.float64).reshape(-1, 1)
        p = np.asarray(r_pred, dtype=np.float64).reshape(-1, 1)
        per_type, n_non = {}, 0
        for i in range(t.shape[1]):
            a, b = sstats.rankdata(t[:, i]), sstats.rankdata(p[:, i])
            if a.std() == 0 or b.std() == 0:
                n_non += 1
                continue
            per_type[str(i)] = float(np.corrcoef(a, b)[0, 1])
        vals = np.asarray(list(per_type.values()))
        return {
            "macro": float(vals.mean()) if vals.size else float("nan"),
            "per_type": per_type,
            "n_qualifying": len(per_type),
            "n_nonqualifying": n_non,
        }

    def ci_mean(vals, level=0.95):
        v = np.asarray(vals, dtype=np.float64).ravel()
        n = v.size
        if n == 0:
            return (float("nan"),) * 3
        mean = float(v.mean())
        if n == 1:
            return (mean, mean, mean)
        half = float(sstats.t.ppf(0.5 + level / 2, n - 1)) * float(v.std(ddof=1)) / np.sqrt(n)
        return (mean, mean - half, mean + half)

    def bh_fdr(pvals, q=0.05):
        p = np.asarray(pvals, dtype=np.float64).ravel()
        m = p.size
        order = np.argsort(p, kind="stable")
        ranked = p[order]
        below = ranked <= q * (np.arange(m) + 1) / m
        reject = np.zeros(m, dtype=bool)
        if below.any():
            reject = p <= ranked[int(np.flatnonzero(below).max())]
        p_adj = np.minimum(1.0, ranked * m / (np.arange(m) + 1))[np.argsort(order)]
        return {"reject": reject, "p_adj": p_adj, "q": q}

    def binarize_events(probs, thr=0.5):
        return np.asarray(probs) >= thr

    def response_sign(deltas, floor):
        mean, lo, hi = ci_mean(deltas)
        if lo > 0.0:
            return "increase"
        if hi < 0.0:
            return "decrease"
        if abs(mean) < floor:
            return "no-effect"
        return "ambiguous"

    def noise_floor(rates_baseline, pct=95.0):
        vals = np.abs(np.asarray(rates_baseline, dtype=np.float64).ravel())
        return float(np.percentile(vals, pct)) if vals.size else float("nan")

    def binomial_test(k, n, p=0.5):
        res = sstats.binomtest(int(k), int(n), float(p), alternative="two-sided")
        return {"k": int(k), "n": int(n), "p_null": float(p), "p": float(res.pvalue)}

    mod.rate_spearman = rate_spearman
    mod.ci_mean = ci_mean
    mod.bh_fdr = bh_fdr
    mod.binarize_events = binarize_events
    mod.response_sign = response_sign
    mod.noise_floor = noise_floor
    mod.binomial_test = binomial_test
    sys.modules["flygpt.evals.common"] = mod
    import flygpt.evals as evals_pkg

    evals_pkg.common = mod
    yield
    sys.modules.pop("flygpt.evals.common", None)


# ---------------------------------------------------------------------------
# Synthetic-bundle-extended fixture: the 6-node connectome + motor rows
# ---------------------------------------------------------------------------

# New node rows appended after the synthetic 6 (node_idx 6..17):
#   6,7  MNw_T1 wing MNs (vnc_motor, T1, L/R)   -> song_pulse
#   8,9  MNw_T2 wing MNs (vnc_motor, T2, L/R)   -> song_sine
#   10,11 MNl_T3 leg MNs (vnc_motor, T3, L/R)   -> locomotion alternation
#   12   MNg_T1 grooming MN (vnc_motor, T1, L)  -> grooming
#   13   MNa_A1 abdominal MN (vnc_motor, A1)    -> rejection (female_only)
#   14,15 MN9 pair (cb_motor)                   -> feeding
#   16   DNp01 descending giant fiber           -> escape
#   17   ORN_x olfactory                        -> control, decodes nothing
MOTOR_ROWS = [
    # (bodyId, type, superclass, subclass, somaSide, somaNeuromere)
    (100, "MNw_T1", "vnc_motor", "wing motor", "L", "T1"),
    (101, "MNw_T1", "vnc_motor", "wing motor", "R", "T1"),
    (102, "MNw_T2", "vnc_motor", "wing motor", "L", "T2"),
    (103, "MNw_T2", "vnc_motor", "wing motor", "R", "T2"),
    (104, "MNl_T3", "vnc_motor", "leg motor", "L", "T3"),
    (105, "MNl_T3", "vnc_motor", "leg motor", "R", "T3"),
    (106, "MNg_T1", "vnc_motor", "grooming", "L", "T1"),
    (107, "MNa_A1", "vnc_motor", "abdominal motor", "L", "A1"),
    (108, "MN9", "cb_motor", "pharyngeal motor", "L", "SEZ"),
    (109, "MN9", "cb_motor", "pharyngeal motor", "R", "SEZ"),
    (110, "DNp01", "descending", "descending neuron", "L", "brain"),
    (111, "ORN_x", "olfactory", "olfactory receptor", "L", "brain"),
]

# Controlled spec exercising every decode rule + masking + a ghost head.
TEST_SPEC = {
    "version": "decoder_v1",
    "heads": {
        "pulse": {
            "sex": "male_only",
            "readout": {"types": ["MNw_T1"]},
            "decode": {"rule": "pulse_train", "bout_gap_s": 0.010},
        },
        "sine": {
            "sex": "male_only",
            "readout": {"types": ["MNw_T2"]},
            "decode": {"rule": "sustained_drive", "min_bout_s": 0.05, "min_rate_hz": 20},
        },
        "walk": {
            "sex": "both",
            "readout": {"types": ["MNl_T3"]},
            "decode": {"rule": "alternating_rate", "window_s": 0.2, "min_rate_hz": 5},
        },
        "escape": {
            "sex": "both",
            "readout": {"types": ["DNp01"]},
            "decode": {"rule": "threshold_events", "min_rate_hz": 20},
        },
        "kick": {
            "sex": "female_only",
            "readout": {"types": ["MNa_A1"]},
            "decode": {"rule": "burst_pattern", "bout_gap_s": 0.05, "min_events": 3},
        },
        "feed": {
            "sex": "both",
            "readout": {"superclasses": ["cb_motor"], "types": ["MN9"]},
            "decode": {"rule": "sustained_drive", "window_s": 0.2, "min_rate_hz": 10},
        },
        "ghost": {
            "sex": "both",
            "readout": {"types": ["NoSuchType"]},
            "decode": {"rule": "threshold_events", "min_rate_hz": 10},
        },
    },
}

SONG_SPEC = {
    "version": "decoder_v1",
    "heads": {
        "song_pulse": {
            "sex": "male_only",
            "readout": {"types": ["MNw_T1"]},
            "decode": {"rule": "pulse_train", "dt": 0.005, "bout_gap_s": 0.010},
        },
        "song_sine": {
            "sex": "male_only",
            "readout": {"types": ["MNw_T2"]},
            "decode": {"rule": "sustained_drive", "dt": 0.005, "min_bout_s": 0.05, "min_rate_hz": 20},
        },
    },
}


def _events(pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """(step, node) pairs -> sorted (steps, nodes) event arrays."""
    arr = np.asarray(sorted(pairs), dtype=np.int64)
    if arr.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return arr[:, 0].copy(), arr[:, 1].copy()


@pytest.fixture()
def motor_bundle(synthetic_bundle, synthetic_nodes_df):
    """synthetic_bundle (6 nodes) extended with 12 motor/DN node rows."""
    rows = {
        "bodyId": [r[0] for r in MOTOR_ROWS],
        "node_idx": list(range(6, 6 + len(MOTOR_ROWS))),
        "type": [r[1] for r in MOTOR_ROWS],
        "instance": [f"{r[1]}_{r[4]}" for r in MOTOR_ROWS],
        "superclass": [r[2] for r in MOTOR_ROWS],
        "class": [("central" if r[2] != "olfactory" else "sensory") for r in MOTOR_ROWS],
        "subclass": [r[3] for r in MOTOR_ROWS],
        "somaSide": [r[4] for r in MOTOR_ROWS],
        "rootSide": [r[4] for r in MOTOR_ROWS],
        "somaNeuromere": [r[5] for r in MOTOR_ROWS],
        "entryNerve": [None] * len(MOTOR_ROWS),
        "exitNerve": ["CvN"] * len(MOTOR_ROWS),
        "statusLabel": ["Reviewed"] * len(MOTOR_ROWS),
        "status": ["Traced"] * len(MOTOR_ROWS),
        "dimorphism": [None] * len(MOTOR_ROWS),
    }
    ext = pl.DataFrame(rows)
    nodes = pl.concat([synthetic_nodes_df, ext], how="diagonal_relaxed")
    n_old = synthetic_bundle.n_nodes
    pre = np.repeat(np.arange(n_old), np.diff(synthetic_bundle.out_indptr))
    post = synthetic_bundle.out_indices
    weight = synthetic_bundle.out_weight
    sign = synthetic_bundle.out_sign
    # add: DNp01(16) -> wing MN(6), wing MN(6) -> leg MN(10)
    pre = np.r_[pre, np.array([16, 6])]
    post = np.r_[post, np.array([6, 10])]
    weight = np.r_[weight, np.array([8, 6])]
    sign = np.r_[sign, np.array([1, 1])]
    return bundle_from_edges(nodes, pre, post, weight, sign)


def _decode(decoder, pairs, n_steps=1200, dt=0.005, sex="male"):
    steps, nodes = _events(pairs)
    return decoder.decode(steps, nodes, n_steps, dt, sex=sex)


# ---------------------------------------------------------------------------
# BehaviorDecoder: resolution coverage + decode rules
# ---------------------------------------------------------------------------


def test_real_spec_resolves_all_heads(motor_bundle):
    """The committed decoder_spec_v1.yaml resolves on the extended table."""
    spec = load_spec(REPO / "configs" / "decoder_spec_v1.yaml")
    dec = BehaviorDecoder(motor_bundle, spec)
    report = dec.coverage_report()
    assert report["unresolved_heads"] == []
    # T1+T2 vnc_motor = wing pairs (6-9) + the grooming T1 node (12)
    assert dec.resolution["song_pulse"]["n_nodes"] == 5
    assert dec.resolution["locomotion"]["n_nodes"] == 7  # T1..T3 = 6..12
    assert dec.resolution["grooming"]["node_idx"].tolist() == [12]
    assert dec.resolution["feeding"]["node_idx"].tolist() == [14, 15]
    assert dec.resolution["escape"]["node_idx"].tolist() == [16]
    assert dec.resolution["rejection"]["node_idx"].tolist() == [13]
    rm = dec.readout_map()
    assert rm["feeding"] == [14, 15] and set(rm) == set(spec["heads"])


def test_pulse_train_bouts(motor_bundle):
    dec = BehaviorDecoder(motor_bundle, TEST_SPEC)
    # node 6: bursts [10,11,12] and [30,31,32]; node 7 joins the first
    pairs = [(s, 6) for s in (10, 11, 12, 30, 31, 32)] + [(11, 7)]
    rows = [r for r in _decode(dec, pairs) if r["behavior"] == "pulse"]
    assert len(rows) == 2
    b0, b1 = rows
    assert b0["t0_s"] == pytest.approx(0.050)
    assert b0["t1_s"] == pytest.approx(0.065)
    assert b0["evidence"]["nodes"] == [6, 7]
    assert b0["evidence"]["n_events"] == 4
    assert b0["masked"] is False
    assert b1["t0_s"] == pytest.approx(0.150)
    assert b1["evidence"]["nodes"] == [6]
    assert b1["evidence"]["n_events"] == 3


def test_sustained_drive_detection(motor_bundle):
    dec = BehaviorDecoder(motor_bundle, TEST_SPEC)
    # nodes 8,9 alternate firing every step over 0.5-1.0 s
    pairs = [(s, 8 if s % 2 == 0 else 9) for s in range(100, 200)]
    rows = [r for r in _decode(dec, pairs) if r["behavior"] == "sine"]
    assert len(rows) == 1
    bout = rows[0]
    assert bout["t0_s"] == pytest.approx(0.5)
    assert bout["t1_s"] == pytest.approx(1.0)
    assert bout["evidence"]["n_events"] == 100
    assert bout["evidence"]["nodes"] == [8, 9]
    # a sparse, non-sustained stream must not decode a bout
    rows2 = [r for r in _decode(dec, [(100, 8), (140, 9), (180, 8)]) if r["behavior"] == "sine"]
    assert rows2 == []


def test_alternating_locomotion(motor_bundle):
    dec = BehaviorDecoder(motor_bundle, TEST_SPEC)
    # anti-phase: L(10) on even steps, R(11) on odd steps for 2 s
    pairs = [(s, 10) for s in range(0, 400, 2)] + [(s, 11) for s in range(1, 400, 2)]
    rows = [r for r in _decode(dec, pairs) if r["behavior"] == "walk"]
    assert len(rows) >= 1
    total = sum(r["t1_s"] - r["t0_s"] for r in rows)
    assert total > 1.0  # sustained alternation across the 2 s span
    assert rows[0]["evidence"]["n_events"] > 200
    # synchronous in-phase firing must NOT decode as locomotion
    sync = [(s, 10) for s in range(0, 400, 2)] + [(s, 11) for s in range(0, 400, 2)]
    rows2 = [r for r in _decode(dec, sync) if r["behavior"] == "walk"]
    assert rows2 == []


def test_threshold_escape_and_burst_pattern(motor_bundle):
    dec = BehaviorDecoder(motor_bundle, TEST_SPEC)
    # DNp01 burst at 2.5 s -> rate 200 Hz instantaneous > 20 Hz
    esc = [r for r in _decode(dec, [(s, 16) for s in range(500, 510)]) if r["behavior"] == "escape"]
    assert len(esc) == 1
    assert esc[0]["t0_s"] == pytest.approx(2.5)
    assert esc[0]["t1_s"] == pytest.approx(2.55)
    # kick (female_only) decodes under sex="female": cluster of 3 kept, singleton dropped
    rows = _decode(dec, [(100, 13), (101, 13), (103, 13), (200, 13)], sex="female")
    kicks = [r for r in rows if r["behavior"] == "kick"]
    assert len(kicks) == 1
    assert kicks[0]["evidence"]["n_events"] == 3


def test_sex_masking_and_zero_resolution(motor_bundle):
    dec = BehaviorDecoder(motor_bundle, TEST_SPEC)
    rows = _decode(dec, [(s, 6) for s in (10, 11, 12)], sex="female")
    masked = {r["behavior"]: r["masked_reason"] for r in rows if r["masked"]}
    # male_only heads masked under female conditioning (T-SEX-4)
    assert masked["pulse"] == "sex" and masked["sine"] == "sex"
    # zero-resolution head is emitted as masked, never silently empty
    assert masked["ghost"] == "unresolved_readout"
    report = dec.coverage_report()
    assert "ghost" in report["unresolved_heads"]
    assert report["per_head"]["ghost"]["resolved"] is False
    # female_only head is the masked one under male conditioning
    rows_m = _decode(dec, [(100, 13), (101, 13), (102, 13)], sex="male")
    kick = [r for r in rows_m if r["behavior"] == "kick"]
    assert kick[0]["masked"] is True and kick[0]["masked_reason"] == "sex"


# ---------------------------------------------------------------------------
# SongDecoder: ethogram stats vs frozen SongExplorer ranges
# ---------------------------------------------------------------------------


def _song_stream(ipi_steps: int, pulses_per_bout: list[int], sine_span=(700, 900)):
    """Plant pulses of 2 spikes (onset, onset+1) spaced ipi_steps apart."""
    pairs = []
    onset = 100
    for n_pulses, gap_after in pulses_per_bout:
        for _ in range(n_pulses):
            pairs += [(onset, 6), (onset + 1, 6)]
            onset += ipi_steps
        onset += gap_after
    for s in range(*sine_span):
        pairs.append((s, 8 if s % 2 == 0 else 9))
    return _events(pairs)


def test_song_ethogram_inside_reference(motor_bundle):
    dec = SongDecoder(motor_bundle, SONG_SPEC)
    # IPI pattern alternates 6/8 steps (30/40 ms -> median 35, cv ~0.14)
    pairs_planted = []
    onset = 100
    for i in range(33):  # bout A: 0.5 s .. ~1.63 s, IPIs alternate 30/40 ms
        pairs_planted += [(onset, 6), (onset + 1, 6)]
        onset += 6 if i % 2 == 0 else 8
    onset = 1200  # bout B: 6.0 s .. ~6.92 s
    for i in range(27):
        pairs_planted += [(onset, 6), (onset + 1, 6)]
        onset += 6 if i % 2 == 0 else 8
    for s in range(700, 900):  # sine drive 3.5-4.5 s on nodes 8,9
        pairs_planted.append((s, 8 if s % 2 == 0 else 9))
    steps, nodes = _events(pairs_planted)
    stats = dec.ethogram(
        steps, nodes, 2400, context=[("near", 0.4, 2.0), ("far", 3.0, 7.5)]
    )
    assert stats["masked"] is False
    assert 30 <= stats["ipi_median_ms"] <= 45
    assert 0.05 <= stats["ipi_cv"] <= 0.5
    assert 0.3 <= stats["pulse_sine_ratio"] <= 0.9
    assert 0.2 <= stats["bout_duration_median_s"] <= 2.0
    assert stats["context_dependence"] == "direction_consistent"
    assert stats["context_switch_log"], "expected pulse<->sine switches"
    ref = json.loads((REPO / "evals" / "song_reference.json").read_text())
    cmp_ = compare_ethogram(stats, ref)
    assert cmp_["n_inside"] == 5 and cmp_["passed"] is True
    assert cmp_["statistics"]["ipi_median_ms"]["inside"] is True
    assert cmp_["statistics"]["ipi_median_ms"]["ks"] is not None


def test_song_ethogram_broken_stream_flagged(motor_bundle):
    dec = SongDecoder(motor_bundle, SONG_SPEC)
    steps, nodes = _song_stream(
        ipi_steps=20,  # 100 ms IPI - far outside [30, 45]
        pulses_per_bout=[(6, 100), (6, 0)],
    )
    stats = dec.ethogram(steps, nodes, 2400)
    ref = json.loads((REPO / "evals" / "song_reference.json").read_text())
    cmp_ = SongDecoder.compare_ethogram(stats, ref)
    assert cmp_["statistics"]["ipi_median_ms"]["inside"] is False
    assert cmp_["n_inside"] < 5
    assert cmp_["passed"] is False


def test_song_masked_on_female(motor_bundle):
    dec = SongDecoder(motor_bundle, SONG_SPEC)
    steps, nodes = _events([(10, 6), (11, 6)])
    stats = dec.ethogram(steps, nodes, 400, sex="female")
    assert stats["masked"] is True and stats["ipi_ms"] == []


def test_song_eval_writes_result(motor_bundle, tmp_path):
    dec = SongDecoder(motor_bundle, SONG_SPEC)
    steps, nodes = _song_stream(ipi_steps=7, pulses_per_bout=[(30, 0)])
    stats = dec.ethogram(steps, nodes, 2400)
    out = song_eval(stats, results_dir=tmp_path)
    artifact = json.loads((tmp_path / "T-REAL-3.json").read_text())
    assert artifact["artifact"] == "T-REAL-3"
    assert artifact["schema"] == "results_v1"
    assert "n_inside_ge_4_of_5" in artifact["checks"]
    assert out["test_id"] == "T-REAL-3"


# ---------------------------------------------------------------------------
# T-REAL-2: GCaMP kernel + Spearman comparison
# ---------------------------------------------------------------------------


def test_gcamp_kernel_shape_and_delta_conv():
    dt = 0.02
    k = gcamp_kernel(dt)
    assert k[0] == pytest.approx(0.0)
    peak = int(np.argmax(k))
    # theoretical peak t* = ln(tau_d/tau_r) / (1/tau_r - 1/tau_d) = 0.128 s
    assert 5 <= peak <= 8
    assert k[peak] == pytest.approx(1.0)
    # decaying tail: after the peak the kernel falls off monotonically
    assert k[peak] > k[peak + 20] > k[peak + 60]
    assert k[-1] < 0.05 * k.max()
    # a delta event reproduces the kernel shape exactly
    pred = np.zeros(300)
    pred[50] = 1.0
    conv = np.convolve(pred, k)[:300]
    assert conv[50] == pytest.approx(0.0)
    assert int(np.argmax(conv)) == 50 + peak
    assert conv[50 + peak] == pytest.approx(1.0)


def test_calcium_spearman_matching_vs_shuffled():
    dt = 0.02
    rng = np.random.default_rng(7)
    t = np.arange(500)
    pred = {"LC4": (rng.random(t.size) < 0.05).astype(float),
            "LC6": (rng.random(t.size) < 0.05).astype(float)}
    kern = gcamp_kernel(dt)
    rec = {lc: np.convolve(pred[lc], kern)[: t.size] for lc in pred}
    out = calcium_comparison(pred, rec, dt)
    assert out["n_qualifying"] == 2
    assert out["macro_spearman"] == pytest.approx(1.0, abs=1e-6)
    shuffled = {lc: rng.permutation(pred[lc]) for lc in pred}
    out_shuf = calcium_comparison(shuffled, rec, dt)
    assert out_shuf["macro_spearman"] < out["macro_spearman"]


# ---------------------------------------------------------------------------
# DandiClient: path filtering + partial download (fully offline)
# ---------------------------------------------------------------------------

LC_DRIVERS = list(load_lc_panel())


def _fake_asset_paths() -> list[str]:
    """~30 recorded-style paths covering all 23 drivers + non-panel subjects."""
    paths = []
    for i, drv in enumerate(LC_DRIVERS):
        fly = i % 3 + 1
        paths.append(f"sub-{drv}-fly{fly}/sub-{drv}-fly{fly}_ses-silencing_behavior.nwb")
        if i % 5 == 0:
            paths.append(f"sub-{drv}-fly{fly}/sub-{drv}-fly{fly}_ses-silencing_song.nwb")
    paths += [
        "sub-CONTROL-fly1/sub-CONTROL-fly1_ses-sham_behavior.nwb",
        "sub-210525/sub-210525_ses-20210525T000000_image+ophys.nwb",
        "derivatives/README.txt",
    ]
    return paths


def test_filter_lc_sessions_discovers_all_23():
    client = DandiClient()
    sessions = client.filter_lc_sessions(_fake_asset_paths())
    assert len(sessions) == 23
    assert sessions["LC4"] == [
        "sub-LC4-fly1/sub-LC4-fly1_ses-silencing_behavior.nwb",
        "sub-LC4-fly1/sub-LC4-fly1_ses-silencing_song.nwb",
    ]
    rep = client.last_filter
    assert rep["n_matched"] + rep["n_unmatched"] == rep["n_paths"]
    assert "sub-CONTROL-fly1" in rep["unmatched_subjects"]
    assert len(rep["discovered_lc_types"]) == 23
    # paths under non-panel or malformed subjects are never bucketed
    assert all("CONTROL" not in p for paths in sessions.values() for p in paths)


def test_download_subset_partial_with_provenance(tmp_path):
    assets = [
        {"path": p, "asset_id": f"asset-{i:03d}", "size": 10 + i}
        for i, p in enumerate(_fake_asset_paths())
        if "sub-LC4" in p or "sub-LC6" in p
    ]
    meta = {
        "id": "DANDI:000951/0.240418.2218",
        "doi": "10.48324/dandi.000951/0.240418.2218",
        "name": "silencing sessions",
        "license": ["spdx:CC-BY-4.0"],
        "access": [{"status": "dandi:OpenAccess"}],
        "citation": "Cowley et al. 2024",
    }

    def fake_fetcher(url, params=None):
        if url.endswith("/versions/0.240418.2218/"):
            return meta
        assert url.endswith("/assets/")
        return {"count": len(assets), "next": None, "results": assets}

    written = {}

    def fake_downloader(url, dest):
        assert url.startswith("https://api.dandiarchive.org/api/assets/")
        content = b"nwb:" + url.encode()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        written[url] = dest
        return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}

    client = DandiClient(fetcher=fake_fetcher, downloader=fake_downloader)
    manifest = client.download_subset(["LC4"], tmp_path / "raw", tmp_path / "prov" / "dandi_000951.json")

    lc4_paths = [p for p in _fake_asset_paths() if "sub-LC4" in p]
    assert len(manifest["files"]) == len(lc4_paths)
    assert {f["path"] for f in manifest["files"]} == set(lc4_paths)
    # partial download only: nothing outside the requested LC types
    assert len(written) == len(lc4_paths)
    assert manifest["license"] == "CC-BY-4.0"
    assert manifest["embargo_status"] == "OPEN"
    assert manifest["partial_download"] is True
    for f in manifest["files"]:
        assert f["sha256"] and f["bytes"] > 0
        assert (tmp_path / "raw" / f["local_name"]).exists()
    on_disk = json.loads((tmp_path / "prov" / "dandi_000951.json").read_text())
    assert on_disk["requested_lc_types"] == ["LC4"]


# ---------------------------------------------------------------------------
# T-REAL-1: silencing_predictions with a planted-direction model_fn
# ---------------------------------------------------------------------------


def _panel_on_fixture() -> dict[str, list[str]]:
    """The real 23 driver names mapped onto types present in the fixture."""
    types = ["MN9", "DNp01", "MNa_A1", "MNw_T1"]
    return {drv: [types[i % len(types)]] for i, drv in enumerate(LC_DRIVERS)}


def test_silencing_predictions_perfect_panel(motor_bundle, tmp_path):
    panel = _panel_on_fixture()
    planted = {}
    for i, lc in enumerate(panel):
        planted[lc] = 5.0 if i < 8 else (-5.0 if i < 16 else 0.0)
    outcomes = {
        lc: ("increase" if d > 0 else "decrease" if d < 0 else "no-effect")
        for lc, d in planted.items()
    }

    def model_fn(lc_type, silenced_nodes, seed, bundle):
        assert lc_type in panel and bundle is motor_bundle
        return planted[lc_type]

    out = silencing_predictions(
        model_fn, panel, motor_bundle, outcomes=outcomes, results_dir=tmp_path
    )
    assert out["n_scored"] == 23 and out["n_correct"] == 23
    assert out["score"] == pytest.approx(1.0)
    assert out["passed"] is True
    assert out["per_lc"]["LC4"]["predicted"] == "increase"
    assert out["per_lc"]["LC4"]["n_silenced_nodes"] == 2  # MN9 pair
    noeff = [lc for lc, d in planted.items() if d == 0.0][0]
    assert out["per_lc"][noeff]["predicted"] == "no-effect"
    # per-type binomial: 3/3 seeds correct vs 1/3 null -> p = (1/3)^3
    assert out["per_lc"]["LC4"]["binomial_p"] == pytest.approx(1.0 / 27.0)
    assert set(out["bh_fdr"]["rejected_lc_types"]) == set(panel)
    artifact = json.loads((tmp_path / "T-REAL-1.json").read_text())
    assert artifact["artifact"] == "T-REAL-1"
    assert artifact["pass"] is True
    assert artifact["checks"]["score_at_least_16_23"] is True


def test_silencing_predictions_partial_score(motor_bundle):
    panel = _panel_on_fixture()
    planted = dict.fromkeys(panel, 5.0)  # model always predicts increase
    outcomes = dict.fromkeys(panel, "increase")
    # flip 5 published outcomes -> 18/23 correct
    for lc in list(panel)[:5]:
        outcomes[lc] = "decrease"

    out = silencing_predictions(
        lambda lc_type, silenced_nodes, seed, bundle: planted[lc_type],
        panel,
        motor_bundle,
        outcomes=outcomes,
        seeds=[1, 2, 3],
    )
    assert out["n_correct"] == 18
    assert out["score"] == pytest.approx(18 / 23)
    flipped = list(panel)[0]
    assert out["per_lc"][flipped]["correct"] is False
    # no correct seeds on flipped types -> high two-sided p, not rejected
    assert out["per_lc"][flipped]["binomial_p"] > 0.5
    assert flipped not in out["bh_fdr"]["rejected_lc_types"]


def test_run_t_real_wrappers_emit_documented_na(tmp_path):
    """Dispatcher entries return NA (not exceptions) when their real
    inputs are absent - matching the results-scoped contract."""
    import flygpt.evals.real_evals as re_

    art = re_.run_t_real_1(tmp_path)
    assert art["artifact"] == "T-REAL-1"
    assert art["pass"] is None and art["status"] == "na"
    art2 = re_.run_t_real_2(tmp_path)
    assert art2["pass"] is None and art2["status"] == "na"
    art3 = re_.run_t_real_3(tmp_path)
    assert art3["pass"] is None and art3["status"] == "na"


def test_sim_silencing_model_scores_with_fixture(motor_bundle, tmp_path):
    """The shipped LIF-sim-as-predictor runs through the real sim backend
    on the fixture and produces a valid (signed) direction per LC type."""
    import flygpt.evals.real_evals as re_

    predict = re_.sim_silencing_model(motor_bundle)
    out = predict("LC4", np.array([6, 7]), 7, motor_bundle)
    # model_fn contract: returns a mean-rate delta; the scorer maps it
    # to a direction via response_sign
    assert np.isfinite(float(out))
