"""Real-data evals T-REAL-1..3: DANDI perturbation, calcium, song.

Implements the real-data validation layer of EVAL.md section 5:

- T-REAL-1 (EVAL 1.6): DANDI 000951 (459 courtship sessions, 23 LC
  types silenced) perturbation benchmark. ``DandiClient`` lists the
  dandiset's assets over the REST API, ``filter_lc_sessions`` buckets
  silencing sessions by driver via the ``sub-<driver>-fly<n>`` subject
  convention (the 23 drivers live in
  ``configs/stim_library_v1.yaml::dandi_lc_panel``), and
  ``download_subset`` pulls a PARTIAL download of the requested
  silencing sessions only, recorded in
  ``provenance/dandi_000951.json`` with per-file sha256 and the
  dandiset's license + embargo status read from its version metadata.
  ``silencing_predictions`` scores a model's predicted direction of
  effect per LC type with the EVAL-1.5 sign rule (mean rate-change sign
  on a defined readout, 95% CI-separated across seeds; "no-effect" iff
  the CI includes zero AND the point estimate sits under the noise
  floor). Score = fraction correct of 23; per-type binomial tests
  across seeds, BH-FDR corrected at q<=0.10 (exploratory panel).

- T-REAL-2 (stretch): predicted population rates convolved through a
  GCaMP6f indicator kernel (double-exponential, tau_r=0.05s,
  tau_d=0.5s sampled at dt, unit-peak) vs recorded dF/F proxies -
  Spearman per aligned type, target >=0.25, reported regardless.

- T-REAL-3: song ethogram scoring delegated to
  ``flygpt.behavior.compare_ethogram`` against the frozen
  ``evals/song_reference.json`` ranges (>=4/5 inside, KS per stat).

Schema: every eval writes ``results/<test-id>.json`` via
``flygpt.schema.write_result`` (impl-spec evals section). Stats helpers
come from ``flygpt.evals.common`` (pinned API: event_f1,
rate_spearman, bh_fdr, ci_mean, binarize_events - plus response_sign,
noise_floor, binomial_test for the 1.5/1.6 rules), imported lazily
inside functions so this module imports without the eval battery.

Plan refs: docs/plan.md T-REAL-1..3, EVAL.md 1.5/1.6 + section 5.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from flygpt.schema import GraphBundle, git_sha, seed_from_config, utcnow, write_result

REPO_ROOT = Path(__file__).resolve().parents[3]
STIM_CONFIG = REPO_ROOT / "configs" / "stim_library_v1.yaml"
SONG_REFERENCE = REPO_ROOT / "evals" / "song_reference.json"

DANDI_BASE_URL = "https://api.dandiarchive.org"
DANDI_SET = "000951"
DANDI_VERSION = "0.240418.2218"
DANDI_MANIFEST = REPO_ROOT / "provenance" / "dandi_000951.json"
DANDI_RAW_DIR = REPO_ROOT / "data" / "raw" / "dandi_000951"

TEST_SILENCING = "T-REAL-1"
TEST_CALCIUM = "T-REAL-2"
TEST_SONG = "T-REAL-3"

# GCaMP6f indicator-kernel approximation (double exponential, unit
# peak): tau_rise ~0.05s, tau_decay ~0.5s per documented physiology
# approximations for T-REAL-2's convolved-trace comparison.
GCAMP6F_TAU_RISE_S = 0.05
GCAMP6F_TAU_DECAY_S = 0.5

CHUNK = 8 * 1024 * 1024
_SUBJECT_RE = re.compile(r"sub-(?P<driver>[A-Za-z0-9]+)-fly(?P<fly>\d+)")


def load_lc_panel(path: Path | None = None) -> dict[str, list[str]]:
    """The 23 DANDI LC silencing drivers -> MaleCNS type sets."""
    path = path or STIM_CONFIG
    cfg = yaml.safe_load(Path(path).read_text())
    return {k: list(v) for k, v in cfg["dandi_lc_panel"].items()}


# ---------------------------------------------------------------------------
# evals/common - pinned API, lazy-imported inside functions per impl-spec
# ---------------------------------------------------------------------------


def _common():
    from flygpt.evals import common

    return common


# ---------------------------------------------------------------------------
# DANDI 000951 client (T-REAL-1 data layer, EVAL 1.6)
# ---------------------------------------------------------------------------


class DandiClient:
    """REST client over api.dandiarchive.org for dandiset 000951.

    ``fetcher``/``downloader`` are injectable callables so tests run
    fully offline: ``fetcher(url, params) -> dict`` performs JSON GETs
    and ``downloader(url, dest_path) -> {"bytes", "sha256"}`` performs
    streamed file downloads. The real implementations are the defaults.
    """

    def __init__(
        self,
        dandiset: str = DANDI_SET,
        version: str = DANDI_VERSION,
        base_url: str = DANDI_BASE_URL,
        fetcher: Callable | None = None,
        downloader: Callable | None = None,
        lc_panel: dict[str, list[str]] | None = None,
    ):
        self.dandiset = dandiset
        self.version = version
        self.base_url = base_url.rstrip("/")
        self._fetcher = fetcher or self._default_fetcher
        self._downloader = downloader or self._default_downloader
        self._lc_panel = lc_panel
        self.last_filter: dict[str, Any] = {}

    @property
    def lc_panel(self) -> dict[str, list[str]]:
        if self._lc_panel is None:
            self._lc_panel = load_lc_panel()
        return self._lc_panel

    @staticmethod
    def _default_fetcher(url: str, params: dict | None = None) -> dict:
        import requests

        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _default_downloader(url: str, dest: Path) -> dict[str, Any]:
        import requests

        dest.parent.mkdir(parents=True, exist_ok=True)
        sha, size = hashlib.sha256(), 0
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for block in r.iter_content(CHUNK):
                    fh.write(block)
                    sha.update(block)
                    size += len(block)
        return {"bytes": size, "sha256": sha.hexdigest()}

    def version_url(self, dandiset: str | None = None, version: str | None = None) -> str:
        d = dandiset or self.dandiset
        v = version or self.version
        return f"{self.base_url}/api/dandisets/{d}/versions/{v}/"

    def assets_url(self, dandiset: str | None = None, version: str | None = None) -> str:
        return f"{self.version_url(dandiset, version)}assets/"

    def _paginate(self, url: str, params: dict | None = None) -> list[dict]:
        """Follow DRF pagination (``next`` URLs) to collect all results."""
        out: list[dict] = []
        while url:
            page = self._fetcher(url, params)
            out.extend(page.get("results", []))
            url = page.get("next") or ""
            params = None  # next URL already carries the query params
        return out

    def list_assets(
        self, dandiset: str | None = None, version: str | None = None
    ) -> list[dict[str, Any]]:
        """All file assets of the version: [{path, asset_id, size}].

        Uses the paginated ``assets/`` listing endpoint - the file-path
        listing of the dandiset. (``assets/paths/`` is a folder-tree
        browser that would need recursive traversal and does not yield
        asset ids, so the flat ``assets/`` pages are the download map.)
        """
        return self._paginate(
            self.assets_url(dandiset, version), params={"page_size": 1000}
        )

    def filter_lc_sessions(self, asset_paths: list[str]) -> dict[str, list[str]]:
        """Bucket asset paths by LC driver via sub-<driver>-fly<n>.

        A path joins driver ``X``'s bucket iff it contains a
        ``sub-X-fly<n>`` subject component with ``X`` in the configured
        23-driver panel. Non-matching paths are counted in
        ``self.last_filter`` (reported, never silently dropped).
        """
        panel = self.lc_panel
        out: dict[str, list[str]] = {}
        unmatched: list[str] = []
        foreign_subjects: set[str] = set()
        for path in asset_paths:
            m = _SUBJECT_RE.search(path)
            if m is None:
                unmatched.append(path)
                continue
            driver = m.group("driver")
            if driver not in panel:
                foreign_subjects.add(m.group(0))
                unmatched.append(path)
                continue
            out.setdefault(driver, []).append(path)
        for paths in out.values():
            paths.sort()
        self.last_filter = {
            "n_paths": len(asset_paths),
            "n_matched": sum(len(v) for v in out.values()),
            "n_unmatched": len(unmatched),
            "unmatched_subjects": sorted(foreign_subjects),
            "discovered_lc_types": sorted(out),
            "panel_size": len(panel),
        }
        return out

    def asset_download_url(self, asset_id: str) -> str:
        """Direct download URL for one asset id."""
        return f"{self.base_url}/api/assets/{asset_id}/download/"

    def dandiset_metadata(self) -> dict[str, Any]:
        """Version metadata: license string + embargo status."""
        meta = self._fetcher(self.version_url(), None)
        licenses = [
            str(x).removeprefix("spdx:") for x in meta.get("license", [])
        ]
        access = meta.get("access") or [{}]
        status = str(access[0].get("status", ""))
        embargo = "OPEN" if status.endswith("OpenAccess") else "EMBARGOED"
        return {
            "id": meta.get("id"),
            "name": meta.get("name"),
            "doi": meta.get("doi"),
            "license": licenses[0] if licenses else None,
            "licenses": licenses,
            "access_status": status,
            "embargo_status": embargo,
            "citation": meta.get("citation"),
            "raw": meta,
        }

    def download_subset(
        self,
        lc_types: list[str],
        dest_dir: Path,
        manifest_path: Path,
        assets: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Partial download of silencing sessions for ``lc_types`` only.

        Every downloaded file lands under ``dest_dir`` at its dandiset
        path and is recorded in ``manifest_path`` with sha256; the
        dandiset license + embargo status come from the version
        metadata (EVAL 1.6 partial-download rule).
        """
        dest_dir = Path(dest_dir)
        manifest_path = Path(manifest_path)
        if assets is None:
            assets = self.list_assets()
        sessions = self.filter_lc_sessions([a["path"] for a in assets])
        by_path = {a["path"]: a for a in assets}

        files: list[dict[str, Any]] = []
        for lc in lc_types:
            for path in sessions.get(lc, []):
                asset = by_path[path]
                dest = dest_dir / path
                got = self._downloader(self.asset_download_url(asset["asset_id"]), dest)
                files.append(
                    {
                        "lc_type": lc,
                        "asset_id": asset["asset_id"],
                        "path": path,
                        "local_name": str(dest.relative_to(dest_dir)),
                        "bytes": got["bytes"],
                        "sha256": got["sha256"],
                        "download_url": self.asset_download_url(asset["asset_id"]),
                        "downloaded_utc": utcnow(),
                    }
                )
        meta = self.dandiset_metadata()
        manifest = {
            "dandiset": self.dandiset,
            "version": self.version,
            "api": self.base_url,
            "doi": meta.get("doi"),
            "license": meta.get("license"),
            "embargo_status": meta.get("embargo_status"),
            "access_status": meta.get("access_status"),
            "citation": meta.get("citation"),
            "partial_download": True,
            "note": (
                "LC-silencing sessions only - the dandiset's behavior/"
                "imaging subset for the T-REAL-1/2 panel; subjects follow "
                "sub-<driver>-fly<n> per configs/stim_library_v1.yaml"
            ),
            "requested_lc_types": sorted(lc_types),
            "discovered_lc_types": self.last_filter.get("discovered_lc_types", []),
            "downloaded_utc": utcnow(),
            "code": {"git_sha": git_sha(), "generator": "flygpt real_evals.py"},
            "files": files,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest


# ---------------------------------------------------------------------------
# T-REAL-1: silencing-direction benchmark (EVAL 1.5 sign rule, 1.6 score)
# ---------------------------------------------------------------------------

DIRECTIONS = ("increase", "decrease", "no-effect", "ambiguous")
CHANCE_P = 1.0 / 3.0  # three-direction outcomes -> binomial null rate


def _resolve_types(bundle: GraphBundle, types: list[str]) -> np.ndarray:
    type_arr = bundle.type_array()
    return np.flatnonzero(np.isin(type_arr, list(types))).astype(np.uint32)


def silencing_predictions(
    model_fn: Callable,
    lc_panel: dict[str, list[str]],
    bundle: GraphBundle,
    *,
    outcomes: dict[str, str] | None = None,
    seeds: list[int] | None = None,
    n_seeds: int = 3,
    noise_floor_hz: float = 1.0,
    control_deltas: Any | None = None,
    baseline_correct: int | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Score predicted silencing directions over the 23-type LC panel.

    ``model_fn(lc_type, silenced_nodes, seed, bundle) -> float`` returns
    the predicted rate change (Hz) on the dandiset's readout population
    with that driver line's MaleCNS types silenced vs its reference
    stimulus - positive = behavior/dynamics increase. Per LC type the
    EVAL-1.5 sign rule (``common.response_sign``) maps the per-seed
    deltas to a direction; ``outcomes`` injects the published direction
    per driver (increase/decrease/no-effect). Per-type significance =
    ``common.binomial_test`` on correct per-seed directions vs the 1/3
    three-way null; the panel is BH-FDR corrected at q<=0.10
    (exploratory). ``control_deltas`` optionally supplies unstimulated
    control |rate changes| for ``common.noise_floor`` - else
    ``noise_floor_hz`` is used as the experiment floor.
    """
    common = _common()
    if seeds is None:
        seeds = [
            seed_from_config({"eval": TEST_SILENCING, "rep": i})
            for i in range(n_seeds)
        ]
    floor = (
        float(common.noise_floor(control_deltas))
        if control_deltas is not None
        else float(noise_floor_hz)
    )
    outcomes = outcomes or {}
    per_lc: dict[str, Any] = {}
    scored: list[str] = []
    for lc, types in lc_panel.items():
        silenced = _resolve_types(bundle, list(types))
        deltas = np.asarray(
            [
                float(
                    model_fn(
                        lc_type=lc,
                        silenced_nodes=silenced,
                        seed=seed,
                        bundle=bundle,
                    )
                )
                for seed in seeds
            ]
        )
        predicted = common.response_sign(deltas, floor)
        per_seed = [
            "increase" if d > 0 else ("decrease" if d < 0 else "no-effect")
            for d in deltas
        ]
        outcome = outcomes.get(lc)
        correct = predicted == outcome if outcome is not None else None
        if outcome is not None:
            scored.append(lc)
        k = int(sum(d == outcome for d in per_seed)) if outcome is not None else 0
        p = None
        if outcome is not None and seeds:
            p = float(common.binomial_test(k, len(seeds), CHANCE_P)["p"])
        mean, lo, hi = common.ci_mean(deltas)
        per_lc[lc] = {
            "malecns_types": list(types),
            "n_silenced_nodes": int(silenced.size),
            "deltas_hz": deltas.tolist(),
            "mean_delta_hz": mean,
            "ci_lo": lo,
            "ci_hi": hi,
            "per_seed_direction": per_seed,
            "seed_correct_rate": (k / len(seeds)) if seeds else None,
            "predicted": predicted,
            "outcome": outcome,
            "correct": correct,
            "binomial_p": p,
        }
    n_scored = len(scored)
    n_correct = int(sum(per_lc[lc]["correct"] for lc in scored))
    pvals = np.array(
        [per_lc[lc]["binomial_p"] for lc in scored], dtype=np.float64
    )
    fdr = (
        common.bh_fdr(pvals, 0.10)
        if pvals.size
        else {"reject": np.zeros(0, dtype=bool), "p_adj": np.zeros(0), "q": 0.10}
    )
    # a type is "significantly predicted" only when the rejection is in
    # the correct direction (two-sided p can also reject when wrong)
    rejected_lcs = [
        lc
        for lc, r in zip(scored, np.asarray(fdr["reject"]).tolist(), strict=False)
        if r and (per_lc[lc]["seed_correct_rate"] or 0.0) > CHANCE_P
    ]
    score = (n_correct / n_scored) if n_scored else None
    margin = (n_correct - baseline_correct) if baseline_correct is not None else None
    result = {
        "test_id": TEST_SILENCING,
        "n_panel": len(lc_panel),
        "n_scored": n_scored,
        "n_correct": n_correct,
        "score": score,
        "per_lc": per_lc,
        "bh_fdr": {
            "q": fdr["q"],
            "n_tests": int(np.asarray(fdr["reject"]).size),
            "p_adj": np.asarray(fdr["p_adj"]).tolist(),
            "rejected_lc_types": rejected_lcs,
        },
        "n_seeds": len(seeds),
        "seeds": [int(s) for s in seeds],
        "noise_floor_hz": floor,
        "baseline_correct": baseline_correct,
        "margin_vs_baseline_types": margin,
        "pass_rule": (
            ">=16/23 correct-direction AND >= influence-matrix baseline + 2 "
            "types; per-type binomial BH-FDR q<=0.10 (EVAL.md T-REAL-1)"
        ),
        "passed": (
            score is not None
            and n_correct >= 16
            and (margin is None or margin >= 2)
        ),
    }
    if results_dir is not None:
        write_silencing_result(results_dir, result)
    return result


# ---------------------------------------------------------------------------
# T-REAL-2: GCaMP kernel convolution + Spearman (stretch)
# ---------------------------------------------------------------------------


def gcamp_kernel(
    dt: float, tau_rise_s: float = GCAMP6F_TAU_RISE_S, tau_decay_s: float = GCAMP6F_TAU_DECAY_S
) -> np.ndarray:
    """Double-exponential GCaMP6f kernel sampled at ``dt``, unit peak.

    k(t) = exp(-t/tau_d) - exp(-t/tau_r) for t>=0, normalized to peak 1.
    Length covers ~6 decay taus so the tail is fully represented.
    """
    n = max(8, int(np.ceil(6 * tau_decay_s / dt)) + 1)
    t = np.arange(n) * dt
    k = np.exp(-t / tau_decay_s) - np.exp(-t / tau_rise_s)
    peak = k.max()
    return k / peak if peak > 0 else k


def calcium_comparison(
    pred_rates: dict[str, Any],
    rec_rates: dict[str, Any],
    dt: float,
    kernel: str = "gcamp6f",
    binarize_pred: bool = False,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Convolve predicted rate traces through the indicator kernel and
    Spearman-compare against recorded traces per aligned type.

    ``pred_rates``/``rec_rates`` map type -> equal-length 1-D traces on
    a shared dt. A type qualifies iff both traces have nonzero variance
    (EVAL 1.3). ``binarize_pred`` routes predictions through
    ``common.binarize_events`` (e=1 iff count>=1) before convolution.
    Stretch target: macro Spearman >= 0.25, reported regardless.
    """
    if kernel != "gcamp6f":
        raise ValueError(f"unknown indicator kernel {kernel!r}")
    common = _common()
    kern = gcamp_kernel(dt)
    shared = [t for t in pred_rates if t in rec_rates]
    per_type: dict[str, float] = {}
    n_qualifying = 0
    for t in sorted(shared):
        pred = np.asarray(pred_rates[t], dtype=np.float64)
        rec = np.asarray(rec_rates[t], dtype=np.float64)
        n = min(pred.size, rec.size)
        pred, rec = pred[:n], rec[:n]
        if binarize_pred:
            pred = common.binarize_events(pred).astype(np.float64)
        conv = np.convolve(pred, kern)[:n]
        if n < 2 or conv.std() == 0 or rec.std() == 0:
            continue
        n_qualifying += 1
        res = common.rate_spearman(
            rec.reshape(-1, 1), conv.reshape(-1, 1)
        )
        per_type[t] = float(res["per_type"]["0"])
    macro = float(np.mean(list(per_type.values()))) if per_type else None
    result = {
        "test_id": TEST_CALCIUM,
        "kernel": {
            "name": kernel,
            "tau_rise_s": GCAMP6F_TAU_RISE_S,
            "tau_decay_s": GCAMP6F_TAU_DECAY_S,
            "dt": dt,
            "n_taps": int(kern.size),
        },
        "n_aligned_types": len(shared),
        "n_qualifying": n_qualifying,
        "per_type_spearman": per_type,
        "macro_spearman": macro,
        "threshold": 0.25,
        "stretch": True,
        "pass_rule": "Spearman >= 0.25 on aligned types (stretch - reported regardless)",
        "passed": (macro is not None and macro >= 0.25),
    }
    if results_dir is not None:
        write_calcium_result(results_dir, result)
    return result


# ---------------------------------------------------------------------------
# T-REAL-3: song ethogram vs SongExplorer reference
# ---------------------------------------------------------------------------


def song_eval(
    song_stats: dict[str, Any],
    reference: dict | Path | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """T-REAL-3: score SongDecoder ethogram stats vs the frozen ranges."""
    from flygpt.behavior import compare_ethogram

    if reference is None:
        reference = SONG_REFERENCE
    comparison = compare_ethogram(song_stats, reference)
    result = {
        "test_id": TEST_SONG,
        **comparison,
    }
    if results_dir is not None:
        write_song_result(results_dir, result)
    return result


# ---------------------------------------------------------------------------
# results/ writers (schema.write_result envelope)
# ---------------------------------------------------------------------------


def write_silencing_result(results_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    return write_result(
        Path(results_dir),
        TEST_SILENCING,
        {
            "score_at_least_16_23": result["n_correct"] >= 16,
            "margin_over_baseline_ge_2": (
                result["margin_vs_baseline_types"] is None
                or result["margin_vs_baseline_types"] >= 2
            ),
            "bh_fdr_q": result["bh_fdr"]["q"],
        },
        inputs={"n_seeds": result["n_seeds"], "noise_floor_hz": result["noise_floor_hz"]},
        metrics={
            "score": result["score"],
            "n_correct": result["n_correct"],
            "n_panel": result["n_panel"],
        },
        passed=result["passed"],
        extra={"per_lc": result["per_lc"], "bh_fdr": result["bh_fdr"]},
    )


def write_calcium_result(results_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    return write_result(
        Path(results_dir),
        TEST_CALCIUM,
        {"macro_spearman_ge_0.25": result["passed"], "stretch": True},
        inputs={"kernel": result["kernel"], "dt": result["kernel"]["dt"]},
        metrics={
            "macro_spearman": result["macro_spearman"],
            "n_qualifying": result["n_qualifying"],
        },
        passed=result["passed"],
        extra={"per_type_spearman": result["per_type_spearman"]},
    )


def write_song_result(results_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    return write_result(
        Path(results_dir),
        TEST_SONG,
        {
            "n_inside_ge_4_of_5": result["n_inside"] >= 4,
            "statistics": {
                k: {"value": v["value"], "inside": v["inside"], "ks": v["ks"]}
                for k, v in result["statistics"].items()
            },
        },
        metrics={"n_inside": result["n_inside"], "n_total": result["n_total"]},
        passed=result["passed"],
    )


# ---------------------------------------------------------------------------
# run + main
# ---------------------------------------------------------------------------


def sim_silencing_model(bundle: GraphBundle, cfg: Any | None = None) -> Callable:
    """LIF-sim predictor for T-REAL-1 (EVAL 1.6 documented substitute).

    Protocol: a reference sensory context (100 Hz Poisson on every
    resolved LC-family + ORN type, 1 s, post-warmup) runs once with and
    once without the driver's population silenced; the predicted
    direction is the mean-rate change on the descending_neuron readout
    population (visual-motor bottleneck). This is the zero-parameter
    prediction; a trained FlyGPT checkpoint substitutes the same
    signature when one exists.
    """

    def _readout_nodes() -> np.ndarray:
        sup = bundle.column("superclass") if "superclass" in bundle.nodes.columns else None
        if sup is None:
            return np.arange(bundle.n_nodes)
        return np.flatnonzero(sup == "descending_neuron")

    def predict(lc_type: str, silenced_nodes: np.ndarray, seed: int, bundle: GraphBundle) -> float:
        from flygpt.sim_backend import SimConfig, simulate

        lc_types = sorted(
            {
                t
                for t in bundle.type_array()
                if t.startswith(("LC", "LPLC", "LLPC", "LPC"))
            }
        )
        ctx_nodes = np.flatnonzero(np.isin(bundle.type_array(), lc_types) |
                                   np.char.startswith(bundle.type_array().astype(str), "ORN_"))
        sil_set = {int(n) for n in silenced_nodes}
        ctx = np.asarray([n for n in ctx_nodes if int(n) not in sil_set], dtype=np.int64)
        cfg_eff = cfg or SimConfig(seed=int(seed), warmup_steps=5)
        program = {
            "drives": [
                {"nodes": ctx.tolist(), "rate_hz": 100.0, "window": [0.0, 1.0],
                 "envelope": "poisson"}
            ],
            "silences": [{"nodes": [int(n) for n in silenced_nodes], "window": [0.0, 1.0]}]
            if silenced_nodes.size
            else [],
            "duration_s": 1.0,
            "dt": getattr(cfg_eff, "dt", 0.02),
        }
        base = dict(program)
        base["silences"] = []
        r_base = simulate(bundle, base, cfg_eff)
        r_sil = simulate(bundle, program, cfg_eff)
        rd = _readout_nodes()
        span = max(r_base.n_steps, 1)
        return float(
            np.bincount(r_sil.events_node[np.isin(r_sil.events_node, rd)], minlength=1).sum()
            / span
            - np.bincount(r_base.events_node[np.isin(r_base.events_node, rd)], minlength=1).sum()
            / span
        ) / getattr(cfg_eff, "dt", 0.02)

    return predict


def _na(results_dir: Path, test_id: str, reason: str) -> dict[str, Any]:
    from flygpt.schema import write_result

    return write_result(
        Path(results_dir),
        test_id,
        checks={"runnable": {"passed": False, "detail": reason}},
        passed=None,
        extra={"status": "na", "reason": reason},
    )


def run_t_real_1(
    results_dir: Path = REPO_ROOT / "results",
    *,
    model_fn: Callable | None = None,
    bundle: GraphBundle | None = None,
    outcomes: dict[str, str] | None = None,
    graph_dir: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """T-REAL-1 dispatcher entry: LC silencing directions vs published.

    Needs a predictor ``model_fn`` (trained checkpoint or the LIF sim
    wrapper) + a graph bundle + the published outcome table. When the
    graph is on disk and ``model_fn`` is omitted, the sim itself is the
    predictor (documented model-substitute per EVAL 1.6).
    """
    if bundle is None and graph_dir is not None:
        from flygpt.schema import load_graph_bundle

        bundle = load_graph_bundle(Path(graph_dir))
    if model_fn is None and bundle is not None:
        model_fn = sim_silencing_model(bundle)
    if model_fn is None or bundle is None:
        return _na(
            Path(results_dir),
            TEST_SILENCING,
            "needs model_fn + graph bundle (checkpoint or LIF predictor) "
            "and evals/silencing_outcomes_v1.json published directions",
        )
    if outcomes is None:
        oc_path = Path(__file__).resolve().parents[3] / "evals" / "silencing_outcomes_v1.json"
        outcomes = json.loads(oc_path.read_text())["outcomes"] if oc_path.exists() else None
    if outcomes is None:
        return _na(
            Path(results_dir),
            TEST_SILENCING,
            "evals/silencing_outcomes_v1.json absent - published outcomes "
            "not recorded yet",
        )
    res = silencing_predictions(
        model_fn, load_lc_panel(), bundle, outcomes=outcomes, results_dir=None
    )
    # return the schema'd write_result dict - the dispatcher keys pass/
    # status off "pass", which only the written envelope carries
    return write_silencing_result(Path(results_dir), res)


def run_t_real_2(
    results_dir: Path = REPO_ROOT / "results",
    *,
    pred_rates: dict[str, Any] | None = None,
    rec_rates: dict[str, Any] | None = None,
    dt: float = 0.02,
    **_: Any,
) -> dict[str, Any]:
    """T-REAL-2 dispatcher entry: calcium Spearman vs recorded activity."""
    if pred_rates is None or rec_rates is None:
        return _na(
            Path(results_dir),
            TEST_CALCIUM,
            "needs pred_rates + rec_rates artifacts (DANDI 000951 partial "
            "download via `python -m flygpt.evals.real_evals --download ...` "
            "plus a run of the model/sim over matched stimuli)",
        )
    res = calcium_comparison(pred_rates, rec_rates, dt, results_dir=None)
    return write_calcium_result(Path(results_dir), res)


def run_t_real_3(
    results_dir: Path = REPO_ROOT / "results",
    *,
    song_stats: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """T-REAL-3 dispatcher entry: song ethogram vs SongExplorer reference."""
    if song_stats is None:
        return _na(
            Path(results_dir),
            TEST_SONG,
            "needs song_stats from a courtship-slice SongDecoder.ethogram "
            "run (behavior.SongDecoder over a dt=5ms corpus slice)",
        )
    res = song_eval(song_stats, results_dir=None)
    return write_song_result(Path(results_dir), res)


def run(
    model_fn: Callable | None = None,
    bundle: GraphBundle | None = None,
    outcomes: dict[str, str] | None = None,
    pred_rates: dict[str, Any] | None = None,
    rec_rates: dict[str, Any] | None = None,
    song_stats: dict[str, Any] | None = None,
    dt: float = 0.02,
    results_dir: Path = REPO_ROOT / "results",
) -> dict[str, Any]:
    """Run whichever T-REAL evals have their inputs provided."""
    out: dict[str, Any] = {}
    if model_fn is not None and bundle is not None:
        out[TEST_SILENCING] = silencing_predictions(
            model_fn, load_lc_panel(), bundle, outcomes=outcomes, results_dir=results_dir
        )
    if pred_rates is not None and rec_rates is not None:
        out[TEST_CALCIUM] = calcium_comparison(
            pred_rates, rec_rates, dt, results_dir=results_dir
        )
    if song_stats is not None:
        out[TEST_SONG] = song_eval(song_stats, results_dir=results_dir)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list-assets", action="store_true",
                    help="list dandiset assets + discovered LC sessions")
    ap.add_argument("--download", type=str, default="",
                    help="comma-separated LC drivers to download (partial subset)")
    ap.add_argument("--dest", type=Path, default=DANDI_RAW_DIR)
    ap.add_argument("--manifest", type=Path, default=DANDI_MANIFEST)
    args = ap.parse_args(argv)

    client = DandiClient()
    if args.list_assets:
        assets = client.list_assets()
        sessions = client.filter_lc_sessions([a["path"] for a in assets])
        print(f"{len(assets)} assets; LC sessions discovered: {len(sessions)}/23")
        for lc, paths in sorted(sessions.items()):
            print(f"  {lc:8s} {len(paths)} file(s)")
        if client.last_filter["n_unmatched"]:
            print(f"  unmatched paths: {client.last_filter['n_unmatched']}")
        return 0
    if args.download:
        lcs = args.download.split(",")
        manifest = client.download_subset(lcs, args.dest, args.manifest)
        print(
            f"downloaded {len(manifest['files'])} files for {lcs}; "
            f"manifest -> {args.manifest}"
        )
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
