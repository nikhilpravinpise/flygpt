"""Model-card and dataset-card generators plus the T-SHIP-4/5 linters.

Public-text hygiene (Phase 6 / EVAL.md T-SHIP table, CLAIMS.md rules):

- ``model_card`` emits a claims table that is a SUBSET of the EVIDENCED
  rows of CLAIMS.md - never UNSUPPORTED/REFUTED/DOWNGRADED (T-SHIP-4).
- The control-suite table is wired from ``results/*.json`` and is the
  product: real vs rewired vs wrong-animal vs dense (the Brunton
  digital-sphinx lesson - a wrong animal's brain can puppeteer a fly
  body, so a demo without connectome-specific controls is dismissed).
- ``LIMITATIONS`` is the verbatim limitation list (the audit's
  section-1.7 objection table, hardened wording from the Eon-backlash
  review in docs/flygpt-deep-research.md 12.2) and must appear verbatim
  in every shipped model card.
- ``dataset_card`` carries "simulated, not recorded", the full
  provenance chain, and the convention choices verbatim (T-SHIP-5).
- ``check_card_claims`` is the CI linter backing T-SHIP-4;
  ``check_dataset_card`` backs T-SHIP-5.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SIMULATED_NOT_RECORDED = "simulated, not recorded"

# The verbatim limitation list - compiled from the audit's section-1.7
# objection table (docs/flygpt-deep-research.md) plus the BANC validation
# asymmetry from plan.md Phase 4. Every item must appear verbatim in the
# shipped model card (T-SHIP-4); do not paraphrase.
LIMITATIONS: tuple[str, ...] = (
    "No morphology: the model sees wiring and cell-type labels, not cell "
    "shape or cable properties.",
    "No receptor dynamics: synapses are static signed weights; receptor "
    "kinetics, adaptation, and short-term plasticity are absent.",
    "No gap junctions: electrical synapses are missing from all EM "
    "connectomes and are functionally important in the fly (shakB innexin).",
    "No non-spiking neurons: every unit is a spiking LIF cell; non-spiking "
    "physiology is unmodeled.",
    "No neuromodulators or neuropeptides: extrasynaptic signaling is "
    "pervasive in the real brain and absent here.",
    "Basal firing is zero unless driven: the unstimulated network is "
    "exactly silent; real brains carry tonic activity.",
    "The corpus is simulated, not recorded: every training trace is LIF "
    "simulator output on the wiring diagram.",
    "Single specimen: the graph is one male fly - inter-individual "
    "variability is unmodeled.",
    "Sim-fidelity ceiling: a model distilled from a simulator cannot "
    "exceed the simulator's fidelity without real-data correction.",
    "Specimen + pipeline + sex confound: 'sex-conditioned' compares "
    "different specimens from different pipelines (MaleCNS vs BANC); read "
    "it as specimen-and-pipeline-conditioned too.",
    "The BANC (female) branch is less validated: no published whole-CNS "
    "LIF validation exists for BANC; shared isomorphic circuits are "
    "cross-checked against the male corpus and published findings, and "
    "the asymmetry is stated, not hidden.",
)

# Convention choices that must appear verbatim in the dataset card
# (T-SHIP-5) - mirror configs/graph_v1.yaml + README conventions.
CONVENTION_MARKERS: tuple[str, ...] = (
    "edges kept at >=5 synapses",
    "autapse policy: keep for MaleCNS",
    "BANC excludes ~2.1% autapses at build",
    "an edge is inhibitory iff >50% of its presynaptic sites are predicted "
    "GABA or glutamate at cleft score >=50",
    "seeds derived as hash(trial config) - never hand-assigned",
    "no repeated (pre, post) pairs survive the build",
)

# The control suite - "the controls are the product" (audit 12.2). Each
# row maps a results/<test-id>.json artifact to the control it carries.
CONTROL_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("T-MODEL-3", "rewired connectome mask (degree+weight preserved) + "
                  "wrong-animal corpus (C. elegans)"),
    ("T-MODEL-4", "permuted node labels"),
    ("T-MODEL-5", "parameter-matched dense transformer"),
    ("T-MODEL-6", "BANC linear influence matrices"),
    ("T-MODEL-7", "flybrain reservoir (Trace/Readout)"),
    ("T-MODEL-10", "GNN message-passing baseline"),
    ("T-MODEL-11", "shuffled-weights corpus slice"),
)

_CLAIM_ID = re.compile(r"\bC(\d+)\b")
_CLAIM_ROW = re.compile(r"^\|\s*(C\d+)\s*\|(.*?)\|\s*(\w+)\s*\|(.*?)\|(.*?)\|\s*$")


def parse_claims(claims_md: str) -> list[dict[str, str]]:
    """Parse the CLAIMS.md register -> [{id, claim, status, artifact, notes}]."""
    rows = []
    for line in claims_md.splitlines():
        m = _CLAIM_ROW.match(line.strip())
        if not m:
            continue
        cid, claim, status, artifact, notes = (x.strip() for x in m.groups())
        if status not in ("EVIDENCED", "UNSUPPORTED", "REFUTED", "DOWNGRADED"):
            continue
        rows.append({
            "id": cid, "claim": claim, "status": status,
            "artifact": artifact, "notes": notes,
        })
    return rows


def _metrics_summary(metrics: dict[str, Any]) -> str:
    """Compact one-line rendering of a result artifact's metrics dict."""
    parts = []
    for k in sorted(metrics):
        v = metrics[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        elif isinstance(v, (int, str, bool)):
            parts.append(f"{k}={v}")
        if len(parts) >= 4:
            break
    return ", ".join(parts) if parts else "-"


def _control_rows(results_dir: Path) -> list[str]:
    rows = []
    for test_id, label in CONTROL_ARTIFACTS:
        path = results_dir / f"{test_id}.json"
        if not path.exists():
            rows.append(f"| {label} | `{test_id}` | pending | - |")
            continue
        try:
            art = json.loads(path.read_text())
        except json.JSONDecodeError:
            rows.append(f"| {label} | `{test_id}` | unreadable | - |")
            continue
        verdict = art.get("pass")
        verdict_s = {True: "PASS", False: "FAIL", None: "reported"}.get(verdict, "reported")
        rows.append(
            f"| {label} | `{test_id}` | {verdict_s} | "
            f"{_metrics_summary(art.get('metrics', {}))} |"
        )
    return rows


def model_card(
    claims_path: Path | str,
    results_dir: Path | str,
    limitations: tuple[str, ...] | list[str] | None = None,
) -> str:
    """Generate ``model_card.md`` text.

    The claims table contains ONLY EVIDENCED rows of CLAIMS.md (T-SHIP-4);
    the control suite is wired from ``results_dir/*.json``; the limitation
    list is emitted verbatim (defaults to :data:`LIMITATIONS`).
    """
    claims_path = Path(claims_path)
    results_dir = Path(results_dir)
    limitations = LIMITATIONS if limitations is None else tuple(limitations)

    rows = parse_claims(claims_path.read_text())
    evidenced = [r for r in rows if r["status"] == "EVIDENCED"]

    lines: list[str] = []
    lines += [
        "---",
        "license: cc-by-4.0",
        "tags:",
        "  - connectome",
        "  - drosophila",
        "  - neural-dynamics",
        "---",
        "",
        "# FlyGPT - model card",
        "",
        "Connectome-constrained generative model of a complete "
        "synapse-resolution animal nervous system's dynamics (MaleCNS v1.0), "
        "with a sex-conditioned branch against the female BANC connectome.",
        "",
        f"**{SIMULATED_NOT_RECORDED}**: every training trace is the output of "
        "a leaky integrate-and-fire simulator run on connectome structure - "
        "the corpus's validity ceiling is the simulator's. This is not a "
        "digital fly, not an upload, and not an embodied agent: it is a "
        "learned model of network dynamics whose claims stand or fall on "
        "held-out controls. The controls are the product.",
        "",
        "## Claims (subset of CLAIMS.md EVIDENCED rows)",
        "",
    ]
    if evidenced:
        lines += [
            "| ID | Claim | Evidence |",
            "|---|---|---|",
        ]
        for r in evidenced:
            lines.append(f"| {r['id']} | {r['claim']} | `{r['artifact']}` |")
    else:
        lines.append("_No claims are EVIDENCED yet; see CLAIMS.md._")
    lines += [
        "",
        "Claims may appear here only when their CLAIMS.md row is EVIDENCED "
        "and links a committed artifact under `results/` (T-SHIP-2/4).",
        "",
        "## Control suite - the controls are the product",
        "",
        "Real model vs each control, on identical splits. A connectome-"
        "specific control is the point of the artifact (Brunton's digital-"
        "sphinx lesson: a wrong animal's wiring can still drive a body).",
        "",
        "| Control | Artifact | Verdict | Headline metrics |",
        "|---|---|---|---|",
        *_control_rows(results_dir),
        "",
        "## Limitations (verbatim - audit section 1.7 objections)",
        "",
    ]
    lines += [f"- {lim}" for lim in limitations]
    lines += [
        "",
        "## Provenance",
        "",
        "- Claims register: `CLAIMS.md` (CI-enforced evidence links)",
        "- Eval protocol: `EVAL.md` (pre-registered thresholds)",
        "- Evidence artifacts: `results/` (results_v1 envelope)",
        "- License: code Apache-2.0; weights + derived data CC-BY-4.0",
        "",
    ]
    return "\n".join(lines)


def dataset_card(manifest: dict[str, Any] | str | Path) -> str:
    """Generate the dataset card text for a corpus manifest.

    ``manifest`` is the corpus/shard manifest dict (or a path to its JSON).
    Carries "simulated, not recorded", the full provenance chain, and the
    convention choices verbatim (T-SHIP-5).
    """
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text())
    m = dict(manifest or {})
    code = m.get("code", {})
    graph = m.get("graph", {})

    def g(*keys, default="not recorded"):
        cur = m
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    lines = [
        "---",
        "license: cc-by-4.0",
        "tags:",
        "  - connectome",
        "  - drosophila",
        "  - simulated-dynamics",
        "---",
        "",
        "# FlyGPT corpus - dataset card",
        "",
        f"**{SIMULATED_NOT_RECORDED}**: this dataset is the output of a "
        "leaky integrate-and-fire simulator run on the MaleCNS v1.0 wiring "
        "diagram. No neural activity was recorded; the validity ceiling of "
        "every trace is the simulator's fidelity.",
        "",
        "## Provenance chain",
        "",
        "| Link | Value |",
        "|---|---|",
        f"| corpus version | `{g('corpus_version')}` |",
        f"| source manifest | `{g('source_manifest')}` |",
        f"| code git_sha | `{code.get('git_sha', 'not recorded')}` |",
        f"| generator | `{code.get('generator', 'not recorded')}` |",
        f"| graph version | `{graph.get('version', 'not recorded')}` |",
        f"| graph sha256 | `{graph.get('sha256', 'not recorded')}` |",
        f"| sim config | `{g('sim_cfg')}` |",
        f"| backend | `{g('backend')}` |",
        f"| gpu class | `{g('gpu_class')}` |",
        f"| dt | `{g('dt')}` |",
        f"| synapse cutoff | `{g('cutoff')}` |",
        f"| slice / split | `{g('slice')}` / `{g('split')}` |",
        f"| created_utc | `{g('created_utc')}` |",
        "",
        "Upstream files are sha256-verified against `provenance/*.json`; "
        "the build fails closed on any checksum mismatch.",
        "",
        "## Convention choices (verbatim)",
        "",
    ]
    lines += [f"- {marker}" for marker in CONVENTION_MARKERS]
    lines += [
        "",
        "## Honesty notes",
        "",
        "- Untyped neurons are retained and route to UNK - never dropped.",
        "- Degenerate trials are a labeled class, never silently dropped.",
        "- FlyWire (CC BY-NC 4.0) is excluded; BANC covers the female side.",
        "- Attribution chain: Berg et al. Cell 2026 (MaleCNS), Eckstein et "
        "al. Cell 2024 (NT predictions), Bates et al. Nature 2026 (BANC).",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Linters - the CI side of T-SHIP-4/5
# ---------------------------------------------------------------------------


def card_claim_ids(card_text: str) -> list[str]:
    """Claim IDs (``C<num>``) referenced anywhere in a card's text."""
    return sorted({f"C{m}" for m in _CLAIM_ID.findall(card_text)}, key=_cid_key)


def _cid_key(cid: str) -> int:
    return int(cid[1:])


def check_card_claims(card_text: str, claims_md: str) -> dict[str, Any]:
    """T-SHIP-4 linter.

    Extracts every claim ID referenced by the card and checks it is a
    subset of the EVIDENCED rows of ``claims_md``; also verifies the
    verbatim limitation list is present. Returns a report dict with
    ``passed`` = both checks clean.
    """
    rows = parse_claims(claims_md)
    evidenced = sorted((r["id"] for r in rows if r["status"] == "EVIDENCED"), key=_cid_key)
    by_status = {r["id"]: r["status"] for r in rows}
    referenced = card_claim_ids(card_text)
    out_of_subset = [c for c in referenced if c not in evidenced]
    non_evidenced_status = {c: by_status.get(c, "not_in_register") for c in out_of_subset}
    missing_limits = [lim for lim in LIMITATIONS if lim not in card_text]
    return {
        "referenced_ids": referenced,
        "evidenced_ids": evidenced,
        "out_of_subset": out_of_subset,
        "out_of_subset_status": non_evidenced_status,
        "limitations_present": len(LIMITATIONS) - len(missing_limits),
        "limitations_missing": missing_limits,
        "disclaimer_present": SIMULATED_NOT_RECORDED in card_text,
        "passed": not out_of_subset and not missing_limits,
    }


def check_dataset_card(card_text: str) -> dict[str, Any]:
    """T-SHIP-5 linter: disclaimer + provenance fields + convention choices."""
    markers = {mk: mk in card_text for mk in CONVENTION_MARKERS}
    prov_fields = ["corpus version", "code git_sha", "graph sha256", "sim config",
                   "backend", "dt", "synapse cutoff", "created_utc"]
    missing_prov = [f for f in prov_fields if f not in card_text]
    return {
        "disclaimer_present": SIMULATED_NOT_RECORDED in card_text,
        "markers": markers,
        "markers_missing": [m for m, ok in markers.items() if not ok],
        "provenance_fields_missing": missing_prov,
        "passed": (
            SIMULATED_NOT_RECORDED in card_text
            and all(markers.values())
            and not missing_prov
        ),
    }


def main(argv: list[str] | None = None) -> int:
    """Emit model_card.md / dataset_card.md; lint a card against CLAIMS.md."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("model-card")
    m.add_argument("--claims", type=Path, default=Path("CLAIMS.md"))
    m.add_argument("--results-dir", type=Path, default=Path("results"))
    m.add_argument("--out", type=Path, default=Path("model_card.md"))
    d = sub.add_parser("dataset-card")
    d.add_argument("--manifest", type=Path, required=True)
    d.add_argument("--out", type=Path, default=Path("dataset_card.md"))
    c = sub.add_parser("check")
    c.add_argument("card", type=Path)
    c.add_argument("--claims", type=Path, default=Path("CLAIMS.md"))
    c.add_argument("--dataset", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "model-card":
        args.out.write_text(model_card(args.claims, args.results_dir))
        print(f"wrote {args.out}")
        return 0
    if args.cmd == "dataset-card":
        args.out.write_text(dataset_card(args.manifest))
        print(f"wrote {args.out}")
        return 0
    text = args.card.read_text()
    report = (check_dataset_card(text) if args.dataset
              else check_card_claims(text, args.claims.read_text()))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
