"""Optional ZeroGPU Gradio variant of the "Probe a Fly Brain" Space.

The robust default is the static page (``space/index.html`` + canned
rollouts): zero live sim calls, zero GPU (T-SHIP-7, ``space/README.md``).
This file is the alternative HF-Space route only - a Gradio UI serving the
SAME canned rollouts, plus a ``flygpt-tiny`` forward path when a checkpoint
is mounted at ``FLYGPT_TINY_CKPT``.

Free-tier mechanics (plan.md Phase 6): all work below runs on CPU - the
documented trick is that CPU work outside a ``spaces.GPU`` function burns
zero visitor quota, so this Space costs nothing to run. ``gradio`` and the
``spaces`` SDK are OPTIONAL and lazily imported: without them this script
prints install/run instructions instead of crashing.

Usage:
    python space/app.py                  # local Gradio launch (if installed)
    python space/app.py --share
    python space/build_canned.py --all   # regenerate the canned files it serves
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

SPACE_DIR = Path(__file__).resolve().parent
CANNED_DIR = SPACE_DIR / "canned"

STIMULUS_ORDER = ["sugar", "bitter", "loom", "female_cue", "antenna_dust"]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def canned_status() -> dict[str, Any]:
    """Which canned rollouts exist - the whole demo path is offline files."""
    out = {}
    for stim in STIMULUS_ORDER:
        p = CANNED_DIR / f"{stim}.json"
        out[stim] = p.exists()
    manifest = _load_json(CANNED_DIR / "manifest.json")
    return {"files": out, "manifest": manifest is not None}


def summarize(stimulus: str) -> dict[str, Any]:
    """CPU-only response builder: canned rollout -> compact demo payload.

    This is deliberately cheap file IO + dict shaping - no GPU, no sim.
    On ZeroGPU this function runs outside any GPU function (free)."""
    obj = _load_json(CANNED_DIR / f"{stimulus}.json")
    if obj is None:
        return {
            "error": f"canned/{stimulus}.json not found",
            "fix": "run `python space/build_canned.py --all` on CPU first",
        }
    ro = obj["readout"]
    payload = {
        "stimulus": obj["stimulus"],
        "title": obj["title"],
        "population": obj["population"]["description"],
        "n_nodes": obj["n_nodes_small"],
        "window_ms": obj["window_ms"],
        "n_events": len(obj["events"]),
        "decoded": ro["decoded"],
        "decoded_label": ro["decoded_label"],
        "programs": {k: v["events"] for k, v in ro["programs"].items()},
        "provenance": {
            "backend": obj["provenance"]["backend"],
            "code_sha": obj["provenance"]["code_sha"],
            "graph_sha": obj["provenance"]["graph_sha"],
            "generated_utc": obj["provenance"]["generated_utc"],
        },
        "disclaimer": "simulated, not recorded",
    }
    if obj.get("ethogram"):
        payload["ethogram"] = obj["ethogram"]["stats"]
    return payload


def tiny_available() -> str:
    """flygpt-tiny live-inference path status (checkpoint mounted?)."""
    ckpt = os.environ.get("FLYGPT_TINY_CKPT", "")
    if not ckpt or not Path(ckpt).exists():
        return ("no flygpt-tiny checkpoint mounted (FLYGPT_TINY_CKPT unset) - "
                "serving canned rollouts only")
    try:
        from flygpt import model  # lazy: model framework is optional
    except ImportError:
        return "flygpt.model unavailable in this env - canned rollouts only"
    if not hasattr(model, "load_checkpoint"):
        return "flygpt.model has no load_checkpoint yet - canned rollouts only"
    return f"flygpt-tiny checkpoint found at {ckpt}"


def build_demo():
    """Construct the Gradio app; returns None when gradio is absent."""
    try:
        import gradio as gr
    except ImportError:
        return None

    def on_pick(stimulus: str):
        return summarize(stimulus), json.dumps(canned_status(), indent=2)

    with gr.Blocks(title="Probe a Fly Brain - FlyGPT") as demo:
        gr.Markdown(
            "# Probe a Fly Brain\n"
            "Canned LIF rollouts on connectome wiring - **simulated, not "
            "recorded**. Zero live-sim and zero GPU in this path; CPU work "
            "outside the GPU function burns zero visitor quota on ZeroGPU."
        )
        pick = gr.Dropdown(
            choices=STIMULUS_ORDER, value="sugar", label="stimulus"
        )
        out = gr.JSON(label="decoded rollout")
        status = gr.JSON(label="canned cache status")
        pick.change(on_pick, inputs=pick, outputs=[out, status])
        demo.load(on_pick, inputs=pick, outputs=[out, status])
        gr.Markdown(tiny_available())
    return demo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args(argv)

    demo = build_demo()
    if demo is None:
        print(
            "gradio is not installed - this file is the OPTIONAL Space variant.\n"
            "The default demo is static and needs nothing:\n"
            "  1. python space/build_canned.py --all\n"
            "  2. open space/index.html (or serve the repo dir)\n"
            "For the Gradio/ZeroGPU route: pip install gradio, then\n"
            "  python space/app.py\n"
            "(spaces SDK optional - CPU work outside spaces.GPU is free)",
            file=sys.stderr,
        )
        return 0
    with contextlib.suppress(ImportError):
        import spaces  # noqa: F401  ZeroGPU SDK - optional, unused on CPU
    demo.launch(share=args.share, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
