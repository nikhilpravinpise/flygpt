# FlyGPT Phase-1 session driver - the pilot gate (Kaggle/Colab/any box).
#
# Thin wrapper around src/flygpt entry points (notebooks/README.md rule).
# Runs: graph load -> stimulus library resolution -> pilot trial set ->
# health metrics -> Shiu spot-checks -> results/pilot_report.json.
# The pilot is the single most important gate in the project: no pilot
# pass, no scale spend.
#
# Env vars:
#   FLYGPT_REPO      git URL or local path to install (default: this repo)
#   FLYGPT_WORK      working dir (default: cwd)
#   FLYGPT_BACKEND   sim backend (default: numpy; jax when [model] installed)
#   FLYGPT_GPU_CLASS recorded gpu class label (default: cpu)

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _pip_install(repo: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", repo])


def main() -> int:
    repo = os.environ.get("FLYGPT_REPO", ".")
    work = Path(os.environ.get("FLYGPT_WORK", ".")).resolve()
    work.mkdir(parents=True, exist_ok=True)

    _pip_install(repo)

    import yaml

    from flygpt.corpus import run_pilot
    from flygpt.schema import load_graph_bundle
    from flygpt.stimuli import load_stim_config

    cfg = yaml.safe_load((Path(repo) / "configs" / "corpus_v1.yaml").read_text())
    stim_cfg = load_stim_config(Path(repo) / "configs" / "stim_library_v1.yaml")
    bundle = load_graph_bundle(work / "data" / "graph", cutoff=5)
    sim_cfg = yaml.safe_load(
        (Path(repo) / "configs" / "sim_v1.yaml").read_text()
    ) or {}
    sim_cfg["backend"] = os.environ.get("FLYGPT_BACKEND", "numpy")
    sim_cfg["gpu_class"] = os.environ.get("FLYGPT_GPU_CLASS", "cpu")
    model = sim_cfg.get("model") or {}
    model["backend"] = sim_cfg["backend"]
    model["gpu_class"] = sim_cfg["gpu_class"]
    sim_cfg["model"] = model

    results = work / "results"
    report = run_pilot(
        bundle,
        None,  # default LIBRARY
        cfg,
        work / "data" / "corpus" / "pilot",
        results,
        stim_cfg=stim_cfg,
        sim_cfg=sim_cfg,
    )
    gate = report.get("checks", {}).get("gate", {})
    print(f"[driver] pilot gate: {gate} - full report at {results/'pilot_report.json'}",
          flush=True)
    return 0 if gate.get("pass", False) else 1


if __name__ == "__main__":
    sys.exit(main())
