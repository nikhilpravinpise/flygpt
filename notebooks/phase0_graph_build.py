# FlyGPT Phase-0 session driver - Kaggle/Colab-compatible script form.
#
# Thin wrapper around src/flygpt entry points (per notebooks/README.md: no
# pipeline logic lives here). Runs the canonical ingest + CSR graph build
# inside a remote session, then stages artifacts for upload to the HF
# dataset repo (the canonical artifact home) or Jio Drive ferry.
#
# On Kaggle: paste into a notebook cell, or keep as a script cell with the
# "internet on" toggle enabled. CPU-only is fine - Phase 0 needs no GPU.
#
# Env vars understood:
#   FLYGPT_REPO      git URL or local path to install (default: this repo)
#   FLYGPT_WORK      working dir (default: /kaggle/working or cwd)
#   FLYGPT_ONLY      comma-separated ingest registry keys (default: all)
#   FLYGPT_SKIP_VERIFY  "1" skips input sha256 re-verification (dev only)

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

    from flygpt.graph import main as graph_main
    from flygpt.ingest import main as ingest_main

    raw = work / "data" / "raw"
    prov = work / "provenance"
    results = work / "results"
    graph_dir = work / "data" / "graph"

    argv = ["--dest", str(raw), "--manifest", str(prov / "malecns_v1.json")]
    if os.environ.get("FLYGPT_ONLY"):
        argv += ["--only", os.environ["FLYGPT_ONLY"]]
    ingest_main(argv)

    argv = [
        "--config", str(Path(repo) / "configs" / "graph_v1.yaml"),
        "--raw-dir", str(raw),
        "--graph-dir", str(graph_dir),
        "--provenance-dir", str(prov),
        "--results-dir", str(results),
        "--manifest", str(prov / "malecns_v1.json"),
    ]
    if os.environ.get("FLYGPT_SKIP_VERIFY") == "1":
        argv.append("--skip-verify")
    graph_main(argv)

    print(f"[driver] artifacts under {work} - push provenance/ + results/ "
          f"to the canonical repo; data/graph regenerates from them.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
