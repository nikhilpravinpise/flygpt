# FlyGPT Phase-5 session driver - eval battery + embedding artifact +
# cards + ship checks.
#
# Thin wrapper around src/flygpt entry points. Runs the full T-* battery
# over artifacts under results/, builds the embedding table
# (data/embeddings/), regenerates model/dataset cards from the claims
# subset, and writes the audit summary.
#
# Env vars:
#   FLYGPT_REPO      git URL or local path to install
#   FLYGPT_WORK      working dir
#   FLYGPT_ONLY      comma-separated test ids (default: all)

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _pip_install(repo: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[model]"])


def main() -> int:
    repo = os.environ.get("FLYGPT_REPO", ".")
    work = Path(os.environ.get("FLYGPT_WORK", ".")).resolve()
    work.mkdir(parents=True, exist_ok=True)

    _pip_install(repo)

    from flygpt.evals import main as eval_main

    argv = [
        "--results-dir", str(work / "results"),
        "--graph-dir", str(work / "data" / "graph"),
        "--corpus-dir", str(work / "data" / "corpus"),
    ]
    only = os.environ.get("FLYGPT_ONLY")
    if only:
        argv += ["--only", only]
    rc = eval_main(argv)
    print(f"[driver] eval battery rc={rc} - artifacts under {work/'results'}",
          flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
