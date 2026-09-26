# FlyGPT Phase-4 session driver - BANC female connectome + sex conditioning.
#
# Thin wrapper around src/flygpt entry points. Downloads the public BANC
# deposit (Lee-lab bucket, CC-BY-4.0), builds the female CSR graph at the
# matching >=5 synapse cutoff, produces the male<->female type alignment
# (curated malecns/manc columns + reviewed NBLAST matches), the
# sex-conditioned vocab + female mask in male space, and a female corpus
# slice over homologous stimuli.
#
# Env vars:
#   FLYGPT_REPO      git URL or local path to install
#   FLYGPT_WORK      working dir
#   FLYGPT_BACKEND   sim backend for the female corpus slice

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

    from flygpt.banc import main as banc_main

    rc = banc_main([
        "--config", str(Path(repo) / "configs" / "banc_v1.yaml"),
        "--raw-dir", str(work / "data" / "raw" / "banc"),
        "--graph-dir", str(work / "data" / "graph_banc"),
        "--provenance-dir", str(work / "provenance"),
        "--results-dir", str(work / "results"),
        "--download", "--build", "--align",
        "--backend", os.environ.get("FLYGPT_BACKEND", "numpy"),
    ])
    print("[driver] Phase 4 complete - alignment at evals/alignment_v1.json, "
          "provenance at provenance/banc_*.json", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
