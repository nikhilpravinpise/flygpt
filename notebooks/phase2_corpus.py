# FlyGPT Phase-2 session driver - corpus generation (burst GPU / Kaggle drip).
#
# Thin wrapper around src/flygpt entry points. Generates corpus shards per
# configs/corpus_v1.yaml: immutable per-task shards, hash-derived seeds,
# per-shard provenance, dead-letter retry, checkpoint-to-remote on a clock
# far under the provider's session cap. Kill the instance at job end.
#
# Env vars:
#   FLYGPT_REPO      git URL or local path to install
#   FLYGPT_WORK      working dir
#   FLYGPT_BACKEND   sim backend (jax on GPU/TPU, numpy on CPU)
#   FLYGPT_GPU_CLASS recorded gpu class (e.g. a100, tpu-v3-8, p100) - REQUIRED
#                    for shards: never auto-guessed
#   FLYGPT_SLICES    comma-separated corpus_v1 slices (default: main)
#   FLYGPT_MAX_SHARDS cap shards this session (checkpoint-resume pattern)

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

    gpu_class = os.environ.get("FLYGPT_GPU_CLASS")
    if not gpu_class:
        raise SystemExit("FLYGPT_GPU_CLASS unset - shard manifests must record "
                         "the real device class, never a guess")

    _pip_install(repo)

    from flygpt.corpus import run_corpus

    slices = os.environ.get("FLYGPT_SLICES", "main").split(",")
    max_shards = int(os.environ.get("FLYGPT_MAX_SHARDS", "0") or 0)
    shards = run_corpus(
        graph_dir=work / "data" / "graph",
        corpus_config=Path(repo) / "configs" / "corpus_v1.yaml",
        stim_config=Path(repo) / "configs" / "stim_library_v1.yaml",
        work_dir=work / "data" / "corpus",
        slices=slices,
        backend=os.environ.get("FLYGPT_BACKEND", "numpy"),
        gpu_class=gpu_class,
        max_shards=max_shards or None,
    )
    print(f"[driver] wrote {len(shards)} shards under {work/'data'/'corpus'} - "
          f"push provenance/ + manifests to the canonical repo.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
