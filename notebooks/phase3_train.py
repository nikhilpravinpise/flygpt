# FlyGPT Phase-3 session driver - model training + control suite.
#
# Thin wrapper around src/flygpt entry points. Trains the dual-axis
# connectome transformer (JAX/Flax - the Kaggle TPU drip framework) with
# checkpoint-to-remote at intervals far under session caps; resume is the
# normal path. Control suite (dense/rewired/permuted/GNN/noedge/shuffled)
# runs as the same entry point with FLYGPT_CONTROL.
#
# Env vars:
#   FLYGPT_REPO      git URL or local path to install
#   FLYGPT_WORK      working dir
#   FLYGPT_CONTROL   control name (dense|gnn|rewired|permuted|noedge|
#                    shuffled|wrong_animal) - empty = real model
#   FLYGPT_SEED      training seed (>=3 per EVAL; enumerated not picked)
#   FLYGPT_DEV       "1" runs the small dev config on synthetic data
#   FLYGPT_BATCHES   pre-materialized batch dir (batch_*.npz); default
#                    streams straight from data/corpus shards

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TRAINABLE_CONTROLS = {
    "dense", "gnn", "rewired", "permuted", "noedge", "shuffled",
    "wrong_animal",
}
# Non-trainable baselines live eval-side (linear/reservoir predictors) -
# they are measured by the T-MODEL checks, not trained here.
NON_TRAINABLE = {"linear", "reservoir"}


def _pip_install(repo: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", f"{repo}[model]"])


def _edge_set(control: str | None, bundle, cfg, seed: int):
    """control -> the edge_set the model trains against (real by default)."""
    from flygpt import controls
    from flygpt.model import edge_set_from_bundle, empty_edge_set

    if control is None or control == "gnn":
        return edge_set_from_bundle(bundle, cfg.max_in_degree)
    if control == "dense":
        return None  # dense arch ignores edges (train() allows None)
    if control == "rewired":
        return edge_set_from_bundle(controls.rewire(bundle, seed), cfg.max_in_degree)
    if control == "permuted":
        return edge_set_from_bundle(
            controls.permuted_labels(bundle, seed), cfg.max_in_degree
        )
    if control == "shuffled":
        return edge_set_from_bundle(
            controls.shuffled_weights(bundle, seed), cfg.max_in_degree
        )
    if control == "noedge":
        return empty_edge_set(bundle.n_nodes, cfg.max_in_degree)
    if control == "wrong_animal":
        from flygpt.model import pad_edge_set

        worm_es = edge_set_from_bundle(
            controls.wrong_animal_bundle(), cfg.max_in_degree
        )
        return pad_edge_set(worm_es, cfg.n_vocab or bundle.n_nodes)
    raise ValueError(f"unhandled control {control!r}")


def main() -> int:
    repo = os.environ.get("FLYGPT_REPO", ".")
    work = Path(os.environ.get("FLYGPT_WORK", ".")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    control = os.environ.get("FLYGPT_CONTROL") or None
    if control in NON_TRAINABLE:
        print(f"[driver] control {control!r} is an eval-time baseline, not "
              f"a trained model - run the T-MODEL battery instead.", flush=True)
        return 0
    if control is not None and control not in TRAINABLE_CONTROLS:
        raise SystemExit(
            f"unknown FLYGPT_CONTROL {control!r} - one of {sorted(TRAINABLE_CONTROLS)}"
        )

    _pip_install(repo)

    from flygpt.model import load_model_config, train
    from flygpt.schema import load_graph_bundle
    from flygpt.tokenize import Vocab, iter_corpus_batches, node_to_token

    dev = os.environ.get("FLYGPT_DEV") == "1"
    seed = int(os.environ.get("FLYGPT_SEED", "1"))
    cfg = load_model_config(Path(repo) / "configs" / "model_v1.yaml", dev=dev)
    cfg.seed = seed
    if control == "dense":
        cfg.arch = "dense"
    elif control == "gnn":
        cfg.arch = "gnn"

    bundle = load_graph_bundle(work / "data" / "graph", cutoff=5)

    batch_dir = os.environ.get("FLYGPT_BATCHES")
    vocab = node_token = None
    if batch_dir:
        import glob

        import numpy as np

        def _npz_iter():
            for f in sorted(glob.glob(str(Path(batch_dir) / "batch_*.npz"))):
                z = np.load(f)
                yield z["x"], z["y_events"], z["y_rates"]

        dataset = _npz_iter()
        z0 = np.load(sorted(glob.glob(str(Path(batch_dir) / "batch_*.npz")))[0])
        cfg.n_vocab = int(z0["x"].shape[-1])
    elif dev:
        from flygpt.model import synthetic_dataset

        cfg.n_vocab = bundle.n_nodes
        dataset = synthetic_dataset(
            bundle, n_trials=cfg.batch_trials * 4, T=9,
            vocab_size=cfg.n_vocab, seed=seed,
        )
    else:
        vocab = Vocab.load(work / "data" / "vocab" / "vocab_v1.json")
        node_token = node_to_token(bundle, vocab)
        cfg.n_vocab = vocab.size()

        class _CorpusDS:
            def __iter__(self):
                return iter_corpus_batches(
                    work / "data" / "corpus", vocab, node_token,
                    T=9, batch_trials=cfg.batch_trials, seed=seed,
                )

        dataset = _CorpusDS()

    # n_vocab must be settled before the edge set (wrong-animal pads to V).
    cfg.edge_set = _edge_set(control, bundle, cfg, seed)

    ckpt = train(cfg, dataset, work / "checkpoints")
    print(f"[driver] checkpoints under {ckpt} - push to remote storage on "
          f"schedule; resume-from-checkpoint is the normal path.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
