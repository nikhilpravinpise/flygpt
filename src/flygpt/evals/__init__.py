"""Evaluation battery package.

Each ``evals/<family>_*.py`` module implements ``run_t_*`` runners that
write ``results/<test-id>.json`` in the results_v1 envelope
(``flygpt.schema.write_result``). This module is the dispatch surface:

    python -m flygpt.evals --results-dir results/ --graph-dir data/graph/

``--only T-SIM-1,T-CORPUS-4`` restricts to selected test ids. Every
runner lazily imports its pipeline dependencies so the package imports
on a minimal install; a runner that cannot find its inputs records a
documented NA result rather than silently passing.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

# test-id -> (module, runner name). Runners are discovered lazily; a
# missing module reports a documented skip, not a crash.
RUNNERS: dict[str, tuple[str, str]] = {
    "T-SIM-1": ("sim_checks", "run_t_sim_1"),
    "T-SIM-2": ("sim_checks", "run_t_sim_2"),
    "T-SIM-3": ("sim_checks", "run_t_sim_3"),
    "T-SIM-4": ("sim_checks", "run_t_sim_4"),
    "T-SIM-5": ("sim_checks", "run_t_sim_5"),
    "T-SIM-6": ("sim_checks", "run_t_sim_6"),
    "T-SIM-7": ("sim_checks", "run_t_sim_7"),
    "T-SIM-8": ("sim_checks", "run_t_sim_8"),
    "T-CORPUS-1": ("corpus_checks", "run_t_corpus_1"),
    "T-CORPUS-2": ("corpus_checks", "run_t_corpus_2"),
    "T-CORPUS-3": ("corpus_checks", "run_t_corpus_3"),
    "T-CORPUS-4": ("corpus_checks", "run_t_corpus_4"),
    "T-CORPUS-5": ("corpus_checks", "run_t_corpus_5"),
    "T-CORPUS-6": ("corpus_checks", "run_t_corpus_6"),
    "T-CORPUS-7": ("corpus_checks", "run_t_corpus_7"),
    "T-TOK-1": ("tok_checks", "run_t_tok_1"),
    "T-TOK-2": ("tok_checks", "run_t_tok_2"),
    "T-TOK-3": ("tok_checks", "run_t_tok_3"),
    "T-TOK-4": ("tok_checks", "run_t_tok_4"),
    "T-TOK-5": ("tok_checks", "run_t_tok_5"),
    "T-STIM-1": ("stim_checks", "run_t_stim_1"),
    "T-STIM-2": ("stim_checks", "run_t_stim_2"),
    "T-STIM-3": ("stim_checks", "run_t_stim_3"),
    "T-STIM-4": ("stim_checks", "run_t_stim_4"),
    "T-STIM-5": ("stim_checks", "run_t_stim_5"),
    "T-EMB-1": ("emb_evals", "run_t_emb_1"),
    "T-EMB-2": ("emb_evals", "run_t_emb_2"),
    "T-EMB-3": ("emb_evals", "run_t_emb_3"),
    "T-EMB-4": ("emb_evals", "run_t_emb_4"),
    "T-MODEL-1": ("model_evals", "run_t_model_1"),
    "T-MODEL-2": ("model_evals", "run_t_model_2"),
    "T-MODEL-3": ("model_evals", "run_t_model_3"),
    "T-MODEL-4": ("model_evals", "run_t_model_4"),
    "T-MODEL-5": ("model_evals", "run_t_model_5"),
    "T-MODEL-6": ("model_evals", "run_t_model_6"),
    "T-MODEL-7": ("model_evals", "run_t_model_7"),
    "T-MODEL-8": ("model_evals", "run_t_model_8"),
    "T-MODEL-9": ("model_evals", "run_t_model_9"),
    "T-MODEL-10": ("model_evals", "run_t_model_10"),
    "T-MODEL-11": ("model_evals", "run_t_model_11"),
    "T-SEX-1": ("sex_evals", "run_t_sex_1"),
    "T-SEX-2": ("sex_evals", "run_t_sex_2"),
    "T-SEX-3": ("sex_evals", "run_t_sex_3"),
    "T-SEX-4": ("sex_evals", "run_t_sex_4"),
    "T-SEX-5": ("sex_evals", "run_t_sex_5"),
    "T-SEX-6": ("sex_evals", "run_t_sex_6"),
    "T-REAL-1": ("real_evals", "run_t_real_1"),
    "T-REAL-2": ("real_evals", "run_t_real_2"),
    "T-REAL-3": ("real_evals", "run_t_real_3"),
    "T-SHIP-2": ("ship_checks", "run_t_ship_2"),
    "T-SHIP-3": ("ship_checks", "run_t_ship_3"),
    "T-SHIP-4": ("ship_checks", "run_t_ship_4"),
    "T-SHIP-5": ("ship_checks", "run_t_ship_5"),
    "T-SHIP-9": ("ship_checks", "run_t_ship_9"),
    "T-SHIP-10": ("ship_checks", "run_t_ship_10"),
    "T-SHIP-1": ("ship_checks", "run_t_ship_1"),
    "T-SHIP-6": ("ship_checks", "run_t_ship_6"),
    "T-SHIP-7": ("ship_checks", "run_t_ship_7"),
    "T-SHIP-8": ("ship_checks", "run_t_ship_8"),
}


def run_test(
    test_id: str,
    *,
    results_dir: Path,
    graph_dir: Path | None = None,
    corpus_dir: Path | None = None,
    vocab_dir: Path | None = None,
    banc_graph_dir: Path | None = None,
    cfg: Any = None,
    index: Any = None,
    library: Any = None,
    repo_root: Path | None = None,
    extra: dict | None = None,
) -> dict:
    """Dispatch a single test id to its runner; documented NA on absence."""
    from flygpt.schema import write_result

    if test_id not in RUNNERS:
        raise KeyError(f"unknown test id {test_id}")
    module_name, fn_name = RUNNERS[test_id]
    try:
        module = importlib.import_module(f"flygpt.evals.{module_name}")
        runner = getattr(module, fn_name)
    except Exception as exc:  # missing module or dep -> recorded NA
        return write_result(
            results_dir,
            test_id,
            checks={"runnable": {"passed": False, "detail": f"runner unavailable: {exc}"}},
            passed=None,
            extra={"status": "na", "reason": str(exc)},
        )
    import inspect

    # Per-family input registry: resolved lazily so absent artifacts
    # produce documented NA rather than crashes or silent passes.
    def _bundle():
        if graph_dir is None:
            return None
        from flygpt.schema import load_graph_bundle

        gd = Path(graph_dir)
        if not (gd / "nodes.parquet").exists():
            return None
        return load_graph_bundle(gd)

    def _vocab():
        if vocab_dir is None:
            return None
        try:
            from flygpt.tokenize import Vocab

            vd = Path(vocab_dir)
            for cand in (vd / "vocab_v1.json", vd / "vocab.json"):
                if cand.exists():
                    return Vocab.load(cand)
        except Exception:
            return None
        return None

    def _cfg():
        if cfg is not None:
            return cfg
        # The calibrated sim contract (w_syn adopted at the pilot gate) -
        # evals run at the deployment config, not the raw default.
        try:
            from flygpt.sim_backend import _cfg_from_yaml

            sim_yaml = Path("configs/sim_v1.yaml")
            if sim_yaml.exists():
                return _cfg_from_yaml(sim_yaml, backend=None, seed=None)
        except Exception:
            pass
        try:
            from flygpt.sim_backend import SimConfig

            return SimConfig()
        except Exception:
            return None

    def _banc_bundle():
        if banc_graph_dir is None:
            return None
        from flygpt.schema import load_graph_bundle

        gd = Path(banc_graph_dir)
        if not (gd / "nodes.parquet").exists():
            return None
        return load_graph_bundle(gd)

    def _index(bundle):
        try:
            from flygpt.stimuli import SensoryIndex, load_stim_config

            scfg = load_stim_config(Path("configs/stim_library_v1.yaml"))
            return SensoryIndex(bundle, scfg)
        except Exception:
            return None

    def _library():
        # The "library" argument is the corpus of generated stimulus
        # programs - reconstructed from corpus trials' program column.
        if corpus_dir is None:
            return None
        import polars as pl

        cd = Path(corpus_dir)
        progs: list[dict] = []
        for tp in sorted(cd.glob("trials-*.parquet")) + sorted(cd.glob("*.trials.parquet")):
            try:
                df = pl.read_parquet(tp, columns=["program"])
            except Exception:
                continue
            progs.extend(json.loads(p) if isinstance(p, str) else p
                         for p in df["program"].to_list())
        return progs or None

    registry: dict[str, Any] = {
        "results_dir": results_dir,
        "corpus_dir": corpus_dir,
        "graph_dir": graph_dir,
        "vocab_dir": vocab_dir,
        "repo_root": repo_root or Path("."),
        "cfg": cfg if cfg is not None else _cfg(),
        "index": index,
        "library": library if library is not None else _library(),
        "bundle": None,
        "vocab": None,
    }
    if extra:
        registry.update(extra)
    if registry["bundle"] is None:
        registry["bundle"] = _bundle()
    if registry["index"] is None:
        registry["index"] = _index(registry["bundle"])
    if registry["vocab"] is None:
        registry["vocab"] = _vocab()
    if registry["bundle"] is not None:
        registry.setdefault(
            "manifest_types", getattr(registry["bundle"], "manifest_types", None)
        )
        if hasattr(registry["bundle"], "type_array"):
            registry.setdefault("node_type", registry["bundle"].type_array())
        registry.setdefault("n_nodes", getattr(registry["bundle"], "n_nodes", None))
        try:
            from flygpt.corpus import region_labels

            registry.setdefault("node_region", region_labels(registry["bundle"]))
        except Exception:
            pass
    banc_bundle = registry.get("banc_bundle") or _banc_bundle()
    registry["banc_bundle"] = banc_bundle
    if banc_bundle is not None:
        # Wire the sex-conditioned graph-swap inputs from real bundles.
        registry.setdefault("female_edges", getattr(banc_bundle, "edges", None))
        if registry["bundle"] is not None:
            registry.setdefault("male_edges", getattr(registry["bundle"], "edges", None))

    sig = inspect.signature(runner)
    wanted = set(sig.parameters)
    if (
        ("node_token" in wanted or "events" in wanted)
        and registry["bundle"] is not None
        and registry["vocab"] is not None
    ):
        # Real tokenization inputs: per-node token map from the built
        # vocab and a short real event stream from the sim on the graph.
        try:
            from flygpt.tokenize import node_to_token

            registry.setdefault(
                "node_token", node_to_token(registry["bundle"], registry["vocab"])
            )
        except Exception:
            pass
        if "events" in wanted and registry.get("events") is None:
            try:
                import numpy as _np

                from flygpt.sim_backend import simulate

                rng = _np.random.default_rng(7)
                prog = {
                    "family_id": "tok1_probe",
                    "drives": [
                        {
                            "nodes": sorted(
                                rng.choice(
                                    registry["bundle"].n_nodes, size=64, replace=False
                                ).tolist()
                            ),
                            "rate_hz": 100.0,
                            "window": [0.0, 0.5],
                        }
                    ],
                    "n_steps": 25,
                    "dt": 0.02,
                }
                tr = simulate(registry["bundle"], prog, registry["cfg"])
                registry["events"] = (tr.events_step, tr.events_node)
                registry.setdefault("n_steps", tr.n_steps)
            except Exception:
                pass
    kwargs: dict[str, Any] = {}
    missing: list[str] = []
    for name, p in sig.parameters.items():
        if p.kind in (p.VAR_POSITIONAL,):
            continue
        if p.kind in (p.VAR_KEYWORD,):
            kwargs.update(extra or {})
            continue
        if name in registry and registry[name] is not None:
            kwargs[name] = registry[name]
        elif p.default is inspect.Parameter.empty:
            missing.append(name)
    if missing:
        return write_result(
            results_dir,
            test_id,
            checks={
                "runnable": {
                    "passed": False,
                    "detail": f"missing required inputs: {missing}",
                }
            },
            passed=None,
            extra={"status": "na", "reason": f"missing inputs: {missing}"},
        )
    return runner(**kwargs)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="flygpt-eval")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--corpus-dir", type=Path, default=None)
    parser.add_argument("--vocab-dir", type=Path, default=None)
    parser.add_argument("--banc-graph-dir", type=Path, default=None,
                        help="BANC/female bundle for sex-conditioned tests")
    parser.add_argument("--only", type=str, default="",
                        help="comma-separated test ids to run")
    parser.add_argument("--list", action="store_true")
    args, remaining = parser.parse_known_args(argv)

    if argv and argv[0] == "claims":
        from flygpt.evals import ship_checks

        return ship_checks.main(["claims", *argv[1:]])

    if args.list:
        for tid in sorted(RUNNERS):
            print(f"{tid}: flygpt.evals.{RUNNERS[tid][0]}.{RUNNERS[tid][1]}")
        return 0

    only = {t.strip() for t in args.only.split(",") if t.strip()}
    ids = [t for t in sorted(RUNNERS) if not only or t in only]
    summary: dict[str, str] = {}
    for tid in ids:
        result = run_test(
            tid,
            results_dir=args.results_dir,
            graph_dir=args.graph_dir,
            corpus_dir=args.corpus_dir,
            vocab_dir=args.vocab_dir,
            banc_graph_dir=args.banc_graph_dir,
        )
        status = ("pass" if result.get("pass") is True
                  else "fail" if result.get("pass") is False
                  else result.get("status", "na"))
        summary[tid] = status
        print(f"{tid}: {status}", flush=True)

    from flygpt.schema import utcnow

    index = {
        "artifact": "eval_battery",
        "created_utc": utcnow(),
        "results": summary,
        "n_pass": sum(1 for s in summary.values() if s == "pass"),
        "n_fail": sum(1 for s in summary.values() if s == "fail"),
        "n_na": sum(1 for s in summary.values() if s == "na"),
    }
    (args.results_dir / "eval_battery.json").parent.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "eval_battery.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"battery: {index['n_pass']} pass / {index['n_fail']} fail / "
          f"{index['n_na']} na -> {args.results_dir/'eval_battery.json'}")
    return 1 if index["n_fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
