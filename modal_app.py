"""Modal entry points - thin remote drivers over src/flygpt.

Per plan.md hard rules: every job entry point lives in ``src/flygpt/`` and
the same code path runs on Kaggle, Vultr, GCP, or Modal. This file only
provides the Modal wiring (image, volumes, timeouts); all logic is in the
package. ``modal`` is an optional dep - install with ``pip install -e
.[remote]`` or ``pip install modal``.

Phase-0 jobs:
    modal run modal_app.py::ingest       # download canonical files + manifest
    modal run modal_app.py::build_graph  # CSR graph build + manifests

Later phases (same code path as local/Kaggle - thin drivers only):
    modal run modal_app.py::pilot        # pilot gate -> results/pilot_report.json
    modal run modal_app.py::eval_battery # T-* battery over results/
    modal run modal_app.py::smoke        # CI CUDA smoke (T-SHIP-8)
    modal run modal_app.py::build_banc   # Phase-4 BANC graph + alignment

Artifacts land on a Modal volume mounted at /data so a later local
``flygpt-ingest`` run resumes/skips cleanly via sha256 provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import modal
except ImportError:  # pragma: no cover - modal is an optional remote dep
    modal = None


if modal is not None:
    app = modal.App("flygpt")

    data_vol = modal.Volume.from_name("flygpt-data", create_if_missing=True)

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy==2.5.3", "pyarrow==25.0.1", "polars==1.44.2",
                     "scipy==1.18.1", "requests==2.34.2", "tqdm==4.70.1",
                     "PyYAML==6.0.3")
        .add_local_dir("src", remote_path="/pkg/src")
        .add_local_dir("configs", remote_path="/pkg/configs")
        .add_local_file("pyproject.toml", remote_path="/pkg/pyproject.toml")
        .add_local_file("README.md", remote_path="/pkg/README.md")
        .workdir("/pkg")
        .pip_install(".")
    )

    DATA = Path("/data")

    @app.function(image=image, volumes={"/data": data_vol}, timeout=6 * 3600,
                  memory=8192)
    def ingest(only: str = ""):
        """Download canonical MaleCNS files to the volume + write manifest."""
        from flygpt.ingest import main as ingest_main

        argv = ["--dest", str(DATA / "raw"),
                "--manifest", str(DATA / "provenance" / "malecns_v1.json")]
        if only:
            argv += ["--only", only]
        rc = ingest_main(argv)
        data_vol.commit()
        return rc

    @app.function(image=image, volumes={"/data": data_vol}, timeout=6 * 3600,
                  memory=28672, cpu=4)
    def build_graph(skip_verify: bool = False, topo_only: bool = False):
        """Full Phase-0 CSR graph build on the volume's raw files."""
        from flygpt.graph import main as graph_main

        argv = [
            "--config", "/pkg/configs/graph_v1.yaml",
            "--raw-dir", str(DATA / "raw"),
            "--graph-dir", str(DATA / "graph"),
            "--provenance-dir", str(DATA / "provenance"),
            "--results-dir", str(DATA / "results"),
            "--manifest", str(DATA / "provenance" / "malecns_v1.json"),
        ]
        if skip_verify:
            argv.append("--skip-verify")
        if topo_only:
            argv.append("--topo-only")
        rc = graph_main(argv)
        data_vol.commit()
        return rc

    @app.function(image=image, volumes={"/data": data_vol}, timeout=6 * 3600,
                  memory=16384, cpu=4, gpu="T4")
    def pilot(backend: str = "jax", gpu_class: str = "t4"):
        """Phase-1 pilot gate on the volume's built graph."""
        import yaml

        from flygpt.corpus import _default_library, run_pilot
        from flygpt.schema import load_graph_bundle

        cfg = yaml.safe_load(
            Path("/pkg/configs/corpus_v1.yaml").read_text()
        )
        cfg["stim_library"] = "/pkg/configs/stim_library_v1.yaml"
        cfg["sim_config"] = "/pkg/configs/sim_v1.yaml"
        cutoff = int((cfg.get("slices") or {}).get("main", {}).get("cutoff", 5))
        bundle = load_graph_bundle(DATA / "graph", cutoff=cutoff)
        report = run_pilot(
            bundle, _default_library(), cfg,
            DATA / "corpus" / "pilot", DATA / "results",
            sim_cfg=_sim_cfg_override(cfg, backend, gpu_class),
        )
        data_vol.commit()
        return report

    def _sim_cfg_override(corpus_cfg: dict, backend: str, gpu_class: str) -> dict:
        """Load the sim config and stamp the Modal backend/gpu_class."""
        import yaml

        path = Path(corpus_cfg.get("sim_config") or "/pkg/configs/sim_v1.yaml")
        sim_cfg = yaml.safe_load(path.read_text()) or {}
        sim_cfg["backend"] = backend
        sim_cfg["gpu_class"] = gpu_class
        model = sim_cfg.get("model") or {}
        model["backend"] = backend
        model["gpu_class"] = gpu_class
        sim_cfg["model"] = model
        return sim_cfg

    @app.function(image=image, volumes={"/data": data_vol}, timeout=4 * 3600,
                  memory=8192, cpu=2)
    def eval_battery():
        """Run the results-battery evals over artifacts on the volume."""
        from flygpt.evals import main as eval_main

        rc = eval_main(["--results-dir", str(DATA / "results"),
                        "--graph-dir", str(DATA / "graph")])
        data_vol.commit()
        return rc

    @app.function(image=image, volumes={"/data": data_vol}, timeout=3600,
                  gpu="T4")
    def smoke():
        """T-SHIP-8: pipeline smoke test on a CUDA device."""
        import subprocess as sp

        return sp.call([sys.executable, "-m", "flygpt.evals.ship_checks", "smoke",
                        "--device", "cuda"])

    @app.function(image=image, volumes={"/data": data_vol}, timeout=6 * 3600,
                  memory=16384, cpu=4)
    def build_banc(download: bool = True):
        """Phase-4: download BANC files + build the female CSR graph."""
        from flygpt.banc import main as banc_main

        argv = ["--config", "/pkg/configs/banc_v1.yaml",
                "--raw-dir", str(DATA / "raw" / "banc"),
                "--graph-dir", str(DATA / "graph_banc"),
                "--provenance-dir", str(DATA / "provenance"),
                "--results-dir", str(DATA / "results"), "--build"]
        if download:
            argv.insert(1, "--download")
        rc = banc_main(argv)
        data_vol.commit()
        return rc
else:
    def _missing(*_a, **_k):
        raise RuntimeError("modal not installed - pip install -e .[remote]")

    app = None
    ingest = _missing
    build_graph = _missing
    pilot = _missing
    eval_battery = _missing
    smoke = _missing
    build_banc = _missing


if __name__ == "__main__":
    sys.exit("usage: modal run modal_app.py::<ingest|build_graph|pilot|eval_battery|smoke|build_banc>")
