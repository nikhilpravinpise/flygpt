"""Canned-rollout generator for the static "Probe a Fly Brain" Space.

Phase 6 / T-SHIP-7 contract: the static Space serves precomputed rollouts
only - zero live sim calls and zero GPU in the demo path
(``space/README.md``). This script runs REAL small sims on a compact
bundled graph fixture (~100 typed nodes spanning the demo's five sensory
populations and their motor readouts), or on a subgraph of ``data/graph``
when ``--graph-dir`` points at a real build, and writes
``space/canned/<stimulus>.json`` + ``space/canned/manifest.json``. It never
touches the 10 GB raw set and finishes in seconds on CPU.

Sim path: ``flygpt.sim_backend`` is preferred via lazy import once its
Phase-1 API (``SimConfig``/``LIFSim``/``simulate``) exists - any failure
there is recorded in provenance, never hidden. Until then a bundled
reference kernel runs the identical ``sim_v1`` discrete update
(configs/sim_v1.yaml), so canned events are real simulator output either
way:

    g[t+1] = g[t]*exp(-dt/tau_g) + sum_{pre fired at t} w_signed(pre,post)
    v[t+1] = v[t]*exp(-dt/tau_v) + drive_scale*g[t+1]
    fired  = v >= threshold ; v[fired] = reset
    w_signed = sign * weight * w_syn            (weight_mode = "count")

Stimulus buttons mirror the demo spec (research doc section 10): each maps
to real annotated sensory populations per ``configs/stim_library_v1.yaml``
grounding queries - sugar = labellar/taste-peg/pharyngeal GRNs, bitter =
leg-bristle GRNs (water/bitter axis), loom = LC4+LPLC2 visual projection
neurons, female-cue = LC10a/LC11 + pheromone ORNs (courtship composite),
antenna-dust = JON chordotonal + antennal mechanosensory bristles.

Every generated file carries full provenance (code sha, graph sha, sim
config, seed, backend, generated_utc) and the dataset-level honesty note:
the traces are simulated, not recorded.

Usage:
    python space/build_canned.py --all                    # regenerate everything
    python space/build_canned.py --stimulus sugar         # one file
    python space/build_canned.py --graph-dir data/graph --all
    python space/build_canned.py --list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from flygpt.schema import (
    bundle_from_edges,
    canonical_json,
    git_sha,
    seed_from_config,
    sha256_file,
    utcnow,
)

SPACE_DIR = Path(__file__).resolve().parent
REPO_DIR = SPACE_DIR.parent
CANNED_DIR = SPACE_DIR / "canned"
SONG_REFERENCE = REPO_DIR / "evals" / "song_reference.json"

CANNED_SCHEMA = "canned_v1"
MANIFEST_SCHEMA = "canned_manifest_v1"
GENERATOR = "space/build_canned.py"

# sim_v1 defaults (configs/sim_v1.yaml) - the demo rollout config; dt and
# window vary per stimulus and are recorded per file.
SIM_CFG: dict[str, Any] = {
    "dt": 0.020,
    "tau_v": 0.100,
    "tau_g": 0.010,
    "w_syn": 0.01,
    "threshold": 1.0,
    "reset": 0.0,
    "refractory_steps": 0,
    "weight_mode": "count",
    "synapse_model": "alpha",
    "drive_scale": 1.0,
    "warmup_steps": 5,
    "backend": "numpy",
}

# Fields mirrored into sim_backend.SimConfig when the Phase-1 API exists.
_SIM_CFG_KEYS = (
    "dt", "tau_v", "tau_g", "w_syn", "threshold", "reset",
    "refractory_steps", "weight_mode", "synapse_model", "drive_scale",
    "warmup_steps", "seed",
)

# ---------------------------------------------------------------------------
# Embedded demo graph fixture (~100 nodes; indices are assigned in order)
# ---------------------------------------------------------------------------
# (count, type, class, subclass, superclass, neuromere, layer)
_NODE_ROWS: tuple[tuple[int, str, str, str, str, str, str], ...] = (
    # --- sensory populations (the five buttons' real grounding queries) ---
    (8, "GRN_sugar", "gustatory", "labellar bristle", "sensory", "SEZ", "sensory"),
    (2, "GRN_sugar", "gustatory", "taste peg", "sensory", "SEZ", "sensory"),
    (2, "GRN_sugar", "gustatory", "pharyngeal sensillum", "sensory", "SEZ", "sensory"),
    (8, "GRN_bitter", "gustatory", "leg bristle", "sensory", "VNC", "sensory"),
    (6, "LC4", "visual", "visual projection neuron", "sensory", "OL", "sensory"),
    (4, "LPLC2", "visual", "visual projection neuron", "sensory", "OL", "sensory"),
    (4, "LC10a", "visual", "visual projection neuron", "sensory", "OL", "sensory"),
    (4, "LC11", "visual", "visual projection neuron", "sensory", "OL", "sensory"),
    (4, "ORN_pheromone", "olfactory", "antennal olfactory receptor", "sensory", "AL", "sensory"),
    (8, "JON", "mechanosensory", "chordotonal organ", "sensory", "AMMC", "sensory"),
    (4, "aMech", "mechanosensory", "mechanosensory bristle", "sensory", "AMMC", "sensory"),
    # --- interneurons ---
    (4, "FeedIN", "central", "feeding interneuron", "cb_intrinsic", "SEZ", "interneuron"),
    (2, "BitterIN", "central", "bitter interneuron", "cb_intrinsic", "SEZ", "interneuron"),
    (2, "EscapeIN", "central", "escape interneuron", "cb_intrinsic", "OL", "interneuron"),
    (6, "CourtIN", "central", "courtship interneuron", "cb_intrinsic", "CB", "interneuron"),
    (4, "GroomIN", "central", "grooming interneuron", "cb_intrinsic", "AMMC", "interneuron"),
    (4, "HubIN", "central", "shared hub interneuron", "cb_intrinsic", "CB", "interneuron"),
    # --- motor / descending readouts ---
    (4, "MN9", "motor", "proboscis motor neuron", "cb_motor", "SEZ", "motor"),
    (2, "DNp01", "descending", "giant-fiber descending neuron", "descending_neuron", "CB", "motor"),
    (4, "GroomMN", "motor", "grooming motor neuron", "vnc_motor", "T1", "motor"),
    (8, "WingMN", "motor", "wing motor neuron", "vnc_motor", "T2", "motor"),
    (2, "WingSine", "motor", "wing motor neuron", "vnc_motor", "T2", "motor"),
    (2, "RejectMN", "motor", "proboscis retract motor neuron", "cb_motor", "SEZ", "motor"),
)

LAYER_X = {"sensory": 0.10, "interneuron": 0.52, "motor": 0.90}


def _idx(name: str) -> list[int]:
    """Node-index range of a demo type (fixture is ordered by _NODE_ROWS)."""
    out, i = [], 0
    for count, typ, *_ in _NODE_ROWS:
        if typ == name:
            out.extend(range(i, i + count))
        i += count
    return out


def _fan(pre: list[int], post: list[int], w: int, s: int) -> list[tuple[int, int, int, int]]:
    return [(p, q, w, s) for p in pre for q in post]


def _demo_edges() -> list[tuple[int, int, int, int]]:
    """(pre, post, weight, sign) demo wiring - hand-laid, deterministic."""
    sugar, bitter = _idx("GRN_sugar"), _idx("GRN_bitter")
    loom = _idx("LC4") + _idx("LPLC2")
    court = _idx("LC10a") + _idx("LC11") + _idx("ORN_pheromone")
    dust = _idx("JON") + _idx("aMech")
    feed_in, bitter_in = _idx("FeedIN"), _idx("BitterIN")
    escape_in, court_in = _idx("EscapeIN"), _idx("CourtIN")
    groom_in, hub_in = _idx("GroomIN"), _idx("HubIN")
    mn9, dnp01 = _idx("MN9"), _idx("DNp01")
    groom_mn, wing_mn = _idx("GroomMN"), _idx("WingMN")
    wing_sine, reject_mn = _idx("WingSine"), _idx("RejectMN")
    sensory = sugar + bitter + loom + court + dust

    edges: list[tuple[int, int, int, int]] = []
    edges += _fan(sugar, feed_in, 24, 1)          # sugar GRNs -> feeding INs
    edges += _fan(bitter, bitter_in, 26, 1)       # leg-bristle GRNs -> bitter INs
    edges += _fan(loom, escape_in, 24, 1)         # loom VPNs -> escape INs
    edges += _fan(_idx("LPLC2"), dnp01, 18, 1)    # direct giant-fiber path
    edges += _fan(court, court_in, 2, 1)          # visual + pheromone -> courtship INs
    edges += _fan(dust, groom_in, 20, 1)          # JON/bristles -> grooming INs
    edges += _fan(sensory, hub_in, 5, 1)          # weak shared hub (background sink)
    edges += _fan(feed_in, mn9, 30, 1)            # feeding program (MN9 axis)
    edges += _fan(bitter_in, reject_mn, 30, 1)    # taste rejection / withdrawal
    edges += _fan(bitter_in, mn9, 24, -1)         # bitter suppresses feeding
    edges += _fan(bitter_in, court_in, 10, -1)    # aversive -> less courtship
    edges += _fan(escape_in, dnp01, 30, 1)        # escape command neuron
    edges += _fan(court_in, wing_mn, 13, 1)       # pulse-song wing drive
    edges += _fan(court_in, wing_sine, 2, 1)      # sine-song wing drive (weak)
    edges += _fan(groom_in, groom_mn, 28, 1)      # grooming program
    edges += _fan(groom_in, feed_in, 12, -1)      # grooming vs feeding cross-talk
    return edges


def _demo_nodes() -> list[dict[str, Any]]:
    """Node table for the embedded fixture; layout x/y included for the
    propagation canvas (deterministic, by layer)."""
    nodes: list[dict[str, Any]] = []
    body = 1000
    layer_sizes: dict[str, int] = {}
    for count, _, _, _, _, _, layer in _NODE_ROWS:
        layer_sizes[layer] = layer_sizes.get(layer, 0) + count
    layer_pos: dict[str, int] = {}
    for count, typ, klass, subclass, superclass, neuromere, layer in _NODE_ROWS:
        for _ in range(count):
            pos = layer_pos.get(layer, 0)
            nodes.append({
                "i": len(nodes),
                "bodyId": body,
                "type": typ,
                "class": klass,
                "subclass": subclass,
                "superclass": superclass,
                "neuromere": neuromere,
                "layer": layer,
                "x": LAYER_X[layer],
                "y": round((pos + 0.5) / layer_sizes[layer], 4),
            })
            layer_pos[layer] = pos + 1
            body += 1
    return nodes


def demo_graph_spec() -> dict[str, Any]:
    """JSON-able spec of the embedded fixture (nodes + signed edges)."""
    return {"nodes": _demo_nodes(), "edges": _demo_edges()}


def graph_sha(spec: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(spec).encode()).hexdigest()


def demo_bundle(nodes: list[dict[str, Any]], edges: list[tuple[int, int, int, int]]):
    """GraphBundle for the embedded fixture (real bundle type)."""
    df = pl.DataFrame({
        "bodyId": [n["bodyId"] for n in nodes],
        "type": [n["type"] for n in nodes],
        "class": [n["class"] for n in nodes],
        "subclass": [n["subclass"] for n in nodes],
        "superclass": [n["superclass"] for n in nodes],
        "somaNeuromere": [n["neuromere"] for n in nodes],
    })
    arr = np.asarray(edges, dtype=np.int64)
    return bundle_from_edges(
        df,
        arr[:, 0].astype(np.uint32),
        arr[:, 1].astype(np.uint32),
        arr[:, 2].astype(np.int32),
        arr[:, 3].astype(np.int8),
    )


def load_real_subgraph(graph_dir: Path, n_max: int = 240) -> dict[str, Any] | None:
    """Best-effort demo subgraph from a real ``data/graph`` build.

    Selects nodes whose type matches the demo populations (LC/VPN types,
    ORN/JON/GRN prefixes, motor readouts), caps at ``n_max``, and induces
    the subgraph edges. Returns a graph spec or ``None`` when the build is
    absent - the embedded fixture is always the documented fallback.
    """
    from flygpt.schema import load_graph_bundle

    if not (graph_dir / "nodes.parquet").exists():
        return None
    bundle = load_graph_bundle(graph_dir, cutoff=5)
    nodes_df = bundle.nodes
    types = bundle.type_array()
    sup = (
        nodes_df["superclass"].to_numpy().astype(str)
        if "superclass" in nodes_df.columns
        else np.full(bundle.n_nodes, "")
    )
    keep_pat = (
        "LC", "LPLC", "LLPC", "LPC", "LCNO", "ORN", "JON", "GRN",
        "DNp", "MN9", "Mech", "aMech",
    )
    wanted = np.array([t.startswith(keep_pat) for t in types])
    motorish = np.array(["motor" in s or "descending" in s for s in sup])
    sensory_idx = np.flatnonzero(wanted & ~motorish)[: n_max // 2]
    motor_idx = np.flatnonzero(motorish)[: n_max // 4]
    pool = np.unique(np.concatenate([sensory_idx, motor_idx]))
    # one hop of out-neighbors from the selected sensory nodes
    extra = np.unique(
        np.concatenate([bundle.out_neighbors(int(i)) for i in sensory_idx])
    ) if sensory_idx.size else np.array([], dtype=np.uint32)
    sel = np.unique(np.concatenate([pool, extra]))[:n_max]
    remap = {int(old): i for i, old in enumerate(sel)}
    edges = []
    for old in sel:
        i = remap[int(old)]
        lo, hi = bundle.out_indptr[int(old)], bundle.out_indptr[int(old) + 1]
        posts = bundle.out_indices[lo:hi]
        for k, q in enumerate(posts):
            j = remap.get(int(q))
            if j is None:
                continue
            w = int(bundle.out_weight[lo + k])
            s = int(bundle.out_sign[lo + k])
            edges.append((i, j, max(w, 5), s if s else 1))
    if not edges:
        return None
    layer_of = {}
    for old in sel:
        s = str(sup[int(old)])
        if "motor" in s or "descending" in s:
            layer_of[int(old)] = "motor"
        elif wanted[int(old)]:
            layer_of[int(old)] = "sensory"
        else:
            layer_of[int(old)] = "interneuron"
    counts = {"sensory": 0, "interneuron": 0, "motor": 0}
    for old in sel:
        counts[layer_of[int(old)]] += 1
    pos: dict[str, int] = {}
    nodes = []
    for i, old in enumerate(sel):
        layer = layer_of[int(old)]
        p = pos.get(layer, 0)
        nodes.append({
            "i": i,
            "bodyId": int(nodes_df["bodyId"][int(old)]),
            "type": str(types[int(old)]) or "untyped",
            "class": "",
            "subclass": "",
            "superclass": s,
            "neuromere": "",
            "layer": layer,
            "x": LAYER_X[layer],
            "y": round((p + 0.5) / max(counts[layer], 1), 4),
        })
        pos[layer] = p + 1
    return {"nodes": nodes, "edges": edges}


# ---------------------------------------------------------------------------
# Stimulus registry - buttons map to real annotated sensory populations
# ---------------------------------------------------------------------------

READOUTS: dict[str, dict[str, Any]] = {
    "feed": {"types": ["MN9"], "label": "proboscis extension - feeding program (MN9 axis)"},
    "flee": {"types": ["DNp01"], "label": "escape - DNp01 giant-fiber takeoff"},
    "groom": {"types": ["GroomMN"], "label": "antennal grooming"},
    "court": {"types": ["WingMN", "WingSine"], "label": "courtship song - pulse/sine wing program"},
    "reject": {"types": ["RejectMN"], "label": "proboscis retraction - taste rejection"},
}

STIMULI: dict[str, dict[str, Any]] = {
    "sugar": {
        "button": "sugar",
        "title": "Sugar on the proboscis",
        "population": "labellar bristle + taste peg + pharyngeal GRNs "
                      "(sugar/feeding axis; stim_library_v1 gustatory_sugar)",
        "query": {"subclasses": ["labellar bristle", "taste peg", "pharyngeal sensillum"]},
        "dt": 0.020,
        "window_s": 1.0,
        "drives": [{"rate_hz": 20.0, "window_s": [0.05, 0.95], "envelope": "poisson"}],
        "expected": "feed",
    },
    "bitter": {
        "button": "bitter",
        "title": "Bitter on the leg",
        "population": "leg-bristle GRNs (water/bitter axis; gustatory_leg)",
        "query": {"subclasses": ["leg bristle"]},
        "dt": 0.020,
        "window_s": 1.0,
        "drives": [{"rate_hz": 25.0, "window_s": [0.05, 0.95], "envelope": "poisson"}],
        "expected": "reject",
    },
    "loom": {
        "button": "shadow overhead",
        "title": "Shadow overhead (loom)",
        "population": "LC4 + LPLC2 visual projection neurons (loom/escape; "
                      "visual_lc - LPLC2 feeds the DNp01 giant fiber)",
        "query": {"types": ["LC4", "LPLC2"]},
        "dt": 0.020,
        "window_s": 1.0,
        "drives": [{"rate_hz": 30.0, "window_s": [0.05, 0.95], "envelope": "poisson"}],
        "expected": "flee",
    },
    "female_cue": {
        "button": "female detected",
        "title": "Female detected (courtship)",
        "population": "LC10a + LC11 visual types + pheromone ORNs "
                      "(courtship_cue composite; courtship slice runs at dt=5ms)",
        "query_multi": [
            {"types": ["LC10a", "LC11"]},
            {"type_prefix": "ORN_"},
        ],
        "dt": 0.005,
        "window_s": 2.0,
        "drives": [
            {"rate_hz": 6.0, "window_s": [0.0, 2.0], "envelope": "poisson"},
            {
                "nodes_types": ["CourtIN"], "envelope": "pulse",
                "period_ms": 35.0, "jitter_steps": 2, "burst_p": 0.85,
                "window_s": [0.3, 0.7],
            },
            {
                "nodes_types": ["CourtIN"], "envelope": "pulse",
                "period_ms": 38.0, "jitter_steps": 2, "burst_p": 0.85,
                "window_s": [0.9, 1.25],
            },
            {
                "nodes_types": ["WingSine"], "envelope": "tonic",
                "rate_hz": 55.0, "window_s": [1.35, 1.95],
            },
        ],
        "expected": "court",
        "ethogram": True,
    },
    "antenna_dust": {
        "button": "dust on antennae",
        "title": "Dust on the antennae",
        "population": "Johnston's-organ chordotonal neurons + antennal "
                      "mechanosensory bristles (mechanosensory; dust -> grooming)",
        "query": {"subclasses": ["chordotonal organ", "mechanosensory bristle"]},
        "dt": 0.020,
        "window_s": 1.0,
        "drives": [{"rate_hz": 20.0, "window_s": [0.05, 0.95], "envelope": "poisson"}],
        "expected": "groom",
    },
}


def resolve_nodes(nodes: list[dict[str, Any]], query: dict[str, Any]) -> list[int]:
    """Tiny SensoryIndex-style resolver over a node dict list."""
    out = []
    for i, n in enumerate(nodes):
        if "types" in query and n["type"] not in query["types"]:
            continue
        if "subclasses" in query and n["subclass"] not in query["subclasses"]:
            continue
        if "type_prefix" in query and not n["type"].startswith(query["type_prefix"]):
            continue
        out.append(i)
    return out


def stimulus_nodes(nodes: list[dict[str, Any]], spec: dict[str, Any]) -> list[int]:
    if "query_multi" in spec:
        out: set[int] = set()
        for q in spec["query_multi"]:
            out.update(resolve_nodes(nodes, q))
        return sorted(out)
    return resolve_nodes(nodes, spec["query"])


def readout_pops(nodes: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {
        name: resolve_nodes(nodes, {"types": spec["types"]})
        for name, spec in READOUTS.items()
    }


# ---------------------------------------------------------------------------
# Reference LIF kernel (sim_v1 discrete update; used until sim_backend lands)
# ---------------------------------------------------------------------------


def _sample_drives(
    drives: list[dict[str, Any]], dt: float, n_steps: int, rng: np.random.Generator
) -> list[set[int]]:
    """Pre-sample exogenous drive spikes per step (Poisson/tonic/pulse)."""
    per_step: list[set[int]] = [set() for _ in range(n_steps)]
    for d in drives:
        nodes = np.asarray(d["nodes"], dtype=np.int64)
        if nodes.size == 0:
            continue
        t0, t1 = d["window_s"]
        s0 = max(0, int(t0 / dt))
        s1 = min(n_steps, int(np.ceil(t1 / dt)))
        if s1 <= s0:
            continue
        env = d.get("envelope", "poisson")
        if env in ("poisson", "tonic"):
            # exact Poisson-train step occupancy, same as sim_backend's driver
            p = float(1.0 - np.exp(-float(d["rate_hz"]) * dt))
            hits = rng.random((s1 - s0, nodes.size)) < p
            for step_off, node_off in np.argwhere(hits):
                per_step[s0 + int(step_off)].add(int(nodes[int(node_off)]))
        elif env == "pulse":
            period_steps = max(1, round(float(d["period_ms"]) / 1000.0 / dt))
            jitter = int(d.get("jitter_steps", max(1, period_steps // 4)))
            bp = float(d.get("burst_p", 1.0))
            bs = s0
            while bs < s1:
                fired = nodes[rng.random(nodes.size) < bp]
                per_step[bs].update(int(x) for x in fired)
                bs += period_steps + int(rng.integers(-jitter, jitter + 1))
        else:
            raise ValueError(f"unknown drive envelope: {env}")
    return per_step


def _mini_lif(
    bundle: Any,
    drives: list[dict[str, Any]],
    cfg: dict[str, Any],
    n_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """sim_v1 alpha-synapse LIF on a GraphBundle - deterministic on CPU.

    Mirrors the documented discrete update (module docstring); driven nodes
    emit an event AND propagate (Shiu-protocol exogenous spikes). Numpy
    backend only; bitwise-deterministic given (seed, cfg).
    """
    rng = np.random.default_rng(seed % (2**63))
    n = bundle.n_nodes
    dg = float(np.exp(-cfg["dt"] / cfg["tau_g"]))
    dv = float(np.exp(-cfg["dt"] / cfg["tau_v"]))
    signed = bundle.signed_weights_out * np.float32(cfg["w_syn"])
    indptr, indices = bundle.out_indptr, bundle.out_indices
    drive = _sample_drives(drives, cfg["dt"], n_steps, rng)
    g = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    refr = int(cfg.get("refractory_steps", 0))
    last_fired = np.full(n, -(refr + 1), dtype=np.int64)
    fired_prev = np.zeros(n, dtype=bool)
    ev_step: list[int] = []
    ev_node: list[int] = []
    warmup = int(cfg.get("warmup_steps", 0))
    total = n_steps + warmup
    for t in range(total):
        contrib = np.zeros(n, dtype=np.float64)
        for pre in np.flatnonzero(fired_prev):
            lo, hi = indptr[pre], indptr[pre + 1]
            contrib[indices[lo:hi]] += signed[lo:hi]
        g = g * dg + contrib
        v = v * dv + cfg["drive_scale"] * g
        fired = v >= cfg["threshold"]
        if refr:
            fired &= (t - last_fired) > refr
        v[fired] = cfg["reset"]
        rec = t - warmup
        ext = drive[rec] if 0 <= rec < len(drive) else set()
        if ext:
            fired[np.fromiter(ext, dtype=np.int64)] = True
        if rec >= 0:
            for node in np.flatnonzero(fired):
                ev_step.append(rec)
                ev_node.append(int(node))
        last_fired[fired] = t
        fired_prev = fired
    return np.asarray(ev_step, dtype=np.uint32), np.asarray(ev_node, dtype=np.uint32)


def _demo_driver(drive_sets: list[set[int]], warmup: int):
    """LIFSim driver: force_spike the pre-sampled exogenous sets per
    recorded step (drive windows live on the recorded timeline)."""

    def driver(t: int, sim) -> None:
        k = t - warmup
        if 0 <= k < len(drive_sets) and drive_sets[k]:
            sim.force_spike(sorted(drive_sets[k]))

    return driver


def _run_rollout(
    bundle: Any,
    program: dict[str, Any],
    cfg: dict[str, Any],
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, str, str | None]:
    """Run the rollout through ``flygpt.sim_backend`` (lazy import) when its
    Phase-1 API exists - ``LIFSim.run`` with a driver compiled from the
    program's drives, the spec'd injection point. Patterned demo envelopes
    (``pulse``) ride the same ``force_spike`` mechanism the backend's own
    driver uses. Falls back to the bundled ``sim_v1`` reference kernel when
    the module is absent; a real-backend failure is recorded, never hidden.
    Returns (steps, nodes, backend_tag, backend_error)."""
    seed = int(program["seed"])
    try:
        from flygpt import sim_backend
    except ImportError:
        sim_backend = None
    if sim_backend is not None and all(
        hasattr(sim_backend, name) for name in ("SimConfig", "LIFSim")
    ):
        err = None
        try:
            sim_cfg = sim_backend.SimConfig(
                **{k: v for k, v in cfg.items() if k in _SIM_CFG_KEYS}
            )
            rng = np.random.default_rng(seed % (2**63))
            drive_sets = _sample_drives(program["drives"], cfg["dt"], n_steps, rng)
            sim = sim_backend.LIFSim(bundle, sim_cfg)
            sim.reset(seed)
            result = sim.run(
                n_steps, driver=_demo_driver(drive_sets, sim_cfg.warmup_steps)
            )
            return (
                np.asarray(result.events_step),
                np.asarray(result.events_node),
                "flygpt.sim_backend",
                None,
            )
        except Exception as exc:  # recorded in provenance, never hidden
            err = f"{type(exc).__name__}: {exc}"
            print(f"[build_canned] sim_backend failed, using mini_lif: {err}",
                  file=sys.stderr)
    else:
        err = "flygpt.sim_backend not built yet"
    steps, nodes = _mini_lif(bundle, program["drives"], cfg, n_steps, seed)
    return steps, nodes, "mini_lif(sim_v1)", err


# ---------------------------------------------------------------------------
# Readout decode + song ethogram
# ---------------------------------------------------------------------------


def decode_readout(
    ev_node: np.ndarray, pops: dict[str, list[int]]
) -> dict[str, Any]:
    """Motor-program readout: per-population event counts + rates -> decode."""
    programs = {}
    best, best_rate = "", -1.0
    for name, idxs in pops.items():
        if not idxs:
            continue
        n_ev = int(np.isin(ev_node, np.asarray(idxs)).sum())
        rate = n_ev / len(idxs)
        programs[name] = {"nodes": idxs, "events": n_ev, "events_per_node": round(rate, 3)}
        if rate > best_rate:
            best, best_rate = name, rate
    return {
        "programs": programs,
        "decoded": best,
        "decoded_label": READOUTS.get(best, {}).get("label", best),
    }


def _group_times(times: list[float], gap_ms: float) -> list[list[float]]:
    groups: list[list[float]] = []
    for t in times:
        if groups and t - groups[-1][-1] <= gap_ms:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def song_ethogram(
    ev_step: np.ndarray,
    ev_node: np.ndarray,
    pulse_nodes: list[int],
    sine_nodes: list[int],
    dt_ms: float,
    phases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Pulse/sine bout segmentation on wing-motor events (SongDecoder rules:
    bout_gap 10ms pulse clustering, sine min-bout 50ms). Statistics land in
    the same fields evals/song_reference.json pins for T-REAL-3."""
    pulse_set, sine_set = set(pulse_nodes), set(sine_nodes)
    t_ms = ev_step.astype(np.float64) * dt_ms
    pulse_t = sorted(float(t_ms[i]) for i in range(len(t_ms)) if int(ev_node[i]) in pulse_set)
    sine_t = sorted(float(t_ms[i]) for i in range(len(t_ms)) if int(ev_node[i]) in sine_set)

    groups = _group_times(pulse_t, 10.0)              # spike clusters -> pulses
    onsets = [g[0] for g in groups]
    bouts: list[list[float]] = []
    cur: list[float] = []
    for o in onsets:                                   # pulses -> bouts (IPI <= 120ms)
        if cur and o - cur[-1] > 120.0:
            if len(cur) >= 3:
                bouts.append(cur)
            cur = []
        cur.append(o)
    if len(cur) >= 3:
        bouts.append(cur)

    ipis = [b[i + 1] - b[i] for b in bouts for i in range(len(b) - 1)]
    ipi_median = float(np.median(ipis)) if ipis else 0.0
    ipi_cv = float(np.std(ipis) / np.mean(ipis)) if ipis and np.mean(ipis) > 0 else 0.0
    bout_durs = [b[-1] - b[0] for b in bouts]
    bout_med = float(np.median(bout_durs)) if bout_durs else 0.0

    sine_groups = _group_times(sine_t, 60.0)
    sine_bouts = [[g[0], g[-1]] for g in sine_groups if g[-1] - g[0] >= 50.0]
    pulse_bout_ms = float(sum(bout_durs))
    sine_bout_ms = float(sum(e[1] - e[0] for e in sine_bouts))
    denom = pulse_bout_ms + sine_bout_ms
    ratio = pulse_bout_ms / denom if denom > 0 else 0.0

    return {
        "phases": phases,
        "pulse_bouts": [[round(b[0], 1), round(b[-1], 1)] for b in bouts],
        "sine_bouts": [[round(e[0], 1), round(e[1], 1)] for e in sine_bouts],
        "pulse_onsets_ms": [round(o, 1) for o in onsets],
        "stats": {
            "ipi_median_ms": round(ipi_median, 2),
            "ipi_cv": round(ipi_cv, 3),
            "n_pulse_bouts": len(bouts),
            "n_sine_bouts": len(sine_bouts),
            "bout_duration_s": round(bout_med / 1000.0, 3),
            "pulse_sine_ratio": round(ratio, 3),
            "context_dependence": "direction_consistent"
            if phases and phases[0].get("mode") == "pulse" else "not_evaluated",
        },
        "reference": _song_reference(),
    }


def _song_reference() -> dict[str, Any]:
    """Static reference ranges (T-REAL-3) - read from evals/ when present."""
    if SONG_REFERENCE.exists():
        ref = json.loads(SONG_REFERENCE.read_text())
        ref = dict(ref)
        ref["loaded_from"] = "evals/song_reference.json"
        return ref
    return {
        "panel": "song_reference_v1",
        "loaded_from": "embedded copy",
        "statistics": [
            {"id": "ipi_median_ms", "expected_range_ms": [30, 45], "reference_value_ms": 37},
            {"id": "pulse_sine_ratio", "expected_range": [0.3, 0.9]},
            {"id": "bout_duration_s", "expected_range_s": [0.2, 2.0]},
            {"id": "ipi_cv", "expected_range": [0.05, 0.5]},
            {"id": "context_dependence", "expected": "direction_consistent"},
        ],
    }


# ---------------------------------------------------------------------------
# Canned-file build + validation
# ---------------------------------------------------------------------------

_PROV_KEYS = ("code_sha", "generator", "graph_sha", "graph_source", "backend",
              "sim_cfg", "seed", "generated_utc")


def build_program(
    nodes: list[dict[str, Any]], stim_id: str, spec: dict[str, Any]
) -> tuple[dict[str, Any], list[int]]:
    """Resolve grounding queries -> drive dicts + the program spec (seeded)."""
    driven = stimulus_nodes(nodes, spec)
    drives = []
    for d in spec["drives"]:
        dd = dict(d)
        if "nodes_types" in dd:
            dd["nodes"] = resolve_nodes(nodes, {"types": dd.pop("nodes_types")})
        else:
            dd["nodes"] = driven
        drives.append(dd)
    spec_drives = [
        {k: v for k, v in d.items() if k != "nodes"} | {"n_nodes": len(d["nodes"])}
        for d in drives
    ]
    program = {
        "family_id": "space_demo",
        "program_id": stim_id,
        "drives": drives,
        "drives_spec": spec_drives,
        "duration_s": spec["window_s"],
        "dt": spec["dt"],
    }
    program["seed"] = seed_from_config(
        {"stimulus": stim_id, "drives": spec_drives,
         "duration_s": spec["window_s"], "dt": spec["dt"]}
    )
    for d in drives:
        d["nodes"] = list(d["nodes"])
    return program, driven


def generate(
    stim_id: str,
    spec_graph: dict[str, Any],
    graph_source: str,
    outdir: Path = CANNED_DIR,
) -> dict[str, Any]:
    """Run one stimulus rollout and write ``<outdir>/<stim_id>.json``."""
    spec = STIMULI[stim_id]
    nodes, edges = spec_graph["nodes"], spec_graph["edges"]
    bundle = demo_bundle(nodes, edges) if graph_source == "embedded_fixture" else _bundle_of(spec_graph)
    program, driven = build_program(nodes, stim_id, spec)
    dt = float(spec["dt"])
    n_steps = int(round(spec["window_s"] / dt))
    cfg = dict(SIM_CFG, dt=dt, seed=int(program["seed"]))

    ev_step, ev_node, backend, backend_err = _run_rollout(
        bundle, program, cfg, n_steps
    )
    order = np.lexsort((ev_node, ev_step))
    ev_step, ev_node = ev_step[order], ev_node[order]
    events = [
        [round(float(s) * dt * 1000.0, 3), int(n)]
        for s, n in zip(ev_step, ev_node, strict=True)
    ]

    pops = readout_pops(nodes)
    readout = decode_readout(ev_node, pops)
    ethogram = None
    if spec.get("ethogram"):
        phases = [
            {"t0_ms": 300.0, "t1_ms": 1250.0, "mode": "pulse", "context": "near"},
            {"t0_ms": 1350.0, "t1_ms": 1950.0, "mode": "sine", "context": "far"},
        ]
        ethogram = song_ethogram(
            ev_step, ev_node,
            resolve_nodes(nodes, {"types": ["WingMN"]}),
            resolve_nodes(nodes, {"types": ["WingSine"]}),
            dt * 1000.0, phases,
        )

    prov = {
        "code_sha": git_sha(),
        "generator": GENERATOR,
        "graph_sha": graph_sha(spec_graph),
        "graph_source": graph_source,
        "backend": backend,
        "sim_cfg": cfg,
        "seed": int(program["seed"]),
        "generated_utc": utcnow(),
    }
    if backend_err:
        prov["backend_error"] = backend_err

    canned = {
        "schema": CANNED_SCHEMA,
        "stimulus": stim_id,
        "button": spec["button"],
        "title": spec["title"],
        "population": {
            "description": spec["population"],
            "query": spec.get("query") or spec.get("query_multi"),
            "driven_nodes": driven,
            "n_driven": len(driven),
            "drives": program["drives_spec"],
        },
        "n_nodes_small": len(nodes),
        "nodes": [
            {"i": n["i"], "type": n["type"], "layer": n["layer"], "x": n["x"], "y": n["y"]}
            for n in nodes
        ],
        "edges": [[int(p), int(q), int(s)] for p, q, _, s in
                  [tuple(e) for e in edges]],
        "dt_ms": round(dt * 1000.0, 3),
        "window_ms": round(spec["window_s"] * 1000.0, 3),
        "n_steps": n_steps,
        "events": events,
        "readout": readout,
        "ethogram": ethogram,
        "provenance": prov,
    }
    errors = validate_canned(canned)
    if errors:
        raise ValueError(f"generated {stim_id} failed validation: {errors}")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{stim_id}.json").write_text(json.dumps(canned) + "\n")
    return canned


def _bundle_of(spec_graph: dict[str, Any]):
    """GraphBundle for a spec dict (real-subgraph path)."""
    return demo_bundle(spec_graph["nodes"], [tuple(e) for e in spec_graph["edges"]])


def validate_canned(obj: Any) -> list[str]:
    """``canned_v1`` schema check - returns a list of violations ([] = ok)."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["canned file is not a JSON object"]
    if obj.get("schema") != CANNED_SCHEMA:
        errs.append(f"schema != {CANNED_SCHEMA}")
    for key in ("stimulus", "title", "population", "n_nodes_small", "nodes",
                "edges", "dt_ms", "window_ms", "n_steps", "events",
                "readout", "provenance"):
        if key not in obj:
            errs.append(f"missing key: {key}")
    if errs:
        return errs
    nodes = obj["nodes"]
    n_small = obj["n_nodes_small"]
    if len(nodes) != n_small:
        errs.append(f"len(nodes)={len(nodes)} != n_nodes_small={n_small}")
    for i, node in enumerate(nodes):
        if node.get("i") != i:
            errs.append(f"nodes[{i}].i != {i}")
        for f in ("type", "layer", "x", "y"):
            if f not in node:
                errs.append(f"nodes[{i}] missing {f}")
    for e in obj["edges"]:
        if len(e) != 3 or not (0 <= e[0] < n_small) or not (0 <= e[1] < n_small):
            errs.append(f"edge out of range: {e}")
    dt_ms, window_ms = float(obj["dt_ms"]), float(obj["window_ms"])
    if abs(obj["n_steps"] - round(window_ms / dt_ms)) > 0:
        errs.append("n_steps != round(window_ms/dt_ms)")
    prev = (-1.0, -1)
    for k, ev in enumerate(obj["events"]):
        if len(ev) != 2:
            errs.append(f"events[{k}] not [t_ms, node]")
            continue
        t_ms, node = ev
        step = t_ms / dt_ms
        if abs(step - round(step)) > 1e-6:
            errs.append(f"events[{k}] t_ms not on dt grid")
        if not (0 <= node < n_small):
            errs.append(f"events[{k}] node {node} out of range")
        if not (0.0 <= t_ms <= window_ms):
            errs.append(f"events[{k}] t_ms outside window")
        if (t_ms, node) < prev:
            errs.append(f"events[{k}] not sorted by (t_ms, node)")
        prev = (t_ms, node)
    ro = obj["readout"]
    if ro.get("decoded") not in ro.get("programs", {}):
        errs.append("readout.decoded not in readout.programs")
    prov = obj["provenance"]
    for key in _PROV_KEYS:
        if key not in prov:
            errs.append(f"provenance missing {key}")
    if "sim_cfg" in prov:
        for f in ("dt", "tau_v", "tau_g", "w_syn", "threshold"):
            if f not in prov["sim_cfg"]:
                errs.append(f"provenance.sim_cfg missing {f}")
    if obj.get("ethogram") is not None:
        etho = obj["ethogram"]
        for key in ("stats", "reference", "pulse_bouts", "sine_bouts"):
            if key not in etho:
                errs.append(f"ethogram missing {key}")
    return errs


def validate_manifest(manifest: Any, canned_dir: Path | None = None) -> list[str]:
    """``canned_manifest_v1`` check ([] = ok); optionally verifies file hashes."""
    errs: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not a JSON object"]
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errs.append(f"schema != {MANIFEST_SCHEMA}")
    for key in ("generated_utc", "code", "graph", "sim_cfg", "stimuli"):
        if key not in manifest:
            errs.append(f"missing key: {key}")
    if errs:
        return errs
    for f in ("git_sha", "generator"):
        if f not in manifest["code"]:
            errs.append(f"manifest.code missing {f}")
    for f in ("source", "sha256", "n_nodes", "n_edges"):
        if f not in manifest["graph"]:
            errs.append(f"manifest.graph missing {f}")
    stim_ids = set()
    for entry in manifest["stimuli"]:
        for f in ("stimulus", "file", "sha256", "n_events", "decoded"):
            if f not in entry:
                errs.append(f"stimuli entry missing {f}: {entry}")
        stim_ids.add(entry.get("stimulus"))
        if canned_dir is not None and "file" in entry:
            path = canned_dir / entry["file"]
            if not path.exists():
                errs.append(f"canned file absent: {entry['file']}")
            elif sha256_file(path) != entry.get("sha256"):
                errs.append(f"sha256 mismatch: {entry['file']}")
    missing = set(STIMULI) - stim_ids
    if missing:
        errs.append(f"stimuli absent from manifest: {sorted(missing)}")
    return errs


def write_manifest(
    generated: list[dict[str, Any]],
    spec_graph: dict[str, Any],
    graph_source: str,
    outdir: Path,
) -> dict[str, Any]:
    entries = []
    for canned in generated:
        path = outdir / f"{canned['stimulus']}.json"
        entries.append({
            "stimulus": canned["stimulus"],
            "title": canned["title"],
            "file": path.name,
            "sha256": sha256_file(path),
            "n_events": len(canned["events"]),
            "decoded": canned["readout"]["decoded"],
            "backend": canned["provenance"]["backend"],
        })
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "generated_utc": utcnow(),
        "code": {"git_sha": git_sha(), "generator": GENERATOR},
        "graph": {
            "source": graph_source,
            "sha256": graph_sha(spec_graph),
            "n_nodes": len(spec_graph["nodes"]),
            "n_edges": len(spec_graph["edges"]),
        },
        "sim_cfg": SIM_CFG,
        "note": "All traces are simulated, not recorded: LIF rollouts on a "
                "connectome wiring fixture, cached so the Space needs zero "
                "live sim calls and zero GPU (T-SHIP-7).",
        "stimuli": entries,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--all", action="store_true", help="regenerate every stimulus")
    ap.add_argument("--stimulus", choices=sorted(STIMULI), default=None)
    ap.add_argument("--outdir", type=Path, default=CANNED_DIR)
    ap.add_argument("--graph-dir", type=Path, default=None,
                    help="optional real data/graph build to subsample instead "
                         "of the embedded fixture")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for stim_id, spec in STIMULI.items():
            print(f"{stim_id:14s} [{spec['button']:17s}] {spec['population']}")
        return 0

    graph_source = "embedded_fixture"
    spec_graph = demo_graph_spec()
    if args.graph_dir is not None:
        real = load_real_subgraph(args.graph_dir)
        if real is None:
            print(f"[build_canned] no usable graph at {args.graph_dir}; "
                  "using embedded fixture", file=sys.stderr)
        else:
            spec_graph, graph_source = real, f"subgraph:{args.graph_dir}"

    if args.all:
        targets = sorted(STIMULI)
    elif args.stimulus:
        targets = [args.stimulus]
    else:
        ap.error("pass --all or --stimulus <id> (or --list)")

    generated = []
    for stim_id in targets:
        canned = generate(stim_id, spec_graph, graph_source, args.outdir)
        print(f"[build_canned] {stim_id:14s} {len(canned['events']):5d} events "
              f"-> decoded {canned['readout']['decoded']} "
              f"({canned['provenance']['backend']})")
        generated.append(canned)

    if args.all:
        manifest = write_manifest(generated, spec_graph, graph_source, args.outdir)
        errs = validate_manifest(manifest, args.outdir)
        if errs:
            print(f"[build_canned] manifest validation errors: {errs}",
                  file=sys.stderr)
            return 1
        print(f"[build_canned] wrote {args.outdir / 'manifest.json'} "
              f"({len(generated)} stimuli)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
