"""Control graph builders: rewired / permuted / wrong-animal / dense.

Implements the T-MODEL control suite (docs/plan.md section 3 T-MODEL-3/4/
5/11; EVAL.md per-test thresholds; docs/impl-spec.md "controls.py").
Every control rebuilds a ``GraphBundle`` in the same ``schema_v1`` layout
as the real graph (``flygpt.schema``) - the sim and the model never see
a different interface, so a margin over a control isolates exactly one
variable:

    rewire              configuration-model rewire preserving in-degree
                        AND out-degree per node and the (weight, sign)
                        multiset - the primary T-MODEL-3 control
    permuted_labels     ``type`` permuted among same-superclass nodes;
                        wiring untouched (T-MODEL-4)
    shuffled_weights    (weight, sign) pairs permuted across edges;
                        topology untouched (T-MODEL-11 / corpus slice
                        ``shuffled``)
    dense_mask_bundle   complete digraph minus self-loops - the edge set
                        behind the parameter-matched dense twin
                        (T-MODEL-5)
    no_edge_bundle      empty edge set - the direct-input floor
    wrong_animal_bundle C. elegans N2U hermaphrodite chemical connectome
                        (the T-MODEL-3b wrong-animal mask)
    linear_influence    normalized-adjacency path-integrated influence
                        matrix - the local stand-in for the BANC linear
                        influence-matrices baseline (T-MODEL-6; the real
                        parquets get loaded by evals when present)

C. elegans source: WormWiring "SI 5 Connectome adjacency matrices,
corrected July 2020" (Cook et al. 2019, Nature). Per the sheet legend:
for chemical connections rows are the pre-synaptic cell and columns the
post-synaptic cell; values are EM serial-section counts. The file is
cached under ``data/raw/worm/`` with a sha256 manifest at
``provenance/celegans_v1.json`` - the same pattern as ingest.py. Sign is
recorded as 0 (undetermined): SI5 publishes connectivity, not NT signs.
"""

from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import requests

from flygpt.schema import (
    GraphBundle,
    bundle_from_edges,
    rng_from_seed,
    sha256_file,
    utcnow,
)

# Dev-layout anchors (repo root = two levels above src/flygpt/). Callers
# in installed environments pass explicit paths; these defaults exist so
# the in-repo pipeline resolves the same locations as ingest.py.
REPO_ROOT = Path(__file__).resolve().parents[2]
WORM_DIR = REPO_ROOT / "data" / "raw" / "worm"
WORM_MANIFEST = REPO_ROOT / "provenance" / "celegans_v1.json"
WORM_URL = (
    "https://wormwiring.org/si/"
    "SI%205%20Connectome%20adjacency%20matrices,%20corrected%20July%202020.xlsx"
)
WORM_LOCAL_NAME = "SI5_connectome_adjacency_2020.xlsx"
WORM_SHEET = "hermaphrodite chemical"  # N2U adult chemical connectome
WORM_DATASET = "celegans-wormwiring:si5-2020"

_DL_CHUNK = 1024 * 1024
_DL_RETRIES = 3


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _edge_arrays(bundle: GraphBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(pre, post, weight, sign) edge arrays in the bundle's CSR order."""
    pre = np.repeat(np.arange(bundle.n_nodes, dtype=np.uint32), np.diff(bundle.out_indptr))
    return pre, bundle.out_indices.copy(), bundle.out_weight.copy(), bundle.out_sign.copy()


def _minimal_nodes(n_nodes: int, superclass: str) -> pl.DataFrame:
    """Minimal nodes table for synthetic-control bundles (bodyId = idx)."""
    return pl.DataFrame(
        {
            "bodyId": np.arange(n_nodes, dtype=np.int64),
            "node_idx": np.arange(n_nodes, dtype=np.int64),
            "type": pl.Series([None] * n_nodes, dtype=pl.String),
            "superclass": pl.Series([superclass] * n_nodes, dtype=pl.String),
        }
    )


def degree_sequences(bundle: GraphBundle) -> tuple[np.ndarray, np.ndarray]:
    """(in_degree, out_degree) per node - the rewire invariants."""
    return np.diff(bundle.in_indptr), np.diff(bundle.out_indptr)


# ---------------------------------------------------------------------------
# Rewire - degree + weight preserving configuration model (T-MODEL-3)
# ---------------------------------------------------------------------------


def rewire(bundle: GraphBundle, seed: int) -> GraphBundle:
    """Degree+weight-preserving rewiring via edge-swap MCMC.

    Repeatedly proposes a target swap on two edges ``(a->b, w_i)`` and
    ``(c->d, w_j)`` giving ``(a->d, w_i)`` and ``(c->b, w_j)``: each edge
    keeps its own (weight, sign), so out-degree, in-degree, the out-weight
    multiset per node, and the global (weight, sign) multiset are all
    preserved exactly. A proposal is rejected when it is a no-op
    (a==c or b==d), would create a self-loop (a==d or c==b), or would
    collide with an existing edge. Deterministic given ``seed``.
    """
    rng = rng_from_seed(("controls:rewire", int(seed), int(bundle.n_nodes), int(bundle.out_indices.size)))
    pre, post, weight, sign = _edge_arrays(bundle)
    n_edges = int(pre.size)
    if n_edges >= 2:
        eset = set(zip(pre.tolist(), post.tolist(), strict=True))
        target_swaps = 4 * n_edges
        max_attempts = max(2_000, 40 * n_edges)
        n_swaps = attempts = 0
        while attempts < max_attempts and n_swaps < target_swaps:
            i, j = (int(x) for x in rng.integers(0, n_edges, 2))
            attempts += 1
            if i == j:
                continue
            a, b = int(pre[i]), int(post[i])
            c, d = int(pre[j]), int(post[j])
            # reject no-ops (a==c, b==d) and new self-loops (a==d, c==b)
            if a in (c, d) or b in (c, d):
                continue
            if (a, d) in eset or (c, b) in eset:
                continue
            eset.discard((a, b))
            eset.discard((c, d))
            eset.add((a, d))
            eset.add((c, b))
            post[i] = d
            post[j] = b
            n_swaps += 1
    return bundle_from_edges(bundle.nodes, pre, post, weight, sign, cutoff=bundle.cutoff)


# ---------------------------------------------------------------------------
# Permuted labels - type->token remap inside superclass strata (T-MODEL-4)
# ---------------------------------------------------------------------------


def permuted_labels(bundle: GraphBundle, seed: int) -> GraphBundle:
    """Permute ``type`` labels among nodes of the same superclass.

    Strata = the ``superclass`` column of the nodes table (a single
    stratum when the column is absent). Within each stratum the non-null
    type labels are reshuffled among the typed nodes only - nulls stay
    null, so the manifest denominator and the typed fraction are
    unchanged. Wiring (all CSR arrays) is bit-identical; node rows keep
    every other column. Deterministic given ``seed``.
    """
    rng = rng_from_seed(("controls:permuted_labels", int(seed), int(bundle.n_nodes)))
    nodes = bundle.nodes
    types = nodes["type"].to_list()
    supers = nodes["superclass"].to_list() if "superclass" in nodes.columns else [""] * nodes.height

    strata: dict[str, list[int]] = {}
    for i, s in enumerate(supers):
        strata.setdefault(s or "", []).append(i)

    new_types = list(types)
    for idxs in strata.values():
        typed = [i for i in idxs if types[i] is not None]
        vals = [types[i] for i in typed]
        perm = rng.permutation(len(typed))
        for k, i in enumerate(typed):
            new_types[i] = vals[perm[k]]

    new_nodes = nodes.with_columns(pl.Series("type", new_types, dtype=pl.String))
    return replace(bundle, nodes=new_nodes)


# ---------------------------------------------------------------------------
# Shuffled weights - topology kept, weight+sign permuted (T-MODEL-11)
# ---------------------------------------------------------------------------


def shuffled_weights(bundle: GraphBundle, seed: int) -> GraphBundle:
    """Permute (weight, sign) pairs across edges; topology untouched.

    The joint permutation keeps each weight glued to its sign, so the
    (weight, sign) multiset is preserved exactly while the assignment of
    weights to edges is scrambled - the corpus slice ``shuffled`` and the
    T-MODEL-11 control. Deterministic given ``seed``.
    """
    rng = rng_from_seed(("controls:shuffled_weights", int(seed), int(bundle.out_indices.size)))
    pre, post, weight, sign = _edge_arrays(bundle)
    perm = rng.permutation(pre.size)
    return bundle_from_edges(
        bundle.nodes, pre, post, weight[perm], sign[perm], cutoff=bundle.cutoff
    )


# ---------------------------------------------------------------------------
# Dense / no-edge floor controls (T-MODEL-5 edge sets)
# ---------------------------------------------------------------------------


def dense_mask_bundle(n_nodes: int, weight: float = 1.0) -> GraphBundle:
    """Complete digraph minus self-loops on a minimal nodes table."""
    n = int(n_nodes)
    if n < 1:
        raise ValueError("n_nodes must be >= 1")
    pre = np.repeat(np.arange(n, dtype=np.uint32), n)
    post = np.tile(np.arange(n, dtype=np.uint32), n)
    keep = pre != post
    return bundle_from_edges(
        _minimal_nodes(n, "dense"),
        pre[keep],
        post[keep],
        np.full(int(keep.sum()), int(round(weight)), dtype=np.int32),
        np.ones(int(keep.sum()), dtype=np.int8),
        cutoff=0,
    )


def no_edge_bundle(n_nodes: int) -> GraphBundle:
    """Empty edge set - the direct-input floor control."""
    n = int(n_nodes)
    if n < 1:
        raise ValueError("n_nodes must be >= 1")
    empty32 = np.empty(0, dtype=np.uint32)
    return bundle_from_edges(
        _minimal_nodes(n, "no_edge"),
        empty32,
        empty32,
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int8),
        cutoff=0,
    )


# ---------------------------------------------------------------------------
# Linear influence proxy (T-MODEL-6 baseline stand-in)
# ---------------------------------------------------------------------------


def linear_influence(bundle: GraphBundle, hops: int = 3, decay: float = 0.5) -> np.ndarray:
    """Normalized-adjacency path-integrated influence matrix, (V, V) f32.

    ``P = D_out^-1 * A`` where ``A[pre, post] = weight`` (unsigned synapse
    count - the influence baseline cares about connectivity strength,
    not NT sign) and rows with zero out-strength stay zero. Returns

        M = sum_{h=1..hops} decay ** (h - 1) * P ** h

    so M[i, j] is the discounted total influence of node i on node j
    along paths of length <= ``hops`` (direct edges weighted 1.0, a k-hop
    path decayed by decay**(k-1)). This is the documented local proxy for
    the BANC linear influence-matrices baseline of T-MODEL-6; evals load
    the real parquets when present.
    """
    n = bundle.n_nodes
    a = np.zeros((n, n), dtype=np.float32)
    np.add.at(
        a,
        (np.repeat(np.arange(n), np.diff(bundle.out_indptr)), bundle.out_indices),
        bundle.out_weight.astype(np.float32),
    )
    rowsum = a.sum(axis=1, keepdims=True)
    p = np.divide(a, rowsum, out=np.zeros_like(a), where=rowsum > 0)
    m = np.zeros_like(p)
    term = p.copy()
    for h in range(1, int(hops) + 1):
        m += (decay ** (h - 1)) * term
        term = term @ p
    return m.astype(np.float32)


# ---------------------------------------------------------------------------
# Wrong-animal control - C. elegans connectome (T-MODEL-3b)
# ---------------------------------------------------------------------------


def _worm_class(name: str) -> str:
    """Coarse cell-class guess from a neuron name (control-grade strata).

    Strips trailing member digits then one trailing side/membership
    suffix (L/R/M/D/V or DL/DR/VL/VR): I1L->I1, VC01->VC, IL1DL->IL1,
    RIPL->RIP. Used only for the ``type`` column so strata-aware
    machinery (permuted_labels, vocab shapes) works on the worm bundle.
    """
    base = re.sub(r"\d+$", "", name)
    stripped = re.sub(r"(DL|DR|VL|VR|L|R|M|D|V)$", "", base)
    return stripped or base


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(f"{ns}t")) for si in root]


def _xlsx_sheet_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    """Resolve a sheet name to its ``xl/worksheets/...`` part path."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    relns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    pkg = "{http://schemas.openxmlformats.org/package/2006/relationships}"
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rid = None
    for sheet in wb.iter(f"{ns}sheet"):
        if sheet.get("name") == sheet_name:
            rid = sheet.get(f"{relns}id")
            break
    if rid is None:
        names = [s.get("name") for s in wb.iter(f"{ns}sheet")]
        raise KeyError(f"sheet {sheet_name!r} not in workbook; sheets: {names}")
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    for rel in rels.iter(f"{pkg}Relationship"):
        if rel.get("Id") == rid:
            return "xl/" + rel.get("Target")
    raise KeyError(f"no relationship {rid} for sheet {sheet_name!r}")


def _col_index(cell_ref: str) -> int:
    """Excel column letters -> 1-based index (D4 -> 4)."""
    n = 0
    for ch in cell_ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch) - 64)
    return n


def _xlsx_rows(zf: zipfile.ZipFile, sheet_path: str) -> dict[int, dict[int, object]]:
    """Parse a worksheet into {row_number: {col_index: value}}.

    ``t="s"`` cells resolve through the shared-string table; numeric
    cells come back as float/int; empty cells are absent.
    """
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    shared = _xlsx_shared_strings(zf) if "xl/sharedStrings.xml" in zf.namelist() else []
    root = ET.fromstring(zf.read(sheet_path))
    rows: dict[int, dict[int, object]] = {}
    for row in root.iter(f"{ns}row"):
        rnum = int(row.get("r"))
        cells: dict[int, object] = {}
        for c in row.iter(f"{ns}c"):
            ref = c.get("r")
            col = _col_index(ref)
            v = c.find(f"{ns}v")
            if v is None or v.text is None:
                continue
            kind = c.get("t")
            if kind == "s":
                cells[col] = shared[int(v.text)]
            elif kind in ("str", "inlineStr"):
                cells[col] = v.text
            else:
                x = float(v.text)
                cells[col] = int(x) if x == int(x) else x
        rows[rnum] = cells
    return rows


def _worm_from_xlsx(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Parse a WormWiring SI5 adjacency sheet.

    Layout (verified against the 2020-corrected file): a header row holds
    the post-synaptic cell names in columns D+; each data row holds the
    pre-synaptic cell name in column C and EM section counts in the
    body. Node set = the row-name list (all neurons; post-only names such
    as body-wall muscle and hypodermis are dropped from edges, not from
    counts of dropped cells reported by callers).
    """
    with zipfile.ZipFile(path) as zf:
        sheet = WORM_SHEET
        names_present = list(zf.namelist())
        if "xl/workbook.xml" not in names_present:
            raise ValueError(f"{path} is not an .xlsx workbook")
        rows = _xlsx_rows(zf, _xlsx_sheet_path(zf, sheet))

    # the column-name row is the densest string row (SI5 has group-label
    # rows above it - PHARYNX / SENSORY NEURONS - with only a few cells)
    header_row = max(
        rows,
        key=lambda r: sum(
            1 for c, v in rows[r].items() if c >= 4 and isinstance(v, str)
        ),
        default=None,
    )
    n_header = (
        sum(1 for c, v in rows[header_row].items() if c >= 4 and isinstance(v, str))
        if header_row is not None
        else 0
    )
    if n_header < 5:
        raise ValueError(f"{path}:{sheet} has no dense header row of post cell names")
    post_of_col = {c: v for c, v in rows[header_row].items() if c >= 4 and isinstance(v, str)}

    pre_names: list[str] = []
    seen: set[str] = set()
    edges: list[tuple[str, str, int]] = []
    for rnum in sorted(rows):
        if rnum <= header_row:
            continue
        name = rows[rnum].get(3)
        if not isinstance(name, str):
            continue
        if name not in seen:
            seen.add(name)
            pre_names.append(name)
        for col, val in rows[rnum].items():
            if col >= 4 and isinstance(val, (int, float)) and val > 0:
                post_name = post_of_col.get(col)
                if post_name is not None:
                    edges.append((name, post_name, int(val)))

    keep = [(a, b, w) for a, b, w in edges if b in seen]
    idx = {name: i for i, name in enumerate(pre_names)}
    pre = np.fromiter((idx[a] for a, _, _ in keep), dtype=np.uint32, count=len(keep))
    post = np.fromiter((idx[b] for _, b, _ in keep), dtype=np.uint32, count=len(keep))
    weight = np.fromiter((w for _, _, w in keep), dtype=np.int32, count=len(keep))
    return pre_names, pre, post, weight


def _worm_from_edgelist(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Parse a local edge-list fixture: ``pre,post,weight[,sign]`` CSV/TSV."""
    text = Path(path).read_text()
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",\t")
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    required = {"pre", "post", "weight"}
    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError(f"{path}: edge list needs columns {sorted(required)}")
    names: list[str] = []
    seen: set[str] = set()
    pre_l: list[int] = []
    post_l: list[int] = []
    w_l: list[int] = []
    for row in rows:
        for key in ("pre", "post"):
            if row[key] not in seen:
                seen.add(row[key])
                names.append(row[key])
        pre_l.append(names.index(row["pre"]))
        post_l.append(names.index(row["post"]))
        w_l.append(int(float(row["weight"])))
    return (
        names,
        np.asarray(pre_l, dtype=np.uint32),
        np.asarray(post_l, dtype=np.uint32),
        np.asarray(w_l, dtype=np.int32),
    )


def _worm_nodes(names: list[str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "bodyId": np.arange(len(names), dtype=np.int64),
            "node_idx": np.arange(len(names), dtype=np.int64),
            "name": pl.Series(names, dtype=pl.String),
            "type": pl.Series([_worm_class(n) for n in names], dtype=pl.String),
            "superclass": pl.Series(["celegans_n2u"] * len(names), dtype=pl.String),
        }
    )


def _ensure_worm_file(worm_dir: Path, manifest_path: Path) -> Path:
    """Return the cached WormWiring file, downloading + manifesting if needed.

    With an existing manifest the cached file's sha256 must match it
    (T-BUILD-5 fail-closed). When offline, raises a clear error - tests
    pass a local fixture to ``wrong_animal_bundle(path=...)`` instead.
    """
    dest = worm_dir / WORM_LOCAL_NAME
    expected = None
    if manifest_path.exists():
        try:
            files = json.loads(manifest_path.read_text()).get("files", [])
            expected = files[0].get("sha256") if files else None
        except (json.JSONDecodeError, KeyError, IndexError):
            expected = None
    if dest.exists():
        sha = sha256_file(dest)
        if expected is not None and sha != expected:
            raise RuntimeError(
                f"sha256 mismatch for cached {dest}: {sha} != manifest {expected} "
                "- fail closed per T-BUILD-5; delete the file to redownload"
            )
        return dest

    worm_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".partial")
    last_exc: Exception | None = None
    for attempt in range(1, _DL_RETRIES + 1):
        try:
            with requests.get(WORM_URL, stream=True, timeout=120) as r:
                r.raise_for_status()
                with part.open("wb") as out:
                    for block in r.iter_content(_DL_CHUNK):
                        out.write(block)
            part.rename(dest)
            break
        except requests.RequestException as exc:
            last_exc = exc
            print(f"[retry {attempt}/{_DL_RETRIES}] {WORM_LOCAL_NAME}: {exc}", flush=True)
    if not dest.exists():
        raise RuntimeError(
            "C. elegans adjacency download failed (offline?) - "
            "pass path=<local adjacency .xlsx/.csv> to wrong_animal_bundle"
        ) from last_exc

    sha = sha256_file(dest)
    if expected is not None and sha != expected:
        dest.unlink()
        raise RuntimeError(f"downloaded {dest} sha256 {sha} != manifest {expected}")
    manifest = {
        "dataset": WORM_DATASET,
        "title": "C. elegans connectome adjacency matrices (Cook et al. 2019, corrected July 2020)",
        "source_url": WORM_URL,
        "index_page": "https://wormwiring.org/pages/adjacency.html",
        "license": "open data (wormwiring.org Emmons lab distribution; wrong-animal control use per plan)",
        "citation": "Cook et al. 2019 Nature - Whole-animal connectomes of both C. elegans sexes",
        "conventions": {
            "sheets": "hermaphrodite chemical / male chemical (gap-jn sheets documented, unused)",
            "matrix": "rows = pre-synaptic cell, columns = post-synaptic cell, values = EM section counts",
            "sign": "0 (undetermined) - SI5 publishes connectivity, not NT signs",
            "role": "wrong-animal control graph; species mismatch is the manipulated variable",
        },
        "code": {"git_sha": "uncommitted", "generator": "flygpt controls.py worm loader"},
        "files": [
            {
                "key": "si5_adjacency",
                "local_name": WORM_LOCAL_NAME,
                "https_url": WORM_URL,
                "bytes": dest.stat().st_size,
                "sha256": sha,
                "downloaded_utc": utcnow(),
            }
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return dest


def wrong_animal_bundle(
    path: Path | str | None = None,
    *,
    worm_dir: Path = WORM_DIR,
    manifest_path: Path = WORM_MANIFEST,
) -> GraphBundle:
    """C. elegans N2U hermaphrodite chemical connectome as a GraphBundle.

    ``path=None`` resolves the cached WormWiring SI5 workbook under
    ``data/raw/worm/`` (downloading + writing the sha256 manifest first
    when absent). ``path=<file>`` loads a local fixture instead:
    ``.xlsx`` files are parsed in the SI5 layout (stdlib zipfile + XML,
    no spreadsheet dependency); any other suffix is read as an edge-list
    ``pre,post,weight`` CSV/TSV.

    Nodes keep the file's row order (bodyId = node idx, name = neuron
    name, type = coarse class guess via ``_worm_class``, superclass =
    ``celegans_n2u``). Postsynaptic cells absent from the pre set -
    muscle, hypodermis, gut - are dropped from edges (documented in the
    module docstring). Sign = 0 (undetermined) throughout.
    """
    if path is None:
        path = _ensure_worm_file(worm_dir, manifest_path)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"worm adjacency not found: {path}")
    if path.suffix.lower() == ".xlsx":
        names, pre, post, weight = _worm_from_xlsx(path)
    else:
        names, pre, post, weight = _worm_from_edgelist(path)
    return bundle_from_edges(
        _worm_nodes(names),
        pre,
        post,
        weight,
        np.zeros(pre.size, dtype=np.int8),  # sign 0 = undetermined (SI5)
        cutoff=0,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worm", action="store_true",
                    help="build the C. elegans control bundle (downloads on first use)")
    ap.add_argument("--stats", action="store_true", help="print bundle summary")
    args = ap.parse_args(argv)
    if args.worm:
        bundle = wrong_animal_bundle()
        print(
            f"wrong_animal_bundle: {bundle.n_nodes} nodes, "
            f"{bundle.out_indices.size} edges, {len(bundle.manifest_types)} types",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
