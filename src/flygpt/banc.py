"""BANC female connectome: GCS download, CSR build, male<->female alignment.

Phase 4 scope - docs/plan.md "Phase 4 - Sex conditioning (BANC)",
docs/impl-spec.md "banc.py", EVAL.md T-SEX-1..6 + section 1.9, RISKS.md
R4 (BANC-sim unvalidated), R9 (convention drift MaleCNS vs BANC), R11
(sex-claim confound - "specimen-conditioned" wording, never bare "sex").

Pipeline:
    download_banc               bucket -> data/raw/banc + provenance/banc_v888.json
    build_banc_graph            meta+edgelist -> nodes/edges parquet + csr npz
                                + provenance/banc_graph_v1.json + results report
    alignment_table             curated male<->female type alignment
    sex_conditioned_vocab       union type vocab (aligned types share a token)
    female_graph_in_male_space  BANC edges re-indexed onto the union vocab
    female_stimulus_map         male program targets -> female homolog types

Conventions that differ from the male build are RECORDED, never aligned
(R9: document asymmetries rather than silently matching them):
- autapses: excluded upstream in the BANC simple edgelist (~2.1%); the
  male build keeps them (configs/graph_v1.yaml autapse_policy: keep).
  The build counts any residual autapse rows and drops them.
- sign_source: per_neuron. The BANC simple edgelist carries no per-T-bar
  NT table, so edge sign = the pre neuron's neurotransmitter_predicted
  class against the same inhibitory set {gaba, glutamate} as the male
  rule; the male build uses the per-edge T-bar majority rule. The
  divergence is recorded verbatim in the manifest (T-SEX-3 honesty).
- synapse_snapshot: v2 (size>=5 detector cutoff - matches the male >=5
  read convention; v3 = size>=10 is kept as a documented slice).
- ids: banc_888_id strings (not int64 bodyIds); nodes are sorted by the
  string id so node_idx is deterministic.

Usage:
    python -m flygpt.banc --config configs/banc_v1.yaml --download
    python -m flygpt.banc --config configs/banc_v1.yaml --build
    python -m flygpt.banc --config configs/banc_v1.yaml --align
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import requests
import yaml

GCS_BUCKET = "lee-lab_brain-and-nerve-cord-fly-connectome"
DATASET_ID = "banc:v888"
DATASET_LICENSE = "CC-BY-4.0"
DATASET_CITATION = (
    "Bates, Phelps, Kim et al. 2026, Nature - BANC female "
    "brain-and-nerve-cord connectome (Lee lab)"
)

CHUNK = 8 * 1024 * 1024  # 8 MiB streaming chunks
RETRIES = 5

SNAPSHOT = "v888"
SYNAPSE_SNAPSHOT = "v2"

# Inhibitory NT classes - identical set to the male nt_sign_rule so the
# per-neuron sign convention matches the male per-edge convention's classes.
INHIBITORY_NT = ("gaba", "glutamate")
# neurotransmitter_predicted values treated as "no usable prediction":
# flagged, never guessed (same low-confidence policy as the male build).
UNKNOWN_NT = frozenset({"", "unknown", "none", "nan", "na", "unclassified"})

SIGN_UNDETERMINED = 0
SIGN_EXCITATORY = 1
SIGN_INHIBITORY = -1

# Alignment iso classes (configs/banc_v1.yaml alignment.iso_classes).
ISO_ISOMORPHIC = "isomorphic"
ISO_DIMORPHIC = "dimorphic"
ISO_MALE_SPECIFIC = "male-specific"
ISO_FEMALE_SPECIFIC = "female-specific"
ISO_MALE_UNMATCHED = "male-only-unmatched"
ISO_FEMALE_UNMATCHED = "female-only-unmatched"
MATCHED_ISO = frozenset({ISO_ISOMORPHIC, ISO_DIMORPHIC})
FEMALE_ONLY_ISO = frozenset({ISO_FEMALE_SPECIFIC, ISO_FEMALE_UNMATCHED})
MALE_ONLY_ISO = frozenset({ISO_MALE_SPECIFIC, ISO_MALE_UNMATCHED})

# Reviewed-matches decision vocabularies (normalized lower/strip).
_DECISION_ACCEPT = frozenset(
    {"1", "true", "t", "yes", "y", "accept", "accepted", "match", "confirmed", "keep"}
)
_DECISION_REJECT = frozenset(
    {"0", "false", "f", "no", "n", "reject", "rejected", "nomatch", "no_match", "none", "drop", ""}
)

# Meta columns carried into nodes.parquet (subset of the 79 meta cols).
META_CARRY_COLUMNS = (
    "root_626",
    "root_888",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "hemilineage",
    "region",
    "side",
    "flow",
    "proofread",
    "position",
    "fafb_cell_type",
    "manc_cell_type",
    "malecns_cell_type",
    "malecns_cell_type_match",
    "neurotransmitter_predicted",
    "neurotransmitter_score",
    "neurotransmitter_verified",
)


@dataclass(frozen=True)
class BancFile:
    """One canonical upstream object in the BANC public bucket."""

    key: str
    name: str  # object path relative to the bucket root
    role: str  # what Phase 4 uses it for
    required: bool = True

    @property
    def local_name(self) -> str:
        """Basename used for the on-disk copy under data/raw/banc/."""
        return self.name.rsplit("/", 1)[-1]


# Registry over the deposit files Phase 4 consumes. Optional files are
# still recorded in the manifest (size + bucket md5) so provenance covers
# the registered set even when they are not downloaded.
BANC_REGISTRY: tuple[BancFile, ...] = (
    BancFile(
        "meta",
        "compiled_data/banc_888/banc_888_meta.feather",
        "node universe: types / classes / sides / cross-dataset matches / NT",
    ),
    BancFile(
        "edgelist_v2",
        "compiled_data/banc_888/banc_888_edgelist_simple_v2.feather",
        "canonical (pre, post, count) edges - size>=5 detector snapshot",
    ),
    BancFile(
        "edgelist_v3",
        "compiled_data/banc_888/banc_888_edgelist_simple_v3.feather",
        "convention variant: size>=10 detector snapshot",
        required=False,
    ),
    BancFile(
        "nt_prediction",
        "compiled_data/banc_888/banc_888_neurotransmitter_prediction_v2.csv",
        "per-neuron NT predictions (backup for the meta NT columns)",
        required=False,
    ),
    BancFile(
        "malecns_matches",
        "nblast/banc_malecns_reviewed_matches.csv",
        "curated BANC<->maleCNS match decisions (alignment override)",
        required=False,
    ),
    BancFile(
        "malecns_nblast",
        "nblast/banc_malecns_v0.9_nblast.feather",
        "NBLAST backup alignment table (~900 MB, on demand only)",
        required=False,
    ),
)

REQUIRED_KEYS = tuple(f.key for f in BANC_REGISTRY if f.required)

# Manifest registry key -> BancSourcePaths field.
KEY_TO_FIELD = {
    "meta": "meta",
    "edgelist_v2": "edgelist",
    "edgelist_v3": "edgelist_v3",
    "nt_prediction": "nt_prediction",
    "malecns_matches": "malecns_matches",
    "malecns_nblast": "malecns_nblast",
}
REQUIRED_SOURCE_KEYS = ("meta", "edgelist_v2")


@dataclass(frozen=True)
class BancSourcePaths:
    """Resolved local paths for the BANC source files (meta+edges required)."""

    meta: Path
    edgelist: Path
    edgelist_v3: Path | None = None
    nt_prediction: Path | None = None
    malecns_matches: Path | None = None
    malecns_nblast: Path | None = None

    def by_key(self, key: str) -> Path | None:
        return getattr(self, KEY_TO_FIELD[key])


@dataclass
class BancBuildStats:
    """Numbers that land in provenance/banc_graph_*.json + results/*.json."""

    wall_seconds: float = 0.0
    n_nodes: int = 0
    n_nodes_typed: int = 0
    n_nonnumeric_ids: int = 0
    meta_columns_absent: list[str] = field(default_factory=list)
    nt_column_present: bool = True
    edges_raw_rows: int = 0
    edges_both_meta: int = 0
    edge_rows_dropped_unannotated: int = 0
    weight_sum_dropped: int = 0
    autapses_found: int = 0
    duplicate_pairs_merged: int = 0
    edges_at_cutoff: dict[str, int] = field(default_factory=dict)
    raw_edges_at_cutoff: dict[str, int] = field(default_factory=dict)
    nodes_nt_inhibitory: int = 0
    nodes_nt_excitatory: int = 0
    nodes_nt_undetermined: int = 0
    nt_edges_inhibitory: int = 0
    nt_edges_excitatory: int = 0
    nt_edges_low_confidence: int = 0


def gcs_url(name: str) -> str:
    return f"https://storage.googleapis.com/{GCS_BUCKET}/{name}"


def list_bucket(prefix: str = "") -> dict[str, dict]:
    """Return {object_name: {size, md5_b64}} for objects under ``prefix``.

    Names are the FULL object paths (registry names are bucket-root
    relative, unlike the single-prefix male flat-connectome dir).
    """
    api = (
        f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
        f"?fields=items(name,size,md5Hash),nextPageToken"
    )
    if prefix:
        api += f"&prefix={prefix}"
    out: dict[str, dict] = {}
    token = ""
    while True:
        url = api + (f"&pageToken={token}" if token else "")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("items", []):
            out[item["name"]] = {"size": int(item["size"]), "md5_b64": item["md5Hash"]}
        token = payload.get("nextPageToken", "")
        if not token:
            return out


def _hash_file(path: Path) -> tuple[str, str, int]:
    """Return (sha256_hex, md5_b64, size) for a local file."""
    sha, md5, size = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            sha.update(block)
            md5.update(block)
            size += len(block)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode(), size


def download_one(
    src: BancFile, dest_dir: Path, meta: dict, getter: Any = None
) -> dict:
    """Download one object with resume + integrity check; manifest entry.

    ``getter`` replaces ``requests.get`` (test seam - signature compatible:
    called as ``getter(url, headers=..., stream=True, timeout=...)``).
    """
    get = getter or requests.get
    dest = dest_dir / src.local_name
    part = dest.with_suffix(dest.suffix + ".partial")
    expected_size = meta["size"]
    expected_md5 = meta["md5_b64"]

    if dest.exists():
        sha, md5, size = _hash_file(dest)
        if size == expected_size and md5 == expected_md5:
            print(f"[skip] {src.local_name} already complete", flush=True)
            return _entry(src, meta, sha, md5, size)
        print(f"[redo] {src.local_name} failed integrity check, redownloading", flush=True)
        dest.unlink()

    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > expected_size:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with get(gcs_url(src.name), headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                if have and r.status_code != 206:
                    # server ignored the Range request - restart cleanly
                    have = 0
                    part.unlink(missing_ok=True)
                sha, md5, size = (
                    hashlib.sha256(),
                    hashlib.md5(),
                    have,
                )
                if have and part.exists():
                    with part.open("rb") as fh:
                        for block in iter(lambda: fh.read(CHUNK), b""):
                            sha.update(block)
                            md5.update(block)
                t0 = time.time()
                with part.open("ab") as out:
                    for block in r.iter_content(CHUNK):
                        out.write(block)
                        sha.update(block)
                        md5.update(block)
                        size += len(block)
                rate = (size - have) / max(time.time() - t0, 1e-6) / 1e6
                print(
                    f"[dl] {src.local_name}: {size / 1e9:.2f} GB at {rate:.0f} MB/s",
                    flush=True,
                )
        except Exception as exc:
            print(f"[retry {attempt}/{RETRIES}] {src.local_name}: {exc}", flush=True)
            time.sleep(min(2**attempt, 30))
            continue

        md5_b64 = base64.b64encode(md5.digest()).decode()
        if size != expected_size or md5_b64 != expected_md5:
            print(
                f"[retry {attempt}/{RETRIES}] {src.local_name}: integrity mismatch "
                f"(size {size} vs {expected_size})",
                flush=True,
            )
            continue
        part.rename(dest)
        return _entry(src, meta, sha.hexdigest(), md5_b64, size)

    raise RuntimeError(f"download failed after {RETRIES} attempts: {src.local_name}")


def _entry(src: BancFile, meta: dict, sha: str, md5_b64: str, size: int) -> dict:
    return {
        "key": src.key,
        "gcs_object": f"gs://{GCS_BUCKET}/{src.name}",
        "https_url": gcs_url(src.name),
        "local_name": src.local_name,
        "role": src.role,
        "bytes": size,
        "md5_gcs": md5_b64,
        "sha256": sha,
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _prefixes_for(sources: list[BancFile]) -> list[str]:
    """Distinct bucket dir prefixes covering the target objects."""
    return sorted({f.name.rsplit("/", 1)[0] + "/" for f in sources if "/" in f.name})


def download_banc(
    keys: list[str] | tuple[str, ...] | None = None,
    dest_dir: Path = Path("data/raw/banc"),
    manifest_path: Path = Path("provenance/banc_v888.json"),
    *,
    listing: Any = None,
    getter: Any = None,
) -> dict:
    """Download registry files and write ``provenance/banc_v888.json``.

    ``keys`` selects registry keys (None -> required set). ``listing``
    replaces the bucket listing call and ``getter`` the HTTP fetch - both
    are test seams so the whole manifest path is exercised without
    network. Returns the manifest dict.
    """
    from flygpt.schema import git_sha

    lister = listing or list_bucket
    if keys is None:
        targets = [f for f in BANC_REGISTRY if f.required]
    else:
        wanted = set(keys)
        targets = [f for f in BANC_REGISTRY if f.key in wanted]
        if len(targets) != len(wanted):
            raise KeyError(f"unknown keys: {wanted - {f.key for f in targets}}")

    bucket: dict[str, dict] = {}
    for prefix in _prefixes_for(list(BANC_REGISTRY)):
        bucket.update(lister(prefix))
    missing = [f.name for f in BANC_REGISTRY if f.name not in bucket]
    if missing:
        raise RuntimeError(f"registry files absent from bucket listing: {missing}")

    # smallest first so schema inspection can start while big files stream
    targets = sorted(targets, key=lambda f: bucket[f.name]["size"])
    dest_dir.mkdir(parents=True, exist_ok=True)
    entries = [download_one(src, dest_dir, bucket[src.name], getter=getter) for src in targets]

    downloaded = {e["local_name"]: e for e in entries}
    files = []
    for src in BANC_REGISTRY:
        if src.local_name in downloaded:
            files.append(downloaded[src.local_name])
            continue
        meta = bucket.get(src.name, {})
        files.append(
            {
                "key": src.key,
                "gcs_object": f"gs://{GCS_BUCKET}/{src.name}",
                "https_url": gcs_url(src.name),
                "local_name": None,
                "role": src.role,
                "bytes": meta.get("size"),
                "md5_gcs": meta.get("md5_b64"),
                "sha256": None,
                "downloaded_utc": None,
                "not_downloaded": True,
            }
        )
    manifest = {
        "dataset": DATASET_ID,
        "title": "BANC v888 brain-and-nerve-cord connectome (female)",
        "source_bucket": f"gs://{GCS_BUCKET}/",
        "license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "conventions": {
            "snapshot": SNAPSHOT,
            "synapse_snapshot": SYNAPSE_SNAPSHOT,
            "synapse_snapshot_rationale": (
                "v2 = size>=5 detector cutoff, chosen to match the male >=5 "
                "read convention; v3 = size>=10 kept as a documented slice"
            ),
            "autapse_policy": "excluded_at_build",
            "autapse_note": (
                "~2.1% of autapses excluded upstream in the BANC simple "
                "edgelist; asymmetry vs male 'keep' documented, never aligned"
            ),
            "edge_weight_semantics": "count = synapse count per (pre, post); norm = count/post_count",
        },
        "code": {"git_sha": git_sha(), "generator": "flygpt banc.py download_banc"},
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] wrote {manifest_path}", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Graph build: banc_888 meta + simple edgelist -> nodes/edges parquet + CSR
# ---------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def resolve_banc_sources(manifest: dict, raw_dir: Path) -> BancSourcePaths:
    """Map manifest keys to local paths; optional files wired when present."""
    by_key = {f["key"]: f for f in manifest["files"]}
    missing = [
        k
        for k in REQUIRED_SOURCE_KEYS
        if k not in by_key or by_key[k].get("local_name") is None
    ]
    if missing:
        raise FileNotFoundError(f"manifest lacks downloaded files for keys: {missing}")

    def local(key: str) -> Path | None:
        entry = by_key.get(key)
        if entry and entry.get("local_name"):
            return raw_dir / entry["local_name"]
        return None

    return BancSourcePaths(
        meta=local("meta"),
        edgelist=local("edgelist_v2"),
        edgelist_v3=local("edgelist_v3"),
        nt_prediction=local("nt_prediction"),
        malecns_matches=local("malecns_matches"),
        malecns_nblast=local("malecns_nblast"),
    )


def verify_banc_source_hashes(manifest: dict, sources: BancSourcePaths) -> dict[str, str]:
    """T-BUILD-5 analogue: recompute sha256 of required inputs; fail closed."""
    from flygpt.schema import sha256_file

    recorded = {f["key"]: f["sha256"] for f in manifest["files"]}
    actual = {}
    for key in REQUIRED_SOURCE_KEYS:
        path = sources.by_key(key)
        digest = sha256_file(path)
        if digest != recorded[key]:
            raise RuntimeError(
                f"sha256 mismatch for {key} ({path.name}): "
                f"{digest} != manifest {recorded[key]}"
            )
        actual[key] = digest
    return actual


def _batches(path: Path, columns: list[str]):
    """Yield record batches of selected columns from a feather/IPC file."""
    reader = ipc.open_file(path)
    names = reader.schema.names
    keep = [c for c in columns if c in names]
    if len(keep) != len(columns):
        raise KeyError(f"{path.name} missing columns {set(columns) - set(keep)}")
    for i in range(reader.num_record_batches):
        yield reader.get_record_batch(i).select(keep)


def _lookup_sorted_ids(sorted_ids: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Row index into ``sorted_ids`` per key (string ids), or -1."""
    if sorted_ids.size == 0:
        return np.full(keys.shape, -1, dtype=np.int64)
    keys = np.asarray(keys, dtype=sorted_ids.dtype)
    pos = np.searchsorted(sorted_ids, keys)
    pos_c = np.clip(pos, 0, sorted_ids.size - 1)
    found = sorted_ids[pos_c] == keys
    return np.where(found, pos_c, -1).astype(np.int64)


def neuron_nt_signs(nodes: pl.DataFrame, inhibitory: tuple[str, ...]) -> np.ndarray:
    """Per-neuron sign i8 from neurotransmitter_predicted.

    inhibitory class -> -1; empty/unknown/null -> 0 (flagged via the
    nt_low_confidence edge column downstream, never guessed); any other
    predicted class -> +1.
    """
    n = nodes.height
    if "neurotransmitter_predicted" not in nodes.columns:
        return np.zeros(n, dtype=np.int8)
    low = (
        nodes["neurotransmitter_predicted"]
        .fill_null("")
        .str.to_lowercase()
        .str.strip_chars()
        .to_numpy()
    )
    return np.where(
        np.isin(low, list(inhibitory)),
        SIGN_INHIBITORY,
        np.where(np.isin(low, list(UNKNOWN_NT)), SIGN_UNDETERMINED, SIGN_EXCITATORY),
    ).astype(np.int8)


def load_banc_nodes(
    sources: BancSourcePaths, cfg: dict[str, Any], stats: BancBuildStats
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Node universe: every banc_888_meta row, sorted by banc_888_id.

    Returns (nodes df with node_idx, sorted string id array, per-neuron
    sign i8). Isolated neurons are retained exactly like the male build.
    """
    meta = pl.from_arrow(feather.read_table(sources.meta))
    if "banc_888_id" not in meta.columns:
        raise KeyError(f"{sources.meta.name} missing banc_888_id column")
    meta = meta.sort("banc_888_id")
    if not meta["banc_888_id"].is_unique().all():
        raise ValueError("banc_888_meta contains duplicate banc_888_id rows")
    meta = meta.with_row_index("node_idx")

    carry = [c for c in META_CARRY_COLUMNS if c in meta.columns]
    stats.meta_columns_absent = [c for c in META_CARRY_COLUMNS if c not in meta.columns]
    nodes = meta.select(["banc_888_id", "node_idx", *carry])
    if "cell_type" in nodes.columns:
        nodes = nodes.with_columns(pl.col("cell_type").alias("type"))
    else:
        nodes = nodes.with_columns(pl.lit(None, dtype=pl.Utf8).alias("type"))
    # GraphBundle API compat: numeric mirror of the string id. BANC root
    # ids are decimal strings; a non-numeric id yields null (counted).
    nodes = nodes.with_columns(
        pl.col("banc_888_id").cast(pl.Int64, strict=False).alias("bodyId")
    )
    stats.n_nonnumeric_ids = int(nodes["bodyId"].null_count())

    stats.nt_column_present = "neurotransmitter_predicted" in nodes.columns
    inhib = tuple(cfg.get("nt_sign_rule", {}).get("inhibitory_classes", INHIBITORY_NT))
    node_sign = neuron_nt_signs(nodes, inhib)
    stats.nodes_nt_inhibitory = int((node_sign == SIGN_INHIBITORY).sum())
    stats.nodes_nt_excitatory = int((node_sign == SIGN_EXCITATORY).sum())
    stats.nodes_nt_undetermined = int((node_sign == SIGN_UNDETERMINED).sum())

    ids_sorted = np.ascontiguousarray(
        nodes["banc_888_id"].to_numpy().astype(str)
    )
    return nodes, ids_sorted, node_sign


def load_banc_edges(
    sources: BancSourcePaths,
    ids_sorted: np.ndarray,
    cfg: dict[str, Any],
    cutoffs: list[int],
    stats: BancBuildStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stream the simple edgelist -> mapped, deduped, autapse-free edges.

    Returns (pre_idx u32, post_idx u32, weight i32, norm f64, post_count
    i32, pre_count i32), all sorted by packed (pre, post) key.
    """
    conv = cfg.get("conventions", {})
    ecols = conv.get("edge_columns", {})
    pre_col = ecols.get("pre", "pre")
    post_col = ecols.get("post", "post")
    w_col = ecols.get("weight", "count")
    norm_col = conv.get("norm_column", "norm")

    names = ipc.open_file(sources.edgelist).schema.names
    for c in (pre_col, post_col, w_col):
        if c not in names:
            raise KeyError(f"{sources.edgelist.name} missing required column {c!r}")
    optional = [c for c in (norm_col, "post_count", "pre_count") if c in names]

    pres: list[np.ndarray] = []
    posts: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    norms: list[np.ndarray] = []
    post_counts: list[np.ndarray] = []
    pre_counts: list[np.ndarray] = []

    cols = [pre_col, post_col, w_col, *optional]
    for batch in _batches(sources.edgelist, cols):
        if (
            batch.column(pre_col).null_count
            or batch.column(post_col).null_count
            or batch.column(w_col).null_count
        ):
            raise ValueError(f"{sources.edgelist.name}: nulls in pre/post/{w_col}")
        pre_s = np.asarray(batch.column(pre_col).to_pylist())
        post_s = np.asarray(batch.column(post_col).to_pylist())
        w = batch.column(w_col).to_numpy().astype(np.int64)
        stats.edges_raw_rows += len(batch)
        for c in cutoffs:
            stats.raw_edges_at_cutoff[str(c)] = stats.raw_edges_at_cutoff.get(
                str(c), 0
            ) + int((w >= c).sum())

        pre_idx = _lookup_sorted_ids(ids_sorted, pre_s)
        post_idx = _lookup_sorted_ids(ids_sorted, post_s)
        ok = (pre_idx >= 0) & (post_idx >= 0)
        stats.edge_rows_dropped_unannotated += int((~ok).sum())
        stats.weight_sum_dropped += int(w[~ok].sum())

        # Autapses are excluded upstream (~2.1%). Any residual row is
        # dropped here AND counted - policy enforced, asymmetry recorded.
        autapse = ok & (pre_idx == post_idx)
        stats.autapses_found += int(autapse.sum())
        keep = ok & ~autapse

        norms_b = (
            batch.column(norm_col).to_numpy().astype(np.float64)
            if norm_col in optional
            else np.full(len(batch), np.nan)
        )
        postc_b = (
            batch.column("post_count").to_numpy().astype(np.int64)
            if "post_count" in optional
            else np.full(len(batch), -1, dtype=np.int64)
        )
        prec_b = (
            batch.column("pre_count").to_numpy().astype(np.int64)
            if "pre_count" in optional
            else np.full(len(batch), -1, dtype=np.int64)
        )
        pres.append(pre_idx[keep].astype(np.uint32))
        posts.append(post_idx[keep].astype(np.uint32))
        weights.append(w[keep].astype(np.int32))
        norms.append(norms_b[keep])
        post_counts.append(postc_b[keep].astype(np.int32))
        pre_counts.append(prec_b[keep].astype(np.int32))

    pre = np.concatenate(pres)
    post = np.concatenate(posts)
    weight = np.concatenate(weights)
    norm = np.concatenate(norms)
    post_count = np.concatenate(post_counts)
    pre_count = np.concatenate(pre_counts)
    stats.edges_both_meta = pre.size

    key = (pre.astype(np.uint64) << 32) | post.astype(np.uint64)
    order = np.argsort(key, kind="stable")
    key, pre, post = key[order], pre[order], post[order]
    weight, norm = weight[order], norm[order]
    post_count, pre_count = post_count[order], pre_count[order]

    # Dedup: repeated (pre, post) pairs merge with weights summed (norm
    # recomputed as count/post_count so the invariant holds post-merge).
    if key.size:
        run_start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        n_dup = int(key.size - run_start.size)
        if n_dup:
            merged_w = np.add.reduceat(weight.astype(np.int64), run_start)
            pc_first = post_count[run_start].astype(np.int64)
            norm_first = norm[run_start]
            merged_norm = np.where(
                pc_first > 0, merged_w / np.maximum(pc_first, 1), norm_first
            )
            key = key[run_start]
            weight = merged_w.astype(np.int32)
            norm = merged_norm
            post_count = post_count[run_start]
            pre_count = pre_count[run_start]
            pre = (key >> 32).astype(np.uint32)
            post = (key & 0xFFFFFFFF).astype(np.uint32)
            stats.duplicate_pairs_merged = n_dup

    return pre, post, weight, norm, post_count, pre_count


def graph_manifest_name(version: str) -> str:
    """graph manifest filename, e.g. banc_v1 -> banc_graph_v1.json."""
    return f"{version.replace('banc_', 'banc_graph_', 1)}.json"


def build_banc_graph(
    cfg: dict[str, Any],
    sources: BancSourcePaths,
    graph_dir: Path,
    prov_dir: Path,
    results_dir: Path,
    input_manifest: dict | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Run the BANC build. Returns the verification report dict."""
    from flygpt.graph import build_csr
    from flygpt.schema import git_sha, utcnow, write_result

    t0 = time.time()
    stats = BancBuildStats()
    version = cfg["version"]
    conv = cfg.get("conventions", {})
    cutoff = int(conv.get("read_time_cutoff", 5))
    cutoffs = sorted({cutoff, *[int(c) for c in cfg.get("sensitivity_cutoffs", [])]})

    hashes = {}
    if verify_hashes and input_manifest is not None:
        hashes = verify_banc_source_hashes(input_manifest, sources)

    print("[1/3] loading node universe (banc_888_meta) ...", flush=True)
    nodes, ids_sorted, node_sign = load_banc_nodes(sources, cfg, stats)
    stats.n_nodes = nodes.height
    stats.n_nodes_typed = int(nodes["type"].is_not_null().sum())

    print("[2/3] loading + filtering edgelist ...", flush=True)
    pre, post, weight, norm, post_count, pre_count = load_banc_edges(
        sources, ids_sorted, cfg, cutoffs, stats
    )
    n_edges = pre.size
    graph_dir.mkdir(parents=True, exist_ok=True)

    # sign_source = per_neuron: edge sign is the pre neuron's predicted-NT
    # class. Recorded verbatim - the male build is per-edge (T-SEX-3).
    edge_sign = node_sign[pre]
    edge_low_conf = (edge_sign == SIGN_UNDETERMINED)
    stats.nt_edges_inhibitory = int((edge_sign == SIGN_INHIBITORY).sum())
    stats.nt_edges_excitatory = int((edge_sign == SIGN_EXCITATORY).sum())
    stats.nt_edges_low_confidence = int(edge_low_conf.sum())

    print("[3/3] writing artifacts ...", flush=True)
    edges_df = pl.DataFrame(
        {
            "pre_idx": pre,
            "post_idx": post,
            "weight": weight,
            "norm": norm,
            "post_count": post_count,
            "pre_count": pre_count,
            "nt_sign": edge_sign.astype(np.int8),
            "nt_low_confidence": edge_low_conf,
        }
    )
    edges_df.write_parquet(graph_dir / "edges.parquet", compression="zstd")

    keep5 = weight >= cutoff
    out_deg = np.bincount(pre[keep5], minlength=stats.n_nodes).astype(np.int32)
    in_deg = np.bincount(post[keep5], minlength=stats.n_nodes).astype(np.int32)
    nodes = nodes.with_columns(
        [
            pl.Series("nt_sign_neuron", node_sign),
            pl.Series("nt_low_confidence_neuron", node_sign == SIGN_UNDETERMINED),
            pl.Series("out_degree", out_deg),
            pl.Series("in_degree", in_deg),
        ]
    )
    nodes.write_parquet(graph_dir / "nodes.parquet", compression="zstd")

    for c in cutoffs:
        csr = build_csr(pre, post, weight, edge_sign.astype(np.int8), stats.n_nodes, c)
        stats.edges_at_cutoff[str(c)] = int(csr["indices"].size)
        np.savez_compressed(graph_dir / f"csr_cutoff{c}.npz", **csr)

    stats.wall_seconds = round(time.time() - t0, 2)
    manifest = {
        "graph_version": version,
        "dataset": cfg["dataset"],
        "built_utc": utcnow(),
        "code": {"git_sha": git_sha(), "generator": "flygpt banc.py"},
        "config_sha256": cfg.get("_config_sha256"),
        "conventions": {
            "snapshot": cfg.get("snapshot", SNAPSHOT),
            "synapse_snapshot": cfg.get("synapse_snapshot", SYNAPSE_SNAPSHOT),
            "synapse_snapshot_rationale": (
                "v2 = size>=5 detector cutoff, chosen to match the male >=5 "
                "read convention; v3 = size>=10 kept as a documented "
                "sensitivity slice"
            ),
            "autapse_policy": "excluded_at_build",
            "autapses_found": stats.autapses_found,
            "autapse_note": (
                "~2.1% of autapses excluded upstream in the BANC simple "
                "edgelist; residual rows dropped + counted. Asymmetry vs "
                "male 'keep' documented, never aligned (R9, T-BUILD-3)"
            ),
            "read_time_cutoff": cutoff,
            "cutoffs_built": cutoffs,
            "sign_source": "per_neuron",
            "sign_source_note": (
                "BANC simple edgelist carries no per-T-bar NT table; edge "
                "sign = pre neuron's neurotransmitter_predicted classed "
                "against the inhibitory set. The male build uses the "
                "per-edge T-bar majority rule - divergence recorded "
                "verbatim (T-SEX-3 honesty), not harmonized"
            ),
            "inhibitory_classes": list(INHIBITORY_NT),
            "norm_semantics": "norm = count/post_count (fraction of the post neuron's inputs)",
            "node_universe": "every banc_888_meta row; isolated rows retained as zero-degree nodes",
            "edge_universe": "edgelist rows with both endpoints in banc_888_meta",
            "meta_columns_absent": stats.meta_columns_absent,
        },
        "input_sha256": hashes or None,
        "counts": {
            "nodes": stats.n_nodes,
            "nodes_typed": stats.n_nodes_typed,
            "nonnumeric_ids": stats.n_nonnumeric_ids,
            "edges_raw_rows": stats.edges_raw_rows,
            "edges_both_meta": stats.edges_both_meta,
            "edge_rows_dropped_unannotated": stats.edge_rows_dropped_unannotated,
            "weight_sum_dropped": stats.weight_sum_dropped,
            "duplicate_pairs_merged": stats.duplicate_pairs_merged,
            "autapses_found": stats.autapses_found,
            "edges_at_cutoff": stats.edges_at_cutoff,
            "raw_edges_at_cutoff": stats.raw_edges_at_cutoff,
        },
        "nt": {
            "sign_source": "per_neuron",
            "nodes_inhibitory": stats.nodes_nt_inhibitory,
            "nodes_excitatory": stats.nodes_nt_excitatory,
            "nodes_undetermined": stats.nodes_nt_undetermined,
            "edges_inhibitory": stats.nt_edges_inhibitory,
            "edges_excitatory": stats.nt_edges_excitatory,
            "edges_low_confidence": stats.nt_edges_low_confidence,
            "nt_column_present": stats.nt_column_present,
        },
    }
    prov_dir.mkdir(parents=True, exist_ok=True)
    (prov_dir / graph_manifest_name(version)).write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )

    test_id = f"banc_build_{version.removeprefix('banc_')}"
    report = write_result(
        results_dir,
        test_id,
        checks={
            "node_count_equals_meta_rows": stats.n_nodes,
            "edges_at_cutoff": stats.edges_at_cutoff,
            "raw_edges_at_cutoff": stats.raw_edges_at_cutoff,
            "edges_both_meta": stats.edges_both_meta,
            "edge_rows_dropped_unannotated": stats.edge_rows_dropped_unannotated,
            "duplicate_pairs_merged": stats.duplicate_pairs_merged,
            "autapses_found": stats.autapses_found,
            "nt_sign": {
                "sign_source": "per_neuron",
                "inhibitory_edges": stats.nt_edges_inhibitory,
                "excitatory_edges": stats.nt_edges_excitatory,
                "low_confidence_edges": stats.nt_edges_low_confidence,
            },
            "typed_fraction": round(stats.n_nodes_typed / max(stats.n_nodes, 1), 6),
        },
        metrics={"wall_seconds": stats.wall_seconds},
        passed=None,
        extra={"manifest": f"provenance/{graph_manifest_name(version)}"},
    )
    print(
        f"[done] {stats.n_nodes} nodes, {n_edges} edges, cutoff {cutoff}: "
        f"{stats.edges_at_cutoff.get(str(cutoff))} edges, {stats.wall_seconds}s",
        flush=True,
    )
    return report


# ---------------------------------------------------------------------------
# Alignment: curated male<->female type table (T-SEX-1/4 inputs)
# ---------------------------------------------------------------------------


def _norm_label(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return s or None


def _dimorphism_map(male_dimorphism: Any) -> dict[str, str]:
    """Normalize the male-annotation dimorphism labels to {type: label}.

    Accepts a dict {type: label} or a DataFrame with a type column
    (``type``/``cell_type``/``male_type``) and a label column
    (``dimorphism``/``dimorphism_label``/``label``). The shared type-name
    namespace is intentional: a female-specific label on a BANC-only type
    name classifies it ``female-specific`` instead of merely unmatched.
    """
    if male_dimorphism is None:
        return {}
    if isinstance(male_dimorphism, dict):
        return {
            str(k): lbl
            for k, v in male_dimorphism.items()
            if (lbl := _norm_label(v)) is not None
        }
    if isinstance(male_dimorphism, pl.DataFrame):
        df = male_dimorphism
        type_col = next(
            (c for c in ("type", "cell_type", "male_type") if c in df.columns), None
        )
        lbl_col = next(
            (c for c in ("dimorphism", "dimorphism_label", "label") if c in df.columns),
            None,
        )
        if type_col is None or lbl_col is None:
            raise ValueError(
                "male_dimorphism DataFrame needs a type column "
                "(type|cell_type|male_type) and a label column "
                "(dimorphism|dimorphism_label|label)"
            )
        return {
            str(t): lbl
            for t, v in zip(df[type_col].to_list(), df[lbl_col].to_list(), strict=True)
            if t is not None and (lbl := _norm_label(v)) is not None
        }
    raise TypeError(f"unsupported male_dimorphism argument: {type(male_dimorphism)}")


def _match_overrides(
    matches_df: pl.DataFrame | None,
) -> tuple[dict[str, str | None], set[str]]:
    """Parse the reviewed-matches table into id/type-keyed overrides.

    Returns (overrides, referenced_male_types). ``overrides`` maps a
    banc_888_id (or ``type:<banc_cell_type>`` when the table has no id
    column) to the curated male type, or to None for an explicitly
    rejected match - the curated file wins over the meta columns.
    """
    overrides: dict[str, str | None] = {}
    referenced: set[str] = set()
    if matches_df is None or matches_df.height == 0:
        return overrides, referenced
    cols = matches_df.columns
    id_col = next(
        (
            c
            for c in (
                "banc_888_id", "banc_id", "root_888", "banc_root_id",
                "query_id", "pt_root_id", "id",
            )
            if c in cols
        ),
        None,
    )
    type_col = next(
        (c for c in ("banc_cell_type", "banc_type", "cell_type") if c in cols), None
    )
    male_col = next(
        (
            c
            for c in ("malecns_cell_type", "male_type", "malecns_type", "match_cell_type")
            if c in cols
        ),
        None,
    )
    dec_col = next(
        (
            c
            for c in (
                "decision", "status", "verdict", "reviewed", "accepted",
                "valid", "validation", "match",
            )
            if c in cols
        ),
        None,
    )
    if id_col is None and type_col is None:
        raise ValueError(f"reviewed matches has no banc id/type column: {cols}")
    if male_col is None:
        raise ValueError(f"reviewed matches has no male-type column: {cols}")

    for row in matches_df.iter_rows(named=True):
        key_val = row.get(id_col) if id_col else row.get(type_col)
        if key_val is None:
            continue
        key = str(key_val) if id_col else f"type:{key_val}"
        male_val = row.get(male_col)
        male = str(male_val) if male_val not in (None, "") else None
        accepted = True
        if dec_col is not None:
            dec = _norm_label(row.get(dec_col)) or ""
            if dec in _DECISION_REJECT:
                accepted = False
            elif dec in _DECISION_ACCEPT:
                accepted = True
            # unknown decision strings: treated as accept when a male type
            # is present (lenient default - curated files are audited once)
        overrides[key] = male if accepted else None
        if male:
            referenced.add(male)
    return overrides, referenced


def alignment_table(
    meta_df: pl.DataFrame,
    matches_df: pl.DataFrame | None = None,
    male_dimorphism: Any = None,
) -> pl.DataFrame:
    """Curated male<->female type alignment.

    One row per typed banc_888 neuron plus one row per male type with no
    aligned partner. Columns: [male_type, banc_type, banc_id, iso_class].

    Primary key: meta's ``malecns_cell_type`` column (config
    alignment.primary), with ``manc_cell_type`` as the cord fallback
    (alignment.secondary). The reviewed-matches table overrides both:
    accepted rows set the male type, rejected rows unmatch the neuron.
    Dimorphism labels (male annotations ``dimorphism`` column, shared
    type namespace) upgrade isomorphic->dimorphic, unmatched->*-specific;
    unmatched-without-label lands in the honest ``*-only-unmatched``
    class - never guessed (plan: coverage reported, not assumed).
    """
    dim = _dimorphism_map(male_dimorphism)
    overrides, referenced = _match_overrides(matches_df)
    cols = meta_df.columns
    if "banc_888_id" not in cols:
        raise KeyError("meta_df missing banc_888_id column")

    rows: list[dict[str, Any]] = []
    matched_male: set[str] = set()
    for r in meta_df.iter_rows(named=True):
        bid = r["banc_888_id"]
        btype = r.get("cell_type")
        male = r.get("malecns_cell_type") or r.get("manc_cell_type")
        if str(bid) in overrides:
            male = overrides[str(bid)]
        elif btype is not None and f"type:{btype}" in overrides:
            male = overrides[f"type:{btype}"]
        if male:
            male = str(male)
            matched_male.add(male)
            iso = ISO_DIMORPHIC if dim.get(male) == ISO_DIMORPHIC else ISO_ISOMORPHIC
            rows.append(
                {
                    "male_type": male,
                    "banc_type": btype,
                    "banc_id": str(bid),
                    "iso_class": iso,
                }
            )
        elif btype:
            iso = (
                ISO_FEMALE_SPECIFIC
                if dim.get(str(btype)) == ISO_FEMALE_SPECIFIC
                else ISO_FEMALE_UNMATCHED
            )
            rows.append(
                {
                    "male_type": None,
                    "banc_type": str(btype),
                    "banc_id": str(bid),
                    "iso_class": iso,
                }
            )
        # untyped + unmatched rows are excluded from the table (they carry
        # no alignable type); the caller reports their count from meta.

    # Male-side universe: dimorphism keys + male types referenced by the
    # reviewed matches + everything matched. Dimorphism keys that already
    # appear as a BANC type in the table (shared namespace ambiguity) are
    # female-side types, not male types - they must not double as
    # male-side rows.
    banc_types_present = {r["banc_type"] for r in rows if r["banc_type"] is not None}
    male_universe = (set(dim) - banc_types_present) | referenced | matched_male
    for mt in sorted(male_universe - matched_male):
        iso = (
            ISO_MALE_SPECIFIC
            if dim.get(mt) == ISO_MALE_SPECIFIC
            else ISO_MALE_UNMATCHED
        )
        rows.append({"male_type": mt, "banc_type": None, "banc_id": None, "iso_class": iso})

    schema = {
        "male_type": pl.Utf8,
        "banc_type": pl.Utf8,
        "banc_id": pl.Utf8,
        "iso_class": pl.Utf8,
    }
    df = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    return df.sort(
        [pl.col("banc_id").is_null(), pl.col("banc_id"), pl.col("male_type").is_null(), pl.col("male_type")]
    )


# ---------------------------------------------------------------------------
# Sex-conditioned union vocab + female graph in the shared type space
# ---------------------------------------------------------------------------


def sex_conditioned_vocab(
    male_bundle: Any, banc_bundle: Any, align_df: pl.DataFrame
) -> dict:
    """Union type-vocab spec for sex-conditioned forward passes.

    Male manifest types keep the male token block (TYPE_TOKEN_BASE + i,
    sorted). Every aligned BANC type shares its male partner's token.
    Female-only types are appended as a contiguous block after the male
    tokens (collision-safe: a BANC name already taken by a male type gets
    a ``__banc`` suffix so the sex-specific sets stay disjoint, T-SEX-4).

    Returns a plain dict (the impl-spec Vocab class is a Phase-3
    construct; this is the deterministic spec it will consume):
    {type_to_id, banc_type_to_id, n_types, sex_specific_sets, ...}.
    """
    from flygpt.schema import TYPE_TOKEN_BASE

    male_types = list(male_bundle.manifest_types)
    type_to_id = {t: TYPE_TOKEN_BASE + i for i, t in enumerate(male_types)}

    banc_to_male: dict[str, str] = {}
    female_only: dict[str, str] = {}
    male_unmatched: dict[str, str] = {}
    for row in align_df.iter_rows(named=True):
        mt, bt, iso = row["male_type"], row["banc_type"], row["iso_class"]
        if bt is not None and mt is not None:
            banc_to_male.setdefault(bt, mt)
        elif bt is not None and iso in FEMALE_ONLY_ISO:
            female_only.setdefault(bt, iso)
        elif mt is not None and bt is None and iso in MALE_ONLY_ISO:
            male_unmatched.setdefault(mt, iso)

    # Male manifest types with no aligned partner are absent under female
    # conditioning even if the alignment artifact never mentioned them.
    aligned_male = set(banc_to_male.values())
    for mt in male_types:
        if mt not in aligned_male and mt not in male_unmatched:
            male_unmatched[mt] = ISO_MALE_UNMATCHED

    banc_type_to_id: dict[str, int] = {}
    alignment_gaps: list[str] = []
    for bt in sorted(banc_to_male):
        mt = banc_to_male[bt]
        if mt in type_to_id:
            banc_type_to_id[bt] = type_to_id[mt]
        else:
            # aligned to a male type absent from this male bundle - the
            # neuron is unrepresentable in the union vocab; reported.
            alignment_gaps.append(f"{bt}->{mt}")

    next_id = TYPE_TOKEN_BASE + len(male_types)
    female_key: dict[str, str] = {}
    for bt in sorted(female_only):
        key = bt if bt not in type_to_id else f"{bt}__banc"
        female_key[bt] = key
        type_to_id[key] = next_id
        next_id += 1
        banc_type_to_id[bt] = type_to_id[key]

    banc_manifest_types = set(banc_bundle.manifest_types)
    uncovered = sorted(banc_manifest_types - set(banc_type_to_id))

    return {
        "version": "sex_union_v1",
        "type_token_base": TYPE_TOKEN_BASE,
        "type_to_id": type_to_id,
        "banc_type_to_id": banc_type_to_id,
        "n_types": len(type_to_id),
        "male_types": male_types,
        "female_only_types": [female_key[bt] for bt in sorted(female_only)],
        "banc_type_map": {
            "aligned": {bt: banc_to_male[bt] for bt in sorted(banc_to_male)},
            "female_only": {bt: female_only[bt] for bt in sorted(female_only)},
        },
        "sex_specific_sets": {
            "male_only": sorted(male_unmatched),
            "female_only": [female_key[bt] for bt in sorted(female_only)],
            "male_specific_annotated": sorted(
                t for t, iso in male_unmatched.items() if iso == ISO_MALE_SPECIFIC
            ),
            "female_specific_annotated": [
                female_key[bt]
                for bt in sorted(female_only)
                if female_only[bt] == ISO_FEMALE_SPECIFIC
            ],
        },
        "uncovered_banc_types": uncovered,
        "alignment_gaps": alignment_gaps,
    }


def female_graph_in_male_space(
    banc_bundle: Any, align_df: pl.DataFrame, vocab: dict, n_types: int
) -> Any:
    """Female edge set re-indexed onto the union type space (T-SEX-5).

    Each BANC neuron maps to its type token: aligned types share the male
    type's token index, female-only types occupy their appended tokens.
    Neuron-level edges are aggregated to TYPE-level edges: the weight of
    type-edge (A -> B) is the SUM of synapse counts over all ordered
    neuron pairs (u, v) with type(u)=A, type(v)=B - multiple BANC neurons
    of one aligned type therefore sum into the shared male type token.
    Edge sign = sign of the summed signed weight (pre-neuron signs are
    near-uniform within a type; ties land on 0). Intra-type neuron pairs
    become type-level self-loops, kept and counted.

    Untyped neurons (no cell_type) and types absent from the vocab are
    unrepresentable in type space: their edges are dropped and counted,
    never silently re-homed. Male-only slots keep zero degree - by
    construction they are absent under female conditioning (T-SEX-4).

    Returns a GraphBundle over ``n_types`` union-vocab nodes (node_idx j
    = token TYPE_TOKEN_BASE + j); build statistics are attached as
    ``bundle.type_space_meta`` for provenance.
    """
    from flygpt.schema import TYPE_TOKEN_BASE, bundle_from_edges

    type_to_id = vocab["type_to_id"]
    banc_type_to_id = vocab["banc_type_to_id"]
    if n_types < len(type_to_id):
        raise ValueError(f"n_types={n_types} smaller than union vocab {len(type_to_id)}")

    col = "cell_type" if "cell_type" in banc_bundle.nodes.columns else "type"
    cell_types = banc_bundle.nodes[col].to_list()
    tok = np.array(
        [banc_type_to_id.get(str(c), -1) if c is not None else -1 for c in cell_types],
        dtype=np.int64,
    )
    tok_idx = np.where(tok >= 0, tok - TYPE_TOKEN_BASE, -1)

    n = banc_bundle.n_nodes
    pre_of_edge = np.repeat(np.arange(n, dtype=np.int64), np.diff(banc_bundle.out_indptr))
    tp = tok_idx[pre_of_edge]
    tq = tok_idx[banc_bundle.out_indices]
    keep = (tp >= 0) & (tq >= 0) & (tp < n_types) & (tq < n_types)
    n_dropped = int((~keep).sum())

    kp, kq = tp[keep], tq[keep]
    kw = banc_bundle.out_weight[keep].astype(np.int64)
    ks = banc_bundle.out_sign[keep].astype(np.int64)
    key = (kp.astype(np.uint64) << 32) | kq.astype(np.uint64)
    order = np.argsort(key, kind="stable")
    key, kw, ks = key[order], kw[order], ks[order]
    if key.size:
        run_start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        agg_w = np.add.reduceat(kw, run_start)
        agg_signed = np.add.reduceat(kw * ks, run_start)
        key = key[run_start]
    else:
        agg_w = np.empty(0, dtype=np.int64)
        agg_signed = np.empty(0, dtype=np.int64)
    agg_sign = np.sign(agg_signed).astype(np.int8)
    agg_pre = (key >> 32).astype(np.uint32)
    agg_post = (key & 0xFFFFFFFF).astype(np.uint32)
    n_intra = int((agg_pre == agg_post).sum())
    n_merged_pairs = int(keep.sum() - agg_pre.size)

    inv = {tid: name for name, tid in type_to_id.items()}
    male_only = set(vocab["sex_specific_sets"]["male_only"])
    female_only = set(vocab["sex_specific_sets"]["female_only"])
    node_rows = []
    for j in range(n_types):
        token = TYPE_TOKEN_BASE + j
        name = inv.get(token)
        if name in female_only:
            scope = "female_only"
        elif name in male_only:
            scope = "male_only"
        else:
            scope = "shared"
        node_rows.append(
            {
                "node_idx": j,
                "token": token,
                "type": name,
                "bodyId": token,
                "present_female": scope != "male_only",
                "sex_scope": scope,
            }
        )
    nodes_df = pl.DataFrame(node_rows)
    out = bundle_from_edges(
        nodes_df, agg_pre, agg_post, agg_w.astype(np.int32), agg_sign,
        cutoff=banc_bundle.cutoff,
    )
    out.type_space_meta = {
        "aggregation_rule": (
            "type-edge (A,B) weight = sum of synapse counts over neuron "
            "pairs (u,v) with type(u)=A, type(v)=B; sign = sign of summed "
            "signed weight (net-zero -> 0); intra-type pairs -> self-loops"
        ),
        "neuron_edges_in": int(banc_bundle.out_indices.size),
        "neuron_edges_dropped_untyped_or_unmapped": n_dropped,
        "type_edges": int(agg_pre.size),
        "neuron_pairs_merged_into_type_edges": n_merged_pairs,
        "intra_type_self_loops": n_intra,
        "male_only_nodes": sorted(j for j in range(n_types) if inv.get(TYPE_TOKEN_BASE + j) in male_only),
    }
    return out


def female_stimulus_map(male_program_spec: dict, align_df: pl.DataFrame) -> dict:
    """Re-resolve male program targets onto female homolog types.

    Every drive carrying a ``types`` (or ``target_types``) list is
    rewritten to the aligned BANC types (isomorphic + dimorphic rows of
    the alignment table). Male types with no female homolog are dropped
    from the drive and recorded - female homolog coverage is REPORTED,
    never assumed (plan Phase 4 open item; T-SEX wording stays honest).
    Drives without a type query pass through flagged as unresolved.
    """
    homologs: dict[str, set[str]] = {}
    for row in align_df.iter_rows(named=True):
        if row["iso_class"] in MATCHED_ISO and row["male_type"] and row["banc_type"]:
            homologs.setdefault(row["male_type"], set()).add(row["banc_type"])

    spec = copy.deepcopy(male_program_spec)
    drives = spec.get("drives", [])
    new_drives: list[dict] = []
    requested: set[str] = set()
    resolved: set[str] = set()
    missing: set[str] = set()
    unresolvable: list[int] = []
    for i, drive in enumerate(drives):
        targets = drive.get("types") or drive.get("target_types")
        d = dict(drive)
        if targets is None:
            unresolvable.append(i)
            d["female_resolution"] = "passthrough_unresolved"
            new_drives.append(d)
            continue
        res: set[str] = set()
        miss: list[str] = []
        for t in targets:
            h = homologs.get(t)
            if h:
                res |= h
                resolved.add(t)
            else:
                miss.append(t)
                missing.add(t)
        requested |= set(targets)
        d["male_types"] = list(targets)
        d["types"] = sorted(res)
        d["unmatched_male_types"] = sorted(miss)
        new_drives.append(d)

    spec["drives"] = new_drives
    n_req = len(requested)
    n_res = len(requested & resolved)
    spec["female_homolog_coverage"] = {
        "requested_male_types": sorted(requested),
        "resolved_male_types": sorted(requested & resolved),
        "missing_male_types": sorted(missing),
        "coverage": round(n_res / n_req, 6) if n_req else None,
        "drives_total": len(drives),
        "drives_fully_resolved": sum(
            1 for d in new_drives if "male_types" in d and not d["unmatched_male_types"]
        ),
        "drives_unresolvable": len(unresolvable),
        "unresolvable_drive_indices": unresolvable,
        "note": (
            "female homolog coverage measured and reported, never assumed "
            "(plan Phase 4 open item)"
        ),
    }
    return spec


# ---------------------------------------------------------------------------
# --align driver: writes evals/alignment_v1.json + updates the graph manifest
# ---------------------------------------------------------------------------


def run_align(
    cfg: dict[str, Any],
    sources: BancSourcePaths,
    prov_dir: Path,
    evals_dir: Path,
    male_dimorphism: Any = None,
) -> dict[str, Any]:
    """Build the alignment table and write its artifacts.

    ``evals/alignment_v1.json`` (results envelope + iso_class counts) and
    ``evals/alignment_v1.parquet`` (full table). The alignment block is
    also recorded in ``provenance/banc_graph_<v>.json`` - the frozen
    dimorphism set fills from this record (evals/dimorphism_set.json).
    """
    from flygpt.schema import sha256_file, write_result

    meta = pl.from_arrow(feather.read_table(sources.meta))
    matches_df = None
    if sources.malecns_matches is not None and Path(sources.malecns_matches).exists():
        matches_df = pl.read_csv(sources.malecns_matches)
    table = alignment_table(meta, matches_df, male_dimorphism)

    evals_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = evals_dir / "alignment_v1.parquet"
    table.write_parquet(parquet_path, compression="zstd")

    iso_counts = {
        k: int(v)
        for k, v in table.group_by("iso_class")
        .len()
        .sort("iso_class")
        .iter_rows()
    }
    untyped = 0
    if "cell_type" in meta.columns:
        mask = pl.col("cell_type").is_null()
        if "malecns_cell_type" in meta.columns:
            mask = mask & pl.col("malecns_cell_type").is_null()
        untyped = int(meta.filter(mask).height)
    align_stats = {
        "table_parquet": str(parquet_path),
        "table_sha256": sha256_file(parquet_path),
        "n_rows": table.height,
        "iso_class_counts": iso_counts,
        "untyped_unmatched_meta_rows_excluded": untyped,
        "inputs": {
            "meta_sha256": sha256_file(sources.meta),
            "matches_sha256": (
                sha256_file(sources.malecns_matches) if matches_df is not None else None
            ),
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report = write_result(
        evals_dir,
        "alignment_v1",
        checks={
            "n_rows": table.height,
            "iso_class_counts": iso_counts,
            "untyped_unmatched_meta_rows_excluded": untyped,
        },
        inputs={
            "meta": str(sources.meta),
            "matches": str(sources.malecns_matches) if matches_df is not None else None,
        },
        passed=None,
        extra={"table_parquet": str(parquet_path)},
    )

    version = cfg["version"]
    mpath = prov_dir / graph_manifest_name(version)
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
    else:
        manifest = {
            "graph_version": version,
            "dataset": cfg.get("dataset", DATASET_ID),
            "note": "graph build artifacts not yet produced; alignment-only record",
        }
    manifest["alignment"] = align_stats
    prov_dir.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", type=Path, default=Path("configs/banc_v1.yaml"))
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--align", action="store_true")
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated registry keys; default = required")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/banc"))
    ap.add_argument("--manifest", type=Path, default=Path("provenance/banc_v888.json"))
    ap.add_argument("--graph-dir", type=Path, default=None,
                    help="default: config output.graph_dir or data/graph_banc")
    ap.add_argument("--provenance-dir", type=Path, default=Path("provenance"))
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--evals-dir", type=Path, default=Path("evals"))
    ap.add_argument("--male-nodes", type=Path, default=Path("data/graph/nodes.parquet"),
                    help="male nodes.parquet; supplies the dimorphism column for --align")
    ap.add_argument("--skip-verify", action="store_true",
                    help="dev only: skip input sha256 re-verification")
    args = ap.parse_args(argv)

    if not (args.download or args.build or args.align):
        ap.error("choose at least one of --download/--build/--align")

    cfg = load_config(args.config)
    version = cfg["version"]
    prov_dir = args.provenance_dir
    out_cfg = cfg.get("output", {})
    graph_dir = args.graph_dir or Path(out_cfg.get("graph_dir", "data/graph_banc"))

    if args.download:
        keys = args.only.split(",") if args.only else None
        download_banc(keys, args.raw_dir, args.manifest)

    if args.build or args.align:
        input_manifest = None
        if args.manifest.exists():
            input_manifest = json.loads(args.manifest.read_text())
            sources = resolve_banc_sources(input_manifest, args.raw_dir)
        elif args.skip_verify:
            sources = BancSourcePaths(
                meta=args.raw_dir / "banc_888_meta.feather",
                edgelist=args.raw_dir / "banc_888_edgelist_simple_v2.feather",
                edgelist_v3=(
                    p if (p := args.raw_dir / "banc_888_edgelist_simple_v3.feather").exists()
                    else None
                ),
                nt_prediction=(
                    p
                    if (
                        p := args.raw_dir / "banc_888_neurotransmitter_prediction_v2.csv"
                    ).exists()
                    else None
                ),
                malecns_matches=(
                    p if (p := args.raw_dir / "banc_malecns_reviewed_matches.csv").exists()
                    else None
                ),
                malecns_nblast=(
                    p if (p := args.raw_dir / "banc_malecns_v0.9_nblast.feather").exists()
                    else None
                ),
            )
        else:
            raise FileNotFoundError(
                f"{args.manifest} missing - run `python -m flygpt.banc --download` "
                "first (or pass --skip-verify for dev builds)"
            )

        graph_manifest = prov_dir / graph_manifest_name(version)
        if args.build or (args.align and not graph_manifest.exists()):
            build_banc_graph(
                cfg,
                sources,
                graph_dir,
                prov_dir,
                args.results_dir,
                input_manifest=input_manifest,
                verify_hashes=not args.skip_verify,
            )
        if args.align:
            male_dimorphism = None
            if args.male_nodes.exists():
                male_nodes = pl.read_parquet(args.male_nodes)
                if "dimorphism" in male_nodes.columns:
                    male_dimorphism = {
                        str(t): d
                        for t, d in male_nodes.select("type", "dimorphism").iter_rows()
                        if t is not None and d is not None
                    }
            run_align(cfg, sources, prov_dir, args.evals_dir, male_dimorphism)
    return 0


if __name__ == "__main__":
    sys.exit(main())
