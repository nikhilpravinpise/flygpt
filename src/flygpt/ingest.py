"""Canonical MaleCNS v1.0 ingest: GCS download + sha256 provenance manifest.

Downloads the canonical Janelia flat-connectome files from
``gs://flyem-male-cns/v1.0/`` (public-read bucket, fetched over HTTPS) and
records sha256 + size + the bucket-reported md5 of every file in
``provenance/malecns_v1.json``.

Per plan.md: canonical Janelia files only - NOT the HF mirror, NOT the
Lee-lab ``malecns_09`` recompilation. ``syn-points`` (geometry) and the
``-significant-only`` / ``-traced-only`` convention variants are registered
in the manifest but not downloaded by default; ``body-stats`` is optional.

Usage:
    python -m flygpt.ingest                     # download required files
    python -m flygpt.ingest --verify-only       # re-hash local files
    python -m flygpt.ingest --list              # print registry
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests

GCS_BUCKET = "flyem-male-cns"
GCS_PREFIX = "v1.0/connectome-data/flat-connectome/"
DATASET_ID = "male-cns:v1.0"
DATASET_LICENSE = "CC-BY-4.0"
DATASET_CITATION = "Berg et al. 2026, Cell - MaleCNS connectome (Janelia FlyEM, Cambridge, Google)"

CHUNK = 8 * 1024 * 1024  # 8 MiB streaming chunks
RETRIES = 5


@dataclass(frozen=True)
class SourceFile:
    """One canonical upstream object in the flat-connectome directory."""

    key: str
    name: str  # object name relative to GCS_PREFIX
    role: str  # what Phase 0 uses it for
    required: bool = True


# Registry covers every object in the flat-connectome directory. Files not
# needed for the Phase-0 graph build are still recorded in the manifest
# (size + md5) so provenance covers the whole release.
REGISTRY: tuple[SourceFile, ...] = (
    SourceFile(
        "body_annotations",
        "body-annotations-male-cns-v1.0-minconf-0.5.feather",
        "node universe: types / classes / sides / instance metadata",
    ),
    SourceFile(
        "body_neurotransmitters",
        "body-neurotransmitters-male-cns-v1.0.feather",
        "per-neuron NT predictions (sign rule input)",
    ),
    SourceFile(
        "connectome_weights",
        "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        "canonical aggregated (pre, post, weight) edges",
    ),
    SourceFile(
        "tbar_neurotransmitters",
        "tbar-neurotransmitters-male-cns-v1.0.feather",
        "per-T-bar NT probabilities (per-edge sign + confidence)",
    ),
    SourceFile(
        "syn_partners",
        "syn-partners-male-cns-v1.0-minconf-0.5.feather",
        "per-synapse partnerships (derived-weight reconciliation)",
    ),
    SourceFile(
        "syn_points",
        "syn-points-male-cns-v1.0-minconf-0.5.feather",
        "synapse-level geometry - not needed for Phase 0",
        required=False,
    ),
    SourceFile(
        "body_stats",
        "body-stats-male-cns-v1.0-minconf-0.5.feather",
        "per-body synapse statistics - optional",
        required=False,
    ),
    SourceFile(
        "weights_significant",
        "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
        "convention variant: significant edges only",
        required=False,
    ),
    SourceFile(
        "weights_traced",
        "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
        "convention variant: traced neurons only",
        required=False,
    ),
    SourceFile(
        "partners_significant",
        "syn-partners-male-cns-v1.0-minconf-0.5-significant-only.feather",
        "convention variant: significant-only synapses",
        required=False,
    ),
    SourceFile(
        "partners_traced",
        "syn-partners-male-cns-v1.0-minconf-0.5-traced-only.feather",
        "convention variant: traced-only synapses",
        required=False,
    ),
)

REQUIRED_KEYS = tuple(f.key for f in REGISTRY if f.required)


def gcs_url(name: str) -> str:
    return f"https://storage.googleapis.com/{GCS_BUCKET}/{GCS_PREFIX}{name}"


def list_bucket() -> dict[str, dict]:
    """Return {object_name: {size, md5_b64}} for the flat-connectome dir."""
    api = (
        f"https://storage.googleapis.com/storage/v1/b/{GCS_BUCKET}/o"
        f"?prefix={GCS_PREFIX}&fields=items(name,size,md5Hash)"
    )
    out: dict[str, dict] = {}
    token = ""
    while True:
        url = api + (f"&pageToken={token}" if token else "")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("items", []):
            short = item["name"][len(GCS_PREFIX):]
            out[short] = {"size": int(item["size"]), "md5_b64": item["md5Hash"]}
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


def download_one(src: SourceFile, dest_dir: Path, meta: dict) -> dict:
    """Download one file with resume + integrity check; return manifest entry."""
    dest = dest_dir / src.name
    part = dest.with_suffix(dest.suffix + ".partial")
    expected_size = meta["size"]
    expected_md5 = meta["md5_b64"]

    if dest.exists():
        sha, md5, size = _hash_file(dest)
        if size == expected_size and md5 == expected_md5:
            print(f"[skip] {src.name} already complete", flush=True)
            return _entry(src, meta, sha, md5, size)
        print(f"[redo] {src.name} failed integrity check, redownloading", flush=True)
        dest.unlink()

    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > expected_size:
            part.unlink()
            have = 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(gcs_url(src.name), headers=headers, stream=True, timeout=120) as r:
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
                # bytes already on disk still count toward the final hashes
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
                print(f"[dl] {src.name}: {size / 1e9:.2f} GB at {rate:.0f} MB/s", flush=True)
        except requests.RequestException as exc:
            print(f"[retry {attempt}/{RETRIES}] {src.name}: {exc}", flush=True)
            time.sleep(min(2**attempt, 30))
            continue

        md5_b64 = base64.b64encode(md5.digest()).decode()
        if size != expected_size or md5_b64 != expected_md5:
            print(
                f"[retry {attempt}/{RETRIES}] {src.name}: integrity mismatch "
                f"(size {size} vs {expected_size})",
                flush=True,
            )
            continue
        part.rename(dest)
        return _entry(src, meta, sha.hexdigest(), md5_b64, size)

    raise RuntimeError(f"download failed after {RETRIES} attempts: {src.name}")


def _entry(src: SourceFile, meta: dict, sha: str, md5_b64: str, size: int) -> dict:
    return {
        "key": src.key,
        "gcs_object": f"gs://{GCS_BUCKET}/{GCS_PREFIX}{src.name}",
        "https_url": gcs_url(src.name),
        "local_name": src.name,
        "role": src.role,
        "bytes": size,
        "md5_gcs": md5_b64,
        "sha256": sha,
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def verify_local(src: SourceFile, dest_dir: Path, meta: dict) -> dict:
    """Re-hash an already-downloaded file and check it against the bucket."""
    dest = dest_dir / src.name
    if not dest.exists():
        raise FileNotFoundError(f"missing {dest}")
    sha, md5, size = _hash_file(dest)
    if size != meta["size"] or md5 != meta["md5_b64"]:
        raise RuntimeError(f"integrity check failed for {src.name}")
    return _entry(src, meta, sha, md5, size)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "uncommitted"


def write_manifest(
    manifest_path: Path, entries: list[dict], bucket: dict, dest_dir: Path
) -> None:
    """Write provenance/malecns_v1.json covering the whole release dir.

    Files verified this run get fresh entries. For registry files not in
    this run's target set (``--only``), an existing on-disk file keeps its
    prior verified manifest entry when the byte size still matches; files
    with no prior entry and no local copy are marked ``not_downloaded``.
    """
    prior: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            prior = {
                f["local_name"]: f
                for f in json.loads(manifest_path.read_text()).get("files", [])
                if f.get("local_name")
            }
        except Exception:
            prior = {}

    downloaded = {e["local_name"]: e for e in entries}
    files = []
    for src in REGISTRY:
        if src.name in downloaded:
            files.append(downloaded[src.name])
            continue
        meta = bucket.get(src.name, {})
        local = dest_dir / src.name
        old = prior.get(src.name)
        if (
            local.exists()
            and old
            and old.get("sha256")
            and local.stat().st_size == meta.get("size") == old.get("bytes")
        ):
            carried = dict(old)
            carried["carried_forward"] = True
            files.append(carried)
        else:
            files.append(
                {
                    "key": src.key,
                    "gcs_object": f"gs://{GCS_BUCKET}/{GCS_PREFIX}{src.name}",
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
        "title": "MaleCNS v1.0 flat connectome",
        "source_bucket": f"gs://{GCS_BUCKET}/{GCS_PREFIX}",
        "license": DATASET_LICENSE,
        "citation": DATASET_CITATION,
        "conventions": {
            "minconf": 0.5,
            "weight_semantics": "weight = synapse count per (pre, post) pair",
            "canonical_files_note": (
                "Janelia canonical files only; NOT the HF mirror and NOT the "
                "Lee-lab malecns_09 recompilation."
            ),
        },
        "code": {
            "git_sha": _git_sha(),
            "generator": f"flygpt {ingest_version()} ingest.py",
        },
        "files": files,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[manifest] wrote {manifest_path}", flush=True)


def ingest_version() -> str:
    from flygpt import __version__

    return __version__


def sha256_of(path: Path) -> str:
    """T-BUILD-5 helper: sha256 of any local file."""
    return _hash_file(path)[0]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", type=Path, default=Path("data/raw"))
    ap.add_argument("--manifest", type=Path, default=Path("provenance/malecns_v1.json"))
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated registry keys; default = all required")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        for f in REGISTRY:
            tag = "required" if f.required else "optional"
            print(f"{f.key:22s} [{tag}] {f.name} - {f.role}")
        return 0

    bucket = list_bucket()
    missing = [f.name for f in REGISTRY if f.name not in bucket]
    if missing:
        raise RuntimeError(f"registry files absent from bucket listing: {missing}")

    if args.only:
        keys = set(args.only.split(","))
        targets = [f for f in REGISTRY if f.key in keys]
        if len(targets) != len(keys):
            raise KeyError(f"unknown keys: {keys - {f.key for f in targets}}")
    else:
        targets = [f for f in REGISTRY if f.required]

    # smallest first so schema inspection can start while big files stream
    targets.sort(key=lambda f: bucket[f.name]["size"])
    args.dest.mkdir(parents=True, exist_ok=True)

    entries = []
    for src in targets:
        if args.verify_only:
            entries.append(verify_local(src, args.dest, bucket[src.name]))
        else:
            entries.append(download_one(src, args.dest, bucket[src.name]))

    write_manifest(args.manifest, entries, bucket, args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
