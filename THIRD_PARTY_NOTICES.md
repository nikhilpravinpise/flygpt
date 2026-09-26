# THIRD_PARTY_NOTICES.md

FlyGPT is Apache-2.0. This file is the dependency + data license matrix
(T-SHIP-3/9). Runtime deps carry permissive licenses only; GPL-3.0 tools
are quarantined in `scripts/` and are never imported by `src/flygpt/`;
non-commercial data is excluded from every shipped artifact.

## Runtime dependencies (pyproject.toml `dependencies`)

| Package | Version | License |
|---|---|---|
| numpy | 2.5.3 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pyarrow | 25.0.1 | Apache-2.0 |
| polars | 1.44.2 | MIT |
| scipy | 1.18.1 | BSD-3-Clause (bundled OpenBLAS BSD-3-Clause; LAPACK BSD-3-Clause-Open-MPI) |
| requests | 2.34.2 | Apache-2.0 |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| PyYAML | 6.0.3 | MIT |

## Optional extras

| Extra | Package | License | Used for |
|---|---|---|---|
| `dev` | pytest 9.1.1 | MIT | test suite |
| `dev` | ruff 0.16.8 | MIT | lint |
| `remote` | modal | MIT | Modal entry points (`modal_app.py`) |
| `remote` | huggingface_hub | Apache-2.0 | HF artifact upload/download |
| `model` | jax 0.11.2 | Apache-2.0 | model + GPU/TPU sim backend |
| `model` | flax 0.12.10 | Apache-2.0 | model framework |
| `model` | optax 0.2.8 | Apache-2.0 | optimizer |
| `flybrain` | flybrain 0.1.0 | MIT | T-SIM-2 cross-check backend only (never corpus) |
| `flybrain` | numba 0.67.0 | BSD-2-Clause | flybrain CPU kernel dep |

## Datasets (derived artifacts carry CC-BY-4.0 with this attribution chain)

| Dataset | License | Attribution |
|---|---|---|
| MaleCNS v1.0 (`gs://flyem-male-cns/v1.0/`) | CC-BY-4.0 | Berg et al., Cell 2026. Janelia FlyEM, University of Cambridge, Google |
| BANC v888 (`gs://lee-lab_brain-and-nerve-cord-fly-connectome/`) | CC-BY-4.0 | Bates, Phelps, Kim et al., Nature 2026 |
| Per-synapse NT predictions | CC-BY-4.0 (ships inside the MaleCNS release) | Eckstein et al., Cell 2024 |
| DANDI 000951 (Cowley et al. silencing/recordings) | CC-BY-4.0 | Cowley et al. 2024; eval inputs only, never in the corpus |
| C. elegans adjacency (WormWiring SI5, corrected July 2020) | open data (wormwiring.org distribution) | Cook et al., Nature 2019; wrong-animal control only |

## Evaluated and deliberately excluded

| Asset | License | Why excluded |
|---|---|---|
| FlyWire FAFB v783 | CC BY-NC 4.0 | Non-commercial - excluded so shipped artifacts stay commercially clean; BANC covers the female side |
| malecns_09 (Lee-lab recompilation) | - | Derived re-format, not canonical - manifests use the Janelia flat-connectome release only |
| natverse / bancr / fafbseg / navis | GPL-3.0 | Quarantined to `scripts/`; never a runtime dependency |

## Verification

- `tests/` T-SHIP-3/9 check the dependency tree and artifact contents.
- Upstream file integrity: sha256 per file in `provenance/*.json`.
- If a dependency or dataset changes license, update this file and re-run
  the license audit before shipping.
