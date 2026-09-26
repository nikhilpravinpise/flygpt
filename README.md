# FlyGPT

![status](https://img.shields.io/badge/status-Phase%200%3A%20ingest%20%2B%20graph%20build-orange)
![license](https://img.shields.io/badge/code-Apache--2.0-blue)
![data](https://img.shields.io/badge/data%20%26%20weights-CC--BY--4.0-blue)

FlyGPT is, to our knowledge, the first trained generative model of a
complete synapse-resolution connectome's dynamics. It is a transformer
trained on simulated leaky-integrate-and-fire dynamics of the male
fruit-fly central nervous system connectome (MaleCNS v1.0, Janelia
FlyEM), with a sex-conditioned branch against the female BANC
connectome.

**Status: Phase 0.** Only ingest and graph build exist right now. There
is no corpus, no model, and no trained artifact yet; anything claiming
otherwise is out of date. See `docs/plan.md` for the phase plan and
`CLAIMS.md` for what is and is not currently supported by evidence.

## Quickstart

```bash
pip install -e .
python -m flygpt.ingest                    # fetch + verify canonical files into data/raw/
python -m flygpt.graph --config configs/graph_v1.yaml   # build the CSR graph into data/graph/
```

Requires Python 3.11+. Graph build needs ~4 GB free RAM.

## Repo layout

```
flygpt/
  pyproject.toml            # Apache-2.0; deps pinned; no GPL runtime deps
  README.md
  LICENSE                   # Apache-2.0
  NOTICE / THIRD_PARTY_NOTICES.md
  CITATION.cff
  CLAIMS.md                 # every public claim -> evidence artifact. CI enforces.
  EVAL.md                   # pre-registered eval protocol, committed BEFORE training
  RISKS.md                  # risk register
  provenance/               # source manifests + sha256 of every upstream file
  configs/                  # corpus/model/eval configs - versioned, one per corpus version
  results/                  # eval outputs (JSON) - the only allowed CLAIMS.md evidence
  src/flygpt/
    graph.py                # MaleCNS/BANC -> CSR build (dedup, autapse policy, NT signs)
    sim_backend.py          # adapter seam: flybrain | own LIF kernel
    stimuli.py              # stimulus program library + coverage tracker
    corpus.py               # trial runner -> event lists -> parquet shards
    tokenize.py             # events -> population vectors/patches
    model.py                # dual-axis transformer, connectome-masked cross-time attn
    controls.py             # rewired/permuted/wrong-animal/dense graph builders
    evals/                  # one module per eval; each writes JSON to results/
    behavior.py             # DN/MN -> behavior head; song decoder
  scripts/                  # GPL-tolerant zone: natverse/bancr prep tools live here only
  notebooks/                # Kaggle/Colab session drivers: checkpoint-resume wrappers around src/
  modal_app.py              # eval battery / CI CUDA smoke / odd jobs
  tests/                    # the real test suite
  space/                    # Gradio "Probe a Fly Brain" (Phase 6)
```

## Data provenance

All connectome data is pulled from the canonical Janelia release at
`gs://flyem-male-cns/v1.0/` - not the Hugging Face mirror, not the
`malecns_09` re-format. Every upstream file's sha256 is recorded in a
manifest under `provenance/` (e.g. `provenance/malecns_v1.json`); the
graph build fails closed on any checksum mismatch.

MaleCNS v1.0 is CC-BY-4.0 (Berg et al., Cell 2026; Janelia FlyEM,
University of Cambridge, Google). Any use of this package's data
artifacts - graph, corpus, embeddings, weights - requires that
attribution. BANC is likewise CC-BY-4.0 (Bates et al., Nature 2026).
See `NOTICE` and `THIRD_PARTY_NOTICES.md`.

## Determinism and conventions

Conventions are explicit, recorded, and versioned in
`configs/graph_v1.yaml` rather than baked into code:

- **Synapse cutoff**: edges kept at >=5 synapses (FlyWire network-stats
  convention); alternate cutoffs 3 and 10 are built as labeled
  sensitivity slices, never mixed into the main split.
- **Autapse policy**: `keep` for MaleCNS, counted and reported in the
  manifest. BANC excludes ~2.1% autapses at build; that asymmetry is
  documented, not hidden.
- **NT sign rule**: an edge is inhibitory iff >50% of its presynaptic
  sites are predicted GABA or glutamate at cleft score >=50, else
  excitatory. Low-confidence edges are flagged, never guessed.
- **Seeds**: derived as `hash(trial config)` - never hand-assigned.
- **Dedup**: no repeated (pre, post) pairs survive the build.
- **Weights check**: edge weights are derived independently from
  `syn-partners` and reconciled against `connectome-weights`; the
  check is that our pipeline reproduces the canonical file.

## Dev notes

- **Windows**: the Hugging Face cache on NTFS can hit long-path limits.
  Set a short `HF_HOME` (e.g. `C:\hf`) or use WSL2 for heavy local work.
- **Nothing trains locally.** The laptop orchestrates only: it launches
  jobs, monitors, and collects artifacts. Corpus generation and training
  run on remote burst/drip compute (see `docs/free-compute-research.md`).
- Every entry point lives in `src/flygpt/`; `notebooks/` and
  `modal_app.py` are thin checkpoint-resume drivers over the same code.

## License

- **Code**: Apache-2.0 (`LICENSE`).
- **Datasets, embeddings, model weights**: CC-BY-4.0 with the upstream
  attribution chain (Berg 2026, Bates 2026, Eckstein 2024).
- **FlyWire (FAFB v783) is CC BY-NC 4.0 and deliberately excluded** so
  every shipped artifact stays commercially clean; BANC covers the
  female side instead.
- GPL-3.0 tools (natverse/bancr/fafbseg/navis) are quarantined in
  `scripts/` and never imported by `src/flygpt`.

## Disclaimer

All training data is **simulated, not recorded**: the corpus is output
of a leaky-integrate-and-fire simulator run on connectome structure, and
the corpus's validity ceiling is the simulator's. FlyGPT is not a
digital fly, not an upload, and not an embodied agent - it is a learned
model of network dynamics whose claims stand or fall on held-out
controls. The controls are the product: rewired, permuted-label,
wrong-animal, dense, and linear baselines ship alongside the model, and
every public claim links to its evidence artifact in `results/` via
`CLAIMS.md`.
