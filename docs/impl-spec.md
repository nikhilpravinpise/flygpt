# FlyGPT implementation spec (contracts)

Companion to `docs/plan.md`. This file pins the exact module APIs, file
schemas, and conventions every component codes against. It exists so
parallel implementers produce compatible code. Where it conflicts with
plan.md, plan.md wins and this file gets fixed.

**Global rules**

- Style: match `src/flygpt/graph.py` + `tests/test_graph.py` - numpy/
  polars/pyarrow, `from __future__ import annotations`, docstring-first
  modules that state schema + plan refs, `main(argv)` CLI entry, ruff
  line-length 110, no em-dashes in comments/strings (plain `-`).
- Every public API is deterministic given `(config, seed)`; seeds come
  only from `flygpt.schema.seed_from_config` - never hand-assigned.
- Every eval writes `results/<test-id>.json` via
  `flygpt.schema.write_result`.
- Tests run in CI on synthetic fixtures ONLY (no 10 GB dataset).
  Real-artifact checks use the `@real_data` skipif pattern from
  `tests/test_graph.py`.
- jax/flax/optax are the model framework (plan Phase 3: TPU drip).
  Import them lazily inside functions/modules that need them -
  `import flygpt` must work without jax installed.
- flybrain is an optional reference backend; import lazily, degrade
  gracefully when absent.

## Shared infra already in place

`src/flygpt/schema.py` (READ THIS FIRST):

- `SCHEMA_VERSION`, `EVENT_*`, `TRIAL_COLUMNS` - event/trial schemas.
- `VOCAB_*`, `TYPE_TOKEN_BASE`, `N_STIMULUS_SLOTS`, `N_BEHAVIOR_SLOTS`.
- `seed_from_config(obj) -> int`, `rng_from_seed(*parts)`,
  `canonical_json`, `git_sha`, `utcnow`, `sha256_file`.
- `GraphBundle` dataclass: `nodes` (pl.DataFrame), `n_nodes`, `cutoff`,
  `out_indptr/out_indices/out_weight/out_sign` (CSR by PRE),
  `in_indptr/in_indices/in_weight/in_sign` (CSR by POST, transposed),
  `signed_weights_out/in`, `node_of_body`, `type_array`,
  `manifest_types`, `in_neighbors`, `out_neighbors`.
- `load_graph_bundle(graph_dir, cutoff)`, `bundle_from_edges(...)`.
- `write_result(results_dir, test_id, checks, inputs=, metrics=,
  passed=, extra=)` - the results envelope.
- `tests/conftest.py::synthetic_build` - session fixture running the
  REAL graph build on a 6-node hand-computed connectome; its
  `data/graph`-shaped output is the standard test graph
  (`root/graph/{nodes,edges}.parquet`, `csr_cutoff{3,5,10}.npz`).

## sim_backend.py - LIF simulator (T-SIM-*)

Shiu-recipe alpha-synapse LIF, bi-exponential discrete form, plus
flybrain's fly64 form as an option (cross-checked vs the real flybrain
adapter). Per-edge sign + count weights from GraphBundle.

```python
@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.020                 # seconds per step
    tau_v: float = 0.100              # membrane tau, seconds
    tau_g: float = 0.010              # synaptic conductance tau
    w_syn: float = 0.01               # THE free param (per-synapse scale);
                                      # pilot-calibrated (sim_v1.yaml)
    threshold: float = 1.0
    reset: float = 0.0
    refractory_steps: int = 0
    weight_mode: str = "count"        # count|binary|normalized (slices)
    tonic_rate_hz: float = 0.0        # global tonic Poisson drive
    tonic_amount: float = 0.0         # per-event tonic voltage kick
    noise_hz: float = 0.0
    noise_amp: float = 0.0
    synapse_model: str = "alpha"      # alpha|instant (fly64 form)
    backend: str = "numpy"            # numpy|jax|flybrain
    warmup_steps: int = 0
    seed: int = 0
    drive_scale: float = 1.0          # g->v coupling constant

@dataclass
class TrialResult:
    events_step: np.ndarray       # u32, sorted by (step, neuron)
    events_node: np.ndarray       # u32
    n_steps: int
    n_nodes: int
    rates: np.ndarray             # f32 per-node rate over recorded span
    config: SimConfig
    seed: int
    program: dict                 # echoed stimulus spec
    meta: dict                    # backend/device/wall seconds
    def events_per_step(self) -> list[np.ndarray]: ...
    def active_mask(self) -> np.ndarray: ...

class LIFSim:
    def __init__(self, bundle: GraphBundle, cfg: SimConfig): ...
    def reset(self, seed: int) -> None
    def run(self, n_steps, driver=None, silence_nodes=None) -> TrialResult
    # driver: callable(step:int, sim) -> None injecting spikes/voltage
    # silence_nodes: nodes whose spikes are dropped pre-propagation AND
    #   never recorded (T-STIM-5)
```

Discrete update (documented + versioned in config):

    g[t+1] = g[t] * exp(-dt/tau_g) + sum_{pre fired at t} w_signed(pre,post)
    v[t+1] = v[t] * exp(-dt/tau_v) + drive_scale * g[t+1] + tonic_v
    fired  = v >= threshold ; v[fired] = reset

`synapse_model="instant"`: `v[t+1] = v[t]*exp(-dt/tau_v) + gain *
sum_spikes w + tonic + noise` (flybrain's fly64 equation; gain=w_syn
role). Stimulation arrives as exogenous spikes on the target nodes
(Shiu protocol: Poisson drive on the identified cell type, 10-260 Hz),
so a driven neuron both emits an event and propagates. Determinism:
same (seed, config, backend, device-class) -> identical event lists;
numpy backend is required to be bitwise-deterministic on CPU (T-SIM-1).

`FlybrainAdapter`: wraps `flybrain.FlyBrain` on ITS OWN baked graph
(different conventions: per-neuron sign incl. histamine, input
normalization - documented in the adapter docstring); used by the
consistency eval only, never for corpus generation.

`simulate(bundle, program, cfg) -> TrialResult`: builds the driver from
`program["drives"]` (see stimuli.py), runs warmup (dropped), returns
events recorded after warmup.

## stimuli.py - stimulus program library + coverage tracker (T-STIM-*)

```python
@dataclass(frozen=True)
class Drive:
    nodes: np.ndarray        # u32 node indices
    rate_hz: float           # Poisson drive rate
    window: tuple[float,float]  # seconds, [t0, t1)
    envelope: str = "poisson"   # poisson|naturalistic|tonic|step
    envelope_id: str = ""       # deterministic envelope key
    # naturalistic only: the sampled OU trace recorded in the artifact
    # (T-STIM-4); the simulator plays it back verbatim
    envelope_values: np.ndarray | None = None
    envelope_dt: float | None = None
    envelope_max: float | None = None

@dataclass(frozen=True)
class StimulusProgram:
    family_id: str
    program_id: str
    drives: tuple[Drive, ...]
    silences: tuple[tuple[np.ndarray, tuple[float,float]], ...]  # nodes,window
    duration_s: float
    dt: float
    meta: dict                 # JSON-able; hashed for the seed
    def spec(self) -> dict: ...       # canonical JSON-able spec
    def seed(self) -> int: ...        # seed_from_config(self.spec())

class SensoryIndex:
    """Resolves grounding queries -> node_idx arrays via nodes table."""
    def __init__(self, bundle: GraphBundle): ...
    def resolve(self, *, types=None, classes=None, subclasses=None,
                superclasses=None, sides=None, neuromeres=None,
                entry_nerves=None) -> np.ndarray

LIBRARY: dict[str, callable]  # family_id -> program generator
class CoverageTracker:
    def record(self, program, result_nodes_fired) -> None
    def report(self) -> dict    # driven/robust per manifest type
```

Families (each a generator `(index, rng, **kw) -> StimulusProgram`):
`olfactory_pulse` (ORN_{glom} types, lateralized),
`visual_lc` / `lc_silence` (the 23 DANDI LC types - config
`configs/eval_panels.json`), `gustatory` (sugar/bitter/water via
subclass: labellar bristle + taste peg + PhG* = feeding/sugar axis;
leg bristle LgA*/LB* = water/bitter axis - document mapping),
`mechanosensory` (chordotonal=JON sound/grooming, wind_gravity,
campaniform, bristle, grooming subclass), `courtship` (visual cue +
pheromone ORN/GRN composite), `optogenetic` (arbitrary type
activation/silencing incl. the LC benchmark family),
`tonic_baseline` (whole-network tonic sweeps over rate levels),
`naturalistic` (envelope-driven sensory programs - bounded,
non-negative OU/pink process sampled at dt),
`babble` (random disjoint subsets, seeded), `silence_test`
(unstimulated controls). Every program records the grounding query in
`meta` so T-STIM-1 can re-resolve it.

## corpus.py - trial runner -> event lists -> parquet shards (T-CORPUS-*)

```python
@dataclass(frozen=True)
class TrialSpec:
    family_id: str; program_id: str; slice: str; split: str
    cfg: SimConfig              # carries dt/backend/conventions
    def program_spec(self) -> dict

def run_trials(bundle, specs, sim_factory, work_dir, shard_prefix,
               trials_per_shard=..., dead_letter=True) -> list[Path]
# events parquet: schema_v1 EVENT_COLUMNS (u64,u32,u32, sorted by
#   trial,step,node) - pyarrow dictionary-encoded where beneficial
# trials parquet: TRIAL_COLUMNS rows
# manifest: <shard>.manifest.json {schema, corpus_version, shard_id,
#   code{git_sha}, graph{version,sha256}, sim_cfg, backend, gpu_class,
#   dt, cutoff, slice, split, trial_ids, counts, source_manifest,
#   created_utc}

def plan_corpus(cfg, index=None, *, bundle=None, ...) -> list[TrialSpec]
def build_splits(cfg, bundle) -> dict   # S-STIM/S-TYPE/S-REGION rules
def health_metrics(events, n_nodes, n_steps, types, dt=None, n_driven=-1) -> dict
def run_pilot(bundle, library, cfg, work_dir, results_dir) -> dict
def run_corpus(graph_dir, corpus_config, ..., slices=None, backend="numpy",
               gpu_class="cpu", max_shards=None) -> list[Path]
#   groups multi-cutoff slices per CSR; manifests record the real cutoff
#   -> results/pilot_report.json {health, spot_checks, throughput}
```

Dead-letter: failed trials -> `deadletter.jsonl` + retry once; shard is
written only when its trial set is complete. `gpu_class` is read from
env/config, never auto-guessed. Splits: family atomic; 5% typed holdout
seeded `hash("corpus_v1:celltype_holdout")` stratified region x NT;
>=1 whole region per compartment held out - all regenerated from the
corpus manifest (T-CORPUS-4).

## tokenize.py - events -> population vectors/patches (T-TOK-*)

```python
@dataclass
class Vocab:
    type_to_id: dict[str,int]     # manifest types -> TYPE_TOKEN_BASE+i
    instance_to_id: dict[int,int] # node_idx -> instance token (designated
                                  # circuits: config list, e.g. vnc_motor +
                                  # courtship types)
    stimulus_to_id: dict[str,int]
    behavior_to_id: dict[str,int]
    def save/load(path); def size() -> int
def build_vocab(bundle, cfg) -> Vocab        # writes vocab manifest json
def node_to_token(bundle, vocab) -> np.ndarray   # per-node token id
def events_to_counts(result|events, node_token, vocab) -> (T,V) u32
def counts_to_events(counts, vocab) -> list[(bin,token,count)]
```

Round-trip exactness: event identity + bin index exact; continuous
values within quantization eps recorded in `configs/tokenizer_v1.yaml`.
Bin index = `floor((t - t0)/dt)` (floor convention, T-TOK-4).
Untyped -> VOCAB_UNK, never dropped (T-TOK-3). Instance slots occupy a
disjoint contiguous block after type tokens (T-TOK-2).

## model.py - dual-axis connectome transformer (T-MODEL-*) [JAX/Flax]

```python
@dataclass
class ModelConfig:
    d_model: int = 512; n_heads: int = 8; d_ff: int = 2048
    n_layers: int = 12              # alternates time/neuron axis
    n_vocab: int                    # = vocab.size()
    max_in_degree: int = 64         # gathered presyn partners cap
    dropout: float = 0.1
    param_target: int = 100_000_000 # headline run; dev configs smaller
    silence_weight: float = 0.1     # loss weight on inactive units
class ConnectomeTransformer(nn.Module):   # flax.linen
    """Tokens (B,T,V=type-space) -> event logits (B,T,V).
    Even layers: causal attention along T per unit.
    Odd layers: connectome-masked attention - unit u at step t attends
    to presynaptic partners of u at t-1 (gathered in-neighbors, padded
    to max_in_degree). mask[post,pre]; transpose-checked (T-MODEL-9)."""
class GNNBaseline(nn.Module)      # same interface (T-MODEL-10)
class DenseTransformer(nn.Module) # param-matched no-mask (T-MODEL-5)
def train(cfg, data, work_dir) -> Path  # checkpoints (normal resume path)
def load_checkpoint(path) -> (params, cfg)
def sex_conditioned_forward(params, mask_male, mask_female, sex) # T-SEX-5
```

Loss: per-unit event BCE + rate term, silence-weighted (silence-robust,
T-MODEL-8). The unit axis is the *type* axis (population vectors);
instance-token circuits pass through the same layer.

## controls.py - control graph builders (T-MODEL-3/4/5, T-CORPUS slices)

```python
def rewire(bundle, seed) -> GraphBundle      # degree+weight preserved
def permuted_labels(bundle, seed) -> GraphBundle  # type->token remap
def dense_mask_bundle(n, weight=1.0) -> GraphBundle
def no_edge_bundle(n) -> GraphBundle          # direct input only
def wrong_animal_bundle(path=None) -> GraphBundle  # C. elegans loader
def shuffled_weights(bundle, seed) -> GraphBundle  # topology kept
def linear_influence(bundle, hops=3, decay=0.5) -> np.ndarray  # (V,V)ish
```

C. elegans: WormWiring adjacency (Cook et al. 2019, N2U hermaphrodite +
male) downloaded into `data/raw/worm/` with `provenance/celegans_v1.json`
- small CSV/Excel, N~300-450. Rewire: stub-matching/configuration model
preserving in+out degree sequences AND the edge-weight multiset.
Permuted labels: permute `type` assignment among same-superclass nodes
(documented strata). Shuffled weights: permute `weight`+`sign` across
edges (corpus slice `shuffled`).

## evals/ - the battery (one module per test family)

`evals/common.py`: `event_f1(y_true,y_pred)` (macro+micro+per_type,
inclusion rule per EVAL 1.2), `rate_pearson`, `rate_spearman`,
`majority_baseline`, `allzero_baseline`, `spike_agreement(a,b,n)` (EVAL
1.4: event / +-1-step / rate-Pearson), `ci_mean(x)` (Student-t),
`bh_fdr(p,q)`, `silence_guard` invariant. `evals/sim_checks.py`
(T-SIM-1..8), `evals/corpus_checks.py` (T-CORPUS-1..7),
`evals/tok_checks.py` (T-TOK-1..5), `evals/stim_checks.py`
(T-STIM-1..5), `evals/emb_evals.py` (T-EMB-1..4), `evals/model_evals.py`
(T-MODEL-1..11 runners over checkpoints+control results),
`evals/sex_evals.py` (T-SEX-1..6), `evals/real_evals.py`
(T-REAL-1..3), `evals/ship_checks.py` (T-SHIP-2/3/4/5/9/10). Every
module: `run(...) -> dict` + `main()` writing results/<test-id>.json.

## banc.py - female connectome (Phase 4, T-SEX-*)

Lee-lab public bucket `lee-lab_brain-and-nerve-cord-fly-connectome`:
`compiled_data/banc_888/{banc_888_meta.feather,
banc_888_edgelist_simple_v2.feather}` (v2 = size>=5 chosen to match the
male >=5 convention - rationale recorded; v3 = size>=10), meta has
79 cols incl. `banc_888_id` (string), `super_class`, `cell_class`,
`cell_sub_class`, `cell_type`, `region`, `side`, `flow`, cross-dataset
`*_cell_type`/`*_match` cols, `neurotransmitter_predicted`,
`proofread`. Edgelist cols: `pre`,`post` (string ids), `count`,
`norm=count/post_count`, `post_count`,`pre_count`. **Autapses excluded
at build (~2.1%) - documented asymmetry vs male `keep`.**
`nblast/banc_malecns_reviewed_matches.csv` + meta `malecns_cell_type`
give the curated alignment; NBLAST table is the backup.

```python
BANC_REGISTRY: tuple[SourceFile-like] (same manifest pattern as ingest)
def download_banc(keys, dest, manifest) -> manifest dict  # sha256'd
def build_banc_graph(cfg, sources, graph_dir, prov_dir, results_dir)
def alignment_table(meta, matches_csv) -> pl.DataFrame
    # columns: male_type, banc_type, banc_id, iso_class
    # iso_class in {isomorphic, dimorphic, male-specific, female-specific,
    #               male-only-unmatched, female-only-unmatched}
def sex_conditioned_vocab(male_bundle, banc_bundle, align) -> Vocab
def female_graph_in_male_space(banc_bundle, align, vocab) -> GraphBundle
# aligned types share the token; male-specific types exist only in the
# male mask, female-specific only in the female mask (T-SEX-4/5)
```

## behavior.py - DN/MN -> behavior head; song decoder (C19, T-REAL-3)

`configs/decoder_spec_v1.yaml` - THE written spec (per behavior:
readout query on superclass/subclass/exitNerve/somaNeuromere + decoding
rule + sex applicability). Wing MNs -> song, leg MNs -> locomotion,
DNp01 -> escape etc. `BehaviorDecoder(bundle, spec).decode(result) ->
[{behavior, t0, t1, evidence}]`. `SongDecoder` at dt=5ms: pulse/sine
bout segmentation, IPI distribution, bout duration, context switching -
`evals/song_reference.json` holds the SongExplorer ground-truth ranges
(EVAL T-REAL-3: IPI median [30,45]ms etc.).

## space/ - "Probe a Fly Brain" (Phase 6, T-SHIP-7)

Static Space (robust default): `space/index.html` + JS rendering
canned rollouts from `space/canned/*.json` produced by
`space/build_canned.py` (runs real small sims locally, caches event
traces + ethogram stats). Zero live GPU/sim calls in the demo path.
Optional `space/app.py` Gradio variant behind `--gradio` (spaces SDK
optional). Stimulus buttons map to real annotated populations.

## Data downloads (new)

- BANC files into `data/raw/banc/` + `provenance/banc_v888.json`.
- C. elegans adjacency into `data/raw/worm/` +
  `provenance/celegans_v1.json`.
- DANDI 000951: `flygpt.evals.real_evals` lists silencing-session
  assets via `api.dandiarchive.org` (subject = `sub-<LCtype>-fly<n>`);
  partial download only, recorded in `provenance/dandi_000951.json`.
- All downloads: stream + sha256 manifest, same pattern as ingest.py.
