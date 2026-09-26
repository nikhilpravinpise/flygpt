# EVAL.md - pre-registered evaluation protocol

**Version**: `eval_v1` | **Date committed**: 2026-09-22 | **Status**: ratified

Committed before Phase 3 training per plan.md section 5 and the hard rules in
section 6. Amendments require a new EVAL version section appended at the bottom
of this file - ratified thresholds are never edited in place. "Beats baseline"
without a number is not a criterion: every threshold below is written before any
training run.

Two implementers given the same eval split and this document must produce the
same number. Anything ambiguous gets nailed here or flagged `set at pilot gate`
with the fixing procedure stated.

---

## 1. Metric definitions

### 1.1 Binned rates and event binarization

- Bin width `dt` = the corpus slice's sim step: 20ms for the main corpus, 5ms
  for the courtship slice. dt is read from the shard manifest; slices never mix
  (T-CORPUS-7).
- `c_{i,t}` = spike count of cell type `i` in bin `t`. Bin index uses the floor
  convention: `t_bin = floor((t_spike - t0) / dt)` (tested by T-TOK-4).
- Binned rate `r_{i,t} = c_{i,t} / dt` in Hz.
- Event indicator `e_{i,t} = 1 iff c_{i,t} >= 1` (any spike in the bin).
- Predicted events: the model emits per-(type, bin) event probability `p_{i,t}`;
  `e_hat_{i,t} = 1 iff p_{i,t} >= 0.5`. The 0.5 threshold is fixed for all evals
  - never tuned per split.

### 1.2 Event-F1

Per type `i` over all concatenated bins of the eval split:
`F1_i = 2TP_i / (2TP_i + FP_i + FN_i)` on the binarized events of 1.1.

- **Macro event-F1** = mean of `F1_i` over *included* types. A type is included
  iff it has >= 1 true event in the split OR >= 1 predicted event. A type silent
  in both truth and prediction is excluded (it contributes no information, and
  counting it as perfect would reward all-zero outputs).
- **Micro event-F1** = F1 computed on the pooled (type, bin) set.
- Report both. Headline numbers use macro unless stated otherwise.

### 1.3 Rate correlation

- `r_i` = Pearson correlation between predicted and true `r_{i,t}` vectors for
  type `i` across concatenated eval bins. A type qualifies iff both sequences
  have nonzero variance. **Macro-r** = mean `r_i` over qualifying types; the
  count of non-qualifying (zero-variance) types is reported alongside.
- Spearman `rho_i` is computed identically on ranks and is the metric for all
  real-data rank/order comparisons (T-REAL-2).

### 1.4 Spike-time agreement rate (T-SIM-1 / T-SIM-2 / T-CORPUS-6)

For two runs A, B of an identical (seed, config, backend) trial, on raw event
lists at native sim resolution (before binning):

- **Event agreement** = fraction of (neuron, step) pairs whose active/inactive
  state matches.
- **+-1-step agreement** = same, but an active/inactive mismatch counts as
  matched if the other run has an event for that neuron in an adjacent step.
- **Rate agreement** = Pearson r over per-neuron event counts pooled across the
  trial.

Thresholds:

| Regime | Requirement |
|---|---|
| CPU backend, same machine | bitwise identical event lists |
| Same GPU class, GPU backend | event agreement >= 99.5%; +-1-step >= 99.9%; rate Pearson >= 0.999 |
| Cross-GPU-class or cross-backend | event agreement >= 99.0% within +-1 step; rate Pearson >= 0.99 - measured and documented only; shards from different backends/GPU classes never mix in one corpus version regardless |

### 1.5 Shiu-protocol agreement (T-SIM-7 / Shiu replication)

Protocol reimplemented, not copied (upstream repo has no license): Poisson
drive 10-260 Hz on the identified cell type/population, 30 trials x 1000 ms per
experiment, matching the reference design.

- A *predicted response* for a recorded target population = sign of mean rate
  change vs the experiment's own baseline window, CI-separated across the 30
  trials (95% CI excludes zero). "No-effect" is predicted iff the CI includes
  zero AND the point estimate is below the experiment noise floor (the 95th
  percentile of |rate change| over unstimulated control trials of the same
  target).
- **Agreement** = fraction of testable predictions whose sign matches the
  published empirical outcome. The frozen experiment list lives in the eval
  config (`evals/shiu_panel.json`) before the eval runs; the reference numbers
  are 91% on 164 testable predictions, 84% excluding the split-GAL4 panel, and
  shuffled-weight controls collapsing to 1/100.

### 1.6 Perturbation-benchmark score (T-REAL-1)

DANDI 000951 (459 sessions, 23 LC types silenced; partial download of the
silencing sessions only). For each silenced LC type the model predicts the
direction of effect on the dandiset's published readout
(increase/decrease/no-effect), using the sign rule of 1.5. Score = fraction of
correct-direction predictions across the 23 types. Per-type significance by
binomial test across seeds; the 23-type panel is BH-FDR corrected.

### 1.7 Silence-robust handling

Raw accuracy is never reported anywhere. Every headline metric is event-F1
(1.2) or an active-subset rate correlation (1.3). Two explicit baselines run on
every eval split: an all-zero predictor and a per-type marginal-rate
("majority") predictor. Sanity invariant (T-MODEL-8): the all-zero predictor
must score <= the majority baseline on every split; if it doesn't, the metric
pipeline is broken and the eval is void.

### 1.8 Coverage (T-CORPUS-3)

A manifest type is **driven** iff it emits >= 1 spike in >= 1 corpus trial, and
**robustly driven** iff it emits >= 5 spikes across >= 10 distinct trials.
Coverage = count / corpus-manifest `types` total - the denominator is the
corpus's own declared driven-type scope (recorded in
`corpus.manifest.json` at generation time), falling back to the brain
manifest when none is declared. A full corpus declares the whole manifest
(~11.7K types; neuPrint v1.0 snapshot reports 11,751, the Cell paper
11,710); a scoped corpus (e.g. the 10-family pilot, 203 declared types) is
judged against what it declared it would drive.

### 1.9 Sex-divergence metric (T-SEX-2)

For homologous stimulus program `s` run under both conditionings, per type `i`
in the pre-registered set: `d_i = |r_male,i - r_female,i| / (0.5*(sd_male,i +
sd_female,i) + eps)`, `eps = 1e-6`. Circuit divergence = mean `d_i` over the
circuit's types. A dimorphic circuit is *divergent* iff its divergence exceeds
the 95th percentile of the same metric computed over matched isomorphic-type
control circuits, with consistent direction across >= 3 seeds.

### 1.10 Speed ratio (T-SHIP-10)

`speedup = model forward steps/sec / sim steps/sec`, both measured on the same
GPU class in the same session. The "faster than the LIF" claim (C18) requires
measured speedup >= 10x; the raw number is reported regardless.

---

## 2. Global policies

- **Seeds**: >= 3 per eval, derived as `hash(trial config)` or enumerated in the
  eval config - never hand-picked post hoc. Report mean + 95% CI (Student-t,
  n-1). A headline claim requires the CI bound on the relevant side to clear the
  threshold, not just the point estimate.
- **Multiple comparisons**: Benjamini-Hochberg FDR within every
  multi-circuit/multi-perturbation panel (the 23-type LC panel, dimorphic
  circuit panels, per-type probe panels). Primary panels use q <= 0.05;
  exploratory panels may report q <= 0.10 and must be labeled exploratory.
  Uncorrected per-item p values are reported alongside.
- **"Set at pilot gate" procedure**: only items explicitly tagged below may be
  set from measurement. Fixing procedure: the pilot writes
  `results/pilot_report.json`; the value is computed by the stated rule from
  that file, recorded in the corpus manifest, and committed before Phase 2
  corpus generation. No other threshold may be set this way.
- **Eval cadence**: the battery runs on intermediate checkpoints, not just the
  final one (late-stage overfitting to sim artifacts is a named failure mode).
- **Claim mapping**: every EVIDENCED row in CLAIMS.md links an artifact produced
  under this protocol. A failed test downgrades or refutes the claim - it never
  quietly disappears.

---

## 3. Split definitions

Split membership is recorded in the corpus manifest (`splits` block) at corpus
generation and is reproducible from the manifest alone. Stimulus *families* are
defined in Phase 2; this file fixes the RULES, the manifest fixes the LIST.

- **S-STIM (held-out stimulus families)**: families are atomic - a `family_id`
  is entirely in train or entirely in test; no trial-level split inside a
  family. Rule: >= 1 held-out family per sensory modality, and the
  naturalistic-envelope family is always held out. Membership enumerated in
  `splits.stimulus_holdout`. Exception recorded in the manifest: the
  benchmark-matched LC-silencing family is train-eligible (matched training
  distribution for the perturbation benchmark) while its eval stimuli stay
  disjoint at the trial level.
- **S-TYPE (held-out cell types)**: seeded stratified sample (seed =
  `hash("corpus_v1:celltype_holdout")`) of 5% of typed cell types, strata =
  region x NT-class, enumerated in `splits.celltype_holdout`. "Held out" means
  the type is never a primary-stimulated target in any train trial; it still
  receives propagated activity - predicting it is the connectome-generalization
  test (T-MODEL-2).
- **S-REGION (held-out regions)**: >= 1 whole annotated region per major
  compartment (optic lobe / central brain / VNC), drawn from the
  body-annotations region field, enumerated in `splits.region_holdout`.
- **Mutual rules**: convention slices (synapse cutoffs 3/5/10) never mix in a
  split; dt slices never mix in a split; male and female corpora are disjoint;
  degenerate trials are a labeled class, never silently dropped. T-CORPUS-4
  regenerates membership from the manifest and asserts zero overlap on all
  three axes.

---

## 4. Pre-registered dimorphism set (T-SEX-1)

Candidate effects drawn from the Cell 2026 male/female comparison (8,069
isomorphic / 138 dimorphic / 289 male-specific / 71 female-specific types;
fru/dsx annotations; male-specific connection "hotspots" in higher centers;
sensory/motor periphery largely isomorphic). The frozen enumeration - exact
type lists from the Cell iso/dimorphic annotation crossed with the BANC
alignment - lands as `evals/dimorphism_set.json` **before Phase 4**. This list
is the pre-registered skeleton:

- **DS-1**: LoVP92 ("love spot") male-specific courtship type: driven by
  female-cue stimulus programs under male conditioning; masked (absent) under
  female conditioning.
- **DS-2**: fru+/dsx+-annotated dimorphic populations show divergent rate
  dynamics under identical homologous courtship stimuli, direction-consistent
  across seeds.
- **DS-3**: propagation differences under male vs female conditioning localize
  to the Cell-documented dimorphic/hotspot type set rather than spreading
  diffusely.
- **DS-4** (negative control): isomorphic sensory/motor periphery shows
  divergence within the isomorphic null band - a no-effect check guarding
  against "everything diverges" artifacts.
- **DS-5**: dimorphic rerouting: downstream activated-type sets differ between
  conditionings on the documented dimorphic types, dynamically reproducing the
  structural rerouting finding.
- **DS-6**: sex-specific type accounting: male-specific types present under
  male conditioning and masked under female; female-specific vice versa
  (shared check with T-SEX-4).

Pass threshold: >= 80% of the frozen DS items reproduced. Kill-condition
mapping in section 6.

---

## 5. Per-test thresholds (every test in plan.md section 3)

### T-BUILD - graph construction

| ID | Pass criterion |
|---|---|
| T-BUILD-1 | Node set == body-annotations row count exactly (expected ~165K; 166,700 per Cell paper / 164,838 per neuPrint v1.0 snapshot - actual recorded in manifest). All annotation rows exist as nodes; isolated rows retained as zero-degree nodes; dropped rows = 0 |
| T-BUILD-2 | Derived (pre, post, weight) table == canonical `connectome-weights` row-for-row at minconf 0.5 after dedup (published totals vary by convention - ~25.6M aggregated pairs, ~151.9M connections in one independent recompute; the check is that OUR pipeline reproduces the canonical file, not a number from a paper). Edge counts reported at cutoffs 3/5/10 |
| T-BUILD-3 | Self-loop count == manifest autapse count exactly; `keep` policy applied. Asymmetry vs BANC (~2.1% excluded there) documented, not aligned |
| T-BUILD-4 | 100% of edges carry the sign implied by `nt_sign_rule` (>50% of presynaptic sites GABA/glutamate at cleft score >= 50 -> inhibitory); low-confidence edges flagged; guessed edges = 0 |
| T-BUILD-5 | sha256 match on 100% of downloaded upstream files vs `provenance/*.json`; any mismatch = fail closed |
| T-BUILD-6 | Duplicate (pre, post) pairs post-build = 0 |
| T-BUILD-7 | Typed fraction recorded; untyped count == manifest; 100% of untyped bodies route to UNK; silently dropped = 0 |

### T-SIM - simulator correctness

| ID | Pass criterion |
|---|---|
| T-SIM-1 | Per 1.4: CPU bitwise; same-GPU-class event agreement >= 99.5%, +-1-step >= 99.9%, rate Pearson >= 0.999 |
| T-SIM-2 | CPU vs GPU same trial: >= 99.0% within +-1 step, rate Pearson >= 0.99. Corpus rule: single backend + single GPU class per corpus version |
| T-SIM-3 | Single-neuron pulse: 100% of t+1 rate changes confined to the pulsed neuron's postsynaptic partners; changes at t = 0 beyond rel eps 1e-6 |
| T-SIM-4 | Known-sign panel (>= 20 excitatory/inhibitory pairs, edges >= 10 synapses): >= 95% correct direction, each CI-separated across trials. Protocol: paired-recording isolation - all nodes except the pair silenced (a single edge is unresolvable under convergent tonic noise on a 211k-node graph); pre silenced in the baseline arm; per-sign tonic levels (subthreshold for excitatory readout, saturating endogenous baseline for inhibitory suppression) |
| T-SIM-5 | Zero-baseline unstimulated network: exactly 0 events. Tonic-drive network: nonzero activity. Exact both ways |
| T-SIM-6 | Pilot health: degenerate-trial fraction <= 5% (degenerate = active-type fraction < 0.5% or > 60% sustained, or split-half stationarity r < 0.5); >= 90% of trials stationary at r >= 0.5; reference band for "sane" rate/activity ranges: **set at pilot gate** (band = pilot median +- 3xMAD per family, committed with `pilot_report.json`). Population-level criteria apply only to population-level stimuli: zero-drive programs are silence controls (degenerate only on >60% blow-up) and drives too small to reach the 0.5%-of-types floor alone (`< 0.005 * n_types` driven cells, ~58 on the real graph) are sparse perturbations (degenerate only on a zero event stream or >60% blow-up), labeled `expected_sparse` per trial |
| T-SIM-7 | >= 8/10 Shiu spot-checks directionally correct per 1.5, in MaleCNS cell-type identities. Panel fixed in `evals/shiu_panel.json`; seeded by: sugar GRN -> proboscis-feeding MNs (companion taste paper), JON -> downstream (postsyn proxy for grooming-circuit propagation), LC4/LPLC2 -> DNp01 escape; inhibitory sanity uses the graph's sign table (Mi9, inhibitory-majority) against a tonic baseline - a decrease is unmeasurable on a silent open network |
| T-SIM-8 | Corpus activity stats vs published physiology ranges: flagged-outlier fraction reported; unreviewed outliers shipped = 0. Stats are mean **per-neuron** sustained rates (events / n_steps / dt / n_nodes), not whole-network event throughput |

### T-CORPUS - dataset integrity

| ID | Pass criterion |
|---|---|
| T-CORPUS-1 | 100% of shards validate the declared schema (trial_id, events, stimulus program, sim config, seed) |
| T-CORPUS-2 | 100% of shard manifests resolve to real source hashes + code SHA |
| T-CORPUS-3 | **X = 90%**: >= 90% of the corpus manifest's declared `types` driven >= once; secondary floor >= 75% robustly driven (1.8). Denominator = the corpus's declared scope, or the full brain manifest when undeclared |
| T-CORPUS-4 | 0 overlapping items between train and test on all three axes (families, cell types, regions), verified by regeneration from manifest |
| T-CORPUS-5 | 100% of convention-variant shards labeled; 0 convention slices mixed within a split |
| T-CORPUS-6 | CI regenerates a small shard from manifest + seed on same backend + GPU class: CPU-backend shards bitwise-identical; GPU within T-SIM-1 tolerance |
| T-CORPUS-7 | 0 mixed-dt trials inside any split; dt field present on 100% of shards |

### T-TOK - tokenizer integrity

| ID | Pass criterion |
|---|---|
| T-TOK-1 | events -> tokens -> vectors -> tokens round-trip: event identity and bin index exact; continuous values within declared quantization eps (recorded in tokenizer config) |
| T-TOK-2 | 0 token-ID collisions across all ~11.7K type slots and all designated neuron-level instance slots (hybrid vocab ranges disjoint) |
| T-TOK-3 | 100% of untyped neurons route to UNK; dropped or merged-into-real-type count = 0 |
| T-TOK-4 | Bin-boundary events land in `floor((t - t0)/dt)`; misassigned boundary test vectors = 0 |
| T-TOK-5 | vocab size == manifest type count + reserved slots (UNK, PAD, sex, stimulus, behavior) exactly |

### T-STIM - stimulus correctness

| ID | Pass criterion |
|---|---|
| T-STIM-1 | 100% of stimulus programs resolve to annotated sensory cell types present in the graph (GRN/JON/LC/ORN families verified against body-annotations); dangling references = 0 |
| T-STIM-2 | 100% of lateralized programs drive the flagged hemisphere's neurons |
| T-STIM-3 | Coverage-tracker output == independent recount from shard metadata, exactly |
| T-STIM-4 | 100% of naturalistic-envelope samples within [0, stated max] and sampled at the declared dt |
| T-STIM-5 | Silenced windows: 0 spikes AND 0 outgoing synaptic effect from silenced populations. Exact |

### T-EMB - embedding artifact

| ID | Pass criterion |
|---|---|
| T-EMB-1 | Held-out cell-type recovery: kNN (k=10) top-1 accuracy >= 2x majority-baseline rate AND >= degree-matched node2vec/spectral baseline on the same graph + 0.05 macro |
| T-EMB-2 | NT-class linear probe (7 classes): accuracy >= majority + 0.15 absolute |
| T-EMB-3 | Iso/dimorphic/male-specific linear probe: balanced accuracy >= 0.50 (chance ~0.33) |
| T-EMB-4 | NBLAST agreement: fraction of each type's top-10 embedding neighbors inside its NBLAST top-50 >= 3x chance overlap |

### T-MODEL - the claims that matter

| ID | Pass criterion |
|---|---|
| T-MODEL-1 | Held-out stimulus families: macro event-F1 >= majority baseline + 0.10, with the 95% CI lower bound > +0.05. CI including <= +0.05 triggers kill condition K1 |
| T-MODEL-2 | Held-out cell types (never primary-stimulated): macro event-F1 >= majority + 0.05 AND macro rate Pearson >= 0.20 |
| T-MODEL-3 | Real mask beats BOTH controls by >= 0.05 macro event-F1 on held-out families, CI-separated: (a) primary = same-size degree+weight-preserving rewired MaleCNS mask on the real corpus; (b) secondary = wrong-animal corpus (C. elegans c302/OpenWorm sim, N=302, architecture accepts arbitrary NxN masks). Real <= either control (CI includes <= 0) triggers K2 |
| T-MODEL-4 | Real node identities >= permuted labels + 0.05 macro event-F1 |
| T-MODEL-5 | Ship gate: masked >= parameter-matched dense transformer - 0.01 (non-inferiority). Architecture claim (C5) upgrades only at masked >= dense + 0.02, CI-separated |
| T-MODEL-6 | Macro event-F1 >= BANC linear influence matrices + 0.05 on held-out activity. Margin <= +0.01 triggers K3 |
| T-MODEL-7 | Macro event-F1 >= flybrain Trace/Readout reservoir + 0.05 on the same tasks |
| T-MODEL-8 | All reported metrics under the silence-robust rules of 1.7; all-zero predictor <= majority baseline on every split (sanity invariant) |
| T-MODEL-9 | Single-upstream-neuron intervention: >= 99% of changed t+1 units are postsynaptic partners; same-time (t) changes = 0 beyond rel eps 1e-6 |
| T-MODEL-10 | Transformer vs GNN on identical splits reported head-to-head; if GNN wins by > 0.02 macro event-F1, ship the GNN and adjust the story |
| T-MODEL-11 | Real-weights corpus model >= shuffled-weights-slice model + 0.05 macro event-F1 on held-out activity; else claims scoped to topology only |

### T-SEX - sex conditioning

| ID | Pass criterion |
|---|---|
| T-SEX-1 | >= 80% of the frozen dimorphism set (section 4) reproduced |
| T-SEX-2 | Dimorphic divergence (1.9) exceeds the isomorphic-null 95th percentile on >= 50% of tested dimorphic circuits; direction-consistent across >= 3 seeds |
| T-SEX-3 | "Specimen-conditioned" ablation reported alongside "sex-conditioned" in every artifact (reporting requirement - no numeric bar, missing report = fail) |
| T-SEX-4 | Cross-sex type leaks = 0: male-specific types absent under female conditioning and vice versa. Exact |
| T-SEX-5 | Female-conditioned forward pass uses the BANC edge set: mask edge indices/counts differ from the male mask exactly on the swapped subgraph and match the BANC manifest (100%) |
| T-SEX-6 | >= 2/3 of checked isomorphic circuits agree between the female corpus and male-corpus/FlyWire-published findings; every failure documented verbatim in the model card. BANC has no published whole-CNS LIF validation - the asymmetry is stated, not hidden |

### T-REAL - real-data validation

| ID | Pass criterion |
|---|---|
| T-REAL-1 | DANDI 000951: >= 16/23 LC types with correct-direction silencing prediction (per 1.6), AND >= influence-matrix baseline + 2 types; per-type binomial tests BH-FDR q <= 0.10 across the 23-type panel |
| T-REAL-2 | Predicted-vs-recorded population dynamics through indicator-kernel convolution: Spearman >= 0.25 on aligned types (stretch - reported regardless of outcome) |
| T-REAL-3 | Song ethograms: >= 4/5 reference statistics within SongExplorer ground-truth ranges - pulse-bout IPI median in [30, 45] ms (reference ~35-40 ms), pulse/sine bout ratio, bout duration distribution, context-dependence direction (near/far switching) preserved; KS statistics reported per stat |

### T-SHIP - release integrity

| ID | Pass criterion |
|---|---|
| T-SHIP-1 | Fresh-venv `pip install flygpt` + minimal inference exits 0 on Windows AND Linux |
| T-SHIP-2 | 100% of EVIDENCED rows in CLAIMS.md link existing committed `results/` artifacts whose stated pass criterion holds |
| T-SHIP-3 | 0 GPL packages in the runtime dep tree; 0 FlyWire-derived files in shipped artifacts; THIRD_PARTY_NOTICES complete |
| T-SHIP-4 | Model-card claims are a subset of EVIDENCED rows (100%); limitation list present verbatim |
| T-SHIP-5 | Dataset card contains "simulated, not recorded" + full provenance chain + convention choices, verbatim |
| T-SHIP-6 | Every headline number reproduces via one documented command, landing inside its reported CI |
| T-SHIP-7 | Space completes its demo path with zero GPU and zero live-sim calls (canned rollouts + `flygpt-tiny`) |
| T-SHIP-8 | CI pipeline smoke test green on CPU and CUDA |
| T-SHIP-9 | 0 secrets in repo; notices complete; 0 GPL in runtime deps |
| T-SHIP-10 | Inference steps/sec vs sim steps/sec measured on the same GPU class and recorded; the speed claim (C18) requires ratio >= 10x |

---

## 6. Kill-criteria thresholds (plan.md section 4, as numbers)

| Kill | Trigger (numeric) | Consequence |
|---|---|---|
| K1 - lookup table | T-MODEL-1 held-out-family event-F1 margin over majority baseline <= +0.05 (95% CI) | Redesign or kill: the model memorized the simulator |
| K2 - wiring doesn't matter | T-MODEL-3: real minus best control <= 0.00 (CI includes <= 0) on the primary rewired-mask control | Architecture claim dead; model may ship without it |
| K3 - linear suffices | T-MODEL-6 margin over BANC linear influence matrices <= +0.01 | Ship corpus + embeddings; dynamics claims DOWNGRADED in CLAIMS.md |
| K4 - sex conditioning empty | T-SEX-1 reproduces < 50% of the frozen DS set, OR T-SEX-2 divergence inside the isomorphic null on > 50% of tested dimorphic circuits | Ship without the flagship sex claim |
| K5 - pilot degenerate | T-SIM-6 fails (degenerate fraction > 5% or sane-band violations unreviewed), OR T-SIM-7 < 8/10 spot-checks | No scale spend; fix the sim first. The pilot gate is the single most important gate in the project |

---

## Amendments

None. `eval_v1` ratified 2026-09-22, before any training run. Amended
thresholds - if ever needed - get a new `eval_vN` section appended below with
date and rationale; this section's numbers stand as committed.
