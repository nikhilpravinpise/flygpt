# FlyGPT - Implementation Plan

**Companion to**: `flygpt-deep-research.md` (the science audit) + `free-compute-research.md` (the compute audit). This file is the executable plan.
**Rule**: every test below must be capable of failing. A test that can't fail is documentation, not a test. Each test lists what it kills if it fails.

---

## 0. Repo layout

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
  modal_app.py              # eval battery / CI CUDA smoke / odd jobs (burst compute lives on Vultr+GCP)
  tests/                    # the real test suite (section 3)
  space/                    # Gradio "Probe a Fly Brain"
```

---

## 1. Phase plan with exit criteria

### Phase 0 - Ingest + graph build (laptop or Oracle/Codespaces, ~1 day; CPU dev work, no training)
- Scaffold artifacts created FIRST: `CLAIMS.md`, `EVAL.md`, `RISKS.md` exist from day one (EVAL thresholds finalized before Phase 3 training). RISKS.md seeds: novelty-perishable (field ships weekly), sim-fidelity ceiling, Eon preemption, BANC-sim unvalidated, corpus determinism, solo-maintainer dep (flybrain).
- **Reserve names early**: HF org `flygpt` + PyPI package at scaffold time - both verified free today; squatter risk is real in this hype cycle.
- Download canonical files (Janelia `gs://flyem-male-cns/v1.0/`, record sha256 in `provenance/malecns_v1.json`). NOT the HF mirror, NOT `malecns_09` (canonical only for the manifest).
- Build CSR graph: dedup edges, explicit autapse policy, per-edge NT sign + confidence, type/instance/hemisphere metadata, `>=5` synapse cutoff (record alternate cutoffs 3/10 for sensitivity slices).
- Edge verification is internal, not just vs published numbers: derive edge weights independently from `syn-partners` and reconcile against `connectome-weights` - published totals vary by convention, so the check is that OUR derivation pipeline reproduces the canonical file.
- **Exit**: `tests/test_graph.py` green; node count = canonical body-annotations row count (expected ~165K: Cell paper reports 166,700, neuPrint male-cns:v1.0 snapshot reports 164,838 - conventions/snapshots differ; exact figure recorded in `provenance/` manifest, not hardcoded); derived-vs-canonical weights reconciled.

### Phase 1 - Pilot gate (laptop or first Kaggle session, ~1-2 days)
- 1,000-trial pilot corpus: 10 stimulus families x 100 trials, batch sims. Runs on the laptop (sim gen, not training) or a free Kaggle GPU session - either is acceptable; nothing about the gate requires owning the hardware.
- Shiu spot-checks run in MaleCNS cell-type identities, not FlyWire body IDs - the companion taste paper gives the male GRN->feeding-MN map to check against.
- **GO only if ALL pass** (see T-SIM tests): non-degenerate activity stats, ~10 known-circuit spot-checks, dynamic range sane, throughput measured (replaces all estimates).
- Output: `pilot_report.json` committed. No pilot pass = no scale spend. This is the single most important gate in the project.

### Phase 2 - Corpus (burst GPU: Vultr/GCP; Kaggle drip as fallback)
- ~1M trials at dt=20ms across stimulus program library (incl. tonic-baseline sweeps + naturalistic envelopes + random babble). Sized at ~10-25 GPU-hours total on a single A100/A40-class card (measured at pilot) - one or two days on the burst credit, ~1-2 weeks on Kaggle quota if burst stock fails.
- **Benchmark-matched stimulus family**: trials replicating DANDI 000951's exact silencing conditions (all 23 LC types) so the perturbation benchmark has matched training distribution where appropriate - but the held-out eval stimuli stay disjoint.
- **Fine-dt courtship slice**: a dedicated ~50-100K-trial corpus slice at dt=5ms for the song fine-tune. A 20ms corpus physically cannot yield 5ms bins - resolves the dt contradiction. Record both dt values in provenance; never mix slices in one split.
- **Shuffled-weights control slice**: a small corpus slice generated on the same topology with shuffled weights - separates "the model learned the topology" from "the model learned the weights" (Shiu's own control, lifted to corpus level).
- Single backend (GPU kernel) for the whole corpus; per-shard provenance manifest; seeds derived as hash(trial config) - never hand-assigned.
- Storage: sparse event lists -> parquet shards; bit-packed where beneficial. Shards push to the HF dataset repo as generated (the canonical home - a cloud VM's disk is not durable storage); Drive as secondary ferry.
- Ops: budget alerts armed on every billed account BEFORE first launch; failed-trial dead-letter queue + retry policy; corpus version tag (`corpus_v1`) immutably recorded; instances killed at job end, not left running.
- **Exit**: coverage metric (fraction of ~11.7K types ever driven; neuPrint v1.0 reports 11,751 - exact count from manifest), health-metric distribution, `tests/test_corpus.py` green on sampled shards.

### Phase 3 - Model + controls (burst A100 headline run; Kaggle TPU/GPU drip for the control suite)
- ~100M-param dual-axis transformer, population-vector inputs, cross-time connectome mask (pre(t) -> post(t+1), transpose-checked), silence-robust loss.
- **Framework: JAX/Flax for the model and our own LIF kernel** - required for the Kaggle TPU drip (v3-8 doesn't run CUDA); PyTorch/XLA acceptable as fallback. flybrain stays the GPU reference/cross-check (T-SIM-2). This is the architecture-level consequence of the compute matrix - decide at scaffold, not at Phase 3.
- **Sizing run is a hard gate before the headline run**: measure tokens/examples-per-second, memory, and early convergence on a ~10% slice in a real session (~free on Kaggle). The "24-48h per run" figure is an unmeasured estimate - the schedule commits only after the measurement.
- **Neuron embeddings artifact trained here** (structure+NT+dimorphism objectives) - cheap, durable, ships early even if the dynamics model underwhelms.
- Control suite trained in parallel: dense baseline, degree+weight-preserving rewire, permuted-node labels, no-edge/direct-input, **GNN message-passing baseline** (research verdict O9: build both - transformer for the foundation story, GNN as ablation), flybrain reservoir Readout.
- Hybrid vocab: type-level tokens + neuron-level tokens for designated circuits (motor neurons, courtship circuit).
- Training ops: checkpoint-to-remote at intervals well under every provider's session cap; resume-from-checkpoint is the normal path, not a recovery path. Headline run on burst A100 (unattended VM); controls/seeds/scaling run as the Kaggle drip queue. **Scaling-curve eval** (train on 10%/50%/100% of corpus) - a data-scaling curve is stronger evidence than a single run.
- **All evals >= 3 seeds, all metrics with CIs, multiple-comparison correction (BH-FDR) across the circuit/perturbation panels.** An uncorrected significance claim across 23 LC types is meaningless.
- **Behavior-decoder spec written before any behavior-head training**: per-behavior readout populations (wing MNs for song, leg MNs for locomotion) grounded in MANC/MaleCNS motor annotations + published song-circuit literature - a defined mapping, not hand-picked neurons. This is the exact spot where Eon got criticized; our decoder spec is a document, not vibes.
- Eval cadence: run eval battery on intermediate checkpoints - catches late-stage overfitting to sim artifacts.
- **Exit**: `tests/test_evals.py` green; results table in `results/` feeds CLAIMS.md.

### Phase 4 - Sex conditioning (BANC)
- **BANC LIF sim must be built - this is a hidden workstream.** flybrain is MaleCNS-only; our `sim_backend.py` kernel runs BANC's edgelist. Non-trivial: BANC conventions differ (autapses excluded at build, `post_count` column semantics, v2/v3 detector snapshots).
- **BANC sim validation caveat**: Shiu validated his LIF against real data; NO published whole-CNS LIF validation exists for BANC. Mitigation: validate shared isomorphic circuits against the male corpus + published FlyWire findings; state the asymmetry in the model card. The female branch is inherently less validated - say so.
- BANC v888 graph via same loader (schema adapter; record v2/v3 snapshot choice + rationale).
- Cell-type alignment via BANC's `fafb_cell_type`/`manc_cell_type` columns (curated), NBLAST as backup.
- Female corpus subset (~200K trials) at **dt=20ms to match male corpus** - cross-sex comparison requires matched temporal resolution. Coverage target is shared/dimorphic circuits, not the whole female CNS.
- Vocab is the union of aligned types; BANC-only types masked under male conditioning just as male-specific types are masked under female.
- Female stimulus library via homologous sensory types; sex-appropriate behavior heads (no song head on female).
- **Exit**: dimorphism validation tests (T-SEX) green; honest confound wording in cards.

### Phase 5 - Real-data validation
- DANDI 000951 LC-silencing perturbation benchmark - **partial download only** (486.5GB total; pull just the silencing sessions via dandi CLI). Cell-type-to-recording alignment via the dandiset's own published annotations.
- SCAPE/LBM correlation where feasible - flag as stretch (alignment is heavy).
- Shiu replication: the 164-prediction protocol reimplemented (not copied - repo has no license).
- Behavior-head labels: song readout validated against SongExplorer ethogram statistics + Lillvis data; locomotion/grooming heads against published DN-function annotations.
- **Exit**: T-REAL tests green OR documented as negative results with claims correspondingly downgraded in CLAIMS.md - a validated-negative outcome still ships, but with honest cards.

### Phase 6 - Ship
- HF org `flygpt` (verified free); dataset repo: preview corpus; model repos: embeddings, base, courtship, `flygpt-tiny` distilled for the demo Space (browser/CPU budget).
- Space "Probe a Fly Brain": HF free tier no longer includes CPU Gradio/Docker Spaces. Two free paths: (a) **static Space** running `flygpt-tiny` in-browser (ONNX/WebGPU) + canned rollout cache; (b) **ZeroGPU Gradio Space** (free accounts get 2) - the documented trick: work outside the `@spaces.GPU` function burns zero quota, so a Gradio demo serving flygpt-tiny on CPU costs nothing while visitors' GPU calls use their own quota. Pick at build time; static is the robust default. Stimulus buttons mapped to real sensory populations; side-by-side generated vs real song ethogram.
- Model card MUST include the control-suite results table (real vs rewired vs wrong-animal) - the controls are the product.
- Clean-env reproduction CI (Windows + Linux), license audit pass, CLAIMS.md evidence check.
- **Exit**: all T-SHIP tests green; every artifact independently loadable in a fresh env.

---

## 2. Compute architecture (verified Sep 2026)

**Budget: $0 out of pocket + $50 Modal credit (user-provided). No local training. No grants. Full scope.**
The laptop is an orchestrator only: it launches jobs, monitors, collects artifacts. All sim/training runs remotely and unattended. Full provider audit lives in `docs/free-compute-research.md` - this section is the operational summary.

### The compute matrix

| Provider | Role | Verified status |
|---|---|---|
| **Vultr** $250-300 new-account credit / **30d expiry** | Burst: corpus gen + headline training | **Likely but unverified** - ToS says "select products only"; affiliates claim GPU covered. Verify GPU deploy at signup before counting; card for identity; regional stock varies |
| **GCP** $300 trial / 90d | Burst: headline A100 runs | GPU unlocks after paid-billing upgrade; trial credits still cover until spent; ~60-80h on-demand A100 (2-3x on spot); budget alert + GPU quota request |
| **Kaggle** | Drip: control suite, seeds, evals; corpus fallback | 30 GPU-h/wk (P100/2xT4) + 20 TPU-h/wk (v3-8); sessions cap ~9-12h; availability is dynamic |
| **Modal** | Eval battery, CI CUDA smoke (T-SHIP-8), odd jobs | $30/mo recurring + $50 user credit; hard spend cap $80 - never corpus/training |
| **Colab** free tier | Overflow only | Jio AI Pro promo is trial-classified - ZERO CCUs confirmed on the account; T4-when-available, ~12h cap |
| **Beam** $30/mo | Secondary recurring credit | metered incl. GPUs |
| **OVH** $200/30d, **Civo** $250/31d, **Thunder** $20 | Burst reserve / parallel burst | OVH GPU-eligible officially ("35h AI1-GPU" example) but auto-bills after credit; Civo unconfirmed; Thunder needs US-univ email - eligibility unverified |
| **Lightning AI** | One-time promo (~80h reported) | unverified until dashboard check |
| **HF public repos + Jio 5TB Drive** | Storage + ferry | HF = canonical artifact home; Drive = checkpoint transport |
| **Oracle** always-free ARM | CPU jobs: ingest, tokenize, CI | 2 OCPU/12GB post-halving |

**Dead ends (verified - do not retry)**: Linode $100 (GPU excluded), Azure all forms incl. student (GPU quota denied on trial subs), Oracle trial GPU (none), Nebius (trial suspended), DigitalOcean student credit (GPU excluded, program ended), TensorDock/Daytona/E2B/Novita/RunPod/Vast/Lambda/Shadeform (no usable GPU credit).

### Two clocks

- **Burst (days 1-7)**: ~$770-1070 credit stack, of which ~$520 is officially GPU-confirmed ≈ ~100-210 A100-equiv-h confirmed floor, ~170-450 if Vultr/Civo land. Pays for: entire corpus (~10-25 GPU-h on one card), headline training run, embeddings, first evals. VMs run 24/7 with the laptop off.
- **Drip (weeks 2+)**: the proof layer - control suite, multi-seed runs, scaling curve, eval battery - rides the weekly Kaggle quota + ~$60/mo serverless credits. Claims in CLAIMS.md upgrade only as evidence rows land in `results/`; v0.1 ships corpus + embeddings + first model with "controls pending" cards, never unvalidated claims.

### Session rules

- Checkpoint to remote (HF repo or Drive) at intervals far under each provider's session cap; resume-from-checkpoint is the normal path.
- Corpus tasks write immutable per-task shards; never append to shared files.
- Kill every instance at job end; budget alerts armed before first launch on every billed account; nothing runs that can silently bill past $0. **OVH explicitly auto-bills the card after credit ends** - teardown is mandatory there, not optional.
- **Expiry discipline**: Vultr/OVH/Civo credits die 30-31 days after activation (OVH's clock starts at first project creation) - activate only when the job is ready to launch, not at account setup.
- Every job entry point exists in `src/` + a thin session driver in `notebooks/` or `modal_app.py` - the same code path runs on Kaggle, Vultr, GCP, or Modal.

---

## 3. The test suite - real tests only

Naming: `T-AREA-N`. Every test states: what it checks / pass criterion / what fails if it dies.

### T-BUILD - graph construction (can't ship a wrong brain)

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-BUILD-1 | Node universe vs canonical | node set = body-annotations row count (expected ~165K: 166,700 per Cell paper / 164,838 per neuPrint v1.0 snapshot - record actual in manifest). NOTE: `connectome-weights` excludes segments with zero synapses - the graph will contain FEWER connected nodes. Correct invariant: all annotation rows exist as nodes; isolated ones retained as zero-degree nodes (not dropped), so model vocab covers them | Spurious fail on a correct build / silent neuron loss |
| T-BUILD-2 | Edge count vs convention | matches published count at declared cutoff; counts reported for 3/5/10 | Graph build |
| T-BUILD-3 | Autapse policy | count of self-loops matches manifest; policy applied | Silent graph-policy asymmetry vs BANC (BANC excludes ~2.1%) |
| T-BUILD-4 | NT sign consistency | every edge's sign follows documented rule; low-confidence edges flagged, not guessed | NT convention contamination |
| T-BUILD-5 | Checksum verify | sha256 of every upstream file matches `provenance/*.json` | Supply-chain corruption |
| T-BUILD-6 | Duplicate-edge dedup | no repeated (pre,post) pairs post-build | Weight inflation |
| T-BUILD-7 | Coverage of types | fraction of neurons with type labels reported; UNK handling verified | Silent vocab loss |

### T-SIM - simulator correctness (the corpus is only as good as the sim)

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-SIM-1 | Determinism | same seed+config+backend+GPU-class -> spike times identical or within documented FP tolerance (GPU scatter/reduction ops are NOT bitwise-deterministic across runs or GPU models - "bitwise" is only required on the CPU backend; GPU reproducibility is measured as spike-time agreement rate >= threshold set in EVAL.md) | Reproducibility claim |
| T-SIM-2 | Backend consistency | CPU vs GPU on same trial: spike agreement within documented tolerance (or single-backend+single-GPU-class corpus enforced per version) | Mixed-distribution corpus |
| T-SIM-3 | Causality unit test | inject single-neuron pulse; only postsynaptic partners' rates change at t+1, nothing at t (same-time) | Mask direction bug |
| T-SIM-4 | Sign sanity | stimulating known excitatory sensory neuron raises downstream rates; inhibitory lowers | Sign inversion bug |
| T-SIM-5 | Silence handling | unstimulated network with zero baseline stays silent; with tonic drive shows activity | Basal-rate pathology unnoticed |
| T-SIM-6 | Pilot health metrics | per-trial %active, rate distribution, stationarity within sane ranges; degenerate trials flagged | Whole corpus garbage |
| T-SIM-7 | **Shiu spot-checks** | ~10 published circuit responses reproduced using MaleCNS cell-type identities (sugar GRN -> proboscis-feeding MNs per the companion taste paper; JON -> grooming; LC4/LPLC2 -> DNp01 escape) | Sim doesn't reproduce known biology -> corpus worthless |
| T-SIM-8 | Physiology-range sanity | corpus activity statistics (rate distributions, spontaneous levels) fall within ranges published physiology supports - flagged outliers get reviewed, not silently shipped | Corpus diverges from biology in ways nobody noticed |

### T-CORPUS - dataset integrity

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-CORPUS-1 | Schema | every shard validates against declared schema (trial_id, events, stimulus program, sim config, seed) | Undocumented corpus |
| T-CORPUS-2 | Provenance | every shard's manifest resolves to real source hashes + code SHA | Unverifiable data |
| T-CORPUS-3 | Coverage floor | >=X% of cell types driven at least once across corpus (X set in EVAL.md) | Model only learns a corner of the brain |
| T-CORPUS-4 | Split hygiene | zero overlap of stimulus families / held-out cell types / regions between train and test splits | Leakage -> inflated scores |
| T-CORPUS-5 | Convention slices | threshold/convention-variant shards correctly labeled and disjoint | Convention/biology conflation |
| T-CORPUS-6 | **Regeneration determinism** | CI job regenerates a small shard from its manifest + seed on the same backend+GPU class -> identical or within the T-SIM-1 tolerance (bitwise only guaranteed for CPU-backend shards) | Corpus unreproducible |
| T-CORPUS-7 | dt integrity | corpus slices never mix dt values inside a split; dt recorded per shard | Temporal-resolution contamination |

### T-TOK - tokenizer integrity (the silent-corruption risk)

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-TOK-1 | Round-trip | events -> tokens -> vectors -> tokens reproduces input within documented precision | Tokenizer data loss |
| T-TOK-2 | No collisions | every cell type AND every designated neuron-level instance maps to exactly one token; no two share an ID (hybrid vocab: type + instance slots must not overlap) | Vocab corruption |
| T-TOK-3 | UNK handling | untyped neurons route to UNK, not silently dropped or merged into a real type | Silent data loss |
| T-TOK-4 | Bin-edge correctness | event at bin boundary lands in the correct bin (define floor convention, test it) | Off-by-one temporal shift |
| T-TOK-5 | Vocab-vs-manifest | vocab size = manifest type count + reserved slots (UNK, PAD, sex, stimulus, behavior) | Vocab drift |

### T-STIM - stimulus correctness (a wrong stimulus mapping falsifies every label in the corpus)

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-STIM-1 | Sensory grounding | every stimulus program resolves to annotated sensory cell types that exist in the graph (GRN/JON/LC/ORN families verified against body-annotations) | Mislabeled corpus |
| T-STIM-2 | Laterality | left/right stimulus flags drive the correct hemisphere's neurons | Crossed wiring |
| T-STIM-3 | Coverage tracker accuracy | tracker output matches independent recount from shard metadata | Fake coverage numbers |
| T-STIM-4 | Envelope sanity | naturalistic envelopes bounded, non-negative, sampled at stated dt | Corrupt stimulus encoding |
| T-STIM-5 | Silencing semantics | "silenced" populations contribute zero spikes AND zero outgoing synaptic effect during the silenced window | Perturbation benchmark invalid |

### T-EMB - embedding artifact eval (or it's just a pretty t-SNE)

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-EMB-1 | Cell-type recovery | embeddings cluster/recover held-out cell types above baseline | Embedding artifact has no value |
| T-EMB-2 | NT-class prediction | linear probe predicts neurotransmitter class above majority | Embeddings carry no biology |
| T-EMB-3 | Dimorphism classification | probe separates iso/dimorphic/male-specific types per Cell labels | Conditioning signal absent |
| T-EMB-4 | NBLAST agreement | embedding nearest-neighbors agree with NBLAST morphological neighbors above chance | Embeddings diverge from morphology |

### T-MODEL - the claims that matter

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-MODEL-1 | **Held-out stimulus families** | next-step prediction on never-seen stimulus *families* beats majority/baseline by margin in EVAL.md | "Memorized the simulator" - model is a lookup table |
| T-MODEL-2 | **Held-out cell types** | predicts activity of types never stimulated during training above baseline | "No connectome generalization" - core claim dies |
| T-MODEL-3 | **Wrong-connectome control (digital sphinx)** | model trained on real graph must beat identical model on degree+weight-matched rewired graph AND on wrong-animal graph. Two distinct controls, don't conflate: (a) rewired/wrong MASK on real corpus - tests that the correct wiring matters to prediction; (b) wrong-animal CORPUS - needs a C. elegans dynamics corpus (c302/OpenWorm sim exists) and is a separate workstream. C. elegans is ~302 neurons - architecture must accept arbitrary NxN masks. Primary = same-size rewired MaleCNS mask on real corpus; wrong-animal corpus is the secondary dramatic demo, sized/scoped in EVAL.md | "Wiring doesn't matter" - architecture story dies |
| T-MODEL-4 | Permuted-labels control | real node identities beat permuted ones | Graph position encoding does all the work |
| T-MODEL-5 | Dense baseline | masked model >= parameter-matched dense transformer | Mask adds nothing (model survives, architecture claim dies) |
| T-MODEL-6 | **BANC linear baseline** | beats precomputed linear influence matrices on held-out activity prediction | "Learned model adds nothing over linear" - weak result |
| T-MODEL-7 | Reservoir baseline | beats flybrain Trace/Readout on same tasks | "A dumb reservoir suffices" |
| T-MODEL-8 | Silence metric guard | reported metrics on active-neuron subsets + event F1; model can't win by predicting all-zero | Vacuous metrics |
| T-MODEL-9 | Mask direction | single-upstream-neuron intervention changes only downstream tokens at t+1 | Causality bug in shipped model |
| T-MODEL-10 | GNN ablation | transformer vs GNN baseline reported head-to-head on identical splits - if GNN wins, ship the GNN and adjust the story | "Transformer was branding, not engineering" |
| T-MODEL-11 | Shuffled-weights corpus control | model trained on real-weights corpus beats model trained on shuffled-weights slice on held-out activity | "Topology carried it, weights didn't matter" - claim scoped accordingly |

### T-SEX - sex conditioning

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-SEX-1 | Dimorphism sanity | female-conditioned run reproduces known structural dimorphism effects from Cell paper (the set pre-registered in EVAL.md) | Conditioning does nothing |
| T-SEX-2 | Same-input divergence | identical homologous stimulus -> measurably divergent dynamics in dimorphic circuits, stable across seeds | Flagship claim evaporates |
| T-SEX-3 | Confound honesty | "specimen-conditioned" ablation reported alongside "sex-conditioned" | Overclaiming |
| T-SEX-4 | Masking correctness | male-specific types absent under female conditioning and vice versa | Conditioning leak |
| T-SEX-5 | **Graph actually swaps** | female-conditioned forward pass uses the BANC edge set - edge indices/counts differ from male mask exactly where expected and match BANC manifest | "Sex conditioning" that's just a soft label |
| T-SEX-6 | Female-branch honesty | female corpus validated on shared circuits vs male + FlyWire-published findings (BANC has no published LIF validation) | Unvalidated female sim undermines flagship claim |

### T-REAL - real-data validation

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-REAL-1 | **DANDI perturbation benchmark** | predicts behavioral/dynamic effect of silencing each of the 23 LC types above baseline | "Validated" claim dies |
| T-REAL-2 | Calcium comparison | predicted-vs-recorded population dynamics, rank/order agreement through indicator-kernel convolution | No real-data grounding |
| T-REAL-3 | Song ethograms | generated pulse/sine bout statistics match SongExplorer ground-truth distributions + context-dependence | Song head is decoration |

### T-SHIP - release integrity

| ID | Test | Pass | Kills |
|---|---|---|---|
| T-SHIP-1 | Clean-env install | fresh venv `pip install flygpt` + minimal inference, Windows AND Linux | Broken release |
| T-SHIP-2 | CLAIMS evidence check | CI: every claim in CLAIMS.md links a `results/` artifact that exists and passes | Shipping unsupported claims |
| T-SHIP-3 | License audit | no GPL in runtime deps; FlyWire-derived files absent from artifacts; THIRD_PARTY_NOTICES complete | Legal exposure |
| T-SHIP-4 | Model card claims | card contains only claims with evidence rows; limitation list present verbatim | Card overstating |
| T-SHIP-5 | Dataset card honesty | "simulated, not recorded" + provenance chain + convention choices stated | Misrepresented data |
| T-SHIP-6 | Reproduction | one-command repro of each headline number in CI or documented env | Non-reproducible results |
| T-SHIP-7 | Space offline | demo runs from canned rollouts + tiny model with zero GPU/live-sim dependency | Space dead on arrival |
| T-SHIP-8 | Device-agnostic | CI runs pipeline smoke test on CPU and CUDA | Windows/local path broken |
| T-SHIP-9 | Secrets/license scan | no API keys in repo; THIRD_PARTY_NOTICES complete; no GPL in runtime dep tree | Legal/security exposure |
| T-SHIP-10 | **Speed benchmark** | measured inference steps/sec vs sim steps/sec recorded - the "faster than the LIF" claim needs a number, not a vibe | Unsupported speed claim |

---

## 4. Kill conditions (from the audit - restated as gates)

1. No held-out stimulus-family generalization -> redesign or kill (lookup table).
2. Real graph <= rewired/wrong-animal graph -> architecture claim dead (model may still ship without the claim).
3. Can't beat BANC linear baseline -> "learned adds nothing"; ship corpus+embeddings, downgrade claims.
4. Sex conditioning shows nothing beyond Cell structure paper -> ship without flagship claim.
5. Pilot degenerate -> no scale spend, fix sim first.

---

## 5. EVAL.md required contents (pre-registration spec)

Committed before any training run. "Beats baseline" without a number is not a criterion.

- **Numeric thresholds** for every test: e.g., "held-out cell-type event-F1 >= dense baseline + X" - X written down now, not after seeing results.
- **Metric definitions**: event-F1 binarization rule, rate-correlation method, Shiu-protocol agreement metric - each defined precisely.
- **Seed policy**: >=3 seeds per eval, mean + CI reported.
- **Multiple-comparison correction**: BH-FDR across all multi-circuit/multi-perturbation panels.
- **Pre-registered dimorphism set**: the specific known-dimorphism effects T-SEX-1 will check, listed in advance.
- **Split definitions**: exact stimulus families / cell types / regions held out - enumerated, not described.
- **Kill-criteria thresholds**: the numbers that trigger each Section-4 kill condition.

---

## 6. Hard rules

- Apache-2.0 code; CC-BY-4.0 datasets/weights; FlyWire-derived anything is NC-flagged or excluded.
- natverse/bancr/fafbseg (GPL) live in `scripts/` only, never runtime deps.
- Canonical Janelia files for manifests; sha256 everything.
- Single sim backend per corpus version; seeds + configs recorded per shard. All shards of one corpus version generated on the same backend AND GPU class - cross-GPU FP drift either prevented or measured and documented (see T-SIM-1/2).
- EVAL.md + CLAIMS.md committed BEFORE any training run.
- No "uploaded fly", no "digital fly", no "first" without timestamp + recheck.
- Every script runs locally AND on at least one remote provider from the compute matrix (Kaggle notebook driver + Modal entry point at minimum).
- Windows dev path documented (HF_HOME short path / WSL2 option).
- Nothing bills past $0: every billed account gets a budget alert before first job; Modal hard-capped at $80 ($50 credit + $30/mo).

---

## 7. Open items to resolve at build time (flagged, not assumed)

- flybrain's exact LIF params + baked graph conventions -> audit `flybrain build` source first; diverge->own kernel.
- ~~flybrain LICENSE~~ -> **MIT, verified (audit §13.2)** - permissive, safe as runtime dep. Still pin `flybrain==0.1.0` (v0.1.0, single maintainer) + keep the `sim_backend.py` adapter seam.
- **CAVE token** (required for BANC queries via bancr/fancr) + **neuPrint token** (annotation pulls) - free registrations, needed by Phase 0/4 respectively.
- DANDI 000951 exact license string -> read dandiset metadata at download.
- Neuroformer/chaobrain/SongExplorer/erojasoficial licenses -> check before any dependency.
- FANC exact count -> verify if ever cited.
- BANC v2 vs v3 snapshot choice -> record decision + rationale.
- Wrong-animal graph source + license for T-MODEL-3 (C. elegans connectome - WormWiring/OpenWorm data; verify redistribution terms before shipping it in a control artifact).
- Account setup checklist (Phase 0): Vultr signup + **GPU deploy test on credit before counting it** (30d expiry - activate only when ready); GCP trial activation + paid-billing upgrade + GPU quota request; OVH trial - **activate at first-project creation only when corpus is ready** (clock starts then; auto-bills after credit); Thunder student eligibility check (US-univ email); Civo GPU attempt; Modal $50 credit applied + $30/mo confirmed + spend cap armed; Beam/Lightning dashboard balances verified; Kaggle phone-verified (required for accelerator quota); HF org `flygpt` created; Drive checkpoint folder on the Jio-linked account.
- Female homolog coverage: what fraction of male stimulus programs have BANC-homologous sensory inputs - measured and reported, not assumed.
- Pilot-measured throughput -> replaces all GPU-hour estimates; feeds the Phase 3 sizing-run gate.
