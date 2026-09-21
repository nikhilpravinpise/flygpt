# FlyGPT / MaleCNS-LM - Deep Research Report

Date: 2026-09-15
Scope: Male fruit fly CNS connectome (MaleCNS v1.0), existing projects, novelty audit, adversarial review, build spec.

---

## 0. TL;DR

Several things we assumed were unbuilt are already built - including a MaleCNS+LLM paper from 4 days ago with a *null result*, and a funded startup (Eon Systems, with the Nature paper's own first author as Head of Engineering) that already embodied a fly connectome in a physics body. After the adversarial pass, exactly three defensible novelty claims survive:

1. **First trained generative/foundation model of a real, complete animal CNS's dynamics** (everyone else ships hand-wired sims or RL controllers - nobody has trained a large model on a real connectome's activity).
2. **First sex-conditioned neural dynamics model** - possible only since June 2026, when the male/female connectome pair became available. Nothing like it exists anywhere.
3. **First brain→VNC→behavior dynamics model** - MaleCNS is the only dataset with the full sensor-to-muscle chain; Eon's embodied fly is brain-only (they admit it) and FlyGM is an RL policy, not a brain.

Recommended framing: **"the first foundation model of a real nervous system"** - NOT "a fly that thinks" and NOT "fly brain inside ChatGPT" (already tried, doesn't work).

---

## 1. The complete landscape audit

### 1.1 Connectome datasets (the raw material)

| Dataset | Scope | Neurons | Synapses | Status |
|---|---|---|---|---|
| **MaleCNS v1.0** (Janelia FlyEM + Cambridge + Google, Cell 2026, Berg et al.) | Male brain + optic lobes + VNC | ~165K (166,700 Cell paper; **164,838 neuPrint v1.0 Sep-2026 snapshot**) | **~171.9M synapses (neuPrint)**; ~25.6M aggregated connections | Released 2026-06-08, CC-BY. Annotated w/ fru/dsx, **11,751 types** (neuPrint; 11,710 in Cell paper) |
| **BANC** (Harvard Lee lab + Zetta + FlyWire, Nature 2026, Bates/Phelps/Kim et al.) | Female brain + VNC | 155,916 proofread (Nature/v626 convention); 188,259 total at v888 | ~169M (v2, size>=5) / ~199M (v3, size>=10); ~11.5-13.5M connections | The correct female counterpart. Includes all-to-all "influence" matrices + built-in cross-dataset cell-type columns (fafb_cell_type, manc_cell_type). Autapses excluded at build (~2.1%). Missing lamina (~9,390 neurons) AND ocellar ganglion |
| **FlyWire/FAFB** (Princeton + Janelia, Nature 2024, Dorkenwald et al.) | Female brain only | 139,255 | ~50M | Flagship connectome; basis of all prior simulations |
| Hemibrain | Partial female brain | ~25K | - | Older |
| MANC | Male VNC only | ~23K | ~10M pre-sites | Subsumed by MaleCNS |
| FANC | Female VNC only | ~15K (unverified; subsumed by BANC) | - | Subsumed by BANC |
| Male/female comparison (Cell paper) | - | - | - | 8,069 isomorphic / 138 dimorphic / 289 male-specific / 71 female-specific types |

Sex-dimorphism facts (Cell 2026): sex-specific/dimorphic neurons concentrate in higher brain centers; sensory/motor periphery largely isomorphic; male-specific connections form "hotspots"; dimorphic neurons reroute information across sexes. Includes the "love spot" LoVP92 courtship neuron.

### 1.2 Simulations already built on connectomes

| Project | What it does | Status vs. our idea |
|---|---|---|
| **Shiu et al. Nature 2024** | LIF model of whole female brain (FlyWire). Predicted feeding/grooming circuits; validated optogenetically. Shuffled-weight controls collapse predictions → connectivity weights matter | The proven recipe; female brain only |
| **Eon Systems** (SF startup; Philip Shiu is Head of Engineering; Michael Andregg CEO; Alex Wissner-Gross advisor) | Embodied fly: FlyWire LIF brain + NeuroMechFly MuJoCo body, closed loop every 15ms. Walks, grooms, feeds, flees. Claims "91% accuracy" vs biological responses. Stated limitation: no VNC/motor neurons - "we plan to scan both brain and body" | **The elephant in the room.** Funded, staffed, 6 months ahead on embodiment - but brain-only, emulation not trained-model |
| **flybrain (PyPI)** | `pip install flybrain` - MaleCNS v1.0 as runnable LIF net, CPU/GPU, batch sims, reservoir readouts (Trace/run/Readout), visual encoder driving VPNs | "First MaleCNS simulation" claim is DEAD |
| **theconsciousness.ai Connectome Atlas** | LIF pass over MaleCNS v1.0; recomputed stats (165,122 traced neurons, 311.8M contacts, 151.9M connections) | Another independent MaleCNS sim |
| **chaobrain** | JAX reproduction of Shiu model, GPU | Template for corpus generation |
| **fwmc** | C++ Izhikevich sim + closed-loop neural twinning | Fringe |
| **Loihi 2 port** (arXiv 2508.16792) | Full FlyWire on 12 Intel neuromorphic chips, orders-of-magnitude faster | Proves cheap compute |
| **NeuroMechFly v2 / FlyGym** | Physics digital twin, vision via FlyVision, RL controllers, fly-following | Embodiment platform |
| **Vaxenburg et al. Nature 2025** | Whole-body MuJoCo fly, RL walking/flight | Body model |

### 1.3 Learned/trained models on connectome data

| Project | What it does | Status vs. our idea |
|---|---|---|
| **Lappalainen et al. Nature 2024 (FlyVision)** | Connectome-constrained recurrent net; learned neuron/synapse params via motion-detection task; matched 26 experimental studies | Structure fixed + params learned - closest recipe to ours, subsystem only |
| **FlyVis-GNN** (Saalfeld lab) | GNN trained on *simulated* visual-system activity recovers effective connectivity, neuron types, nonlinearities; predicts causal effects of ablations | Proves dynamics→mechanism inference works; 13.7K neurons only |
| **FlyGM** (arXiv 2602.17997) | Whole-brain connectome instantiated as message-passing GNN controller; RL-trained for locomotion of biomechanical fly; beats rewired/random/MLP baselines | Connectome-as-controller is DONE |
| **FLYNN** (arXiv 2607.00025) | Connectome-topology RNN for robot navigation; superior OOD/sensory-loss robustness vs hand-crafted nets | Connectome-as-RL-architecture is DONE |
| **"Flies Are All You Need"** (artificialscientific.com, 2026-09-11) | MaleCNS v1.0 graph as token-driven reservoir inside a frozen 1.17B LLM; trained 278K-param readout. Result: −0.0222 nats vs frozen model, but a parameter-matched direct-input readout was *slightly better*. Honest null result | "MaleCNS + LLM" is DONE - and doesn't work for language. Key negative evidence shaping our design |
| **ConnectomeGPT-Worm** (HF) | C. elegans 302-neuron connectome as fixed layer inside a GPT; WikiText-103/MMLU evals | Biological layer inside LM - done at worm scale |
| **TuKoResearch ConnectomeGPT100M** | "Model connectomes": learned sparse wiring of the LM itself, not biology | Name collision only |
| **NTAC** | Cell typing from connectivity alone, 95% acc, minutes on laptop | Prior art for structure→type |
| **CLOVE/Node2vec hyperbolic embeddings** | FlyWire embedded in hyperbolic space; quality saturates ~d=64 | Neuron embeddings exist as analysis, not as released model artifact |

### 1.4 Neural foundation models (the "GPT for brains" lineage)

| Model | Data | Notes |
|---|---|---|
| **Neuroformer** (ICLR 2024) | Mouse spikes | Neuron-as-token GPT; on simulated data it *inferred underlying circuit connectivity including direction*. The architectural template |
| BrainLM (ICLR 2024) | 6,700h fMRI | Masked prediction; perturbation prompting |
| NDT/NDT2, POYO+, MtM, NEDS, NeuroPaint, CalM | Spikes/calcium | Decoding, forecasting |
| OmniMouse (2026) | 150B neural tokens, 3.1M neurons, 73 mice | Brain models are data-limited, not parameter-limited - inverts LLM scaling |
| Zebrafish SBM | Whole-brain calcium | Dual-axis attention (neuron axis + time axis); brain-state→behavior readout |

### 1.5 LLM-over-connectome tools (query/interpretation)

- **Codex "Exodiac"** - AI assistant on FlyWire Codex (Gemini Flash): natural-language query help, cell-type summaries. Exists.
- **LLantia** (arXiv 2608.00059) - LLM-driven circuit function inference over the fly connectome, literature-distilled prompts per cell type. Exists.
- **GPTCelltype** - GPT-4 cell-type annotation for scRNA-seq.

### 1.6 Real functional data for validation (this is a goldmine nobody's using)

- **CRCNS fly-1**: ~965 GB whole-brain light-field calcium/voltage recordings, up to 200Hz, behaving flies
- **SCAPE microscopy** (NatComms 2023): whole-brain volumetric imaging during spontaneous behaviors - locomotion-correlated and behavior-independent structure
- **LBM** (NatComms 2026): light-beads microscopy, 28-60 volumes/s brain-wide in behaving flies - resolves responses to *single courtship song pulses*
- **Locomotion maps** (Curr Biol 2023): whole-brain walking dynamics registered to the connectome
- **DANDI 000951** (Cowley et al. 2024): 459 courtship sessions, 23 LC types silenced, SLEAP tracks + song labels + 2p imaging
- **Lillvis et al. 2023** (Janelia): courtship song datasets w/ optogenetics, SongExplorer ethograms
- **BANC influence matrices**: all-to-all linear-dynamical influence scores, precomputed - a free baseline "dynamics model" to beat

### 1.7 Known scientific objections (the grilling)

| # | Objection | Evidence | Severity |
|---|---|---|---|
| 1 | "A connectome is not enough" - no weights (only synapse counts), no neuromodulators, no gap junctions, no neuropeptides, no receptor kinetics, no neuronal dynamics parameters | Scheffer & Meinertzhagen 2021; Marder; neuropeptidergic connectome work shows extrasynaptic signaling is pervasive | HIGH |
| 2 | Shiu model already fails on specific neurons - Usnea/Phantom not predicted (they're likely neuropeptidergic; Amontillado knockdown phenocopies silencing) | Shiu et al. Nature 2024 extended data | HIGH - your model inherits these failure modes |
| 3 | Basal firing = 0 assumption → inhibitory neurons can't act without drive; real brains have tonic activity | Shiu et al. methods | MED - fixable w/ tonic baseline params |
| 4 | Functional maps poorly predicted by synaptic connectome (extrasynaptic transmission) | Nature 2023 neuropeptidergic C. elegans work ("it's not all about the synapses") | HIGH |
| 5 | Training a transformer on a simulator = distillation of your own assumptions; model can't exceed the LIF's fidelity | logical | MED - mitigated by real-data validation |
| 6 | Male vs female comparison confounded: different specimens, different pipelines (MaleCNS vs BANC/FlyWire), single-animal each | - | MED - frame as "specimen-conditioned" too |
| 7 | "Fly inside an LLM" shown not to help language (Flies Are All You Need null result) | 2026-09-11 paper | Settles a sub-question; our target is dynamics not language |
| 8 | Single specimen = no inter-individual variability; "the" male brain is one fly | - | MED |
| 9 | Scale: is a transformer even needed if dynamics are low-dimensional? SCAPE data suggests much activity is behavior-correlated + high-dim residual | NatComms 2023 | LOW-MED - actually argues a learned model can capture what LIF can't |
| 10 | Eon Systems is 6+ months ahead, funded, and plans to scan body+VNC | eon.systems | HIGH for embodiment claims; LOW for trained-model claims (they do emulation, not learned models) |

---

## 2. What survives - sharpened novelty claims

After the adversarial pass, the defensible claims are:

**A. First trained generative model of a complete real connectome's dynamics.**
Everything existing is either a hand-wired sim (flybrain, consciousness.ai, Eon, Shiu), an RL policy on connectome topology (FlyGM, FLYNN), or a foundation model on *recorded* neural data (Neuroformer, OmniMouse, SBM). Nobody has trained a foundation-scale autoregressive model where the training signal is generated by a real, complete, proofread wiring diagram. FlyVis-GNN is the closest precedent and it's a 13.7K-neuron subsystem.

**B. First sex-conditioned neural dynamics model.**
The male/female connectome pair exists only since June 2026. The Cell paper compared structure; the bioRxiv VNC paper aligned topology; nobody has built a single model that generates male-vs-female dynamics by conditioning. This is the most scientifically interesting and least crowded claim.

**C. First brain→VNC→motor-program dynamics model.**
MaleCNS is the only dataset containing the full chain: sensory neurons → brain interneurons → descending neurons → VNC motor neurons. Eon's fly is brain-only (they state it). flybrain produces spikes/readouts, not decoded behavior programs. The claim: "first model where you stimulate a sensory neuron and read out a motor program."

**D (supporting). First released neuron-embedding model for the fly CNS.**
CLOVE/Node2vec embeddings exist as paper analyses; no released embedding artifact conditioned on type/NT/dimorphism exists on HF.

### What NOT to claim

- First MaleCNS simulation - taken (flybrain, consciousness.ai)
- Fly connectome + LLM - taken, and negative result published
- Connectome-as-controller - taken (FlyGM, FLYNN)
- Embodied fly - taken (Eon, NeuroMechFly)
- Ask-the-connectome chat - taken (Codex Exodiac, LLantia)

---

## 3. Refined proposal - "FlyGPT" (final spec)

**The first connectome-constrained foundation model: an autoregressive transformer over the male fly's complete nervous system, sex-conditioned against the female connectome, with a motor-program output head.**

### 3.1 Tokenization (updated per audit O1/O4 - the event-token approach is dead at this scale)

- **Primary representation**: population vectors per time bin - an ~11.7K-dim (type-level) rate/activity vector per step, modeled autoregressively over time (OmniMouse/SBM-style patches), NOT per-event tokens (10^4-10^6 events/trial blows up sequence length). The corpus is *stored* as sparse events; the model *reads* binned vectors.
- **Neuron vocab**: ~165K instances is unwieldy; **~11.7K cell types** is a clean GPT-2-scale vocab. Type-level for v1, instance-level as v2. Must handle untyped neurons (UNK token) - not all ~165K neurons carry a type label; report the untyped fraction.
- **Time bins**: 20ms default (matches validation observables - calcium imaging ~100-500ms); the model is a *rate-dynamics* model, not a spike model. Song head may use 5-10ms bins (pulse-song IPI ~35-40ms, individual pulses ~5-10ms).
- **Modality/stimulus tokens**: sensory input channels (visual loom, odor classes, gustatory sugar/bitter/water, mechanosensory JON, female-silhouette courtship cues) as context tokens.
- **Behavior tokens**: decoded from descending/motor neuron populations → locomotion, grooming, proboscis extension, courtship song (pulse/sine), escape.
- **Sex token**: conditions the model on male vs female wiring - AND on female-side stimulus inputs (see 14.4) and sex-appropriate behavior vocab (females don't sing; the song head is male-only).

### 3.2 Architecture

- Dual-axis transformer (SBM/CalM lineage): sparse attention across the neuron axis **masked by the actual MaleCNS adjacency matrix** (connectome-as-attention-prior - the wiring is in the architecture, not just the data); causal attention along time.
- **Sex conditioning**: shared parameters on the 8,069 isomorphic types; dimorphic/sex-specific types get sex-specific attention masks (the graph delta IS the conditioning mechanism - more principled than a soft embedding).
- ~50-200M params. This is a wiring-prior project, not a scale project.
- Optional GNN message-passing variant (FlyGM-style) as baseline.

### 3.3 Training corpus - the brain writes its own data

No whole-brain fly recordings exist, so:

1. Port Shiu LIF recipe to MaleCNS (JAX; ~25.6M connection pairs - the flybrain package proves it runs on commodity hardware).
2. Massive rollout generation: randomized stimulus programs (odor pulses, looming, sugar/bitter, mechanosensory, courtship cues, optogenetic-style perturbations, tonic-baseline variations to fix the basal-rate problem).
3. Emit (cell_type, time) spike streams → the pretraining corpus.
4. Objectives: masked-neuron reconstruction (spatial BERT), next-timestep autoregression (temporal GPT), stimulus→behavior prediction, held-out-edge link prediction.

### 3.4 Validation against real data (what makes this science, not a demo)

- Reproduce Shiu's optogenetic validations (sugar→MN9, JON→grooming) in the *transformer*, not just the LIF.
- **DANDI 000951**: does the model predict the behavioral effect of silencing each of the 23 LC types? This is a real held-out perturbation benchmark.
- **LBM/SCAPE recordings**: correlation of predicted vs recorded population dynamics during behavior.
- **BANC influence matrix**: the linear-dynamical influence scores are a free baseline - beat them.
- Song: generate ethograms; compare pulse/sine bout statistics vs SongExplorer ground truth and context-dependence (near/far female distance switching, per Nature 2023 song-sequencing paper).

### 3.5 The sex-conditioned experiment (the flagship result)

- Train on male corpus; the isomorphic types (8,069) are shared.
- Condition "female": swap in BANC's connectivity for the dimorphic/female-specific types.
- Testable predictions: same sensory input → divergent downstream dynamics in higher centers. Does the model reproduce the structural finding that dimorphism propagates brain-wide via dimorphic connectivity? Do "antagonistic circuit switches" emerge as divergent attractor states?
- This is a **modeling result the Cell paper could not produce** - they have structure, not dynamics.

### 3.6 HF artifacts

| Repo | Content |
|---|---|
| `flygpt/malecns-spike-corpus` | Simulated dynamics dataset under labeled stimulus programs (the corpus) |
| `flygpt/fly-neuron-embeddings` | Pretrained per-type embeddings (structure+NT+dimorphism) - the broadly useful artifact |
| `flygpt/flygpt-base` | Pretrained dynamics model |
| `flygpt/flygpt-courtship` | Fine-tune w/ behavior+song head |
| Space "Probe a Fly Brain" | Gradio: stimulus picker → activity propagation visualization on real wiring → generated behavior/song ethogram |

---

## 4. The grilling - honest answers

**"Isn't this just distilling a broken simulator?"**
Partially. The LIF is crude (objections 1-4 above). But: (a) Shiu proved it predicts real circuit responses well enough to generate testable hypotheses; (b) the transformer is a *surrogate with a learning budget* - fine-tuned on real recordings it can correct LIF biases; (c) FlyVis-GNN showed learned models recover mechanistic truth better than the simulator they're trained on. Frame the LIF corpus as pretraining, real data as the corrective.

**"Eon Systems will crush this."**
Eon does emulation (hand-wired LIF + body), not trained foundation models. Their published method is Shiu 2024 + NeuroMechFly. They're scanning toward bigger brains; a MaleCNS-trained model is orthogonal - arguably complementary (a distilled transformer is ~1000x faster than LIF for inference, which is exactly what embodied closed-loop needs).

**"The sex-conditioned model is confounded by specimen differences."**
True - male and female are different individuals from different pipelines. Mitigation: report it honestly, use the Cell paper's own isomorphic/dimorphic classification as ground truth for the sex split, and treat "specimen" as an explicit nuisance variable. The scientific framing survives: *cross-sex comparison of dynamics conditioned on a documented dimorphism map*.

**"Why a transformer at all? The LIF already runs cheap."**
Three reasons: (1) inference speed for closed-loop embodiment; (2) a learned model can be *corrected by data* - the LIF can't be (its params aren't learned); (3) the foundation-model frame lets you do masked inference - predict unrecorded neurons from recorded ones, which is a real neuroscience use case (cf. NeuroPaint).

**"Isn't FlyGM already 'the connectome does control'?"**
FlyGM is an RL policy that borrows topology - it doesn't model neural dynamics, can't predict spike trains, can't be validated against recordings. Different object.

---

## 5. Compute & feasibility (updated with verified numbers)

- Corpus generation: flybrain runs MaleCNS LIF at **1.4ms/step on a 4060-class GPU** (our 5050 Laptop should be comparable). At default dt=20ms a 1s trial = 50 steps ≈ 70ms wall → ~14 trials/s unbatched, ~1.2M trials/GPU-day; `batch=N` multiplies this. At spike-resolved dt=1ms: ~0.7 trials/s → ~60K trials/GPU-day unbatched.
- Corpus cost on Modal depends on the dt decision: ~1M trials at dt=20ms ≈ 1-2 GPU-days (~$30-80); at dt=1ms ≈ 17+ GPU-days (~$400-600 on A10Gs). The rate-model design (20ms) keeps corpus gen cheap - another argument for it. Pilot measures real throughput.
- Storage (corrected math, dt=20ms, 50 steps/1s trial): dense float32 = 33MB/trial (~33TB/1M) - infeasible; sparse event lists at ~10% activity ≈ 5MB/trial (~5TB); bit-packed spikes ≈ 1MB/trial (~1TB/1M) - feasible remotely, still too big for casual HF download. Ship a ~50-100GB preview shard (parquet, event-schema) + regeneration code; full corpus streams to HF public repos as generated.
- Model: 50-200M params. Training on the laptop's 8GB card is tight (small dev runs only); pretraining is a burst-A100 + Kaggle-TPU-drip job (~24-48h unmeasured - sizing run gates).
- The hard budget is *engineering time*, not FLOPs: the tokenizer, the sex-conditioned attention mask, the behavior decoder, validation harnesses.

## 6. Kill criteria / go-no-go

- If the transformer can't beat BANC's linear influence matrix at predicting held-out activity → the learned model adds nothing; pivot to the embedding/Space artifacts.
- If sex-conditioned dynamics produce no divergent predictions beyond what the Cell paper already described structurally → claim shrinks to "dynamics explorer."
- If corpus gen at scale proves unstable (LIF tonic-rate tuning is finicky) → first fallback is reduced MaleCNS scope or our own Shiu-recipe kernel. NOTE: the old "fall back to FlyWire" fallback is now LICENSE-CONSTRAINED - FlyWire is CC BY-NC 4.0, so a FlyWire-derived model can't be released under the same permissive terms. If we ever use it, that artifact gets marked NC explicitly.

## 7. One-paragraph pitch

> "Every connectome project so far ships a wiring diagram or a hand-wired simulator. We shipped a brain's own training data. FlyGPT is the first foundation model whose vocabulary is a real nervous system's ~11.7K cell types, whose attention is wired by the actual synaptic graph of a complete male fruit-fly CNS, and whose output is motor programs - including courtship song. It's sex-conditioned against the female connectome, so one checkpoint can run the same circuit in male or female configuration. It learns from the connectome's own simulated dynamics, corrects against real recordings, and predicts the behavioral effect of silencing individual cell types - a benchmark no existing model can take."

---

## 8. Implementation dossier (extracted from the papers)

### 8.1 The LIF recipe (Shiu et al. Nature 2024, methods)

- LIF with alpha-synapse dynamics, two ODEs (membrane potential `v` decaying toward rest; conductance `g` decaying with tau; on presynaptic spike, `g += w`).
- Edge weight `w = connectome synapse count × sign × W_syn`. Sign: +1 excitatory / -1 inhibitory per presynaptic neuron.
- **W_syn is the single free parameter** of the whole model, calibrated so sugar GRNs at 100 Hz drive ~80% of max MN9 firing.
- NT sign rule: inhibitory iff >50% of a neuron's presynaptic sites are predicted GABA or Glut (cleft score >= 50); else excitatory.
- Stimulation: Poisson input at 10-260 Hz on identified sensory neurons; 30 trials x 1000 ms per experiment; 127,400 neurons in Brian2.
- Known stated limitations: no morphology, no receptor dynamics, no gap junctions, no non-spiking neurons, no internal state, no neuropeptides, basal rate = 0 (inhibitory edges are dead when target is silent).
- Reported accuracy: **91% of 164 testable predictions** consistent with empirical results; 84% excluding the split-GAL4 optogenetic panel. Shuffled-weight control collapses MN9 prediction to 1/100 - the wiring itself carries the signal.

### 8.2 The corpus generator already exists (2 days old)

`flybrain` (PyPI, released 2026-09-13, github.com/alextitonis/fly.ai, MIT):
- ~165K neurons / 25.6M signed connections baked from MaleCNS v1.0 (~260 MB brain files, ~1.1 GB from raw release).
- LIF stepping: **1.4 ms/step on RTX 4060**, 12-15 ms/step on 24 CPU threads, `batch=N` parallel flies per process.
- Ships `Eyes`/`FeatureDetectors` visual encoder (drives visual projection neurons) and `Trace`/`Readout` reservoir readouts.
- Repo extras signal a hobbyist ecosystem forming fast: `world/`, `talk/`, `valentfly/` (yes, a Valentine's-themed fly - the love-spot angle is already memeing).

**Corpus math (corrected):** at default dt=20ms, 1s trial = 50 steps ≈ 70ms wall on a 4060 (~14 trials/s, ~1.2M trials/GPU-day unbatched). At spike-resolved dt=1ms it's ~0.7 trials/s (~60K trials/GPU-day). `batch=N` multiplies throughput roughly linearly until VRAM saturates. So a 1M-trial corpus is ~1-2 GPU-days at dt=20ms (~$30-80 on A10Gs) but ~17+ GPU-days at dt=1ms (~$400-600) - the dt choice IS the budget decision. The bottleneck was never compute - it was that nobody wired stimulus programs to the generator yet.

### 8.3 Data files (exact paths)

- `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather` - full connection graph, 1.1 GB
- `body-annotations-male-cns-v1.0-minconf-0.5.feather` - types/classes/sides, 13 MB
- `body-neurotransmitters-male-cns-v1.0.feather` - per-neuron NT predictions, 42 MB
- `tbar-neurotransmitters-male-cns-v1.0.feather` - per-synapse NT probabilities, 2.7 GB
- `syn-partners-...feather` (6.8 GB), `syn-points-...feather` (12.7 GB) - synapse-level geometry if needed
- Skeletons: `gs://flyem-male-cns/v1.0/segmentation/skeletons-malecns/skeletons-swc/`
- neuPrint API: `dataset='male-cns:v1.0'`, free token. BANC via bancr/fancr + CAVE.
- HF mirror: `svgmediabills/malecns-connectome` (gated - accept terms once, then `huggingface_hub` pull).

---

## 9. The Modal build plan

> **SUPERSEDED 2026-09-17** - the compute strategy moved to $0 + new-account GPU credits + free weekly quotas; this section remains as the original cost baseline. **The operative compute plan is `plan.md` §2 + `free-compute-research.md`.** The "< $500" figure was also under-scoped (~10-20 required runs, not 1); see the compute audit for the corrected budget.

Architecture: corpus generation is embarrassingly parallel (independent stimulus trials) - exactly what Modal's `.map()` fan-out is for. Training is a single-GPU job. Total expected spend: **< $500**.

```python
# sketch - app layout
app = modal.App("flygpt")
vol = modal.Volume.from_name("flygpt-data")          # connectome + corpus + ckpts

@app.function(volumes={"/data": vol}, timeout=86400)
def ingest():                                        # MaleCNS feathers -> CSR graph

@app.function(gpu="A10G", volumes={"/data": vol}, timeout=3600)
def gen_trials(program_shard):                       # N stimulus programs -> parquet shards
    # flybrain batch sims; emits (cell_type_id, t_bin, trial_id) spike events

@app.function(gpu="H100", volumes={"/data": vol}, timeout=86400)
def train(cfg):                                      # dual-axis connectome-masked transformer

@app.function(gpu="A10G", volumes={"/data": vol}, timeout=7200)
def eval_battery(ckpt):                              # Shiu replication + DANDI + BANC baseline
```

### Phase plan

| Phase | Job | GPU | Est. cost |
|---|---|---|---|
| 0. Ingest | MaleCNS feathers -> CSR + type vocab + NT signs | CPU | ~$1 |
| 1. Corpus | ~1M stimulus trials fanned out via `.map()` over ~64 A10Gs | GPU | $30-80 |
| 2. Tokenize | events -> (type, t_bin) sequences, shard to HF datasets format | CPU | ~$5 |
| 3. Pretrain | ~100M-param connectome-masked transformer, ~24-48h | H100 | $100-300 |
| 4. Fine-tune | courtship/behavior head + sex-conditioning pass on BANC | A10G | $20-50 |
| 5. Eval | perturbation benchmark (DANDI LC silencing, Shiu replication) | A10G | $10-30 |
| 6. Ship | `huggingface_hub` upload, model cards, Space (Gradio CPU) | CPU | ~free |

Stimulus program library (the "prompts" the corpus is conditioned on): odor pulses across glomeruli, looming/escape (LC4/LPLC2 -> DNp01 giant fiber), sugar/bitter/water GRNs, JON mechanosensory, courtship cues (female silhouette, pheromone), optogenetic-style cell-type activation/silencing, tonic-baseline sweeps (fixes the basal-rate-0 pathology), and random "babble" baseline.

---

## 10. The fun layer

This project has unfair advantages in the meme economy. Use them.

**Naming candidates:** `FlyGPT`, `MushroomGPT` (after the mushroom body), `LoVP` (the love-spot neuron), `Bzzt`, `Lord-of-the-Flies`, `fruGPT` (the fruitless gene literally gates courtship circuitry). Recommend **`flygpt`** for the org + a model named **`lovp92`** or **`drosophilm`**.

**Hooks that are literally true:**

- *"The only model on HuggingFace whose tokenizer was proofread by 100+ human experts over 10 years."* (The connectome was human-proofread - the vocab is real cell types.)
- *"The only checkpoint with a `sex=` flag."* Same weights, male or female wiring.
- *"The only LLM whose output vocabulary includes flirting."* Courtship song - pulse vs sine bouts - is a real generative grammar the male brain speaks. The demo: prompt = "a female is 3 mm away" -> model sings.
- *"100% organic training data - generated by a brain, not scraped from Reddit."*
- *"Contains an actual panic button."* DNp01 (giant fiber) - the demo's "scare the fly" easter egg.
- *"It dreams faster than the fly thinks."* LIF corpus at 20ms/step vs transformer inference.

**Demo ("Probe a Fly Brain" Space):** stimulus buttons - `sugar`, `bitter`, `shadow overhead`, `female detected`, `dust on antennae` - each mapped to real sensory neuron populations; a propagation animation over the actual wiring; output readout shows which motor program decoded (feed / flee / groom / court). The viral clip: `female detected` -> generated pulse-sine song bout, compared side-by-side with a real SongExplorer ethogram.

**Milestones, named:** `v0.1 "First Spike"` (corpus gen), `v0.5 "It Sings"` (song head), `v1.0 "Battle of the Sexes"` (sex conditioning), `v2.0 "The Fly Awakens"` (closed-loop embodiment via FlyGym).

**A leaderboard nobody has:** per-cell-type perturbation prediction accuracy vs DANDI 000951 - "how well does your model know a fly?"

---

## 11. THE AUDIT - every claim verified, every design decision grilled

### 11.1 New prior art found during the audit

| Artifact | What it is | Impact |
|---|---|---|
| `ngxson/fly-hf` | 49,393-cell MaleCNS subset as reservoir trained on TinyStories, no frozen backbone. Cited by the FLM paper as prior work | Even the "subset reservoir on text" corner is taken |
| `erojasoficial-byte/fly-brain` + Zenodo preprint | FlyWire v783 (138,639 neurons) LIF at dt=0.1ms + NeuroMechFly body + Hebbian plasticity; claims "emergent individuality" - two identical flies diverge after 24h embodied experience (81% vs 47% escape rates, 76K divergent synapses) | Second embodied-fly effort, still brain-only (no VNC) |
| FLM details confirmed | Frozen LFM2.5-1.2B + full MaleCNS graph + 278K-param readout. Null result replicated across 3 seeds w/ parameter-matched control | Solid, honest negative result. "Fly wiring helps language" is answered: it doesn't (at this scale/method) |

### 11.2 Claim-by-claim verification

| Claim | Verdict | Notes |
|---|---|---|
| MaleCNS: ~165K neurons, ~11.7K types | VERIFIED w/ snapshot caveat | Cell paper final: 166,700 / 11,710. **Current neuPrint male-cns:v1.0 snapshot (Sep 2026, post-June updates): 164,838 neurons / 11,751 types / 171.9M synapses** - use manifest counts, not hardcoded numbers |
| Dimorphism: 8,069 iso / 138 dim / 289 male / 71 female types | VERIFIED | Cell final numbers (bioRxiv had 7,205/114/262/69 - use final) |
| ~25.6M connections vs ~172M synapses (neuPrint) | VERIFIED - explain | Flat file = neuron-pair edges (aggregated weights); synapses counted individually. Consciousness.ai recomputed 311.8M contacts / 151.9M connections w/ different thresholds - counting conventions vary |
| BANC is the female counterpart | VERIFIED w/ corrections | BANC v626 = **155,916 neurons** (not 160K or 188K - those are different counting conventions). Missing ~9,390 lamina neurons. Differs from MaleCNS by ~11K - specimen + pipeline + coverage differences, not pure sex |
| Shiu model: single free param, 91% accuracy | VERIFIED | W_syn calibrated on sugar→MN9; 91% on 164 testable predictions (84% excl. split-GAL4 panel). Caveat: predictions were selected for testability |
| flybrain: 1.4ms/step on RTX 4060 | VERIFIED | PyPI released 2026-09-13. Default dt=20ms - see objection below |
| "First trained generative model of a real complete connectome" | QUALIFIED - still standing | Neuroformer = technique on recorded/simulated data; FlyVis-GNN = 13.7K-neuron subsystem; nobody has done complete CNS. But must say "to our knowledge" - field moving weekly |
| "First sex-conditioned dynamics model" | STANDING | C. elegans dimorphic circuit simulations exist (Setty/Oren-Suissa) but at small scale, hand-built. First *learned, whole-CNS* sex-conditioned model is defensible |
| "First brain→VNC→motor-program model" | STANDING | Eon = brain-only (self-acknowledged); flybrain emits spikes not behavior; FlyGM is an RL policy |
| Corpus gen is cheap | VERIFIED w/ correction | At dt=20ms: ~14 trials/s/GPU. At spike-resolved dt=1ms: ~0.7 trials/s/GPU (~20x cost). ~10-25 GPU-h total on an A100-class card - fits inside free credit |

### 11.3 Technical design objections (the real grilling)

**O1. The tokenization scale problem.**
Neuroformer-style (neuron, time) event tokens work for ~hundreds of recorded neurons. Our corpus has ~11.7K types / 166K instances; a 1s trial can emit 10^4-10^6 events - sequences get quadratically expensive. *Fix*: model population vectors per time-bin (masked modeling over an ~11.7K-dim vector, diffusion/autoencoder style) OR OmniMouse-style per-neuron temporal patches. The "GPT" branding survives via autoregression over *time*, not event-by-event.

**O2. The attention mask is pointed the wrong way.**
Adjacency-masked attention over same-time tokens models correlation, not causation - synaptic influence runs pre(t) → post(t+δ). The connectome mask belongs on *cross-time* attention, and multi-hop influence needs depth. Alternative: connectome as positional-bias/graph-prior (PageRank-style encoding) + a GNN message-passing baseline to compare. Honestly, message-passing may be the more natural primitive - plan an ablation.

**O3. The determinism ceiling.**
The LIF sim is near-deterministic given a stimulus program. A transformer distilled from it can at best equal the LIF - it learns the simulator's quirks, not new biology. *Value must come from*: (a) masked inference/infilling (predict unrecorded populations - a real use case), (b) fine-tuning on real recordings to correct sim bias (the LIF can't be corrected - it's not learned), (c) speed for closed-loop, (d) the sex-conditioning experiment. If we only deliver "faster LIF," that's an engineering demo, not a model.

**O4. dt=20ms vs reality.**
flybrain defaults to 20ms steps - too coarse for spikes but *defensible*: our ground-truth validation data is calcium imaging (GCaMP ~100-500ms effective resolution), so a 20ms model is already finer than the observable. Design choice: match the observable, not the biophysics. But then the model is a *rate-dynamics* model, not a spike model - say so.

**O5. Validation mapping is lossy.**
Our output = rates/spikes; ground truth = deconvolved calcium. Must convolve predictions through an indicator kernel for comparison - adds noise. Do rank/order-based comparisons (which types respond, sign of change) rather than absolute rates - same epistemics as Shiu.

**O6. The sex claim is confounded three ways.**
BANC vs MaleCNS differ by specimen, pipeline, AND sex; BANC is missing the lamina; type systems differ. *Mitigation*: use the Cell paper's own iso/dimorphic type mapping (computed between MaleCNS and FlyWire/hemibrain) as ground truth for the sex split; report "specimen-conditioned" alongside "sex-conditioned"; use the bioRxiv VNC-alignment paper's matching for the cord. The result is still first-of-kind - just word it precisely.

**O7. "First" is perishable.**
flybrain (2 days ago), FLM (4 days ago), fly-hf, two embodied-fly repos, FlyGM, FLYNN - this space now ships weekly. Every claim needs a timestamp and the model needs to ship *fast*. The durable claim isn't "first to touch MaleCNS" - it's "first *trained foundation model* of a complete connectome," which has a higher bar than any of the hobby artifacts.

**O8. Basal-rate-zero pathology inherits to the corpus.**
Shiu's assumption (all neurons silent absent drive) means the corpus has no spontaneous activity - a huge fraction of real brain dynamics. *Fix*: sweep tonic baseline rates as a stimulus-program dimension so the corpus includes "resting-state" regimes; compare vs SCAPE spontaneous-activity data.

**O9. Is a transformer even the right object?**
FlyVis-GNN showed a GNN recovers mechanistic truth from dynamics. If our goal is interpretability/causal probing, a GNN student may beat a transformer. *Decision*: build both - transformer for the foundation-model story + masked-inference use cases, GNN as baseline/ablation. The HF release can ship the better one.

**O10. The "nobody's done it" meta-risk.**
Eon Systems has the recipe's author, funding, and a stated roadmap to scan brain+body. If they publish a MaleCNS trained model first, claims shrink to "first open HF artifact." Hedge: move fast, ship the *dataset + embeddings* artifacts early (those are useful regardless), and keep the science claims precise.

### 11.4 What would actually kill this project

1. Corpus generation produces trivially learnable dynamics (transformer memorizes deterministic LIF) → detectable early: train a small model on 10% of corpus, check held-out *stimulus-type* generalization (not just held-out trials). If it can't generalize to unseen stimuli, kill or redesign.
2. Sex-conditioned predictions don't diverge from what the Cell paper already described structurally → flagship result evaporates; pivot to embeddings + Space + corpus dataset as the deliverables.
3. The connectome-masked attention provides no gain over an equivalent dense/sparse baseline → the architecture story dies, but the model + corpus still stand.
4. Model can't beat BANC's linear influence matrices on held-out activity prediction → "learned model adds nothing over linear" is itself a publishable negative, but a weak HF release.

### 11.5 Revised defensible claims (final wording)

1. "To our knowledge, the first **trained** generative model of a complete, synapse-resolution animal connectome's dynamics." (Timestamp: Sept 2026. Check before shipping.)
2. "The first model conditioned on biological sex at the level of a complete CNS wiring diagram."
3. "The first model to span an animal's full sensor→interneuron→motor-neuron chain in a single learned object."
4. "The first public corpus of connectome-generated neural dynamics" (the dataset is valuable independent of the model).
5. NOT claimed: first fly simulation, first connectome+LLM, first embodied fly, first connectome chatbot.

### 11.6 Audit verdict

**The idea survives - with a sharper spine.** (See Section 12 for the final-pass updates.) The "fly brain as LLM" framing is dead on arrival (null result published). The surviving core: *a foundation model of a real nervous system's dynamics, where the connectome is the architecture prior, the training data is the brain's own simulated activity, the output is behavior, and sex is a conditioning variable.* The moat isn't compute or data access - it's being first to close the loop from wiring diagram to *learned, validated, downloadable* dynamics model, on a dataset released 14 weeks ago.

---

## 12. FINAL PASS - what changed in the last sweep

### 12.1 Housekeeping

- **`huggingface.co/flygpt` is unclaimed (404, re-verified 2026-09-15).** `pypi.org/project/flygpt` also 404 - both names free. Register before shipping.
- MaleCNS companion papers confirmed as validation targets: **visual pathways** (Cell 2026 - feedforward/feedback pathway classification, receptive-field predictions "consistent with physiological data" across thousands of uncharacterized types), **taste-feeding connectome** (Champalimaud/Ribeiro - complete GRN→feeding-motor-neuron map, cross-sex dimorphic taste circuits), **social behavior** (Curr Biol 2026).

### 12.2 The Eon backlash - required reading for our model card

The community response to Eon's "uploaded fly" (March 2026) defines what NOT to do:

- **The Verge / experts** (Bakhtiari, Bates): criticized missing reproducible details, unexplained "91%" figure, "for a claim of this magnitude, spell out the whole approach in specifics."
- **Carboncopies Foundation**: rigid body constraints + closed loop "can hide a model's failures" - almost any input can be made to look like behavior. Validation must look at microcircuits, not outputs.
- **Brunton's "digital sphinx" preprint** (the killer control): a *C. elegans* connectome - the wrong animal's brain - wired to a fly body and RL-trained still produced walking. Embodiment demos prove nothing without connectome-specific controls. Cowley: "could a randomly connected connectome do the same behaviors?"
- **Turaga (Janelia)**: connectome models miss biophysics, neuromodulator pools, and the body.
- **Eon's own caveats**: DN→behavior mappings "somewhat arbitrarily chosen by hand"; "not yet proof that structure alone is sufficient"; VNC "another useful direction" (i.e., NOT done).

**Consequence for us**: (a) never claim "uploaded fly" or "digital fly"; (b) ship code + data + configs for everything; (c) the eval battery MUST include a **wrong-connectome control** (shuffled graph or C. elegans graph as the attention mask - the digital-sphinx control) and a degree-matched rewired control; (d) claim dynamics prediction at microcircuit granularity vs perturbation data, not embodied behavior theater.

### 12.3 Technical confirmations

- **Gap junctions confirmed missing from all EM connectomes** and functionally important in the fly (shakB innexin: VNC sensorimotor processing, visual-neuron intrinsic stability). Model card must list this.
- **VNC motor neurons are mostly terminal outputs** (rare presynaptic sites; ~10 in DLM MNs) - natural readout endpoints for the behavior head; the connectome ends at motor neurons, not muscles.
- **NT classes available**: 7 (ach, GABA, glut, dopamine, octopamine, serotonin, **histamine** - the photoreceptor transmitter, matters here because the optic lobes are in this dataset); ach=exc, GABA=inh, glut≈inh in fly brain. Per-synapse probabilities ship in `tbar-neurotransmitters` and per-synapse scores in malecns_09 (`syn_top_nt`, `syn_top_p`, + all 7 score columns).
- **Standard connection threshold**: >=5 synapses per edge (FlyWire network-stats convention) - adopt for the adjacency mask.
- **Real-data caveat for corpus**: flybrain's default 20ms step + calcium-imaging timescale of validation data (~100-500ms) means a *rate-dynamics* model is the right object - confirmed design choice.

### 12.4 Final go/no-go

**GO - with the corrected claims.** Build order: (1) corpus dataset + generation code (useful regardless of model outcome), (2) neuron embeddings (cheap, durable value), (3) the dynamics model with wrong-connectome control baked in, (4) sex-conditioned experiment, (5) Space. Every artifact stands alone; the model card cites every limitation in Section 1.7 verbatim and never says "uploaded."

**The single most important sentence from this whole audit**: a wrong animal's brain can puppeteer a fly body (Brunton). Any demo we ship without a connectome-specific control will be dismissed the same way. The controls are the product.

---

## 13. HARDWARE + LICENSE AUDIT

### 13.1 Local hardware

**RTX 5050 Laptop, 8 GB VRAM, CUDA 13.3, 24 GB RAM (measured 2026-09-15).**

| Task | Feasible locally? |
|---|---|
| flybrain corpus smoke tests / dev | YES - expect ~1.4-2ms/step (4060-class ballpark), batch sims work |
| CSR graph build (25.6M edges) | YES w/ caveat - needs ~2-4 GB free RAM; currently only ~2 GB free, close apps first |
| Small-model smoke training (<=50M params) | YES - tight, use bf16 + gradient checkpointing + small batches |
| Full corpus generation at scale | NO - burst GPU credit (Vultr/GCP/OVH) or Kaggle drip; ~10-25 GPU-h on one card per the compute audit |
| Pretraining 100M+ params | NO - burst A100 headline + Kaggle TPU drip; local is dev loops only |
| Space demo / local inference | YES |

Windows notes: PyTorch+CUDA works natively under WDDM; flash-attn and Triton-based kernels are smoother on WSL2 or just stay remote for anything heavy. numba/CuPy (flybrain's engines) work on native Windows.

### 13.2 License matrix - verified

**Data:**

| Asset | License | Consequence |
|---|---|---|
| **MaleCNS v1.0** | **CC-BY 4.0** | Attribution required (cite Berg et al., Cell 2026). Derivatives (corpus, embeddings, weights trained on it) may be redistributed - keep CC-BY chain |
| **BANC** | **CC-BY 4.0** (HHMI open-access policy) | Attribution (Bates et al., Nature 2026). Clean for the female half of sex-conditioning |
| **FlyWire (FAFB v783)** | **CC BY-NC 4.0** | **NONCOMMERCIAL - the big catch.** Any FlyWire-derived artifact can't be used commercially. Prefer BANC for the female CNS (CC-BY + correct structural counterpart). Only touch FlyWire if unavoidable, and mark that artifact NC |
| DANDI 000951 | Published v0.240418.2218, embargo OPEN, 486.5GB/1413 assets; DANDI requires a license at publish - confirm CC-BY vs CC0 from the dandiset metadata at download | Attribution |
| Janelia song data (Lillvis) | Janelia figshare, CC-BY-style | Attribution |
| GCS buckets (flyem-male-cns, lee-lab) | Public read, same CC-BY terms | Pull directly; avoids the HF mirror's contact-gate |

**Code:**

| Package | License | Handling |
|---|---|---|
| flybrain | MIT | Build freely, keep notice |
| PyTorch / HF transformers+datasets / JAX / pyarrow / Numba / CuPy / MuJoCo / FlyGym | Apache-2.0 / BSD / MIT family | Safe |
| **navis, fafbseg, bancr (natverse stack)** | **GPL-3.0** | Copyleft. Keep OUT of the shipped package's required deps - use them only in `scripts/` data-prep tools, or reimplement needed queries on neuprint-python + pyarrow (Apache-friendly path). If we end up depending on them at runtime, our code must go GPL-3.0 - decide early |
| neuprint-python | Apache-2.0 | Preferred query path |
| Brian2 | CeCILL 2.1 (copyleft) | Avoid in shipped code - flybrain covers the sim |
| Neuroformer / chaobrain / SongExplorer | verify at integration | Don't vendor code; reimplement or depend + respect license |
| Modal SDK | free client, paid compute | No license issue |
| LFM2.5 (LiquidAI) | custom restricted license | Not needed - skip |

### 13.3 Our artifact licenses (decision)

- **Code: Apache-2.0** - permissive, HF-standard, keeps the door open for Eon/labs to use it. Quarantine any GPL-dependent prep scripts in `scripts/` with their own note.
- **Datasets (corpus, embeddings): CC-BY-4.0** + required citations block (Berg 2026, Bates 2026, Eckstein 2024, Dorkenwald 2024, Cowley 2024 for the DANDI 000951 benchmark).
- **Model weights: CC-BY-4.0** for male-only artifacts; **CC-BY-NC-4.0** for anything trained on FlyWire-derived data (default: don't use FlyWire - BANC covers the female side).
- `THIRD_PARTY_NOTICES.md` listing every upstream project + license + citation.
- Model card disclaimer block: the Section 1.7 limitation list verbatim + "not a digital fly, not an upload" + the digital-sphinx control results.

**Bottom line: we're clean.** Core pipeline is CC-BY + permissive code all the way down. The only real constraint: keep FlyWire out of anything we want to be commercially usable, and don't let GPL natverse tools into runtime dependencies.

---

## 14. EDGE-CASE AUDIT + FOOL-PROOFING

Third pass. Everything below is a concrete failure mode with a designed-in fix, plus new facts that change earlier assumptions.

### 14.1 New facts that change the plan

- **flybrain internals confirmed**: `FlyBrain(device, batch=N, dt, sensory_input, refractory)`, `batch` runs N independent flies per step (perfect for parallel corpus fan-out), `Trace`/`run`/`Readout` ship a built-in reservoir+linear-readout baseline (our free "dumb reservoir" control), `Eyes`/`FeatureDetectors` drive real visual projection neurons. Prebuilt brain files = 260MB sha256-checked; `flybrain build` rebuilds from the raw release (~1.1GB). Neuron model follows "Fly64" (Jessica Paquette) - verify its exact parameters at integration; it's v0.1.0 from a single maintainer, so **pin the version AND keep a thin `sim_backend.py` adapter** so we can swap in our own ~300-line numba/CuPy LIF kernel (Shiu recipe) if it breaks.
- **BANC hosts `malecns_09` in BANC's own schema** (`gs://lee-lab_brain-and-nerve-cord-fly-connectome/compiled_data/malecns_09`: meta.feather, simple/split edgelist, 301M-row synapses.parquet with per-synapse NT scores). MaleCNS and BANC are available in ONE format - a single loader serves both sexes, and BANC's NBLAST cross-dataset matching ships in the deposit (reuse it for cell-type alignment instead of computing our own).
- **"Fly plays Doom/Mario/Minecraft" already happened** (within days of release, plus the erojasoficial FlyWire+MuJoCo embodied fly). Embodiment/reservoir demos are crowded - our demo is "predict and explain the fly's response," not "fly plays game." Differentiation stays on learned+validated dynamics.
- **Shiu's official repo exists**: `philshiu/Drosophila_brain_model` (Brian2, activation+silencing protocol implemented). It's our replication benchmark reference - no license file, so reimplement the protocol, don't copy code.
- **HF mechanics**: parquet shards ~5GB, `push_to_hub(max_shard_size=...)` resumes on failure, watch 429 commit quotas on large uploads.

### 14.2 Corpus-generation failure modes

| Edge case | Fix |
|---|---|
| **Dense storage is infeasible**: ~165K neurons x 50 steps (1s @ dt=20ms) x float32 = 33MB/trial → ~33TB per 1M trials | Store sparse event lists (~5MB/trial @ ~10% activity) or bit-packed spikes (~1MB/trial → ~1TB/1M). HF ships a ~50-100GB preview shard + the generation code; full corpus stays regenerable remotely + streams to HF public repos. Sizing confirmed by pilot measurements, not these estimates |
| **Degenerate dynamics** (global silence or global synchrony regimes) | Pilot gate: 1K trials, per-trial health metrics (% active neurons, rate distributions, stationarity). Auto-flag degenerate trials as a labeled corpus class - don't silently drop them |
| **Basal rate = 0 bakes in "the brain is always silent unless poked"** | Corpus stimulus programs include tonic-baseline drive conditions as an explicit dimension; without them the model can't represent resting state |
| **W_syn is arbitrary** (single free param calibrated to one assay) | Sweep W_syn across trials, record it per-trial in metadata - turns an arbitrary constant into a documented uncertainty axis |
| **Determinism/reproducibility** | Per-shard provenance manifest: sim version, seed, stimulus config, git SHA, source-file sha256s. Bitwise reproducibility verified in CI on a small shard |
| **Threshold conventions baked in** (minconf-0.5 weights file, >=5 synapse edge cutoff, Shiu sign rule) | Generate corpus slices under 2-3 conventions (cutoffs 3/5/10; binarized vs count-weighted edges) - measures how much of the "result" is convention vs biology |
| **Ambiguous NTs**: low-confidence `syn_top_p`, mixed predictions, co-release ignored; dopamine/octopamine/serotonin are diffuse modulators IRL, not point synapses | Record per-edge NT confidence + sign-source in graph metadata; mark modulatory-class edges with a flag the model can learn on; sensitivity ablation flips low-confidence signs |
| **Stimulus coverage**: if 90% of cell types never get driven, model only learns 10% of the brain | Coverage tracking per program family; report "fraction of ~11.7K types ever active" as a headline corpus metric |
| **Autapses/self-loops/duplicate edges** | Explicit dedup + self-loop policy at graph build, documented; count and report them. NOTE: BANC excludes autapses at build (~2.1%) - if our MaleCNS build keeps them, the sex comparison inherits an asymmetric graph policy; align or document |
| **Trial cold-start**: every rollout starting from identical all-rest state is unphysical and teaches the model a fake transient | Warmup/burn-in steps per trial (drop the first ~100-200ms); randomize initial state where feasible |
| **Stimulus realism gap**: square-wave Poisson drives ≠ naturalistic input; model may learn the stimulator not the animal | Include naturalistic envelopes (real loom traces, odor plumes, song-context sequences) alongside synthetic programs |
| **Backend nondeterminism**: numba-CPU vs CuPy-GPU kernels may differ numerically - a corpus mixing backends isn't one distribution | Generate the whole corpus on ONE backend; record backend in the manifest |
| **flybrain's baked-in conventions**: its prebuilt graph embeds its own threshold/sign/binning choices - our corpus inherits them invisibly | Audit `flybrain build` source; extract and record its exact convention choices in the provenance manifest; prefer building our own graph from canonical files if any choice diverges from Shiu conventions |
| **Untyped neurons**: not all ~165K neurons carry a cell-type label - type-level vocab needs them somewhere | UNK token + report untyped fraction; instance-level fallback path for untyped populations |

### 14.3 Model failure modes

| Edge case | Fix |
|---|---|
| **Silence-dominated loss**: most neurons quiet most steps -> model learns to predict all-zero | Rate/BCE-weighted loss; metrics reported on active neurons + event F1, never raw accuracy alone |
| **Mask direction**: pre(t) -> post(t+1), and "who I attend to" = my presynaptic partners (transposed correctly) | Unit test: flip a single upstream neuron, verify only downstream tokens change |
| **Held-out rigor**: same-stimulus-family splits leak | Three split axes: held-out stimulus *families*, held-out *cell types* (predict neurons never driven), held-out regions. The cell-type holdout is the real connectome-generalization test |
| **Controls must be airtight**: rewired control confounded by degree changes | Degree-AND-weight-preserving rewiring (configuration model); + permuted-node-labels control; + wrong-animal graph (digital-sphinx test); + flybrain's built-in reservoir Readout as the "dumb" baseline; + BANC's precomputed linear influence matrix as the "is learning even needed" baseline |
| **Token granularity**: ~11.7K type tokens lose within-type variation (~14 neurons/type) | Hybrid vocab: type-level tokens + neuron-level tokens for designated circuits (motor neurons, courtship circuit) |
| **Song timing**: pulse-song IPI ~35ms vs 20ms bins is marginal | Song head evaluates at native resolution; consider 5-10ms bins for the courtship fine-tune |
| **OOV cell types** in future connectome versions | Type embeddings keyed by name hash + reserved slots |
| **Corpus = simulator output**: model learns Shiu-convention artifacts (alpha kernel shape, sign rule) as if they were biology | Dataset card states "simulated, not recorded" loudly; eval on real data is the only arbiter; report it as distillation-with-validation, not discovery |
| **Embedding artifact needs its own eval** or it's just a pretty t-SNE | Fixed eval suite for `fly-neuron-embeddings`: cell-type recovery, NT-class prediction, iso/dimorphic classification, NBLAST agreement - cheap, quantitative, citable |

### 14.4 Scientific-claim edge cases

- **Sex conditioning is sex+specimen+pipeline entangled.** Claim precisely: "cell-type-aligned male/female conditioning," validated only on dimorphisms the Cell paper already established structurally. Male-specific types get masked under female conditioning. Alignment: prefer BANC's **built-in cross-dataset type columns** (`fafb_cell_type`, `manc_cell_type` in the v888 metadata - curated matches, better than raw NBLAST) with the shipped NBLAST matrices as backup.
- **Female-side stimulus library is a real dependency we missed**: stimulus tokens are defined on male sensory neurons; female conditioning needs the homologous female inputs. Resolve via BANC's cross-dataset cell-type matching - and where no homolog exists, that stimulus simply isn't defined for the female branch (report coverage).
- **Sex-appropriate behavior vocab**: pulse/sine song is male-only; the female branch needs its own behavior head set (locomotion, grooming, feeding, rejection/kick behaviors). Mask male-only heads under female conditioning.
- **malecns_09 caveat**: the Lee-lab compiled copy is a derived re-formatting (their conventions, incl. autapse-excluded edgelists and v2/v3 detector snapshots) - use canonical Janelia files for the release manifest; malecns_09 only if schema unification is worth the provenance indirection. Document which snapshot (v2 size>=5 vs v3 size>=10) everywhere.
- **The corpus's validity ceiling is the LIF's.** Explicitly report corpus stats vs published physiology where they exist (spontaneous rates, response latencies) - a "simulator fidelity appendix" preempts the sharpest reviewer question.
- **"Fly plays Doom" is claimed territory** - reservoir+readout demos aren't novel. If we ship any game/interactive demo, cite flybrain's lineage and keep the headline on validation.
- **Firstness is perishable**: every "first" claim in the release gets a timestamp + "to our knowledge" + re-checked the week before upload.

### 14.5 Engineering/ops edge cases

- **Windows**: CUDA 13.3 driver runs CuPy cuda12 wheels fine; HF cache on NTFS can hit long-path limits - set a short `HF_HOME` or use WSL2 for heavy local work.
- **Don't pull the 30GB raw release** for corpus gen - flybrain's 260MB sha256-checked prebuilt files suffice. Raw pull only for the graph provenance manifest + synapse-level NT metadata if needed.
- **Storage/checkpointing**: corpus shards push to HF public repos as generated (canonical home); checkpoints to the Jio 5TB Drive + HF. Every billed account gets a budget alert; instances killed at job end; every script must also run locally (device-agnostic) so the pipeline never hard-depends on one provider.
- **HF upload**: 5GB parquet shards via `push_to_hub` (resumable); dataset viewer-friendly schema (long-format events or per-trial rows); private repo until launch.
- **Space**: static Space with distilled `flygpt-tiny` in-browser (ONNX/WebGPU) + canned rollout cache; ZeroGPU Gradio as alternative (free CPU work outside the GPU function). Demo never needs live GPU.
- **Supply chain**: flybrain is v0.1.0/3-days-old - pin `flybrain==0.1.0`, keep the adapter seam, engage the author (MIT, active) for goodwill + early warning on breaking changes.

### 14.6 Process fool-proofing (the "Garry-proof" layer)

1. **Pilot gate before scale**: 1K-trial pilot must pass (a) non-degenerate activity stats, (b) Shiu spot-checks on ~10 known circuits, (c) dynamic-range sanity, before any 1M-trial spend. No pilot pass, no corpus.
2. **Pre-registration**: `EVAL.md` (protocol, splits, metrics, kill criteria) + `CLAIMS.md` (every public claim -> the eval artifact that backs it) committed and timestamped BEFORE training. If a claim has no evidence row, it doesn't ship.
3. **Risk register** maintained in-repo: each risk gets likelihood x impact x mitigation x owner.
4. **Reproducibility CI**: small shard regenerates bitwise-identical; `pip install` from clean env tested on Windows AND Linux before release.
5. **One-command reproduction** documented for every headline number.

### 14.7 Updated go/no-go

Everything above strengthens rather than weakens GO. The two genuinely new constraints: (a) corpus economics - preview-shard + regeneration code instead of raw upload; (b) sex-conditioning claims must be cell-type-level and specimen-confound-aware. The pilot gate is the single most important new addition: **the project's first real experiment is verifying the simulator produces non-degenerate, Shiu-consistent dynamics on MaleCNS - before spending a dollar on scale.**

---

## References

- Berg et al. 2026, Cell - MaleCNS connectome + sex-dimorphism comparison
- Dorkenwald et al. 2024, Nature - FlyWire connectome (data license: CC BY-NC 4.0, join.flywire.ai/guidelines)
- Bates/Phelps/Kim et al. 2026, Nature - BANC female brain+VNC (Dataverse doi:10.7910/DVN/7WTH1N; CC BY 4.0)
- Shiu et al. 2024, Nature - whole-brain LIF model; reference impl: philshiu/Drosophila_brain_model (Brian2, no license file - reimplement protocol only)
- Lappalainen et al. 2024, Nature - connectome-constrained deep net (FlyVision)
- Eckstein et al. 2024, Cell - neurotransmitter prediction from EM
- Wang-Chen et al. 2024 / Ozdil et al. 2024 - NeuroMechFly v2
- Vaxenburg et al. 2025, Nature - whole-body MuJoCo fly
- Antoniades et al. 2024, ICLR - Neuroformer (neuron-token GPT)
- FlyVis-GNN (Saalfeld lab) - GNN dynamics inference
- FlyGM (arXiv 2602.17997) - connectome RL controller
- FLYNN (arXiv 2607.00025) - connectome-topology RNN navigation
- "Flies Are All You Need" / FLM (artificialscientific.com, 2026-09-11) - MaleCNS-in-LLM, null result
- ngxson/fly-hf - 49K-cell MaleCNS reservoir on TinyStories (precedes FLM)
- ConnectomeGPT-Worm (HF) - C. elegans layer in GPT; TuKoResearch/ConnectomeGPT100M - name collision only
- LLantia (arXiv 2608.00059) - LLM circuit inference
- Scheffer & Meinertzhagen 2021 - "A connectome is not enough"
- Brunton et al. - "digital sphinx" preprint (wrong-connectome control origin)
- DANDI 000951 (Cowley et al. 2024) - courtship silencing dataset, v0.240418.2218, 486.5GB
- Lillvis et al. 2023 - song data; Roemschied et al. 2023 Nature - song sequencing
- flybrain (PyPI, alextotonis/fly.ai, MIT) - neuron model follows "Fly64" (Jessica Paquette); MaleCNS-to-game demos (Doom/Mario/Minecraft) shipped within days of release
- erojasoficial-byte/fly-brain + Zenodo - FlyWire+NeuroMechFly embodied fly
- sjcabs/fly_connectome_data_tutorial - unified malecns_09/BANC schema docs (v2 vs v3 conventions, autapse exclusion, post_count column differences)
- Virtual Fly Brain neuron-counts page - counting-convention reference
- theconsciousness.ai Connectome Atlas; eon.systems
