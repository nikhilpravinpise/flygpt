# Future work - what is left and what it needs

Status baseline: commits `15318bd` + `e1a7fba` + `b6d2bf7`. Battery on real
artifacts: **31 pass / 1 fail / 27 na**. Tests: 275 pass / 1 skip. Pilot
corpus (1,000 real trials, 428.7M events, 0% degenerate) on disk under
`data/corpus/pilot/pilot/`. Everything below is honest remaining work; none
of it is pretend.

## 1. Train the model (blocks 21 of 27 NAs)

The single gate: no trained checkpoint exists, so every model-, embedding-,
and sex-conditioned check is `na`. The machinery is complete and verified
end-to-end on CPU (real corpus -> vocab mask -> JAX step -> checkpoint).

What a real run means:

- `configs/model_v1.yaml`: ~100M params, `total_steps: 200_000`,
  `batch_trials: 8`, `seeds: [1,2,3]` - three independent runs minimum
  (EVAL.md requires mean+CI across seeds).
- Training data: full corpus is ~1M trials at dt=20ms (the committed pilot
  is 1,000 - a 1,000x scale-up), plus a ~100K-trial dt=5ms slice and a
  ~200K-trial female corpus.
- Controls to train alongside the headline: dense, GNN, rewired,
  permuted-label, no-edge, shuffled-weights, wrong-animal. Linear and
  reservoir are eval-time baselines, not trained.

### Why this is infeasible on CPU

Measured real numbers, not estimates:

- Corpus generation on this CPU ran at **0.80 trials/s**
  (`results/pilot_report.json`, 1,247s for 1,000 trials). The full
  ~1M-trial corpus at that rate is ~14 days of continuous CPU sim -
  and that is before a single training step.
- The LIF sim step is memory-bound sparse propagation over 6.3M edges; on
  an A100/A40 it is expected ~0.4-0.7ms/step vs the CPU's much slower
  sparse ops. `docs/free-compute-research.md` sizes the full corpus at
  **~10-25 GPU-hours**; same work on this CPU is measured at ~14 days.
- Training: 200K steps x 3 seeds x (1 headline + 6 trained controls).
  Even at a hypothetical 1 step/s on CPU (optimistic for a 100M-param
  transformer over (B=8, T=9, V=13,966) batches) that is ~70 days. On an
  A100-class card the sizing figure is ~6-12h per run
  (`free-compute-research.md` - unmeasured, flagged as the deciding
  unknown; a Kaggle TPU sizing run is the hard Phase-3 gate).
- The checkpoint cadence (`checkpoint_every: 5000`, resume-is-normal) is
  designed for free-tier session caps (~12h Kaggle/Colab), not for a CPU
  that would need weeks inside a single run.

CPU can hold correctness. It cannot hold throughput. The pilot proves
the pipeline is right; the GPU is what makes it *finish*.

### The free-GPU path (researched in `docs/free-compute-research.md`)

| Source | Budget | Fit |
|---|---|---|
| Kaggle TPU v3-8 | 20 h/wk, ~9h sessions | JAX/flax is the native target (model_v1.yaml already JAX); TPU-CPU mixes needed for the sim |
| Kaggle GPU (P100/2xT4) | 30 h/wk | sim + training overflow |
| Modal | $30/mo + $50 one-time | eval battery, CI CUDA smoke - not corpus/training |
| GCP trial $300 | ~60-80 A100-h (after paid upgrade) | corpus burst + headline runs |
| Vultr $250-300 (30d) | ~110-130 A100-40h, unverified | verify GPU deploy before counting |
| OVH $200 (30d) | ~35 GPU-h | teardown discipline - auto-bills |

Total confirmed floor ~100-210 A100-equivalent hours against an estimated
~350-1,000 A100-h full-suite demand - the burst covers corpus + headline;
the proof layer rides the weekly Kaggle drip.

## 2. Remaining NAs and what unblocks each

| Test(s) | Needs |
|---|---|
| T-MODEL-1..11 | trained checkpoint(s); runs vs controls live in `evals/model_evals.py` |
| T-EMB-1..4 | trained embedding artifact |
| T-SEX-1..5 | trained model + BANC female edge set (graph ready) |
| T-SEX-6 | isomorphic-circuit agreement run across male/female checkpoints |
| T-REAL-2 | `pred_rates`/`rec_rates`: DANDI 000951 partial download (`python -m flygpt.evals.real_evals --download ...`) + model/sim over matched stimuli |
| T-REAL-3 | `song_stats` from `behavior.SongDecoder.ethogram` over a courtship dt=5ms slice |
| T-STIM-3 | a coverage-tracker report emitted during corpus gen (needs the full corpus run, not just pilot shards) |
| T-SHIP-4 | model card (`docs/model_card.md` or equivalent) written |
| T-SHIP-5 | dataset card written |
| T-SHIP-10 | the CUDA smoke run on a GPU provider (Modal T4 path exists; needs `MODAL_TOKEN_ID` secret wired into CI) |

## 3. The one recorded failure (not a bug - a result)

`T-REAL-1` ran for real: 23 LC-driver silencing experiments on the MaleCNS
graph, 3 seeds each, LIF sim as the predictor. Verdict: **0/23 directional
calls** - every per-seed CI straddles zero (`results/T-REAL-1.json`). Two
interpretations, both honest:

- the LIF sim is too noisy at n=3 seeds to resolve silencing direction
  (more seeds or control-delta flooring might move it), or
- the sim-as-model-substitute (EVAL 1.6) is too weak a predictor for
  published silencing outcomes, and the test is waiting on the trained
  model.

Either way it stays a `fail` in `results/` - not hidden, not relabeled.

## 4. Claim maturation order (when a GPU lands)

1. Sizing run on Kaggle TPU (free) - hard gate before committing credit.
2. Corpus burst (GCP/Vultr/OVH credit) - ~10-25 GPU-h.
3. Headline + controls training.
4. Eval battery rerun -> T-MODEL/T-EMB turn real.
5. DANDI partial download -> T-REAL-2/3.
6. BANC female corpus slice (~200K trials) -> T-SEX.
7. Cards + Space refresh -> T-SHIP-4/5/10.

Every check above is registered, dispatched, and already writes honest
`na` artifacts. The moment a GPU exists, the battery goes from
31/27 evidence to a real verdict - no engineering debt in the way.
