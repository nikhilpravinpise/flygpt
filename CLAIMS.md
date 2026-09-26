# CLAIMS.md - public-claim evidence register

**Rule**: a claim may appear in README, model cards, dataset cards, the Space, or
any HF artifact ONLY if its row here is `EVIDENCED` and links a committed artifact
under `results/`. CI enforces the link (T-SHIP-2): every `EVIDENCED` row's
artifact must exist and the artifact's stated pass criterion must hold.

**Status enum**: `UNSUPPORTED` (default - no artifact yet), `EVIDENCED`,
`REFUTED` (committed evidence contradicts the claim; the row stays, the claim is
removed from public text), `DOWNGRADED` (weaker wording survives; notes record
the downgrade).

The evidence-artifact column names the file that will carry the result. For
`UNSUPPORTED` rows that file does not exist yet; the path is the CI-checked
field once the row is upgraded. Each row cites the plan.md section-3 test that
produces its evidence. Thresholds for every pass criterion live in `EVAL.md` -
this file records verdicts, not criteria.

**Current status**: graph build, simulator, stimulus grounding, vocabulary,
and the pilot corpus are evidenced (`EVIDENCED` rows below). Model claims
(C1-C9), sex-conditioning claims (C10-C11), real-data claims (C12-C14), and
ship claims (C17-C21) remain `UNSUPPORTED` until a trained model and the
remaining artifacts exist - no fabricated evidence.

| ID | Claim | Status | Evidence artifact | Notes |
|---|---|---|---|---|
| C0 | Graph built from canonical MaleCNS v1.0 (`gs://flyem-male-cns/v1.0/`, not the HF mirror, not `malecns_09`) with sha256-verified provenance: node set = body-annotations row count, derived edge weights reconciled against `connectome-weights`, dedup/autapse/NT-sign conventions per `configs/graph_v1.yaml` | EVIDENCED | `results/graph_build_v1.json` | Phase-0 build-verification JSON written by the graph build + `tests/test_graph.py` (T-BUILD-1..7); provenance in `provenance/malecns_v1.json` |
| C1 | Model generalizes to held-out stimulus *families* above the majority/marginal baseline (not a lookup table of the simulator) | UNSUPPORTED | `results/T-MODEL-1.json` | Margin: event-F1 >= baseline + 0.10, CI lower bound > +0.05 (EVAL.md); fails -> kill condition K1 |
| C2 | Model predicts activity of cell types never stimulated during training (connectome generalization) | UNSUPPORTED | `results/T-MODEL-2.json` | Held-out-type split per EVAL.md split rules |
| C3 | Real connectome mask beats degree+weight-preserving rewired control AND wrong-animal (C. elegans) control - wiring matters, not just the architecture | UNSUPPORTED | `results/T-MODEL-3.json` | The digital-sphinx control. Fails -> kill condition K2 (architecture claim dead, model may still ship) |
| C4 | Real node identities beat permuted-label control | UNSUPPORTED | `results/T-MODEL-4.json` | Guards "position encoding does all the work" |
| C5 | Connectome-masked transformer >= parameter-matched dense transformer on identical splits | UNSUPPORTED | `results/T-MODEL-5.json` | Non-inferiority to ship; claim upgrades only at masked >= dense + 0.02 |
| C6 | Model beats BANC precomputed linear influence matrices on held-out activity prediction | UNSUPPORTED | `results/T-MODEL-6.json` | "Learned adds nothing over linear" -> kill condition K3; ship corpus+embeddings, downgrade claims |
| C7 | Model beats flybrain Trace/Readout reservoir baseline on same tasks | UNSUPPORTED | `results/T-MODEL-7.json` | "A dumb reservoir suffices" is a live alternative |
| C8 | Model trained on real-weights corpus beats identical model trained on shuffled-weights slice | UNSUPPORTED | `results/T-MODEL-11.json` | Separates "learned topology" from "learned weights" (Shiu's control lifted to corpus level) |
| C9 | Transformer >= GNN message-passing baseline head-to-head on identical splits, or GNN shipped instead with story adjusted | UNSUPPORTED | `results/T-MODEL-10.json` | Comparative, per audit O9 (build both) |
| C10 | Sex conditioning reproduces the pre-registered known-dimorphism set from the Cell 2026 comparison | UNSUPPORTED | `results/T-SEX-1.json` | Frozen dimorphism list lands before Phase 4 (EVAL.md section 4) |
| C11 | Identical homologous stimulus produces measurably divergent dynamics in dimorphic circuits under male vs female conditioning, stable across seeds | UNSUPPORTED | `results/T-SEX-2.json` | Divergence vs isomorphic-type null; fails -> kill condition K4 |
| C12 | Model predicts the direction of the behavioral/dynamic effect of silencing each of the 23 LC types in DANDI 000951 above baseline | UNSUPPORTED | `results/T-REAL-1.json` | Real held-out perturbation benchmark; BH-FDR across the 23-type panel |
| C13 | Predicted population dynamics correlate with recorded calcium imaging (SCAPE/LBM where alignment feasible) through an indicator kernel | UNSUPPORTED | `results/T-REAL-2.json` | Stretch goal per plan Phase 5; rank/order agreement, not absolute rates |
| C14 | Generated song ethograms match SongExplorer ground-truth bout/IPI statistics and context-dependence | UNSUPPORTED | `results/T-REAL-3.json` | Courtship slice runs at dt=5ms; never mixed with 20ms corpus in one split |
| C15 | Simulator + corpus reproduce known circuit responses in MaleCNS cell-type identities (Shiu-protocol spot-checks) | EVIDENCED | `results/pilot_report.json`, `results/T-SIM-7.json` | T-SIM-7: 10/10 directionally correct on the real graph (paired-recording sign panel + postsyn-propagation protocol); pilot: 1000 trials, 0% degenerate, 429.9M events |
| C16 | Corpus is reproducible: shard regeneration from manifest + seed is bitwise-identical on CPU backend and within documented FP tolerance on the same GPU class | EVIDENCED | `results/T-SIM-1.json`, `results/T-CORPUS-6.json` | T-SIM-1 bitwise-identical CPU rerun on the real graph; T-CORPUS-6 manifest+seed regeneration verified. GPU bitwise determinism is NOT claimed - FP tolerance per EVAL.md |
| C17 | Neuron embeddings recover held-out cell types, NT class, iso/dimorphic labels, and NBLAST morphological neighborhoods above baselines | UNSUPPORTED | `results/T-EMB-1.json` .. `results/T-EMB-4.json` | Embedding artifact ships early; fixed eval suite per audit 14.3 |
| C18 | Inference is faster than the LIF simulator it was distilled from, measured in steps/sec on the same GPU class | UNSUPPORTED | `results/T-SHIP-10.json` | Claim requires measured ratio >= 10x; a number, not a vibe |
| C19 | Single learned object spans the full sensory -> interneuron -> descending -> VNC motor-neuron chain, with decoded motor programs | UNSUPPORTED | `results/T-MODEL-1.json`, `results/T-REAL-3.json`, behavior-head evals | MaleCNS is the only dataset with the full chain; behavior decoder spec written before any head training |
| C20 | To our knowledge, the first trained generative model of a complete, synapse-resolution animal connectome's dynamics | UNSUPPORTED | `results/firstness_check.json` | Novelty claim - perishable. Requires timestamped recheck the week before upload (audit O7); never claimed without "to our knowledge" + date |
| C21 | To our knowledge, the first neural dynamics model conditioned on biological sex at the level of a complete CNS wiring diagram | UNSUPPORTED | `results/firstness_check.json`, `results/T-SEX-1.json` | Subordinate to C10/C11 evidence; confound wording mandatory ("cell-type-aligned male/female conditioning", specimen-conditioned ablation reported) |

## Explicitly not claimed

Per the audit's "what NOT to claim" list - these appear nowhere in public text:

- First MaleCNS simulation (taken: flybrain, theconsciousness.ai)
- Fly connectome + LLM (taken, and a published null result)
- Connectome-as-controller (taken: FlyGM, FLYNN)
- Embodied fly / "digital fly" / "uploaded fly" (taken: Eon, NeuroMechFly - and forbidden framing per the Eon-backlash review)
- Ask-the-connectome chat (taken: Codex Exodiac, LLantia)
- Bitwise GPU reproducibility (FP non-determinism; only tolerance-level agreement is claimed)
- Any "first" without a timestamp + "to our knowledge" + a recheck artifact

## Reproduction commands

Each evidence artifact is regenerated by an explicit command (run from the
repo root with the project venv; `graph_dir` points at the built
`data/graph/` from the Phase-0 build):

```bash
# C0 - graph build + verification (results/graph_build_v1.json)
python -m flygpt.graph --config configs/graph_v1.yaml --raw-dir data/raw --graph-dir data/graph --results-dir results

# C15 - pilot gate: 10 grounded families x 100 trials on the real graph
python -m flygpt.corpus --config configs/corpus_v1.yaml --graph-dir data/graph --pilot --work-dir data/corpus/pilot --results-dir results

# C16 - bitwise determinism on the real graph (results/T-SIM-1.json)
# plus every other T-id -> results/T-*.json
python -m flygpt.evals --graph-dir data/graph --results-dir results --only T-SIM-1

# full battery -> results/T-*.json + results/eval_battery.json
# (corpus-dir/vocab-dir/banc-graph-dir unlock the corpus, tokenizer,
# and sex-conditioning checks against real artifacts)
python -m flygpt.evals --graph-dir data/graph --corpus-dir data/corpus/pilot/pilot --vocab-dir data/vocab --banc-graph-dir data/graph_banc --results-dir results

# release checks (claims / license / secrets / all)
python -m flygpt.evals.ship_checks all --results-dir results
```
