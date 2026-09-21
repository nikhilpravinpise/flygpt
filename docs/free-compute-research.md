# FlyGPT on $0 - Free Compute Deep Research (v3, audited)

Date: 2026-09-17 (audit pass: same date)
Constraints (final): **$0 new spend, no local training, no grant programs, no scope reduction.**
Assets in hand: **GitHub Student Pack + Google AI Pro via the Jio 18-month promo** - **verified on-account 2026-09-17: zero Colab CCUs** ("You are not subscribed... zero compute units available"). The promo is trial-classified; Colab on this account behaves as free tier. Branch B confirmed.

Grant programs (TRC/AWS-research/NVIDIA/ACCESS) are excluded by decision. Compute comes from: (a) recurring free quotas needing no application, (b) benefits inside subscriptions the user already holds, (c) one-time cloud trial credits. The laptop orchestrates only.

---

## 0. TL;DR

**The plan holds at full scope - and it now has a days-first shape.** The deep audit of ~55 credit offers found the real burst stack: **~$770-1070 of new-account credit, of which ~$520 is officially GPU-confirmed** (GCP $300 post-upgrade + OVH $200 + Thunder $20-if-eligible; Vultr $250-300 and Civo $250 are likely-but-unofficial for GPU) ≈ **~100-210 A100-equivalent hours confirmed-floor, ~170-450 if Vultr/Civo land** - all burnable in days. Corpus gen on a real GPU is ~10-25 hours total (the old 4060-class number made it look like a week) - **v0.1 artifacts (corpus + embeddings + first trained model) ship inside week one**. The following weeks-months are the *proof* layer - controls, seeds, eval battery - riding the free weekly drip. Claims mature as evidence lands; nothing ships unvalidated.

- **Google AI Pro via Jio promo: NO Colab CCUs** - verified on-account (zero units, unsubscribed status). The 200-CCU benefit excludes trial-type memberships and the Jio redemption counts as one. The account still carries the promo's real value for us: **5TB Drive** as the checkpoint/shard ferry. Colab itself remains usable at free tier (T4-when-available, ~12h cap - the panel shows up to 75h CPU runtimes).
- **Colab tier ceiling found**: H100 and G4 (RTX PRO 6000) are **Pro+-only as of 2026**. Our tier tops out at A100-80 - fine for this project's scale.
- **Student Pack audit**: DigitalOcean $200 is dead for GPU (excluded June 2026, program expired July 2026). **Azure student credit cannot buy GPUs** - student subs are vCPU-capped and GPU quota requests are denied unless you upgrade to pay-as-you-go (violates $0). Azure demoted to CPU/storage only. Codespaces remains a free CPU dev box.
- **Math correction from v2**: mixed GPU-hours are now converted to honest A100-equivalents (a P100-hour is ~6% of an A100, not 100%).

**The deciding unknown**: actual training throughput. The "24-48h/run" figure is unmeasured - a sizing run in a Kaggle TPU session (free) is now a hard Phase 3 gate.

---

## 1. Assets in hand - audited inventory

### Google AI Pro - via Jio promo (existing subscription)

**Jio offer**: 18 months of full Google AI Pro free for Jio unlimited-5G users (Rs 35,100 value; 5TB storage per jio.com's current page). Activated via MyJio on whichever Google account was used at redemption - all assets live on that account.

**CCU status - RESOLVED 2026-09-17**: account shows "not subscribed, zero compute units" -> the Jio promo is trial-classified and the 200-CCU benefit does NOT apply. Colab on this account is plain free tier.

| Benefit | Value | Use here |
|---|---|---|
| **Colab free tier** | T4 when available, ~12h cap, dynamic ~15-30h/wk | Overflow/evals only - NOT a budgeted source |
| 5TB Drive | on the Jio offer page, reliable | **The** checkpoint + shard ferry for Kaggle/GCP runs |
| $10/mo GCP credit | Google Developer Program - verify it appears on promo accounts | Drip-feeds the GCP bucket |

### GitHub Student Pack (already held)

| Offer | Audited status |
|---|---|
| Azure for Students $100/yr | **No GPU path** - ~3 vCPU cap, GPU quota denied on student subs (verified via MS Q&A threads). Credit usable for CPU VMs/storage only. Demoted |
| DigitalOcean $200 | **Dead** - GPU droplets excluded June 2026; program ended July 31, 2026 |
| Codespaces | ~180 core-h/mo CPU dev box |
| Actions | free CI on public repo |

### No-application recurring floor

| Provider | Capacity | Verified limits |
|---|---|---|
| **Kaggle GPU** | 30 h/wk, P100 16GB or 2xT4 | 12h sessions, phone-verify, internet toggle; `/kaggle/working` ~20GB; datasets 200GB/each private cap |
| **Kaggle TPU** | v3-8 (128GB HBM, ~420 bf16 TFLOPS - raw-FLOPS parity with an A100) | 20 h/wk, ~9h sessions; PyTorch/XLA or JAX - **not** CuPy |
| **Colab free** | T4 when available | dynamic ~15-30h/wk, 12h cap; unbudgeted overflow (no CCUs on this account - confirmed) |
| **Modal** | $30/mo credit + **$50 user-provided credit** | ~27 A10G-h/mo + ~45 A10G-h one-time - reserved for eval battery / CI CUDA smoke / odd jobs, never corpus/training |
| **Lightning** | ~80 GPU-h one-time + ~15 cr/mo | evals/dev |
| **Oracle Always Free** | 2 OCPU/12GB ARM (halved 2026) + 200GB + 10TB egress | CPU pipeline/orchestration |
| **HF Hub** | public best-effort free; private 100GB | canonical artifact store |
| **HF Spaces** | static free; 2 ZeroGPU Spaces free (acct >30d, verified email); free CPU Gradio gone | ZeroGPU or in-browser ONNX demo |
| **Paperspace** | M4000 8GB, 6h sessions | marginal |

### One-time fuel: the new-account GPU-credit stack (signup, NOT grants)

Deep audit of ~55 free-credit/free-tier offers (Sep 2026): **only a handful can buy a single GPU minute**. Verified GPU-eligible:

| Provider | Credit | GPU-eligible? | Yield | Catch |
|---|---|---|---|---|
| **GCP trial** | $300 / 90d | YES after paid-billing upgrade (credits still apply) | ~60-80h A100 on-demand; ~2-3x more on spot | card + budget alert + GPU quota request post-upgrade |
| **Vultr** | $250-300 / **30d expiry** | **LIKELY, unverified** - ToS says "select products only"; affiliate sources claim full stack incl. GPU. Verify GPU deploy works before counting | ~110-130h A100-40 PCIe ($2.27/h), ~52-63h H100 | card for identity; new-account-only; regional stock |
| **OVHcloud** | $200 / 30d | **YES - officially GPU-exampled** ("35h AI1-1-GPU notebook", "5h AI Training on 4 GPU nodes") | ~35 GPU-h on their AI1 SKU + CPU/storage | **auto-bills after credit ends** - teardown discipline mandatory; clock starts at first project creation |
| **Civo** | $250 / 31d | "not excluded, not confirmed" - attempt, verify at deploy | L40S-B200 class | 31d expiry |
| **Thunder Compute** | $20 | YES, A6000/L40 only | ~57 A6000h or ~25 L40h | **US-university email required** - user eligibility unverified |
| **Beam** | $30/mo recurring | YES, metered incl. GPUs | ~15-25 A10G-equiv-h/mo | second recurring credit alongside Modal |
| **Azure** (any) | $200 new / $100 student | **NO - GPU quota denied on trial+student subs** | 0 | dead for GPU |
| **Linode/Akamai** | $100 / 60d | **NO - GPU plans explicitly excluded** | 0 | CPU only |
| **Oracle trial** | $300 / 30d | NO GPU on free tier | 0 | CPU/ARM only |
| Nebius, TensorDock, Daytona, E2B, Novita, RunPod, Vast, Lambda, Shadeform, Salad, Cerebrium, Prime Intellect | - | **NO** (no credit, trial suspended, or GPU excluded) | 0 | skip all |

**Burst total, honestly split**: ~$520 officially GPU-confirmed (GCP $300 + OVH $200 + Thunder $20-if-eligible) + ~$500-550 likely-but-unverified (Vultr $250-300 + Civo $250) = **~$770-1070 headline**. In A100-equivalents: ~100-210h confirmed floor, ~170-450h if Vultr/Civo land - plus ~$60/mo recurring serverless credit (Modal + Beam) and the $50 Modal credit. Either tier covers corpus + headline run + embeddings with margin. This is the "few days" fuel that makes the fast path real.

**Corrected corpus timing**: flybrain's ~1.4ms/step figure is a 4060-class number. On an A100/A40 the sparse LIF step should run ~0.4-0.7ms (memory-bound, verify at pilot). Math: 1.2M trials x ~50 steps = 60M steps -> ~30-50 trials/s ideal -> **~7-12 GPU-h at speedup, ~23h even at zero speedup (4060 parity); budget ~10-25 GPU-h for the whole corpus with IO overhead** (~$25-60 of credit - a day or two, not a week). The dt=5ms slice (~100K trials x 200 steps) and female corpus (~200K x 50) add ~4-10h. Kaggle remains the $0-credit fallback path.

---

## 2. Audited capacity math (A100-equivalents, not raw hours)

De-rating: P100 ~0.06x, T4 ~0.1x, T4x2 ~0.2x, A10G/L4 ~0.3x, TPU v3-8 ~0.8-1.2x, A100 = 1x (approximate, workload-dependent - the sizing run replaces these with measurements).

| Source | Raw h/mo | A100-equiv h/mo |
|---|---|---|
| ~~Colab 200 CCU~~ - **confirmed absent** | 0 | 0 |
| Kaggle TPU v3-8 | ~86 | ~70-100 |
| Kaggle GPU | ~129 | ~8-26 |
| Colab free T4 (unbudgeted floor) | ~60-120 dynamic | ~6-15 |
| Modal + Lightning + Paperspace | ~50 mixed | ~10-15 |
| **Recurring total (confirmed)** | | **~95-155 A100-equiv h/mo** |
| One-time: GCP $300 (+$10/mo drip) | | ~60-80 on-demand A100 (~160-240 spot) |
| One-time: Vultr $250-300 (if GPU-eligible) | | ~110-130 A100-40 PCIe |
| One-time: OVH $200 + Thunder $20 + Civo $250-attempt | | ~45-55 + upside |

**Suite demand**: ~10-20 full-equivalent runs (main + dense/rewired/permuted/no-edge + GNN + scaling slices + multi-seed claim runs + fine-tunes + distillation) x ~24-48h = **~350-1000 A100-hours**.

**Fit**: with the burst stack (~100-210 A100-equiv-h confirmed floor, up to ~450 with Vultr/Civo) front-loading corpus + headline runs + embeddings, **v0.1 lands in week one**; the remaining suite demand (~350-1000 A100-h minus burst spend) completes on the recurring drip over the following weeks-months. If the sizing run shows ~6-12h convergence (plausible for population-vector modeling at ~50 steps/trial), the whole suite compresses into the drip faster. Either way the *artifacts* ship early; only the *proof* is scheduled.

**Schedule model - two clocks: the burst and the drip**:

- **Burst week (days 1-7)**: burn the ~$770-1070 one-time credit stack on the heavy contiguous jobs - corpus gen (~10-25 GPU-h on one A100/A40), headline training run (~24-48h), embeddings, first evals. VMs run 24/7 unattended; laptop is launch/monitor only. **Expiry discipline**: Vultr/OVH/Civo credits die in 30-31 days and OVH's clock starts at first project creation - activate only when the job is ready to launch.
- **The drip (weeks 2+)**: the maturation layer - control suite, multi-seed runs, eval battery - rides the weekly Kaggle quota and ~$60/mo serverless credits. This is where the months live, and it's *proof work*, not scope.

| Phase | Burst path (days) | Drip continuation |
|---|---|---|
| 0-1 graph+pilot | days 1-2: Oracle CPU + first Vultr/GCP session | - |
| 2 Corpus | days 2-3: ~10-25 GPU-h on A100/A40 -> HF shards | convention slices continue on Kaggle |
| 3 Training | days 3-6: main run on GCP/Vultr A100 + distill | **controls/seeds/scaling = the drip queue** |
| 4 BANC | female corpus slice is cheap, queues behind | same |
| 5 Eval | headline evals day 6-7 | full battery on drip |
| 6 Ship | v0.1 artifacts end of week 1: preview corpus + embeddings + base model + honest "controls pending" card | claims upgrade as controls land |

**The claim phasing is the honest part**: v0.1 ships with corpus + embeddings + a trained model - NOT with "beats rewired controls" (T-MODEL-3) or multi-seed significance. Those mature on the drip over following weeks. The kill-conditions don't care about speed; claiming them early does - that's the Eon mistake (audit §12.2). Wording discipline: "first trained dynamics model on a real connectome *artifact*" is timestampable now; "validated" claims wait for evidence rows in CLAIMS.md.

**Session mechanics**: the laptop only launches/collects - Kaggle background runs execute to session cap unattended, GCP/Vultr VMs run 24/7, checkpoint-to-remote makes every cap harmless. Weekly quotas are use-it-or-lose-it - the schedule is set by quota-weeks consumed, not laptop uptime.

**Engineering time is the other axis** (audit doc: "the hard budget is engineering time, not FLOPs"): pipeline code doesn't exist yet - graph loader, stimulus programs, tokenizer, mask, evals are days-to-weeks of dev regardless of compute. Burst compute and code-writing run in parallel: kick off corpus gen while building the model.

**Budget rules**:
1. Cloud credits never buy what a free quota covers - corpus gen and CPU work never touch them.
2. Controls/scaling runs go on the cheapest converging accelerator (Kaggle GPU/T4); A100-equivalents reserved for headline + seed runs.
3. Early-stopping rules in EVAL.md - controls stop at validation plateau, not fixed steps.
4. Corpus gen realism: 30 Kaggle GPU-h/wk x ~10-14 trials/s x ~70% uptime-overhead ≈ **~0.75-1.1M trials/week** - the 1M corpus in ~1-1.5 weeks, zero credits spent.

## 3. Phase map (nothing trains locally, no grants)

| Phase | Runs on | Source |
|---|---|---|
| 0 Ingest/graph | Oracle ARM / Codespaces / laptop (CPU dev) | free |
| 1 Pilot | Kaggle GPU or laptop (verify CuPy on Kaggle image) | free quota |
| 2 Corpus | **Burst GPU (Vultr/GCP/OVH)** -> HF public shards; Kaggle GPU fallback | trial credit + free quota |
| 3 Suite | **Burst A100 (headline run)** + Kaggle TPU/GPU drip (controls/seeds/scaling) | trial credit + free quota |
| 4 BANC | same; JAX kernel unlocks TPU-side gen | same |
| 5 Eval | Modal credits + Kaggle/Colab-free; DANDI -> Drive/cloud | credit + free |
| 6 Ship | HF repos + static-ONNX/ZeroGPU Space + Actions | free |

## 4. Risks (post-audit)

| Risk | LxI | Mitigation |
|---|---|---|
| GCP trial GPU quota/availability post-upgrade | MED x MED | Verify an A100 can actually be provisioned before building the schedule around it |
| **Vultr credit turns out GPU-ineligible** ("select products" ToS ambiguity) | MED x MED | GPU deploy test is step 1 at signup; fallbacks: OVH (officially GPU-covered), Civo attempt, GCP spot |
| OVH auto-bill after credit exhaustion | CERTAIN x MED if forgotten | Teardown script + billing alert mandatory; clock starts at project creation - don't create early |
| Thunder $20 student eligibility (US-univ email) | UNKNOWN x LOW | 5-min signup test; negligible either way |
| ~~Jio promo trial-classified~~ **CONFIRMED - no CCUs** | - | Resolved: suite rides Kaggle TPU + GCP credit; Colab demoted to unbudgeted free floor |
| GCP trial requires paid-billing upgrade (card on file) | CERTAIN x LOW | Budget alert + teardown script; credits burn first, nothing bills under $300 |
| Free-tier churn (2026 precedent: Studio Lab, HF CPU Spaces, DO student GPU, Oracle halving) | HIGH x MED | Provider-agnostic pipeline, no single-host dependency |
| Suite timeline vs novelty window | MED x HIGH | Corpus + embeddings ship first; sizing run sets the real calendar in week 1 |
| Kaggle CuPy/flybrain compat unverified | LOW x MED | Phase 1 pilot IS the verification - first session tests it |
| Drive staging bloat | LOW x MED | Drive = transit only; HF canonical; cleanup rule |

## 5. plan.md integration status (APPLIED 2026-09-17)

- [x] Compute architecture section added to plan.md §2 (matrix, two clocks, session rules)
- [x] Phase 1: laptop-or-Kaggle pilot option
- [x] Phase 2: burst GPU corpus + Kaggle fallback + HF shard push
- [x] Phase 3: burst A100 headline + Kaggle TPU control drip + sizing-run hard gate
- [x] Phase 6: static ONNX/WebGPU Space, ZeroGPU upgrade path
- [x] Hard rules: nothing bills past $0, Modal capped at credits
- [x] Open items: Phase 0 account checklist
- [ ] RISKS.md: seed with the §4 table (file not yet created - Phase 0 scaffold task)

**Post-audit delta to propagate**: Vultr downgraded to likely-unverified (30d expiry); Thunder needs US-univ email; OVH auto-bills post-credit; corpus ~10-25 GPU-h; GCP ~60-80h on-demand A100.
