# AMBER — Implementation Details

This document maps every part of the paper to the code, lists all hyper-parameters, and explains how each experiment is produced. Section and equation numbers refer to the manuscript.

## 1. Code map

| Paper element | File | Function / class |
|---|---|---|
| Simulation parameters (Table I) | `src/env.py` | `Config` |
| Indoor-factory drop, 3GPP TR 38.901 InF-SL path loss | `src/env.py` | `Scenario.__init__`, `pathloss_db` |
| Incident power at tags, Eq. (1) | `src/env.py` | `Scenario.incident` |
| Backscatter power and carrier leakage, Sec. III-C | `src/env.py` | `Scenario.rx_power`, `Scenario.leak` |
| SINR and read success, Eqs. (2)–(3) | `src/env.py` | `Scenario.evaluate` |
| Legacy UE interference constraint, Eq. (4) | `src/env.py` | `Scenario.feasible` |
| UE mobility (time-varying constraint) | `src/env.py` | `Scenario.move_ues` |
| Inventory round over T slots, Eq. (5) | `src/env.py` | `run_episode` |
| Feasibility projection layer, Proposition 1 | `src/env.py` | `Scenario.project` |
| Graph state features, Sec. IV-A | `src/amber.py` | `graph_inputs` |
| Stage 1 encoder and CE activation, Eq. (6) | `src/amber.py` | `encode`, `ce_logits` |
| Stage 2 conditioned tag features and scores | `src/amber.py` | `stage2_feats`, `tag_logits` |
| Plackett–Luce ranking, Eq. (7); Gumbel-top-C sampling | `src/amber.py` | `rank_lists`, `pl_logprob` |
| Policy log-likelihood and entropy | `src/amber.py` | `logprob` |
| SINR-certified admission decoder, Sec. IV-E | `src/hybrid.py` | `sinr_admit`, `make_hybrid` |
| Phase A: behaviour cloning, Algorithm 1 lines 2–4 | `src/bc.py` | `make_data`, `bc_loss` |
| Phase B: self-critical policy gradient, Algorithm 1 lines 5–13 | `src/train.py` | `run` |
| Adam optimiser | `src/amber.py` | `Adam` |
| Benchmarks (Sec. V-B) | `src/baselines.py` | `random_policy`, `allon_policy`, `greedy_snr_policy`, `greedy_sinr_policy`, `upper_bound` |
| Experiments E1–E4 | `src/exp.py` | modes `e1`, `e2`, `e3`, `e4` |
| Paired significance test | `src/paired.py` | — |
| Figures 1–7 | `src/figs.py` | — |

## 2. System model as implemented

* Hall 60 m × 60 m, carrier 900 MHz. 16 CEs on a 4 × 4 grid with 1.5 m jitter, 4 readers at quadrant centres with 2 m jitter, tags drawn around 10 random work cells with 4 m spread, 8 legacy UEs uniform.
* Path loss: 3GPP InF-SL. LOS probability exp(−d/20 m). Shadowing 4 dB (LOS) / 5.7 dB (NLOS). NLOS loss is max(PL_LOS, PL_NLOS).
* All gains are large-scale (no small-scale fading). All powers are in mW internally; `LIN` and `DB` convert.
* A tag is activated if its incident power Q_k ≥ P_sens = −20 dBm. Carriers from different CEs add in power.
* Tag k is decoded at the reader with the highest interference-free SNR. The SINR counts co-subchannel tags at that reader, residual leakage ε = −80 dB of every active carrier, and noise −174 dBm/Hz + 10log10(15 kHz) + 7 dB.
* A read succeeds if the tag is unread, activated, scheduled and SINR ≥ 3 dB. Since γ_th > 1, at most one tag per (reader, subchannel) succeeds, so reads/slot ≤ R·C = 16.
* Legacy constraint: Σ_m a_m P_c α g_ml ≤ I_max for every UE, with α = −45 dB and I_max = −62 dBm. Each slot the UEs take a Gaussian random-walk step (σ = 1 m) and their shadowing is redrawn, so the admissible CE sets change continuously.

## 3. AMBER policy

**Graph state (Sec. IV-A).**
CE node features (7): x, y, worst normalised legacy interference in dB/20, strongest leakage to a reader over noise in dB/40, fraction of unread tags it activates alone, fraction of unread tags for which it is the strongest emitter, and an individually-feasible flag.
Tag node features (5): unread flag, x, y, incident-power margin over P_sens and best-reader SNR, both computed with all individually-feasible CEs on and clipped.
CE→tag edge weight: E_mk = P_c g_mk u_k / Σ_n P_c g_nk (row-normalised for CE aggregation). Tag–tag edges: 8 nearest neighbours.

**Encoder.** Input embeddings of width H = 32, then two residual message-passing rounds:
h_m ← h_m + ReLU(W_a h_m + W_b Σ_k Ẽ_mk h_k)
h_k ← h_k + ReLU(W_c h_k + W_d Σ_m E_mk h_m + W_e mean_{j∈N(k)} h_j)
Setting `FLAGS["graph"] = False` skips message passing (the AMBER-Lite ablation).

**Stage 1.** π_m = σ(v_cᵀ h_m + b_c). At training time â ~ Bernoulli(π); at inference â = 1[π > 0.5]. Then a = `project(â, prio=π)`.

**Projection (Proposition 1).** While some UE is violated, find the most violated UE l*, and switch off the active CE with the largest a_m·P_c α g_ml* / π_m. The loop terminates in at most M steps, because the empty set is feasible.

**Stage 2.** Four extra tag features under the realised set a: activated-and-unread flag, incident margin, best-reader SNR including leakage, and the number of competing activated tags at that reader. A second encoder pass (input weights `t2_in`) produces bounded scores z_k = 5·tanh(logit/5). Setting `FLAGS["stage2"] = False` zeroes these four features (ablation).

**Tag ordering.** For each reader, the candidates are the activated unread tags it serves. Training samples an ordering of up to C tags by Gumbel-top-C (exact Plackett–Luce sampling); inference sorts by z_k.

**Decoders.**
*Rank decoder* (used during RL training): subchannel = position in the order.
*SINR-certified decoder* (used for all reported AMBER results): visit tags in descending z_k, try every free subchannel of the tag's reader, and admit on the one that maximises total successful reads, only if this increases the count.

## 4. Training

| Item | Phase A (behaviour cloning) | Phase B (self-critical PG) |
|---|---|---|
| Script | `bc.py <tag> <epochs>` | `train.py <tag> <total_iters>` |
| Data / episodes | 50 teacher rounds × 15 slots = 750 states, K ∈ {100, 200, 300} | 6 fresh drops per iteration × 12 slots; K ∈ {100, 200, 300}; random warm start of 0–13 greedy slots |
| Teacher / baseline | Greedy-SINR | Reward of the policy's own deterministic action in the same state |
| Loss / objective | BCE(CE decisions) − Plackett–Luce log-likelihood of the teacher's per-reader order | (r − r̄)/(R·C) · log π(â, order) + 0.005 · H(π_CE) |
| Optimiser | Adam, lr 3e-3, mini-batch 16 | Adam, lr 5e-4, gradient clipping ±5 |
| Length | 25 epochs | 775 iterations |
| Checkpoint | `models/bc_<tag>.pkl` every epoch | `models/rl_<tag>.pkl` every iteration (resumable) |

Gradients are computed with `autograd` (pure NumPy). No GPU is needed; full training takes about 15–20 minutes on one CPU core.

## 5. Experiments

All test layouts use seeds of 2,000,000 and above, which are disjoint from training seeds (1,000 to 1,000,000). Every scheme uses the projection layer, so none of them violates the constraint.

| Mode | Command | Setting | Output |
|---|---|---|---|
| E1 | `python exp.py e1` | K = 200, T = 30, 20 layouts, 6 schemes | `results/e1.json` → Table II, Fig. 4 |
| E2 | `python exp.py e2 <K>` | K ∈ {100, 200, 300, 500, 800}, T = ⌈1.5K/16⌉; 8 layouts (K ≤ 300) or 4 | `results/e2_<K>.json` → Table III, Fig. 5 |
| E3 | `python exp.py e3 <I_max>` | I_max ∈ {−68, −65, −62, −59, −56} dBm, 10 layouts | `results/e3_<I>.json` → Fig. 6 |
| E4 | `python exp.py e4` | ablations, 12 layouts | `results/e4.json` → Table IV, Fig. 7 |
| Paired | `python paired.py` | AMBER vs Greedy-SINR on 30 further layouts | `results/paired.json` |

## 6. Main results (from the files in `results/`)

| Scheme (K = 200) | Read @10 | Read @20 | Read @30 |
|---|---|---|---|
| Greedy-SNR | 116.5 | 173.8 | 178.8 |
| Greedy-SINR | 123.6 | 176.6 | 179.6 |
| AMBER (BC only) | 123.3 | 168.7 | 172.0 |
| **AMBER** | **123.6** | **178.4** | **185.2** |

Paired test after 30 slots: +5.9 ± 1.7 tags (95% CI), 26 wins, 2 ties, 2 losses; paired t-test p = 1.4 × 10⁻⁷.

## 7. Known limitations (as reported in the paper)

* The gain appears in the late inventory phase; with short horizons AMBER is on par with Greedy-SINR.
* The NumPy prototype is slower than the greedy benchmarks (2.6 ms at K = 100 up to 35 ms at K = 800), mainly because of the dense K × K nearest-neighbour search.
* Performance degrades for interference limits much stricter than the training value (−68 dBm).
* Graph message passing gives no measurable benefit at the studied scales (AMBER-Lite performs equally).
* Large-scale fading only; ideal subchannel orthogonality; single CE power level.
