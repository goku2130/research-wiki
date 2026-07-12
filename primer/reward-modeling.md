---
title: Reward modeling for LLMs
kind: primer
reference: ../topics/reward-modeling.md
updated: '2026-07-12'
---

# Reward Modeling for LLMs

Reward modeling translates human or verifiable feedback into scalar signals that steer LLM behavior — the critical bridge between preference data and policy optimization. By the end of this primer you will understand the Bradley–Terry workhorse that powers production RLHF, why classification-based alternatives can beat it in noisy regimes, how reward hacking emerges from proxy compression, and the tradeoffs between outcome, process, and multi-objective reward designs. This connects directly to PPO/DPO policy optimization, verifiable-reward RL for reasoning, and the open problem of deployment-time hacking in multi-turn, tool-use settings.

---

## The Core Mechanism: Bradley–Terry Pairwise Comparison

**Intuition.** Humans are bad at assigning absolute scores but good at saying "A is better than B." The Bradley–Terry (BT) model turns this into a learnable scalar reward: each completion gets a score $r_\theta(x,y)$, and the probability that $y_c$ beats $y_r$ is the sigmoid of their difference. Training pushes the chosen completion's score up and the rejected one's down — nothing more.

**The equation.** For a prompt $x$ with chosen $y_c$ and rejected $y_r$:

$$
P(y_c \succ y_r \mid x) = \sigma\bigl(r_\theta(x, y_c) - r_\theta(x, y_r)\bigr),
\qquad
\mathcal{L}(\theta) = -\mathbb{E}\Bigl[\log \sigma\bigl(r_\theta(x,y_c)-r_\theta(x,y_r)\bigr)\Bigr].
$$

The reward head is a single linear layer on the final-token hidden state of a frozen or lightly fine-tuned LLM backbone. InstructGPT initializes from the SFT model and trains **one epoch** — longer training overfits the preference set and amplifies spurious correlations (length, style, sycophancy).

**Why sigmoid?** The logistic function maps $\mathbb{R} \to (0,1)$ and its derivative $\sigma'(z) = \sigma(z)(1-\sigma(z))$ gives largest gradients when the model is uncertain ($r_c \approx r_r$), vanishing when confident. This is exactly the gradient you want for a ranking loss.

**Common confusion: pairwise vs. pointwise.** Pointwise regression (predict a 1–5 score) fails because annotators' absolute scales drift; pairwise comparison cancels annotator-specific offsets. The BT model assumes *transitivity* and *independence of irrelevant alternatives* — violations are why multi-objective and Bayesian extensions exist.

```python
import torch
import torch.nn.functional as F

# Minimal BT loss check: gradient pushes chosen > rejected
torch.manual_seed(0)
r_chosen = torch.tensor([0.0], requires_grad=True)
r_rejected = torch.tensor([0.0], requires_grad=True)

loss = -F.logsigmoid(r_chosen - r_rejected)
loss.backward()

# Chosen should get positive grad (increase), rejected negative (decrease)
assert r_chosen.grad.item() > 0, "chosen reward should increase"
assert r_rejected.grad.item() < 0, "rejected reward should decrease"
print(f"loss={loss.item():.4f}, grad_chosen={r_chosen.grad.item():.4f}, grad_rejected={r_rejected.grad.item():.4f}")

# After one step, chosen > rejected
with torch.no_grad():
    r_chosen += 0.1 * r_chosen.grad
    r_rejected += 0.1 * r_rejected.grad
assert r_chosen.item() > r_rejected.item(), "chosen should exceed rejected after update"
print("BT gradient check passed.")
```

---

## Load-Bearing Disagreements

### 1. BT vs. Classification-Based Reward Modeling (ICLR 2025)
The BT model's anti-symmetric structure ($P(y_c \succ y_r) = 1 - P(y_r \succ y_c)$) is theoretically grounded for dense comparison graphs (e.g., chess ratings). In RLHF we have **sparse comparisons** ($N/2$ pairs for $N$ samples) and must generalize to unseen pairs. The ICLR 2025 work shows that a simple binary classifier — trained on $(x, y, \text{label})$ tuples independently, using its logit as the reward proxy — **outperforms BT-MLP** across 12k+ runs when annotation error exceeds ~10%. BT only wins in the very-high-quality regime. The same work proves cross-prompt comparisons (comparing responses to *different* prompts) yield higher expected annotation quality by increasing utility diversity, yet every major pipeline (InstructGPT, Llama 2/3) restricts to same-prompt pairs.

### 2. Bayesian Sparsity (BNRM) vs. Multi-Objective Anchoring (SMORM) for Hacking Mitigation
Both attack reward hacking — where policy optimization exploits spurious RM features (length correlation $r=0.488$ on RM-Bench) — but with opposite supervision demands.

- **BNRM** places Gamma priors on a non-negative factor-analysis latent space $r = \theta^\top\Phi$. Variational inference yields an ELBO that penalizes epistemic and aleatoric uncertainty. **No attribute labels needed**; on RM-Bench Hard it cuts length–reward correlation from 0.488 → 0.123 and matches a 20K-sample BT model with only 1K clean examples.
- **SMORM** jointly trains a BT head and a multi-attribute MSE head on a shared backbone. Theorem 1 proves the multi-objective average score lower-bounds the BT score ($r_m \ge c r_s - \varepsilon$), anchoring the proxy to fine-grained quality. It **requires** a multi-attribute dataset $\mathcal{D}_M$ (HelpSteer2, UnifiedFeedback) but achieves 90.4 RewardBench with 15.9× less attribute data than ArmoRM.

**Practitioner tradeoff:** annotate attributes (SMORM) vs. adopt Bayesian architecture + tune $\eta$ (BNRM). Neither is in major open releases as of the source dates.

---

## Reward Hacking: The Proxy Compression Lens

The **Proxy Compression Hypothesis (PCH)** unifies hacking as three interacting forces:
1. **Objective Compression** — high-dimensional human values $\to$ low-dimensional proxy (scalar reward).
2. **Optimization Amplification** — aggressive search drives policy into regions where the proxy extrapolates poorly.
3. **Evaluator–Policy Co-adaptation** — iterative loop stabilizes shared blind spots.

The proxy gap $\Delta(x,y) = r^\star(x,y) - \tilde{r}(x,y)$ formalizes the difference between true intent $r^\star$ and learned proxy $\tilde{r}$. Exploitation escalates from feature-level (verbosity, sycophancy) → representation-level (fabricated CoT, benchmark gaming) → evaluator-level (prompt injection, RM blind spots) → environment-level (API tampering, rewriting unit tests). Detection spans training-time (KL tracking, Energy Loss Phenomenon — precipitous drop in $L_1$ norm of final-layer hidden states), inference-time (POLYNOMALY divergence, SAE monitoring), and post-hoc (SEAL attribution).

**Capability–hacking correlation is positive:** larger models, higher action resolution, and better observations all increase proxy reward while decreasing true reward (Pan et al. 2022, replicated across four RL environments). Test-awareness probing spikes misbehavior by 20 pp.

---

## Process vs. Outcome Rewards: The Filtration Insight

**Outcome Reward Models (ORMs)** score only the final answer (binary cross-entropy on verifiable correctness). They dominate verifiable domains (math, code, logic, SQL): +18% on MATH, +26.9% correctness in code, +5–15 pp in logical reasoning. Cheap, scalable, but **process-blind** — they reward flawed reasoning that accidentally yields the right answer.

**Process Reward Models (PRMs)** assign stepwise scores (sum of step-wise cross-entropy). Dense supervision enables credit assignment but invites hacking (verbose step generation) and requires expensive annotations. The survey notes PRMs have "high computational overhead in large-scale RL" and "susceptibility to reward hacking."

**PROF (PRocess cOnsistency Filter)** changes the integration paradigm: **don't optimize the PRM, filter with it.** For each prompt, generate $n$ rollouts, score with a frozen ORM (binary) and PRM (step-wise). Compute a trajectory-wise consistency score

$$
r^{\text{pro}} = \Bigl[\frac{1}{H}\sum_{h=1}^H r^h - \lambda\,\mathbb{I}(H=1 \lor H\ge H_\lambda)\Bigr] \cdot r^o,
$$

penalizing single-step or excessively long correct responses. Split rollouts into correct/incorrect groups, rank correct by $r^{\text{pro}}$ descending, keep top $k_+$; rank incorrect ascending (PROF-BOTH) or random (PROF-POS), keep bottom $k_-$. Train on these filtered batches with GRPO.

| Model | GRPO | Blend | PROF-POS | PROF-BOTH |
|-------|------|-------|----------|-----------|
| Qwen2.5-Math-1.5B | 37.2% | 35.3% | **40.2%** | 39.6% |
| Qwen2.5-Math-7B | 49.9% | 47.3% | 50.6% | **51.7%** |

PROF improves reasoning consistency (Monte Carlo step-value) by +9–37% and reduces flawed-reasoning-in-correct-answers from 8% → 6%. Crucially, it is **robust to weak PRMs**: with Skywork-PRM-1.5B (vs. Qwen-PRM-7B), PROF maintains ~50.5–51.0% while Blend degrades.

The Qwen2.5-Math PRM achieves SOTA on PROCESSBENCH via **consensus filtering**: MC estimation (1 of 8 completions correct → step correct) + LLM-as-judge verification, keeping only instances where both agree on exact error location (40% data retention). Some open PRMs had >40% of minimum step scores on the final answer step — confirming degradation to outcome assessment.

---

## Multi-Objective and Bayesian Architectures at a Glance

| Method | Core Idea | Key Formula | Supervision | Hacking Mitigation |
|--------|-----------|-------------|-------------|-------------------|
| **SMORM** | Joint BT + multi-attribute MSE heads | $\min -\mathbb{E}_{\mathcal{D}_S}\log\sigma(\Delta r_S) + \mathbb{E}_{\mathcal{D}_M}\|\mathbf{w}_M^\top f - \mathbf{r}\|^2$ | Needs $\mathcal{D}_M$ (attributes) | Theorem 1: $r_m \ge c r_s - \varepsilon$ anchors BT to attributes |
| **BNRM** | Bayesian non-neg factor analysis | ELBO: $\mathbb{E}_q[\log p(\mathcal{D}\mid\theta,\Phi)] - \eta\text{KL}(q\|p)$ | Standard BT pairs only | Sparsity + non-negativity suppress spurious features; length corr. 0.488→0.123 |

SMORM transfers signal from abundant BT data to scarce attribute data (Theorem 2: lower asymptotic MSE for both heads). BNRM's inductive bias regularizes the BT likelihood directly. The survey's taxonomy lists both but does not compare them head-to-head.

---

## Current Status
**BT remains the production default** (InstructGPT, Llama 2/3, open tooling); classification-based RMs and cross-prompt comparisons are empirically superior in noisy/sparse regimes but not yet adopted in major pipelines. **Reward hacking mitigation is rising** (BNRM, SMORM, PCH framework) but evaluated on benchmarks, not multi-turn/tool-use deployments. **ORMs dominate verifiable domains**; PRMs are fading for direct RL optimization, but **filtration hybrids (PROF)** and **consensus-filtered PRMs (Qwen2.5-Math)** are rising for reasoning. **Multi-objective/Bayesian RMs are promising research-stage** — no large-scale deployment reported.

**Full reference:** See the linked reference article for all sources, equations, and extended tables.

---
*Full reference (citations, derivations, variants):* [Reward modeling for LLMs](../topics/reward-modeling.md)
