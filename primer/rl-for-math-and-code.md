---
title: RL for math and code
kind: primer
reference: ../topics/rl-for-math-and-code.md
updated: '2026-07-12'
---

# Reinforcement Learning with Verifiable Rewards (RLVR) for Math and Code

**Scaffold.** RLVR replaces the learned reward model of classical RLHF with a *deterministic verifier*—a unit-test suite, compiler, formal proof checker, or SQL executor—that returns a binary reward (pass/fail) for each model completion. This eliminates reward-model overoptimization, makes every reward traceable, and has become the default paradigm for frontier mathematical reasoning and code generation in open LLMs. By the end of this primer you will understand how Group Relative Policy Optimization (GRPO) computes advantages without a critic, why verifier completeness is the load-bearing design choice, how co-evolution frameworks (CURE, UTRL) bootstrap verifiers from scarce ground truth, and why the field still debates whether RLVR gains reflect *search compression* (sampling known paths more efficiently) or genuine *capability expansion* (learning new reasoning strategies).

---

## Core Mechanism: GRPO with Deterministic Verifiers

Classical PPO trains a separate critic network to estimate state values. GRPO removes the critic entirely. For a prompt $q$, sample a *group* of $G$ completions $\{o_i\}_{i=1}^G$ from the current policy $\pi_{\theta_{\text{old}}}$. Each completion receives a scalar reward $r_i \in \{0, \gamma\}$ from the deterministic verifier. The advantage for every token in completion $i$ is the same group-normalized scalar:

$$
\hat{A}_i = \frac{r_i - \text{mean}(\{r_1,\dots,r_G\})}{\text{std}(\{r_1,\dots,r_G\})}
$$

The GRPO objective is a token-level clipped surrogate with a KL penalty against a reference policy $\pi_{\text{ref}}$ (typically the initial SFT or base model):

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\left( \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})} \hat{A}_i,\; \text{clip}\Bigl(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\varepsilon, 1+\varepsilon\Bigr) \hat{A}_i \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

The KL term uses the unbiased estimator $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$.

**Why this works.** The group mean/std normalization centers and scales rewards *per prompt*, so the policy learns *relative* quality within each context. No critic means no critic bias, no extra memory, and no value-function approximation error. The binary verifier reward makes the advantage signal sparse but *grounded*—every non-zero advantage corresponds to a verifiably correct completion.

**Common confusion:** The advantage $\hat{A}_i$ is a *single scalar per completion*, not per token. The token-level sum in the objective simply distributes that scalar uniformly across the completion's tokens. This is a deliberate design choice: with outcome-only verification, there is no per-token credit assignment signal, so uniform distribution is the maximum-entropy assumption.

```python
import torch

def grpo_advantage(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute GRPO group-normalized advantages.
    rewards: shape (G,) — scalar reward per completion in the group.
    Returns: shape (G,) — advantage per completion (same for all its tokens).
    """
    assert rewards.ndim == 1, "rewards must be 1D (group size G)"
    G = rewards.shape[0]
    assert G > 1, "need at least 2 samples for std normalization"
    mean = rewards.mean()
    std = rewards.std(unbiased=False) + eps
    return (rewards - mean) / std

# Quick sanity check
rewards = torch.tensor([0.0, 1.0, 1.0, 0.0])  # 2 passes, 2 fails
adv = grpo_advantage(rewards)
print("Rewards:", rewards.tolist())
print("Advantages:", adv.tolist())
# Passes get positive advantage, fails get negative; mean(adv) ≈ 0, std(adv) ≈ 1
assert abs(adv.mean().item()) < 1e-6
assert abs(adv.std(unbiased=False).item() - 1.0) < 1e-6
print("✓ Group advantages centered at 0, unit variance")
```

---

## Verifier Design: Completeness Is Non-Negotiable

The verifier is a function $r(s, a) \in \{0, \gamma\}$. **Partial verifiers**—e.g., a SQL parser that checks syntax but does not execute the query—create an exploitable gap: the model learns to generate syntactically valid but semantically wrong outputs that still receive reward. This is *reward hacking by design*. The RLVR loop therefore includes **verifier refinement**: when exploitation is detected, the verifier is hardened (e.g., adding execution, expanding test coverage) and training continues.

**Spurious rewards** are a documented pathology: Qwen2.5-Math-7B improved 21.4% on MATH-500 with *random* rewards versus 29.1% with ground-truth rewards, implying the RL process itself can refine internal pathways independent of the reward signal. This warns against over-attributing gains to the verifier alone.

**Meta-verification** (DeepSeekMath-V2) addresses the "generation-verification gap" in theorem proving. A verifier $\pi_\varphi$ scores proofs on a 3-point rubric; a *meta-verifier* checks whether the verifier's identified defects actually exist and whether the score matches the findings. The enhanced reward becomes $R_V = R_{\text{format}} \cdot R_{\text{score}} \cdot R_{\text{meta}}$, preventing the verifier from hallucinating issues to justify low scores. Verifier analysis quality rose from 0.85 to 0.96 under meta-verification.

---

## Co-Evolution: Bootstrapping Verifiers from Scarce Ground Truth

### CURE (Co-Evolving LLM Coder and Unit Tester)

CURE trains a single policy to act as both **coder** and **unit tester**. For a task $q$, the model generates $n$ candidate solutions $\{s_j\}$ and $m$ task-derived unit tests $\{u_k\}$. Execution yields a binary matrix $\mathcal{B}^\star \in \{0,1\}^{n \times (m + t_q)}$ where $t_q$ is the number of ground-truth tests.

- **Solution reward** = count of passed ground-truth tests: $\mathcal{R}_{s_j}^{\star} = \sum_{l=1}^{t_q} \mathcal{B}_{j, m+l}^{\star}$.
- **Unit-test reward (reward precision)** penalizes "naive" tests that pass incorrect solutions. Let $\mathcal{I}_{s_j} = \prod_{l=1}^{t_q} \mathcal{B}_{j, m+l}^{\star}$ indicate whether $s_j$ passes *all* ground-truth tests. The reward for generated test $u_k$ is:

$$
\mathcal{R}_{u_k}^{\star} = -\sum_{l=1}^{n}(1-\mathcal{I}_{s_l})\mathcal{B}_{l,k}^{\star} + \left(\prod_{l=1}^{n}\mathcal{I}_{s_l}\mathcal{B}_{l,k}^{\star}\right)\left(\sum_{l=1}^{n}(1-\mathcal{I}_{s_l})\right)
$$

  The first term penalizes tests that pass incorrect solutions; the second rewards tests that *only* pass correct solutions (when at least one incorrect solution exists). This optimizes the **precision** of the generated test as a classifier of solution correctness.

**Limitation:** CURE still requires ground-truth *unit tests* to define $\mathcal{I}_{s_j}$; it eliminates the need for ground-truth *code* but not for ground-truth *tests*.

### UTRL (Adversarial Co-Evolution Without Ground-Truth Tests)

UTRL removes the need for *any* ground-truth unit tests by framing test generation as an adversarial game between a test generator $\mathcal{M}_{\text{UT}}$ and a code generator $\mathcal{M}_{\text{code}}$, given only instruction-code pairs $(I, C^*)$.

- **Discrimination reward** $R_{\text{disc}}$: fraction of sampled imperfect codes $\mathcal{C}$ detected by at least one valid test.
- **Validity reward** $R_{\text{valid}}$: tests pass on $C^*$, normalized by $\max(|\mathcal{T}|, \tau)$ to prevent trivial test sets.
- Test generator reward: $r_{\text{UT}} = \lambda R_{\text{disc}} + (1-\lambda) R_{\text{valid}}$.
- Code generator reward: $R_{\text{code}}(C, \mathcal{T}, C^*) = \frac{\sum_{T \in \mathcal{T}} \text{Pass}(C, T) \cdot \text{Pass}(C^*, T)}{\sum_{T \in \mathcal{T}} \text{Pass}(C^*, T)}$.

As $\mathcal{M}_{\text{code}}$ improves, $\mathcal{M}_{\text{UT}}$ must generate increasingly discriminative tests, creating co-evolution. UTRL Qwen3-14B achieves Spearman's $\rho=0.827$ fidelity to ground-truth test evaluation (outperforming GPT-4.1), and the UTRL code generator reaches 15.3% pass@1 (vs. 15.9% upper bound with ground-truth tests).

---

## DeepSeekMath and DeepSeek-R1: Key Empirical Findings

| Stage | Data / Method | Key Result |
|-------|---------------|------------|
| **DeepSeekMath-Base** | 500B tokens (56% DeepSeekMath Corpus, 20% GitHub code, 10% arXiv, 10% NL, 4% AlgebraicStack) | Base 7B: 36.2% MATH (beats Minerva 540B at 33.6%) |
| **Code pre-training** | GitHub code in corpus | **Significantly benefits math reasoning** both with and without tool use |
| **arXiv data** | MathPile, ArXiv-RedPajama | **No notable improvement** on adopted benchmarks |
| **DeepSeekMath-RL (GRPO)** | Rule-based verifier (answer match, execution) | 51.7% MATH Top1, 88.2% GSM8K |
| **DeepSeek-R1-Zero** | Pure GRPO on DeepSeek-V3-Base, *no SFT* | AIME 2024 Pass@1: 15.6% → 77.9% (86.7% self-consistency) |
| **DeepSeek-R1 (final)** | 4-stage: cold-start SFT → RL → rejection sampling + SFT → RL (rule + model-based) | AIME 79.8%, MATH-500 97.3%, Codeforces 2029 |

**Emergent behavior in R1-Zero:** During pure RL, the model spontaneously develops long CoT with self-reflection and verification ("aha moment"), suggesting genuine capability expansion—not just search compression.

**Limitations documented:** suboptimal structured output/tool use; "overthinking" on simple queries; language mixing outside EN/ZH; high prompt sensitivity (zero-shot recommended); limited software-engineering gains due to RL evaluation cost; pure RL susceptible to reward hacking with *neural* (non-rule-based) reward models.

---

## Theoretical Insight: Not All Reward Errors Are Harmful

A linear-softmax bandit analysis categorizes proxy reward errors by their effect on ground-truth reward increase:

| Type | Proxy $r_P$ behavior | Effect |
|------|----------------------|--------|
| Harmful I (Hacking) | High $r_P$ for low $r_G$ | Policy converges to bad outputs |
| Harmful II (Stalling) | Mediocre $r_P$ for low $r_G$ | Policy concentrates on suboptimal outputs and stalls |
| Benign I | $r_P(y) < V_P(\theta_0)$ | Output never attracts probability |
| Benign II | Negligible $\pi_{\theta_0}(y)$ | Output never explored |
| **Beneficial I** | **Low $r_P$ for *mediocre* $r_G$** | **Prevents stalling, steers policy toward optimal $y_\star$** |

**Key theorem:** With orthonormal features, if initial $\pi_{\theta_0}(y_\star)$ is small relative to a mediocre $y_{\text{med}}$, time to reach high $r_G$ when maximizing $r_G$ directly is $\Omega(\pi_{\theta_0}(y_\star)^{-14/13})$, while maximizing a proxy $r_P$ that assigns *low* reward to $y_{\text{med}}$ requires only $\mathcal{O}(\pi_{\theta_0}(y_\star)^{-1})$. The gap can be arbitrarily large.

**Partial rewards can cap performance:** In instruction-following, rewarding partial correctness (e.g., 0.5 for 1 of 2 constraints) causes the policy to learn only the easier constraint if the initial policy is significantly more likely to produce partially correct than fully correct outputs.

---

## Load-Bearing Disagreement: Compression vs. Capability Expansion

**Promptfoo's compression ratio** quantifies search compression:

$$
\frac{\text{RLVR pass@1} - \text{Base pass@1}}{\text{Base pass@k} - \text{Base pass@1}}
$$

A ratio $> 0.7$ suggests search compression dominates. In one case, pass@1 rose from 40% to 65% (+25 pp) while pass@8 rose only from 75% to 77% (+2 pp), yielding ~71% compression.

**DeepSeek-R1-Zero's "aha moment"** argues for capability expansion: the model *emergently* develops self-reflection and verification behaviors during pure RL, which were not present in the base model.

The survey of RL for LRMs lists this as a "foundational problem" under active debate. **Entropy collapse**—where in-distribution accuracy rises but OOD performance deteriorates—is a documented failure mode correlated with excessive compression.

---

## Current Status

RLVR with GRPO is the **default, rising paradigm** for math and code reasoning in open LLMs; infrastructure is standardizing on **verl**, **OpenRLHF**, and **open-r1**. Fundamental disagreements persist on gain attribution (compression vs. expansion), entropy collapse remains unsolved, and overthinking is a deployment concern. The field is accelerating (135+ papers integrated from ICLR/ICML 2026 alone in the RLVR repo).

**Full reference:** See the linked reference article for all sources, equations, and extended results tables.

---
*Full reference (citations, derivations, variants):* [RL for math and code](../topics/rl-for-math-and-code.md)
