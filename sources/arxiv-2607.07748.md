---
id: arxiv:2607.07748
type: paper
title: 'Selective Left-Shift: Turning Test-Time Compute and Difficulty-based Curation
  into Training Data for Low-Resource Code Generation'
url: https://arxiv.org/abs/2607.07748
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# Selective Left-Shift: Turning Test-Time Compute and Difficulty-based Curation into Training Data for Low-Resource Code Generation

### Core Problem
Improving Small Language Models (SLMs) for Low-Resource Programming Languages (LRPLs)—such as Julia and Ballerina—is hindered by a "trilemma":
1. **Data Scarcity:** Supervised Fine-Tuning (SFT) is bottlenecked by a lack of organic training examples.
2. **Inference Cost:** Test-time scaling (e.g., iterative self-correction) improves performance but is too computationally expensive for real-world deployment.
3. **Sparse Rewards:** Reinforcement Learning (RL) applied to base models often fails because the models generate frequent syntax errors, resulting in near-zero positive feedback.

### Method
The authors propose a three-phase pipeline that "left-shifts" expensive inference-time compute to an offline data synthesis stage, decoupling syntax acquisition from algorithmic reasoning.

#### Phase 1: Data Synthesis via Offline Inference Scaling
The pipeline repurposes iterative refinement as an offline curation engine. For a set of language-independent seed problems $P$ and test suites $T$, the base model $M_\theta$ generates a candidate solution $c_i^{(0)}$. If the solution fails, the system provides structured feedback $e_i$ (compilation errors or a single failed test case showing expected vs. actual output). The model iteratively refines the code:

$$
c_{i}^{(k+1)}=M_{\theta}\Big(p_{i},\;c_{i}^{(k)},\;e_{i}^{(k)}\Big)
$$

Only solutions that pass all tests are added to the verified dataset $\mathcal{D}_L$.

#### Phase 2: Syntax-Aware SFT
The base model is fine-tuned on $\mathcal{D}_L$ to embed strong syntactic and grammatical priors (e.g., Julia's 1-based indexing). The objective minimizes cross-entropy loss over verified prompt-completion pairs:

$$
\mathcal{L}_{\mathrm{SFT}}=-\sum_{(p,c^{*})\in\mathcal{D}_{L}}\sum_{t=1}^{|c^{*}|}\log\pi_{\theta}(c_{t}^{*}\mid p,c_{<t}^{*})
$$

#### Phase 3: RLVR with Difficulty-Curated Data
The SFT-initialized model undergoes Reinforcement Learning with Verifiable Reward (RLVR) using Group Relative Policy Optimization (GRPO).
1. **Difficulty Curation:** To maximize the learning signal, problems are selected based on ELO ratings to target the "edge of capability":

$$
D_{curated}=\{d\in D \text{ s.t. } ELO_{m}\leq d.elo\leq ELO_{m}+400D.elo\}
$$

2. **Reward Function:** A composite reward $r(c,p)$ is used:

$$
r(c,p)=r_{\mathrm{test}}(c,p)+\alpha\cdot r_{\mathrm{build}}(c,p)
$$

   where $r_{\mathrm{test}}$ is the proportion of passed tests: $r_{\mathrm{test}}(c, p) = \min \left(\frac {p}{\left| \mathcal {T} _ {p} \right|}, 1. 0\right)$, and $r_{\mathrm{build}}$ rewards compilation success and structural compliance.
3. **Zero-Advantage Masking:** To prevent the policy from converging on suboptimal heuristics, if no completion in a group achieves the maximum reward ($r_{\max} < 1.0$), the advantage $\hat{A}_i$ is set to zero for the entire group, discarding the update.
4. **GRPO Objective:**

$$
\mathcal{L}_{\mathrm{GRPO}}=\mathbb{E}_{p\sim\mathcal{P}}\left[\frac{1}{G}\sum_{i=1}^{G}\hat{A}_{i}\cdot\log\pi_{\theta}(c_{i}\mid p)\right]-\beta\cdot D_{\mathrm{KL}}(\pi_{\theta}\|\pi_{\mathrm{ref}})
$$

   where $\hat{A}_{i}=\frac{r_{i}-\mu_{G}}{\sigma_{G}+\epsilon}$.

### Key Quantitative Results
Evaluated on **Qwen3-8B**, the full pipeline achieved the following Pass@1 improvements:

| Language | Benchmark | Base Model | Full Pipeline | $\Delta$ |
| :--- | :--- | :--- | :--- | :--- |
| **Julia** | MultiPL-E | 44.0% | 68.6% | +24.6 |
| **Julia** | Ag-LCB | 9.0% | 39.2% | +30.2 |
| **Ballerina** | MultiPL-E | 4.4% | 49.7% | +45.3 |
| **Ballerina** | Ag-LCB | 2.9% | 25.0% | +22.1 |

**Efficiency and Cost:**
* The pipeline outperformed the previous SOTA (Agnostics) on Julia by **+7.6 points** (MultiPL-E) and **+14.2 points** (Ag-LCB).
* It used **1/3 of the data** and **1/6 of the cost** compared to the previous SOTA.
* Total cost was **$54.02**, compared to **$320.3** for the Agnostics pipeline.

### Limitations
The pipeline's effectiveness is dependent on the availability of a working compiler and language-agnostic I/O test suites for the target LRPL. Additionally, the RL phase requires a robust SFT prior; without it, the exploration space is too unconstrained, and the model spends its compute budget rediscovering basic syntax rather than learning algorithmic logic.
