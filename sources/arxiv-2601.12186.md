---
id: arxiv:2601.12186
type: paper
title: 'Aletheia: What Makes RLVR For Code Verifiers Tick?'
url: https://arxiv.org/abs/2601.12186
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# Aletheia: Analysis of RLVR for Code Verifiers

**Aletheia** is a study and execution-grounded testbed designed to analyze the performance-cost tradeoffs of Reinforcement Learning with Verifiable Rewards (RLVR) when training surrogate code-execution verifiers. The core problem addressed is the prohibitive cost of the full RLVR pipeline, which has hindered the adoption of generative verifiers in code generation compared to their use in math and science.

### The Aletheia Testbed
To isolate the effects of training recipes from data contamination, the authors developed the **Aletheia** testbed. The curation pipeline involves:
1. Generating solutions for competitive programming questions from **CodeContests** using a mix of "Weak" and "Strong" LLMs.
2. Calculating ground-truth pass rates ($\text{PR}$) via **SandboxFusion**.
3. Constructing lists of 2–5 candidates where exactly one is fully correct ($\max_n \text{p}_n = 1$).
4. Partitioning data into **Easy** ($\text{PR}_{\text{incorrect}} < 0.5$) and **Hard** ($\text{PR}_{\text{incorrect}} \in [0.7, 0.9]$) buckets.

The testbed evaluates robustness across three covariate shifts: **Aletheia-Strong** (stronger generators), **Aletheia-Hard** (semantically close distractors), and **Aletheia-Adv** (adversarial modifications).

### Method and Training Recipe
The study ablates three primary RLVR components across model sizes of 1.5B, 7B, and 14B:
*   **Thinking:** Comparing short chain-of-thought (GRPO-Instruct) against long reasoning traces (GRPO-Think) with varying reasoning budgets $B_{tr} = \text{len}(\mathbf{z}) \in \{4\text{k}, 8\text{k}, 16\text{k}\}$.
*   **Online Learning:** Comparing purely offline learning (**DPO-Think**), semi-online batch learning (**BO-GRPO**), and fully online learning (**GRPO-Think**).
*   **Negative Samples:** Comparing **GRPO** (which uses negatives) against **RAFT**, which trains only on correct responses via next-token prediction.

#### Key Formulas
The verifier predicts an index $\mathbf{o}$ for the best candidate. The reward is defined as:

$$
\mathbf{1}[\mathbf{o}=\arg\max_{n}\mathrm{p}_{n}]
$$

To evaluate the verifier's utility as an RL reward model, the authors use **Kendall’s $\tau$-b ($K\tau$)** to measure the reconstruction of the full candidate ranking:

$$
\mathsf{K}\tau=\frac{n_{C}-n_{D}}{\sqrt{(n_{C}+n_{D}+T_{w})(n_{C}+n_{D}+T_{p})}}
$$

where $n_C$ and $n_D$ are concordant and discordant pairs, and $T_w, T_p$ are ties in predicted wins and pass rates, respectively.

### Key Quantitative Results
The analysis reveals that the optimal recipe is scale-dependent:

*   **Thinking Traces:** Benefits increase monotonically with scale. For 1.5B models, performance saturates beyond $B_{tr}=8\text{k}$, but 7B and 14B models continue improving up to 16k. Thinking is essential for **Easy-to-Hard** generalization; for example, GRPO-Think-16k (14B) achieves 66.84% BoN accuracy on Aletheia-Hard compared to 44.15% for GRPO-Instruct.
*   **On-Policy Learning:** Critical for small verifiers but diminishes in importance at scale. At 14B, **DPO-Think** (offline) achieves 73.89% average BoN accuracy, nearly matching the more expensive **BO-GRPO** (75.29%) and significantly outperforming its 1.5B counterpart.
*   **Negative Samples:** Consistently boost top-1 selection (BoN). For RL ranking, the gap between GRPO and RAFT grows with scale (from 7.2 $K\tau$ at 1.5B to 10.8 at 14B). RAFT becomes unstable at larger sizes, with reward curves flatlining or degrading.

**Pareto Optimality:** **DPO-Think-14B** is identified as a highly efficient choice, achieving 14B-scale performance at $1/5.2\times$ the cost of GRPO-Think-14B. **GRPO-Instruct-7B** serves as an optimal low-budget baseline.

### Stated Limitations
*   **Parseability:** DPO-Think-1.5B suffers from a low response parse rate (43%), making its $K\tau$ scores noisy.
*   **Generalization:** Easy-to-Hard generalization remains a significant challenge for all model sizes.
*   **Inference Scaling:** While self-consistency (SC@K) provides modest gains, it cannot compensate for the absence of thinking traces or negative samples during training.
