---
id: arxiv:2605.29275
type: paper
title: Prompt-Level Reward Specifications for Open-Ended Post-Training
url: https://arxiv.org/abs/2605.29275
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# Prompt-Level Reward Specifications for Open-Ended Post-Training

### Core Problem
Open-ended post-training for Large Language Models (LLMs)—specifically in instruction following, creative writing, and decision support—suffers from the **prompt-level reward specification problem**. Standard reward mechanisms often rely on implicit criteria (RLHF), require unavailable reference answers (RLVR), or use opaque LLM-as-a-judge scalar scores. The authors argue that success conditions for open-ended tasks are heterogeneous, requiring a combination of local requirement satisfaction, holistic quality judgment, and deterministic constraint verification.

### Method
The proposed framework separates reward **specification** (constructed offline) from reward **computation** (executed online).

#### 1. Offline Reward Specification Construction
For each prompt $x$, the framework builds two reusable reward artifacts:
*   **Task-Adaptive Rubrics ($\mathcal{R}_x$):** A lightweight classifier assigns a coarse task label to the prompt. This label, combined with a shared template and task-specific module, generates a set of weighted, atomic, and discriminative criteria.
*   **Hard-Constraint Checkers ($\mathcal{C}_x$):** The framework extracts explicit surface-level constraints (e.g., length, required strings, formatting) and compiles them into executable Python code.

#### 2. Online Hybrid Reward Computation
Each prompt-response pair $(x, y)$ is scored using three complementary signals, all normalized to a $[0, 1]$ scale:

*   **Rubric-based Score ($s_r$):** An LLM judges each criterion $r_i$ in $\mathcal{R}_x$ as "yes" (1), "part" (0.5), or "no" (0). The score is the weighted average:

$$
s_{r}(x,y)=\frac{\sum_{i=1}^{m}w_{i}v_{i}}{\sum_{i=1}^{m}w_{i}}
$$

*   **Global Score ($s_g$):** An LLM provides a holistic raw score $g(x,y) \in [0, 10]$, which is normalized:

$$
s_{g}(x,y) = \mathrm{clip}(g(x,y)/10, 0, 1)
$$

*   **Code-based Score ($s_c$):** Executable checkers $\mathcal{C}_x$ return binary outcomes $b_j \in \{0, 1\}$. The score is the pass rate:

$$
s_{c}(x,y)=\frac{1}{n}\sum_{j=1}^{n}b_{j}
$$

These signals are aggregated into a unified hybrid reward. During online Reinforcement Learning (RL), the weight of the global score ($\alpha$) is linearly decayed from 1 to 0 over $T_{decay} = 800$ steps to shift focus from general helpfulness to fine-grained requirement satisfaction.

### Key Quantitative Results
The framework was evaluated using Group Sequence Policy Optimization (GSPO) across multiple backbones.

**Offline RM-style Ranking (RewardBench v2):**
Using Qwen3.5-35B-A3B as the evaluator, the Hybrid Reward achieved an **Overall score of 85.1**, significantly improving upon the "Global only" baseline (80.0) and "Rubric only" baseline (77.0).

**Online RL Performance (Average Score Gains):**
The hybrid reward improved policy performance across diverse benchmarks (IFEval, IFBench, Arena-Hard-v2.0, etc.):
*   **DeepSeek-R1-Distill-Qwen-7B:** 32.3 $\rightarrow$ 39.8 (+7.5)
*   **Qwen3-4B:** 51.4 $\rightarrow$ 60.1 (+8.7)
*   **GLM-4.7-Flash:** 63.9 $\rightarrow$ 71.9 (+8.0)
*   **Qwen3-30B-A3B:** 60.7 $\rightarrow$ 65.4 (+4.7)

**Reliability and Robustness:**
*   **Code-based Verification:** On 100 VERINSTRUCT prompts, adding the code component increased the Top-1 exact pass rate from **48.0% to 69.5%** and reduced constraint-discordant inversions from 14.8% to 3.1%.
*   **Extractor Robustness:** Replacing the proprietary GPT-5 rubric extractor with an open-weight Qwen3.5-397B-A17B model resulted in only a minor drop in the hybrid reward's overall score (85.1 $\rightarrow$ 84.5).

### Stated Limitations
*   **Evaluator Dependence:** The framework relies on strong general-purpose LLMs for artifact construction and scoring; rubric extraction remains a source of variability.
*   **Verification Scope:** Executable checkers are limited to surface-level constraints and cannot verify semantic correctness, factuality, or reasoning validity.
*   **Safety Risks:** Generated executable code requires sandboxing and auditing to prevent prompt-injection or execution-related risks.
*   **Evaluation Scale:** Human preference checks were small-scale and conducted by non-experts, serving only as auxiliary sanity checks.
