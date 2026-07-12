---
id: arxiv:2509.03403
type: paper
title: 'Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training'
url: https://arxiv.org/html/2509.03403v1
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-modeling
---

## Summary of "Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training"

### Core Problem

The core problem addressed is the "process-outcome mismatch" in Reinforcement Learning with Verifiable Rewards (RLVR) for reasoning tasks. While RLVR effectively improves final-answer accuracy, it often fails to reliably enhance the quality of the reasoning process itself. This occurs because outcome rewards (ORMs) only assess final answers, leading to "outcome reward hacking" where flawed reasoning can still receive maximal reward if it coincidentally produces a correct outcome. This generates biased gradients, hindering the learning of faithful reasoning. Conversely, Process Reward Models (PRMs), which provide step-wise supervision, can be unstable under distribution shifts during online RL training, and direct optimization or naive combination of PRMs with ORMs can lead to reward hacking (e.g., over-generating verbose steps to game the PRM).

The paper theoretically analyzes this challenge by defining a latent state variable $z$, where $z=1$ denotes a valid intermediate reasoning process. The probability of generating valid reasoning is $\alpha_{\pi}=\mathbb{P}(z=1|\pi)$. If an incorrect process ($z=0$) can yield a correct answer ($r=1$) with a small probability $\epsilon$ (i.e., $\mathbb{P}(r=1|z=0)\leq\epsilon$), the expected reward is:

$$
\mathbb{E}_{\pi}[r] = \sum_{z\in\{0,1\}}P(r=1|z)P(z|\pi) \approx \alpha_{\pi} + \epsilon(1-\alpha_{\pi}) = (1-\epsilon)\alpha_{\pi} + \epsilon
$$

The ideal objective is to maximize $\alpha_{\pi}$, but the term $\epsilon(1-\alpha_{\pi})$ introduces gains from spurious successes, leading to biased gradients that reinforce flawed reasoning.

### Method: PRocess cOnsistency Filter (PROF)

PROF is a data curation method designed to robustly integrate pre-trained PRMs into online RL training by filtering trajectories based on PRM-ORM consistency, rather than directly optimizing PRM rewards. It aims to harmonize accurate but coarse-grained ORMs with fine-grained but potentially noisy PRMs.

**Step-by-step Recipe:**

1.  **Generate Samples and Outcome Rewards:** For a given prompt $x$, generate $n$ responses (rollouts) $\{a_1, \ldots, a_n\}$ and obtain their binary outcome rewards $\{r_1^o, \ldots, r_n^o\}$, where $r^o \in \{-1, 1\}$.
2.  **Compute Process Rewards and Trajectory-wise Consistency Score:** For each rollout $a_i$ with $H_i$ steps, call a pre-trained PRM to generate step-level rewards $(r_i^1, \ldots, r_i^{H_i})$. Compute the trajectory-wise consistency score $r_i^{\mathrm{pro}}$ using the following formula:

$$
r_i^{\mathrm{pro}} = \left[\frac{1}{H_i}\sum_{h=1}^{H_i}r_i^h - \lambda I(H_i=1 \text{ or } H_i \geq H_{\lambda})\right] \cdot r_i^o
$$

    where $\lambda$ is a regularization parameter and $H_{\lambda}$ is a threshold for penalized step numbers. This regularization penalizes samples with no steps or excessively long steps in the correct group.
3.  **Divide Rollouts into Groups:** Separate rollouts into a correct group $\mathcal{G}_+$ (where $r_i^o = 1$) and an incorrect group $\mathcal{G}_-$ (where $r_i^o = -1$). Let $n_+$ and $n_-$ be the number of samples in each group, respectively, such that $n_+ + n_- = n$.
4.  **Determine Number of Samples to Keep:** Calculate the number of samples to keep from each group, $k_+ \in [n_+]$ and $k_- \in [n_-]$, such that $k_+ + k_- = m$ (the policy update size) and $k_+k_-$ is maximized. This ensures a balanced ratio of correct and incorrect responses in the final training batch. Maximizing $k_+k_-$ is equivalent to making $k_+$ as close as possible to $m/2$.
5.  **Rank and Filter Samples:**
    *   **Correct Group ($\mathcal{G}_+$):** Rank samples in $\mathcal{G}_+$ by their $r^{\mathrm{pro}}$ scores in descending order (higher consistency is better). Keep the top $k_+$ samples, denoted as $\mathcal{K}^+$.
    *   **Incorrect Group ($\mathcal{G}_-$):** Two variants are proposed:
        *   **PROF-POS:** Randomly pick $k_-$ samples from $\mathcal{G}_-$.
        *   **PROF-BOTH:** Rank samples in $\mathcal{G}_-$ by their $r^{\mathrm{pro}}$ scores in ascending order (lower consistency is better, implying more "flawed" incorrect responses). Keep the bottom $k_-$ samples, denoted as $\mathcal{K}^-$.
6.  **Output:** The combined set of kept trajectories $\mathcal{K}^+ \cup \mathcal{K}^-$ with a total size of $m$ is used for policy updates (e.g., with GRPO).

### Key Quantitative Results and Numbers

*   **Process-Outcome Mismatch:** Empirical analysis on 2k samples from Qwen2.5-Math-7B revealed that **26.28%** of correct responses still contained flawed reasoning (judged by Claude). Within this flawed-correct subset, PROF identified and filtered **65.88%** of these problematic samples.
*   **Accuracy Improvement (Average@16):**
    *   **Qwen2.5-Math-1.5B-base:**
        *   GRPO: **37.2%**
        *   Blend: **35.3%**
        *   PROF-POS: **40.2%**
        *   PROF-BOTH: **39.6%**
    *   **Qwen2.5-Math-7B-base:**
        *   GRPO: **49.9%**
        *   Blend: **47.3%**
        *   PROF-POS: **50.6%**
        *   PROF-BOTH: **51.7%**
    *   **LLaMA-3.2-3B-instruct:**
        *   GRPO: **23.6%**
        *   Blend: **15.7%**
        *   PROF-POS: **25.4%** (outperforms GRPO by **1.8%**)
*   **Matched-Cost Comparison (Qwen2.5-Math-7B-base):** PROF achieved larger gains than GRPO at the same computational cost, especially on harder problems (Level 4).
*   **Reasoning Consistency (Monte Carlo Step-Value Scores):** PROF consistently achieved higher average MC estimates across five benchmarks compared to GRPO. Specific improvements: Math500 (**9.2%**), Minerva Math (**37.4%**), Olympiad Bench (**15.9%**), AMC2023 (**9.2%**), and AIME2024 (**11.1%**). These gaps were "much larger" than outcome-accuracy gaps.
*   **Flawed Reasoning Rate within Correct Responses (Claude Audit):** The flawed-reasoning rate in correct responses decreased from **8%** for GRPO to **6%** for PROF.
*   **Robustness to PRM Quality:** When using a weaker Skywork-PRM-1.5B (instead of Qwen2.5-Math-PRM-7B) for Qwen2.5-Math-7B-base, Blend achieved lower accuracies, while PROF-POS (51.0%) and PROF-BOTH (50.5%) maintained performance close to the model trained with the 7B PRM (51.7%).
*   **Filtration Method Variants (Qwen2.5-Math-7B-base):**
    *   Mean (PROF's default): **51.7%**
    *   Minimum: **50.9%**
    *   Sum: **50.6%**
    *   Ratio (preserving original correct/incorrect distribution): **50.6%**
    *   Filter-Nstep (filtering by step count): **47.6%**

### Stated Limitations

*   **Computational Overhead:** PROF requires more computation than "Blend" or "vanilla GRPO" because it involves oversampling and subsequent filtering. Balancing efficiency and reasoning quality remains an important future direction.
*   **Dependence on Pre-trained PRMs:** The filtration mechanism relies on pre-trained PRMs. If these PRMs harbor biases towards specific reasoning patterns or languages, PROF could inadvertently amplify such biases by filtering out diverse but valid solutions.
*   **Approximate Reasoning Quality Signal:** Claude-based auditing for reasoning flaws is an approximate signal and cannot fully replace careful human judgment on step granularity, subtle unsupported jumps, or the level of detail.
