---
id: arxiv:2404.14367
type: paper
title: Is RLHF Dead? A Survey of Offline Preference Optimization for LLMs
url: https://arxiv.org/abs/2404.14367
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

# Summary: Preference Fine-Tuning of LLMs Should Leverage Suboptimal, On-Policy Data

### Core Problem
The authors address the lack of clarity regarding the necessity of on-policy reinforcement learning (RL) versus offline contrastive or supervised methods for preference fine-tuning of Large Language Models (LLMs). Specifically, they investigate whether on-policy sampling and the use of "negative gradients" (explicitly pushing down the likelihood of dispreferred responses) are critical for performance, and how these factors interact with the geometric alignment between the reference policy $\pi_{\text{ref}}$ and the ground-truth reward function $r^*$.

### Method and Recipe
The researchers employ a three-tiered analysis framework to isolate the effects of different fine-tuning components:
1.  **Didactic N-dimensional Bandit Problems:** Used to study geometric relationships without sampling error.
2.  **Synthetic LLM Problems:** Three scenarios—**Min Length** (reward peak in low-likelihood regions of $\pi_{\text{ref}}$), **Mode Length** (reward peak aligned with $\pi_{\text{ref}}$), and **Skew Length** (preference data distribution skewed away from $\pi_{\text{ref}}$)—to test reward model approximation and data coverage.
3.  **Full-scale LLM Problems:** Experiments using Pythia-1.4B and Mistral-7B models on the AlpacaFarm and UltraFeedback datasets.

They categorize fine-tuning methods along three axes: **on-policy sampling**, **sample reuse** (multiple gradient updates per sample), and **negative gradient**. To systematically test these, they implement a unified algorithm that varies the batch size $B$ (controlling "on-policyness") and the number of inner iteration steps $T$ (controlling sample reuse).

### Key Formulas
The study frames preference fine-tuning as a KL-constrained reward optimization:

$$
\max _ {\pi_ {\theta}} \mathbb {E} _ {\boldsymbol {x} \sim \mathcal {D} _ {\text {pref}}, \boldsymbol {y} \sim \pi_ {\theta} (\cdot | \boldsymbol {x})} [ r ^ {*} (\boldsymbol {x}, \boldsymbol {y}) ] - \beta \mathbb {D} _ {\mathrm{KL}} [ \pi_ {\theta} (\cdot | \boldsymbol {x}) | | \pi_ {\text {ref}} (\cdot | \boldsymbol {x}) ]
$$

For contrastive methods like DPO, the implicit reward is defined as:

$$
r _ {\theta} (\boldsymbol {x}, \boldsymbol {y}) = \beta \left[ \log \pi_ {\theta} (\boldsymbol {y} | \boldsymbol {x}) - \log \pi_ {\text {ref }} (\boldsymbol {y} | \boldsymbol {x}) \right]
$$

The authors unify these methods under the notion of **mode-seeking objectives**. A general contrastive update with a negative gradient is represented as:

$$
\theta_{t+1} \leftarrow \theta_t + \eta \mathbb{E}_{\boldsymbol{x}, \boldsymbol{y}_w, \boldsymbol{y}_l \sim \mathcal{D}} \left[ \nabla_\theta \log \pi_\theta(\boldsymbol{y}_w | \boldsymbol{x}) \cdot c_1(\boldsymbol{x}, \boldsymbol{y}_w, \boldsymbol{y}_l) - \nabla_\theta \log \pi_\theta(\boldsymbol{y}_l | \boldsymbol{x}) \cdot c_2(\boldsymbol{x}, \boldsymbol{y}_w, \boldsymbol{y}_l) \right]
$$

where $c_2 > 0$ denotes the presence of a negative gradient.

### Key Quantitative Results
*   **On-Policy Sampling:** In the "Min Length" and "Skew Length" scenarios (where $r^*$ and $\pi_{\text{ref}}$ are misaligned), on-policy sampling significantly improves performance and convergence speed. Conversely, in the "Mode Length" scenario, on-policy sampling has a negligible effect.
*   **Negative Gradients:** Offline methods with negative gradients (e.g., DPO, IPO) consistently outperform maximum likelihood methods (e.g., Pref-FT, RWR) when the reward peak is in low-likelihood regions of $\pi_{\text{ref}}$. In the "Mode Length" setting, the performance gap narrows.
*   **Complementary Effects:** Combining on-policy sampling with negative gradients (e.g., on-policy DPO) yields the best results, outperforming both on-policy RL (PPO/REINFORCE) and offline contrastive methods.
*   **Sample Reuse:** Moderate sample reuse ($T > 1$) can improve sample efficiency. However, excessive reuse hurts exploration in non-RL methods; PPO is more robust to this due to its off-policy correction mechanisms.
*   **Theoretical Insight:** The authors prove that on-policy RL and contrastive methods with negative gradients are "mode-seeking" (optimizing reverse KL), which allows them to relocate probability mass more aggressively than "mode-covering" supervised objectives (optimizing forward KL).

### Stated Limitations
*   **Lack of Statistical Guarantees:** The work provides conceptual and empirical evidence but lacks rigorous statistical guarantees.
*   **Under-studied Negative Gradients:** The authors note that the statistical properties of negative gradients are not fully explored in existing literature.
*   **Coverage Simplification:** The analysis considers preference data coverage relative to $\pi_{\text{ref}}$ but does not account for the coverage of the original pre-training distribution.
*   **Reward Model Quality:** The study does not explore the impact of reward model quality or specific parameterizations on the results.
