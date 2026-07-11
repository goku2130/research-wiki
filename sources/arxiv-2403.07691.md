---
id: arxiv:2403.07691
type: paper
title: 'ORPO: Monolithic Preference Optimization without Reference Model'
url: https://arxiv.org/abs/2403.07691
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

# ORPO: Monolithic Preference Optimization without Reference Model

## Core Problem
Traditional preference alignment pipelines for large language models (LLMs) typically require a multi-stage process: an initial supervised fine-tuning (SFT) phase followed by a preference alignment phase (e.g., RLHF or DPO). This approach is computationally expensive because it requires maintaining a separate, frozen reference model (the SFT model) to prevent the policy from deviating too far during alignment. Furthermore, the authors observe that standard cross-entropy loss used in SFT lacks a mechanism to penalize rejected responses; consequently, while SFT increases the likelihood of the desired domain, it can inadvertently increase the likelihood of generating tokens in undesirable styles.

## Method
Odds Ratio Preference Optimization (ORPO) is a monolithic alignment algorithm that integrates preference optimization directly into the SFT process. It eliminates the need for a separate warm-up phase and a reference model by appending an odds ratio-based penalty to the standard negative log-likelihood (NLL) loss.

### Step-by-Step Recipe
1. **Input Data**: Utilize a pairwise preference dataset consisting of a query $x$, a favored response $y_w$, and a disfavored response $y_l$.
2. **SFT Loss Calculation**: Compute the standard causal language modeling NLL loss ($\mathcal{L}_{SFT}$) to maximize the likelihood of the favored response $y_w$.
3. **Odds Ratio Calculation**: 
    - Calculate the probability $P_\theta(y|x)$ for both $y_w$ and $y_l$.
    - Compute the **odds** for each response, defined as the probability of generating the sequence divided by the probability of not generating it.
    - Calculate the **odds ratio** ($\mathbf{OR}$), which represents how much more likely the model is to generate $y_w$ than $y_l$.
4. **OR Loss Calculation**: Wrap the log odds ratio in a log sigmoid function to create the relative ratio loss ($\mathcal{L}_{OR}$), which is minimized as the odds ratio increases.
5. **Monolithic Optimization**: Combine both losses using a weighting hyperparameter $\lambda$ and optimize the model parameters $\theta$ in a single training stage.

### Key Formulas
The average log-likelihood of generating a sequence $y$ of length $m$ is:

$$
\log P_{\theta}(y | x) = \frac{1}{m} \sum_{t=1}^{m} \log P_{\theta}(y_t | x, y_{<t})
$$

The odds of generating sequence $y$ are:

$$
\mathbf{odds}_{\theta}(y | x) = \frac{P_{\theta}(y | x)}{1 - P_{\theta}(y | x)}
$$

The odds ratio between the favored response $y_w$ and rejected response $y_l$ is:

$$
\mathbf{OR}_{\theta}(y_w, y_l) = \frac{\mathbf{odds}_{\theta}(y_w | x)}{\mathbf{odds}_{\theta}(y_l | x)}
$$

The total ORPO objective function is:

$$
\mathcal{L}_{ORPO} = \mathbb{E}_{(x, y_w, y_l)} \left[ \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR} \right]
$$

where the odds ratio loss is defined as:

$$
\mathcal{L}_{OR} = - \log \sigma \left(\log \frac{\mathbf{odds}_{\theta}(y_w | x)}{\mathbf{odds}_{\theta}(y_l | x)}\right)
$$

## Key Quantitative Results
ORPO was evaluated across models ranging from 125M to 7B parameters using datasets such as UltraFeedback and HH-RLHF.

### Leaderboard Performance
Fine-tuning Mistral (7B) with ORPO on UltraFeedback alone surpassed several larger state-of-the-art models:
*   **Mistral-ORPO-$\beta$ (7B)**: Achieved **12.20%** on AlpacaEval$_{2.0}$, **7.32** on MT-Bench, and **66.19%** on IFEval (instruction-level loose).
*   **Mistral-ORPO-$\alpha$ (7B)**: Achieved **11.33%** on AlpacaEval$_{2.0}$, **7.23** on MT-Bench, and **61.63%** on IFEval (instruction-level loose).
*   **Llama-2 (7B) + ORPO**: Scored **81.26%** on AlpacaEval$_{1.0}$ and **9.44%** on AlpacaEval$_{2.0}$, outperforming Llama-2 Chat (13B) on AlpacaEval$_{2.0}$ (7.70%).
*   **Phi-2 (2.7B) + ORPO**: Achieved **71.80%** on AlpacaEval$_{1.0}$ and **6.35%** on AlpacaEval$_{2.0}$.

### Reward Model Win Rates (OPT Models)
Using an OPT-1.3B reward model, ORPO demonstrated superior win rates over other methods:
*   **HH-RLHF**: ORPO achieved win rates of up to **84.0%** vs SFT, **79.4%** vs PPO, and **70.9%** vs DPO (for the 1.3B model).
*   **UltraFeedback**: ORPO achieved win rates of up to **80.5%** vs SFT and **85.8%** vs PPO.

## Stated Limitations
The authors identify the following limitations:
*   **Algorithm Scope**: The study did not incorporate a comprehensive range of all existing preference alignment algorithms.
*   **Model Scale**: The method was not scaled to models larger than 7B parameters.
*   **Dataset Diversity**: Fine-tuning was limited to a few datasets; the generalizability across diverse NLP downstream tasks and data qualities remains to be verified.
*   **Internal Analysis**: The internal impact of the ORPO method on the pre-trained language model's weights and representations was not fully explored.
