---
id: arxiv:2403.07691
type: paper
title: 'ORPO: Monolithic Preference Optimization without Reference Model'
url: https://arxiv.org/abs/2403.07691
retrieved: '2026-07-12'
maturity: comprehensive
topic: dpo-variants
---

# ORPO: Monolithic Preference Optimization without Reference Model

## Core Problem
Traditional preference alignment pipelines for Large Language Models (LLMs), such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO), typically require a multi-stage process. This involves an initial Supervised Fine-Tuning (SFT) warm-up phase followed by a preference alignment phase that requires a separate, frozen reference model to prevent the policy from drifting. 

The authors identify a critical flaw in SFT: the standard cross-entropy loss only penalizes the model if predicted logits for reference answers are low, providing no direct penalty for non-answer (rejected) tokens. Consequently, during SFT, the log probabilities of rejected responses often increase alongside chosen responses, failing to discern between favored and disfavored generation styles.

## Method
Odds Ratio Preference Optimization (ORPO) is a monolithic alignment algorithm that integrates preference optimization directly into the SFT process. It eliminates the need for a separate SFT stage and a reference model by appending an odds ratio-based penalty to the negative log-likelihood (NLL) loss.

### Step-by-Step Recipe
1. **Input Data**: Utilize a pairwise preference dataset containing a query $x$, a favored response $y_w$, and a disfavored response $y_l$.
2. **SFT Loss Calculation**: Compute the standard causal language modeling NLL loss ($\mathcal{L}_{SFT}$) to maximize the likelihood of the favored response $y_w$.
3. **Odds Ratio Calculation**: Calculate the odds of generating the favored and disfavored responses. The odds ratio ($\mathbf{OR}$) determines how much more likely the model is to generate $y_w$ than $y_l$.
4. **Relative Ratio Loss Calculation**: Compute the relative ratio loss ($\mathcal{L}_{OR}$) by wrapping the log odds ratio in a log sigmoid function. This penalizes the model for assigning high probabilities to the rejected response.
5. **Monolithic Optimization**: Combine both losses using a weighting hyperparameter $\lambda$ and minimize the total objective:

$$
\mathcal{L}_{ORPO} = \mathbb{E}_{(x, y_w, y_l)} \left[ \mathcal{L}_{SFT} + \lambda \cdot \mathcal{L}_{OR} \right]
$$

6. **Training**: Fine-tune the pre-trained model (e.g., Mistral, Llama-2, Phi-2) using the combined loss for several epochs (e.g., 10 epochs for ORPO).

## Key Formulas
The average log-likelihood of generating output $y$ given input $x$ is:

$$
\log P_{\theta}(y|x) = \frac{1}{m} \sum_{t=1}^{m} \log P_{\theta}(y_t | x, y_{<t})
$$

The odds of generating sequence $y$ are defined as:

$$
\mathbf{odds}_{\theta}(y|x) = \frac{P_{\theta}(y|x)}{1 - P_{\theta}(y|x)}
$$

The odds ratio between the favored response $y_w$ and rejected response $y_l$ is:

$$
\mathbf{OR}_{\theta}(y_w, y_l) = \frac{\mathbf{odds}_{\theta}(y_w|x)}{\mathbf{odds}_{\theta}(y_l|x)}
$$

The relative ratio loss is:

$$
\mathcal{L}_{OR} = -\log \sigma \left(\log \frac{\mathbf{odds}_{\theta}(y_w|x)}{\mathbf{odds}_{\theta}(y_l|x)}\right)
$$

## Key Quantitative Results
ORPO demonstrated superior performance across various model sizes (125M to 7B) and benchmarks:

*   **Mistral-ORPO (7B)**: Using the UltraFeedback dataset, Mistral-ORPO-$\beta$ achieved **12.20% on AlpacaEval$_{2.0}$**, **7.32 on MT-Bench**, and **66.19% on IFEval** (instruction-level loose). Mistral-ORPO-$\alpha$ achieved **11.33% on AlpacaEval$_{2.0}$**, **7.23 on MT-Bench**, and **61.63% on IFEval**.
*   **Llama-2 (7B)**: ORPO achieved **81.26% on AlpacaEval$_{1.0}$** and **9.44% on AlpacaEval$_{2.0}$**, surpassing the RLHF-trained Llama-2 Chat (7B and 13B).
*   **Phi-2 (2.7B)**: ORPO achieved **71.80% on AlpacaEval$_{1.0}$** and **6.35% on AlpacaEval$_{2.0}$**.
*   **Win Rates (OPT models)**: On the HH-RLHF dataset, ORPO showed win rates against SFT up to **84.0%**, against PPO up to **79.4%**, and against DPO up to **70.9%** (for the 1.3B model).
*   **Efficiency**: ORPO requires half the number of forward passes per batch compared to DPO/RLHF because it does not require a frozen reference model.

## Stated Limitations
*   **Algorithm Scope**: The study did not incorporate a comprehensive range of all existing preference alignment algorithms.
*   **Scaling**: The method was not tested on models larger than 7B parameters.
*   **Generalizability**: Fine-tuning was limited to specific datasets; further verification across diverse domains and qualities is needed.
*   **Internal Analysis**: The internal impact of the ORPO method on the pre-trained language model's weights and representations remains understudied.
