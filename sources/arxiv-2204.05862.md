---
id: arxiv:2204.05862
type: paper
title: Training a Helpful and Harmless Assistant with Reinforcement Learning from
  Human Feedback
url: https://arxiv.org/abs/2204.05862
retrieved: '2026-07-11'
maturity: comprehensive
topic: ppo-for-llms
---

The authors of "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback" address the core problem of training AI agents to be simultaneously helpful, honest, and harmless, recognizing the inherent tension between these objectives. They aim to evaluate the effectiveness of preference modeling (PMing) and Reinforcement Learning from Human Feedback (RLHF) in achieving this alignment.

The method involves a multi-stage process:

1.  **Data Collection:**
    *   Human crowdworkers engage in open-ended conversations with language models (predominantly 52B models).
    *   For "helpfulness" tasks, workers solicit assistance for text-based tasks and choose the more helpful and honest response between two model-generated options.
    *   For "harmlessness" (red-teaming) tasks, workers adversarially probe models to elicit harmful responses and choose the *more harmful* response.
    *   This process generates two distinct human-preference datasets: helpfulness and harmlessness.
    *   Data is collected in three tranches: from initial context-distilled LMs, with rejection sampling against early PMs, and from iterated online RLHF models.
    *   Crowdworker preferences are also used to compute Elo scores for model comparisons.

2.  **Preference Modeling (PMing):**
    *   Language models (ranging from 13M to 52B parameters) are trained as Preference Models.
    *   PMs are trained to assign a higher score to the preferred response in each human comparison.
    *   Training typically occurs for a single epoch.
    *   PMs are evaluated for accuracy, calibration (how well predicted probabilities match actual human preferences), and their ability to detect helpful/harmless behavior on independent benchmarks (e.g., HHH Evaluation, Bot Adversarial Dialogues).

3.  **Reinforcement Learning from Human Feedback (RLHF):**
    *   Prompts are extracted from the PM dataset, and an RL policy is trained to generate responses auto-regressively.
    *   The reward signal for the RL policy is provided by the PM score at the end of each generated response.
    *   Proximal Policy Optimization (PPO) is used to stabilize training.
    *   A KL penalty term is added to the reward function:
        $r_{total} = r_{PM} - \lambda_{KL} D_{KL}(policy \parallel policy_0)$
        where $r_{PM}$ is the preference model score, $\lambda_{KL} \geq 0$ is a hyperparameter (typically $0.001$), and $D_{KL}(policy \parallel policy_0)$ is the KL divergence between the current policy and its initial state.
    *   The relationship between PM scores and human preference probability is given by:
        $P(A > B) = \frac{1}{1 + e^{r_{PM}(B)}-r_{PM}(A)}$
    *   Additional prompts for RLHF training are generated using few-shot learning from a large LM.

4.  **Iterated Online RLHF:**
    *   The best RLHF policy is used to collect new comparison data from crowdworkers.
    *   This new data is mixed with existing data, and new PMs are trained.
    *   These updated PMs are then used to train new RLHF policies.
    *   This process is iterated to continuously improve PMs and policies, particularly in the high-score regime.

**Key Quantitative Results and Numbers:**

*   **Alignment Bonus:** For 13B and 52B models, RLHF training improves performance on zero-shot NLP evaluations and maintains performance on few-shot evaluations (Figure 3). Smaller models (e.g., 12B) show performance degradation.
*   **PM Accuracy:** PMs achieve 86% accuracy on the HHH alignment evaluation dataset, outperforming previous models and human average (75%).
*   **Crowdworker-Anthropic Agreement:** Average agreement between Anthropic researchers and crowdworkers was about 63%.
*   **Scaling Laws:** Preference model accuracy shows roughly log-linear trends with dataset and model size (Figure 7).
*   **RL Robustness:** Training is robust up to ~150k training samples, after which train and test PM scores diverge, indicating overfitting (Figure 4). Larger PMs are more robust.
*   **KL-Reward Relationship:** An approximately linear relationship is observed between $\sqrt{D_{KL}(\pi||\tau_0)}$ and the PM reward during RLHF training (Figures 4 and 13).
*   **Human Preference:** Crowdworkers prefer online HH models to human writers approximately 57% of the time.
*   **Sentiment Bias:** RLHF training tends to make model sentiment much more positive across racial and religious groups, though it does not necessarily remove bias (Figure 17).
*   **Specialized Skills:** Mixing PM training for HH with summarization or applying natural language RLHF to code-finetuned models incurs no degradation in performance for larger models (Figures 20, 21). For 52B models, RLHF improves programming ability on HumanEval for all pass@k.
*   **Helpfulness vs. Harmlessness Tension:** Training purely on one objective leads to performance on the other that is significantly worse than chance (Figure 19). Larger models are more robust to the mixture of helpfulness/harmlessness data and loss weighting.

**Stated Limitations:**

*   The definition of 'helpful' and 'harmless' is largely left to crowdworkers' interpretation, not explicitly defined or prescribed by the authors.
*   The harmlessness data collection method (choosing the *more harmful* response) made it difficult for models to learn sophisticated "hostage negotiation" behaviors, instead favoring simple refusals, leading to over-optimization for harmlessness and under-optimization for helpfulness in early RLHF policies.
*   PMs become less calibrated and robust at higher scores due to a lack of data in this regime.
*   RLHF tends to decrease policy entropy, potentially limiting the diversity of data collected in online training.
*   Evaluations using rigidly formatted NLP tasks can be difficult for RLHF models due to their narrower, lower-entropy output distributions.
*   The sentiment analysis model used for bias evaluations may have limitations (e.g., questionable scores for certain groups).
*   BBQ-Lite evaluation results were inconclusive, suggesting limitations in the measurement rather than definitive statements about model bias.
*   Gender bias evaluations were challenging because RLHF models were significantly less likely to use gendered terms, making direct comparison difficult.
*   Improvements in performance from RLHF are modest in some evaluations (e.g., HumanEval, where simple prompting of a base code model performed slightly better).
