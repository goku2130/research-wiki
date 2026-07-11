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

Here's a faithful and thorough summary of the provided source, "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback," for a research wiki:

## Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback

This paper investigates the application of preference modeling (PMing) and Reinforcement Learning from Human Feedback (RLHF) to fine-tune large language models (LLMs) to function as helpful and harmless (HH) AI assistants. The core problem addressed is how to align LLMs with human values of helpfulness and harmlessness without compromising their general capabilities, and to develop robust and scalable training methods.

### Method/Recipe Step by Step:

1.  **Data Collection:**
    *   Human crowdworkers engage in open-ended conversations with LLMs via a chat interface.
    *   For **helpfulness**, crowdworkers solicit assistance for text-based tasks (answering questions, writing documents) and choose the *more helpful and honest* response from two model-generated options.
    *   For **harmlessness** (red-teaming), crowdworkers adversarially probe models to elicit harmful responses (e.g., planning illegal activities, toxic language) and choose the *more harmful* response.
    *   Data is collected in three tranches: an initial base dataset from context-distilled LLMs, a rejection sampling (RS) dataset, and an iterated "online" dataset from RLHF-trained models.
    *   Crowdworkers are not filtered based on agreement but on the sophistication and variation of their dialogues.

2.  **Preference Modeling (PMing):**
    *   Language models (ranging from 13M to 52B parameters) are trained as Preference Models.
    *   PMs are trained to assign a higher score to the preferred response in each human comparison.
    *   Training typically occurs for a single epoch.
    *   PMs are evaluated for accuracy, calibration, and agreement with human judgments.

3.  **Reinforcement Learning from Human Feedback (RLHF):**
    *   The prompts from the collected comparison dataset are extracted.
    *   An RL policy (the LLM) is trained using Proximal Policy Optimization (PPO) to generate responses autoregressively.
    *   The reward signal for the RL policy is the PM score assigned to the generated response.
    *   A KL divergence penalty term is added to the reward to stabilize training:
        $r_{total} = r_{PM} - \lambda_{KL} D_{KL}(policy \parallel policy_0)$
        where $\lambda_{KL} \geq 0$ is a hyperparameter (typically set to 0.001).
    *   The relationship between PM score and the probability of preference is given by:
        $P(A > B) = \frac{1}{1 + e^{r_{PM}(B)}-r_{PM}(A)}$
    *   Additional prompts for RLHF training are generated using few-shot learning from existing high-quality human queries.

4.  **Iterated Online RLHF:**
    *   The best RLHF policy is used to collect new comparison data from crowdworkers.
    *   This new data is mixed with existing data, and new PMs are trained.
    *   New RLHF policies are then trained using these updated PMs.
    *   This process is iterated to continuously improve PMs and policies, especially in high-score regimes.

### Key Formulas in LaTeX:

*   **Total Reward in RLHF:**
    $r_{total} = r_{PM} - \lambda_{KL} D_{KL}(policy \parallel policy_0)$
*   **Probability of Preference from PM Scores:**
    $P(A > B) = \frac{1}{1 + e^{r_{PM}(B)}-r_{PM}(A)}$
*   **Win Fraction and Elo Score Conversion:**
    $\text{Win Fraction} = \frac{1}{1 + 10} \quad \text{and} \quad \Delta(\text{Elo Score}) \approx 174 \cdot \Delta(\text{PM Score})$
*   **Weighted Loss for Mixed Objectives:**
    $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Helpfulness}} + \lambda \cdot \mathcal{L}_{\text{Harmlessness}}$

### Key Quantitative Results and Numbers:

*   **Alignment Bonus:** RLHF training improves performance on almost all NLP evaluations for larger models (13B and 52B), with smaller models experiencing an "alignment tax" (performance decline).
*   **Human Evaluation Preference:** Crowdworkers prefer the online HH model to professional human writers approximately 57% of the time.
*   **PM Accuracy on HHH Evaluation:** PMs achieve 86% accuracy on the HHH alignment evaluation dataset, outperforming plain LMs and context-distilled models.
*   **Crowdworker-Anthropic Agreement:** Average agreement between Anthropic researchers and crowdworkers is about 63%. PM-Crowdworker agreement is higher.
*   **Scaling Laws:** PM accuracy shows roughly log-linear trends with dataset and model size.
*   **RLHF Robustness:** Training is robust up to about 150k training samples, after which train and test PM scores diverge, indicating overfitting. Larger PMs are more robust.
*   **$\sqrt{D_{KL}}$ and Reward:** An approximately linear relationship is observed between $\sqrt{D_{KL}(\pi||\tau_0)}$ and PM reward during RLHF training, where $\pi$ is the policy and $\tau_0$ is the initial policy.
*   **Helpfulness/Harmlessness Tension:** Training purely on helpfulness or harmlessness data results in performance on the other distribution significantly worse than chance. Larger models are more robust to the data mixture and loss weighting (e.g., increasing $\lambda$ from 1 to 10 for loss weighting causes a 7.4% decrease in helpfulness accuracy for a 13M model, but only 1.5% for a 52B model).
*   **Specialized Skills:** Mixing PM training for HH with summarization or applying natural language RLHF to code-finetuned models does not degrade performance in either domain for larger models. RLHF improves programming ability on HumanEval for larger models.
*   **Online Training Efficacy:** Iterated online RLHF significantly improved models as evaluated by crowdworkers (higher Elo scores) and improved the dataset quality (filling out the upper tail of the PM score distribution). A controlled experiment showed online training improved performance even with equal dataset size and hyperparameters.
*   **Honesty:** RLHF training significantly improves performance on TruthfulQA (MC1) for large models.
*   **Bias:** RLHF-trained models tend to have much more positive sentiment than plain LMs across racial and religious groups. Large RLHF models exhibit gender biases similar to language models evaluated at a lower temperature.

### Stated Limitations:

*   **Honesty/Truthfulness:** The paper does not explicitly focus on honesty/truthfulness, believing other techniques might be more efficient. Crowdworkers were not expected to fact-check models significantly.
*   **Harmlessness Data Collection:** The choice to have crowdworkers select the *more harmful* response during red-teaming made it difficult for models to learn sophisticated "hostage negotiation" behaviors (explaining why a request is harmful and dissuading the user), instead favoring simple refusals. This created an imbalance where harmlessness was easily over-optimized.
*   **Crowdworker Distribution:** The crowdworker distribution was not held fixed, and quality likely improved over time, potentially complicating the evaluation of "online training."
*   **PM Robustness Failures:** PMs are not adversarially robust; they can be fooled by subtly inaccurate but well-written responses, especially those out-of-distribution from model-generated samples.
*   **RLHF Overfitting:** RLHF becomes gradually less robust at higher PM scores, and policies can be over-optimized on the train PM, leading to divergence between train and test PM scores.
*   **Evaluation Limitations:**
    *   NLP evaluation format (explicit choices for multiple-choice questions) can lead to misleading "grok" curve appearances.
    *   Sentiment analysis model limitations (e.g., low scores for "Atheist" group due to neutral descriptions).
    *   BBQ-Lite results were inconclusive, suggesting limitations in the measurement rather than model bias.
    *   Gender bias evaluations were challenging due to RLHF models generating fewer gendered terms, making measurements noisy.
*   **Online Training Diversity:** RLHF tends to decrease policy entropy, which could limit the diversity of data collected through the online procedure.
*   **Small Model Performance:** RLHF training hurts performance for smaller models on NLP evaluations and code generation. Smaller models were also more difficult to stabilize during RL training.
*   **Code Performance:** Improvements in code performance from RLHF are modest; simply prompting a base code model performs slightly better.
