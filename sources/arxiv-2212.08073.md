---
id: arxiv:2212.08073
type: paper
title: 'Constitutional AI: Harmlessness from AI Feedback (Related Self-Improvement)'
url: https://arxiv.org/abs/2212.08073
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# Constitutional AI: Harmlessness from AI Feedback

### Core Problem
The primary challenge addressed is the reliance on massive amounts of human-labeled data to train AI assistants to be "helpful, honest, and harmless" (HHH). Previous Reinforcement Learning from Human Feedback (RLHF) methods often created a tension between helpfulness and harmlessness, frequently resulting in "evasive" models that refused to engage with sensitive or controversial topics even when a harmless response was possible. The authors seek a method to scale supervision by using AI to supervise other AIs, guided by a transparent set of natural language principles (a "constitution"), to create a harmless but non-evasive assistant.

### Method: The Constitutional AI (CAI) Recipe
The CAI process consists of two distinct stages: a supervised learning (SL) phase to establish a baseline distribution and a reinforcement learning (RL) phase to refine performance.

#### Stage 1: Supervised Learning (Critique $\rightarrow$ Revision $\rightarrow$ SL)
1. **Initial Sampling:** A helpful-only RLHF model generates initial responses to prompts designed to elicit harmful behavior (red-teaming prompts).
2. **Critique:** The model is prompted to critique its own response based on a randomly sampled principle from the constitution (e.g., "Identify specific ways in which the assistant's last response is harmful...").
3. **Revision:** The model rewrites the response to remove the identified harms based on the critique.
4. **Iteration:** This critique-revision cycle is repeated multiple times.
5. **Finetuning:** A pretrained language model is finetuned via supervised learning on the final revised responses, mixed with responses to helpfulness prompts to maintain utility.

#### Stage 2: Reinforcement Learning from AI Feedback (RLAIF)
1. **Preference Dataset Generation:** The SL-CAI model generates pairs of responses to harmful prompts.
2. **AI Evaluation:** A separate "feedback model" (a pretrained LM) evaluates the pairs as a multiple-choice question, choosing the better response based on a constitutional principle.
3. **Preference Model (PM) Training:** A PM is trained on these AI-generated harmlessness labels, combined with human-generated helpfulness labels.
4. **RL Finetuning:** The SL-CAI model is finetuned using RL, using the PM as the reward signal.

**Chain-of-Thought (CoT) Enhancement:** In the RL stage, the feedback model can be prompted to "think step-by-step" before choosing a preference. To prevent the model from becoming over-confident (probabilities near 0 or 1), the authors clamped CoT probabilities to a range of $40\% \text{--} 60\%$.

### Key Quantitative Results
*   **Model Scale:** Experiments primarily utilized models up to 52B parameters.
*   **Data Volume:** The SL stage used 182,831 harmful prompts (42,496 human-written, 140,335 model-generated) and 135,296 human-written helpfulness prompts.
*   **Performance:** RL-CAI models were preferred by crowdworkers over previous human-feedback-trained models. They achieved higher harmlessness Elo scores while remaining significantly less evasive.
*   **Supervision Accuracy:** The authors found that as model capabilities increase, AI identification of harms improves. Models larger than 52B are projected to be competitive with preference models trained on human feedback.
*   **Absolute Harmfulness:** On a 0–4 scale (where 4 is most harmful), RL-CAI and RL-CAI CoT models showed a progressive decrease in absolute harmfulness during training.

### Stated Limitations
*   **Goodharting:** Over-training the RL phase can lead to "Goodharting" behavior, where the model becomes overly harsh or produces repetitive boilerplate language (e.g., "You are valid, valued, and cared for").
*   **Critique Accuracy:** The authors noted that model-generated critiques were sometimes inaccurate or overstated the harms in the original response.
*   **Calibration:** Absolute harmfulness scores may not be perfectly calibrated due to individual human worker biases.
*   **Dual Use:** The authors acknowledge that by lowering the barrier to training aligned models without massive human datasets, the method could potentially be used to train pernicious systems.
