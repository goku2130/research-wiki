---
id: arxiv:2212.08073
type: paper
title: 'Constitutional AI: Harmlessness from AI Feedback'
url: https://arxiv.org/abs/2212.08073
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

# Constitutional AI: Harmlessness from AI Feedback

### Core Problem
The authors address the challenge of training AI assistants to be helpful, honest, and harmless (HHH) without relying on massive datasets of human labels for harmlessness. Previous Reinforcement Learning from Human Feedback (RLHF) methods often created a tension between helpfulness and harmlessness, frequently resulting in "evasive" models that refused to engage with controversial but benign queries. The goal of Constitutional AI (CAI) is to scale supervision by using AI to supervise other AIs, guided by a small set of human-defined natural language principles (a "constitution"), thereby reducing human labeling effort and eliminating evasiveness.

### Method
The CAI process consists of two primary stages: a supervised learning (SL) phase and a reinforcement learning (RL) phase.

#### 1. Supervised Stage: Critique $\rightarrow$ Revision $\rightarrow$ SL
This stage aims to move the model "on-distribution" regarding harmlessness:
1. **Initial Sampling:** A helpful-only RLHF model generates responses to "red teaming" prompts designed to elicit harmful behavior.
2. **Critique:** The model is prompted to critique its own response based on a randomly sampled principle from the constitution (e.g., identifying racist or toxic content).
3. **Revision:** The model rewrites the response to remove the identified harms.
4. **Iteration:** This critique-revision loop is applied repeatedly.
5. **Finetuning:** A pretrained language model is finetuned via supervised learning on the final revised responses, mixed with samples from a helpful RLHF model to maintain helpfulness.

#### 2. RL Stage: AI Feedback $\rightarrow$ Preference Model $\rightarrow$ RL
This stage, termed Reinforcement Learning from AI Feedback (RLAIF), refines the model:
1. **Pair Generation:** The SL-CAI model generates two potential responses to a prompt.
2. **AI Evaluation:** A separate "feedback model" (a pretrained LM) evaluates the pair as a multiple-choice question, selecting the better response based on a constitutional principle.
3. **Preference Modeling:** The log probabilities of the feedback model's choices are used as targets to train a Preference Model (PM). This PM is trained on a mixture of AI-generated harmlessness labels and human-generated helpfulness labels.
4. **RL Finetuning:** The SL-CAI model is finetuned via RL using the PM as the reward signal.

**Chain-of-Thought (CoT) Enhancement:** To improve accuracy, the feedback model can be prompted to "think step-by-step" before choosing a response. To prevent over-confidence, the authors clamped CoT probabilities to a range of $40\%-60\%$.

### Key Quantitative Results
*   **Dataset Scale:** The SL stage used 182,831 red teaming prompts (42,496 human-written, 140,335 model-generated) and 135,296 human-written helpfulness prompts.
*   **Performance:** RL-CAI models (specifically the 52B parameter version) were preferred by crowdworkers over HH RLHF models. They achieved higher harmlessness Elo scores while remaining more helpful than SL-CAI models.
*   **Evasiveness:** RL-CAI was found to be "virtually never evasive," providing nuanced explanations for refusals rather than canned responses like "I cannot respond to this content."
*   **Absolute Harmfulness:** On a 0–4 scale (where 4 is most harmful), RL-CAI and RL-CAI with CoT showed a progressive decrease in absolute harmfulness during training.
*   **AI Supervision:** Pretrained LMs using CoT reasoning showed significant improvements in binary accuracy when identifying the more harmless response, approaching the performance of human-trained preference models.

### Limitations
*   **Goodharting:** Over-training can lead to "Goodharting" behavior, where the model becomes overly harsh or adopts repetitive boilerplate language (e.g., "You are valid, valued, and cared for").
*   **Critique Accuracy:** The authors noted that model-generated critiques were sometimes inaccurate or overstated, although the resulting revisions remained generally harmless.
*   **Calibration:** Preference model scores were observed to become less calibrated at higher values.
*   **Dual Use:** The authors acknowledge that reducing the need for human feedback lowers the barrier for creating pernicious systems and may lead to the deployment of models that have not been thoroughly observed by humans.
