---
id: arxiv:2212.08073
type: paper
title: 'Constitutional AI: Harmlessness from AI Feedback (RLAIF)'
url: https://arxiv.org/abs/2212.08073
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

# Constitutional AI: Harmlessness from AI Feedback (RLAIF)

### Core Problem
The authors address the challenge of training AI assistants to be both helpful and harmless without relying on massive datasets of human labels for harmlessness. A primary issue identified in prior Reinforcement Learning from Human Feedback (RLHF) is the tension between helpfulness and harmlessness: models trained for harmlessness often become "evasive," refusing to engage with sensitive but non-harmful queries. The goal of Constitutional AI (CAI) is to produce a harmless, non-evasive assistant that explains its objections to harmful requests, using a small set of human-written principles (a "constitution") to steer behavior.

### Method
The CAI process consists of two primary stages: a supervised learning phase and a reinforcement learning phase.

#### Stage 1: Supervised Learning (SL)
This stage aims to move the model "on-distribution" to reduce exploration needs during RL.
1. **Initial Sampling:** A helpful-only RLHF model generates responses to "red teaming" prompts designed to elicit harmful behavior.
2. **Critique:** The model is prompted to critique its own response based on a randomly sampled principle from the constitution (e.g., "Identify specific ways in which the assistant's last response is harmful...").
3. **Revision:** The model rewrites the response to remove the identified harmful content.
4. **Iteration:** This critique-revision cycle is repeated multiple times.
5. **Finetuning:** A pretrained language model is finetuned via supervised learning on the final revised responses, mixed with responses to helpfulness prompts to maintain utility.

#### Stage 2: Reinforcement Learning from AI Feedback (RLAIF)
This stage refines the model using a preference model (PM) derived from AI evaluations.
1. **AI Comparison:** The SL-CAI model generates pairs of responses to prompts. A separate "feedback model" (a pretrained LM) evaluates these pairs as a multiple-choice question based on constitutional principles.
2. **Label Generation:** The feedback model's preference is captured via the log probabilities of the options (A) and (B).
3. **Chain-of-Thought (CoT):** To improve accuracy, the feedback model can be prompted to "think step-by-step" before choosing the better response.
4. **Preference Model Training:** A PM is trained on a dataset combining human labels for helpfulness and AI-generated labels for harmlessness.
5. **RL Finetuning:** The SL-CAI model is finetuned via RL using the PM as the reward signal.

### Key Technical Details
The feedback model's preference is determined by the normalized log probabilities of the multiple-choice answers. For CoT-based feedback, the authors found that probabilities were often over-confident (near 0 or 1); to mitigate this, they clamped the probabilities to a range of $40\%–60\%$ to produce more robust behavior.

### Quantitative Results
* **Performance:** RL-CAI models were preferred by crowdworkers over previous RLHF models for harmlessness. They achieved higher harmlessness Elo scores while remaining more helpful than the SL-CAI baseline.
* **AI Supervision Capability:** Pretrained models using CoT reasoning showed significant improvements in binary accuracy when identifying the more harmless response, suggesting that models larger than 52B parameters can be competitive with human-trained preference models.
* **Data Scale:** The SL phase utilized 182,831 red teaming prompts and 135,296 helpfulness prompts. The RL phase used 182,831 AI-generated harmlessness comparisons and 135,296 human helpfulness comparisons.
* **Absolute Harmfulness:** On a 0–4 scale (where 0 is least harmful), RL-CAI and RL-CAI CoT models showed a progressive decrease in absolute harmfulness during training.

### Limitations
* **Goodharting:** Over-training can lead to "Goodharting" behavior, where the model becomes overly harsh or generates repetitive boilerplate language (e.g., "You are valid, valued, and cared for").
* **Critique Accuracy:** The authors noted that model-generated critiques were sometimes inaccurate or overstated, although the resulting revisions were still generally more harmless.
* **Human Dependency:** While harmlessness labels were automated, the system still relied on human feedback labels for helpfulness and instruction-following.
