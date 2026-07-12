---
id: huggingface:sources-arxiv-2212-08073-md-rl-llm-wiki-
type: web
title: sources/arxiv-2212.08073.md · rl-llm-wiki/knowledge-base ...
url: https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/blob/refs%2Fpr%2F273/sources/arxiv-2212.08073.md
retrieved: '2026-07-12'
maturity: comprehensive
topic: rlaif
---

The paper "Constitutional AI: Harmlessness from AI Feedback" by Bai et al. (2022) introduces Constitutional AI (CAI), a method for training helpful and harmless AI assistants without relying on human labels for harmlessness. Instead, it uses a set of natural language principles, termed a "constitution," and AI feedback. This approach addresses the helpfulness-harmlessness tension and evasiveness observed in prior RLHF models.

**Core Problem:**
The core problem CAI addresses is the need to train AI assistants that are both helpful and harmless, particularly in situations where collecting extensive human labels for harmlessness is impractical or undesirable. Previous methods like HH-RLHF often resulted in evasive responses when faced with harmful prompts, failing to engage with the user's query while still avoiding harm. CAI aims to resolve this by enabling models to explain their objections and provide non-evasive, harmless responses.

**Method/Recipe Step by Step:**
CAI operates in two main stages, bootstrapping from a helpful-only RLHF model:

**Stage 1 — Supervised (SL-CAI): Critique → Revision → SL.**
1.  **Prompt and Sample:** A helpful model is given a "red-team prompt" designed to elicit harmful responses, and a (typically harmful) response is sampled.
2.  **Critique Generation:** A critique request, linked to a randomly selected constitutional principle, is appended to the prompt and response. The model then generates a self-critique.
3.  **Revision Generation:** A revision request is appended, prompting the model to generate a revised, less-harmful response.
4.  **Fine-tuning:** This critique-revision loop is repeated multiple times (e.g., 4 pairs per prompt, with a fresh principle each step). A pretrained model is then fine-tuned on these final revisions, combined with helpfulness samples to maintain helpfulness. This results in the SL-CAI model. The purpose of this stage is to shift the response distribution towards harmlessness efficiently, reducing the exploration needed in the subsequent RL stage.

**Stage 2 — RL from AI Feedback (RL-CAI = RLAIF): AI comparisons → Preference Model → RL.**
1.  **Response Generation:** The SL-CAI model generates two responses for each red-team prompt.
2.  **AI Preference Labeling:** A "feedback model" (a pretrained language model) is presented with the pair of responses as a multiple-choice question, along with a constitutional principle (e.g., "Which response is less harmful?"). The normalized log-probabilities of choosing response (A) versus (B) are used as soft preference labels. These labels are ensembled over multiple constitutional principles for robustness.
3.  **Preference Model Training and RL Fine-tuning:** This AI-generated harmlessness preference data is mixed with human helpfulness preference data. A preference model (PM) is trained on this combined dataset. Finally, the SL-CAI model is fine-tuned using Reinforcement Learning (Proximal Policy Optimization - PPO) against this preference model. The pipeline for RL fine-tuning is identical to standard RLHF, with the key difference being the source of harmlessness labels (AI instead of human).

**Chain-of-thought feedback:** The feedback model can use chain-of-thought reasoning ("Let's think step by step") before making a preference judgment, which improves its ability to discriminate harm. To prevent over-confident labels from CoT leading to extreme RL targets, the probabilities are clamped to a range of 40-60%.

**Key Formulas (Conceptual, as specific mathematical formulas are not provided in the text):**
*   **Preference Model (PM) Training:** The PM is trained to predict preference labels, which are derived from the normalized log-probabilities of the feedback model's choices.
*   **Reinforcement Learning (PPO):** The RL stage uses PPO to fine-tune the language model, maximizing a reward signal provided by the trained preference model. This typically involves optimizing an objective function that balances policy improvement with a KL divergence constraint to prevent large policy shifts.

**Key Quantitative Results and Numbers:**
*   **SL-CAI Training Data:** 16 harmlessness principles were used. 182,831 red-team prompts were used for revisions, supplemented by 135,296 helpfulness samples.
*   **Pareto Improvement:** RL-CAI demonstrates a Pareto improvement over standard RLHF models (helpful-RLHF and HH-RLHF) in terms of harmlessness at a given helpfulness level, as measured by crowdworker Elo scores. RL-CAI with Chain-of-Thought (CoT) pushes harmlessness even further.
*   **AI Harm Identification Scaling:** On 438 HHH binary comparisons, larger models combined with CoT achieved performance comparable to human-feedback-trained PMs (>90% accuracy on an easy set), indicating that "models can now help supervise models."
*   **Evasiveness Reduction:** RL-CAI is "virtually never evasive," resolving the helpful-vs-harmless tension by engaging with prompts and explaining objections, unlike prior HH-RLHF models.
*   **Critiques and Revisions:** Revisions monotonically improve harmlessness. Critiques are beneficial for smaller models and roughly equivalent for larger ones.
*   **Soft vs. Hard Labels:** Soft labels (probabilities) are superior to hard labels (binary choices) when CoT is not used.
*   **CoT Label Clamping:** Probabilities from CoT feedback were clamped to 40-60% to ensure better-calibrated, less-extreme RL targets.

**Stated Limitations:**
*   **Model Scope:** The results are based on 52B-class Anthropic models from 2022 and primarily focus on harmlessness; helpfulness still relies on human labels. CAI is "no human labels *for harm*," not "no human supervision" entirely.
*   **Ad Hoc Principles:** The 16 constitutional principles were chosen "in a fairly ad hoc way for research purposes" and are not a validated constitution. Outcomes may be sensitive to the specific wording of these principles.
*   **Calibration Intervention:** The clamping of CoT probabilities to 40-60% is a calibration intervention, not an inherent property of AI feedback, suggesting a need for careful tuning.
*   **Goodharting:** AI feedback still leads to a proxy reward that can be Goodharted, meaning over-optimization can result in undesirable behaviors like overly harsh responses or boilerplate text ("you are valid, valued, and cared for"). This indicates that AI feedback relocates, but does not eliminate, the problem of reward misspecification.
*   **Dual-Use Potential:** The authors acknowledge that by lowering the barrier to human supervision, CAI could also facilitate the training of pernicious systems or the deployment of under-observed models.
*   **RL Specifics:** The novelty of CAI lies in the label source (AI + constitution) and the critique-revision SL stage, not in the underlying RL optimizer (PPO) or the general RLHF pipeline, which are adopted from prior work.
