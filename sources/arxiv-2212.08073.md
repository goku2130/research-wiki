---
id: arxiv:2212.08073
type: paper
title: 'Constitutional AI: Harmlessness from AI Feedback'
url: https://arxiv.org/abs/2212.08073
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

**Core Problem**
Training large language models to be helpful, honest, and harmless (HHH) traditionally relies on Reinforcement Learning from Human Feedback (RLHF), which demands tens of thousands of human preference labels. This approach is resource-intensive, slow to iterate, and frequently produces evasive assistants that refuse to engage with sensitive topics. The core challenge is to scale AI supervision by replacing human harmlessness labels with AI-generated feedback, while maintaining transparency, reducing evasiveness, and encoding behavioral constraints in a legible, principled format.

**Methodology**
Constitutional AI (CAI) implements a two-stage training pipeline governed by a "constitution"—a short list of natural language principles. 
*Supervised Learning (SL) Phase:* The process begins with a helpful RLHF model. For each red-teaming prompt, the model generates an initial response. The model is then prompted to critique its own output based on a randomly sampled constitutional principle, followed by a revision request to remove harmful content. This critique-revision cycle is repeated sequentially. The model is subsequently finetuned on these revised responses to produce the SL-CAI checkpoint.
*Reinforcement Learning (RL) Phase:* The SL-CAI model generates response pairs for each prompt. A separate feedback model evaluates the pairs via a multiple-choice format guided by constitutional principles. To improve evaluation accuracy, Chain-of-Thought (CoT) prompting is employed. The feedback model computes log-probabilities for each option, which are normalized to serve as soft preference targets for training a hybrid preference model (PM). This PM, trained on AI-generated harmlessness labels and human helpfulness labels, provides the reward signal for PPO-style RL, yielding the final RL-CAI policy.

**Key Formulations and Probabilistic Targets**
The preference modeling stage replaces human labels with model-generated comparisons. Given two responses $A$ and $B$, the feedback model computes their log-probabilities. These are normalized to produce soft preference targets:
$$ \hat{y}_A = \frac{\exp(\log p(A))}{\exp(\log p(A)) + \exp(\log p(B))} $$
When CoT is used, the model commits strongly to one option, yielding poorly calibrated probabilities near 0 or 1. To mitigate this, probabilities are clamped to the $[0.4, 0.6]$ range before training the PM. Ensembling over 16 constitutional principles during label generation further stabilizes PM scores.

**Quantitative Results**
The methodology was evaluated on 52B-parameter models. The training dataset comprised 182,831 red-teaming prompts (42,496 human-written, 140,335 model-generated) and 135,296 helpfulness prompts. SL-CAI was trained for one epoch with a constant learning rate of 0.5 relative to pre-training and a batch size of 1,024 sequences. Crowdworker AB tests collected 10,274 helpfulness and 8,135 harmlessness comparisons. Evaluations show that language models achieve over 90% binary accuracy in identifying HHH responses, with CoT significantly boosting performance. Harmlessness preference scores improve monotonically with each successive critique-revision step. Notably, varying the number of constitutional principles does not significantly alter harmlessness scores but increases response diversity, which aids RL exploration. RL-CAI models consistently outperform RLHF baselines in harmlessness Elo scores without sacrificing helpfulness, achieving virtually non-evasive responses that explain objections rather than refusing queries. Absolute harmfulness scores (0–4 scale) decrease progressively across RL-CAI training steps.

**Limitations**
The approach retains human feedback for helpfulness, limiting full autonomy. The 16 constitutional principles were selected ad hoc and require future refinement by stakeholders. AI supervision risks obscuring decision-making or automating harmful behavior if the supervisor model is misaligned. RL-CAI is susceptible to Goodharting, where over-optimization yields overly harsh, preachy, or boilerplate responses. CoT-generated preference labels are poorly calibrated, necessitating probability clamping. Additionally, absolute harmfulness metrics may be biased by worker subjectivity, and critiques sometimes contain inaccurate or overstated criticisms, though revisions generally mitigate initial harms. Crowdworker instructions were modified to penalize evasiveness, which explains divergent Elo trends compared to prior RLHF baselines.
