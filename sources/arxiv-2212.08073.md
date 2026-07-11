---
id: arxiv:2212.08073
type: paper
title: 'Constitutional AI: Harmlessness from AI Feedback'
url: https://arxiv.org/abs/2212.08073
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

**Core Problem**
Training AI assistants to be helpful, honest, and harmless (HHH) traditionally relies on Reinforcement Learning from Human Feedback (RLHF), which requires tens of thousands of human preference labels. This approach is resource-intensive, opaque, and creates a tension between helpfulness and harmlessness, often causing models to adopt evasive, non-interactive refusal patterns. The core problem addressed is the need to scale AI supervision by replacing human harmlessness labels with AI-generated feedback, while maintaining transparency, reducing evasiveness, and enabling precise behavioral control through a minimal set of explicit rules.

**Methodology**
The Constitutional AI (CAI) framework operates in two sequential stages, guided by a "constitution" comprising a short list of natural language principles. 
*Supervised Learning Stage (SL-CAI):* Starting with a helpful RLHF model, the system samples red-teaming prompts and generates initial responses. It then appends a critique request based on a randomly sampled constitutional principle, samples a critique, and appends a revision request to generate a revised response. This critique-revision loop is repeated sequentially, randomly sampling principles at each step. The model is then finetuned via supervised learning on the final revised responses, alongside helpfulness prompts, to bootstrap harmless behavior without human harmlessness labels.
*Reinforcement Learning Stage (RL-CAI/RLAIF):* The finetuned SL-CAI model generates pairs of responses to prompts. A feedback model evaluates each pair as a multiple-choice question guided by a constitutional principle. The system computes the log probability of each response and uses the normalized probabilities as soft preference targets. These AI-generated harmlessness labels are mixed with human helpfulness labels to train a hybrid Preference Model (PM). Finally, the SL-CAI policy is optimized via reinforcement learning using the PM as the reward signal, a process termed RL from AI Feedback (RLAIF). Chain-of-thought (CoT) prompting can be applied during evaluation to improve reasoning and transparency, with probabilities clamped to the 40–60% range to prevent overconfident targets.

**Key Quantitative Results**
Experiments utilized 52B-parameter models. The SL stage trained on 182,831 red-teaming prompts (42,496 human-written, 140,335 model-generated) and 135,296 human-written helpfulness prompts, using a batch size of 1,024 sequences and a learning rate of 0.5 relative to pre-training for one epoch. The RL stage incorporated 491,142 red-teaming and 474,300 helpfulness prompts. Evaluation across 10,274 helpfulness and 8,135 comparison tests demonstrated that RL-CAI models significantly outperform both RLHF and SL-CAI baselines in harmlessness while maintaining helpfulness. Crucially, RL-CAI models virtually eliminate evasive responses, engaging with sensitive queries transparently rather than shutting down. CoT prompting further improves evaluation accuracy and harmlessness scores, though it slightly reduces helpfulness. Scaling analysis indicates that increasing the number of revisions monotonically improves harmlessness preference scores, while varying the number of constitutional principles does not significantly alter scores but improves response diversity for RL exploration.

**Stated Limitations**
The authors acknowledge several constraints. The 16 constitutional principles were selected ad hoc for research and require future refinement by stakeholders and adaptation to specific deployment contexts. The method still relies on human feedback for helpfulness, leaving full self-supervision as future work. RL-CAI is susceptible to over-training, resulting in "Goodharting" where models produce overly harsh or boilerplate responses (e.g., repetitive affirmations). Additionally, absolute harmfulness scores may lack calibration due to annotator bias, and AI critiques can sometimes be inaccurate or overstated, though revisions still yield net harmlessness improvements. Finally, while AI supervision scales oversight, it risks automating and obscuring decision-making, necessitating careful transparency measures.
