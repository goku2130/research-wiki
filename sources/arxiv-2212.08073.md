---
id: arxiv:2212.08073
type: paper
title: 'Constitutional AI: Harmlessness from AI Feedback'
url: https://arxiv.org/abs/2212.08073
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

**Core Problem**
Training AI assistants to be helpful, honest, and harmless (HHH) traditionally relies on reinforcement learning from human feedback (RLHF), which requires tens of thousands of human preference labels. This approach is costly, opaque, and difficult to scale as model capabilities grow. Moreover, human-labeled RLHF creates a tension between helpfulness and harmlessness, frequently causing models to adopt evasive, unhelpful refusal patterns when confronted with sensitive queries. The authors seek to eliminate human labels for harmlessness while preserving helpfulness and transparency.

**Method/Recipe**
The proposed Constitutional AI (CAI) pipeline replaces human harmlessness labels with a "constitution"—a short list of natural language principles—and operates in two sequential stages:
1. **Supervised Learning (SL-CAI):** Starting from a helpful RLHF model, the system generates responses to red-teaming prompts. It then iteratively critiques its own output based on a randomly sampled constitutional principle, followed by a revision step. This critique-revision sequence is repeated multiple times per prompt. The model is subsequently finetuned via supervised learning on the final revised responses, combined with a separate set of helpfulness prompts.
2. **Reinforcement Learning from AI Feedback (RL-CAI):** The SL-CAI model generates response pairs for prompts. A separate feedback model evaluates which response is less harmful using the constitutional principles formatted as multiple-choice questions. These AI-generated harmlessness labels are mixed with human helpfulness labels to train a Preference Model (PM). Finally, the SL-CAI model is optimized via reinforcement learning using the PM as a reward signal. Chain-of-thought (CoT) prompting can be applied to the feedback model to improve evaluation accuracy, with probability targets clamped to prevent overconfidence.

**Key Technical Specifications & Formulations**
The source describes the training algorithmically rather than through explicit loss functions. Key mathematical specifications include sampling at temperature $T = 1$, a constant learning rate of $0.5$ relative to the pre-training learning rate, and a batch size of $1024$ sequences. For the RL stage, preference targets are derived from normalized log-probabilities. When CoT prompting is used, the feedback model's output probabilities are clamped within the interval $[0.40, 0.60]$ to maintain calibration and prevent extreme reward signals that cause over-optimization.

**Quantitative Results**
The study utilized $182,831$ red-teaming prompts ($42,496$ human-written + $140,335$ model-generated) and $135,296$ helpfulness prompts. Evaluation relied on $10,274$ helpfulness and $8,135$ harmlessness crowdworker comparisons. On $52$-billion parameter models, RL-CAI achieved significantly higher harmlessness scores than both standard RLHF and SL-CAI baselines while maintaining comparable helpfulness. Crowdworkers consistently preferred RL-CAI outputs over prior human-feedback-trained models. Absolute harmfulness ratings (on a $0$–$4$ scale) demonstrated a progressive decline across RL-CAI training steps. The authors report that AI evaluators using CoT reasoning approach human-trained preference model accuracy, with models exceeding $52$B parameters predicted to become competitive.

**Stated Limitations**
The authors acknowledge several constraints. The method still requires human feedback for helpfulness, though full automation is a long-term goal. The constitutional principles were selected ad hoc and require future refinement by diverse stakeholders. Over-optimization (Goodharting) can occur, leading to overly preachy, accusatory, or boilerplate responses. Critiques generated during SL-CAI are sometimes inaccurate or overstated, though revisions still mitigate harm. Absolute harmfulness scores may lack calibration due to subjective worker biases. Additionally, the evaluation protocol explicitly penalized evasiveness, which may influence comparative scores relative to prior work. The authors also warn of dual-use risks, noting that reducing human oversight lowers the barrier to training potentially pernicious systems.
