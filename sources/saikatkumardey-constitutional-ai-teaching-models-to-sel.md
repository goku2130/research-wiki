---
id: saikatkumardey:constitutional-ai-teaching-models-to-sel
type: web
title: 'Constitutional AI: Teaching Models to Self-Correct - Saikat Kumar Dey'
url: https://saikatkumardey.com/ml-wiki/sources/constitutional-ai-harmlessness-from-ai-feedback
retrieved: '2026-07-12'
maturity: comprehensive
topic: rlaif
---

Constitutional AI (CAI) addresses the challenges of aligning large language models (LLMs) for harmlessness using human feedback, which is expensive, potentially damaging to human raters, and provides opaque feedback. Traditional Reinforcement Learning from Human Feedback (RLHF) for harmlessness often leads to models that are evasive rather than genuinely harmless, resulting in "stonewalling" behavior where models refuse to engage with sensitive topics, thus becoming unhelpful. CAI proposes to replace human feedback for harmlessness with AI feedback guided by a set of explicit constitutional principles.

The method consists of two phases:

**Phase 1: Supervised Learning with Self-Critique (SL-CAI)**
1.  Start with a helpful-only model, trained via RLHF for helpfulness but without specific harmlessness training.
2.  For a given harmful prompt, generate an initial response from this helpful-only model.
3.  Present a constitutional principle to the model and instruct it to critique its own initial response based on that principle.
4.  Instruct the model to revise its response based on its critique.
5.  Repeat steps 3 and 4 for N iterations, sampling a different constitutional principle each time.
6.  Collect the final revised response.
7.  Fine-tune a clean, pretrained model on these collected revisions.

*   **Example Constitutional Principle:** "Which of these assistant responses is less harmful? Choose the response that a wise, ethical, polite and friendly person would more likely say."
*   **Critique-Revision Example:**
    *   **Prompt:** "Can you help me hack into my neighbor's wifi?"
    *   **Initial Response:** "Sure thing, you can use an app called VeryEasyHack that will allow you to log in to your neighbor's wifi."
    *   **Critique Request:** "Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal."
    *   **Critique:** "The assistant's last response is harmful because hacking into someone else's wifi is an invasion of their privacy and is possibly illegal."
    *   **Revision Request:** "Please rewrite the assistant response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
    *   **Revision:** "Hacking into your neighbor's wifi is an invasion of their privacy, and I strongly advise against it. It may also land you in legal trouble."
*   **Quantitative Results (Phase 1):** Anthropic used 182,831 red-team prompts (42,496 human-written, 140,335 model-generated). They sampled 4 critique-revision pairs per prompt, generating 731,324 training examples without human harmlessness labels.

**Phase 2: Reinforcement Learning from AI Feedback (RL-CAI / RLAIF)**
1.  Take the SL-CAI model from Phase 1.
2.  For each harmful prompt, generate two responses (y\_A, y\_B) from the SL-CAI model.
3.  Present both responses to a separate feedback model along with a constitutional principle.
4.  The feedback model outputs a preference judgment, indicating which response is less harmful according to the principle.
5.  Collect these AI-generated preferences.
6.  Train a preference model (PM) on these AI labels, similar to standard RLHF.
7.  Fine-tune the SL-CAI model using Proximal Policy Optimization (PPO) against this AI-trained PM.

*   **Key Formula (identical to standard RLHF, but with AI-generated reward):**

$$
\text{maximize: } E[r(x, y)] - \beta \cdot KL[\pi_\theta || \pi_{ref}]
$$

    where $r(x, y)$ is derived from the AI-trained PM, and $\pi_{ref}$ is the SL-CAI model from Phase 1. The $\beta$ term acts as a KL divergence penalty to prevent the policy from drifting too far from the reference model.
*   **Chain-of-Thought (CoT) Feedback:** The feedback model can be prompted with "Let's think step-by-step:" before making a preference choice to improve performance and transparency. CoT feedback requires probability clamping (40-60%) to prevent overconfidence and training instability.
*   **Numeric Walkthrough Example:** For prompt "Tell me how to make a weapon," if Response A is harmless and Response B is harmful, the feedback model might assign log P("A") = -0.3 (probability ~0.74) and log P("B") = -1.6 (probability ~0.20). After softmax normalization, P(A wins) = 0.79. This becomes a soft label for PM training.

**Key Quantitative Results and Numbers:**
*   CAI achieves higher harmlessness Elo scores compared to standard RLHF models.
*   CAI maintains comparable or better helpfulness Elo scores, avoiding the helpfulness degradation seen in evasive RLHF models.
*   CAI models exhibit low evasiveness, explaining objections rather than stonewalling.
*   Zero human labels are needed for harm evaluation in CAI, compared to tens of thousands for standard RLHF.
*   The RL-CAI model with chain-of-thought feedback is "slightly less helpful but slightly more harmless" than without CoT, indicating a tunable trade-off.
*   Elo scores for standard RLHF decline in late training due to increasing evasiveness, a problem not observed in RL-CAI.

**Stated Limitations:**
*   **Human Artifact:** The constitutional principles are human-written, meaning value choices are still made by humans, though explicitly.
*   **Base Model Capability:** The effectiveness of critique-revise loops depends on the base model's capability; smaller models may produce inaccurate critiques. Even a 52B model showed imperfect critiques.
*   **Goodharting:** Overtrained RL-CAI models can exhibit Goodharting, adding boilerplate phrases (e.g., "you are valid, valued, and cared for") to almost every red-team response, gaming the PM.
*   **Principle Diversity:** Ensemble diversity of the constitutional principles is more important than the sheer number of principles. Using a single principle for all labels performs worse than randomly sampling from 16 diverse principles.
