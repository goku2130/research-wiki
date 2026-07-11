---
id: arxiv:2507.04136
type: paper
title: A Technical Survey of Reinforcement Learning Techniques for Large Language
  Models
url: https://arxiv.org/html/2507.04136v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

The provided text is a technical survey of Reinforcement Learning (RL) techniques for Large Language Models (LLMs).

**Core Problem:**
The core problem addressed is the "alignment problem" in Large Language Models (LLMs). While LLMs have grown significantly in scale and capability (e.g., GPT-3 with 175 billion parameters, LLaMA 3.1 with 405 billion, DeepSeek-V3 with 671 billion), their pre-training objective (minimizing negative log-likelihood of the next token) prioritizes statistical probability over factual correctness, ethical safety, or precise instruction following. This leads to issues like hallucinations, generation of harmful content, and failure to follow complex instructions. Reinforcement Learning is proposed as the critical methodological framework to bridge this "Alignment-Optimization Gap" by enabling LLMs to optimize for complex, multi-faceted objectives beyond static text mimicry.

**Method/Recipe Step by Step:**

The survey details several RL techniques adapted for LLMs, categorized into "Alignment" and "Reasoning" objectives.

**1. Reinforcement Learning from Human Feedback (RLHF):**
*   **Stage 1: Supervised Fine-Tuning (SFT):** A pre-trained LLM is fine-tuned on a dataset of human-written demonstrations of desired behavior to provide a coherent baseline.
    *   **Objective:** Maximize the likelihood of producing desired output.
    *   **Loss Function:** $\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}} [\log p_\theta(y|x)]$
*   **Stage 2: Reward Model (RM) Training:** A separate reward model ($r_\phi$) is trained to predict human preferences between different model outputs. Human annotators provide preference judgments (which response is preferred).
    *   **Objective:** Minimize the loss to reflect human preferences.
    *   **Loss Function:** $\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}_{\text{pref}}} [\log \sigma(r_\phi(x,y_w) - r_\phi(x,y_l))]$
*   **Stage 3: Reinforcement Learning Optimization:** The SFT model (now the policy model) is fine-tuned using an RL algorithm (typically PPO) to maximize the reward predicted by the RM. A KL-divergence penalty is added to prevent deviation from the original SFT model.
    *   **Objective:** Maximize PPO objective while constraining policy changes.
    *   **Loss Function:** $\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{PPO}}(\theta) - \beta \mathcal{D}_{\text{KL}}(p_\theta(\cdot|x) || p_{\text{SFT}}(\cdot|x))$
    *   **PPO Clipped Surrogate Objective:** $\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$
    *   **Probability Ratio:** $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$
    *   **KL Divergence:** $\mathcal{D}_{\text{KL}}(P \parallel Q) = \sum_x P(x) \log \left( \frac{P(x)}{Q(x)} \right)$

**2. Reinforcement Learning from AI Feedback (RLAIF):**
*   Mirrors RLHF but replaces human preference collection with AI-driven evaluations (e.g., GPT-4, specialized classifiers).
*   AI systems provide preference scores or judgments to train the reward model.
*   Policy optimization uses PPO based on the AI-generated reward model.

**3. Constitutional AI:**
*   A structured variant of RLAIF where feedback is guided by explicit principles (a "constitution").
*   **Phase 1: Supervised Learning (SL-CA):** Model self-critiques and revises its outputs against constitutional principles. Revised responses serve as positive examples for fine-tuning.
*   **Phase 2: Reinforcement Learning (RL-CA):** Model generates response pairs, and a feedback model (guided by the constitution) assigns preferences, used for scalable optimization (e.g., PPO).
*   Employs "red-teaming" to proactively uncover and mitigate failure modes.

**4. Direct Preference Optimization (DPO):**
*   Simplifies RLHF by directly optimizing the policy using preference pairs, bypassing explicit reward modeling and reinforcement learning stages.
*   **Core Insight:** Optimal policy under a given reward function can be expressed in terms of a reference policy.
*   **Implicit Reward:** $r(x, y) = \frac{1}{\beta} \log \frac{p_{\theta}(y|x)}{p_{\text{ref}}(y|x)} + Z(x)$
*   **Loss Function:** $\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[ \log \sigma \left( \beta \log \frac{p_{\theta}(y_w|x)}{p_{\text{ref}}(y_w|x)} - \beta \log \frac{p_{\theta}(y_l|x)}{p_{\text{ref}}(y_l|x)} \right) \right]$

**5. Unified Alignment (UNA):**
*   Generalizes various alignment methods (PPO, DPO, KTO) under a single supervised regression objective.
*   Treats the policy as a parameterized reward estimator.
*   **Implicit Reward:** $\hat{r}_\theta(x, y)$ is defined as the scaled log-likelihood ratio between the current policy and the reference policy $\pi_{\text{ref}}$.
*   **Generalized Loss Function:** $\mathcal{L}_{\text{UNA}}(\theta) = \mathbb{E}_{(x,y,z)\sim\mathcal{D}} \left[ \ell \left( \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}, z \right) \right]$
    *   $\ell(\cdot, \cdot)$ is a loss function (e.g., MSE for scalar scores, BCE for binary labels, logistic margin for pairwise preferences).

**6. Outcome-Based Reinforcement Learning for Reasoning (OB-RL):**
*   A sparse-reward framework that optimizes for functional success by rewarding only the correctness of the final answer.
*   **Objective:** Maximize the reward for correct final answers.
*   **Loss Function:** $\mathcal{L}_{\text{OB-RL}}(\theta) = -\mathbb{E}_{(x,\tau)\sim\pi_\theta}[R(\tau) \log \pi_\theta(\tau \mid x)]$
    *   $R(\tau)$ is a scalar reward (e.g., binary success/failure) from an automated verifier.

**7. Chain-of-Thought Reward Optimization (CoT-RO):**
*   A dense-reward framework that scores each intermediate step in a chain of thought.
*   **Objective:** Maximize the discounted return over the entire trajectory with step-wise rewards.
*   **Loss Function:** $\mathcal{L}_{\text{CoT-RO}}(\theta) = -\mathbb{E}_{(x,\tau)\sim\pi_\theta} \left[ \sum_{t=1}^T \gamma^{t-1} r_t \log \pi_\theta(y_t \mid x, y_{<t}) \right]$
    *   $r_t$ is an immediate reward from a lightweight evaluator (e.g., symbolic verifier).

**8. Verifier-Guided Reinforcement Learning (V-RL):**
*   Augments an LLM's policy with an external verifier ($V_\phi$) that continuously evaluates candidate outputs and supplies the reward signal.
*   **Objective:** Maximize the reward assigned by the verifier.
*   **Loss Function:** $\mathcal{L}_{\text{V-RL}}(\theta) = -\mathbb{E}_{(x,\tau)\sim\pi_\theta}[V_\phi(x, \tau) \log \pi_\theta(\tau \mid x)]$
    *   $V_\phi$ can be a learned model, heuristics, or a specialized tool.

**9. Proximal Policy Optimization (PPO):**
*   A general RL algorithm used in RLHF and RLAIF.
*   **Objective:** Maximize expected returns while constraining policy changes.
*   **Clipped Surrogate Objective:** $\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$
    *   $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ (probability ratio), $\hat{A}_t$ = estimated advantage, $\epsilon$ = clip parameter.

**10. Q-Learning and Off-Policy RL (e.g., ILQL, VerifierQ):**
*   Address sample-efficiency limitations by learning from static datasets.
*   **Implicit Language Q-Learning (ILQL):** Offline RL algorithm leveraging Q-learning on static datasets.
    *   **Training Objective:** $\mathcal{L}_{\text{ILQL}}(\theta) = \mathcal{L}_{\text{TD}}(\theta) + \alpha \cdot \mathcal{L}_{\text{conservatism}}(\theta)$
*   **VerifierQ:** Employs offline Q-learning to train a verifier for multi-step reasoning.
    *   **Loss Function:** $\mathcal{L}_{\text{VerifierQ}}(\theta) = \mathcal{L}_{\text{Bellman}}(\theta) + \beta \cdot \mathcal{L}_{\text{CQL}}(\theta)$
*   **Conservative Q-Learning (CQL):** Stabilizes training and mitigates distributional shift.
    *   **General Formulation:** $\mathcal{L}_{\text{CQL}}(\theta) = \mathcal{L}_{\text{Bellman}}(\theta) + \alpha \cdot \left( \mathbb{E}_{(s,a)\sim\mu}[Q_\theta(s,a)] - \mathbb{E}_{(s,a)\sim D}[Q_\theta(s,a)] \right)$

**11. Group Relative Policy Optimization (GRPO):**
*   Streamlined alternative to PPO, eliminating the separate value estimator (critic).
*   Estimates relative advantages by normalizing rewards across a group of sampled outputs.
*   **Adapted PPO Clipped Surrogate Objective:** $\mathcal{L}^{\mathrm{GRPO}}(\theta) = \mathbb{E}_{(s, \{a_i\}) \sim \pi_{\theta_{\mathrm{old}}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( r_i(\theta) \hat{A}_i^{\mathrm{GR}}, \text{clip}(r_i(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i^{\mathrm{GR}} \right) \right]$
    *   $r_i(\theta) = \frac{\pi_{\theta}(a_i|s)}{\pi_{\theta_{\mathrm{old}}}(a_i|s)}$
    *   Group-normalized advantage: $\hat{A}_i^{\mathrm{GR}} = \frac{r(a_i) - \mu}{\sigma}$ (where $\mu$ and $\sigma$ are mean and std dev of rewards within the group).

**Key Quantitative Results and Numbers:**
*   LLM parameter counts: GPT-3 (175 billion), LLaMA 3.1 (405 billion), DeepSeek-V3 (671 billion).
*   PPO clip parameter ($\epsilon$): typically 0.1 or 0.2.
*   GRPO group size ($G$): typically $G \ge 64$ to maintain stability.
*   PPO memory footprint: $\mathcal{M}_{\text{total}} \approx \mathcal{M}_\pi + \mathcal{M}_{\text{ref}} + \mathcal{M}_r + \mathcal{M}_V$, approaching $4N$ for models with $N$ parameters.

**Stated Limitations:**
*   **General RL Challenges:** Exploration-exploitation tradeoff, credit assignment problem, sample efficiency (especially for computationally expensive LLMs), stability during learning.
*   **RLHF:**
    *   Scalability limitations due to time and cost of human annotation.
    *   "Reward hacking" (Goodhart's Law): models exploit loopholes in reward functions.
    *   "Alignment Tax": trade-off between alignment fidelity and distributional entropy, leading to "mode-seeking behavior" and reduced policy entropy (pruning diverse answers).
    *   "Alignment Tax" can lead to regressions on some public NLP evaluations.
    *   Excessive safety tuning can induce "exaggerated safety behaviors" (refusing safe prompts).
*   **RLAIF:**
    *   AI evaluators may not fully capture nuanced human values, potentially diminishing feedback quality.
    *   AI evaluators may propagate biases inherent in their own training data.
    *   "Recursive alignment issue" / "model collapse" / "sycophancy loops": if AI evaluators are biased, their flaws can be amplified, leading to systematic bias (deterministic error $\delta(x)$ that doesn't cancel out).
*   **Constitutional AI:**
    *   "Critique-Revision Gap": model's discriminative capability exceeds its generative corrective capacity, especially for models below a critical parameter threshold (e.g., $\approx 50\text{B}$).
    *   "Performative alignment": model learns to parrot safety language without semantic understanding.
    *   "Over-Refusal Gap": tendency to minimize risk by refusing ambiguous queries, leading to excessive caution.
    *   Trade-off between interpretability (explicit rules) and flexibility.
*   **DPO:**
    *   May be less effective at exploring diverse outputs compared to traditional RL.
    *   Quality highly dependent on accuracy and representativeness of human preference data.
    *   Susceptible to length bias and overfitting on noisy datasets due to unbounded implicit reward.
    *   "Catastrophic overfitting" on noisy labels if implicit reward explodes.
    *   "Open gap in robustness to label noise" compared to PPO.
*   **UNA:**
    *   "Exploration deficit": inherent difficulty in generating novel, high-utility reasoning paths using a strictly supervised loss function.
    *   "Functional ceiling": model discouraged from discovering super-human strategies if they exceed target scores in training distribution.
    *   "Support mismatch problem": if reference model assigns negligible probability to a valid high-scoring response, implicit reward estimate explodes.
*   **OB-RL:**
    *   High-variance gradients typical of sparse-reward settings.
    *   Acute credit assignment problem: valid computations wasted if final answer is incorrect.
    *   Slow convergence without additional techniques.
    *   Can produce opaque/convoluted rationales or reasoning shortcuts (correct answer through flawed logic).
    *   "Vanishing gradient regime" in early training due to sparse reward landscape.
    *   "Causal validity" blindness: treats lucky guesses or shortcuts as rigorous proofs.
*   **CoT-RO:**
    *   Computational expense: step-level evaluation can be costly.
    *   Dense feedback may encourage verbose yet shallow reasoning.
    *   Training a reliable Process Reward Model (PRM) often requires expensive human annotation at the step level.
    *   "Bias of proxy rewards": PRM approximation errors (e.g., confusing confident tone with logical validity) can be integrated over trajectory length, leading to "reasoning hacking."
    *   "Oversight scalability gap": difficulty in robustly distinguishing necessary deductions from plausible hallucinations with automated supervision.
*   **V-RL:**
    *   Effectiveness hinges on verifier quality: weak/biased verifier may mis-reward spurious solutions ("proxy gaming").
    *   Overly strict verifier can stifle exploration.
    *   Computational cost: every policy sample must be scored by $V_\phi$, doubling inference cost.
    *   "Verification-Generation Asymmetry": when $V_\phi$ is a learned model, policy optimization can degrade into an "adversarial attack on the verifier" ("Adversarial Goodharting").
    *   "Scalable oversight" remains an open challenge for subjective tasks.
*   **PPO:**
    *   Inherently sample-inefficient, requiring new rollouts for every update step.
    *   "Four-Model Bottleneck": active memory footprint scales as $\mathcal{M}_{\text{total}} \approx \mathcal{M}_\pi + \mathcal{M}_{\text{ref}} + \mathcal{M}_r + \mathcal{M}_V$, approaching $4N$.
    *   Stability hinges on KL-penalty coefficient $\beta$, where lower $\beta$ risks "exponential error amplification (instability)."
*   **Q-Learning and Off-Policy RL:**
    *   Intrinsically limited by the support of the offline dataset.
    *   "Overestimation bias" from Bellman optimality operator, leading to "delusional value estimation" for out-of-distribution tokens.
    *   Learned value is upper-bounded by the quality of the static data.
    *   Better for "alignment constraints" (suppressing undesirable behaviors) than complex reasoning (requires active on-policy exploration).
*   **GRPO:**
    *   "Sample variance": approximating state-value baseline with Monte Carlo mean introduces stochastic noise ($O(1/\sqrt{G})$).
    *   Requires increasing group size $G$ (e.g., $\ge 64$) to maintain stability, converting memory bottleneck into compute bottleneck.
    *   "Singularity in homogeneous reward regimes": if all rewards are same (all 0 or all 1), standard deviation $\sigma \to 0$, causing learning signal to vanish or explode.
