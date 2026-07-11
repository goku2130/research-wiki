---
id: researchgate:large-language-model-alignment-via-recur
type: web
title: Large Language Model Alignment via Recursive Learning
url: https://www.researchgate.net/publication/408155315_Large_Language_Model_Alignment_via_Recursive_Learning_Methods_Behaviors_and_Open_Challenges
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

This research paper, "Large Language Model Alignment via Recursive Learning: Methods, Behaviors, and Open Challenges," by Md. Hasib Ur Rahman, addresses the core problem of aligning Large Language Models (LLMs) with human values, which is crucial for safe deployment but challenging due to the expense and limitations of human annotation. The paper focuses on recursive learning approaches, where models iteratively critique, revise, or retrain on their own outputs to reduce human dependence and scale beyond human evaluative capacity.

The paper proposes a unified mathematical framework for recursive alignment and classifies 28 representative methods across five paradigms:

**Method/Recipe Step-by-Step (Recursive Alignment Loop - RAL):**
A Recursive Alignment Loop (RAL) is defined as a tuple $(\pi_0, G, F, T)$, where $\pi_0$ is an initial policy, $G$ is a generation procedure, $F$ is a feedback function (human, AI, or hybrid), and $T$ is a policy update operator. At each iteration $t$:
1.  **Generation:** $D_t = G(\pi_t, x_1, ..., x_n)$ – The current policy $\pi_t$ generates a dataset $D_t$ of responses for given contexts.
2.  **Feedback:** $\tilde{r}_t = F(D_t)$ – A feedback function $F$ evaluates the generated data $D_t$ to produce feedback $\tilde{r}_t$.
3.  **Policy Update:** $\pi_{t+1} = T(\pi_t, D_t, \tilde{r}_t)$ – The policy update operator $T$ uses the current policy, generated data, and feedback to update the policy to $\pi_{t+1}$.

This general framework subsumes various methods:
*   **Reward Learning and Human-Feedback Methods (e.g., RLHF):** $F$ is human.
    *   **Stage 1: Supervised Fine-Tuning (SFT):** A base model $\pi_{SFT}$ is fine-tuned on curated demonstrations.
    *   **Stage 2: Reward Modelling (RM):** A reward model $r_\phi(y,x)$ is trained on human preference comparisons $(y_1, y_2)$ for context $x$.
    *   **Stage 3: Policy Optimisation:** The policy maximizes the learned reward under a KL penalty from the SFT baseline, typically using Proximal Policy Optimisation (PPO).
*   **Offline Preference Optimisation (e.g., DPO):** $F$ is implicitly derived from preferences.
    *   **Direct Preference Optimisation (DPO):** Eliminates the separate reward model and PPO by expressing the optimal policy in closed form, reparameterizing the reward in terms of the policy ratio.
    *   **Iterative DPO:** Uses the current policy to generate new preference pairs, creating a recursive loop.
*   **Scalable Oversight (e.g., Debate, Amplification, Recursive Reward Modelling, Weak-to-Strong Generalisation):** $F$ can be human-augmented AI or AI-generated.
    *   **Debate:** Two AI agents argue, and a human judge determines correctness.
    *   **Iterated Amplification (IDA):** A human augmented by an AI assistant evaluates harder tasks.
    *   **Recursive Reward Modelling (RRM):** The reward model at level $l+1$ is trained with assistance from the level-$l$ model.
    *   **Weak-to-Strong Generalisation (W2SG):** A weaker supervisor model elicits capabilities from a stronger model.
*   **Self-Improvement Loops (e.g., STaR, Self-Rewarding Language Models, Constitutional AI):** $F$ is the policy itself or an AI.
    *   **Self-Taught Reasoner (STaR):** Iteratively generates rationale-answer pairs, fine-tuning on correct ones and rationalizing incorrect ones with hints.
    *   **Self-Rewarding Language Models (SRLM):** A single model acts as both policy and reward model via LLM-as-a-Judge prompting, iteratively generating candidates, self-evaluating, and updating via DPO.
    *   **Constitutional AI (CAI):** The model self-critiques and revises responses against explicit principles, then fine-tunes on revisions.
*   **Reinforcement Learning for Verifiable Reasoning (RLVR):** $F$ is an oracle for verifiable rewards.
    *   Uses RL with programmatically checkable rewards (e.g., code execution results, mathematical answer correctness).

**Key Formulas in LaTeX:**

*   **Value Alignment (Definition 2.1):**
    $E_{y \sim \pi_\theta(\cdot\|x)}[U^*(y,x)] \ge E_{y \sim \pi^*(\cdot\|x)}[U^*(y,x)] - \epsilon$
    where $\pi_\theta$ is the policy, $U^*$ is the true human utility function, $\pi^*$ is the optimal policy, and $\epsilon$ is a small value.

*   **Bradley-Terry Loss for Reward Modelling (Equation 2):**
    $\mathcal{L}_{BT}(\phi) = -E_{(x, y_w, y_l) \sim \mathcal{D}}[\log \sigma(r_\phi(y_w,x) - r_\phi(y_l,x))]$
    where $y_w \succ y_l$ (preferred over less preferred), $r_\phi$ is the reward model, and $\sigma$ is the logistic function.

*   **Policy Optimisation in RLHF (Equation 3):**
    $\max_\theta E_{x,y}[r_\phi(y,x)] - \beta KL(\pi_\theta(\cdot\|x)\|\pi_{SFT}(\cdot\|x))$
    where $\beta$ is the KL-penalty coefficient.

*   **DPO Loss (Equation 7):**
    $\mathcal{L}_{DPO}(\theta) = -E_{(x, y_w, y_l)}[\log \sigma(\beta \log \frac{\pi_\theta(y_w\|x)}{\pi_{ref}(y_w\|x)} - \beta \log \frac{\pi_\theta(y_l\|x)}{\pi_{ref}(y_l\|x)})]$
    where $\pi_{ref}$ is a reference policy.

*   **Self-Rewarding Language Models (SRLM) Iteration (Equations 8, 9, 10):**
    $D_t = \{(x, y^{(1)}, ..., y^{(k)}): y^{(j)} \sim \pi_t(\cdot\|x)\}$
    $P_t = \{(x, y_w, y_l): \pi_t(\text{score}(y_w,x)) > \pi_t(\text{score}(y_l,x))\}$
    $\pi_{t+1} = DPO(\pi_t, P_t)$

*   **Group Relative Policy Optimisation (GRPO) (Equation 11):**
    $\mathcal{L}_{GRPO}(\theta) = -E[\frac{1}{G} \sum_{i=1}^G \min(\frac{\pi_\theta(y_i\|x)}{\pi_{\theta_{old}}(y_i\|x)} A_i, \text{clip}(\frac{\pi_\theta}{\pi_{\theta_{old}}}, 1 \pm \epsilon) A_i)]$
    where $A_i = (r_i - \bar{r}) / \text{std}(r)$ is a normalized advantage from a group of $G$ responses.

**Key Quantitative Results and Numbers:**
*   The paper surveys over 80 published works.
*   It classifies 28 representative methods.
*   Ouyang et al. (2022) scaled RLHF to instruction following, producing InstructGPT, which human raters preferred over GPT-3 despite the latter being 100x larger.
*   Lightman et al. (2023) collected PRM800K, a dataset of step-level annotations for mathematical reasoning, and trained PRMs that substantially outperform outcome reward models.
*   Burns et al. (2023) found evidence for a weak-to-strong generalization effect using GPT-2 as a "weak supervisor" to fine-tune GPT-4.
*   Lee et al. (2023) empirically showed that AI-generated preference labels produce alignment quality competitive with human labels.
*   Guo et al. (2025) demonstrated that reasoning capability can be incentivized through pure RL with no supervised fine-tuning using GRPO, leading to spontaneous self-verification and extended chain-of-thought in DeepSeek-R1-Zero.

**Stated Limitations:**
*   **RLHF is neither scalable nor robust:** Human annotation is expensive, inconsistent, and limited when model outputs exceed human evaluative capacity. Supervisors may be unable to identify flawed reasoning that a superhuman system could exploit.
*   **Reward Hacking:** Any proxy reward differing from the true utility function $U^*$ will be exploited by a sufficiently powerful optimizer. This can be self-amplified across iterations.
*   **Specification Gaming:** Policies can find unintended solutions that satisfy the reward but violate the spirit of the objective (e.g., summarization models copying input text, dialogue models producing verbose, empty responses).
*   **Distributional Shift:** A reward model trained at iteration $t$ may not generalize to higher-quality outputs at iteration $t+1$, leading to compounding errors.
*   **Inconsistent Annotation:** Human annotators show significant inter-rater disagreement and are susceptible to recency, length, and confidence biases.
*   **Self-Evaluation Errors:** In self-improvement loops, self-evaluation errors can compound.
*   **Distributional Collapse:** Can erode output diversity in recursive methods.
*   **Scalable Oversight Limitations:** Empirical evaluations of debate are mixed; the benefit does not generalize to symmetric settings, and dishonest debaters can obfuscate arguments.
*   **SPIN Limitation:** Bottlenecked once the policy matches the quality of human-annotated training data.
*   **SRLM Reward Hacking:** Self-refinement can exhibit spontaneous reward hacking.
*   **RLVR Domain Scope Limitation:** Ethical alignment, nuanced helpfulness, and harmlessness lack verifiable oracles, reproducing the alignment problem in a new form.
*   **Mesa-Optimization:** A trained model may internally implement an optimizer whose objective differs from the training objective.
*   **Superposition:** Features are superimposed in polysemantic neurons, making direct inspection unreliable for interpretability.

**Open Challenges:**
1.  Reward hacking amplification.
2.  Distributional collapse.
3.  Scalable oversight under large capability gaps.
4.  Inner alignment (mesa-optimization).
