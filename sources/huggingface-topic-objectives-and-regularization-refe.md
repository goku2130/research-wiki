---
id: huggingface:topic-objectives-and-regularization-refe
type: web
title: 'topic: objectives-and-regularization/reference-model-and-kl'
url: https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/commit/a03840159b93b042b5460ac2233c7acf82ca7842
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

The reference-KL regularizer is a universal component in RL-based Large Language Model (LLM) post-training objectives, preventing policy drift and collapse while optimizing for reward.

**Core Problem:**
LLMs trained with Reinforcement Learning (RL) can "over-optimize" for a learned reward signal, leading to nonsensical or collapsed generations. The core problem is to balance maximizing a reward function with maintaining the coherence and quality of the generated text, preventing the policy from diverging too far from a trusted prior.

**Method/Recipe:**

1.  **Objective Formulation:** The standard KL-regularized objective aims to maximize expected reward while penalizing the KL-divergence from a frozen reference policy ($\pi_{\text{ref}}$), typically the Supervised Fine-Tuning (SFT) model.
    *   **Formula:**

$$
\max_{\pi_\theta}\ \mathbb{E}_{x,\,y\sim\pi_\theta}\big[r(x,y)\big]-\beta\, \mathbb{D}_{\mathrm{KL}}\big[\pi_\theta(y\mid x)\,\|\,\pi_{\text{ref}}(y\mid x)\big]
$$

        where $r(x,y)$ is the reward for generating response $y$ given prompt $x$, $\pi_\theta$ is the current policy, $\pi_{\text{ref}}$ is the frozen reference policy, and $\beta$ is the penalty strength.
    *   This objective was introduced for language models by Ziegler et al.

$$
source:arxiv:1909.08593] and used in InstructGPT
$$

source:arxiv:2203.02155].
    *   The penalty is a **reverse KL**, $\mathbb{D}_{\mathrm{KL}}(\pi\|\pi_{\text{ref}})$, which is mode-seeking, meaning the policy concentrates on a subset of the reference's support.

2.  **Origin (KL-control and Entropy Bonus):** The mechanism predates LLMs, originating from KL-control (stochastic optimal control). Jaques et al.'s Sequence Tutor fine-tuned a generator by penalizing KL from a frozen pretrained model treated as a prior

$$
source:arxiv:1611.02796].
    *   **Objective (Sequence Tutor):**
$$

L(q)=\mathbb{E}_{q(\tau)}[r(\tau)]/c-\mathbb{D}_{\mathrm{KL}}\big[q(\tau)\,\|\,p(\tau)\big]

$$
Minimizing KL to the prior automatically supplies an entropy bonus, leading to a high-entropy/stochastic optimal policy, beneficial for diverse generation.

3.  **Three Roles of Reference-KL:**
    *   **Anti-over-optimization anchor:** Prevents the policy from exploiting inaccuracies in the learned reward model, which can lead to "reward-hacking gibberish" if removed
$$

source:arxiv:1909.08593].
    *   **Diversity / anti-mode-collapse:** Through the implicit entropy bonus, it preserves generation diversity and prevents collapse onto a few high-reward strings

$$
source:arxiv:1611.02796, source:arxiv:2305.18290].
    *   **Task definition for style tasks:** For tasks where human judgment involves style or coherence, staying near the fluent reference is part of achieving good performance
$$

source:arxiv:1909.08593].

4.  **Setting the Coefficient ($\beta$):** $\beta$ determines the trade-off between reward and divergence, operating on the KL-vs-reward Pareto frontier

$$
source:arxiv:1909.08593, source:arxiv:2009.01325].
    *   **Adaptive $\beta$ (Ziegler et al.):** Adjusts $\beta$ to hit a target KL using a log-space proportional controller.
        *   **Formula:** $e_t=\mathrm{clip}\big(\tfrac{\mathrm{KL}_t-\mathrm{KL}_{\text{targ}}}{\mathrm{KL}_{\text{targ}}},-0.2,0.2\big)$, $\beta_{t+1}=\beta_t(1+K_\beta e_t)$, with $K_\beta=0.1$. Target KL values mentioned are 10 nats (sentiment) or 6 nats (descriptiveness)
$$

source:arxiv:1909.08593].
    *   **Fixed $\beta$:** A constant $\beta$ value.

5.  **Placement of KL Penalty:**
    *   **In the reward (per-token):** The penalty is folded into the per-token reward, $r_t=r_\phi-\beta\log\frac{\pi}{\pi_{\text{ref}}}$, and seen by the advantage estimator

$$
source:arxiv:2203.02155].
    *   **In the loss:** The term $-\beta\,\mathbb{D}_{\mathrm{KL}}[\pi_\theta\|\pi_{\text{ref}}]$ is added directly to the objective, keeping the advantage estimate separate. GRPO uses an unbiased, always-positive "$k3$" estimator for this term: $\frac{\pi_{\text{ref}}}{\pi_\theta}-\log\frac{\pi_{\text{ref}}}{\pi_\theta}-1$
$$

source:arxiv:2402.03300].

6.  **Reference-free and Implicit KL:**
    *   **DPO's implicit KL:** DPO lacks an online KL term, but its implicit reward $\hat r=\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ incorporates the KL contribution into a supervised loss. $\beta$ acts as the implicit anchor strength, and $\pi_{\text{ref}}$ (or a surrogate) is still required

$$
source:arxiv:2305.18290].

**Key Quantitative Results and Numbers:**

*   **Ziegler et al. (2019):** Adaptive $\beta$ targeting 10 nats (sentiment) or 6 nats (descriptiveness) KL, or fixed $\beta$ values of 0.1/0.03
$$

source:arxiv:1909.08593].
*   **InstructGPT (2022):** Fixed $\beta=0.02$ for per-token KL-in-reward

$$
source:arxiv:2203.02155].
*   **DPO (2023):** Fixed $\beta=0.1$ (or 0.5 for TL;DR tasks) for implicit KL-in-loss
$$

source:arxiv:2305.18290].
*   **GRPO (DeepSeekMath) (2024):** Fixed $\beta=0.04$ for KL-in-loss

$$
source:arxiv:2402.03300].
*   **DeepSeek-R1 (2025):** Fixed $\beta=0.001$ with a verifier reward, placed in loss
$$

source:arxiv:2501.12948].

**Stated Limitations:**

*   **Divergence Choice:** It is an open question whether reverse KL $\mathbb{D}_{\mathrm{KL}}(\pi\|\pi_{\text{ref}})$ is the optimal divergence, as it is mode-seeking. Forward-KL and other f-divergence variants exist but are not yet covered in the corpus.
*   **Cross-Recipe $\beta$ Inference:** The observation that $\beta$ is significantly smaller (e.g., DeepSeek-R1's 0.001 vs. InstructGPT's 0.02) when the reward is more trustworthy (verifier-based vs. preference RM) is an inference across recipes, not a stated result, and is flagged as an open question.
*   **Reference-Free Variants:** Reference-free variants (e.g., SimPO/ORPO), which entirely drop $\pi_{\text{ref}}$, are not yet processed in the corpus, leaving a gap in understanding how much of the anchor's benefit survives without it.
*   **Distinct KLs:** The reference-KL (regularizer) should not be conflated with the old-vs-new-policy KL used by PPO/TRPO for step-size control (clip/trust region). RLHF uses both simultaneously.
