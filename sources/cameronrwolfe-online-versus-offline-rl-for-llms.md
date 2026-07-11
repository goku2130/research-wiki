---
id: cameronrwolfe:online-versus-offline-rl-for-llms
type: web
title: Online versus Offline RL for LLMs
url: https://cameronrwolfe.substack.com/p/online-rl
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

The article "Online versus Offline RL for LLMs" by Cameron R. Wolfe, Ph.D., discusses the performance gap between online and offline Reinforcement Learning (RL) methods for aligning Large Language Models (LLMs).

**Core Problem:**
The primary challenge addressed is the complexity and computational expense of traditional online RL algorithms, specifically PPO-based Reinforcement Learning from Human Feedback (RLHF), for LLM alignment. PPO-based RLHF is an online algorithm because it actively generates "on-policy" samples using the current LLM during training. This leads to difficulties in orchestration, stability issues, significant memory overhead (requiring multiple copies of the LLM), and a wide range of training settings that need careful management. The article explores whether simpler, offline, or RL-free alignment methods can achieve comparable performance.

**Method/Recipe Step-by-Step:**

LLM training generally involves several stages:
1.  **Pretraining:** Training the LLM from scratch on internet-scale text data using a next-token prediction objective.
2.  **Supervised Finetuning (SFT) / Instruction Finetuning (IFT):** Training the LLM on a smaller dataset of high-quality completions using a supervised next-token prediction objective to emulate desired responses.
    *   **SFT Process:**
        *   Curate a dataset of high-quality prompt-response pairs.
        *   Train the LLM to predict the next token in a sequence, given prior tokens.
        *   Often uses a completion-only loss, masking prompt tokens and applying loss only to response tokens.
    *   **Rejection Sampling (Online SFT variant):**
        *   Start with a dataset of prompts.
        *   Generate completions for each prompt using the current LLM.
        *   Score completions using a reward model or LLM judge.
        *   Select top-scoring prompt-completion pairs.
        *   Perform SFT over these selected examples. This process is typically iterative.
    *   **Other Online SFT Variants:**
        *   **Supervised Iterative Learning from Human Feedback (SuperHF):** Samples on-policy outputs, filters with a reward model, and optimizes with a supervised objective under a KL divergence constraint.
        *   **Reinforced Self-Training (ReST):** Iteratively samples on-policy data, scores with a reward model, and trains on the best samples.
        *   **Reward-Weighted Regression (RWR):** Generates on-policy samples, scores them with a reward model, and uses these scores to weight each sample in the training loss.
        *   **Reward Ranked Finetuning (RAFT):** Similar to rejection sampling, samples online completions and filters them using reward model scores for SFT.
3.  **Reinforcement Learning from Human Feedback (RLHF) / Preference Finetuning (PreFT):** Training the LLM with RL using human preference data.
    *   **RLHF Process:**
        *   Collect a dataset of preference pairs (prompt, chosen completion, rejected completion).
        *   Train a reward model on this preference dataset.
        *   Generate completions over a set of prompts using the current LLM (on-policy sampling).
        *   Compute rewards for these completions using the reward model.
        *   Use rewards to derive a policy update (update LLM parameters) with an RL optimizer (e.g., PPO, REINFORCE, GRPO).
4.  **Reinforcement Learning with Verifiable Rewards (RLVR) / Reinforcement Finetuning (RFT):** Training the LLM with RL using rewards derived deterministically from rules or heuristics.
    *   **RLVR Process:** Similar to RLHF, but rewards come from verifiable tasks rather than a human preference model.
5.  **Direct Alignment Techniques (Offline):**
    *   **Direct Preference Optimization (DPO):** Avoids training an explicit reward model by implicitly deriving a reward signal from the LLM itself.

**Key Formulas in LaTeX:**

*   **Next-token prediction loss (Cross-Entropy):**
    ```latex
    \text{loss} = F.\text{cross\_entropy}(\text{logits}, \text{targets})
    ```
    where `logits` are the model's output probabilities for the next token and `targets` are the actual next tokens.
*   **RL Training Objective:** Maximizes reward while minimizing KL divergence from a reference model.
    ```latex
    \max_{\theta} \mathbb{E}_{(x,y) \sim D} [R(x,y) - \beta D_{KL}(\pi_\theta(y|x) || \pi_{\text{ref}}(y|x))]
    ```
    where $\theta$ are the LLM parameters, $x$ is the prompt, $y$ is the completion, $R(x,y)$ is the reward for completion $y$ given prompt $x$, $\beta$ is a coefficient, $D_{KL}$ is the KL divergence, $\pi_\theta$ is the current policy, and $\pi_{\text{ref}}$ is the reference policy.

**Key Quantitative Results and Numbers:**

*   LIMA aligned an LLM using SFT with a curated dataset of **only 1K examples**.
*   Tulu-3 is trained with **~1M SFT examples**.
*   Llama-2 alignment process uses **four rounds of rejection sampling** before RL-based RLHF.
*   PPO-based RLHF stores **four different copies of the LLM** during training (policy, reference policy, value model, reward model).
*   REINFORCE and GRPO can reduce overhead; GRPO, when used without a reward model (common in RLVR), stores **only two copies of the LLM** (policy and reference policy).

**Stated Limitations:**

*   PPO-based RLHF is complex to implement, difficult to efficiently orchestrate (especially in synchronous training setups), and often suffers from stability issues.
*   PPO requires significant memory overhead and high hardware requirements due to storing multiple copies of the LLM.
*   PPO involves a wide range of training settings and design decisions that must be managed for successful training.
*   Simpler alignment algorithms (offline or RL-free) tend to come at a cost in performance compared to online RL, creating an "online-offline performance gap."
