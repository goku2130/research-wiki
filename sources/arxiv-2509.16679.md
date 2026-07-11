---
id: arxiv:2509.16679
type: paper
title: 'Reinforcement Learning Meets Large Language Models: A Survey of Advancements
  and Applications Across the LLM Lifecycle'
url: https://arxiv.org/html/2509.16679v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

The provided text, "Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle," by Liu et al. (2025), surveys the integration of Reinforcement Learning (RL) with Large Language Models (LLMs) across their lifecycle, with a particular focus on Reinforcement Learning with Verifiable Rewards (RLVR).

**Core Problem:**
Despite remarkable performance in various tasks, LLMs often struggle with reliably capturing nuanced human intentions, producing misleading or unsafe outputs, and exhibiting shortcomings in reasoning capabilities. The core problem is to effectively align LLMs with human preferences and values, enhance their reasoning abilities for complex problems, and address the limitations of existing RL-augmented LLM surveys which often have limited scope.

**Method/Recipe Step-by-Step:**
The survey systematically reviews theoretical and practical advancements of RL in empowering LLMs, particularly RLVR, across the LLM lifecycle:

1.  **Introduction to Basic RL Theory:** Briefly introduces the fundamentals of RL, modeling problems as Markov Decision Processes (MDPs) with state space, action space, state transition probability distribution, and a reward function. The objective is to learn an optimal policy $\pi^*$ that maximizes expected long-term cumulative reward.
2.  **Policy Learning Methods:**
    *   **Policy Gradient:** Directly optimizes policy $\pi(a|s; \theta)$ parameters $\theta$ via gradient ascent to maximize expected returns.
        *   **REINFORCE:** A fundamental Monte Carlo policy gradient method.
        *   **Baseline for Variance Reduction:** Introduces a baseline function $b(s)$ to reduce variance without biasing the gradient estimate.
        *   **Actor-Critic (AC) Method:** Combines policy gradient with value function approximation, using an actor (policy) and a critic (value function) to estimate advantage $A(s,a)$ or Temporal-Difference (TD) error $\delta$.
        *   **Trust Region Policy Optimization (TRPO):** Addresses instability from large policy updates by formulating optimization as a constrained problem, maximizing advantage while constraining KL divergence between new and old policies.
        *   **Proximal Policy Optimization (PPO):** Uses a clipped surrogate objective for multi-step gradient updates without policy collapse.
        *   **Group Relative Policy Optimization (GRPO):** Samples a set of outputs for each prompt and uses relative differences in intra-group feedback to guide policy updates, using the intra-group average reward as a dynamic baseline, thus avoiding a separate value network.
3.  **Value Learning Methods:**
    *   **Q-learning:** A model-free, off-policy approach to approximate the optimal action-value function $Q^*(s,a)$ by iteratively updating value estimates based on the Bellman optimality equation.
    *   **SARSA:** An on-policy value learning method that evaluates action values according to the currently executed policy and updates estimates using samples from that policy.
    *   **Deep Q-Network (DQN):** Introduces neural function approximation into Q-learning, using a deep neural network $Q(s,a;\theta)$ to approximate the action-value function and training it to satisfy the Bellman optimality equation.
4.  **Application Strategies Across LLM Lifecycle:**
    *   **Pre-training Phase:** RL is introduced to next-token prediction tasks, allowing verifiable rewards for correct predictions. Mid-training strategies are used to adapt pre-trained models for RL.
    *   **Alignment Fine-tuning Phase:** RL, particularly RLHF, is used to align LLMs with human instructions and preferences. This includes methods like Direct Preference Optimization (DPO), Reinforcement Learning with AI Feedback (RLAIF), and various reward model designs (e.g., Reward Reasoning Model (RRM), generative process reward models).
    *   **Reinforced Reasoning Phase (RLVR):** Emphasizes RL methods for advancing model reasoning, especially with verifiable rewards (e.g., programmatic checks, proofs of correctness). This includes applications in multimodal reasoning, adaptive thinking, and agent training.
5.  **Datasets and Benchmarks:** Collates existing datasets (human-annotated, AI-assisted preference, program-verification-style) and evaluation benchmarks for RL fine-tuning, covering alignment/dialogue, code, math, general knowledge, logical reasoning, and agentic tasks.
6.  **Open-source Tools and Training Frameworks:** Reviews mainstream open-source tools and frameworks available for RL-enhanced LLMs.

**Key Formulas in LaTeX:**

*   **Policy Gradient (REINFORCE):**

$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta}}\left[\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})R_{t}\right]
$$

    where $R_t = \sum_{k=t}^{T}\gamma^{k-t}r_{k}$ is the discounted return.
*   **Policy Gradient with Baseline:**

$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta}}\left[\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t}|s_{t})(R_{t}-b(s_{t}))\right]
$$

    where $A_t = R_t - b(s_t)$ is the advantage function.
*   **TRPO Objective:**

$$
\underset{\theta}{\text{max}}L(\theta)=\mathbb{E}_{s\sim\pi_{\mathrm{old}}}\left[\sum_{a}\frac{\pi_{\theta}(a|s)}{\pi_{\mathrm{old}}(a|s)}A^{\pi_{\mathrm{old}}}(s,a)\right]
$$

$$
\text{s.t.}\mathbb{E}_{s\sim\pi_{\mathrm{old}}}\left[D_{\mathrm{KL}}(\pi_{\mathrm{old}}(\cdot|s)\parallel\pi_{\theta}(\cdot|s))\right]\leq\delta
$$

*   **PPO Clipped Objective:**

$$
L^{\mathrm{PPO}}(\theta)=\mathbb{E}_{t}\,[\min(r_{t}(\theta)\hat{A}_{t}, \mathrm{clip}(r_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t})]
$$

    where $r_t(\theta)$ is the probability ratio of new and old policies.
*   **GRPO Advantage Value (Normalized):**

$$
\hat{A}_{i,t}=\frac{R_{i}-\text{max}(\{R_{i}\}_{i=1}^{G})}{\text{std}(\{R_{i}\}_{i=1}^{G})}
$$

*   **GRPO Objective Function:**

$$
\mathcal{L}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{(q,a)\sim\mathcal{D}_{1},\{\sigma_{t}\}_{t=1}^{G}\sim\pi_{\mathrm{old}}(\cdot|q)}\left[\frac{1}{G}\sum_{t=1}^{G}\frac{1}{|\sigma_{t}|}\sum_{t=1}^{|\sigma_{t}|}\left(\min\left(r_{t,t}(\theta)\hat{A}_{t,t},\mathrm{~clip~}\left(r_{t,t}(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{t,t}\right)-\beta D_{\mathrm{KL}}(\pi_{\theta}||\pi_{\mathrm{old}})\right)\right]
$$

*   **Q-learning Update Rule:**

$$
Q_{\mathrm{new}}(s_{t},a_{t})\gets Q(s_{t},a_{t})+\alpha\left[r_{t}+\gamma\text{max}_{a^{\prime}}Q(s_{t+1},a^{\prime})-Q(s_{t},a_{t})\right]
$$

*   **SARSA Update Rule:**

$$
Q_{\mathrm{new}}(s_{t},a_{t})\leftarrow Q(s_{t},a_{t})+\alpha\left[r_{t}+\gamma Q(s_{t+1},a_{t+1})-Q(s_{t},a_{t})\right]
$$

*   **DQN Loss Function:**

$$
L(\theta)=\left(r+\gamma\text{max}_{a^{\prime}}Q(s^{\prime},a^{\prime};\theta^{-})-Q(s,a;\theta)\right)^{2}
$$

*   **Policy Performance vs. Entropy (Cui et al. [28]):**

$$
P = -c \exp(H) + 1
$$

    where $P$ is policy performance, $H$ is entropy, and $c$ is a constant.

**Key Quantitative Results and Numbers:**
Table 1 demonstrates performance improvements of RL-trained models over baselines:

*   **DeepSeek-R1-Zero:**
    *   AIME2024: +31.8
    *   GPQA-Diamond: +14.2
    *   LiveCodeBench: +13.8
    *   MATH-500: +5.7
*   **DeepSeek-R1:**
    *   AIME2024: +40.6
    *   GPQA-Diamond: +12.4
    *   LiveCodeBench: +29.7
    *   MATH-500: +7.1
    *   MMLU: +2.3
    *   SWE-benchVerified: +7.2
*   **Magistral Small-RL#:**
    *   AIME2024: +33.6
    *   GPQA: +5.4
    *   LiveCodeBench v5: +23.7
    *   MATH-500: +2.2
*   **OpenAI-o1-1217:**
    *   AIME2024: +70.2
    *   GPQA-Diamond: +25.8
    *   LiveCodeBench: +30.5
    *   MATH-500: +21.8
    *   MMLU: +4.6
    *   SWE-benchVerified: +10.1

Table 4 compares well-known reasoning LLMs:

*   **MMLU-Redux:** OpenAI-o1 (92.8), DeepSeek-R1 (92.9), Gemini2.5-Pro (93.7), Qwen3-235B-A22B (92.7).
*   **GPQA-Diamond:** OpenAI-o1 (78.0), DeepSeek-R1 (71.5), Grok-3-Beta (80.2), Gemini2.5-Pro (84.0), Qwen3-235B-A22B (71.1).
*   **MATH-500:** OpenAI-o1 (96.4), DeepSeek-R1 (97.3), Gemini2.5-Pro (98.8), Qwen3-235B-A22B (98.0).
*   **AIME'24:** OpenAI-o1 (74.3), DeepSeek-R1 (79.8), Grok-3-Beta (83.9), Gemini2.5-Pro (92.0), Qwen3-235B-A22B (85.7).
*   **LiveCodeBenchv5:** OpenAI-o1 (63.9), DeepSeek-R1 (64.3), Grok-3-Beta (70.6), Gemini2.5-Pro (70.4), Qwen3-235B-A22B (70.7).
*   **ZebraLogic:** OpenAI-o1 (81.0), DeepSeek-R1 (78.7), Gemini2.5-Pro (87.4), Qwen3-235B-A22B (80.3).

**Stated Limitations:**

1.  **RLVR's True Impact on Reasoning:** It remains debated whether RLVR truly expands LLM reasoning capabilities beyond pre-training or merely amplifies high-reward outputs already present in the base model's distribution. Some studies suggest RLVR models primarily improve sampling efficiency of correct reasoning paths, with all generated paths existing in the base model's sampling distribution, implying inherent limitations by the base model.
2.  **Application of RL Techniques:** There is no clear consensus on how different RL techniques should be best applied at various stages of the LLM lifecycle (pre-training, instruction alignment, post-training inference optimization).
3.  **Practical Issues in RL:** Challenges persist in data curation (constructing high-quality reward datasets via human preference, AI-assisted preference, or programmatic rewards) and optimization strategy (choosing appropriate RL algorithms like policy gradients vs. reward model optimization).
4.  **Efficiency and Stability of RL Fine-tuning:** Implementing RL fine-tuning efficiently at scale without destabilizing the model's performance is still not fully resolved.
5.  **Resource Consumption in Pre-training RL:** Introducing RL into pre-training can consume excessive resources and requires an excellent base model with existing reasoning capabilities.
6.  **Memory/Computation Costs and Instability in PPO for Reasoning:** PPO in LLM fine-tuning incurs added memory/computation costs from extra value networks and can suffer instability in long sequences due to inaccurate value estimates.
7.  **Overestimation Bias in Q-learning:** The maximization step in Q-learning can yield upward-biased value estimates.
8.  **Action Space Complexity for Value-based Methods:** The immense and complex action space in LLMs makes it infeasible to explicitly construct Q-tables or networks that evaluate the value of every possible output, limiting the direct applicability of value-based methods.
9.  **Diversity Collapse and Forgetting in RLVR:** RLVR, while an efficient sampler, can suffer from diversity collapse and may cause the model to forget problems the base model already knows how to solve.
10. **Entropy Depletion and Computational Power Bottleneck:** Policy performance in RL is achieved at the cost of entropy consumption, limiting it by entropy depletion and leading to a computational power expansion bottleneck without proper entropy management.
11. **Necessity of Explicit Thinking Processes:** For some tasks and models, removing or adjusting the explicit thinking process in rule-based reinforcement fine-tuning can improve performance and efficiency, questioning its universal necessity.
12. **Side Effects of Training Settings:** Increases in response length observed in RL training might be a side effect of training settings rather than an improvement in reasoning ability.
13. **Limitations of Supervised Fine-tuning (SFT):** SFT is prone to generating "pseudo reasoning paths," failing to enhance complex reasoning and impairing subsequent RL performance.
14. **Accuracy Collapse and Inefficient Reasoning in LRMs:** Large Reasoning Models (LRMs) show significant limitations in complex tasks, including accuracy collapse at high complexity and inefficient reasoning processes.
15. **Hallucinations with Stronger Reasoning:** Stronger reasoning can sometimes increase hallucinations, as longer chains shift attention from image content to language priors.
16. **Context Window Constraints in Agentic Tasks:** Limited context windows constrain long-term dynamic reasoning in agentic tasks.
17. **External Supervision Dependence:** RLHF and RLVR require extensive external supervision, and RLVR does not bring truly novel external information to LLMs.
18. **Low Data Utilization and Text Bias in GRPO:** GRPO can suffer from low data utilization (models struggle with hard samples) and text bias (models ignore images and rely only on text).
19. **Fragmented Tooling:** RL tooling for LLMs is fragmented, with libraries varying in interfaces and scope, complicating pipeline integration.
