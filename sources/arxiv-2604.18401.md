---
id: arxiv:2604.18401
type: paper
title: 'StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning'
url: https://arxiv.org/html/2604.18401v4
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

**Core Problem**
Agentic reinforcement learning (RL) for large language models (LLMs) suffers from a fundamental granularity mismatch. Existing RL algorithms, such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO), adopt a token-centric paradigm where tokens are the basic units for modeling states, actions, and policy updates. However, LLM agents interact with environments at the step level, where the agent receives an observation, generates a complete response (which may include tool calls), and transitions to the next state based on environmental feedback. This discrepancy misaligns the Markov Decision Process (MDP) formulation and credit assignment with the natural decision granularity of multi-turn agent behaviors.

**Method**
StepPO addresses this gap by proposing a step-aligned policy optimization paradigm. The method proceeds through the following steps:
1. **Step-level MDP Formulation:** StepPO reformulates agentic RL as a step-level MDP. A trajectory is represented as $\tau = \{(s_{1:M_t}^{(t)}, a_{1:L_t}^{(t)}, r_t, s_{1:M_{t+1}}^{(t+1)})\}_{t=1}^T$, where $s_{1:M_t}^{(t)}$ is the state history, $a_{1:L_t}^{(t)}$ is the complete action response, and $r_t$ is the step reward. The policy objective is defined as $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^T \gamma^{t-1} r_t \right]$.
2. **Step-native Trajectory Representation:** To avoid retokenization drift and maintain token-space consistency, StepPO stores trajectories as step-native records rather than flat token sequences. Each record contains state tokens, action tokens, and rewards, ensuring that replay units align with MDP transitions.
3. **Step-level Credit Assignment:** StepPO computes advantages at the step granularity. The state value is estimated at the final token before action generation, $V_\phi(s_{M_t}^{(t)})$. The step-level temporal-difference (TD) residual is $\delta_t = r_t + \gamma V_\phi(s_{M_{t+1}}^{(t+1)}) - V_\phi(s_{M_t}^{(t)})$. The step-level advantage is then computed using Generalized Advantage Estimation (GAE): $\hat{A}_t^{\text{Step}} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$.
4. **Step-level Importance Sampling:** For multi-token actions, directly multiplying token-level importance ratios leads to extreme values. StepPO stabilizes updates by using the geometric mean of token ratios: $\bar{w}_t(\theta) = \exp \left( \frac{1}{L_t} \sum_{i=1}^{L_t} \log w_t^i(\theta) \right)$. Theoretical analysis shows that while the variance of the exact product ratio grows linearly with action length $L_t$, the variance of the geometric-mean ratio decreases with $L_t$, providing length-invariant update scales.
5. **Step-level Actor Objective:** The policy is optimized using a clipped PPO objective: $\mathcal{J}_{\text{actor}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{old}}} \left[ \frac{1}{T} \sum_{t=1}^{T} \min \left( \bar{w}_t(\theta) \hat{A}_t^{\text{Step}}, \text{clip}_\epsilon \left( \bar{w}_t(\theta) \right) \hat{A}_t^{\text{Step}} \right) \right]$.

**System Requirements**
Training StepPO requires specific infrastructure. It necessitates step-native data representation, data management via gateway and datapool abstractions to handle heterogeneous agents, computational efficiency optimization through shared-prefix reuse, and asynchronous training design to decouple rollout and training engines.

**Results**
StepPO was evaluated on Qwen3-1.7B and Qwen3-4B-Instruct-2507 across multi-hop QA (HotpotQA, 2Wiki, MuSiQue), academic paper search (RealResearchQuery), ALFWorld, and WebShop. StepPO consistently outperforms baselines including PPO, GRPO, and GiGPO. On Qwen3-4B, StepPO achieves 63.78 accuracy on HotpotQA (vs. 58.14 for GiGPO), 66.16 on 2Wiki, 29.87 on MuSiQue, 92.14 win rate on ALFWorld seen tasks, and 77.52 score on WebShop. Ablation studies confirm that removing step-level GAE or step-level importance sampling degrades performance. Efficiency analysis shows StepPO maintains similar per-iteration training times to PPO (e.g., 93.18s critic update vs. 104.60s for PPO). Training dynamics show StepPO achieves higher rewards and lower critic loss than PPO and GRPO.

**Limitations**
The source does not explicitly detail limitations of StepPO. It notes that token-level GAE suffers a larger score drop (13.09%) than step-level GAE (5.53%) when the discount factor $\gamma$ decreases from 0.99 to 0.95 on WebShop, highlighting the robustness of the step-centric approach.
