---
id: arxiv:2503.18130
type: paper
title: Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization
url: https://arxiv.org/html/2503.18130v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

The paper "Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization" (arXiv:2503.18130v1) addresses the problem of reward over-optimization in Reinforcement Learning from Human Feedback (RLHF) for Large Language Models (LLMs).

**Core Problem:**
Reward over-optimization in RLHF leads to a discrepancy between an LLM's performance under a proxy reward model and its true alignment with human objectives (gold reward). This issue primarily stems from extrapolation errors by the reward model when evaluating out-of-distribution (OOD) responses generated during the reinforcement learning (RL) process. Existing methods often fail to prevent the increasing generation of OOD responses and struggle with these extrapolation errors, or they pessimistically handle OOD responses in a way that also negatively impacts in-distribution (ID) evaluations, leading to suboptimal solutions.

**Method/Recipe Step by Step (Behavior-Supported Policy Optimization - BSPO):**

1.  **Define Behavior Policy for OOD Detection:**
    *   Formalize language generation as a token-level Markov Decision Process (MDP) where states are token sequences and actions are next tokens.
    *   Define a *behavior policy* $\beta: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$ as the next-token distribution observed in the reward training dataset. This policy characterizes the in-distribution (ID) region of the reward model.
    *   An action $a$ in state $s$ is "behavior-supported" if $\beta(a|s) > 0$. Actions with $\beta(a|s) = 0$ are considered OOD.
    *   The paper hypothesizes that OOD actions lead to accumulation of extrapolation errors in resulting unsupported responses. Empirical evidence (Figure 1c) shows that the proxy model's prediction accuracy drops significantly for responses containing unsupported actions (58.10%) compared to supported ones (75.91%).

2.  **Introduce Behavior-Supported Bellman Operator:**
    *   Define a minimum possible return $Q_{\min} \doteq \sum_{t=0}^{\infty} \gamma^t r_{\min} = \frac{r_{\min}}{1-\gamma}$, where $r_{\min}$ is the minimum reward.
    *   Propose the behavior-supported Bellman operator $\mathcal{T}_{\beta}^{\pi}Q(s,a)$ for Q-values:

$$
\mathcal{T}_{\beta}^{\pi}Q(s,a)\doteq\left\{\begin{aligned}{}&{{}\mathcal{T}^{\pi}Q(s,a),}&{}&{{}\text{i f}\quad\beta(a|s)>0,}\\ {}&{{}Q_{\text{m i n}},}&{}&{{}\text{o t h e r w i s e}.}\\ \end{aligned}\right.
$$

        where $\mathcal{T}^{\pi}Q(s,a)\doteq r(s,a)+\gamma\mathbb{E}_{a^{\prime}\sim\pi(\cdot|s^{\prime})}\left[Q(s^{\prime},a^{\prime})\right]$ is the standard Bellman operator. This operator penalizes OOD values by setting them to $Q_{\min}$ without affecting ID values.

3.  **Policy Optimization with Regularized Values:**
    *   The fixed point $Q_{\beta}^{\pi}$ of $\mathcal{T}_{\beta}^{\pi}$ is the behavior-supported Q-value function. It underestimates future returns for OOD actions, thus disadvantaging them.
    *   Policy iteration is performed using these behavior-supported Q-values:

$$
\pi_{k+1}(a|s) = \mathbb{I}\left[a=\text{arg\,max}_{a^{\prime}\in\mathcal{A}}Q_{\beta}^{\pi_{k}}(s,a^{\prime})\right]
$$

        where $V_{\beta}^{\pi_{k}}(s)=\mathbb{E}_{a\sim\pi_{k}(\cdot|s)}\left[Q_{\beta}^{\pi_{k}}(s,a)\right]$ and $A_{\beta}^{\pi_{k}}(s,a)=Q_{\beta}^{\pi_{k}}(s,a)-V_{\beta}^{\pi_{k}}(s)$. This process reinforces behavior-supported actions and weakens unsupported ones, leading to a "behavior-supported policy" $\pi \in \Pi_{\beta}$.

4.  **Implementation Details (ScoreLM and V-value Regularization):**
    *   **Behavior Distribution Prediction:** A single "ScoreLM" model is used to predict both rewards and behavior distributions. It retains the original LLM head for next-token distribution and adds a score head for reward prediction. The training loss is:

$$
\mathcal{L}_{\text{ScoreLM}}(\phi;\mathcal{D})=-\underbrace{\mathbb{E}_{\mathcal{D}}\Big[\text{log}\sigma\Big(R(x,y_{w};\phi)-R(x,y_{l};\phi)\Big)\Big]}_{\text{Preference Loss}}-\alpha\underbrace{\mathbb{E}_{\mathcal{D}}\Big[\text{log}\beta\big(a_{t}|x\cup a_{0,t-1};\phi\big)\Big]}_{\text{Supervised Loss}}
$$

        where $\alpha$ balances the two losses.
    *   **Behavior-Supported V-Value Function:** For stability, behavior-supported V-values are predicted instead of Q-values, leveraging the deterministic state transition in token-level MDPs. The behavior-supported Bellman V-operator is:

$$
\mathcal{T}_{\beta,V}^{\pi}V(x\cup a_{0:t-1})\doteq\begin{cases}{\mathcal{T}_{V}^{\pi}V(x\cup a_{0:t-1}),}&{\text{i f}\quad\beta(a_{t-1}|x\cup a_{0:t-2})>0,}\\ {\frac{1}{\gamma}\left[Q_{\text{m i n}}-r(x\cup a_{0:t-2},a_{t-1})\right],}&{\text{o t h e r w i s e},}\\ \end{cases}
$$

        where $\mathcal{T}_{V}^{\pi}V(x\cup a_{0:t-1})\doteq\mathbb{E}_{a_{t}\sim\pi(\cdot|x\cup a_{0:t-1})}\left[r(x\cup a_{0:t-1},a_{t})+\gamma V(x\cup a_{0:t})\right]$ is the standard Bellman V-operator.
    *   **Integration with PPO:** BSPO combines the behavior-supported method with Proximal Policy Optimization (PPO). The LLM $\pi_{\theta}$ is optimized by minimizing:

$$
\mathcal {L} _ {\pi} (\theta) = - \mathbb {E} _ {\tau \sim \pi_ {\theta_ {k}}} \left[ \min \left(\rho_ {t} (\theta) A ^ {\pi_ {\theta_ {k}}} \left(s _ {t}, a _ {t}\right), \text {clip} \left(\rho_ {t} (\theta), 1 - \epsilon , 1 + \epsilon\right) A ^ {\pi_ {\theta_ {k}}} \left(s _ {t}, a _ {t}\right)\right) \right]
$$

        where $\rho_{t}(\theta)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{k}}(a_{t}|s_{t})}$ and $A^{\pi_{\theta_{k}}}$ are advantage estimates based on predictions from the behavior-supported value model $V_{\phi_{k}}$.

**Key Formulas (in LaTeX):**

*   **Behavior-Supported Bellman Operator (Q-values):**

$$
\mathcal{T}_{\beta}^{\pi}Q(s,a)\doteq\left\{\begin{aligned}{}&{{}\mathcal{T}^{\pi}Q(s,a),}&{}&{{}\text{i f}\quad\beta(a|s)>0,}\\ {}&{{}Q_{\text{m i n}},}&{}&{{}\text{o t h e r w i s e}.}\\ \end{aligned}\right.
$$

*   **ScoreLM Training Loss:**

$$
\mathcal{L}_{\text{ScoreLM}}(\phi;\mathcal{D})=-\underbrace{\mathbb{E}_{\mathcal{D}}\Big[\text{log}\sigma\Big(R(x,y_{w};\phi)-R(x,y_{l};\phi)\Big)\Big]}_{\text{Preference Loss}}-\alpha\underbrace{\mathbb{E}_{\mathcal{D}}\Big[\text{log}\beta\big(a_{t}|x\cup a_{0,t-1};\phi\big)\Big]}_{\text{Supervised Loss}}
$$

*   **Behavior-Supported Bellman Operator (V-values):**

$$
\mathcal{T}_{\beta,V}^{\pi}V(x\cup a_{0:t-1})\doteq\begin{cases}{\mathcal{T}_{V}^{\pi}V(x\cup a_{0:t-1}),}&{\text{i f}\quad\beta(a_{t-1}|x\cup a_{0:t-2})>0,}\\ {\frac{1}{\gamma}\left[Q_{\text{m i n}}-r(x\cup a_{0:t-2},a_{t-1})\right],}&{\text{o t h e r w i s e},}\\ \end{cases}
$$

*   **PPO Policy Loss (with behavior-supported advantage):**

$$
\mathcal {L} _ {\pi} (\theta) = - \mathbb {E} _ {\tau \sim \pi_ {\theta_ {k}}} \left[ \min \left(\rho_ {t} (\theta) A ^ {\pi_ {\theta_ {k}}} \left(s _ {t}, a _ {t}\right), \text {clip} \left(\rho_ {t} (\theta), 1 - \epsilon , 1 + \epsilon\right) A ^ {\pi_ {\theta_ {k}}} \left(s _ {t}, a _ {t}\right)\right) \right]
$$

**Key Quantitative Results and Numbers:**

*   **Reward Model Extrapolation Error:** For comparison pairs consisting of supported responses, the proxy model achieved an average prediction accuracy of **75.91%**. For comparison pairs with unsupported responses, accuracy dropped significantly to **58.10%**.
*   **Performance against Baselines (Figure 3):** BSPO consistently achieved the highest "gold reward" across three proxy model scales (774M, 1.1B, 2.7B), indicating effective mitigation of reward over-optimization. Other baselines (Standard PPO, KL-Penalty, CPPO, ENS-UWO, ENS-WCO) showed varying degrees of over-optimization (proxy reward increasing while gold reward stagnates or decreases).
*   **Reduction of OOD Responses (Figure 4b):** BSPO maintained a consistently low number of behavior-unsupported actions (tokens) per response during RL training, while traditional methods showed a significant increase in such actions as over-optimization occurred.
*   **Win Rate against Initial Model (Figure 4a):** BSPO achieved the highest win rate against the initial Alpaca-7B model, outperforming all baselines.
*   **Comparison with KL Penalty (Figure 4c):** BSPO avoided over-optimization even at larger KL divergence distances, whereas the KL penalty method failed at closer distances.

**Stated Limitations:**

1.  **Synthetic Setup:** The evaluation relies on a synthetic setup using a "gold-standard" reward model instead of human evaluations due to cost. This raises questions about the generalizability of results to real-world scenarios, where human preferences are more variable.
2.  **Assumption of Well-Trained Reward Model:** BSPO primarily addresses over-optimization during the RL phase, assuming the reward model itself is well-trained and accurately captures preferences. It does not tackle issues arising from an inaccurate reward model during the reward learning phase.
3.  **OOD Prompts:** The current work does not explicitly address the presence of OOD prompts in the RL phase, where the reward model might not have ID responses. This implies a need for further study on data distribution shifts between the reward modeling and RL phases.
4.  **Multi-Objective Scenarios:** Extending BSPO to multi-objective scenarios (e.g., multiple reward models) is a direction for future work.
