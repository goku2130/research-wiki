---
id: arxiv:2503.11019
type: paper
title: 'Residual Policy Gradient: A Reward View of KL-regularized Objective'
url: https://arxiv.org/abs/2503.11019
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Residual Policy Gradient: A Reward View of KL-regularized Objective

### Core Problem
The authors address the challenge of **policy customization**: adapting a prior policy $\pi$ (trained on a basic reward $r$) to meet new task-specific requirements (defined by an add-on reward $r_R$) while preserving the original policy's inherent properties. While Residual Q-Learning (RQL) provided a principled value-based framework for this, it had not been extended to policy gradient methods. This limitation is significant because policy gradient methods often outperform value-based methods in large-scale parallel environments.

### Method/Recipe
The authors propose **Residual Policy Gradient (RPG)**, which extends the RQL framework to gradient-based RL. The method follows these steps:

1.  **Derive Soft Policy Gradient (SPG):** To optimize a maximum-entropy objective, the authors derive a concise gradient estimator. They show that maximizing the entropy-regularized objective is equivalent to applying a standard policy gradient over a reformulated reward: $r^{\text{SPG}} = r(\boldsymbol{s}_t, \boldsymbol{a}_t) - \alpha \log \pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t)$.
2.  **Formulate the Augmented MDP:** Following RQL, the customization problem is framed as solving an augmented MDP $\mathcal{M}^{\text{aug}} = (\mathcal{S}, \mathcal{A}, \omega' \log \pi + r_R, p)$, where $\omega' = \omega \alpha$ balances the basic and add-on tasks.
3.  **Implement Residual Policy Gradient (RPG):** RPG applies the SPG derivation to the augmented MDP. The resulting reward used for advantage estimation is:

$$
r^{\text{RPG}}(s_t, a_t) = r_R(s_t, a_t) + \omega' \log \pi(a_t|s_t) - \hat{\alpha} \log \pi_{\theta}(a_t|s_t)
$$

4.  **Practical Deployment (Residual PPO):** To improve stability and data efficiency, the authors integrate this reformulated reward into the Proximal Policy Optimization (PPO) framework. **Residual PPO** computes the advantage $A_t^{\text{RPG}}$ using $r^{\text{RPG}}$ and removes the repeated entropy term from the actor loss.

### Key Formulas
The maximum-entropy objective is defined as:

$$
J(\pi_\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=0} \gamma^t \left( r(\boldsymbol{s}_t, \boldsymbol{a}_t) + \alpha \mathbb{E}_{\boldsymbol{a}_t \sim \pi_\theta(\cdot|\boldsymbol{s}_t)} \left[ -\log \pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t) \right] \right) \right]
$$

The resulting **Soft Policy Gradient** estimator is:

$$
\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_{t=0} \nabla_\theta \log \pi_\theta(\boldsymbol{a}_t|\boldsymbol{s}_t) \left( \sum_{t'=t} \gamma^{t'-t} \left( r(\boldsymbol{s}_{t'}, \boldsymbol{a}_{t'}) - \alpha \log \pi_\theta(\boldsymbol{a}_{t'}|\boldsymbol{s}_{t'}) \right) \right) \right]
$$

The authors also rethink the **KL-regularized objective** as a special case of RPG where $\omega' = \hat{\alpha} = \beta$:

$$
r^{\mathrm{KL}}(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}) = r_{R}(\boldsymbol{s}_{t}, \boldsymbol{a}_{t}) + \beta \log \pi(\boldsymbol{a}_{t} | \boldsymbol{s}_{t}) - \beta \log \pi_{\theta}(\boldsymbol{a}_{t} | \boldsymbol{s}_{t})
$$

### Key Quantitative Results
Experiments were conducted in MuJoCo environments (HalfCheetah, Hopper, Ant).

*   **Soft PPO Performance:** Soft PPO generally matched or outperformed other entropy-regularized PPO variants. In HalfCheetah, it achieved a best performance of $5896.5 \pm 371.8$, significantly outperforming "Repeat-Entropy PPO" ($-38.9 \pm 58.2$).
*   **Residual PPO Customization:**
    *   **HalfCheetah:** Residual PPO achieved a total reward of $5488.3 \pm 75.6$, which is comparable to the "Full Policy" upper bound of $5556.2 \pm 57.8$.
    *   **Ant:** Residual PPO significantly outperformed KL PPO, achieving a total reward of $5568.3 \pm 852.4$ compared to KL PPO's $4019.4 \pm 1103.4$.
    *   **Hopper:** Residual PPO achieved a total reward of $4401.4 \pm 505.3$, improving upon the Prior Policy ($5133.7 \pm 33.2$ total, but only $3689.4 \pm 29.2$ basic reward) and Greedy PPO ($3714.5 \pm 7.1$).

### Stated Limitations
The authors note that while decoupling $\omega'$ and $\hat{\alpha}$ provides flexibility, it can lead to suboptimal results if there are "excessively large assumption misalignments," as observed in the Hopper task. They conclude that for foundation models to be more adaptable, more adequate modeling of complex distributions is required.
