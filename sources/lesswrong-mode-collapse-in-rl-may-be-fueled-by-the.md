---
id: lesswrong:mode-collapse-in-rl-may-be-fueled-by-the
type: web
title: Mode collapse in RL may be fueled by the update equation
url: https://www.lesswrong.com/posts/A7RgYuYH4HywNeYWD/mode-collapse-in-rl-may-be-fueled-by-the-update-equation
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Mode Collapse in RL via Action-Conditioned TD Error (ACTDE)

### Core Problem: Infinite Logit Updates and Mode Collapse
The authors argue that standard policy gradient methods, such as Proximal Policy Optimization (PPO), are prone to "mode collapse" because their update equations allow a network to extract an unbounded amount of reinforcement from a single type of experience. 

In PPO, the optimization objective is proportional to the advantage:

$$
A^\pi(s,a) := \mathbb{E}_{s' \sim T(s,a)}[R(s,a,s') + \gamma v^\pi(s')] - v^\pi(s)
$$

The problem arises from the state-value baseline $v^\pi(s)$. In a simple bandit scenario with two actions (e.g., "wedding" with $R=1$ and "party" with $R=0.5$), the policy may alternate between actions. When the agent takes the lower-reward action, $v^\pi(s)$ decreases; when it subsequently takes the higher-reward action, the advantage $A^\pi(s,a)$ becomes positive again relative to the now-lower baseline. This cycle can continue indefinitely, driving the logits of the higher-reward action toward infinity. This process penalizes exploration and can lead the agent to collapse onto a single action (mode collapse) or aggressively pursue "wireheading" opportunities.

### Method: Action-Conditioned TD Error (ACTDE)
To mitigate this, the authors propose **Action-Conditioned TD Error (ACTDE)**. The core modification replaces the state-value baseline $v^\pi(s)$ with an action-value baseline $q^\pi(s,a)$ to account for the specific action taken:

$$
A^{\pi*}(s,a) := \mathbb{E}_{s' \sim T(s,a)}[R(s,a,s') + \gamma v^\pi(s')] - q^\pi(s,a)
$$

By using $q^\pi(s,a)$, the update mimics a reward prediction error. If the agent takes an off-policy action, the optimization term accounts for the expected value of that specific action rather than the average value of the state.

**Theoretical Recipe/Behavior:**
1. **Tabular Convergence:** In tabular settings, ACTDE does not converge to an "optimal" (greedy) policy. Instead, it converges to a softmax distribution over the rewards.
2. **Logit Addition:** Reward logits are effectively added to the initialization logits $\pi_0$.
3. **Fixed Reinforcement:** A single reward event provides a finite amount of updating, preventing the "infinite logit" failure.

### Quantitative Results
**Tabular Example:**
For rewards $R(\text{pizza})=10$ and $R(\text{cookies})=11$, PPO converges to a policy with arbitrarily high logits on cookies. ACTDE converges to a softmax-over-reward policy: $\{\text{pizza}: 27\%, \text{cookies}: 73\%\}$.

**Iterated Prisoner's Dilemma Experiment:**
The authors tested ACTDE using a `minGPT` model (3 layers, embedding dimension 12) playing against its past self. The reward matrix was:
*   Cooperate (t) / Cooperate (t-1): $0.5$
*   Defect (t) / Cooperate (t-1): $2$
*   Cooperate (t) / Defect (t-1): $-1.76$
*   Defect (t) / Defect (t-1): $-0.74$

With $\gamma=0.5$, the optimal strategy is alternating (`cdcd...`) with a discounted return of $1.492$.

*   **PPO Results:** Immediately learned the alternating strategy and collapsed onto a pure strategy.
*   **ACTDE Results:** Did not collapse onto a pure strategy. The model initially alternated with high probability but tended to regress toward a uniform or softmax-return policy over 10k epochs. While $\pi(\text{alternating}) > \pi(\text{cooperate}), \pi(\text{defect})$, convergence was very slow and inconsistent across trials.

### Stated Limitations
*   **Convergence Properties:** In non-tabular settings (like the `minGPT` experiment), ACTDE showed unclear convergence properties and may potentially converge to uniform policies.
*   **Sensitivity:** Results were sensitive to algorithmic variations, including the use of whitening for advantages, the detaching of value/Q-heads, and the choice of loss function (PPO vs. ILQL).
*   **Scalability:** The authors speculate that the method might not perform well for RLHF at scale due to the finicky nature of deep RL.
*   **Sub-optimality:** By design, ACTDE does not train an optimal greedy policy, which may be undesirable in scenarios where one action is decisively superior (e.g., "ice cream" vs. "slap in the face").
