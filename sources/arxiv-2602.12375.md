---
id: arxiv:2602.12375
type: paper
title: Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning
url: https://arxiv.org/abs/2602.12375
retrieved: '2026-07-12'
maturity: comprehensive
topic: entropy-and-exploration
---

# Value Bonuses using Ensemble Errors (VBE) for Exploration in RL

### Core Problem
Directed exploration in reinforcement learning (RL) often relies on optimistic value estimation. However, existing methods face a trade-off between usability and effectiveness. Reward-bonus approaches (e.g., RND, ACB) typically lack **first-visit optimism** because they only increase value bonuses retroactively after a reward bonus is encountered, failing to encourage the agent to visit a state for the first time. Conversely, Bootstrap DQN (BDQN) provides first-visit optimism but is computationally onerous, requiring large ensembles and specific design choices regarding how often to switch between value functions.

### Method
The proposed **Value Bonuses using Ensemble Errors (VBE)** algorithm directly estimates a value bonus $b(s, a)$ that can be layered on top of any base value-based RL algorithm (e.g., Double DQN). VBE maintains an ensemble of $k$ random action-value functions (RQFs).

**Step-by-Step Recipe:**
1. **Initialization:** Initialize a main action-value function $q_w$ and its target network $\tilde{q}_w$. Initialize an ensemble of $k$ fixed target RQFs $f_i$ and $k$ learnable predictor RQFs $f_{w_i}$ with their corresponding target networks $\tilde{f}_{w_i}$.
2. **Action Selection:** Use an optimistic behavior policy $\pi_b$ that selects actions greedily based on the sum of the main value estimate and a scaled value bonus:

$$
\pi_b(s) = \text{argmax}_{a \in \mathcal{A}} q_w(s, a) + c \cdot b(s, a)
$$

   where $c$ is the bonus scale and the value bonus is the maximum absolute error across the ensemble:

$$
b(s, a) = \max_{i \in [k]} |f_{w_i}(s, a) - f_i(s, a)|
$$

3. **Main Value Update:** Update $q_w$ using standard Double DQN updates.
4. **Ensemble Update:** To minimize computational overhead, randomly select one predictor RQF $f_{w_i}$ per step to update.
5. **Random Reward Construction:** Define a stochastic reward $r_i$ consistent with the target RQF $f_i$:

$$
r_i = f_i(s, a) - \gamma f_i(s', \text{argmax}_{a' \in \mathcal{A}} q_w(s', a'))
$$

6. **Predictor Update:** Update $f_{w_i}$ using temporal difference (TD) learning on the random reward $r_i$:

$$
w_i \leftarrow w_i + \eta \delta_i \nabla f_{w_i}(s, a) \quad \text{where} \quad \delta = r_i + \gamma f_{\tilde{w}_i}(s', \text{argmax}_{a' \in \mathcal{A}} q_w(s', a')) - f_{w_i}(s, a)
$$

### Key Quantitative Results
VBE was evaluated against BDQN, RND, and ACB across several benchmarks:

*   **Pure Exploration (Tabular Deepsea):** VBE was the only algorithm to cover the entire state space across all tested grid sizes. BDQN failed on larger grids, while ACB and RND failed to cover the state space entirely due to a lack of first-visit optimism.
*   **Classic Control:** VBE learned faster and achieved the best final performance in Sparse Mountain Car, Puddle World, River Swim, and Deepsea. In Mountain Car, ACB and RND failed to learn entirely.
*   **Atari:** VBE scaled successfully to image-based domains. It performed on par with or better than baselines in Breakout, Pong, Qbert, and Pitfall. In Gravitar, RND showed stronger initial performance but eventually matched VBE.

### Stated Limitations
*   **Theoretical Convergence:** While the authors provide convergence proofs for fixed policies, they note that there is currently no theory to guarantee convergence when both the behavior and target policies change over time, although empirical results show bonuses eventually converge to zero.
*   **Hyperparameter Sensitivity:** Performance is sensitive to the ensemble size $k$ and bonus scale $c$. For example, in River Swim (hard exploration), larger $k$ and $c$ improved performance, whereas in Puddle World and Mountain Car (easier exploration), increasing these parameters harmed performance.
