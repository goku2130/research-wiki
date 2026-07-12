---
id: arxiv:2409.02392
type: paper
title: Building Math Agents with Multi-Turn Iterative Preference Learning
url: https://arxiv.org/abs/2409.02392
retrieved: '2026-07-12'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Building Math Agents with Multi-Turn Iterative Preference Learning

### Core Problem
Large Language Models (LLMs) often struggle with complex mathematical reasoning involving basic arithmetic and symbolic computations. While integrating external tools (e.g., code interpreters) and Supervised Fine-Tuning (SFT) improve performance, existing direct preference learning algorithms (like DPO) are designed for single-turn chat tasks. They fail to address the complexities of multi-turn reasoning and the integration of external environment feedback, which are essential for tool-integrated mathematical agents.

### Method
The authors formulate the tool-integrated reasoning process as a Markov Decision Process (MDP) defined by the tuple $(\mathcal{S}, \mathcal{A}, H, \mathbb{P}, d_0)$, where $H$ is the episode length and $\mathbb{P}$ represents state transition kernels. The agent observes a state $s_h$ (history), takes an action $a_h$ (reasoning and code execution), and receives an observation $o_h$ from the environment.

#### Multi-Turn Preference Learning
To optimize the policy, the authors derive multi-turn variants of direct preference learning. Assuming deterministic transitions (where code execution results are determined by the history), they introduce the **Multi-turn Direct Preference Optimization (M-DPO)** and **Multi-turn KTO (M-KTO)** losses. A critical implementation detail is masking out user messages and external tool observations during training to ensure the model only learns to optimize its own actions.

The **M-DPO loss** is defined as:

$$
\mathcal{L}_{\mathrm{M-DPO}}(\theta)=-\sum_{((x,\tau^{w},\tau^{l})\in\mathcal{D}}\log\sigma\Big(\eta\sum_{h=1}^{H}\Big[\log\frac{\pi_{\theta,h}(a_{h}^{w}|s_{h}^{w})}{\pi_{\mathrm{ref},h}(a_{h}^{w}|s_{h}^{w})}-\log\frac{\pi_{\theta,h}(a_{h}^{l}|s_{h}^{l})}{\pi_{\mathrm{ref},h}(a_{h}^{l}|s_{h}^{l})}\Big]\Big)
$$

where $\tau^w$ and $\tau^l$ are the winning and losing trajectories, respectively.

The **M-KTO loss** is defined as:

$$
\mathcal{L}_{\mathrm{M-KTO}}(\theta)=\mathbb{E}_{x,y\sim\mathcal{D}}\big[\lambda_{y}-v(x,y)\big]
$$

with the utility function $u(x,y)=\eta\sum_{h=1}^{H}\log\frac{\pi_{u,h}(a_{h}|s_{h})}{\pi_{\mathrm{ref},h}(a_{h}|s_{h})}$.

#### Online Iterative Framework
The training follows an iterative recipe:
1. **Data Collection:** Sample trajectories using a behavior policy pair. To balance exploration and exploitation, the authors use **mixture sampling** (e.g., 20 trajectories from the current model and 10 from the previous iteration).
2. **Preference Labeling:** Use final result checking against gold answers to divide responses into winning and losing sets.
3. **Policy Update:** Optimize the model using M-DPO or M-KTO.
4. **Reference Model Update:** To optimize the non-regularized reward (improving in-domain performance), the reference model $\pi_{\text{ref}}$ is updated to the policy learned in the previous iteration ($\pi_{t, \text{ref}} = \pi_{t-1}$).

### Key Quantitative Results
The framework was evaluated on GSM8K and MATH benchmarks using various base models. Results demonstrate that iterative M-DPO significantly outperforms SFT and other RLHF baselines (like RAFT):

*   **Gemma-1.1-it-7B:** Performance increased from **77.5% $\to$ 83.9%** on GSM8K and **46.1% $\to$ 51.2%** on MATH.
*   **Gemma-2-it-9B:** Performance increased from **84.1% $\to$ 86.3%** on GSM8K and **51.0% $\to$ 54.5%** on MATH.
*   **Pass@n Analysis:** Preference learning primarily improves the quality of top responses (pass@1). When $n > 16$, the performance of preference-learned models converges with SFT models, indicating that the method elicits existing knowledge rather than injecting new knowledge.

### Limitations
*   **Reward Sparsity:** The current implementation relies on final result checking, meaning it cannot distinguish between two trajectories that are both correct or both incorrect.
*   **Environment Determinism:** The M-DPO/M-KTO losses are derived for deterministic transitions. They are not directly applicable to stochastic environments (e.g., human-in-the-loop chat) without constructing a value network for adaptive margins.
*   **Dependency on SFT:** The success of the preference learning stage is heavily dependent on a well-trained initial SFT model.
