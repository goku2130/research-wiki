---
id: arxiv:2312.11456
type: paper
title: 'Iterative Preference Learning from Human Feedback: Bridging Theory and Practice
  for RLHF under KL-Constraint'
url: https://arxiv.org/abs/2312.11456
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Summary: Iterative Preference Learning from Human Feedback

### Core Problem
Existing Reinforcement Learning from Human Feedback (RLHF) methods, such as offline PPO and offline DPO, suffer from a lack of strategic exploration. These methods typically require preference datasets with uniform coverage over the entire prompt-response space to converge to an optimal policy. Because the response space is exponentially large, this coverage is rarely achieved in practice, leading to "reward hacking" (over-optimization of an imperfect proxy reward) and performance degeneration.

### Mathematical Formulation
The authors formulate RLHF as a reverse-KL regularized contextual bandit problem. Given a pretrained policy $\pi_0$, a prompt distribution $d_0$, and a ground-truth reward function $r^*(x, a)$, the goal is to find a policy $\pi$ that maximizes:

$$
J (\pi) = \mathbb {E} _ {x \sim d _ {0}} \left[ \mathbb {E} _ {a \sim \pi (\cdot | x)} [ r ^ {*} (x, a) ] - \eta D _ {\mathrm{KL}} (\pi (\cdot | x) \| \pi_ {0} (\cdot | x)) \right]
$$

where $\eta > 0$ is the KL penalty coefficient. Preferences are assumed to follow the Bradley-Terry model:

$$
\mathbb{P}(a^1 \succ a^2 | x, a^1, a^2) = \sigma(r^*(x, a^1) - r^*(x, a^2))
$$

The reward function is parameterized linearly as $r_{\theta}(x,a) = \langle \theta, \phi(x,a) \rangle$.

### Methods and Recipes

#### 1. Offline Learning with Pessimism (GSHF)
To mitigate distribution shift in offline datasets $\mathcal{D}_{\text{off}}$, the authors propose **Gibbs Sampling from Human Feedback (GSHF)** using the principle of pessimism (conservative reward estimation).
*   **Option I:** Penalize the reward by an expected uncertainty estimator $\Gamma^e(\pi, \nu, \mathcal{D}_{\text{off}})$.
*   **Option II:** Use a point-wise pessimistic reward: $\hat{r}(x, a) = r_{\text{MLE}}(x, a) - \beta \cdot \Gamma(x, a, \nu, \mathcal{D}_{\text{off}})$, where $\beta$ is a scaling factor and $\Gamma$ is an uncertainty bonus.

#### 2. Online Iterative Learning
The authors propose a non-symmetric structure for online interaction:
*   **Main Agent ($\pi_t^1$):** Exploits historical data by optimizing the MLE reward $r_{\text{MLE}}$ under the KL constraint.
*   **Enhancer ($\pi_t^2$):** Explores by maximizing relative uncertainty compared to $\pi_t^1$ while remaining within a KL-divergence bound:

$$
\eta \cdot \mathbb {E} _ {x \sim d _ {0}} D _ {\mathrm{KL}} (\pi'(\cdot|x), \pi_t^1(\cdot|x)) \leq \Gamma_t^m(\lambda, \pi_t^1, \pi')
$$

#### 3. Practical Implementations
*   **Direct Preference Learning with Pessimism:** A modified DPO loss incorporating an adaptive margin based on uncertainty differences:

$$
\mathcal {L} _ {\mathcal {D} _ {\mathrm{off}}} (\theta , \pi_ {0}) = \sum_{(x, a^w, a^l) \in \mathcal{D}_{\text{off}}} \log \sigma \left(\eta \log \frac {\pi_ {\theta} (a ^ {w} | x)}{\pi_ {0} (a ^ {w} | x)} - \eta \log \frac {\pi_ {\theta} (a ^ {l} | x)}{\pi_ {0} (a ^ {l} | x)} + (\Gamma (x , a ^ {w}) - \Gamma (x , a ^ {l}))\right)
$$

*   **Multi-step RSO:** To improve the low acceptance rate of rejection sampling, the authors propose a progressive approach: $\pi_0 \to \pi_0 \exp(\frac{1}{\eta_1}r) \to \dots \to \pi_0 \exp(\frac{1}{\eta_N}r)$, where $\eta_N = \eta$.

### Key Quantitative Results
*   **HH-RLHF Experiments:** Hybrid-GSHF-DPO outperformed DPO and RSO in gold reward and GPT-4 evaluations. It demonstrated higher robustness to out-of-distribution (OOD) data, showing a smaller difference ($\Delta$) between in-domain and OOD rewards.
*   **Scaling-up (Mistral-7B):** Using Online-GSHF-DPO with DPO as the oracle and rejection sampling for exploration, the model achieved a length-control win rate of **34.79%** on the AlpacaEval-2 benchmark (utilizing prompt augmentation and a filtered seed set).
*   **Multi-step RSO:** This method produced a reward-KL curve that strictly dominated the original RSO.

### Stated Limitations
*   **Length Bias:** The authors observed that as Hybrid-GSHF-DPO iterates, average output length increases (e.g., from 161 to 263 tokens), suggesting that reward models may favor wordier responses (reward hacking).
*   **Uncertainty Estimation:** Calculating analytical uncertainty for deep neural networks is an open problem; the authors rely on heuristic methods like ensembles.
*   **Alignment Tax:** The paper acknowledges the general challenge of performance degeneration when imposing strong optimization pressure on imperfect reward functions.
