---
id: arxiv:1911.09615
type: paper
title: Sample-Efficient Reinforcement Learning with Maximum Entropy Mellowmax Episodic
  Control
url: https://arxiv.org/abs/1911.09615
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

# Maximum Entropy Mellowmax Episodic Control (MEMEC)

### Core Problem
Deep Reinforcement Learning (DRL) agents typically suffer from poor sample efficiency, requiring significantly more experience than biological agents to reach similar performance levels. While Episodic Control (EC) methods—which use non- or semi-parametric models to store and retrieve previously experienced transitions—improve this, they often rely on naive $\epsilon$-greedy exploration. The core problem addressed by MEMEC is the need for a more principled exploration strategy that can improve sample efficiency and final rewards without altering the underlying objective function to a maximum entropy RL objective.

### Method
MEMEC integrates Episodic Control (either Model-Free EC [MFEC] or Neural EC [NEC]) with a maximum entropy mellowmax policy for action selection.

**1. Episodic Control Foundation:**
*   **MFEC:** Uses a fixed-size table per action to store the highest return ever obtained for a state-action pair. It estimates values based on the average of the $k$-nearest states.
*   **NEC:** Employs a convolutional neural network (CNN) to map states to embeddings (keys) and a Differentiable Neural Dictionary (DND) to store and retrieve the $k$ most similar keys and their associated $Q$-values via an inverse distance weighted kernel.

**2. Mellowmax Exploration Recipe:**
Instead of $\epsilon$-greedy sampling, MEMEC samples actions according to a Boltzmann distribution with a state-dependent temperature $\beta$.
*   **Step 1:** Compute the state-action values $Q(s, a)$ using the EC model (MFEC or NEC).
*   **Step 2:** Apply the mellowmax operator $\text{mm}_\omega(Q)$ to the $Q$-values. This operator acts as a non-expansion in the infinity norm, ensuring stability.
*   **Step 3:** Determine the optimal temperature $\beta$ by solving for the root of the maximum entropy constraint equation using a root-finding algorithm (e.g., Brent’s method).
*   **Step 4:** Sample the action $a$ from the resulting Boltzmann distribution $\pi_{mm}(a|s)$.

### Key Formulas
The **Boltzmann operator** is defined as:

$$
\text{boltz}_{\beta}(Q)=\frac{\sum_{i=1}^{n}Q_{i}e^{\beta Q_{i}}}{\sum_{i=1}^{n}e^{\beta Q_{i}}}
$$

The **mellowmax operator** (with hyperparameter $\omega$) is:

$$
\text{mm}_{\omega}(Q)=\frac{\log\left(\frac{1}{n}\sum_{i=1}^{n}e^{\omega Q_{i}}\right)}{\omega}
$$

The **maximum entropy mellowmax policy** is:

$$
\pi_{m m}(a|s)=\frac{e^{\beta Q(s, a)}}{\sum_{a \in \mathcal{A}} e^{\beta Q(s, a)}}
$$

The temperature $\beta$ is solved via the following equation:

$$
\sum_{a \in \mathcal{A}} e^{\beta(Q(s, a)-\text{m m}_{\omega}(Q(s, \cdot)))} (Q(s, a)-\text{m m}_{\omega}(Q(s, \cdot)))=0
$$

### Key Quantitative Results
MEMEC was evaluated against $\epsilon$-greedy EC and a Dueling Double DQN (D3QN) baseline across classic control, gridworld, and Atari environments.

**Atari Performance (Final Averaged Rewards):**
MFEC-based MEMEC significantly outperformed $\epsilon$-greedy MFEC and D3QN in several games:
*   **Q*Bert:** MEMEC achieved $11610 \pm 898$, compared to $\epsilon$-greedy MFEC ($3896 \pm 710$) and D3QN ($1480 \pm 271$).
*   **Ms. Pac-Man:** MEMEC achieved $6968 \pm 787$, compared to $\epsilon$-greedy MFEC ($4178 \pm 510$) and D3QN ($1851 \pm 98$).
*   **Space Invaders:** MEMEC achieved $1029 \pm 157$, compared to $\epsilon$-greedy MFEC ($672 \pm 13$) and D3QN ($756 \pm 30$).
*   **Bowling:** MEMEC achieved $68 \pm 7$, slightly better than $\epsilon$-greedy MFEC ($62 \pm 8$) and D3QN ($22 \pm 16$).
*   **Pong:** MEMEC performed worse than $\epsilon$-greedy MFEC ($7 \pm 4$ vs $17 \pm 2$).

In classic control (Acrobot), MEMEC achieved higher final rewards and faster learning speeds than other exploration strategies. In gridworlds, MFEC with softmax-based exploration performed best, while NEC generally struggled.

### Limitations
The primary limitation of MEMEC is the sensitivity of the $\omega$ hyperparameter. The optimal $\omega$ varies across different domains (e.g., $\omega=25$ for Pong, $\omega=60$ for Ms. Pac-Man), necessitating a grid search that is computationally prohibitive in high-dimensional domains like Atari.
