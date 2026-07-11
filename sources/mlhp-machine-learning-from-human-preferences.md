---
id: mlhp:machine-learning-from-human-preferences
type: web
title: Machine Learning from Human Preferences
url: https://mlhp.stanford.edu/src/chap4.html
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

# Chapter 4: Decisions in Preference Learning

**Core Problem**
This chapter addresses sequential decision-making under preference uncertainty, reframing the objective from passive measurement (reducing uncertainty about a preference function) to active reward maximization (managing uncertainty to make optimal decisions). The central challenge is the exploration–exploitation tradeoff: balancing the selection of items believed to yield high utility against the testing of uncertain options to gather information for future rounds. The theoretical framework assumes stationarity, meaning item utilities remain fixed over time. While this enables standard regret bounds, the source notes that real-world systems (e.g., evolving user tastes or updated LLMs) often violate this assumption, requiring explicit drift-handling mechanisms.

**Method and Algorithmic Recipe**
Thompson Sampling (TS) is presented as a Bayesian decision rule that converts posterior beliefs into a randomized selection policy. The algorithm operates in two primary regimes:
1. **Linear Objectives:** The agent maintains a posterior over a latent user embedding $U_i$. At each round $t+1$, the agent computes the MAP estimate $\hat{U}_i$, approximates the posterior via Laplace approximation, samples a draw $\tilde{U}_i^{(t)}$ from the approximate posterior, and selects the item maximizing the sampled expected utility.
2. **Nonlinear Objectives:** The method extends to Gaussian Processes (GPs) by placing a prior over the utility function $h_i$. The agent approximates the intractable GP classification posterior using Laplace approximation, samples a function realization $\tilde{h}_i$ from the approximate posterior, and greedily selects the item with the highest sampled utility. In both cases, exploration emerges naturally from posterior variance without explicit exploration schedules.

**Key Formulas**
Preference feedback follows a logistic link: 

$$
p(Y_{ij}=1 \mid U_i, V_j) = \sigma(U_i^\top V_j), \tag{4.1}
$$

The log-posterior is defined as:

$$
\ell(U_i) = \log p(U_i) + \sum_{j \in \mathcal{D}_t} \log p(Y_{ij} \mid U_i). \tag{4.2}
$$

The Laplace approximation quadratically expands $\ell(U_i)$ around $\hat{U}_i$, yielding a Gaussian posterior:

$$
p(U_i \mid \mathcal{D}_t) \approx \mathcal{N}(\hat{U}_i, G^{-1}), \tag{4.4}
$$

where the Hessian is $G = I + \sum_{j} p_j(1-p_j) V_j V_j^\top$ (4.5). The MAP estimate solves:

$$
\hat{U}_i = \arg\max_{U_i} \Big[\sum_{j\in\mathcal{D}_t} Y_{ij}\log\sigma(U_i^\top V_j) + (1-Y_{ij})\log(1-\sigma(U_i^\top V_j)) - \tfrac{1}{2}|U_i|^2 \Big]. \tag{4.6}
$$

The TS selection rule is $j^* = \arg\max_j \tilde{U}_i^{(t)\top} V_j$ (4.9). For nonlinear cases, a GP prior $h_i \sim \mathcal{GP}(m(\cdot), k(\cdot,\cdot))$ (4.10) is used, typically with an RBF kernel $k(V, V') = \sigma^2 \exp\left(-\frac{\|V - V'\|^2}{2\ell^2}\right)$ (4.16). The approximate posterior at a new point $V_*$ is $\mathcal{N}(\mu_t(V_*), \sigma_t^2(V_*))$ (4.29–4.30), and TS selects $j^* = \arg\max_j \tilde{h}_i(V_j)$ (4.32).

**Quantitative Results and Numbers**
The source provides a concrete simulation illustrating TS dynamics with three items and a one-dimensional user embedding. After five observations, the posterior is $\mathcal{N}(1.2, 0.5^2)$. A high sample draw of $\tilde{U}=1.6$ yields expected utilities of $0.80$, $1.60$, and $-0.48$ for items with features $0.5$, $1.0$, and $-0.3$, leading to the selection of item 2. A low draw of $\tilde{U}=0.7$ maintains this choice with a narrower margin. A rare tail draw of $\tilde{U}=-0.1$ produces utilities of $-0.05$, $-0.10$, and $0.03$, triggering exploration by selecting item 3. Historically, TS was proposed in 1933 for clinical trials, fell into obscurity, and was rediscovered in the 2010s for online advertising and recommendation systems, where it empirically outperforms frequentist alternatives like UCB.

**Stated Limitations**
The framework's regret guarantees strictly depend on the stationarity assumption; when utilities drift, standard bounds fail and algorithms must incorporate discounting or change-point detection. The Laplace approximation requires the posterior to be unimodal; applying it to multimodal posteriors causes TS to either overconfidently exploit suboptimal items or waste rounds exploring. The source explicitly warns against adding explicit exploration bonuses (e.g., UCB-style terms) to TS, as the inherent posterior randomization already provides the necessary exploration, and adding bonuses disrupts this balance. For nonlinear GP implementations, exact inference is intractable, and practical scalability requires external libraries like `gpytorch` or `scikit-learn`. The provided text concludes mid-section, leaving dueling bandit regret bounds and advanced acquisition functions partially outlined.
