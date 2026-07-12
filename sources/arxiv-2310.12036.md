---
id: arxiv:2310.12036
type: paper
title: Identity Preference Optimization (IPO)
url: https://arxiv.org/abs/2310.12036
retrieved: '2026-07-12'
maturity: comprehensive
topic: dpo-and-preference-optimization
---

# Identity Preference Optimization (IPO)

### Core Problem
The authors address a theoretical and practical vulnerability in Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO). Both methods rely on the **Bradley-Terry (BT) model**, which assumes that pairwise preferences can be substituted with pointwise rewards (Elo scores). 

The core issue is that the BT model is unbounded; when preferences are deterministic (probability of 1 or 0), the required reward difference becomes infinite. In practice, this causes the KL-regularization term—intended to keep the learned policy $\pi$ close to a reference policy $\pi_{\text{ref}}$—to be ignored. Consequently, DPO is prone to overfitting, often converging to "greedy" deterministic policies regardless of the regularization parameter $\tau$, especially in finite data regimes where empirical preferences may appear deterministic even if the true preferences are not.

### The $\Psi$PO Paradigm and IPO Method
To resolve this, the authors propose $\Psi$-Preference Optimization ($\Psi$PO), a general theoretical framework that expresses the objective in terms of pairwise preferences rather than pointwise rewards. The general objective is defined as:

$$
\max_{\pi}\mathop{\mathbb{E}}_{\substack{x\sim \rho \\ y\sim \pi (\cdot |x)\\ y^{\prime}\sim \mu (\cdot |x)}}\left[\Psi (p^{*}(y\succ y^{\prime}|x))\right] - \tau D_{\mathrm{KL}}(\pi \mid \mid \pi_{\mathrm{ref}})
$$

where $\Psi$ is an arbitrary non-decreasing mapping. The authors demonstrate that RLHF and DPO are special cases of $\Psi$PO where $\Psi(q) = \log(q/(1 - q))$.

**Identity Preference Optimization (IPO)** is a specific instance of $\Psi$PO where $\Psi$ is the **identity mapping**. By using a bounded mapping, IPO ensures that KL-regularization remains effective even with deterministic preferences. The IPO objective is:

$$
\max _ {\pi} p _ {\rho} ^ {*} (\pi \succ \mu) - \tau D _ {\mathrm{KL}} (\pi | | \pi_{\mathrm{ref}})
$$

### Implementation Recipe
To avoid the costs of RL and reward modeling, the authors derive a computationally efficient offline sampled loss function:

1. **Define the log-likelihood ratio gap** $h_\pi$ for a context $x$ and actions $y, y'$:

$$
h _ {\pi} (y, y ^ {\prime}, x) = \log \left(\frac {\pi (y | x) \pi_ {\mathrm{ref}} (y ^ {\prime} | x)}{\pi (y ^ {\prime} | x) \pi_ {\mathrm{ref}} (y | x)}\right)
$$

2. **Initialize** the policy $\pi$ as the reference policy $\pi_{\text{ref}}$.
3. **Minimize the sampled loss** over the preference dataset $\mathcal{D}$ (consisting of preferred $y_w$ and dispreferred $y_l$ actions):

$$
\underset {(y _ {w}, y _ {l}, x) \sim D} {\mathbb {E}} \left(h _ {\pi} (y _ {w}, y _ {l}, x) - \frac {\tau^ {- 1}}{2}\right) ^ {2}
$$

   This effectively regresses the gap between the policy's log-likelihood ratios and the reference policy's ratios to a constant $\frac{\tau^{-1}}{2}$.

### Key Quantitative Results
The authors tested IPO and DPO in a bandit setting with three actions $\{y_a, y_b, y_c\}$ using the Adam optimizer (learning rate 0.01, mini-batch size 9, 18,000 steps).

*   **Avoidance of Greedy Policies:** In a total ordering dataset $\mathcal{D}_1 = \{(y_a, y_b), (y_b, y_c), (y_a, y_c)\}$, DPO converged to a deterministic policy ($\pi(y_a)=1$) for all values of $\tau$, ignoring $\pi_{\text{ref}}$. IPO remained close to $\pi_{\text{ref}}$ when $\tau$ was large and only became greedy as $\tau \to 0$.
*   **Action Preservation:** In scenarios where an action never wins in the dataset, DPO pushed that action's probability to 0 regardless of $\tau$. IPO maintained the action's probability relative to the strength of $\tau$.
*   **Unobserved Pairs:** With a dataset $\mathcal{D}_3 = \{(y_a, y_b), (y_b, y_a)\}$ where the pair $(y_a, y_c)$ was unobserved, DPO ignored $\pi_{\text{ref}}$ entirely, while IPO's solution scaled gradually with $\tau$.

### Stated Limitations
The authors note that their empirical evaluations were conducted on "simple bandit examples" and "minimal experiments." They state that future work is required to scale these findings to more complex settings, specifically the training of large language models on human preference data.
