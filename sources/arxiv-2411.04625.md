---
id: arxiv:2411.04625
type: paper
title: Sharp Analysis for KL-Regularized Contextual Bandits and RLHF
url: https://arxiv.org/abs/2411.04625
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

The authors, Zhao, Ye, Gu, and Zhang, address the problem of the sample complexity of KL-regularized contextual bandits and Reinforcement Learning from Human Feedback (RLHF). Existing theoretical analyses for KL-regularized RLHF typically yield an $\mathcal{O}(1/\epsilon^2)$ sample complexity, similar to non-regularized objectives, failing to capture the empirical benefits of KL-regularization. The core problem is to theoretically demonstrate the advantages of KL-regularization and the impact of reference policy coverage on sample complexity, aiming for a sharper $\mathcal{O}(1/\epsilon)$ rate.

The proposed method is a **Two-stage Mixed-Policy Sampling (TMPS) algorithm** for KL-regularized contextual bandits (Algorithm 1) and its extension for RLHF with preference feedback (Algorithm 2).

**Algorithm 1: Two-stage Mixed-Policy Sampling (TMPS) for KL-Regularized Contextual Bandits**

1.  **Input**: KL-regularization coefficient $\eta$, desired suboptimality $\epsilon$, reference policy $\pi_0$, and reward function class $\Theta$.
2.  **Stage 1: Initial Data Collection and Reward Estimation**
    *   For $i = 1, \ldots, m$ rounds:
        *   Sample context $x_i^0 \sim d_0$ and action $a_i^0 \sim \pi_0(\cdot | x_i^0)$.
        *   Observe reward $r_i^0 = R(\theta_*, x_i^0, a_i^0) + \epsilon_i^0$, where $\epsilon_i^0$ is random noise.
    *   Compute the least squares estimate $\widehat{\theta}_0$ of the reward function based on the collected data $D_0 = \{(x_i^0, a_i^0, r_i^0)\}_{i=1}^m$:

$$
\widehat {\theta} _ {0} \leftarrow \underset {\theta \in \Theta} {\text{argmin}} \sum_ {i = 1} ^ {m} (R (\theta , x _ {i} ^ {0}, a _ {i} ^ {0}) - r _ {i} ^ {0}) ^ {2}
$$

    *   Apply a planning oracle to compute an intermediate policy $\pi_{\widehat{\theta}_0}^\eta$:

$$
\pi_{\widehat{\theta}_0}^\eta (\cdot |\cdot)\propto \pi_0(\cdot |\cdot)\exp \bigl (\eta R(\widehat{\theta}_0,\cdot ,\cdot)\bigr)
$$

3.  **Stage 2: Refined Data Collection and Reward Estimation**
    *   For $i = 1, \ldots, n$ rounds:
        *   Sample context $x_i \sim d_0$ and action $a_i \sim \pi_{\widehat{\theta}_0}^\eta(\cdot |x_i)$.
        *   Observe reward $r_i = R(\theta_*, x_i, a_i) + \epsilon_i$, where $\epsilon_i$ is random noise.
    *   Compute a refined least squares estimate $\widehat{\theta}$ of the reward function using data from both stages, $D_0$ and $\{(x_i, a_i, r_i)\}_{i=1}^n$:

$$
\widehat {\theta} \leftarrow \underset {\theta \in \Theta} {\text{argmin}} \sum_ {i = 1} ^ {m} (R (\theta , x _ {i} ^ {0}, a _ {i} ^ {0}) - r _ {i} ^ {0}) ^ {2} + \sum_ {i = 1} ^ {n} (R (\theta , x _ {i}, a _ {i}) - r _ {i}) ^ {2}
$$

4.  **Output**: The final policy $\pi_{\widehat{\theta}}^{\eta}(\cdot |\cdot)\propto \pi_0(\cdot |\cdot)\exp \bigl (\eta R(\widehat{\theta},\cdot ,\cdot)\bigr)$.

**Algorithm 2: Two-stage Mixed-Policy Sampling from Preference Feedback (TMPS-PF)** (Extension of Algorithm 1 for RLHF)

1.  **Input**: $\eta, \epsilon, \pi_0, \Theta$.
2.  **Stage 1: Initial Data Collection and Reward Estimation**
    *   For $i = 1, \ldots, m$ rounds:
        *   Sample context $\widetilde{x}_i \sim d_0$ and two actions $\widetilde{a}_i^1, \widetilde{a}_i^2 \sim \pi_0(\cdot|\widetilde{x}_i)$.
        *   Observe preference label $\widetilde{y}_i \in \{0, 1\}$ from a preference oracle (Bradley-Terry model).
    *   Compute the Maximum Likelihood Estimator (MLE) $\widehat{\theta}_0$ of the reward function based on the collected data:

$$
\widehat {\theta} _ {0} \leftarrow \underset {\theta} {\text{argmax}} \sum_ {i = 1} ^ {m} \widetilde {y} _ {i} \cdot \log \sigma (R (\theta , \widetilde {x} _ {i}, \widetilde {a} _ {i} ^ {1}) - R (\theta , \widetilde {x} _ {i}, \widetilde {a} _ {i} ^ {2})) + (1 - \widetilde {y} _ {i}) \cdot \log \sigma (R (\theta , \widetilde {x} _ {i}, \widetilde {a} _ {i} ^ {2}) - R (\theta , \widetilde {x} _ {i}, \widetilde {a} _ {i} ^ {1}))
$$

    *   Apply a planning oracle to compute an intermediate policy $\pi_{\widehat{\theta}_0}^\eta$:

$$
\pi_{\widehat{\theta}_0}^\eta (\cdot |\cdot)\propto \pi_0(\cdot |\cdot)\exp \bigl (\eta R(\widehat{\theta}_0,\cdot ,\cdot)\bigr)
$$

3.  **Stage 2: Refined Data Collection and Reward Estimation**
    *   For $i = 1, \ldots, n$ rounds:
        *   Sample context $x_i \sim d_0$ and two actions $a_i^1, a_i^2 \sim \pi_{\widehat{\theta}_0}^\eta(\cdot |x_i)$.
        *   Observe preference label $y_i \in \{0, 1\}$ from the preference oracle.
    *   Compute a refined MLE $\widehat{\theta}$ of the reward function using data from both stages:

$$
\widehat {\theta} \leftarrow \underset {\theta} {\text{argmax}} \sum_ {i = 1} ^ {m} \widetilde {y} _ {i} \cdot \log \sigma (R (\theta , \widetilde {x} _ {i}, \widetilde {a} _ {i} ^ {1}) - R (\theta , \widetilde {x} _ {i}, \widetilde {a} _ {i} ^ {2})) + (1 - \widetilde {y} _ {i}) \cdot \log \sigma (R (\theta , \widetilde {x} _ {i}, \widetilde {a} _ {i} ^ {2}) - R (\theta , \widetilde {x} _ {i}, \widetilde {a} _ {i} ^ {1})) + \sum_ {i = 1} ^ {n} y _ {i} \cdot \log \sigma (R (\theta , x _ {i}, a _ {i} ^ {1}) - R (\theta , x _ {i}, a _ {i} ^ {2})) + (1 - y _ {i}) \cdot \log \sigma (R (\theta , x _ {i}, a _ {i} ^ {2}) - R (\theta , x _ {i}, a _ {i} ^ {1}))
$$

4.  **Output**: The final policy $\pi_{\widehat{\theta}}^{\eta}(\cdot|\cdot)\propto\pi_{0}(\cdot|\cdot)\exp\big(\eta R(\widehat{\theta},\cdot,\cdot)\big)$.

**Key Formulas (in LaTeX):**

*   **KL-regularized objective function**:

$$
Q(\pi) = \mathbb{E}_{x \sim d_0} \mathbb{E}_{a \sim \pi(\cdot|x)} \left[ R(\theta_*, x, a) - \eta^{-1} \log \frac{\pi(a|x)}{\pi_0(a|x)} \right]
$$

*   **Policy Improvement Oracle**:

$$
\pi_\theta^\eta(\cdot|x) := \underset{\pi(\cdot|x) \in \Delta(\mathcal{A})}{\text{argmax}} \mathbb{E}_{a \sim \pi(\cdot|x)} \left[ R(\theta, x, a) - \eta^{-1} \log \frac{\pi(a|x)}{\pi_0(a|x)} \right] \propto \pi_0(\cdot|x) \cdot \exp\left( \eta R(\theta, x, \cdot) \right)
$$

*   **Data Coverage (Definition 3.5)**: $D^2$ is the minimum positive real number satisfying $\forall (x, a) \in \mathcal{X} \times \mathcal{A}$, $\pi(a|x) > 0$, we have for any pair of $\theta, \theta' \in \Theta$,

$$
\frac{[R(\theta', x, a) - R(\theta, x, a)]^2}{\mathbb{E}_{x' \sim d_0, a' \sim \pi_0(\cdot|x')} [(R(\theta', x', a') - R(\theta', x', a'))^2]} \leq D^2
$$

    For RLHF (Definition 4.5), there exists $b: \mathcal{X} \to [-B, B]$ such that:

$$
\frac{|R(\theta', x, a) - R(\theta, x, a) - b(x)|^2}{\mathbb{E}_{x' \sim d_0} \text{Var}_{a' \sim \pi_0(\cdot|x')} [R(\theta', x', a') - R(\theta, x', a')]} \le D^2
$$

*   **Bradley-Terry Model (Definition 4.2)**:

$$
\mathbb{P}(x, a_1, a_2) = \frac{\exp(R(\theta_*, x, a_1))}{\exp(R(\theta_*, x, a_1)) + \exp(R(\theta_*, x, a_2))} = \sigma(R(\theta_*, x, a_1) - R(\theta_*, x, a_2))
$$

*   **Suboptimality decomposition (Lemma 3.9)**:

$$
Q(\pi^*) - Q(\pi_{\widehat{\theta}}^\eta) \leq \eta \mathbb{E}_{x \sim d_0} \left[ \sum_{a \in \mathcal{A}} \pi_f^\eta(a|x) \Delta^2(x, a) \right]
$$

    where $\Delta(x, a) = R(\widehat{\theta}, x, a) - R(\theta_*, x, a)$ and $f(\cdot, \cdot) = \gamma R(\widehat{\theta}, \cdot, \cdot) + (1 - \gamma) R(\theta_*, \cdot, \cdot)$ for $\gamma \in (0, 1)$.

**Key Quantitative Results and Numbers:**

*   **Lower Bound (Theorem 3.6 & 4.6)**: For KL-regularized contextual bandits and RLHF, the sample complexity is at least $\Omega\big(\min(\frac{\eta \log N_{\mathcal{R}}(\epsilon)}{\epsilon}, \frac{\log N_{\mathcal{R}}(\epsilon)}{\epsilon^2})\big)$ to achieve an $\epsilon$ suboptimality gap when $\epsilon \in (0, 1/256)$ and $\eta > 4$. This implies an $\Omega(\eta \log N_{\mathcal{R}}(\epsilon)/\epsilon)$ lower bound when $\epsilon$ is sufficiently small.
*   **Upper Bound for Contextual Bandits (Theorem 3.8)**: With sample sizes $m = \Theta ( \eta ^ {2} D ^ {2} \cdot B ^ {2} \log ( 2 N _ { \mathcal { R } } ( \epsilon _ { c } ) / \delta ) )$ and $n = \Theta ( \eta / \epsilon \cdot B ^ {2} \log ( N _ { \mathcal { R } } ( \epsilon _ { c } ) / \delta ) )$, where $\epsilon _ { c } = \text { m i n } \{ \frac { \epsilon } { 2 ( 1 + c _ { m , n } ^ { - 1 } ) B } , \frac { 1 } { 8 ( 1 + c _ { m , n } ) B \eta ^ { 2 } D ^ { 2 } } \}$, Algorithm 1 achieves $\mathcal{O}(\epsilon)$ optimality with probability at least $1 - 5\delta$. The sample complexity is $\mathcal{O}(\max(\eta^2 D^2, \eta/\epsilon) \log N_{\mathcal{R}}(\epsilon/\delta))$.
*   **Upper Bound for RLHF (Theorem 4.7)**: With sample sizes $m = \Theta(\eta^2 D^2 \cdot e^B \log(N_{\mathcal{R}}(\epsilon_c)/\delta))$ and $n = \Theta(\eta/\epsilon \cdot e^B \log(N_{\mathcal{R}}(\epsilon_c)/\delta))$, where $\epsilon_c = \min\{\frac{\epsilon}{2(1+c_m,n)} e^B, \frac{1}{8(1+c_m,n)} e^B \eta^2 D^2\}$, Algorithm 2 achieves $\mathcal{O}(\epsilon)$ optimality with probability at least $1 - 6\delta$. The sample complexity is $\mathcal{O}(\max(\eta^2 D^2, \eta/\epsilon) \log N_{\mathcal{R}}(\epsilon/\delta))$.
*   **Dependence on Coverage**: The proposed algorithms achieve an *additive* dependence on the coverage coefficient $D^2$ (e.g., $\mathcal{O}(\eta^2 D^2 + \eta/\epsilon)$), which is sharper than previous results that often show a *multiplicative* dependence (e.g., $\mathcal{O}(C_{\rho_{\text{KL}}} \eta/\epsilon)$ under local coverage, or $\mathcal{O}(C_{\rho_{\text{KL}}}^2/\epsilon^2)$ in prior work).
*   **Reward Scale**: The RLHF sample complexity includes a factor of $e^B$, where $B$ is the bound on the reward function.

**Stated Limitations:**

*   The current analysis for KL-regularized RLHF (Theorem 4.7) includes an $e^B$ factor in the sample size, which can be exponentially large. This term is attributed to the non-linearity of the link function for the preference model and is common in RLHF literature.
*   The analysis is primarily for contextual bandits and RLHF, and extending these techniques to the Markov Decision Process (MDP) setting is left for future work.
*   It is an open question whether an additive dependence on global and local coverage conditions (Definitions 3.10 and 3.11) can be achieved, similar to the data coverage condition. The current result under local-coverage (Corollary 3.13, 4.10) shows a multiplicative dependence $\Theta(C_{\rho_{\text{KL}}} \eta/\epsilon)$.
