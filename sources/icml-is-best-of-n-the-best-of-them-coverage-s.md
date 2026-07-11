---
id: icml:is-best-of-n-the-best-of-them-coverage-s
type: web
title: Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time
  Alignment
url: https://icml.cc/virtual/2025/poster/45322
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment

### Core Problem
The authors address the problem of **inference-time alignment**, where a pre-trained base policy $\pi_{\text{ref}}$ is used to generate candidate responses for a prompt $x$, and a reward model $\hat{r}$ is used to select the best response. A common heuristic for this is **Best-of-N (BoN) sampling**, which generates $N$ candidates and returns the one with the highest score under $\hat{r}$. 

The central issue is that BoN suffers from **reward hacking** (or overoptimization): as the computational budget $N$ increases, reward model errors at the tail of the response distribution accumulate. This causes the algorithm to select responses with high modeled rewards but poor actual task performance, leading to a degradation in quality as $N$ scales. The authors seek to determine the theoretical limits of inference-time alignment and develop an algorithm where performance scales monotonically with computation.

### Theoretical Framework
The authors formalize inference-time alignment as a statistical problem with the following components:
*   **Base Policy:** $\pi_{\text{ref}} : X \to \Delta(Y)$, mapping a prompt $x$ to a distribution over responses $y \in Y$.
*   **True Reward:** $r^* : X \times Y \to [0, R_{\max}]$, the unobserved ground-truth quality of a response.
*   **Reward Model:** $\hat{r}$, an imperfect proxy for $r^*$.
*   **Objective:** Produce a response $\hat{y}$ that approximately maximizes the true reward:

$$
\max_{\hat{y}} r^*(x, \hat{y})
$$

The authors identify an information-theoretic barrier: the ability to optimize $r^*$ is limited by the mean-squared error of $\hat{r}$ and the **coverage** of the base policy $\pi_{\text{ref}}$ over high-quality responses.

### Method: InferenceTimePessimism
To mitigate reward hacking, the authors introduce **InferenceTimePessimism**. Unlike BoN, which naively maximizes $\hat{r}$, this algorithm implements "pessimism in the face of uncertainty" to penalize responses that have high modeled rewards but high uncertainty.

**The Recipe:**
1.  **$\chi^2$-Regularization:** The algorithm applies $\chi^2$-regularization to the reward objective, a technique known to mitigate overoptimization.
2.  **Inference-Time Implementation:** This regularization is implemented purely at inference time using a novel **rejection sampling scheme**.
3.  **Decoupling Compute and Penalty:** The method separates the amount of computation spent (the number of samples $N$) from the strength of the uncertainty penalty (the regularization parameter).

### Key Results
The authors provide theoretical guarantees and empirical evidence to contrast BoN with InferenceTimePessimism:

**Theoretical Findings:**
*   **BoN Limitations:** BoN can achieve optimal regret only under stringent "uniform" or $L_\infty$-type coverage. It provably suffers from overoptimization as $N$ scales past a critical threshold and fails to provide tight guarantees under more realistic "average-case" or $L_1$-type coverage.
*   **Optimality of InferenceTimePessimism:** The proposed algorithm is **regret-optimal**, matching the theoretical "skyline" of the best possible reward achievable given $\hat{r}$.
*   **Scaling-Monotonicity:** InferenceTimePessimism is proven to be scaling-monotonic, meaning its performance does not degrade as $N \to \infty$.

**Quantitative/Empirical Results:**
*   **Task/Model:** Evaluated on the **GSM8K** dataset using the **OASST** reward model across various base policies $\pi_{\text{ref}}$.
*   **Performance Trends:** While BoN accuracy initially improves and then degrades as $N$ increases, InferenceTimePessimism remains monotone.
*   **Robustness:** InferenceTimePessimism outperforms BoN in maximal reward when the computational budget is untuned.

### Limitations
The authors note that there is an inherent, information-theoretic barrier to optimizing the true reward $r^*$ because it is unobserved; consequently, if the reward model $\hat{r}$ is highly inaccurate, the achievable quality is fundamentally limited regardless of the algorithm used.
