---
id: arxiv:2001.08361
type: paper
title: Scaling Laws for Neural Language Models
url: https://arxiv.org/abs/2001.08361
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

# Scaling Laws for Neural Language Models

### Core Problem
The researchers investigate the empirical relationship between the performance of Transformer language models—measured by cross-entropy loss—and three primary scale factors: the number of model parameters ($N$), the size of the dataset ($D$), and the amount of compute used for training ($C$). The goal is to determine if these relationships follow predictable patterns and to identify the optimal allocation of a fixed compute budget to maximize performance.

### Method and Recipe
The study utilizes decoder-only Transformer models trained on **WebText2**, a dataset containing $2.29 \times 10^{10}$ tokens. The primary performance metric is the cross-entropy loss averaged over a 1024-token context.

**Step-by-Step Procedure:**
1.  **Parameter Definition:** The authors define model size $N$ as the number of non-embedding parameters.
2.  **Compute Estimation:** Training compute is estimated as $C \approx 6NBS$, where $B$ is the batch size and $S$ is the number of training steps. This estimate excludes context-dependent terms, which are negligible when $d_{model} \gg n_{ctx}/12$.
3.  **Scaling Analysis:** The authors train models across wide ranges of $N$, $D$, and $C$ to fit power-law equations.
4.  **Overfitting Study:** By varying $N$ and $D$ simultaneously and using early stopping, they characterize the boundary where models begin to overfit.
5.  **Batch Size Optimization:** They measure the "critical batch size" ($B_{crit}$), the point providing an optimal tradeoff between training time (serial steps) and compute efficiency.
6.  **Budget Allocation:** Using the derived power laws, they determine the optimal growth rates for $N$, $B$, and $S$ as the total compute budget $C_{min}$ increases.

### Key Formulas
The performance $L$ follows power-law relationships with scale factors:

*   **Model Size:** $L(N) = (N_c / N)^{\alpha_N}$
*   **Dataset Size:** $L(D) = (D_c / D)^{\alpha_D}$
*   **Minimum Compute:** $L(C_{min}) = (C_c^{min} / C_{min})^{\alpha_{C}^{min}}$
*   **Overfitting Boundary:** To avoid a performance penalty, dataset size should scale sub-linearly with model size: $D \propto N^{\alpha_N / \alpha_D} \sim N^{0.74}$
*   **Critical Batch Size:** $B_{crit}(L) \approx B_* / L^{1/\alpha_B}$

### Quantitative Results
The authors find that performance depends strongly on scale but weakly on architectural hyperparameters (e.g., depth vs. width).

**Empirical Fit Parameters:**
*   $\alpha_N \approx 0.076, N_c \approx 8.8 \times 10^{13}$
*   $\alpha_D \approx 0.095, D_c \approx 5.4 \times 10^{13}$
*   $\alpha_{C}^{min} \approx 0.050, C_c^{min} \approx 3.1 \times 10^8 \text{ PF-days}$
*   $\alpha_B \approx 0.21, B_* \approx 2 \times 10^8 \text{ tokens}$
*   $\alpha_S \approx 0.76, S_c \approx 2.1 \times 10^3 \text{ steps}$

**Optimal Compute Allocation:**
For a fixed compute budget $C_{min}$, the optimal parameters scale as:
*   **Model Size:** $N(C_{min}) \propto C_{min}^{0.73}$
*   **Batch Size:** $B \propto C_{min}^{0.24}$
*   **Dataset Size:** $D \propto C_{min}^{0.27}$
*   **Training Steps:** $S_{min} \propto C_{min}^{0.03}$

This indicates that most of a compute increase should be allocated to increasing model size, with a modest increase in data and batch size, and negligible increase in serial training steps.

### Stated Limitations
*   **Compute Estimation:** The $C \approx 6NBS$ formula may be confounded if context length is very large ($n_{ctx} \gtrsim 12d_{model}$).
*   **Data Regime:** Fits for $L(N, D)$ were poor for very small dataset sizes.
*   **Hyperparameters:** The authors did not thoroughly experiment with regularization or data augmentation, which could alter results.
*   **Extrapolation:** The authors are less confident in the $B_{crit}(L)$ prediction for loss values far outside their explored range.
*   **Learning Rates:** Optimal learning rates are sensitive to the target loss; the authors did not experiment with higher learning rates for short runs that did not converge.
