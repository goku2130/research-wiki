---
id: arxiv:2402.10221
type: paper
title: Mathematical Discoveries from Prompt Learning (Chen et al., 2024)
url: https://arxiv.org/abs/2402.10221
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

The provided text analyzes the convergence properties of the classical projected subgradient method (PSG) for nonsmooth convex optimization. The core problem investigates whether PSG equipped with a time-varying step-size $\eta_s = \frac{R}{L\sqrt{s}}$ achieves the optimal ergodic convergence rate of $\mathcal{O}(1/\sqrt{t})$, or if it inherently suffers from a sub-optimal $\log(t)$ factor as previously claimed in the literature. The optimization framework assumes a compact convex feasible set $\mathcal{X} \subset \mathbb{R}^n$ contained within a ball of radius $R$ centered at the optimal solution $x^*$, and a convex, Lipschitz continuous objective function $f$ with constant $L$ (i.e., $\|g\| \leq L$ for all $g \in \partial f(x)$).

The algorithmic recipe proceeds iteratively for $t \geq 1$. At each step, a subgradient $g_t \in \partial f(x_t)$ is computed, followed by an intermediate update $y_{t+1} = x_t - \eta_t g_t$. The next iterate is obtained via Euclidean projection onto the feasible set: $x_{t+1} = \operatorname{argmin}_{x \in \mathcal{X}} \|x - y_{t+1}\|$. The step-size sequence is defined as $\eta_s = \frac{R}{L\sqrt{s}}$ for $s = 1, \dots, t$. The convergence proof relies on a telescoping inequality derived from the subgradient condition and the projection theorem, establishing the recursive bound:
\[
f(x_s) - f(x^*) \leq \frac{1}{2\eta_s}(\|x_s - x^*\|^2 - \|x_{s+1} - x^*\|^2) + \frac{\eta_s}{2}L^2.
\]

The key quantitative results are formalized in two theorems. Theorem 1 proves that with the specified time-varying step-size, the standard uniform average satisfies the optimal bound:
\[
f \left( \frac{\sum_{s=1}^{t} x_s}{t} \right) - f(x^*) \leq \frac{3RL}{2\sqrt{t}}.
\]
Theorem 2 generalizes this analysis by introducing a weighting parameter $k \geq -1$, yielding:
\[
f \left( \frac{\sum_{s=1}^{t} \frac{1}{\eta_s^2} x_s}{\sum_{s=1}^{t} \frac{1}{\eta_s^2}} \right) - f(x^*) \leq \frac{\frac{R^2}{\eta_s^{k+1}} + L^2 \sum_{s=1}^{t} \frac{1}{\eta_s^{k-1}}}{2 \sum_{s=1}^{t} \frac{1}{\eta_s^2}}.
\]
Setting $k = -1$ recovers the classical bound $\frac{R^2 + L^2 \sum_{s=1}^t \eta_s^2}{2 \sum_{s=1}^t \eta_s}$, which for the given step-size produces the previously reported sub-optimal rate $\frac{2RL + RL \log t}{4(\sqrt{t+1}-1)}$. However, setting $k = 0$ immediately yields Theorem 1, demonstrating that the $\log(t)$ penalty is an artifact of the $k = -1$ analysis. For any $k > -1$, the bound remains $\mathcal{O}(1/\sqrt{t})$ without logarithmic factors. The authors term the case $k > 0$ a “weak” ergodic convergence rate, noting that the weighting scheme $\frac{\sum_{s=1}^{t} \eta_s^{-k} x_s}{\sum_{s=1}^{t} \eta_s^{-k}}$ increasingly emphasizes recent iterates as $k$ grows.

The stated limitations and scope of the analysis are strictly defined by the problem assumptions and the averaging mechanism. The results apply only to nonsmooth convex optimization over compact sets with Lipschitz continuous objectives and require the step-size sequence to be positive and non-increasing. The generalized bound depends critically on the choice of $k$; while $k=0$ recovers the standard uniform average, higher $k$ values alter the ergodic sense by biasing toward the final iterates. The authors note that the proof techniques extend to mirror descent and Nesterov’s dual averaging, but the paper’s quantitative guarantees are confined to the classical PSG framework. Additionally, the analysis assumes exact projection onto $\mathcal{X}$ and does not account for computational, numerical, or stochastic approximation errors.
