---
id: arxiv:2603.01162
type: paper
title: 'Demystifying Group Relative Policy Optimization: Its Policy Gradient is a
  U-Statistic'
url: https://arxiv.org/abs/2603.01162
retrieved: '2026-07-10'
maturity: comprehensive
topic: grpo
---

**Core Problem**
Group Relative Policy Optimization (GRPO) has become a foundational reinforcement learning with verifiable rewards (RLVR) algorithm for scaling reasoning capabilities in large language models, yet its theoretical properties remain largely uncharacterized. The authors identify four critical gaps: the underlying reasons for GRPO’s empirical effectiveness, the statistical rationale for replacing a critic network with a group mean, the convergence behavior of the algorithm, and the principled selection of the number of sampled outputs per prompt. To bridge these gaps, the work establishes a unified statistical framework that maps GRPO’s policy gradient to classical U-statistics, enabling rigorous finite-sample and asymptotic analysis.

**Methodological Recipe & U-Statistic Framework**
GRPO operates as a critic-free policy-based algorithm. For each input prompt, the model samples $G$ distinct reasoning traces. Instead of training a separate value network to estimate policy quality, the algorithm computes the average reward across this group and uses it as a baseline proxy for the critic. The policy parameters are then updated via a policy gradient estimator that leverages this group-relative advantage. The authors demonstrate that this gradient estimator is inherently a U-statistic. By applying the Hoeffding decomposition, they decompose the estimator into orthogonal components, revealing that the first-order projection dominates the variance. This statistical mapping provides a principled justification for the group-mean approximation and enables precise characterization of both the gradient estimator and the suboptimality gap.

**Key Formulas**
The theoretical foundation rests on two central expressions. First, a U-statistic of order $m$ is defined as:
$$U_n = \binom{n}{m}^{-1} \sum_{1 \le i_1 < \dots < i_m \le n} h(X_{i_1}, \dots, X_{i_m})$$
where $h$ is a symmetric kernel function. Second, the Hoeffding decomposition expresses the estimator’s deviation from its mean as:
$$U_n - \theta = \frac{2}{n} \sum_{i=1}^n h_1(X_i) + R_n$$
where $\theta = \mathbb{E}[h(X_1, X_2)]$ and $h_1(x) = \mathbb{E}[h(x, X_2)] - \theta$ denotes the first-order projection. In the non-degenerate case, the first-order term governs the variance at rate $O(n^{-1})$, while the remainder decays faster at $O(n^{-2})$. This decomposition directly characterizes GRPO’s gradient behavior and establishes its asymptotic equivalence to a simple average of independent samples.

**Quantitative Results & Theoretical Properties**
The paper derives finite-sample mean squared error (MSE) bounds for the policy gradient (Theorem 2, Proposition 3) and establishes error bounds for the suboptimality gap (Lemma 6). It proves parameter consistency and derives the asymptotic distribution of the suboptimality gap without requiring parameter identifiability (Theorem 8). The analysis yields two major properties: the oracle property, showing GRPO is asymptotically equivalent to an algorithm with access to a true critic network (Corollaries 4 & 9), and optimality, demonstrating that GRPO asymptotically minimizes both MSE and the suboptimality gap among a broad class of policy gradient algorithms (Corollaries 5 & 10). Additionally, the work derives a universal scaling law for the optimal group size (Theorem 7), which depends exclusively on the training data and model architecture, remaining invariant to training budgets or iteration counts. Empirical validation confirms this universality and the oracle property. Practically, GRPO’s efficiency is highlighted by DeepSeek-R1’s post-training requiring only approximately 147,000 H800 GPU-hours, an order of magnitude less than contemporary reasoning models.

**Stated Limitations**
The analysis explicitly addresses theoretical constraints present in prior literature. Earlier studies on GRPO only examined gradient biases, bounded the expected squared norm of the gradient, or characterized objective transformations, lacking a unified finite and asymptotic characterization. Furthermore, classical asymptotic theory typically assumes parameter identifiability, an assumption explicitly violated by overparameterized LLMs. The authors’ framework circumvents this limitation by deriving asymptotic distributions without identifiability requirements. While the theoretical analysis provides rigorous statistical guarantees, it is grounded in learning theory and assumes the policy gradient estimator can be accurately mapped to U-statistic properties; practical deployment may still require empirical validation of hyperparameters outside the scaling law’s scope. The source material concludes the theoretical exposition at this point, leaving further algorithmic variants and extended empirical benchmarks for subsequent investigation.
