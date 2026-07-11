---
id: arxiv:2512.11391
type: paper
title: Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization
url: https://arxiv.org/abs/2512.11391
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

# Null-Space Constrained Policy Optimization (NSPO)

### Core Problem: The Safety Alignment Tax
Large Language Models (LLMs) undergoing safety alignment via Reinforcement Learning (RL) often experience a "safety alignment tax," where the optimization of safety objectives leads to the forgetting or degradation of general capabilities (e.g., mathematical reasoning, coding, and instruction following). The authors identify the root cause as a conflict between safety and general capability objectives, where safety gradients weaken the model's learned general abilities.

### Method: Null-Space Constrained Policy Optimization
NSPO mitigates the alignment tax by geometrically projecting safety policy gradients into the null space of general task representations. This ensures that parameter updates are orthogonal to the subspace spanned by the model's core capabilities, thereby preserving them.

**Step-by-Step Recipe:**
1. **Representation Capture:** Feed general reasoning data into the base model $\pi_{\text{base}}$ to capture the embedding matrix $K \in \mathbb{R}^{d \times N}$ (where $d$ is the representation dimension and $N$ is the number of tokens).
2. **Covariance Computation:** To reduce computational complexity, compute the non-central covariance matrix $KK^T \in \mathbb{R}^{d \times d}$.
3. **SVD Decomposition:** Apply Singular Value Decomposition (SVD) to the covariance matrix:

$$
\{U, A, U^T\} = \text{SVD}(KK^T)
$$

4. **Null Space Identification:** Define a submatrix $\hat{U}$ by removing eigenvectors in $U$ corresponding to eigenvalues above a threshold (set to $5e^{-4}$).
5. **Projection Matrix Construction:** Define the projection matrix as $P = \hat{U}\hat{U}^T$.
6. **Gradient Projection:** Compute the safety policy gradient $\nabla_W \mathcal{J}$ (based on Group Relative Policy Optimization, GRPO) and project it into the null space:

$$
\nabla_W \mathcal{J}_{\text{NSPO}} = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \hat{A}_i \frac{1}{|y|} \sum_{t=1}^{|y|} r_{i,t}(\theta) \nabla_W \log \pi_\theta (y_i | x, y_{i<t}) \right] \hat{U}U^T
$$

7. **Weight Update:** Update the model parameters $W$ using $\nabla_W \mathcal{J}_{\text{NSPO}}$. Notably, the KL divergence penalty is removed to avoid conflicts with the safety objective.

### Theoretical Guarantees
*   **Gradient Stability:** The projection is a non-expansive mapping, meaning the spectral norm of the projected gradient is bounded by the original: $\|\nabla_{W} J_{\text{NSPO}}\|_2 \leq \|\nabla_{W} J\|_2$.
*   **Valid Descent Direction:** There exists a learning rate $\eta > 0$ such that $J(W - \eta \nabla_{W} J_{\text{NSPO}}) \leq J(W)$, ensuring that the safety objective is still effectively optimized.

### Key Quantitative Results
NSPO was evaluated using Llama3-8B-Instruct and Qwen2.5-7B-Instruct.

*   **Safety Performance:** NSPO achieved state-of-the-art results across seven safety benchmarks. For Qwen2.5-7B-Instruct, NSPO achieved an Attack Success Rate (ASR) of **0.52%** on PKU-Safe, **0.50%** on HarmBench, **0.67%** on JailbreakBench, **0.11%** on HarmfulQA, and **0.81%** on ALERT.
*   **General Capability Preservation:** Performance degradation on general tasks was marginal, typically within **1%**, with a maximum degradation of **2.67%**. This significantly outperformed other safety alignment methods that caused substantial drops in math and code benchmarks.
*   **Data Efficiency:** NSPO achieved strong safety performance using only **40%** of the public human-annotated safety data from PKU-SafeRLHF.
*   **Computational Efficiency:** The initial SVD is a one-time cost of $O(d^2)$. During training, the projection complexity is $O(d^2)$, and GPU memory overhead is limited to $O(d^2)$ via an offloading mechanism.

### Stated Limitations and Trade-offs
The authors note a trade-off regarding the construction of the projection matrix $K$:
*   **Sample Size:** Increasing the sample size of general data improves utility (preserves capabilities better) but compromises safety, as a larger sample size results in a smaller null space, restricting the model's exploration space.
*   **Thresholding:** The eigenvector selection threshold requires a balance; smaller thresholds preserve general capabilities better but degrade safety, while larger thresholds sacrifice utility for safety.
