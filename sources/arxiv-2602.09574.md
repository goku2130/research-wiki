---
id: arxiv:2602.09574
type: paper
title: Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling
  of LLMs
url: https://arxiv.org/abs/2602.09574
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Budget-Guided MCTS (BG-MCTS)

**Budget-Guided MCTS (BG-MCTS)** is a tree-search decoding algorithm designed for test-time scaling of Large Language Models (LLMs) under fixed per-query token budgets.

### Core Problem
Standard tree-search policies are typically **budget-agnostic**, treating the token budget $B$ merely as a termination condition. This mismatch leads to two primary failure modes:
1. **Late-stage over-branching:** The search introduces new alternatives too late to refine them, exhausting the budget before reaching a completed answer.
2. **Premature termination:** The search stops before the budget is fully utilized.

The goal of BG-MCTS is to align the search policy with the remaining budget, shifting the search behavior from broad exploration (wide) to refinement and completion (deep) as the budget is consumed.

### Method
BG-MCTS modifies the standard Monte Carlo Tree Search (MCTS) loop by conditioning selection and widening decisions on the **budget sufficiency ratio** $\rho$:

$$
\rho = 1 - \frac{C_{\text{used}}}{B}
$$

where $C_{\text{used}}$ is the cumulative number of output tokens generated.

#### 1. Budget-Guided Selection
BG-MCTS replaces the standard PUCT score with a budget-conditioned version (**BG-PUCT**) that anneals exploration and biases selection toward deeper nodes as $\rho$ decreases:

$$
\text{BG-PUCT}(p, s, \rho) = \frac{\tilde{W}(s, \rho)}{m_s} + \rho c P(s | p) \sqrt{\frac{\ln(m_p)}{m_s}}
$$

The corrected accumulated value $\tilde{W}(s, \rho)$ incorporates a **completion bias** to encourage deepening:

$$
\tilde{Q}(x, \rho) = Q(x) + \kappa (1 - \rho) \frac{d(x)}{\tilde{d}_{\text{ans}}}
$$

where $d(x)$ is the node depth, $\tilde{d}_{\text{ans}}$ is the average depth of completed answers, and $\kappa$ is a constant.

#### 2. Budget-Guided Tree Widening
To prevent wasteful late-stage branching, BG-MCTS introduces a **virtual generative child** $s_{\text{gen}}(p)$. The decision to widen the tree (generate a new child) versus deepening an existing one is determined by the generative score:

$$
E_{\text{gen}}(p, \rho) = \mu(p) + \lambda \rho \sigma^2(p)
$$

where $\mu(p)$ and $\sigma^2(p)$ are the mean and variance of child values $Q(s)$. The $\rho$ factor ensures that the incentive to widen is strong early in the search and suppressed as the budget runs out.

#### 3. Unified Selection
At each internal node $p$, the algorithm selects the action $s^*$ that maximizes the score:

$$
\text{Score}(p, s, \rho) = \begin{cases} \text{BG-PUCT}(p, s, \rho), & s \in \mathcal{S}_{\text{std}}(p) \\ E_{\text{gen}}(p, \rho), & s = s_{\text{gen}}(p) \end{cases}
$$

### Key Quantitative Results
BG-MCTS was evaluated using Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Qwen3-32B across budgets $B \in \{10\text{k}, 20\text{k}, 30\text{k}\}$.

*   **Mathematical Reasoning:** On **AIME24/25**, BG-MCTS consistently outperformed budget-agnostic MCTS. For Qwen3-32B at $B=30\text{k}$, BG-MCTS achieved **0.350** accuracy compared to **0.300** for MCTS and **0.267** for Greedy decoding.
*   **Physics Reasoning:** On **UG-Physics**, BG-MCTS achieved the best or near-best accuracy. For Qwen3-32B at $B=30\text{k}$, BG-MCTS reached **0.372** accuracy, surpassing MCTS (**0.356**).
*   **Efficiency:** Analysis on AIME24/25 (Llama-3.1-8B) showed that BG-MCTS produced significantly more correct answered nodes than baselines. At $B=30\text{k}$, BG-MCTS produced **413.7** correct nodes, while MCTS produced **143.7** and Repeated Sampling produced **48.0**.

### Limitations
*   **Reward Model Saturation:** The authors noted that the process reward model (GenPRM-7B) often produces near-binary, saturated scores, which limits the granularity of the search signal.
*   **Coverage Trade-off:** BG-MCTS may reach final answers on fewer total problem instances (lower tree-level reach rate) than some baselines, as it trades broad coverage for higher-precision completed candidates.
