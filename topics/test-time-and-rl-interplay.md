---
title: Test-time compute and RL interplay
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2410.18905
- arxiv:2402.05402
- arxiv:2305.18290
- arxiv:2306.05673
- arxiv:2310.10303
- arxiv:2404.01080
- arxiv:2401.10020
open_questions:
- How do inference scaling laws vary across different model sizes and architectures?
  Are there predictable relationships between model capacity and the efficiency of
  test-time compute?
- Can self-rewarding models escape the "alignment tax" observed on standard NLP benchmarks,
  or is this a fundamental trade-off?
- What are the limits of verifiable rewards for open-ended tasks (e.g., creative writing,
  dialogue)? Can hybrid reward models (e.g., combining verifiable and LLM-as-a-Judge
  rewards) bridge this gap?
- How can MCTS-guided decoding be optimized for real-time applications (e.g., chatbots)
  without sacrificing performance? Are there principled ways to trade off latency
  and quality?
---

# Test-Time Compute and RL Interplay: Search, Inference Scaling, and MCTS-Guided Decoding

Large language models (LLMs) have demonstrated remarkable capabilities when scaled during training, but recent work reveals that *test-time compute*—additional computation applied during inference—can unlock further performance gains, particularly for complex reasoning tasks. The interplay between test-time compute and reinforcement learning (RL) is emerging as a critical axis of optimization, where search algorithms (e.g., Monte Carlo Tree Search, MCTS) and RL-based decoding strategies (e.g., guided by learned policies or reward models) jointly determine the efficiency and effectiveness of inference scaling. This deep dive explores the theoretical foundations, empirical trade-offs, and open challenges in this nascent but rapidly evolving area.

---

## Core Concepts and Mechanisms

### 1. Test-Time Compute and Inference Scaling
Test-time compute refers to the additional computational budget expended during inference to improve model outputs, distinct from training-time scaling. This includes:
- **Iterative refinement**: Generating multiple candidate outputs and selecting or refining the best one (e.g., via rejection sampling or Best-of-N).
- **Search-based decoding**: Using algorithms like MCTS to explore the space of possible outputs, guided by learned policies or reward models.
- **Dynamic computation**: Allocating more compute to "hard" inputs (e.g., via early-exit mechanisms or adaptive depth).

**Inference scaling laws** formalize the relationship between test-time compute and performance. Empirical studies suggest that for tasks like mathematical reasoning or code generation, performance scales predictably with the number of candidate generations or search iterations [source:arxiv:2402.05402]. For example, [source:arxiv:2402.05402] demonstrates that GSM8K accuracy improves by up to 13 percentage points when using Best-of-N sampling with N=16 compared to greedy decoding.

### 2. RL for Test-Time Compute Optimization
RL provides a framework for optimizing how test-time compute is allocated. Key components include:
- **Policy-guided search**: The LLM acts as a policy $\pi_\theta(y|x)$, where $x$ is the input prompt and $y$ is the output. RL fine-tunes $\pi_\theta$ to maximize a reward $r(x, y)$, which may encode task-specific metrics (e.g., correctness, coherence) or learned preferences (e.g., from human feedback).
- **Reward models**: A separate model $r_\phi(x, y)$ is trained to predict the quality of $(x, y)$ pairs, often via preference learning (e.g., Bradley-Terry models) or verifiable rewards (e.g., unit tests for code).
- **Exploration vs. exploitation**: During inference, the policy must balance exploring new outputs (e.g., via temperature sampling) and exploiting high-reward candidates (e.g., via greedy decoding or Best-of-N). Search algorithms like MCTS provide a principled way to manage this trade-off by maintaining a tree of partial outputs and prioritizing high-value branches.

### 3. MCTS-Guided Decoding
MCTS is a search algorithm that combines tree traversal with Monte Carlo rollouts to estimate the value of partial outputs. In the context of LLMs:
1. **Tree structure**: Each node represents a partial output (e.g., a sequence of tokens), and edges represent token extensions. The root node is the input prompt $x$.
2. **Selection**: Starting from the root, the algorithm traverses the tree using the Upper Confidence Bound for Trees (UCT) criterion:

$$
\text{UCT}(s) = \frac{Q(s)}{N(s)} + c \cdot \sqrt{\frac{\log N(\text{parent}(s))}{N(s)}},
$$

   where $Q(s)$ is the estimated value of node $s$, $N(s)$ is its visit count, and $c$ is an exploration constant.
3. **Expansion**: When a leaf node is reached, the policy $\pi_\theta$ generates $k$ candidate tokens to extend the sequence.
4. **Simulation**: For each candidate, a rollout is performed (e.g., greedy decoding to completion) to estimate its value using the reward model $r_\phi$.
5. **Backpropagation**: The estimated value is propagated back up the tree to update $Q(s)$ and $N(s)$ for all ancestor nodes.

**Key advantages**:
- **Adaptive depth**: MCTS dynamically allocates compute to promising branches, avoiding exhaustive search.
- **Reward integration**: The reward model $r_\phi$ guides search toward high-quality outputs without requiring ground-truth labels during inference.
- **Uncertainty awareness**: The UCT criterion balances exploration (visiting under-explored nodes) and exploitation (favoring high-reward nodes).

**Limitations**:
- **Compute overhead**: MCTS requires multiple forward passes per token, increasing latency.
- **Reward sparsity**: If $r_\phi$ is poorly calibrated or sparse (e.g., only provides feedback at the end of generation), MCTS may struggle to guide search effectively.
- **Policy-reward misalignment**: If $\pi_\theta$ and $r_\phi$ are trained on different distributions, the search may converge to suboptimal outputs.

---

## RL and Test-Time Compute: Interplay and Trade-offs

### 1. RL as a Test-Time Compute Optimizer
RL can be used to *learn* how to allocate test-time compute efficiently. For example:
- **Dynamic rollout policies**: RL can learn to adapt the number of search rollouts based on input difficulty or intermediate rewards.
- **Early stopping**: RL can learn to terminate generation early if the reward model predicts low-quality outputs, saving compute.
- **Adaptive branching**: The policy $\pi_\theta$ can learn to generate fewer candidate tokens for "easy" inputs and more for "hard" ones.

**Empirical trade-offs**:
- **Compute vs. performance**: For tasks like GSM8K (math word problems), increasing the number of search rollouts can improve accuracy, but the optimal compute budget depends on the task and the quality of the reward model [source:arxiv:2402.05402].
- **Latency vs. throughput**: MCTS-guided decoding is inherently sequential, limiting parallelization. Techniques like batched rollouts or speculative decoding can mitigate this but introduce additional complexity.

### 2. RL for Reward Model Training
The quality of the reward model $r_\phi$ is critical for test-time compute efficiency. RL can be used to:
- **Improve reward modeling**: Self-rewarding language models iteratively refine $r_\phi$ by using the LLM itself to generate and evaluate candidate outputs, reducing reliance on human annotations [source:arxiv:2401.10020]. For example, [source:arxiv:2401.10020] demonstrates that self-rewarding models achieve AlpacaEval win rates of up to 20.44% against GPT-4 Turbo.
- **Mitigate reward hacking**: RL can learn to penalize reward model exploitation (e.g., generating verbose outputs to maximize length-based rewards) via auxiliary losses or adversarial training.

**Disagreement in the literature**:
- **Reward model generalization**: Some work suggests that reward models trained on narrow distributions (e.g., math problems) may not generalize well to out-of-distribution inputs, limiting the effectiveness of search. However, self-rewarding models can iteratively expand the reward model's coverage [source:arxiv:2401.10020].
- **Reward sparsity**: Verifiable rewards (e.g., unit tests for code) enable more effective search than sparse human preferences, but their applicability is limited to tasks with objective success criteria [source:arxiv:2404.01080].

### 3. RL and Search Synergy
The interplay between RL and search algorithms like MCTS can be synergistic:
- **Policy initialization**: RL fine-tuning (e.g., via PPO or DPO) can initialize $\pi_\theta$ to generate higher-quality partial outputs, reducing the search space for MCTS.
- **Search-augmented training**: MCTS can be used during training to generate high-quality rollouts for RL, improving sample efficiency.
- **Joint optimization**: The policy $\pi_\theta$ and reward model $r_\phi$ can be co-trained to improve alignment, e.g., via iterative self-rewarding [source:arxiv:2401.10020].

**Key challenges**:
- **Credit assignment**: RL struggles to attribute rewards to early tokens in long sequences, as the reward is typically only observed at the end of generation. MCTS mitigates this by propagating rewards back through the tree.
- **Exploration in high-dimensional spaces**: The space of possible outputs grows exponentially with sequence length, making exhaustive search infeasible. RL must learn to explore efficiently, e.g., via intrinsic motivation or curiosity-driven rewards.

---

## Empirical Results and Benchmarks

### 1. Performance Gains from Test-Time Compute
Empirical studies demonstrate that test-time compute can significantly improve performance on reasoning-heavy tasks. For example:
- **GSM8K (Math)**: Best-of-N sampling (e.g., generating 16 candidates and selecting the highest-reward one) can improve accuracy by up to 13 percentage points over greedy decoding [source:arxiv:2402.05402].
- **AlpacaEval (Instruction Following)**: Self-rewarding models achieve win rates of up to 20.44% against GPT-4 Turbo, compared to 9.94% for the baseline [source:arxiv:2401.10020].

**Observations**:
- Search-based methods (e.g., MCTS) consistently outperform brute-force methods (e.g., Best-of-N), suggesting that guided search is more efficient.
- The gains are largest for tasks with verifiable rewards (e.g., math, code), where the reward model can provide dense feedback.

### 2. Compute Efficiency
The performance-compute trade-off varies by task and method:
- **Best-of-N**: Performance scales sublinearly with the number of candidates, with diminishing returns setting in quickly (e.g., beyond 16 candidates for GSM8K) [source:arxiv:2402.05402].
- **MCTS**: More compute-efficient than Best-of-N, as it avoids generating low-quality full outputs. However, the marginal gains decrease with additional rollouts.
- **RL fine-tuning**: Shifts the performance-compute curve upward, enabling higher performance at the same compute budget. For example, larger models may benefit more from test-time compute due to their capacity to leverage additional search iterations [source:arxiv:2402.05402].

### 3. Reward Model Quality
The effectiveness of search depends heavily on the reward model $r_\phi$. Key findings include:
- **Verifiable rewards** (e.g., unit tests for code) provide dense, unambiguous feedback and are highly effective for MCTS [source:arxiv:2404.01080].
- **LLM-as-a-Judge rewards** (e.g., [source:arxiv:2401.10020]) are more general but prone to bias (e.g., favoring verbose outputs) and may require iterative refinement.
- **Human preferences** are sparse and expensive to collect but provide high-quality feedback for tasks like instruction following.

---

## Current Status and Trajectory

### 1. Rising Techniques
- **MCTS-guided decoding**: Increasingly adopted for reasoning-heavy tasks (e.g., math, code), where the compute overhead is justified by performance gains.
- **Self-rewarding models**: Iterative self-improvement (e.g., [source:arxiv:2401.10020]) is gaining traction as a way to reduce reliance on human annotations and improve reward model generalization.
- **Dynamic compute allocation**: Techniques like early-exit decoding or adaptive rollouts are being explored to reduce latency while preserving performance.

### 2. Default Techniques
- **Best-of-N sampling**: Remains the default for many applications due to its simplicity and parallelizability. Widely used in production systems for tasks like creative writing or chatbot responses.
- **Greedy decoding**: Dominates for latency-sensitive applications (e.g., real-time chat), where test-time compute is prohibitive.

### 3. Fading Techniques
- **Static reward models**: Frozen reward models (e.g., trained once on human preferences) are becoming less common, as they limit the scalability of test-time compute. Iterative or self-rewarding approaches are replacing them [source:arxiv:2401.10020].
- **Brute-force search**: Exhaustive search is rarely used due to its exponential compute cost. Guided search methods like MCTS are preferred.

### 4. Trajectory and Open Challenges
- **Scaling laws**: The field is still characterizing how test-time compute scales with model size, task complexity, and reward model quality. Early results suggest that the optimal compute budget depends on the task and reward model, with larger models potentially benefiting more from additional test-time compute [source:arxiv:2402.05402].
- **Reward model generalization**: Self-rewarding models show promise, but their long-term stability and safety are unproven. There is disagreement over whether iterative reward modeling leads to reward hacking or genuine generalization.
- **Latency optimization**: MCTS and other search methods are computationally expensive. Techniques like speculative decoding or batched rollouts are being explored to reduce latency, but their effectiveness is task-dependent.
- **Task-specific vs. general methods**: Verifiable rewards work well for math and code but are less applicable to open-ended tasks. The field is still developing general-purpose reward models that can guide search across diverse domains.

---

## Key Takeaways
- **Test-time compute scales performance**: For reasoning-heavy tasks, additional compute during inference (e.g., via MCTS or Best-of-N) yields performance gains, though the exact scaling behavior is task-dependent [source:arxiv:2402.05402].
- **RL enables efficient compute allocation**: RL fine-tunes policies and reward models to guide search algorithms like MCTS, improving compute efficiency compared to brute-force methods.
- **MCTS outperforms Best-of-N**: MCTS is more compute-efficient than Best-of-N because it avoids generating low-quality full outputs, but it introduces latency overhead.
- **Reward model quality is critical**: The effectiveness of MCTS depends on the reward model's ability to provide dense, unambiguous feedback. Verifiable rewards work well for math/code, while LLM-as-a-Judge rewards are more general but prone to bias [source:arxiv:2401.10020].
- **Self-rewarding models are rising**: Iterative self-improvement (e.g., self-rewarding LLMs) reduces reliance on human annotations and improves reward model generalization, but long-term stability is unproven [source:arxiv:2401.10020].
- **Compute-performance trade-offs**: The optimal test-time compute budget depends on the task, reward model, and latency constraints. Diminishing returns set in quickly for some tasks [source:arxiv:2402.05402].
- **Disagreements persist**: The field is divided on whether verifiable rewards or LLM-as-a-Judge rewards are superior for MCTS, and whether iterative reward modeling leads to reward hacking or genuine generalization.

---

## Related Topics
- [[RL for reasoning models](rl-for-reasoning.md)]: How RL is applied to improve LLM reasoning capabilities, including test-time compute strategies.
- [[Reward modeling for LLMs](reward-modeling.md)]: Techniques for training and evaluating reward models, including verifiable rewards and LLM-as-a-Judge.
- [[Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)]: Brute-force test-time compute methods and their trade-offs.
- [[RL for math and code](rl-for-math-and-code.md)]: Domain-specific applications of RL and test-time compute for mathematical reasoning and code generation.
- [[Self-improvement and self-play RL](self-improvement-and-self-play.md)]: Iterative self-improvement techniques, including self-rewarding models.
- [[Process vs outcome reward models](process-vs-outcome-rewards.md)]: How reward models can provide feedback on intermediate steps (process rewards) vs. final outputs (outcome rewards).
- [[Reward hacking in RLHF](reward-hacking.md)]: Challenges and mitigation strategies for reward model exploitation, particularly in test-time compute settings.

---

##

## References
- [source:arxiv:2410.18905] [Scaling LLM Test-Time Compute Optimally and Efficiently](https://arxiv.org/abs/2410.18905)
- [source:arxiv:2402.05402] [Inference Scaling Laws: An Empirical Analysis of Compute-optimal Inference](https://arxiv.org/abs/2402.05402)
- [source:arxiv:2305.18290] [Reasoning with Language Model is Planning with World Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2306.05673] [Monte Carlo Tree Search for Text Generation](https://arxiv.org/abs/2306.05673)
- [source:arxiv:2310.10303] [ReST: Reinforced Self-Training for Language Models](https://arxiv.org/abs/2310.10303)
- [source:arxiv:2404.01080] [RLVR: Reinforcement Learning with Verifiable Rewards for Language Model Alignment](https://arxiv.org/abs/2404.01080)
- [source:arxiv:2401.10020] [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
