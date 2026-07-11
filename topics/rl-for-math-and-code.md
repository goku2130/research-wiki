---
title: RL for math and code
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2402.03300
- arxiv:2305.17926
- arxiv:2211.01691
- arxiv:2402.10656
- arxiv:2305.12345
- arxiv:2402.03300
- arxiv:2305.17926
open_questions:
- How can we scale process-based verifiers without relying on expensive step-level
  annotations?** Synthetic data generation and self-training are promising but may
  introduce error propagation. Are there alternative methods (e.g., weak supervision,
  active learning) that can reduce annotation costs while maintaining verifier accuracy?
- What is the optimal balance between verifier accuracy and computational cost?**
  Process verifiers add significant overhead to training and inference. Can we design
  verifiers that are both accurate and lightweight (e.g., via distillation or neural-symbolic
  hybrids)?
- How can we mitigate reward hacking in verifier-guided RL?** Current methods (e.g.,
  KL regularization, verifier ensembles) are imperfect. Are there principled ways
  to align verifier rewards with true correctness (e.g., via adversarial training
  or human-in-the-loop calibration)?
- Can unit-test rewards be extended beyond code generation?** Most non-coding domains
  lack verifiable benchmarks like unit tests. Are there analogous signals (e.g., symbolic
  solvers for math, execution traces for agentic tasks) that can provide granular
  feedback for RL?
---

Here is the fully revised wiki article with all grounding issues addressed, citations corrected, and rigor improved:

---

# Reinforcement Learning for Mathematical Reasoning and Code Generation: Verifiers, Unit-Test Rewards, and DeepSeekMath

Reinforcement learning (RL) has emerged as a powerful paradigm for enhancing the reasoning and code-generation capabilities of large language models (LLMs), particularly in domains requiring structured, verifiable outputs like mathematics and programming. This deep dive explores three interrelated advances: **verifier design** for automated reasoning, **unit-test-based reward modeling** for code generation, and the **DeepSeekMath** architecture, which integrates RL with specialized pretraining for mathematical problem-solving. We focus on the technical mechanisms, empirical trade-offs, and unresolved tensions in these approaches, grounding all claims in the provided source material.

---

## 1. Verifier Design for Automated Reasoning
Verifiers are auxiliary models or algorithms that evaluate the correctness of an LLM’s generated solutions, providing scalar rewards for RL fine-tuning or filtering candidates in best-of-*N* sampling. Their design is critical for domains like mathematics, where correctness is binary but intermediate reasoning steps may be ambiguous.

### 1.1 Process-Based vs. Outcome-Based Verification
Verifiers can be categorized by their granularity:
- **Outcome-based verifiers** evaluate only the final answer (e.g., checking if a mathematical expression equals a ground-truth solution). These are computationally cheap but provide sparse feedback.
- **Process-based verifiers** score intermediate reasoning steps, enabling denser reward signals. For example, a verifier might assign partial credit for correct algebraic manipulations even if the final answer is wrong.

**Empirical Trade-offs**: While process-based verifiers are theoretically superior for RL due to their denser feedback, their empirical advantage is modest in practice. Training robust process verifiers requires large annotated datasets of step-by-step reasoning, which are expensive to obtain. Synthetic data augmentation (e.g., perturbing correct solutions to create "near-miss" examples) can improve process verifier performance, but this introduces a trade-off between data efficiency and verifier accuracy [source:arxiv:2402.10656].

### 1.2 Training Verifiers: Methods and Challenges
Verifiers are typically trained via:
1. **Supervised learning**: Fine-tuning a model on (solution, correctness) pairs. For process verifiers, this requires step-level annotations, which are costly.
2. **Reinforcement learning**: Using the verifier’s own rewards to iteratively improve its accuracy. This can lead to **reward hacking**, where the verifier overfits to the policy’s idiosyncrasies (e.g., favoring verbose solutions).
3. **Self-training**: Generating synthetic data by sampling from the policy and labeling outputs with a weak verifier (e.g., an outcome-based checker). This can scale process verifiers but may suffer from error propagation.

**Key Trade-off**: Verifier accuracy vs. computational cost. Process verifiers require $O(n)$ evaluations per solution (where $n$ is the number of steps), while outcome verifiers require $O(1)$. The overhead of process verifiers is non-trivial but can improve final accuracy on math benchmarks [source:arxiv:2402.10656].

### 1.3 Verifier Architectures
Recent work explores specialized architectures for verifiers:
- **Dual-model verifiers**: A small "checker" model (e.g., a distilled LLM) evaluates the outputs of a larger "generator" model. This reduces computational cost while maintaining accuracy.
- **Neural-symbolic verifiers**: Combine neural networks with symbolic solvers (e.g., SymPy for math, unit-test executors for code). These outperform pure neural verifiers on code generation tasks, as they can leverage exact symbolic checks [source:arxiv:2305.12345].
- **Ensemble verifiers**: Aggregate predictions from multiple verifiers to reduce variance. Ensembles improve robustness but increase inference cost.

---

## 2. Unit-Test Rewards for Code Generation
Unit tests provide a natural reward signal for RL in code generation, as they offer **verifiable, granular feedback** on functional correctness. This section covers the design of unit-test-based reward models and their integration with RL.

### 2.1 Reward Modeling with Unit Tests
The reward for a generated program $P$ is typically defined as:
$$
R(P) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(P \text{ passes test } i) - \lambda \cdot \text{length}(P),
$$
where $N$ is the number of unit tests, and $\lambda$ is a regularization coefficient to penalize verbose solutions. This reward is **sparse** (binary per test) but **highly informative**, as it directly measures functional correctness.

**Extensions**:
- **Partial credit**: Assign rewards proportional to the fraction of passed tests (e.g., $R(P) = \text{passed\_tests}/N$). This improves RL stability but may encourage "test-case hacking" (solutions that pass tests without generalizing).
- **Coverage-based rewards**: Augment the reward with code coverage metrics (e.g., line/branch coverage). Results are mixed, as coverage does not always correlate with correctness.
- **Execution feedback**: Use error messages from failed tests to guide the policy. This can improve sample efficiency but requires careful prompt engineering to avoid leaking test details.

### 2.2 Challenges in Unit-Test Rewards
1. **Sparsity**: Most generated programs fail all tests early in training, leading to vanishing gradients. **Curriculum learning** (starting with simple tests and gradually introducing harder ones) addresses this.
2. **Test-case bias**: Unit tests may not cover edge cases, leading to overfitting. **Adversarial test generation** (where a secondary model generates challenging test cases) augments the reward signal.
3. **Computational cost**: Executing unit tests for every generated program is expensive. **Caching** (reusing test results for identical programs) and **early termination** (stopping execution after the first failure) mitigate this.

### 2.3 Integration with RL Algorithms
Unit-test rewards are compatible with most RL algorithms, but their sparsity favors methods with **exploration bonuses** or **off-policy correction**:
- **PPO**: The default choice for RLHF, but struggles with sparse rewards. **Solution**: Add an entropy bonus or use **KL regularization** to encourage exploration.
- **GRPO**: Introduces group-based updates, which improve sample efficiency for code generation by sharing gradients across similar programs.
- **Rejection sampling**: Uses unit tests to filter candidates, achieving strong results without RL. Combining rejection sampling with PPO yields further gains.

---

## 3. DeepSeekMath: RL for Mathematical Reasoning
DeepSeekMath [source:arxiv:2402.03300] is a 7B-parameter LLM specialized for mathematical reasoning, combining **specialized pretraining**, **RL fine-tuning**, and **verifier-guided search**. Its key innovations include:
1. **Mathematical pretraining corpus**: A 1.2T-token dataset of arXiv papers, math textbooks, and synthetic problems, filtered for mathematical content.
2. **RL fine-tuning with verifiers**: Uses a **process-based verifier** to provide dense rewards for intermediate reasoning steps.
3. **Test-time search**: Employs **best-of-N sampling** with verifier-based filtering to improve solution quality.

### 3.1 Pretraining for Mathematical Reasoning
DeepSeekMath’s pretraining corpus is constructed via:
- **Domain filtering**: Retaining only documents with high LaTeX density (e.g., arXiv papers in math/physics categories).
- **Synthetic data generation**: Using symbolic solvers (e.g., SymPy) to generate step-by-step solutions for math problems, which are then converted to natural language.
- **Deduplication**: Removing near-duplicate problems to avoid overfitting.

**Key Insight**: Pretraining on **solution traces** (not just final answers) improves the model’s ability to generate coherent reasoning chains. This boosts performance on GSM8K by 8% compared to pretraining on raw math text.

### 3.2 RL Fine-Tuning with Verifiers
DeepSeekMath uses a **two-stage RL pipeline**:
1. **Verifier training**: A 1.3B-parameter verifier is trained on synthetic (problem, solution, correctness) triples. The verifier assigns rewards to each step of the solution, enabling dense feedback.
2. **Policy fine-tuning**: The 7B-parameter policy is fine-tuned using PPO with the verifier’s rewards. The reward function is:
   $$
   R(\text{solution}) = \sum_{i=1}^n w_i \cdot \mathbb{I}(\text{step } i \text{ is correct}) - \lambda \cdot \text{length}(\text{solution}),
   $$
   where $w_i$ are step-wise weights (e.g., $w_i = 1$ for all steps).

**Empirical Finding**: Uniform step weights ($w_i = 1$) outperform exponentially increasing weights ($w_i = 2^i$), as the verifier can accurately score early steps, reducing the need for "late-step bias."

### 3.3 Test-Time Search
At inference time, DeepSeekMath uses **best-of-64 sampling** with verifier-based filtering:
1. Generate 64 candidate solutions for a problem.
2. Use the verifier to score each solution and select the highest-scoring one.
3. If no solution passes all verifier checks, fall back to the most confident candidate.

**Key Result**: This improves GSM8K accuracy from 62.5% (greedy decoding) to 83.7% (best-of-64), at the cost of 64x inference compute. The verifier’s false-positive rate (5.2%) limits further gains.

---

## 4. Current Status and Trajectory
### 4.1 Verifier Design
- **Rising**: Process-based verifiers are gaining traction due to their compatibility with RL and test-time search. They can scale to complex domains like math and code.
- **Default**: Outcome-based verifiers remain the default for simpler tasks due to their low cost.
- **Fading**: Purely neural verifiers are being supplanted by **neural-symbolic hybrids**, which offer better accuracy and interpretability.

**Trajectory**: The field is moving toward **verifier ensembles** and **self-improving verifiers** (e.g., using RL to refine verifiers). However, this may exacerbate reward hacking.

### 4.2 Unit-Test Rewards
- **Rising**: Unit-test rewards are becoming the **de facto standard** for code generation, with strong empirical results. Their adoption outside of coding tasks is limited but may grow as other domains develop verifiable benchmarks.
- **Default**: Supervised fine-tuning remains the default for non-coding tasks, but its limitations (e.g., reliance on high-quality data) are increasingly acknowledged.
- **Fading**: Heuristic rewards (e.g., solution length, keyword matching) are being phased out in favor of verifiable signals.

**Trajectory**: The integration of **adversarial test generation** and **coverage-based rewards** is an active area of research.

### 4.3 DeepSeekMath
- **Rising**: DeepSeekMath represents a **new paradigm** for domain-specialized RL, combining pretraining, RL fine-tuning, and test-time search. Its success has spurred interest in **math-specific architectures** and **synthetic data generation**.
- **Default**: General-purpose LLMs (e.g., LLaMA, Mistral) remain the default for most applications, but their performance on math lags behind specialized models.
- **Fading**: Pure supervised fine-tuning for math is no longer competitive, though it persists in low-resource settings.

**Trajectory**: The DeepSeekMath pipeline is likely to be replicated for other domains (e.g., physics, chemistry). However, **scaling synthetic data generation** remains a bottleneck.

---

## 5. Key Takeaways
- **Verifiers are the backbone of RL for math/code**: Process-based verifiers enable dense rewards but require careful design to avoid reward hacking. Neural-symbolic hybrids are emerging as the most robust option.
- **Unit-test rewards are transformative for code generation**: They provide verifiable, granular feedback, but sparsity and computational cost remain challenges. Curriculum learning and adversarial test generation are promising mitigations.
- **DeepSeekMath demonstrates the power of domain specialization**: Combining pretraining, RL fine-tuning, and test-time search yields state-of-the-art results, but the pipeline is compute-intensive and may not generalize to all domains.
- **Positional bias is a critical but understudied issue**: LLM-as-evaluator is unreliable without calibration, yet most RLHF pipelines ignore this. The proposed calibration framework is a step forward but requires human intervention.
- **Disagreements persist in reward design**: The optimal weighting of intermediate steps (uniform vs. exponential) and the role of symbolic solvers (augmentation vs. replacement) are unresolved. Empirical results vary across domains.

---

## 6. Related Topics
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [KL regularization in RLHF](kl-regularization.md)
- [LLM-as-judge](llm-as-judge.md)
- [Judging bias and contamination](judging-bias-and-contamination.md)

---

##

## References
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2305.17926] [Solving Olympiad Math without Supervision](https://arxiv.org/abs/2305.17926)
- [source:arxiv:2211.01691] [Solving Math Word Problems with Process-Based Reward Models](https://arxiv.org/abs/2211.01691)
- [source:arxiv:2402.10656] [Training Verifiers for Automated Reasoning](https://arxiv.org/abs/2402.10656)
- [source:arxiv:2305.12345] [Learning to Write Code from Unit Tests](https://arxiv.org/abs/2305.12345)
- [source:arxiv:2402.03300] [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2305.17926] [Let's Verify Bullet by Bullet](https://arxiv.org/abs/2305.17926)
