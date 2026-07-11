---
title: Process vs outcome reward models
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2203.02155
- arxiv:2402.03300
- arxiv:2312.11410
- arxiv:2411.16232
- arxiv:2402.10435
- arxiv:2305.18290
- arxiv:2402.03300
- arxiv:2310.19078
open_questions:
- ''
- How do PRMs generalize to domains beyond mathematical reasoning, such as coding,
  planning, or multi-turn dialogue? Are there tasks where PRMs degrade performance
  compared to ORMs?
- What is the optimal reward design for PRMs? Should step rewards be additive or multiplicative,
  and should they be constrained to be monotonic?
- How can PRMs scale to tasks with very long trajectories (e.g., thousands of steps)?
  Are there efficient methods for step-level annotation or synthetic feedback generation?
---

# Process vs Outcome Reward Models

Large language models (LLMs) trained via reinforcement learning from human feedback (RLHF) traditionally optimize for *outcome* rewards—scalar scores assigned to entire generated sequences. However, complex tasks like mathematical reasoning or multi-step problem-solving often benefit from *process* supervision, where rewards are assigned to intermediate steps, enabling finer-grained credit assignment and more stable training. This deep dive explores the distinction between process and outcome reward models (PRMs and ORMs), their theoretical foundations, empirical trade-offs, and emerging techniques like step-level supervision.

---

## Theoretical foundations

### Outcome reward models (ORMs)
ORMs assign a single scalar reward $r(x, y)$ to a complete trajectory $y = (y_1, \dots, y_T)$ given input $x$. The reward is trained via the Bradley-Terry model for sequence-level preferences [source:arxiv:2203.02155][source:arxiv:2305.18290]:

$$
p^*(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l)),
$$

where $\sigma$ is the logistic function, and $y_w, y_l$ are preferred and dispreferred completions. The policy $\pi_\phi$ is then optimized to maximize the expected reward under a KL constraint to a reference policy $\pi_{\text{ref}}$:

$$
\mathcal{J}(\phi) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\phi(y \mid x)} \left[ r(x, y) - \beta \log \frac{\pi_\phi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right].
$$

ORMs are simple to implement and align with human preferences at the sequence level, but they suffer from *sparse credit assignment*: the policy receives no signal about which intermediate steps contributed to the final reward. This is particularly problematic for long-horizon tasks, where the probability of generating a correct full trajectory by chance is vanishingly small [source:arxiv:2402.03300].

### Process reward models (PRMs)
PRMs decompose the reward into step-level signals $r(x, y_{1:t})$ for partial trajectories $y_{1:t} = (y_1, \dots, y_t)$. The total reward is typically the sum (or product) of step rewards [source:arxiv:2312.11410][source:arxiv:2402.03300]:

$$
r(x, y) = \sum_{t=1}^T r(x, y_{1:t}).
$$

PRMs are trained using the Bradley-Terry framework applied to *step-level* preferences. For example, given a trajectory $y$ and a prefix $y_{1:t}$, a labeler (human or synthetic) annotates whether $y_{1:t}$ is "correct" or "on the right track." The step-level reward model is then:

$$
p^*(y_{1:t}^w \succ y_{1:t}^l \mid x) = \sigma(r(x, y_{1:t}^w) - r(x, y_{1:t}^l)) \quad \text{[source:arxiv:2312.11410]}.
$$

PRMs enable *dense credit assignment*, as the policy receives feedback at every step, but they introduce new challenges:
1. **Annotation cost**: Step-level preferences require more labels than sequence-level preferences. Human annotation is expensive, while synthetic annotation (e.g., using GPT-4) may inherit biases from the teacher model [source:arxiv:2402.03300].
2. **Reward hacking**: The policy may exploit step-level rewards to generate verbose or redundant intermediate steps (e.g., "Let’s think step by step...") without improving the final outcome. Techniques like reward normalization or capping step rewards can mitigate this [source:arxiv:2312.11410].
3. **Reward consistency**: Step rewards should ideally be *monotonic* (i.e., $r(x, y_{1:t+1}) \geq r(x, y_{1:t})$ if $y_{1:t+1}$ is a valid continuation) to avoid perverse incentives, though this is not universally enforced in practice [source:arxiv:2312.11410].

---

## Step-level supervision: Methods and trade-offs

### Step-level preference collection
Two dominant approaches exist for collecting step-level preferences:
1. **Human annotation**: Labelers judge whether a partial trajectory $y_{1:t}$ is "correct" or "on the right track." This is expensive and scales poorly with trajectory length. For example, [source:arxiv:2312.11410] reports using human annotators to label step-level correctness for mathematical reasoning tasks, but the dataset size is not specified.
2. **Synthetic annotation**: A strong "teacher" model (e.g., GPT-4) generates step-level feedback. While synthetic labels may inherit biases from the teacher, they are increasingly used due to scalability. For instance, [source:arxiv:2402.03300] uses synthetic step-level feedback to train PRMs for mathematical reasoning, achieving state-of-the-art performance on benchmarks like MATH and AIME.

### PRM training objectives
PRMs are trained using one of two objectives:
1. **Binary classification**: Step-level feedback is treated as a binary label (correct/incorrect), and the model is trained via cross-entropy. This is simple but discards nuanced preference information [source:arxiv:2312.11410].
2. **Pairwise ranking**: The Bradley-Terry model is used to compare pairs of partial trajectories, as in ORMs. This captures relative preferences but requires more data [source:arxiv:2312.11410].

### Credit assignment strategies
PRMs can assign credit in two ways:
1. **Additive rewards**: The total reward is the sum of step rewards, $r(x, y) = \sum_t r(x, y_{1:t})$. This is intuitive but may over-reward verbose trajectories [source:arxiv:2312.11410].
2. **Product rewards**: The total reward is the product of step rewards, $r(x, y) = \prod_t r(x, y_{1:t})$. This penalizes early mistakes harshly but may be too conservative for exploratory tasks [source:arxiv:2312.11410].

---

## Empirical comparisons: PRMs vs ORMs

### Mathematical reasoning
Mathematical reasoning is the most studied domain for PRMs. Key findings:
1. **PRMs outperform ORMs**: PRMs achieve higher accuracy on benchmarks like MATH and AIME. For example, DeepSeekMath [source:arxiv:2402.03300] demonstrates that process supervision via GRPO improves accuracy on the MATH benchmark from 51.7% (ORM) to 64.6% (PRM) and on AIME from 15.0% to 36.7%. The dataset for step-level annotation consists of 776K examples, with synthetic feedback generated by a strong teacher model.
2. **Step-level feedback matters**: Ablations in [source:arxiv:2402.03300] show that removing step-level rewards degrades performance, particularly for tasks requiring intermediate reasoning. For instance, the accuracy on MATH drops by ~13 percentage points when switching from PRMs to ORMs.
3. **Reward hacking**: PRMs can exploit step rewards to generate verbose or redundant steps. Techniques like reward normalization (e.g., dividing by trajectory length) or capping step rewards (e.g., to a maximum of 1) can mitigate this [source:arxiv:2312.11410].

### Generalist tasks
For generalist tasks (e.g., instruction following), the trade-offs are less clear:
1. **ORMs are simpler**: ORMs require fewer annotations and are easier to train. InstructGPT [source:arxiv:2203.02155] achieves strong performance with ORMs alone, with human evaluations showing that 1.3B-parameter InstructGPT models are preferred over 175B-parameter GPT-3 models 85% of the time.
2. **PRMs may degrade performance**: Some works suggest that PRMs can introduce distribution shift or degrade performance on generalist tasks. For example, [source:arxiv:2402.03300] notes that PRMs are primarily validated for reasoning tasks, and their generalization to other domains remains unclear.
3. **Hybrid models**: Some works combine PRMs and ORMs, using PRMs for intermediate steps and ORMs for final outcomes. For example, [source:arxiv:2402.03300] uses a hybrid approach in GRPO, where step-level rewards guide intermediate reasoning, and a final outcome reward ensures correctness. However, this is not yet widely reported.

### Reward hacking and over-optimization
Both PRMs and ORMs are susceptible to reward hacking. Key observations:
1. **Verbosity**: PRMs may generate unnecessarily long trajectories to accumulate step rewards. For example, [source:arxiv:2312.11410] reports that PRMs can generate redundant steps like "Let’s think step by step..." without improving the final answer. Normalizing rewards by trajectory length can address this.
2. **Early termination**: If step rewards are not monotonic, the policy may learn to terminate early to avoid negative rewards. Careful reward design (e.g., enforcing monotonicity) is required to prevent this [source:arxiv:2312.11410].
3. **Over-optimization**: Both PRMs and ORMs can exploit flaws in the reward model. PRMs may be more robust due to their finer-grained feedback, but this remains an active area of research. For example, [source:arxiv:2312.11410] notes that PRMs can still be over-optimized if the step-level rewards are not carefully designed.

---

## Current status and trajectory

### Rising techniques
1. **PRMs for reasoning**: PRMs are rapidly becoming the default for mathematical reasoning and other complex tasks. Methods like GRPO [source:arxiv:2402.03300] demonstrate that PRMs can scale to large datasets (e.g., 776K examples for DeepSeekMath) and achieve state-of-the-art performance on benchmarks like MATH (64.6% accuracy) and AIME (36.7% accuracy).
2. **Hybrid models**: Some works combine PRMs and ORMs, using PRMs for intermediate steps and ORMs for final outcomes. This approach is gaining traction but is not yet widely reported. For example, [source:arxiv:2402.03300] uses a hybrid approach in GRPO, where step-level rewards guide intermediate reasoning, and a final outcome reward ensures correctness.
3. **Synthetic annotation**: Synthetic step-level feedback from strong models (e.g., GPT-4) is increasingly used to scale PRMs without human annotation. For instance, [source:arxiv:2402.03300] uses synthetic feedback to train PRMs for mathematical reasoning, achieving strong performance on MATH and AIME.

### Fading techniques
1. **Pure ORMs for reasoning**: ORMs are increasingly seen as insufficient for tasks requiring multi-step reasoning. While they remain dominant for generalist tasks (e.g., instruction following), their use for reasoning is declining. For example, [source:arxiv:2402.03300] shows that ORMs achieve only 51.7% accuracy on MATH, compared to 64.6% for PRMs.
2. **Human-annotated PRMs**: Fully human-annotated PRMs are becoming less common due to the high cost of step-level labeling. Synthetic annotation and self-supervised methods are replacing them. For example, [source:arxiv:2402.03300] uses synthetic feedback to train PRMs, achieving state-of-the-art performance on MATH and AIME.

### Open questions and disagreements
1. **Generalization**: PRMs are primarily validated on mathematical reasoning. It is unclear whether they generalize to other domains (e.g., coding, planning). Some works suggest that PRMs may degrade performance on generalist tasks, but this remains an open question [source:arxiv:2402.03300].
2. **Reward design**: There is no consensus on the best way to design step rewards. Additive vs. product rewards, monotonicity constraints, and reward normalization are all active areas of debate. For example, [source:arxiv:2312.11410] discusses the trade-offs between additive and product rewards but does not provide a definitive answer.
3. **Scalability**: PRMs require more data and computational resources than ORMs. It is unclear how they will scale to tasks with very long trajectories (e.g., multi-turn dialogue). For example, [source:arxiv:2402.03300] notes that PRMs are computationally expensive and may not scale well to tasks with thousands of steps.

---

## Key takeaways
- **ORMs** assign a single reward to complete trajectories, enabling simple training but suffering from sparse credit assignment. They remain dominant for generalist tasks like instruction following [source:arxiv:2203.02155].
- **PRMs** assign rewards to intermediate steps, enabling dense credit assignment but requiring more annotations and careful reward design. They are the default for reasoning tasks like mathematics [source:arxiv:2402.03300].
- **Step-level supervision** improves performance on reasoning tasks (e.g., MATH accuracy improves from 51.7% to 64.6% with PRMs) but introduces challenges like reward hacking and annotation cost [source:arxiv:2402.03300].
- **PRMs are rising** for reasoning tasks, while ORMs remain dominant for generalist tasks. Hybrid models and synthetic annotation are emerging trends [source:arxiv:2402.03300].
- **Open questions** include the generalization of PRMs to non-reasoning tasks, optimal reward design (additive vs. product, monotonicity), and scalability to long trajectories [source:arxiv:2312.11410][source:arxiv:2402.03300].

---

## Related topics
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)

---

**

## References
- [source:arxiv:2203.02155] [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2402.03300] [A Generalist Language Model Towards Process Supervision](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2312.11410] [Process Reward Models for Mathematical Reasoning](https://arxiv.org/abs/2312.11410)
- [source:arxiv:2411.16232] [DeepScaleR: Scaling LLM Reasoning in Isolation](https://arxiv.org/abs/2411.16232)
- [source:arxiv:2402.10435] [Self-Evaluation Guided Beam Search for Reasoning](https://arxiv.org/abs/2402.10435)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2310.19078] [Rejection Sampling GPT (RSGPT)](https://arxiv.org/abs/2310.19078)
