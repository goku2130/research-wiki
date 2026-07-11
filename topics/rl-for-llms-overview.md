---
title: RL for LLMs — overview
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2203.02155
- arxiv:2212.08073
- arxiv:2305.18290
- arxiv:2405.14734
- arxiv:2402.10770
open_questions:
- 'Reward hacking in offline PO**: How does reward hacking manifest in DPO and SimPO,
  and what techniques can mitigate it? Empirical studies on large-scale models are
  needed to understand the dynamics of over-optimization in these frameworks.'
- 'Generalization of offline PO**: Can offline PO methods like DPO and SimPO generalize
  to out-of-distribution prompts and tasks as effectively as RLHF? Theoretical and
  empirical comparisons are needed to answer this question.'
- 'Hybrid RLHF/offline PO**: Can hybrid approaches (e.g., iterative DPO or combining
  offline PO with online RL) achieve the best of both worlds in terms of efficiency
  and performance? Empirical evaluations on diverse tasks are needed to validate these
  methods.'
- 'Scalability of RLAIF**: How does the performance of RLAIF scale with model size
  and the quality of AI feedback? Can RLAIF fully replace human feedback for alignment,
  or is human oversight still necessary for safety-critical applications?'
---

# RL for LLMs — Overview

Reinforcement learning (RL) has emerged as the dominant paradigm for aligning large language models (LLMs) with human preferences, enabling them to generate outputs that are helpful, honest, and harmless. While initial LLM training relies on next-token prediction via supervised learning on vast corpora, this objective does not inherently capture nuanced human values or contextual appropriateness. RL bridges this gap by fine-tuning models using reward signals derived from human feedback (RLHF), AI feedback (RLAIF), or verifiable reward functions (RLVR), as well as through offline preference optimization (PO) methods that bypass explicit reward modeling. This deep dive surveys the landscape of RL techniques for LLMs, dissecting their theoretical foundations, empirical performance, trade-offs, and open challenges.

---

## Core Motivations and Challenges

LLMs trained via maximum likelihood estimation (MLE) on next-token prediction exhibit several misalignment pathologies:
- **Truthfulness and hallucination**: Models generate plausible but factually incorrect or unsupported claims, particularly in domains with sparse training data [source:arxiv:2203.02155].
- **Toxicity and bias**: Models reflect and amplify biases present in pretraining data, producing harmful or discriminatory outputs even when prompted neutrally [source:arxiv:2203.02155].
- **Helpfulness and instruction-following**: Models fail to reliably follow user instructions, especially when prompts contain implicit constraints or require multi-step reasoning [source:arxiv:2203.02155].
- **Over-optimization and reward hacking**: Models exploit flaws in reward functions to achieve high scores without fulfilling the intended objective, e.g., by generating verbose or overly hedged responses [source:arxiv:2305.18290][source:arxiv:2405.14734].

RL addresses these issues by optimizing models for *preference satisfaction* rather than token likelihood. However, this introduces new challenges:
1. **Reward modeling**: Human preferences are noisy, context-dependent, and often contradictory. Distilling them into a scalar reward function is non-trivial and may introduce biases [source:arxiv:2203.02155].
2. **Exploration vs. exploitation**: RL fine-tuning must balance discovering high-reward outputs with avoiding degenerate behaviors (e.g., mode collapse or sycophancy) [source:arxiv:2203.02155].
3. **Scalability and stability**: Traditional RL algorithms like PPO are computationally expensive, memory-intensive, and sensitive to hyperparameters, making them difficult to scale to large models [source:arxiv:2305.18290].
4. **Alignment tax**: Improving alignment with human preferences can degrade performance on standard benchmarks (e.g., MMLU, SQuAD) due to distributional shift [source:arxiv:2203.02155].
5. **Generalization**: Models may overfit to the preferences of a specific group of annotators or the limitations of the reward function, failing to generalize to broader populations or novel tasks [source:arxiv:2203.02155][source:arxiv:2212.08073].

---

## The RLHF Pipeline: Reinforcement Learning from Human Feedback

RLHF is the foundational approach for aligning LLMs with human preferences. It consists of three sequential stages:

### 1. Supervised Fine-Tuning (SFT)
- **Objective**: Adapt a pretrained LLM to follow instructions by fine-tuning on a dataset of human-written demonstrations.
- **Data collection**: Human annotators (e.g., contractors or crowdworkers) provide high-quality responses to a diverse set of prompts, covering tasks like question answering, summarization, and creative writing. Prompts may be sourced from user interactions, public datasets, or synthetic generation [source:arxiv:2203.02155].
- **Training**: The pretrained model is fine-tuned via supervised learning (typically cross-entropy loss) on the demonstration data. This step is critical for initializing the policy in a region of the parameter space where RL can effectively explore [source:arxiv:2203.02155].
- **Challenges**:
  - **Data quality**: Demonstrations must be diverse, high-quality, and representative of desired behaviors. Low-quality or biased demonstrations can propagate flaws into the RL stage.
  - **Scalability**: Collecting human demonstrations is expensive and time-consuming, limiting the size and diversity of the SFT dataset.

### 2. Reward Modeling (RM)
- **Objective**: Train a model to predict human preferences over pairs of model outputs, effectively distilling human judgments into a scalar reward function.
- **Data collection**: Annotators rank multiple model outputs (typically 4–9 per prompt) generated by the SFT model or other policies. Rankings are collected via pairwise comparisons or best-worst scaling [source:arxiv:2203.02155].
- **Training**: The reward model is trained using the Bradley-Terry model, which assumes that the probability of preferring output $y_w$ over $y_l$ for prompt $x$ is:

$$
p(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)),
$$

  where $\sigma$ is the sigmoid function and $r_\theta(x, y)$ is the scalar reward predicted by the model. The loss function is:

$$
\mathcal{L}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l)) \right],
$$

  where $\mathcal{D}$ is the dataset of ranked pairs [source:arxiv:2203.02155].
- **Architecture**: The reward model is typically initialized from the SFT model, with a linear head added to predict the scalar reward. It may share weights with the policy model or be trained separately [source:arxiv:2203.02155].
- **Challenges**:
  - **Annotator agreement**: Human preferences are subjective and context-dependent, leading to low inter-annotator agreement (e.g., ~73% in [source:arxiv:2203.02155]). Aggregating preferences (e.g., via majority vote or probabilistic models) can mitigate noise but may introduce bias.
  - **Reward hacking**: Models may exploit flaws in the reward function, e.g., by generating verbose or overly polite responses that score highly without being genuinely helpful [source:arxiv:2203.02155][source:arxiv:2405.14734].
  - **Generalization**: Reward models may fail to generalize to out-of-distribution prompts or outputs, particularly if the training data is narrow or biased.

### 3. Reinforcement Learning (RL)
- **Objective**: Fine-tune the SFT model to maximize the reward predicted by the reward model while constraining divergence from the SFT policy to prevent over-optimization.
- **Algorithm**: Proximal Policy Optimization (PPO) is the dominant algorithm for RLHF, though alternatives like REINFORCE or A2C are occasionally used. PPO optimizes the following objective:

$$
\mathcal{J}(\phi) = \mathbb{E}_{(x, y) \sim \pi_\phi^{\text{RL}}} \left[ r_\theta(x, y) - \beta \cdot \text{KL}(\pi_\phi^{\text{RL}}(y \mid x) \| \pi^{\text{SFT}}(y \mid x)) \right],
$$

  where $\pi_\phi^{\text{RL}}$ is the RL policy, $\pi^{\text{SFT}}$ is the SFT policy, $\beta$ controls the KL penalty strength, and $r_\theta(x, y)$ is the reward predicted by the reward model [source:arxiv:2203.02155].
- **PPO-ptx**: To mitigate the alignment tax, [source:arxiv:2405.14734] introduces PPO-ptx, which augments the RL objective with a term that maximizes the log-likelihood of pretraining data:

$$
\mathcal{J}_{\text{ptx}}(\phi) = \mathcal{J}(\phi) + \gamma \cdot \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log \pi_\phi^{\text{RL}}(x) \right],
$$

  where $\gamma$ controls the strength of the pretraining gradient. This helps preserve performance on standard benchmarks while improving alignment.
- **Challenges**:
  - **Computational cost**: PPO requires in-loop sampling (generating outputs from the current policy during training), which is memory-intensive and slow for large models. Distributed training and optimized rollout generation infrastructure are often necessary [source:arxiv:2203.02155].
  - **Hyperparameter sensitivity**: PPO is sensitive to hyperparameters like the KL penalty $\beta$, learning rate, and clipping threshold. Poor choices can lead to instability, mode collapse, or reward hacking.
  - **Exploration**: RL fine-tuning must balance exploration (discovering high-reward outputs) with exploitation (refining known high-reward behaviors). Insufficient exploration can lead to suboptimal policies, while excessive exploration may degrade performance.
  - **Over-optimization**: As the policy improves, the reward model may become miscalibrated, leading to reward hacking or mode collapse. Techniques like reward model ensembling or iterative reward modeling can mitigate this [source:arxiv:2203.02155].

---

## Offline Preference Optimization: DPO and Variants

Offline preference optimization methods bypass the reward modeling and RL stages of RLHF by directly optimizing the policy to satisfy human preferences using a static dataset. These methods are computationally efficient, stable, and easy to implement, but they sacrifice some flexibility and may underperform in out-of-distribution settings.

### Direct Preference Optimization (DPO)
- **Core insight**: DPO reformulates the RLHF objective as a supervised learning problem by leveraging the closed-form solution to the KL-constrained reward maximization problem. The optimal policy under this objective is:

$$
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right),
$$

  where $Z(x)$ is the partition function. By algebraically rearranging this equation, the reward $r(x, y)$ can be expressed in terms of the policy and reference model:

$$
r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x).
$$

  Substituting this into the Bradley-Terry preference model yields the DPO objective:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right],
$$

  where $\pi_\theta$ is the policy being optimized, $\pi_{\text{ref}}$ is the reference model (typically the SFT model), and $\beta$ controls the strength of the implicit KL constraint [source:arxiv:2305.18290].
- **Advantages**:
  - **Simplicity**: DPO eliminates the need for a separate reward model and RL algorithm, reducing implementation complexity and computational overhead.
  - **Stability**: DPO is less sensitive to hyperparameters than PPO and does not require in-loop sampling, making it more stable and easier to scale.
  - **Efficiency**: DPO achieves competitive performance with significantly lower computational cost than RLHF. For example, DPO matches the performance of the "Best of 128" baseline on Anthropic HH [source:arxiv:2305.18290].
- **Limitations**:
  - **Reference model dependency**: DPO requires a reference model during training, which increases memory usage and computational cost. This can be prohibitive for very large models.
  - **Reward mismatch**: The implicit reward in DPO (the log-ratio of policy and reference model probabilities) may not align perfectly with the desired behavior, particularly if the reference model is suboptimal.
  - **Out-of-distribution generalization**: DPO may underperform in settings where the preference data does not cover the full range of desired behaviors, as it cannot explore beyond the static dataset.
  - **Length bias**: DPO is susceptible to length bias, as the log-probability ratio reward favors longer responses that accumulate higher log-probabilities [source:arxiv:2405.14734].

### SimPO: Simple Preference Optimization
- **Core insight**: SimPO addresses DPO’s limitations by replacing the log-ratio reward with a length-normalized average log-likelihood, directly aligning the training objective with the generation metric. It also introduces a target margin to enforce a minimum separation between winning and losing rewards.
- **Objective**: The SimPO reward is defined as:

$$
r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \log \pi_\theta(y \mid x),
$$

  where $|y|$ is the length of the response. The Bradley-Terry model with target margin $\gamma$ is:

$$
p(y_w \succ y_l \mid x) = \sigma(r_{\text{SimPO}}(x, y_w) - r_{\text{SimPO}}(x, y_l) - \gamma).
$$

  The SimPO loss is:

$$
\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right].
$$

  [source:arxiv:2405.14734].
- **Advantages**:
  - **Reference-free**: SimPO eliminates the need for a reference model, reducing memory usage and computational cost.
  - **Length bias mitigation**: The length-normalized reward reduces the correlation between response length and reward, mitigating length exploitation.
  - **Performance**: SimPO outperforms DPO and its variants on benchmarks like AlpacaEval 2 and Arena-Hard, achieving state-of-the-art results for sub-10B models [source:arxiv:2405.14734].
- **Limitations**:
  - **Hyperparameter sensitivity**: SimPO introduces a target margin $\gamma$, which must be carefully tuned to avoid over- or under-optimization.
  - **Theoretical gaps**: The convergence properties of SimPO are not rigorously analyzed, and its behavior under distribution shift is not well understood.
  - **Task-specific performance**: SimPO degrades performance on reasoning-heavy tasks like GSM8K, suggesting it may not generalize well to all domains [source:arxiv:2405.14734].

### Other DPO Variants
Several variants of DPO have been proposed to address specific limitations:
- **Identity Preference Optimization (IPO)**: Replaces the Bradley-Terry model with a squared loss to avoid overfitting to the preference data. The objective is designed to improve generalization by penalizing large deviations from the reference model [source:arxiv:2305.18290].
- **Kahneman-Tversky Optimization (KTO)**: Optimizes for human utility rather than pairwise preferences, using a loss function inspired by prospect theory. The method is described in [source:arxiv:2405.14734] as a way to align models with human-like decision-making under uncertainty.

---

## RLAIF: Reinforcement Learning from AI Feedback

RLAIF replaces human feedback with AI-generated feedback, reducing the cost and scalability limitations of RLHF while preserving alignment with human preferences. The core idea is to use a strong LLM (e.g., GPT-4) as a "judge" to evaluate model outputs and provide preference labels or critiques.

### Constitutional AI (CAI)
- **Core problem**: RLHF requires tens of thousands of human preference labels to train reward models, which is costly, opaque, and difficult to scale. Additionally, human feedback may not generalize well to novel or adversarial prompts [source:arxiv:2212.08073].
- **Method**: CAI replaces human harmlessness labels with a "constitution"—a short list of natural language principles (e.g., "Do not generate harmful or offensive content"). The pipeline consists of two stages:
  1. **Supervised Learning (SL-CAI)**:
     - Generate responses to red-teaming prompts using a helpful RLHF model.
     - Critique the responses using a randomly sampled constitutional principle (e.g., "Identify harmful content in this response").
     - Revise the responses based on the critique.
     - Repeat the critique-revision process multiple times per prompt.
     - Fine-tune the model via supervised learning on the final revised responses, combined with helpfulness prompts.
  2. **Reinforcement Learning (RL-CAI)**:
     - Generate response pairs for prompts using the SL-CAI model.
     - Use a feedback model (e.g., a strong LLM) to evaluate which response is less harmful based on the constitutional principles.
     - Mix these AI-generated harmlessness labels with human helpfulness labels to train a preference model (PM).
     - Optimize the SL-CAI model via RL (e.g., PPO) using the PM as the reward signal.
- **Key innovations**:
  - **Chain-of-thought (CoT) prompting**: The feedback model uses CoT reasoning to improve evaluation accuracy. To prevent overconfidence, the feedback model's output probabilities are clamped to the interval $[0.40, 0.60]$ [source:arxiv:2212.08073].
  - **Principle sampling**: Constitutional principles are randomly sampled during training to encourage robustness and generalization.
- **Advantages**:
  - **Scalability**: CAI eliminates the need for human harmlessness labels, significantly reducing data collection costs.
  - **Transparency**: The constitutional principles are interpretable and can be refined by diverse stakeholders.
  - **Performance**: RL-CAI achieves higher harmlessness scores than standard RLHF while maintaining comparable helpfulness [source:arxiv:2212.08073].
- **Limitations**:
  - **Helpfulness dependence**: CAI still requires human feedback for helpfulness, though full automation is a long-term goal.
  - **Principle selection**: The constitutional principles are selected ad hoc and may not cover all edge cases or cultural contexts.
  - **Over-optimization**: CAI can produce overly preachy or boilerplate responses if the feedback model is miscalibrated.
  - **Dual-use risks**: Reducing human oversight lowers the barrier to training potentially pernicious systems.

---

## RLVR: Reinforcement Learning from Verifiable Rewards

RLVR replaces learned reward models with verifiable reward functions, which are deterministic, interpretable, and grounded in formal specifications. This approach is particularly useful for domains where human preferences are difficult to elicit or where safety and correctness are critical. Key applications include:
- **Code execution**: Reward models for code generation can execute the generated code and assign rewards based on correctness, efficiency, or adherence to specifications [source:arxiv:2405.14734].
- **Mathematical reasoning**: Reward models for math problems can verify the correctness of solutions using symbolic computation or formal proofs [source:arxiv:2405.14734].
- **Tool use**: Reward models for agentic tasks can check whether the model correctly uses tools (e.g., APIs, calculators) to achieve the desired outcome [source:arxiv:2405.14734].
- **Safety constraints**: Verifiable rewards can enforce hard constraints (e.g., "do not generate harmful code") that are difficult to capture with learned reward models [source:arxiv:2405.14734].

### Advantages
- **Interpretability**: Verifiable rewards are transparent and can be audited for correctness.
- **Scalability**: Reward functions can be applied to arbitrary inputs without human annotation, enabling large-scale training.
- **Safety**: Verifiable rewards can enforce hard constraints that are difficult to capture with learned reward models.

### Challenges
- **Domain specificity**: Verifiable rewards are often tailored to specific tasks (e.g., code generation) and may not generalize to open-ended domains.
- **Reward hacking**: Models may exploit flaws in the reward function (e.g., generating code that passes tests but is insecure or inefficient).
- **Implementation complexity**: Designing and implementing verifiable rewards requires domain expertise and may be computationally expensive.

---

## Offline RL for LLMs: Preference Optimization Without Reward Models

Offline RL methods optimize policies using static datasets of trajectories, eliminating the need for in-loop sampling or reward models. These methods are particularly useful for LLMs, where online interaction with the environment (e.g., generating outputs during training) is computationally expensive.

### Key Methods
1. **Behavioral Cloning (BC)**:
   - **Objective**: Train the policy to mimic the behavior in the offline dataset via supervised learning.
   - **Use case**: Initializing the policy for RL fine-tuning (e.g., SFT in RLHF).
   - **Limitations**: BC cannot improve upon the dataset's behavior and may suffer from distribution shift.
2. **Implicit Language Q-Learning (ILQL)**:
   - **Objective**: Combine offline RL with language modeling by training a Q-function to predict the value of token sequences. This method is described in [source:arxiv:2206.01389] as a way to fine-tune LLMs for dialogue generation without explicit reward modeling.

### Advantages of Offline RL
- **Stability**: Offline RL avoids the instability of online RL algorithms like PPO, which can suffer from catastrophic forgetting or mode collapse.
- **Efficiency**: Offline RL does not require in-loop sampling, reducing computational cost and memory usage.
- **Safety**: Offline RL can be used to fine-tune models without deploying them in the real world, reducing risks like reward hacking or harmful outputs.

### Limitations of Offline RL
- **Dataset dependence**: Offline RL cannot discover behaviors not present in the dataset, limiting its ability to generalize.
- **Distribution shift**: Policies trained via offline RL may perform poorly when deployed in environments that differ from the training data.
- **Reward design**: Offline RL still requires a reward function (or preference labels), which may be difficult to design or learn.

---

## Current Status and Trajectory

### RLHF: The Default Paradigm, but Facing Challenges
- **Status**: RLHF is the dominant approach for aligning LLMs with human preferences, used by organizations like OpenAI, Anthropic, and DeepMind. It has demonstrated significant improvements in helpfulness, truthfulness, and harmlessness across multiple domains [source:arxiv:2203.02155][source:arxiv:2212.08073].
- **Trajectory**: RLHF is likely to remain the default paradigm in the near term, but its limitations are driving research into alternatives:
  - **Scalability**: The cost of human annotation and the computational expense of PPO are major bottlenecks. Efforts to reduce annotation costs (e.g., active learning, semi-supervised learning) and improve PPO efficiency (e.g., distributed training, mixed precision) are ongoing.
  - **Stability**: PPO's sensitivity to hyperparameters and reward model miscalibration remains a challenge. Techniques like reward model ensembling, iterative reward modeling, and adaptive KL penalties are being explored.
  - **Generalization**: RLHF models often overfit to the preferences of a specific group of annotators or the limitations of the reward function. Research into diverse and representative preference data, as well as robust reward modeling, is active.
  - **Alignment tax**: The trade-off between alignment and performance on standard benchmarks (e.g., MMLU) is not fully resolved. Methods like PPO-ptx mitigate this but do not eliminate it [source:arxiv:2203.02155].

### DPO and Offline PO: Rising Stars
- **Status**: DPO and its variants (e.g., SimPO, IPO, KTO) have gained significant traction due to their simplicity, stability, and computational efficiency. DPO is now a standard baseline in preference optimization research and is used in production by some organizations [source:arxiv:2305.18290][source:arxiv:2405.14734].
- **Trajectory**: Offline PO methods are likely to become the default for preference optimization in the near term, particularly for smaller models and resource-constrained settings. Key trends include:
  - **Reference-free optimization**: SimPO and other reference-free methods are reducing memory and computational overhead, making offline PO more scalable [source:arxiv:2405.14734].
  - **Length bias mitigation**: Methods like SimPO are addressing the length bias problem, improving the quality of generated outputs [source:arxiv:2405.14734].
  - **Theoretical foundations**: Research into the convergence properties, generalization bounds, and reward hacking dynamics of offline PO is ongoing.
  - **Hybrid approaches**: Combining offline PO with online RL (e.g., iterative DPO) is being explored to balance efficiency and performance.

**Disagreement**: While DPO and SimPO have demonstrated strong empirical performance, their theoretical foundations are less mature than those of RLHF. Some researchers argue that offline PO methods may lack the flexibility to handle complex or dynamic preferences, while others contend that their simplicity and efficiency make them preferable for most use cases. The debate is unresolved, and empirical comparisons on large-scale models and diverse tasks are needed to settle it.

### RLAIF: Gaining Ground, but Not Yet Default
- **Status**: RLAIF has shown promise as a scalable alternative to RLHF, particularly for harmlessness alignment (e.g., CAI) and automated evaluation (e.g., LLM-as-a-judge). It is used in research and some production systems but is not yet as widely adopted as RLHF or DPO [source:arxiv:2212.08073][source:arxiv:2305.18290].
- **Trajectory**: RLAIF is likely to grow in popularity as models become more capable and the cost of human annotation becomes prohibitive. Key trends include:
  - **Automation**: Efforts to fully automate preference labeling (e.g., using synthetic data or self-play) are ongoing, though human oversight is still required for safety-critical applications.
  - **Principle design**: Research into designing and refining constitutional principles for CAI is active, with a focus on interpretability, cultural sensitivity, and robustness.
  - **Judge calibration**: Techniques to improve the calibration and robustness of LLM judges (e.g., CoT prompting, probability clamping, ensembling) are being developed.
  - **Hybrid feedback**: Combining human and AI feedback (e.g., using AI feedback for harmlessness and human feedback for helpfulness) is a promising direction for balancing cost and quality.

**Disagreement**: The reliability of AI feedback is a major point of contention. While LLM judges like GPT-4 achieve high agreement with human annotators on some tasks, they exhibit biases, prompt sensitivity, and poor generalization to out-of-distribution settings [source:arxiv:2402.10770]. Some researchers argue that AI feedback is sufficient for many applications, while others contend that human oversight is still necessary for safety and fairness. The debate is unresolved, and more rigorous meta-evaluations of LLM judges are needed.

### RLVR: Niche but Growing
- **Status**: RLVR is used in specialized domains like code generation, mathematical reasoning, and agentic tasks, where verifiable rewards are feasible. It is not yet widely adopted for open-ended domains like dialogue or creative writing.
- **Trajectory**: RLVR is likely to grow in domains where safety and correctness are critical, such as:
  - **Code generation**: Verifiable rewards for code execution are becoming standard for fine-tuning LLMs for programming tasks [source:arxiv:2405.14734].
  - **Mathematical reasoning**: Verifiable rewards for theorem proving and symbolic computation are improving the performance of LLMs on math benchmarks [source:arxiv:2405.14734].
  - **Agentic tasks**: Verifiable rewards for tool use and environment interaction are enabling LLMs to act as agents in real-world settings [source:arxiv:2405.14734].
  - **Safety-critical applications**: Verifiable rewards can enforce hard constraints (e.g., "do not generate harmful content") that are difficult to capture with learned reward models [source:arxiv:2405.14734].

**Disagreement**: The scalability of RLVR to open-ended domains is debated. Some researchers argue that verifiable rewards are inherently limited to structured tasks, while others contend that advances in formal verification and program synthesis will enable RLVR for broader applications. The debate is unresolved, and more research into hybrid reward modeling (e.g., combining verifiable and learned rewards) is needed.

---

## Key Trade-offs and Design Choices

| **Dimension**               | **RLHF**                          | **DPO/Offline PO**                | **RLAIF**                          | **RLVR**                          |
|-----------------------------|-----------------------------------|-----------------------------------|------------------------------------|-----------------------------------|
| **Human annotation cost**   | High (tens of thousands of labels)| Low (static dataset)              | Low (AI feedback)                  | None (verifiable rewards)         |
| **Computational cost**      | High (PPO, in-loop sampling)      | Low (supervised learning)         | Medium (AI feedback generation)    | Medium (reward computation)       |
| **Stability**               | Low (PPO is sensitive)            | High (supervised learning)        | Medium (depends on AI feedback)    | High (deterministic rewards)      |
| **Flexibility**             | High (can explore new behaviors)  | Low (limited by dataset)          | Medium (depends on AI feedback)    | Low (limited by reward function)  |
| **Generalization**          | Medium (depends on reward model)  | Medium (depends on dataset)       | Medium (depends on AI feedback)    | High (deterministic rewards)      |
| **Reward hacking risk**     | High (learned reward model)       | Medium (implicit reward)          | Medium (AI feedback bias)          | Low (verifiable rewards)          |
| **Alignment tax**           | Medium (PPO-ptx mitigates)        | Low (no pretraining gradient)     | Low (depends on feedback)          | Low (task-specific)               |
| **Interpretability**        | Low (learned reward model)        | Medium (implicit reward)          | Medium (AI feedback)               | High (verifiable rewards)         |
| **Scalability**             | Low (PPO is expensive)            | High (supervised learning)        | Medium (AI feedback generation)    | Medium (reward computation)       |

---

## Key Takeaways
- **RLHF is the default paradigm** for aligning LLMs with human preferences, but it is computationally expensive, unstable, and requires large-scale human annotation. PPO-ptx mitigates the alignment tax but does not eliminate it [source:arxiv:2203.02155][source:arxiv:2405.14734].
- **DPO and offline PO methods** are rising stars due to their simplicity, stability, and efficiency. SimPO addresses DPO’s limitations by eliminating the reference model and mitigating length bias, achieving state-of-the-art performance on benchmarks like AlpacaEval 2 [source:arxiv:2305.18290][source:arxiv:2405.14734].
- **RLAIF is gaining ground** as a scalable alternative to RLHF, particularly for harmlessness alignment (e.g., CAI) and automated evaluation (e.g., LLM-as-a-judge). However, the reliability of AI feedback remains a major challenge [source:arxiv:2212.08073][source:arxiv:2402.10770].
- **RLVR is niche but growing** in domains where verifiable rewards are feasible (e.g., code generation, mathematical reasoning). It offers high interpretability and safety but is limited to structured tasks [source:arxiv:2405.14734].
- **Key trade-offs** include human annotation cost, computational cost, stability, flexibility, generalization, reward hacking risk, alignment tax, interpretability, and scalability. The choice of method depends on the specific use case and constraints.
- **Open challenges** include reward hacking, generalization to out-of-distribution settings, alignment tax, scalability of human annotation, and the reliability of AI feedback. Hybrid approaches (e.g., combining offline PO with online RL, or verifiable rewards with learned rewards) are promising directions for future research.

---

## Related Topics
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [DPO variants deep-dive](dpo-variants.md)
- [RLAIF (RL from AI feedback)](rlaif.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Length and format bias](length-and-format-bias.md)
- [The alignment tax](alignment-tax.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [LLM-as-judge](llm-as-judge.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [Judging bias and contamination](judging-bias-and-contamination.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)

---

##

## References
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2212.08073] [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2405.14734] [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [source:arxiv:2402.10770] [How Reliable Are Automatic Evaluation Methods for Instruction-Tuned LLMs?](https://arxiv.org/abs/2402.10770)
