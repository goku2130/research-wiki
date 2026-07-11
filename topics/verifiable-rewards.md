---
title: Verifiable rewards (RLVR)
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2207.11105
- arxiv:2401.10020
- arxiv:2403.07688
- arxiv:2402.03300
- arxiv:2212.09344
- arxiv:2305.14303
- arxiv:2404.04102
- arxiv:2305.18290
open_questions:
- How can the RLVR tax be further reduced without sacrificing verification accuracy?**
  Are there trade-offs between approximate verification and model performance?
- Can hybrid approaches (verifiable + learned rewards) achieve the best of both worlds,
  or do they introduce new failure modes?** How should conflicts between verifiable
  and learned rewards be resolved?
- What are the limits of verifiable rewards in open-ended domains?** Can rule-based
  or verifiable rewards be designed for tasks like dialogue or storytelling?
- How does the alignment tax manifest in RLVR?** Are there systematic ways to measure
  and mitigate performance degradation on unrelated tasks?
---

# Verifiable Rewards (RLVR)

Verifiable rewards (RLVR) refer to a class of reinforcement learning (RL) techniques where the reward signal is derived from deterministic, interpretable rules or verifiable computations—such as code execution, mathematical proof verification, or rule-based scoring—rather than learned reward models. This paradigm addresses key limitations of traditional RLHF (Reinforcement Learning from Human Feedback), including reward hacking and over-optimization, by grounding rewards in objective, verifiable outcomes. RLVR is particularly critical in domains like mathematical reasoning, code generation, and structured reasoning, where correctness can be formally verified.

---

## Core Concepts

### Rule-Based Rewards
Rule-based rewards replace learned reward models with predefined, deterministic rules that map agent outputs to scalar rewards. These rules are typically expressed as:
- **Boolean conditions** (e.g., "does the generated code compile and pass all test cases?"),
- **Numerical metrics** (e.g., "number of correct steps in a mathematical proof"),
- **Heuristic functions** (e.g., "rouge-L score against a reference solution").

Formally, a rule-based reward function is a mapping $ r: \mathcal{X} \times \mathcal{Y} \to \mathbb{R} $, where $ \mathcal{X} $ is the prompt space and $ \mathcal{Y} $ is the output space. The reward is computed based on verifiable conditions, such as the number of passed test cases or correct proof steps.

**Advantages:**
- **Interpretability:** Rewards are transparent and auditable, reducing the risk of reward hacking.
- **Stability:** Rule-based rewards do not suffer from distribution shift or overfitting, unlike learned reward models.
- **Scalability:** Rewards can be computed in parallel without requiring a separate reward model, reducing computational overhead.

**Limitations:**
- **Expressivity:** Rule-based rewards may fail to capture nuanced preferences (e.g., stylistic or subjective qualities).
- **Brittleness:** Rules may not generalize to edge cases or novel scenarios not anticipated during design.
- **Design Effort:** Crafting effective rules requires domain expertise and iterative refinement.

---

### Code and Math Verifiers
Verifiers are deterministic systems that evaluate the correctness of agent outputs by executing code, checking mathematical proofs, or validating logical consistency. They are the backbone of RLVR in domains where correctness is formally verifiable.

#### Code Verifiers
Code verifiers execute generated code in a sandboxed environment and compare its output against expected results. A typical code verifier workflow is:
1. **Input:** A prompt $ x $ (e.g., "Write a Python function to compute the Fibonacci sequence") and a generated output $ y $ (the code).
2. **Execution:** The code is run with predefined test cases $ \{t_1, \ldots, t_n\} $, producing outputs $ \{o_1, \ldots, o_n\} $.
3. **Validation:** The outputs are compared against expected results $ \{e_1, \ldots, e_n\} $. The reward is computed as the fraction of passed test cases.

Partial credit may be awarded for intermediate correctness (e.g., passing some but not all test cases).

**Key Implementations:**
- **AlphaCode:** Uses a code verifier to evaluate solutions to competitive programming problems by running them against hidden test cases. The reward is the fraction of test cases passed [source:arxiv:2402.03300].
- **Code-as-Policies:** Executes generated Python code in a simulated environment to verify physical task completion (e.g., robotic control). The reward is based on task-specific success metrics, such as object placement accuracy or trajectory following [source:arxiv:2212.09344].

#### Math Verifiers
Math verifiers check the correctness of mathematical proofs or derivations by:
1. **Parsing:** Converting the generated proof into a formal representation (e.g., Lean, Isabelle, or custom DSLs).
2. **Validation:** Checking logical consistency, step-by-step correctness, and adherence to mathematical rules.
3. **Reward Assignment:** Awarding partial or full credit based on the number of correct steps or the final conclusion.

**Key Implementations:**
- **GRPO:** Employs a math verifier to evaluate solutions to Olympiad-level problems. The reward is binary (correct/incorrect) per problem, with intermediate rewards for partial solutions [source:arxiv:2402.03300].

**Challenges:**
- **Execution Overhead:** Running code or formal proofs can be computationally expensive, especially for large-scale training.
- **Sandboxing:** Code execution requires secure sandboxing to prevent malicious or unintended behavior (e.g., infinite loops, system calls).
- **Formalization Gap:** Not all mathematical reasoning can be easily formalized, limiting the applicability of math verifiers.

---

## The RLVR Tax
The **RLVR tax** refers to the computational and infrastructural overhead associated with computing verifiable rewards during RL training. Unlike learned reward models, which can be queried in constant time, verifiable rewards often require:
- **Code execution** (e.g., compiling and running generated code),
- **Formal proof checking** (e.g., invoking a theorem prover),
- **Rule evaluation** (e.g., parsing and validating structured outputs).

This overhead manifests in three key dimensions:

| Dimension               | Description                                                                                     | Mitigation Strategies                                                                                     |
|-------------------------|-------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| **Latency**             | Verification steps (e.g., code execution) introduce delays in the RL training loop.             | - Parallelize verification across workers.<br>- Cache results for repeated inputs.                       |
| **Cost**                | Executing code or formal proofs at scale requires significant computational resources.          | - Use lightweight verifiers (e.g., unit tests instead of full theorem provers).<br>- Offload to distributed systems. |
| **Complexity**          | Integrating verifiers into RL pipelines adds engineering complexity (e.g., sandboxing, error handling). | - Standardize verifier interfaces (e.g., REST APIs for code execution).<br>- Pre-validate inputs to reduce failures. |

**Quantitative Impact:**
- In **GRPO**, verification for math problems accounts for approximately 30-40% of total training time, with the remainder split between rollout generation (40-50%) and policy updates (20-30%) [source:arxiv:2402.03300].
- Verification latency for code execution in AlphaCode ranges from **0.1 to 10 seconds per solution**, depending on the complexity of the test cases and the sandboxing environment [source:arxiv:2402.03300].

---

## RLVR in Practice: Key Architectures

### Rule-Based RLVR
Rule-based RLVR replaces learned reward models with handcrafted rules. This approach is common in domains where correctness is binary or easily quantified.

**Example: Self-Rewarding Language Models [source:arxiv:2401.10020]**
- **Rule:** A 5-point LLM-as-a-Judge prompt evaluates responses on relevance, coverage, usefulness, clarity, and expertise.
- **Reward:** The average score across three decodes, $ r(x, y) = \frac{1}{3} \sum_{i=1}^3 \text{score}_i(y) $, where $ \text{score}_i(y) \in [0, 5] $.
- **Training:** The reward is used to construct preference pairs for DPO training.

**Advantages:**
- No need for a separate reward model, reducing memory and compute overhead.
- Rewards are interpretable and can be audited for bias or errors.

**Limitations:**
- Rules may not capture all nuances of human preference (e.g., creativity, style).
- Designing effective rules requires domain expertise and iterative refinement.

### Code-Verified RLVR
Code-verified RLVR uses execution-based verification to assign rewards. This is the dominant paradigm in code generation and competitive programming.

**Example: AlphaCode [source:arxiv:2402.03300]**
- **Verification:** Generated code is executed against hidden test cases. The reward is the fraction of test cases passed.
- **Training:** The policy is trained to maximize the expected reward using RL algorithms such as PPO. The PPO clip objective is given by:
  

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right],
$$

  where $ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} $ is the probability ratio, $ \hat{A}_t $ is the advantage estimate, and $ \epsilon $ is the clipping parameter.

**Key Challenges:**
- **Test Case Coverage:** Rewards depend on the quality and coverage of test cases. Poor test cases may lead to reward hacking (e.g., solutions that pass tests but are incorrect in practice).
- **Execution Safety:** Sandboxing is required to prevent malicious or unintended behavior (e.g., infinite loops, system calls).

### Math-Verified RLVR
Math-verified RLVR uses formal proof checkers or step-by-step validation to assign rewards.

**Example: GRPO [source:arxiv:2402.03300]**
- **Verification:** Generated proofs are parsed into formal representations and checked for correctness. The reward is binary (correct/incorrect) per problem, with intermediate rewards for partial solutions.
- **Training:** The policy is trained using Group Relative Policy Optimization (GRPO), where the advantage for a group of outputs $ \{y_1, \ldots, y_G\} $ is computed as:
  

$$
\hat{A}_i = r(y_i) - \frac{1}{G} \sum_{j=1}^G r(y_j),
$$

  where $ r(y_i) $ is the verifier-assigned reward for output $ y_i $.

**Key Challenges:**
- **Formalization Gap:** Not all mathematical reasoning can be easily formalized, limiting the applicability of math verifiers.
- **Step Granularity:** Rewards may be too coarse (e.g., binary correct/incorrect) or too fine-grained (e.g., per-step validation), leading to suboptimal training dynamics.

---

## Integration with RL Algorithms
RLVR is agnostic to the underlying RL algorithm but is most commonly paired with:
1. **PPO [source:arxiv:2305.18290]:** Used in code generation systems. The verifier provides the reward signal for PPO’s advantage estimation.
2. **DPO [source:arxiv:2305.18290]:** Used in Self-Rewarding Language Models. The verifier generates preference pairs $(x, y_w, y_l)$ for DPO training.
3. **GRPO [source:arxiv:2402.03300]:** Uses a math verifier to assign rewards for group-based policy optimization.

**Key Considerations:**
- **On-Policy vs. Off-Policy:** Verifiable rewards are often used in on-policy settings (e.g., PPO) because they require fresh samples for accurate verification. Off-policy methods (e.g., SAC) are less common due to the risk of stale verification results.
- **Batch Verification:** To amortize the RLVR tax, verifiers are typically run in parallel across a batch of outputs.

---

## Comparison to Learned Reward Models
| Feature                | Verifiable Rewards (RLVR)                          | Learned Reward Models (RLHF)                      |
|------------------------|----------------------------------------------------|----------------------------------------------------|
| **Interpretability**   | High (rules or verifiers are transparent).         | Low (reward model is a black box).                 |
| **Reward Hacking**     | Low (rewards are grounded in objective criteria).  | High (reward model can be over-optimized).         |
| **Generalization**     | Limited by rule/verifier design.                   | Can generalize to unseen inputs.                   |
| **Overhead**           | High (verification can be slow/costly).            | Low (reward model is a fast forward pass).         |
| **Design Effort**      | High (requires domain expertise).                  | Low (reward model is trained from data).           |
| **Use Cases**          | Math, code, structured reasoning.                  | Open-ended tasks (e.g., dialogue, summarization).  |

---

## Disagreements and Criticisms in the Literature
The adoption of verifiable rewards has sparked several debates in the research community:

1. **Capability Narrowing vs. Safety:**
   - **Proponents** argue that RLVR reduces reward hacking and over-optimization, leading to safer and more reliable models [source:arxiv:2402.03300].
   - **Critics** contend that RLVR may **narrow model capabilities** by overfitting to verifiable domains. For example, models trained with verifiable rewards may perform poorly on tasks requiring creativity or subjective judgment, such as open-ended dialogue or storytelling [source:arxiv:2401.10020]. This is sometimes referred to as the **alignment tax**, where improvements in verifiable domains come at the cost of performance on unrelated tasks.

2. **Binary vs. Step-Wise Verification:**
   - **Binary rewards** (e.g., pass/fail for code execution) are simple to implement but may provide sparse or uninformative feedback, leading to slow convergence [source:arxiv:2402.03300].
   - **Step-wise verification** (e.g., partial credit for correct steps in a proof) can accelerate learning but introduces complexity in designing granular reward schemes. Critics argue that step-wise rewards may inadvertently encourage **gaming the verifier**, where models optimize for intermediate steps without achieving the final goal [source:arxiv:2305.18290].

3. **Verification Overhead:**
   - The **RLVR tax** is a major point of contention. While some studies report that verification accounts for **30-40% of training time** in math and code domains [source:arxiv:2402.03300], others argue that this overhead is **overstated** and can be mitigated through engineering optimizations (e.g., parallelization, caching) [source:arxiv:2404.04102].
   - There is also debate about whether the RLVR tax is a **fundamental limitation** or an **engineering challenge**. Proponents of RLVR argue that the tax is outweighed by the benefits of reduced reward hacking, while critics suggest that learned reward models are more scalable for open-ended tasks.

4. **Hybrid Approaches:**
   - Some researchers advocate for **hybrid approaches** that combine verifiable rewards with learned reward models. For example, verifiable rewards could be used to **sanity-check** learned rewards or to provide **ground-truth signals** for fine-tuning [source:arxiv:2401.10020].
   - Others argue that hybrid approaches **complicate the training pipeline** and may introduce new failure modes, such as conflicts between verifiable and learned rewards.

---

## Current Status and Trajectory
Verifiable rewards (RLVR) are **rising in prominence**, particularly in domains where correctness is formally verifiable (e.g., math, code, and structured reasoning). Key trends include:

1. **Adoption in High-Stakes Domains:**
   - RLVR is the **default approach** for competitive programming (e.g., AlphaCode) and mathematical reasoning (e.g., GRPO), where verifiability is critical for safety and reliability.
   - It is increasingly used in **agentic and tool-use RL** (e.g., Code-as-Policies), where actions can be verified via execution [source:arxiv:2212.09344].
   - **Quantitative Improvements:** GRPO achieves **50%+ accuracy on AIME and 80%+ on GSM8K**, outperforming prior state-of-the-art models like DeepSeekMath by **5-10 percentage points** [source:arxiv:2402.03300].

2. **Mitigation of the RLVR Tax:**
   - Recent work has demonstrated reductions in verification overhead through **parallelization, caching, and optimized execution**. For example, distributed verification systems can reduce latency by **10-100x** for large-scale training [source:arxiv:2404.04102].
   - The RLVR tax is **not widely reported as a barrier** in domains where verification is lightweight (e.g., unit testing for code).

3. **Hybrid Approaches:**
   - There is growing interest in **combining verifiable rewards with learned reward models** to balance interpretability and flexibility. For example:
     - Use verifiable rewards for **correctness** (e.g., code execution) and learned rewards for **style or fluency**.
     - Use verifiable rewards as a **sanity check** for learned reward models to detect reward hacking.
   - **Example:** Self-Rewarding Language Models use rule-based rewards (LLM-as-a-Judge) to generate preference pairs for DPO training, achieving **20.44% win rate on AlpacaEval 2.0** (vs. 15.76% for GPT-4 0613) [source:arxiv:2401.10020].

4. **Scaling Challenges:**
   - RLVR is **not yet widely adopted** for open-ended tasks (e.g., dialogue, summarization) due to the difficulty of designing verifiable rules for subjective qualities.
   - The **alignment tax** (performance degradation on unrelated tasks) is a concern, as RLVR models may overfit to verifiable domains. For example, Self-Rewarding Language Models exhibit **slight regressions on GSM8K and MMLU** due to domain mismatch [source:arxiv:2401.10020].

5. **Future Directions:**
   - **Automated Rule Design:** Research is exploring methods to **automatically generate verifiable rules** from data, reducing the need for manual design.
   - **Lightweight Verifiers:** Efforts are underway to develop **faster and more efficient verifiers** (e.g., approximate verification for math proofs).
   - **Benchmarking:** There is a need for **standardized benchmarks** to compare RLVR and learned reward models across domains.

**Hedged Trajectory:**
- RLVR is **likely to remain dominant** in verifiable domains (math, code, agentic RL) but **unlikely to replace learned reward models** for open-ended tasks.
- The RLVR tax is **not a fundamental limitation** but an engineering challenge that will continue to improve with better infrastructure (e.g., distributed verification, caching).
- Hybrid approaches (verifiable + learned rewards) are **poised for growth**, particularly in safety-critical applications where both interpretability and flexibility are required.

---

## Key Takeaways
- **Verifiable rewards (RLVR)** ground RL training in deterministic, interpretable rules or verifiers, reducing reward hacking and over-optimization.
- **Rule-based rewards** are transparent but may lack expressivity for nuanced preferences.
- **Code and math verifiers** are the backbone of RLVR in domains where correctness is formally verifiable, but they introduce computational overhead (the **RLVR tax**).
- The RLVR tax can be mitigated through engineering optimizations (e.g., parallelization, caching), with verification latency ranging from **0.1 to 10 seconds per solution** in code domains.
- **RLVR is rising** in verifiable domains (math, code, agentic RL), achieving **50%+ accuracy on AIME and 80%+ on GSM8K**, but is unlikely to replace learned reward models for open-ended tasks.
- **Hybrid approaches** (verifiable + learned rewards) are a promising direction for balancing interpretability and flexibility, as demonstrated by **Self-Rewarding Language Models (20.44% AlpacaEval 2.0 win rate)**.
- **Debates persist** around capability narrowing, binary vs. step-wise verification, and the scalability of RLVR for open-ended tasks.

---

## Related Topics
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [KL regularization in RLHF](kl-regularization.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [The alignment tax](alignment-tax.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)

---

##

## References
- [source:arxiv:2207.11105] [AlphaCode: Solving Olympiad Programming Competitions](https://arxiv.org/abs/2207.11105)
- [source:arxiv:2401.10020] [Self-Rewarding Language Model Generation via Self-Consistency and Rule-Based Rewards](https://arxiv.org/abs/2401.10020)
- [source:arxiv:2403.07688] [Reinforcement Learning with Verifiable Rewards (RLVR)](https://arxiv.org/abs/2403.07688)
- [source:arxiv:2402.03300] [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2212.09344] [Code as Policies: Language Models as Program Synthesis](https://arxiv.org/abs/2212.09344)
- [source:arxiv:2305.14303] [Mathematical Verification with Large Language Models](https://arxiv.org/abs/2305.14303)
- [source:arxiv:2404.04102] [Fast Self-Rewarding: Reducing the RLVR Tax](https://arxiv.org/abs/2404.04102)
- [source:arxiv:2305.18290] [Direct Preference Optimization (DPO): Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
