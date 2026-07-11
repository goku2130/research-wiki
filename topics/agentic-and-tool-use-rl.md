---
title: Agentic and tool-use RL
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:1905.09836
- arxiv:2210.03629
- arxiv:2302.04761
- arxiv:2303.11366
- arxiv:2305.10601
- arxiv:2406.11903
- arxiv:2401.10020
- arxiv:2406.13479
open_questions:
- How can tool chaining be explicitly integrated into self-supervised tool-learning
  frameworks like Toolformer?
- What are the scaling laws for self-improving systems (e.g., self-rewarding models)
  beyond 3 iterations?
- Can hybrid approaches (e.g., search-augmented memory) reconcile the trade-offs between
  ToT and Reflexion?
- How can verbal reinforcement learning (Reflexion) be extended to tasks requiring
  highly diverse exploration (e.g., WebShop)?
---

# Agentic and Tool-Use Reinforcement Learning: A Deep Dive

Reinforcement learning (RL) has evolved to enable language models (LMs) to interact with complex, multi-step environments through *agentic* behavior—autonomous decision-making that integrates reasoning, memory, and tool use. This paradigm shift addresses a fundamental limitation of static LMs: their inability to dynamically gather information, correct errors, or adapt strategies mid-task. Agentic RL bridges this gap by formalizing tool use as a sequential decision process, where models learn to interleave reasoning traces with environmental actions to maximize task success. The core challenges—multi-turn interaction, exploration in high-dimensional discrete spaces, and scalable reward modeling—define the frontier of this field.

---

## Core Concepts and Definitions

### Agentic Behavior
An *agentic* LM operates as a closed-loop system: it observes an environment, selects actions (including tool invocations), receives feedback, and updates its internal state to achieve a goal. This contrasts with *reactive* models (e.g., standard autoregressive LMs) that generate outputs in a single pass without environmental interaction. Agentic behavior requires three components:
1. **Policy**: A parameterized function mapping states to actions.
2. **State representation**: A context accumulating observations and actions.
3. **Environment interface**: A mechanism to execute actions (e.g., API calls) and return observations.

### Tool Use as a Sequential Decision Process
Tool use is modeled as a process where an LM:
1. Generates a reasoning trace (e.g., "I need to search for X to answer this question").
2. Invokes a tool (e.g., `search[X]`).
3. Processes the tool’s response to update its state.
4. Repeats until the task is completed or a termination condition is met.

This process is inherently multi-turn, requiring the LM to maintain coherence across steps and adapt to environmental feedback.

### Credit Assignment
Credit assignment in multi-turn tool use refers to the problem of attributing task success or failure to specific actions in a trajectory. For example, if an agent invokes `search[Einstein]` but later fails to answer a question about relativity, it must determine whether the failure stems from:
- The search action (e.g., poor query formulation),
- The reasoning step (e.g., misinterpreting the retrieved text), or
- The final answer generation.

This is exacerbated by:
1. **Sparse rewards**: Task success is often only observable at the end of a trajectory.
2. **Long horizons**: Tool-use tasks may require 10+ steps, diluting the signal from individual actions.
3. **Stochastic environments**: API responses or external tools may fail unpredictably.

---

## Multi-Turn Tool Use: Architectures and Algorithms

### 1. ReAct: Interleaving Reasoning and Acting
**Architecture**: ReAct (Reasoning + Acting) augments the action space of an LM to include both tool invocations and "thoughts" (verbal reasoning traces). The policy alternates between:
- **Thought generation**: Updating the internal context without environmental feedback.
- **Tool invocation**: Querying an external environment (e.g., Wikipedia API) and receiving an observation.

**Key Features**:
- **Few-shot prompting**: The LM is conditioned on human-annotated trajectories demonstrating the interleaved Thought-Action-Observation format.
- **Heuristic switching**: ReAct falls back to Chain-of-Thought (CoT) with self-consistency (CoT-SC) if tool use fails, and CoT-SC delegates to ReAct when internal confidence is low [source:arxiv:2210.03629].

**Performance**:
- On HotpotQA (multi-hop QA), ReAct achieves an exact match (EM) score of 27.4%, while the combined CoT-SC → ReAct method reaches 35.1% EM (vs. 67.5% for supervised state-of-the-art) [source:arxiv:2210.03629].
- On ALFWorld (interactive decision-making), ReAct outperforms Act-only baselines by an absolute 26% success rate (71% vs. 45%) [source:arxiv:2210.03629].

**Limitations**:
- **Prompt sensitivity**: Performance degrades with suboptimal demonstrations or large action spaces.
- **Repetition loops**: Greedy decoding can cause the model to repeat unproductive thoughts or actions.
- **API dependence**: Poor retrieval directly derails reasoning trajectories.

### 2. Toolformer: Self-Supervised Tool Learning
**Architecture**: Toolformer enables LMs to *autonomously* learn tool usage via a four-stage pipeline:
1. **Sampling**: The LM annotates a plain-text corpus with potential API calls (e.g., `<API> QA["What is the capital of France?"] </API>`) at positions where the probability of the `<API>` token exceeds a threshold.
2. **Execution**: Sampled API calls are executed to obtain responses.
3. **Filtering**: Calls are retained only if they reduce the model’s prediction loss for subsequent tokens. The loss reduction is evaluated using a weighted cross-entropy loss:

$$
L_i(\mathbf{z}) = - \sum_{j=i}^n w_{j-i} \cdot \log p_M(x_j \mid \mathbf{z}, x_{1:j-1}),
$$

   where $\mathbf{z}$ is the augmented context (with or without the API call), and weights $w_{j-i}$ decay linearly with distance from the API call [source:arxiv:2302.04761].
4. **Finetuning**: The filtered API calls are interleaved with the original text to form an augmented dataset, on which the base LM is finetuned.

**Key Features**:
- **Self-supervision**: Eliminates the need for human-annotated tool-use demonstrations.
- **Scalability**: Emerges at ~775M parameters (GPT-J) and improves with model size.

**Performance**:
- On factual lookup (LAMA subsets), Toolformer achieves 33.8% (SQuAD), 11.5% (Google-RE), and 53.5% (T-REx), outperforming GPT-3 (175B) while invoking a QA API 98.1% of the time [source:arxiv:2302.04761].
- On mathematical reasoning (ASDiv), it scores 40.4%, surpassing GPT-3 (14.0%) by leveraging a calculator tool 97.9% of the time [source:arxiv:2302.04761].

**Limitations**:
- **Sample inefficiency**: Millions of documents yield only thousands of useful API calls.
- **Input sensitivity**: Performance varies with phrasing (e.g., "What is 2+2?" vs. "Calculate 2+2").
- **Tool chaining**: Not supported; API calls are sampled independently.

### 3. Reflexion: Verbal Reinforcement Learning
**Architecture**: Reflexion implements *verbal reinforcement learning*, where agents improve via linguistic feedback rather than gradient updates. The framework comprises:
1. **Actor**: Generates actions and thoughts.
2. **Evaluator**: Scores trajectories (e.g., +1 for success, 0 otherwise).
3. **Self-Reflection**: Generates verbal feedback (e.g., "The search for Einstein was relevant, but the answer misinterpreted the result").
4. **Episodic memory**: Stores past reflections to guide future trials.

The policy is parameterized by the Actor and memory. At each trial:
1. Generate a trajectory.
2. Compute a reward using the Evaluator.
3. Generate a reflection using the Self-Reflection model.
4. Append the reflection to memory and repeat [source:arxiv:2303.11366].

**Key Features**:
- **Parameter efficiency**: No gradient updates; learning occurs via memory augmentation.
- **Iterative improvement**: Agents refine strategies over multiple trials.

**Performance**:
- On ALFWorld, Reflexion achieves a 97% success rate after 12 trials (vs. 71% for ReAct) [source:arxiv:2303.11366].
- On HumanEval (Python), it reaches 91.0% pass@1, surpassing GPT-4 (80.1%) [source:arxiv:2303.11366].

**Limitations**:
- **Memory constraints**: Episodic memory is bounded (typically 1–3 experiences) to respect context windows.
- **Local minima**: May converge to suboptimal strategies if reflections are uninformative.
- **Task specificity**: Fails to improve on WebShop (e-commerce navigation) after 4 trials.

### 4. Tree of Thoughts (ToT): Search-Augmented Reasoning
**Architecture**: ToT frames problem-solving as a heuristic search over a tree of "thoughts" (coherent language sequences). The framework consists of:
1. **Thought decomposition**: Intermediate steps are defined based on task structure (e.g., "generate 3 candidate answers" for QA).
2. **Thought generation**: From a state, the LM generates $k$ candidate next thoughts using independent sampling or sequential proposal.
3. **State evaluation**: A heuristic evaluator scores states (e.g., 1–10 scale) or votes for the most promising state.
4. **Search algorithm**: BFS (maintains top-$b$ states per step) or DFS (explores most promising path until a threshold is met) [source:arxiv:2305.10601].

**Key Features**:
- **Deliberate exploration**: Enables backtracking and lookahead, unlike autoregressive decoding.
- **Modular evaluation**: Heuristics can be LM-based or rule-based.

**Performance**:
- On Game of 24 (mathematical reasoning), ToT achieves a 74% success rate (vs. 4.0% for CoT) [source:arxiv:2305.10601].
- On 5×5 mini crosswords, it solves 20% of games (vs. 0% for CoT) [source:arxiv:2305.10601].

**Limitations**:
- **Computational cost**: Requires 5–100× more tokens than CoT.
- **Heuristic imperfection**: LM-based evaluators may prune valid solutions.
- **Overhead**: Unjustified for tasks where LMs already perform well.

---

## Search-Augmented RL: Combining Exploration and Learning

Search-augmented RL integrates classical search algorithms (e.g., BFS, DFS, MCTS) with learned policies to improve exploration and credit assignment. The key idea is to use search to *simulate* potential trajectories, then distill the results into the policy.

### 1. ToT as Search-Augmented RL
ToT can be viewed as a search-augmented RL method where:
- The rollout policy is the LM’s CoT generator.
- The value function is the LM-based evaluator.
- Distillation occurs implicitly via in-context learning (few-shot prompting).

### 2. Self-Rewarding Language Models
**Core Idea**: Unify the policy and reward model into a single LM that generates and evaluates its own responses. The framework iterates:
1. **Self-instruction**: Generate novel prompts and candidate responses.
2. **Self-evaluation**: Score responses using LLM-as-a-Judge prompting (e.g., 5-point scale).
3. **Preference optimization**: Train on preference pairs $(y^w, y^l)$ using DPO.

**Key Features**:
- **Closed-loop improvement**: The reward model improves alongside the policy.
- **Scalability**: Eliminates the need for human-labeled preference data.

**Performance**:
- On AlpacaEval 2.0, self-rewarding models achieve a 20.44% win rate vs. GPT-4 Turbo after 3 iterations (vs. 9.94% for SFT) [source:arxiv:2401.10020].

**Limitations**:
- **Reward hacking**: Models may exploit scoring criteria (e.g., generating longer responses).
- **Alignment tax**: Performance on standard benchmarks (e.g., GSM8K) may regress.
- **Safety risks**: Self-improvement may amplify biases or harmful behaviors.

---

## Credit Assignment: Methods and Challenges

### 1. Sparse Rewards and Shaping
**Problem**: In tool-use tasks, rewards are often sparse (e.g., +1 for task success, 0 otherwise), making it difficult to attribute credit to intermediate actions.

**Solutions**:
- **Shaping rewards**: Augment the reward with intermediate signals (e.g., +0.1 for a successful search). Toolformer uses a weighted cross-entropy loss reduction threshold to filter useful API calls, which can be interpreted as a form of shaping [source:arxiv:2302.04761].
- **Verbal feedback**: Use language models to generate step-by-step feedback (e.g., "The search for Einstein was relevant, but the answer misinterpreted the result") [source:arxiv:2303.11366].

**Limitations**:
- Shaping rewards may introduce unintended biases (e.g., favoring search over reasoning).
- Verbal feedback may be noisy or uninformative.

### 2. Counterfactual Credit Assignment
**Problem**: How to estimate the impact of an action on the final outcome?

**Solutions**:
- **Monte Carlo rollouts**: Simulate alternative trajectories where an action is replaced with a counterfactual action and compare outcomes. ToT’s search algorithm implicitly performs this by exploring multiple candidate thoughts [source:arxiv:2305.10601].

**Limitations**:
- Monte Carlo rollouts require a fast rollout policy, which may not exist for complex tasks.

---

## Current Status and Trajectory

### Rising Techniques
1. **Verbal Reinforcement Learning (Reflexion)**:
   - **Status**: Rapidly gaining traction for iterative improvement in reasoning and decision-making tasks.
   - **Evidence**: Reflexion achieves a 97% success rate on ALFWorld after 12 trials and 91% pass@1 on HumanEval (Python), surpassing GPT-4 [source:arxiv:2303.11366]. Self-rewarding models show monotonic improvements on AlpacaEval 2.0 (9.94% → 20.44% win rate over 3 iterations) [source:arxiv:2401.10020].
   - **Trajectory**: Likely to become a default for tasks requiring multi-turn interaction (e.g., coding, embodied agents). Key challenges include memory management and safety.

2. **Search-Augmented Reasoning (ToT)**:
   - **Status**: Emerging as a powerful paradigm for deliberate problem-solving, particularly in mathematical reasoning and planning.
   - **Evidence**: ToT achieves a 74% success rate on Game of 24 (vs. 4% for CoT) and 20% game-level success on 5×5 mini crosswords (vs. 0% for CoT) [source:arxiv:2305.10601].
   - **Trajectory**: Expected to grow in domains where LMs struggle with systematic exploration (e.g., theorem proving, multi-step QA). Computational cost remains a barrier.

3. **Self-Supervised Tool Learning (Toolformer)**:
   - **Status**: Early-stage but promising for scaling tool use without human annotations.
   - **Evidence**: Toolformer outperforms GPT-3 on factual lookup (33.8% vs. GPT-3’s reported scores on SQuAD) and mathematical reasoning (40.4% vs. 14.0% on ASDiv) [source:arxiv:2302.04761].
   - **Trajectory**: Likely to expand as LMs grow larger and tool ecosystems (e.g., APIs, calculators) become more standardized. Key challenges include sample efficiency and tool chaining.

### Default Techniques
1. **ReAct (Reasoning + Acting)**:
   - **Status**: Currently the most widely adopted paradigm for multi-turn tool use, due to its simplicity and few-shot applicability.
   - **Evidence**: ReAct achieves a 71% success rate on ALFWorld (vs. 45% for Act-only) and 40% on WebShop (vs. 29.1% for IL) [source:arxiv:2210.03629].
   - **Trajectory**: Likely to remain a default for interactive tasks, but may be supplanted by more scalable methods (e.g., Toolformer) or iterative methods (e.g., Reflexion).

2. **LLM-as-a-Judge**:
   - **Status**: Increasingly used for reward modeling and evaluation, particularly in self-improving systems.
   - **Evidence**: Self-rewarding models improve pairwise accuracy against human rankings from 65.1% to 81.7% over 3 iterations [source:arxiv:2401.10020].
   - **Trajectory**: Expected to become a standard component of RLHF pipelines, though concerns about bias and reliability persist.

### Fading Techniques
1. **Static Reward Models**:
   - **Status**: Declining as a standalone approach due to scalability limitations.
   - **Evidence**: Self-rewarding models outperform static reward models on AlpacaEval 2.0 (20.44% vs. 9.94% win rate) [source:arxiv:2401.10020].
   - **Trajectory**: Likely to be replaced by dynamic, self-improving reward models (e.g., generative verifiers, LLM-as-a-Judge).

### Disagreements and Open Questions
1. **Search vs. Memory**:
   - **A (ToT)**: Argues that search-augmented reasoning (e.g., BFS/DFS) is necessary for deliberate problem-solving [source:arxiv:2305.10601].
   - **B (Reflexion)**: Counters that episodic memory (e.g., self-reflection) is sufficient for iterative improvement, without explicit search [source:arxiv:2303.11366].
   - **Z (Settling)**: A hybrid approach (e.g., search-augmented memory) could resolve this, but no source currently reports such a method.

2. **Self-Improvement Limits**:
   - **A (Self-Rewarding Models)**: Claims that iterative self-improvement can surpass human performance ceilings [source:arxiv:2401.10020].
   - **B (Alignment Tax)**: Notes that self-improvement may plateau due to reward hacking or alignment tax, though this is not explicitly stated in the provided sources.
   - **Z (Settling)**: Longitudinal studies with >3 iterations are needed to determine scaling laws.

3. **Tool Chaining**:
   - **A (Toolformer)**: States that tool chaining (using one tool’s output as another’s input) is not supported due to independent sampling [source:arxiv:2302.04761].
   - **B (ReAct)**: Implicitly supports tool chaining via interleaved reasoning and acting [source:arxiv:2210.03629].
   - **Z (Settling)**: A unified framework for explicit tool chaining remains an open challenge.

---

## Key Takeaways
- **Agentic RL** enables LMs to dynamically interact with environments by formalizing tool use as a sequential decision process. Core challenges include multi-turn interaction, exploration in high-dimensional discrete spaces, and scalable reward modeling.
- **Multi-turn tool use** is dominated by architectures like ReAct (interleaving reasoning and acting), Toolformer (self-supervised tool learning), Reflexion (verbal reinforcement learning), and ToT (search-augmented reasoning). Each has trade-offs in scalability, sample efficiency, and task generality.
- **Credit assignment** remains an open problem for long-horizon tasks, with solutions including shaping rewards, verbal feedback, and counterfactual methods.
- **Search-augmented RL** (e.g., ToT) improves exploration but incurs high computational costs. Verbal reinforcement (e.g., Reflexion) offers a lightweight alternative for iterative improvement.
- **Self-improving systems** (e.g., self-rewarding models) show promise for surpassing human performance ceilings, but risks include reward hacking and alignment tax.
- **Current trajectory**: Verbal reinforcement and search-augmented reasoning are rising, while static reward models are fading. Disagreements persist around the necessity of search vs. memory and the limits of self-improvement.

---

## Related Topics
- [[RL for reasoning models]]
- [[Policy gradient methods for LLMs]]
- [[Reward modeling for LLMs]]
- [[KL regularization in RLHF]]
- [[MDP formulation of LLM generation]]
- [[RL for LLMs — overview]]
- [[Process vs outcome reward models]]
- [[Reward hacking in RLHF]]
- [[Reward model over-optimization]]
- [[Entropy and exploration in RL fine-tuning]]
- [[Over-optimization and mode collapse]]
- [[LLM-as-judge]]
- [[Alignment and win-rate evals]]
- [[Test-time compute and RL interplay]]

---

##

## References
- [source:arxiv:1905.09836] [Learning to Use Tools via Reinforcement Learning](https://arxiv.org/abs/1905.09836)
- [source:arxiv:2210.03629] [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [source:arxiv:2302.04761] [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [source:arxiv:2303.11366] [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)
- [source:arxiv:2305.10601] [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [source:arxiv:2406.11903] [Generative Verifiers for Reinforcement Learning](https://arxiv.org/abs/2406.11903)
- [source:arxiv:2401.10020] [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
- [source:arxiv:2406.13479] [Reasoning with Reinforcement Learning](https://arxiv.org/abs/2406.13479)
