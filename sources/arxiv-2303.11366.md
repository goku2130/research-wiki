---
id: arxiv:2303.11366
type: paper
title: 'Reflexion: Language Agents with Verbal Reinforcement Learning'
url: https://arxiv.org/abs/2303.11366
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

**Core Problem**
Large language models (LLMs) increasingly function as autonomous agents interacting with external environments, yet they struggle to learn efficiently from trial-and-error. Traditional reinforcement learning methods demand extensive training samples and computationally expensive model fine-tuning, which are impractical for massive parameterized models. Consequently, there is a need for a lightweight, parameter-efficient paradigm that enables agents to iteratively improve their decision-making without gradient-based weight updates.

**Method and Recipe**
Reflexion addresses this by implementing verbal reinforcement learning, where agents learn through linguistic feedback rather than weight updates. The framework operates iteratively using three modular components: an Actor ($M_a$) that generates text and actions, an Evaluator ($M_e$) that scores outputs, and a Self-Reflection model ($M_{sr}$) that generates verbal feedback. The process follows a strict sequence:
1. Initialize the Actor, Evaluator, and Self-Reflection models, along with an episodic memory buffer ($mem$).
2. Generate an initial trajectory $\tau_0$ using the policy $\pi_{\theta}(a_{i}|s_{i})$, where the parameter set $\theta$ comprises the Actor and the memory buffer.
3. Evaluate $\tau_0$ via $M_e$ to compute a reward score $r_0$.
4. Feed the trajectory, reward, and current memory into $M_{sr}$ to generate a verbal self-reflection summary $sr_0$.
5. Store $sr_0$ in $mem$. The trajectory history serves as short-term memory, while $mem$ acts as long-term episodic storage.
6. Iterate: Generate $\tau_t$, evaluate to obtain $r_t$, generate $sr_t$, and append $sr_t$ to $mem$. Continue until the Evaluator passes the trajectory or a maximum trial limit is reached. Memory is bounded by a capacity $\Omega$ (typically 1–3 experiences) to respect context window constraints.

**Key Formulas**
The policy is parameterized as $\pi_{\theta}(a_{i}|s_{i}), \theta = \{M_{a}, mem\}$. The reward signal at trial $t$ is computed deterministically via the evaluator: $r_t = M_e(\tau_t)$. The self-reflection model processes the trial history and reward to produce verbal feedback $sr_t$, which is iteratively appended to the memory buffer.

**Quantitative Results**
Reflexion demonstrates significant performance gains across sequential decision-making, reasoning, and programming tasks. In the AlfWorld decision-making suite, the agent achieved a 22% absolute improvement over the ReAct baseline, successfully completing 130 out of 134 tasks after 12 iterative learning steps. For the HotPotQA reasoning benchmark, Reflexion yielded a 20% overall improvement; specifically, when provided with ground-truth context, the baseline Chain-of-Thought agent failed on 39% of questions, but Reflexion corrected these errors to improve accuracy by 14%. An ablation study confirmed that self-reflection provided an 8% absolute boost over simple episodic memory retrieval. In code generation, Reflexion surpassed previous state-of-the-art results, achieving 91.0% pass@1 accuracy on HumanEval (Python), outperforming GPT-4’s 80.1%. It also scored 68.0% on HumanEval (Rust) versus GPT-4’s 60.0%, and 15.0% on the newly introduced LeetcodeHardGym benchmark versus GPT-4’s 7.5%. On MBPP (Python), it achieved 77.1%, while on MBPP (Rust) it reached 75.4%. Ablation tests on HumanEval Rust confirmed that both test generation and self-reflection are necessary, as omitting either component reduced pass@1 accuracy to 52% or 60%, respectively.

**Limitations**
The authors acknowledge several constraints. Reflexion may converge to non-optimal local minima and lacks formal guarantees for success, relying instead on the LLM’s self-evaluation capabilities. The episodic memory is restricted to a sliding window of fixed capacity, preventing unbounded experience accumulation. In programming, the approach struggles with non-deterministic functions, impure API interactions, hardware-dependent outputs, and concurrent behaviors. Furthermore, on the WebShop e-commerce navigation benchmark, Reflexion failed to improve after four trials, indicating difficulty with tasks requiring highly diverse exploration and ambiguous natural language interpretation.
