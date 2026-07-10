---
id: arxiv:2502.10867
type: paper
title: 'A Tutorial on LLM Reasoning: Relevant Methods behind ChatGPT o1'
url: https://arxiv.org/abs/2502.10867
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

**Core Problem**
Autoregressive large language models (LLMs) traditionally optimize for next-token prediction, a paradigm that inherently caps intelligence at the quality of training demonstrations—a constraint termed the "intelligence upper bound." While prompting techniques like Chain-of-Thought (CoT) can elicit intermediate reasoning steps, they do not natively embed deliberative reasoning within the model architecture. The field requires a fundamental shift from fast, direct autoregressive generation to slow, inference-time computation that simulates human System 2 thinking. This transition demands mechanisms capable of exploring multiple reasoning trajectories, dynamically updating problem representations, and leveraging reinforcement learning rather than solely minimizing prediction errors.

**Methodological Recipe**
The tutorial formalizes native reasoning as a Markov Decision Process (MDP) where the state $s_t$ comprises the initial question $Q$ and previously generated reasoning steps $\{R_1, \dots, R_{t-1}\}$, actions $a_t$ correspond to selecting the next reasoning step or final answer $A$, and the policy $\pi_{LLM}$ governs token generation. The implementation follows a structured pipeline:
1. **Data Acquisition:** Employ the Self-Taught Reasoner (STaR) framework to autonomously generate intermediate steps $\{R\}$ conditioned on $Q$ and $A$, then verify correctness by checking if the model predicts $A$ from $Q$ and $\{R\}$. Monte Carlo Tree Search (MCTS) can extend this for longer sequences.
2. **Process-Reward Model (PRM) Training:** Train a verifier to evaluate reasoning quality. Instead of independent step classification, the PRM is optimized via value iteration using the Bellman equation to predict cumulative rewards across reasoning trajectories.
3. **Policy Optimization:** Fine-tune the LLM using Group Relative Policy Optimization (GRPO). The model samples a group of reasoning trajectories, normalizes rewards across the group, and computes advantages based on cumulative future rewards.
4. **Inference-Time Computation:** Deploy non-autoregressive decoding strategies like beam search or MCTS guided by the PRM to explore branching reasoning paths.
5. **Native CoT Formation:** The integrated system autonomously performs structured, step-by-step reasoning without external prompts, formalized as Native Chain-of-Thought (NCoT).

**Key Formulas**
The autoregressive foundation remains $P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$. State transitions are deterministic: $s_{t+1} = s_t + a_t$. The PRM value function is updated via the Bellman equation:
$$V_{\theta}(s) = r(s) + \gamma \max_{a} V_{\theta}(a + s),$$
with the temporal difference loss:
$$L(\theta) = \sum_{i=1}^{N} \left( V_{\theta}(s_i) - \left[ r(s_i) + \gamma \max_{a} V_{\theta}(s_i + a) \right] \right)^2.$$
GRPO optimization maximizes:
$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{K_i} \sum_{t=1}^{K_i} \min(\hat{\rho}_{i,t} A_{i,t}, \text{clip}(\hat{\rho}_{i,t}, 1 - \epsilon, 1 + \epsilon) A_{i,t}) - \beta D_{\text{KL}}(\pi_{\theta} \| \pi_{\theta_{old}}) \right],$$
where the advantage $A_{i,t} = \sum_{j=t}^{K_i} \bar{r}_i^{(j)}$ and normalized rewards are $\bar{r}_i^{(t)} = \frac{r_i^{(t)} - \text{mean}(R)}{\text{std}(R)}$.

**Quantitative Results**
OpenAI’s o1 demonstrates substantial performance gains over ChatGPT 4o, achieving five times greater proficiency in mathematics and coding. The model ranks in the 89th percentile for competitive programming, places among the top 500 students in a U.S. math olympiad qualifier, and surpasses human PhD-level accuracy in physics, biology, and chemistry benchmarks. In coding competitions, o1 scored the 49th percentile in the 2024 International Olympiad in Informatics and outperformed 93% of human competitors in simulated Codeforces contests.

**Stated Limitations**
The source identifies several constraints. Autoregressive next-token prediction inherently caps model capabilities at the skill level of training demonstrations. Standard CoT prompting acts as a limited, write-only memory that cannot overwrite or delete prior outputs, failing to natively integrate into the decoding stage. Transformer architectures suffer from quadratic computational complexity, complicating multi-step mathematical challenges. Additionally, the PRM classification approach treats reasoning steps as independent, ignoring the future impact of intermediate actions. Finally, while self-critique and reasoning emergence show promise, they remain limited and typically require sufficiently large models, and the closed-source nature of o1 obscures its exact implementation details.
