---
id: arxiv:2305.10601
type: paper
title: 'Tree of Thoughts: Deliberate Problem Solving with Large Language Models'
url: https://arxiv.org/abs/2305.10601
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

**Core Problem**
Standard autoregressive language models (LMs) operate via token-level, left-to-right decoding, functioning analogously to a fast, automatic cognitive mode. This paradigm lacks mechanisms for deliberate exploration, strategic lookahead, and backtracking, causing models to fail on tasks requiring multi-step planning, search, or where early decisions are pivotal. Existing prompting methods like Chain-of-Thought (CoT) sample continuous sequences without local branching or global evaluation, limiting their problem-solving capacity.

**Method/Recipe Step by Step**
The Tree of Thoughts (ToT) framework addresses these limitations by framing problem-solving as a heuristic search over a combinatorial tree of coherent language sequences, termed "thoughts." The recipe proceeds through four modular steps: (1) **Thought Decomposition:** Intermediate steps are explicitly defined based on task structure, balancing granularity so thoughts are neither too small to evaluate nor too large to generate coherently. (2) **Thought Generation:** From a given state, the LM generates $k$ candidate next thoughts using either independent i.i.d. sampling from a CoT prompt for rich thought spaces, or a sequential proposal prompt to prevent duplication in constrained spaces. (3) **State Evaluation:** A heuristic evaluator assesses candidate states to guide search. This is implemented either by independently scoring each state (e.g., assigning a 1–10 value or classifying it as sure/maybe/impossible via lookahead and commonsense) or by voting across states to select the most promising partial solution. (4) **Search Algorithm:** The framework integrates classical search algorithms, specifically Breadth-First Search (BFS) to maintain a fixed breadth of top states per step, or Depth-First Search (DFS) to explore the most promising path until a threshold is met, pruning unpromising subtrees, and backtracking to parent nodes when necessary.

**Key Formulas**
The mathematical formalization defines a state as $s = [x, z_{1\dots i}]$, representing the input $x$ and the sequence of thoughts generated so far. The thought generator $G(p_\theta, s, k)$ operates via two strategies: independent sampling $z^{(j)} \sim p_\theta^{CoT}(z_{i+1}|s) = p_\theta^{CoT}(z_{i+1}|x, z_{1\dots i})$ for $j=1\cdots k$, or sequential proposal $[z^{(1)}, \cdots, z^{(k)}] \sim p_\theta^{propose}(z_{i+1}^{(1\dots k)} \mid s)$. The state evaluator $V(p_\theta, S)$ either assigns an independent value $V(p_\theta, S)(s) \sim p_\theta^{value}(v|s)$ or performs voting via an indicator function $V(p_\theta, S)(s) = \mathbb{1}[s = s^*]$, where the winning state $s^*$ is sampled from $p_\theta^{vote}(s^*|S)$.

**Quantitative Results**
Evaluated on GPT-4 across three novel tasks, ToT significantly outperforms standard prompting baselines. In the Game of 24 (a mathematical reasoning task), ToT achieved a 74% success rate with a breadth of $b=5$, compared to 4.0% for CoT and 7.3% for Input-Output (IO) prompting. For creative writing, ToT attained a GPT-4 coherency score of 7.56, surpassing CoT (6.93) and IO (6.19), with human evaluators preferring ToT outputs in 41 out of 100 paired comparisons. In $5 \times 5$ mini crosswords, ToT reached a 78% letter-level success rate and solved 4 out of 20 games (20% game-level success), dramatically exceeding IO (38.7% letter, 0% game) and CoT (40.6% letter, 1% game). Ablation studies confirmed that both pruning and backtracking are critical for crossword performance.

**Stated Limitations**
The authors note that ToT demands substantially higher computational resources, requiring roughly 5–100 times more generated tokens and API costs than IO or CoT methods. The framework is most beneficial for tasks where LMs struggle with deliberate reasoning; for already-solved tasks, the overhead is unjustified. Additionally, the LM-based evaluation heuristics can be imperfect, occasionally pruning valid solutions due to unfamiliar vocabulary or knowledge gaps. Finally, the current implementation relies on off-the-shelf LMs without task-specific fine-tuning, leaving room for future research into training models for high-level counterfactual decision-making.
