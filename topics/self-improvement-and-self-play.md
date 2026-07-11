---
title: Self-improvement and self-play RL
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2203.14465
- arxiv:2308.08998
- arxiv:2505.03335
- arxiv:1809.02923
- arxiv:2409.12917
- arxiv:2401.10020
- arxiv:2406.01495
open_questions:
- 'Scaling Laws for Self-Improvement**:'
- How does performance scale with the number of self-improvement iterations for methods
  like Self-Rewarding Models and Absolute Zero? Are there diminishing returns or emergent
  capabilities?
- What are the compute-optimal trade-offs between model size, iteration depth, and
  data generation volume?
- 'Safety and Alignment in Autonomous Systems**:'
---

# Self-Improvement and Self-Play in Reinforcement Learning for LLMs

Self-improvement and self-play in reinforcement learning (RL) for large language models (LLMs) enable autonomous, iterative refinement by leveraging the model’s own outputs as training signals. These methods address key limitations of static, human-supervised alignment—such as data scarcity, reward hacking, and scalability—through algorithmic innovations like bootstrapping, growing-batch RL, and self-generated task synthesis.

---

## Core Problems and Motivations

The primary challenges these methods address include:

1. **Data Scarcity and Cost**: Manual annotation of rationales or preferences is prohibitively expensive for complex tasks, particularly in reasoning domains [source:arxiv:2203.14465]. Methods like STaR and ReST mitigate this by bootstrapping synthetic training data from the model’s own generations.
2. **Reward Hacking and Misalignment**: Online RLHF methods are vulnerable to reward model exploitation, where policies optimize for proxy rewards rather than true objectives [source:arxiv:2308.08998]. ReST and Absolute Zero decouple data generation from optimization to reduce overfitting.
3. **Autonomy and Scalability**: Dependence on human-curated datasets or verifiers limits long-term scalability [source:arxiv:2505.03335]. Absolute Zero eliminates this bottleneck by enabling models to generate, solve, and learn from entirely synthetic tasks.
4. **Intrinsic Self-Correction**: LLMs fail to revise their own errors without external feedback, suffering from distribution shift and behavior collapse [source:arxiv:2409.12917]. SCoRe and Re-ReST introduce multi-turn RL frameworks to explicitly incentivize corrective progress.

---

## Methodological Deep Dive

### 1. Self-Taught Reasoner (STaR)
**Problem**: Inducing step-by-step reasoning in LLMs without manual rationale annotation or template-based methods.

**Mechanism**:
STaR iteratively refines a model’s reasoning by alternating between *forward rationale generation* and *backward rationalization* [source:arxiv:2203.14465]. Given a dataset $\mathcal{D} = \{(x_i, y_i)\}$ and a few-shot prompt set $\mathcal{P}$ with initial rationales:
1. **Generation**: The model $M$ generates a rationale $\hat{r}_i$ and predicted answer $\hat{y}_i$ for each $x_i$ using $\mathcal{P}$.
2. **Rationalization**: For incorrect predictions ($\hat{y}_i \neq y_i$), the model is prompted with $x_i$, the ground-truth answer $y_i$, and $\mathcal{P}$ to generate a corrected rationale $\hat{r}_i^{\text{rat}}$.
3. **Filtering**: Only rationales yielding correct answers are retained.
4. **Fine-tuning**: The model is fine-tuned on the filtered dataset, and the loop repeats.

**Mathematical Formulation**:
STaR approximates a policy gradient objective with a discrete latent variable model:

$$
J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot|x_i)} \mathbb{1}(\hat{y}_i = y_i),
$$

where the gradient is computed via the log-derivative trick:

$$
\nabla J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot|x_i)} \left[ \mathbb{1}(\hat{y}_i = y_i) \cdot \nabla \log p_M(\hat{y}_i, \hat{r}_i | x_i) \right].
$$

The indicator function $\mathbb{1}(\hat{y}_i = y_i)$ acts as a reward filter, discarding gradients for incorrect rationales.

**Key Results**:
- On GSM8K, STaR improves accuracy from 5.8% (direct fine-tuning) to 10.7% (with rationalization) [source:arxiv:2203.14465].
- On CommonsenseQA, STaR achieves 72.5% accuracy (with rationalization) vs. 68.8% (without rationalization) [source:arxiv:2203.14465].
- Human evaluations show STaR-generated rationales are preferred over few-shot rationales (30% more favorable, $p=.039$) [source:arxiv:2203.14465].

**Limitations**:
- Requires non-trivial few-shot reasoning capability; smaller models (e.g., GPT-2) fail to bootstrap.
- Tasks with high chance-performance (e.g., binary decisions) generate excessive poor rationales, confounding filtering.
- Rationalization relies on a hint format that may not generalize across domains.

---

### 2. Reinforced Self-Training (ReST)
**Problem**: Computational inefficiency and reward hacking in online RLHF, coupled with dataset-quality limitations in offline RL.

**Mechanism**:
ReST decouples data generation (*Grow*) from policy optimization (*Improve*) in a growing-batch RL framework [source:arxiv:2308.08998]:
1. **Grow**: Sample multiple outputs $y$ from the current policy $\pi_\theta$ for each context $x \in \mathcal{D}$, forming an augmented dataset $\mathcal{D}_g$ scored by a reward model $R(x, y)$.
2. **Improve**: Filter $\mathcal{D}_g$ using a reward threshold $\tau$ and fine-tune $\pi_\theta$ on the retained samples. Repeat with increasing thresholds $\tau_1 < \tau_2 < \dots$.

**Mathematical Formulation**:
The policy $\pi_\theta(y \mid x) = \prod_{t=1}^T \pi_\theta(y_t \mid y_{1:t-1}, x)$ is optimized via:

$$
J(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}_g} \left[ F(x, y; \tau) \mathcal{L}(x, y; \theta) \right],
$$

where $F(x, y; \tau) = \mathbb{1}_{R(x, y) > \tau}$ and $\mathcal{L}$ is typically the NLL loss.

**Key Results**:
- On IWSLT 2014 De-En, ReST (G=2, I=3) achieves an average reward of 83.1, outperforming online PPO (71.6) and supervised fine-tuning (70.9) [source:arxiv:2308.08998].
- Human evaluations confirm ReST’s superiority over supervised baselines, though reward model rankings misalign with human preferences.

**Limitations**:
- Reward models are imperfect proxies for human preferences, leading to misalignment as policies diverge.
- Repeated Grow steps risk overfitting to the reward model.
- Simple sampling in the Grow step limits exploration; Monte Carlo Tree Search is suggested as a mitigation [source:arxiv:2308.08998].

---

### 3. Absolute Zero
**Problem**: Scalability of zero-shot reasoning paradigms that still depend on manually curated question-answer pairs.

**Mechanism**:
Absolute Zero eliminates external data by enabling a model to autonomously generate, solve, and learn from synthetic tasks [source:arxiv:2505.03335]. The **Absolute Zero Reasoner (AZR)** operates in a self-play loop with two roles:
1. **Proposer**: Generates code-based reasoning tasks (deduction, abduction, induction) conditioned on past examples.
2. **Solver**: Attempts to solve proposed tasks, with learning guided by:
   - A *learnability reward* for the proposer (incentivizing moderate-difficulty tasks).
   - A *correctness reward* for the solver (binary success/failure).

**Mathematical Formulation**:
The composite objective generalizes RLVR to self-generated tasks:

$$
\mathcal{J}(\theta) = \max_\theta \mathbb{E}_{z \sim p(z)} \left[ \mathbb{E}_{(x, y^*) \sim f_e(\cdot | \tau), \tau \sim \pi_\theta^{\text{propose}}(\cdot | z)} \left[ \lambda r_e^{\text{propose}}(\tau, \pi_\theta) + \mathbb{E}_{y \sim \pi_\theta^{\text{solve}}(\cdot | x)} \left[ r_e^{\text{solve}}(y, y^*) \right] \right] \right],
$$

where the proposer reward is $r_{\text{propose}} = 1 - \bar{r}_{\text{solve}}$ if $\bar{r}_{\text{solve}} > 0$, else $0$.

**Key Results**:
- AZR-Coder-7B surpasses models trained on tens of thousands of human-curated examples by an average of 1.8 points in combined reasoning scores [source:arxiv:2505.03335].
- Math accuracy improves by 15.2 points for AZR-Coder-7B vs. 0.65 points for expert-trained code models.
- Ablations show all three reasoning modes (deduction, abduction, induction) are essential for performance.

**Limitations**:
- Tasks are restricted to deterministic programs to ensure verifiable rewards.
- Safety concerns arise from occasional "uh-oh moments" (concerning chains of thought).
- Scaling laws for deeper iterations or smaller models remain unexplored.

---

### 4. Iterated Amplification via Comparison-Based Algorithm (CBA)
**Problem**: Unbiased gradient estimation in settings with only comparative feedback (e.g., censored demand, preference surveys).

**Mechanism**:
Iterated amplification (as formalized in [source:arxiv:1809.02923]) addresses one-dimensional stochastic convex optimization where only binary comparisons are observable. The **Comparison-Based Algorithm (CBA)**:
1. Samples $\xi_t$ from the underlying distribution and compares it to the current decision point $x_t$.
2. Uses auxiliary sampling densities $f_-(x, z)$ and $f_+(x, z)$ to construct an unbiased gradient estimate:

$$
\hat{g}_t = \frac{\mathbb{1}_{\xi_t > x_t}}{f_+(\xi_t, x_t)} - \frac{\mathbb{1}_{\xi_t < x_t}}{f_-(\xi_t, x_t)}.
$$

3. Updates $x_{t+1} = \text{Proj}_{[\ell, u]}(x_t - \eta_t \hat{g}_t)$.

**Mathematical Formulation**:
The gradient estimate is unbiased:

$$
\mathbb{E}[\hat{g}_t \mid x_t] = \nabla H(x_t),
$$

where $H(x) = \mathbb{E}_{\xi}[h(x, \xi)]$.

**Key Results**:
- CBA achieves convergence rates comparable to full-information methods in simulated inventory optimization [source:arxiv:1809.02923].

**Limitations**:
- Restricted to one-dimensional convex optimization; extensions to high-dimensional or non-convex settings are non-trivial.
- Requires careful design of auxiliary densities $f_\pm$ to ensure integrability and variance control.

---

### 5. Self-Correction via Reinforcement Learning (SCoRe)
**Problem**: Intrinsic self-correction failure in LLMs due to distribution shift and behavior collapse.

**Mechanism**:
SCoRe trains a model to revise its own errors via a two-stage RL framework [source:arxiv:2409.12917]:
1. **Stage I**: Optimize second-attempt accuracy while constraining the first-attempt distribution to match the base model via KL-divergence.
2. **Stage II**: Jointly optimize both attempts with a *progress bonus* $\hat{b}(y_2|y_1, y^*) = \alpha \cdot (\hat{r}(y_2, y^*) - \hat{r}(y_1, y^*))$ to incentivize corrective transitions.

**Mathematical Formulation**:
Stage I objective:

$$
\max_\theta \mathbb{E} \left[ \hat{r}(y_2, y^*) - \beta_2 D_{\text{KL}}(\pi_\theta(\cdot | x_1) || \pi_{\text{ref}}(\cdot | x_1)) \right].
$$

Stage II objective:

$$
\max_\theta \mathbb{E} \left[ \sum_{i=1}^2 \hat{r}(y_i, y^*) - \beta_1 D_{\text{KL}}(\pi_\theta(\cdot | x_i) || \pi_{\text{ref}}(\cdot | x_i)) \right].
$$

**Key Results**:
- On MATH, SCoRe improves accuracy from 52.6% (turn 1) to 64.4% (turn 2), with a 15.6% absolute self-correction gain [source:arxiv:2409.12917].
- On HumanEval, SCoRe yields a 9.1% absolute self-correction gain, improving accuracy from 53.7% to 64.6% across two turns.

**Limitations**:
- Trained for only one correction round (two attempts total); performance on deeper iterations is unknown.
- Two-stage training requires separate runs, complicating deployment.

---

### 6. Self-Rewarding Language Models
**Problem**: Static reward models in RLHF/DPO cap the quality of training signals at human performance.

**Mechanism**:
Self-Rewarding Language Models iteratively self-align by generating and evaluating their own responses via LLM-as-a-Judge prompting [source:arxiv:2401.10020]:
1. **Initialization**: Start with an SFT model $M_0$ on seed IFT and EFT data.
2. **Self-Instruction Creation**: $M_t$ generates prompts $x_i$, samples $N$ responses $\{y_i^n\}$, and evaluates them using a structured LLM-as-a-Judge prompt (scoring 0–5 across five criteria).
3. **Preference Pair Construction**: Highest- and lowest-scoring responses are paired as $(x_i, y_i^w, y_i^l)$.
4. **Iterative DPO Training**: Train $M_{t+1}$ on the self-generated preference data.

**Mathematical Formulation**:
The iterative training dynamics are:

$$
M_{t+1} = \text{DPO}(M_t, \text{AIFT}(M_t)),
$$

where $\text{AIFT}(M_t)$ is the self-generated preference dataset.

**Key Results**:
- On AlpacaEval 2.0, $M_3$ achieves a 20.44% win rate vs. GPT-4 Turbo, surpassing Claude 2 (17.19%) [source:arxiv:2401.10020].
- Reward modeling accuracy improves from 65.1% (SFT) to 81.7% ($M_3$).

**Limitations**:
- Length bias correlates with quality improvements; further analysis is needed to isolate genuine gains.
- Safety evaluations are absent, though the authors suggest the loop could enhance safety alignment.

---

### 7. Re-ReST: Reflection-Reinforced Self-Training
**Problem**: Self-training discards low-quality generations, limiting sample efficiency for complex tasks.

**Mechanism**:
Re-ReST introduces a *reflector model* to actively refine failed trajectories using environmental feedback [source:arxiv:2406.01495]:
1. **Initial Generation**: The agent model $\mathcal{M}$ samples $k$ trajectories for each input $x$; an environment $E$ evaluates them.
2. **Reflection**: For failed samples, a reflector model $\mathcal{R}$ generates corrected trajectories $\tilde{y}^j$ using $x$, the failed generation $\hat{y}^j$, and feedback $\mathcal{E}(x, \hat{y}^j)$.
3. **Training**: $\mathcal{M}$ is fine-tuned on the combined dataset of successful and reflected trajectories.

**Mathematical Formulation**:
The reflector is trained via:

$$
\mathcal{L}_{MLE}(\theta_{\mathcal{R}}) = - \mathbb{E}_{(x, y^l, y^w) \sim \mathcal{D}_{\mathcal{M}}^{\mathcal{R}} \cup \mathcal{D}_{\mathcal{R}}^{\mathcal{R}}} \log p_{\theta_{\mathcal{R}}}(y^w \mid x, y^l).
$$

**Key Results**:
- On AlfWorld, Re-ReST improves success rates from 37.3% (self-training) to 51.4% [source:arxiv:2406.01495].
- On MBPP, Re-ReST achieves a Pass@1 score of 56.4% vs. 54.5% for self-training.

**Limitations**:
- Requires ground-truth environmental feedback, limiting applicability to general language modeling.
- Risk of amplifying base model biases.

---

## Current Status and Trajectory

### Rising Techniques
1. **Self-Rewarding Models**: The self-rewarding paradigm is rapidly gaining traction, with [source:arxiv:2401.10020] demonstrating scalable self-improvement over three iterations. The method’s ability to simultaneously enhance instruction-following and reward-modeling capabilities positions it as a leading candidate for autonomous alignment. However, its long-term scaling behavior and safety implications remain understudied.
2. **Absolute Zero**: The zero-data paradigm represents a radical departure from traditional alignment, with [source:arxiv:2505.03335] showing state-of-the-art performance on reasoning benchmarks. Its trajectory hinges on addressing safety concerns and extending beyond deterministic tasks.
3. **Multi-Turn Self-Correction**: SCoRe [source:arxiv:2409.12917] and Re-ReST [source:arxiv:2406.01495] demonstrate that multi-turn training is essential for intrinsic self-correction. While SCoRe’s two-stage training is complex, its empirical gains suggest multi-turn RL will become a default component of alignment pipelines.

### Default Techniques
1. **ReST**: Reinforced self-training has become a default offline RL method for language modeling, with [source:arxiv:2308.08998] demonstrating its superiority over online PPO in machine translation. Its growing-batch architecture is widely adopted for its computational efficiency.
2. **STaR**: While STaR’s rationale bootstrapping was groundbreaking, its adoption has plateaued due to its dependence on few-shot prompting and filtering heuristics. It remains a default for reasoning tasks where rationale quality is critical.

### Fading Techniques
1. **Iterated Amplification (CBA)**: The comparison-based algorithm [source:arxiv:1809.02923] laid the groundwork for preference-based RL but has seen limited direct adoption in LLM alignment. Its one-dimensional convex optimization setting is too restrictive for modern applications. The core ideas persist in reward modeling and preference optimization, but the original CBA is fading as a standalone method.

### Disagreements and Open Challenges
1. **Reward Model Fidelity vs. Autonomy**:
   - ReST [source:arxiv:2308.08998] and Self-Rewarding Models [source:arxiv:2401.10020] differ fundamentally in their reliance on reward models. ReST uses a static reward model, while self-rewarding models eliminate this dependency entirely. The trade-off between reward model fidelity (ReST) and autonomy (self-rewarding) is unresolved.
   - **Settling the Disagreement**: A head-to-head comparison of the two approaches on a shared benchmark (e.g., GSM8K or MATH) with identical model architectures and compute budgets would clarify their relative merits.

2. **Exploration in Self-Play**:
   - Absolute Zero [source:arxiv:2505.03335] and ReST [source:arxiv:2308.08998] both note that simple sampling limits exploration. Absolute Zero proposes task diversity incentives, while ReST suggests Monte Carlo Tree Search. However, neither has empirically validated these solutions.
   - **Settling the Disagreement**: Ablation studies comparing task diversity incentives, MCTS, and other exploration strategies (e.g., entropy regularization) in a controlled self-play setting would determine the most effective approach.

3. **Safety in Autonomous Systems**:
   - Absolute Zero [source:arxiv:2505.03335] reports "uh-oh moments" (concerning chains of thought), while Self-Rewarding Models [source:arxiv:2401.10020] lack safety evaluations entirely. The field has not established whether autonomous self-improvement inherently amplifies or mitigates safety risks.
   - **Settling the Disagreement**: Longitudinal safety evaluations of self-improving models (e.g., tracking sycophancy, misgeneralization, or reward hacking over iterations) are needed to quantify risks.

---

## Key Takeaways
- **Autonomy vs. Control**: Methods like Absolute Zero and self-rewarding models prioritize autonomy, while ReST and STaR retain human-designed components (reward models, few-shot prompts). The optimal balance depends on the domain and safety requirements.
- **Multi-Turn RL**: SCoRe and Re-ReST demonstrate that multi-turn training is essential for intrinsic self-correction, but deeper iterations (>2 turns) remain unexplored.
- **Reward Hacking Mitigation**: Decoupling data generation from optimization (ReST) or eliminating reward models entirely (self-rewarding) reduces overfitting, but at the cost of training signal fidelity.
- **Scaling Laws**: Absolute Zero and self-rewarding models show performance scales with model size, but the limits of iterative self-improvement are unknown.
- **Safety Gaps**: All methods lack rigorous safety evaluations. "Uh-oh moments" in Absolute Zero and length bias in self-rewarding models highlight unresolved risks.

---

## Related Topics
- [Reward modeling for LLMs](reward-modeling.md): Foundational for ReST and STaR, but challenged by self-rewarding models.
- [RL for reasoning models](rl-for-reasoning.md): STaR and Absolute Zero are specialized instances of this broader problem.
- [KL regularization in RLHF](kl-regularization.md): Critical for SCoRe’s two-stage training and ReST’s stability.
- [Reward hacking in RLHF](reward-hacking.md): ReST and self-rewarding models address this, but with different trade-offs.
- [LLM-as-judge](llm-as-judge.md): Central to self-rewarding models and Re-ReST’s reflection mechanism.
- [Alignment and win-rate evals](alignment-and-winrate-evals.md): Benchmarks like AlpacaEval 2.0 are used to evaluate self-rewarding models.
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md): SCoRe and Re-ReST explore this for self-correction.

---

##

## References
- [source:arxiv:2203.14465] [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- [source:arxiv:2308.08998] [Reinforced Self-Training (ReST) for Language Modeling](https://arxiv.org/abs/2308.08998)
- [source:arxiv:2505.03335] [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335)
- [source:arxiv:1809.02923] [Deep reinforcement learning in iterated amplification](https://arxiv.org/abs/1809.02923)
- [source:arxiv:2409.12917] [Training Language Models to Self-Correct via Reinforcement Learning](https://arxiv.org/abs/2409.12917)
- [source:arxiv:2401.10020] [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
- [source:arxiv:2406.01495] [Re-ReST: Reflection-Reinforced Self-Training for Language Agents](https://arxiv.org/abs/2406.01495)
