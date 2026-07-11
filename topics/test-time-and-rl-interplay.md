---
title: Test-time compute and RL interplay
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2510.09988
- arxiv:2304.01132
- arxiv:2403.09629
- wellecks:l1-controlling-how-long-a-reasoning-mode
- deepmind:alphageometry-2-new-geometry-reasoning-s
- arxiv:2501.02497
- arxiv:2203.14465
- nature:olympiad-level-formal-mathematical-reaso
- arxiv:2503.24235
- arxiv:2602.09574
- arxiv:2109.13582
- arxiv:2303.05510
- arxiv:2505.23614
- arxiv:2506.06444
- arxiv:2310.17022
open_questions:
- What is the **universal inference scaling law** relating model size, token budget,
  search width/depth, and verifier quality to task performance? Can the proposed metrics
  (Scalability, Controllability, $\eta$, $\xi_{UT}$, $\xi_P$) form a predictive theory?
- 'Prompt Space Search**: How to effectively optimize over reasoning structures/templates
  $p \in \mathcal{P}$ jointly with answer traces $s \in \mathcal{S}_p$? Current methods
  fix $p$; [source:arxiv:2510.09988] identifies this as a key limitation.'
- 'Reward Model Saturation**: PRMs often output near-binary scores [source:arxiv:2602.09574],
  limiting MCTS signal granularity. How to train calibrated, fine-grained process
  rewards without excessive annotation cost?'
- 'Internal vs. External Scaling Trade-off**: When does distillation (internalizing
  search) hit diminishing returns vs. continuing to scale external search? The survey
  notes 32B $\approx$ teacher on competition math [source:arxiv:2503.24235], but what
  about harder/out-of-distribution tasks?'
---

Test-time compute scaling has emerged as the primary lever for eliciting System-2 reasoning capabilities from LLMs, shifting the bottleneck from parameter count to inference-time search guided by learned reward signals. The interplay between reinforcement learning (RL) and test-time search creates a dual optimization loop where RL internalizes search strategies into policy parameters while search externalizes computation to solve specific instances beyond the policy's zero-shot capability.

## Test-Time Compute Paradigms: System-1 Adaptation vs System-2 Reasoning

The literature distinguishes two fundamentally different uses of test-time compute [source:arxiv:2501.02497]. **Test-Time Adaptation (TTA)** targets System-1 models—fast, pattern-based predictors—by updating parameters (Test-Time Training, FTTA), optimizing in-context demonstrations, editing representations via steering vectors, or calibrating outputs with $k$-NN retrieval. These methods aim at robustness to distribution shift but do not induce deliberate reasoning. **Test-Time Reasoning** targets System-2 models by structuring generation into explicit, multi-step reasoning chains with feedback and search. This paradigm decomposes into three components: *Feedback Modeling* (ORMs, PRMs, generative critics), *Search Strategies* (repeated sampling, self-correction, tree search), and *Improvement Training* (distilling search traces back into the policy) [source:arxiv:2501.02497]. The boundary is porous: STaR [source:arxiv:2203.14465] and Quiet-STaR [source:arxiv:2403.09629] use self-generated rationales filtered by correctness (a form of outcome verification) to fine-tune the base model, effectively converting test-time search successes into training data.

### A Unified Taxonomy of Test-Time Scaling

A comprehensive survey [source:arxiv:2503.24235] proposes a hierarchical four-axis taxonomy to organize the rapidly expanding TTS literature:

**1. What to Scale (Scaling Paradigms)**
*   **Parallel Scaling:** Generating multiple candidate responses in parallel and aggregating them.

$$
\text{Set of solutions } S = \{s_{m,i} \mid m \le M, i \le k_m\}, \text{ Final answer } \hat{s} = A(s_{1,1}, \dots, s_{M,k_M})
$$

*   **Sequential Scaling:** Iteratively updating intermediate states $n_t$ based on the problem context $p$.

$$
n_{t+1} = R(n_t, p)
$$

*   **Hybrid Scaling:** Combining parallel expansion ($E$) and sequential filtering ($S$).

$$
F_{t+1} = S(E(s)) \text{ for } s \in F_t
$$

*   **Internal Scaling:** Training the model to autonomously determine computation allocation via internal parameters.

$$
z_{t+1} = f_\theta(z_t), \text{ stop}(z_t) = \pi_\theta(z_t)
$$

**2. How to Scale (Implementation)**
*   **Tuning-based:** 
    *   *Supervised Fine-Tuning (SFT):* Includes imitation of long Chain-of-Thought (CoT) traces, distillation from stronger models, and SFT warmup.
    *   *Reinforcement Learning (RL):* Divided into reward model-free (e.g., rule-based rewards in DeepSeek R1) and reward model-based (e.g., PPO, GRPO).
*   **Inference-based:** 
    *   *Stimulation:* Using prompts (e.g., "think step by step"), decoding strategies (e.g., filler tokens), latent strategies (hidden space reasoning), or mixture-of-model strategies.
    *   *Verification:* Outcome verification (scoring final answers) and Process verification (step-level Process Reward Models or PRMs).
    *   *Search:* Utilizing Tree-of-Thoughts, MCTS, or graph search.
    *   *Aggregation:* Selection (e.g., Majority Voting, Best-of-N) or Fusion (merging multiple samples).

**3. Where to Scale (Application Domains)**
*   **Reasoning-intensive:** Mathematics, Programming/Code, Game Playing, and Scientific Reasoning.
*   **Agentic Tasks:** Scaling via design choice (multi-agent), environment feedback, or simulated environments.
*   **Others:** General benchmarks, knowledge-intensive tasks, and evaluation tasks (LLM-as-a-judge).

**4. How Well to Scale (Evaluation Metrics)**
*   **Performance:** $\text{Pass@1}$ and $\text{Pass@k}$.

$$
\text{P a s s@k}=\frac{1}{n}\sum_{i=1}^{n}\left(1-\frac{\binom{N-C_{i}}{k}}{\binom{N}{k}}\right)
$$

*   **Efficiency:** Token cost, FLOPs, KV cache size. Reasoning efficiency $\eta$:

$$
\eta(\mathcal{M})=\mathbb{E}_{\mathcal{T}\sim p(\mathcal{T})}\left[\frac{Q(\mathcal{M},\mathcal{D})}{\mathcal{C}(\mathcal{M},\mathcal{D})}\right]
$$

*   **Reasoning Quality:** Underthinking score ($\xi_{UT}$) and Process efficiency ($\xi_P$).

$$
\xi_{\mathrm{U T}}=\frac{1}{N_{\mathrm{i n c}}}\sum_{i=1}^{N_{\mathrm{i n c}}}\left(1-\frac{\hat{T}_{i}}{T_{i}}\right), \quad \xi_P = \frac{1}{N} \sum_{i=1}^N \frac{D_i}{T_i}
$$

*   **Controllability:** Adherence to compute budgets via the Control metric:

$$
\text{Control} = \frac{1}{|A|} \sum_{a \in A} I(a_{\min} \le a \le a_{\max})
$$

*   **Scalability:** Average slope of performance gains relative to compute.

$$
\text{S c a l i n g}=\frac{1}{\binom{|\mathcal{A}|}{2}}\sum_{a,b\in{\mathcal{A}}\atop b>a}\frac{f(b)-f(a)}{b-a}
$$

The survey notes key limitations per paradigm: **Parallel Scaling** suffers diminishing returns once solution coverage saturates; **Sequential Scaling** is prone to error accumulation and coherence challenges over long horizons; **Internal Scaling** lacks interpretability/controllability and risks "logical drift" without external guidance [source:arxiv:2503.24235].

## Search Strategies at Test Time

### Repeated Sampling: Best-of-N and Self-Consistency

The simplest search generates $N$ independent completions $y^{(1)},\dots,y^{(N)} \sim \pi_\theta(\cdot|x)$ and aggregates. **Best-of-N (BoN)** selects $\arg\max_i V(y^{(i)})$ using a verifier $V$ (ORM or PRM) [source:arxiv:2501.02497; wellecks:l1-controlling-how-long-a-reasoning-mode]. **Self-Consistency (voting)** selects the most frequent answer $\arg\max_a \sum_i \mathbb{I}[a_i = a]$, optionally weighted by $V$ [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. The voting accuracy converges as $N\to\infty$ to a limit determined by the generator $g$ and verifier $v$ [source:wellecks:l1-controlling-how-long-a-reasoning-mode]:

$$
\frac{1}{M}\sum_{i=1}^{M}{\mathbb{I}}\left[a_{i}^{*}=\arg\max_{a}\sum_{z}v(x,z,a)g(z,a|x)\right]
$$

where $z$ is the solution path. Self-consistency CoT improves math reasoning by $\sim 18\%$ over vanilla CoT [source:arxiv:2501.02497]. However, BoN suffers from **over-optimization** when $V$ is imperfect: as $N$ grows, the selected output exploits reward model errors rather than true quality [source:wellecks:l1-controlling-how-long-a-reasoning-mode; arxiv:2510.09988].

### Self-Correction and Refinement

Refinement iterates $y^{(i+1)} = g(x, y^{(i)}, F(y^{(i)}))$ where $F$ provides feedback [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. **Extrinsic feedback** (code execution, theorem provers, retrievers) works reliably because it injects ground-truth signals and localizes errors [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. **Intrinsic feedback** (self-critique) yields mixed results in math: models often fail to locate their own errors even when capable of correcting them [source:arxiv:2501.02497; wellecks:l1-controlling-how-long-a-reasoning-mode]. AlphaProof uses Lean verification as extrinsic feedback at every tactic step, making refinement the core of its search [source:nature:olympiad-level-formal-mathematical-reaso].

### Tree Search: MCTS and Structured Exploration

Tree search explores a reasoning tree where nodes are partial states $s$ (e.g., prefix of CoT), actions $a$ are next-step generations, and transitions $s\to s'$ append tokens. **Monte Carlo Tree Search (MCTS)** operates in four phases [source:arxiv:2510.09988]:
1. **Selection**: UCT policy $a = \arg\max_{a\in A(s)} \left( Q(s,a) + c\sqrt{\frac{\ln N(s)}{N(s,a)}} \right)$
2. **Expansion**: Sample new child nodes via LLM policy $\pi_\theta$
3. **Simulation**: Rollout to terminal state (or heuristic evaluation)
4. **Backpropagation**: Update $Q(s,a)$ and visit counts $N(s,a)$

**Tree-of-Thought (ToT)** uses BFS/DFS with a value function; **Reward Balanced Search (Rebase)** allocates a compute budget $P$ to balance exploration/exploitation [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. AlphaProof employs an AND-OR tree search over Lean tactic states with a learned value function, where the return for decomposed subgoals is $G_t = \min(\text{returns of subgoals})$ to enforce all subgoals must be solved [source:nature:olympiad-level-formal-mathematical-reaso]. The **unified framework** of [source:arxiv:2510.09988] formalizes test-time scaling as:

$$
p^* = \arg\max_{p \in \mathcal{A}(\pi, Q, \mathcal{C}_{\text{infer}})} V(p)
$$

where $\mathcal{A}$ is the search algorithm, $\mathcal{C}_{\text{infer}}$ the compute budget, and $V$ the value function.

#### Budget-Guided MCTS (BG-MCTS)

Standard tree-search policies are typically **budget-agnostic**, treating the token budget $B$ merely as a termination condition, leading to late-stage over-branching or premature termination [source:arxiv:2602.09574]. **Budget-Guided MCTS (BG-MCTS)** aligns the search policy with the remaining budget via the **budget sufficiency ratio** $\rho = 1 - C_{\text{used}} / B$.

*   **Budget-Guided Selection (BG-PUCT):** Anneals exploration and biases selection toward deeper nodes as $\rho$ decreases:

$$
\text{BG-PUCT}(p, s, \rho) = \frac{\tilde{W}(s, \rho)}{m_s} + \rho c P(s | p) \sqrt{\frac{\ln(m_p)}{m_s}}
$$

    with a **completion bias** in the corrected value:

$$
\tilde{Q}(x, \rho) = Q(x) + \kappa (1 - \rho) \frac{d(x)}{\tilde{d}_{\text{ans}}}
$$

    where $d(x)$ is node depth and $\tilde{d}_{\text{ans}}$ is average depth of completed answers.
*   **Budget-Guided Tree Widening:** Introduces a **virtual generative child** $s_{\text{gen}}(p)$ to decide between widening vs. deepening:

$$
E_{\text{gen}}(p, \rho) = \mu(p) + \lambda \rho \sigma^2(p)
$$

    where $\mu, \sigma^2$ are mean/variance of child $Q$-values. The $\rho$ factor suppresses widening as budget depletes.
*   **Unified Selection:** At each node, select $s^*$ maximizing $\text{Score}(p, s, \rho)$ over standard children and the virtual generative child.

**Results:** On AIME24/25 (Qwen3-32B, $B=30\text{k}$), BG-MCTS achieves **0.350** accuracy vs. MCTS **0.300** and Greedy **0.267**. On UG-Physics, BG-MCTS reaches **0.372** vs. MCTS **0.356**. BG-MCTS produces significantly more correct answered nodes (413.7 at $B=30\text{k}$) vs. MCTS (143.7) and Repeated Sampling (48.0) [source:arxiv:2602.09574]. **Limitation:** Process reward model (GenPRM-7B) often produces saturated, near-binary scores, limiting search signal granularity; BG-MCTS may achieve lower tree-level reach rate (fewer total problems with any completed answer) as it trades breadth for precision [source:arxiv:2602.09574].

#### PPL-MCTS: Discriminator-Guided MCTS for Constrained Generation

**PPL-MCTS** (Plug and Play Language - MCTS) treats constrained text generation (sentiment, style, emotion) as tree exploration without fine-tuning the LM [source:arxiv:2109.13582]. It uses a pre-trained LM for token probabilities and a separate discriminator to score constraint satisfaction $p(c|x)$.

*   **Selection:** Standard PUCT with LM prior $p_\theta(x_i|x_{1:t-1})$.
*   **Expansion:** LM generates top-$k$ children.
*   **Simulation:** Random walk to terminal node.
*   **Backpropagation:** Final score $p(x|c) \propto p_D(c|x)^\alpha p_\theta(x)^{1-\alpha}$ propagated via averaging.
*   **Repetition Penalty:** Temperature scaling factor $I(i) > 1$ for tokens already in sequence.

**Results:** On `amazon_polarity`, `CLS` (French), and `emotion` datasets, PPL-MCTS achieves SOTA accuracy (0.97 on `amazon_polarity`) competitive with task-specific models (GeDi-Classloss 0.96), with better readability (perplexity) than pure re-ranking baselines. Human evaluation shows parity with GeDi on polarity (4.43 vs 4.46) and readability (4.05 vs 4.19). Accuracy plateaus after ~5 roll-out tokens on `CLS` [source:arxiv:2109.13582]. **Limitation:** Roll-out phase is expensive for long sequences; reducing it diminishes "long-term vision."

#### Planning-Guided Transformer Decoding (PG-TD) for Code Generation

**PG-TD** treats code generation as an MDP with test-case feedback *during* decoding, not just post-hoc filtering [source:arxiv:2303.05510].

*   **MDP:** State = problem + partial program; Action = next token; Reward = pass rate on public test cases (0 for partial programs).
*   **Selection:** **P-UCB** balancing exploitation, exploration, and Transformer prior:

$$
\mathrm{P-UCB}(s,a)=Q(s,a)+\beta(s)\cdot P_{\mathrm{Transformer}}(a|s)\cdot\frac{\sqrt{\log(s.visits)}}{1+s'.visits}
$$

$$
\beta(s)=\log\left(\frac{s.visits+c_{base}+1}{c_{base}}\right)+c
$$

*   **Evaluation:** `BEAM SEARCH` completes partial program; result executed for reward.
*   **Efficiency:** Tree Structure Caching (reuses beam search tree) + Sequence Caching (caches completed programs/rewards).

**Results:** On APPS Introductory (GPT-2), PG-TD ($c=4$) achieves **26.70%** pass rate vs. Sampling+Filtering **25.19%** and Beam Search **11.95%**. On CodeContests: **24.05%** vs. **20.40%**. Caching reduces compute from 2206s to 1312s (100 problems). Compilation errors drop from 5.58% to 1.93%; runtime errors from 32.95% to 19.5%. Fine-tuning GPT-2 (2.7B) on PG-TD samples improves pass rate from 14.32% to 16.54% [source:arxiv:2303.05510]. **Limitation:** Requires test cases (auto-generated can mitigate); more expensive than pure beam search due to repeated beam search calls.

| Search Method | Compute Profile | Feedback Type | Typical Use Case |
|---------------|-----------------|---------------|------------------|
| Best-of-N / Voting | Parallel, $N\times$ | Outcome (ORM) / None | Quick quality boost |
| Self-Correction | Sequential, iterative | Extrinsic (tools) / Intrinsic | Code, formal proof |
| MCTS / ToT | Tree, adaptive | Process (PRM) / Value net | Hard reasoning, planning |
| **BG-MCTS** | Tree, budget-aware | Process (PRM) / Value net | Fixed token budget reasoning |
| **PPL-MCTS** | Tree, discriminator-guided | Discriminator score | Constrained generation (style/sentiment) |
| **PG-TD** | Tree, test-case guided | Execution (pass rate) | Code generation |

## Reward Modeling for Search

Search requires a reward signal $R(s,a)$ or $V(s)$. Three families dominate:

1. **Outcome Reward Models (ORMs)**: Score final answer correctness. Cheap but sparse; cannot guide intermediate steps.
2. **Process Reward Models (PRMs)**: Score each reasoning step. ThinkPRM achieves comparable performance with $1\%$ of process supervision data vs discriminative PRMs [source:arxiv:2501.02497]. Generative PRMs (LLM-as-judge) can be trained with $\sim 40$K SFT+DPO samples [source:arxiv:2501.02497].
3. **Generative Critics**: LLMs produce natural language critiques. Training-free (prompting) or SFT/DPO-trained. Used in self-correction loops [source:arxiv:2501.02497].

AlphaProof uses **Lean verification as perfect binary reward** at the tactic level ($r_t = -1$ per tactic to incentivize brevity) [source:nature:olympiad-level-formal-mathematical-reaso]. The unified framework treats reward as a **unified signal** serving two roles [source:arxiv:2510.09988]:
- **Internalization (RL)**: $\theta^* = \arg\max_\theta \mathbb{E}_{\tau\sim\pi_\theta}[G(\tau)] - \lambda \int_{s\in\tau} D_{\text{KL}}(\pi_\theta(\cdot|s)\|\pi_{\mathcal{P}}(\cdot|s)) ds$
- **Externalization (Search)**: $p^* = \arg\max_{p\in\mathcal{P}_{\text{plan}}} \left[ \sum_{t=0}^{T-1} \gamma^t R_{\text{ext}}(s_t,a_t) + \mathcal{H}_\theta(s_T,p) \right]$

### Controlled Decoding (CD): Inference-Time Alignment via Prefix Scorer

**Controlled Decoding (CD)** solves a tokenwise KL-regularized RL objective at inference time using a separate **prefix scorer** module $V_\theta$, leaving the base model $\pi_{\text{ref}}$ frozen [source:arxiv:2310.17022].

*   **Training Prefix Scorer:**
    *   **CD-FUDGE (Supervised):** Minimizes squared error between prefix score and final reward $r$:

$$
\ell_{F}(\mathbf{x},\mathbf{y};\theta)=\frac{1}{2}\sum_{t}\left(V_{\theta}([\mathbf{x},y^{t}])-r([\mathbf{x},\mathbf{y}])\right)^{2}
$$

    *   **CD-Q (Off-policy DQN-style):** Minimizes Bellman residual with target $\dot{v}_t$ (expected next-token value or terminal reward):

$$
\ell_{Q}(\mathbf{x},y^{t};\theta)=\frac{1}{2}\sum_{t}\big(V_{\theta}([\mathbf{x},y^{t}])-\dot{v}_{t}\big)^{2}
$$

*   **Inference Strategies:**
    *   **Tokenwise Sampling:** $\pi_{\theta}(z|[\mathbf{x},y^{t}]) \propto \pi_{\text{ref}}(z|[\mathbf{x},y^{t}])e^{\lambda V_{\theta}([\mathbf{x},y^{t},z])}$
    *   **Blockwise Best-of-$K$:** Sample $K$ continuation blocks of length $M$; select highest $V_\theta$; repeat until EOS.

**Theoretical Grounding:** The optimal policy for the tokenwise KL-regularized objective $J_\lambda = \lambda A - D$ is $\pi_{\lambda}^{\star}(z|[\mathbf{x},y^{t}]) \propto p(z|[\mathbf{x},y^{t}])e^{\lambda V^{\star}([\mathbf{x},y^{t},z])}$ [source:arxiv:2310.17022].

**Results:** Blockwise CD-Q matches best-of-$K$ reward-KL tradeoff with far fewer samples (CD-Q $K=6$ $\approx$ BoN $K=50$). CD-Q/FUDGE outperform IPO/PPO on HH win rates. Prefix scorers transfer across model sizes (PaLM 2-XXS $\to$ S/XS) without retraining. Hybrid DPO + blockwise CD-Q improves efficiency (KL=5 at $K=8$ vs $K=32$ for vanilla CD-Q) [source:arxiv:2310.17022]. **Limitations:** Tokenwise RL more restrictive than sequence-level RLHF/DPO; prefix scorers show lower classification accuracy (~0.6 vs RM ~0.7) due to noisy training data; blockwise CD not optimal for sequence-level objective.

### SAFFRON: Safety Inference Scaling via Multifurcation Reward Model

**SAFFRON-1** addresses the **exploration–efficiency dilemma** in safety test-time scaling: tree search (Beam, MCTS) requires frequent PRM calls, but safety tasks lack self-consistency (no majority voting), making PRM overhead outweigh gains vs. simple Best-of-$N$ [source:arxiv:2506.06444].

*   **Multifurcation Reward Model (MRM):** Decoder-only Transformer $M_\theta: \mathcal{V}^+ \to \mathbb{R}^{\mathcal{V}}$ predicts reward vector for *all* next tokens in one forward pass (vs. scalar PRM per token).
*   **Partial Supervision Training:** LoRA on Llama Guard 3 1B + trainable unembedding bias. Loss on corpus $\mathcal{C}$:

$$
\mathcal{L}_{\mathsf{M R M}}(\mathbf{s}_{[0:j+1)}) := (M_{\theta}(\mathbf{s}_{[0:j)})_{s_{j}} - R(\mathbf{s}_{[0:j+1)}))^{2}
$$

*   **Conservative Exploration Constraint:** Masks "unseen tokens" (not in training corpus $\mathcal{C}$) to $-\infty$:

$$
M_{\mathsf{cons}}(\mathfrak{s})_{a} := \begin{cases} -\infty, & a \in \mathcal{V}_{\text{unseen}} \\ M_{\theta}(\mathfrak{s})_{a}, & \text{otherwise} \end{cases}
$$

*   **Trie-based KV Caching:** Shared prefixes share KV cache tensors (Trie structure), reducing time/space complexity.
*   **Inference:** Beam search variant; MRM called once per beam sequence to score all top-$p$ tokens; select top-$N$ by predicted reward.

**Results:** On **Ai2 Refusals**, SAFFRON-1 achieves ASR **0.175** vs. BoN (0.285), Rebase (0.415), DeAL (0.435). On **Harmful HEx-PHI**, ASR **0.409** vs. BoN (0.582), Rebase (0.758), DeAL (0.794). **Scaling Efficiency:** To reach ASR $\approx 0.4$, SAFFRON-1 needs $\approx \mathbf{60}$ TFLOP vs. $\approx \mathbf{190}$ TFLOP for strongest baseline. Search width $N=1\to16$ reduces ASR from 0.827 to 0.497 on HEx-PHI. Efficiency metric: $\mathrm{ScalEff} := \frac{\log \frac{\mathrm{TFLOP}_{\mathrm{Lim}}}{\mathrm{TFLOP}}}{\mathrm{ASR}}$ [source:arxiv:2506.06444]. **Limitations:** Applies only to closed-source LLMs (policy frozen); MRM is tokenizer-dependent (different tokenizer $\to$ different MRM).

**Disagreement**: [source:arxiv:2501.02497] emphasizes PRMs as critical for step-wise search guidance; [source:nature:olympiad-level-formal-mathematical-reaso] shows AlphaProof succeeds with only sparse outcome rewards (Lean proof completion) plus a learned value function for intermediate states, suggesting PRMs may be unnecessary when a perfect verifier exists. [source:arxiv:2510.09988] argues the reward design must match the search algorithm: MCTS needs a value function $Q(s,a)$, while BoN only needs a terminal score. [source:arxiv:2506.06444] demonstrates that for safety, scalar PRMs are *too slow* for tree search, motivating a structural change (MRM) rather than better PRM training.

## RL and Search Interplay: The Dual Optimization Paradigm

The central insight is that **RL and search are two optimizers for one objective** [source:arxiv:2510.09988]. RL *internalizes* search behavior into policy weights $\theta$ (training-time scaling in latent space $\Theta$). Search *externalizes* computation at test time to find $p^*$ for a specific problem $Q$ (test-time scaling in objective space $\mathcal{P}(Q)$). This creates a flywheel:

1. **Search generates high-quality traces** $\tau^*$ for hard problems (using current policy $\pi_\theta$ as proposal + reward guidance).
2. **RL trains on $\tau^*$** (SFT or policy gradient) to improve $\pi_\theta$, reducing the need for search at future test time.
3. **Improved $\pi_\theta$ enables more efficient search** (better proposals, better value estimates).

**STaR** [source:arxiv:2203.14465] implements this loop: generate rationales $\to$ filter by correctness $\to$ fine-tune on correct rationales $\to$ iterate. The objective is $\nabla J = \sum_i \mathbb{E}[\mathbb{1}(\hat{y}_i=y_i) \nabla \log p_M(\hat{y}_i,\hat{r}_i|x_i)]$—a policy gradient with binary reward. **Rationalization** (conditioning on ground-truth answer to generate rationale) prevents plateauing on unsolved problems. **Quiet-STaR** [source:arxiv:2403.09629] generalizes to *every token position*: generate parallel rationales $\to$ reward by log-likelihood improvement on future tokens $\to$ REINFORCE update. The reward is $r_j = \log p_{j:j+n_{\text{true}}}^{\text{talk}}(X_{j+1:j+n_{\text{true}}+1}) - \log \overline{p}_{j:j+n_{\text{true}}}^{\text{talk}}(\cdots)$.

**ReST-MCTS** [source:arxiv:2510.09988] uses MCTS to generate training data for iterative SFT/RL, achieving SOTA on MATH, GPQA, CEval. **AlphaProof** [source:nature:olympiad-level-formal-mathematical-reaso] runs a massive RL loop: Gemini auto-formalizes 1M NL problems $\to$ 80M Lean problems $\to$ distributed actors use MCTS to prove/disprove $\to$ Lean-verified outcomes update proof network (policy + value). At **test time**, AlphaProof performs *Test-Time RL (TTRL)*: generates millions of problem variants $\to$ runs focused RL on them to adapt to the target problem structure.

**Joint Space Optimization** [source:arxiv:2510.09988] extends the duality: search should optimize over both **Prompt Space $\mathcal{P}$** (reasoning structure/template) and **Answer Space $\mathcal{S}$** (solution trace given template): $s^* = \arg\max_{p\in\mathcal{P}, s\in\mathcal{S}_p} V(s)$. Most current methods fix $p$ (single prompt template), searching only $\mathcal{S}$—a key limitation.

The survey [source:arxiv:2503.24235] frames this as **Internal Scaling** (RL internalizing computation into $\theta$) vs. **External Scaling** (inference-time search), noting that distillation (e.g., 32B student from top-tier reasoner teacher) can nearly match teacher performance on competition math, effectively converting external search compute into internalized capability.

## Inference Scaling Laws and Compute-Optimal Tradeoffs

Empirical evidence suggests **smaller models with more test-time tokens can outperform larger models with fewer tokens** [source:wellecks:l1-controlling-how-long-a-reasoning-mode]. This inverts the training-time scaling law (where larger models are more compute-efficient). The compute-optimal inference strategy depends on the task difficulty distribution: easy tasks need only System-1; hard tasks justify System-2 search. No universal test-time scaling law exists yet [source:arxiv:2501.02497]. DeepSeek-R1-style long CoT models regularly exceed 10,000 tokens/response [source:wellecks:l1-controlling-how-long-a-reasoning-mode], making token budget a first-order cost factor.

The survey [source:arxiv:2503.24235] proposes quantitative **Scalability** and **Controllability** metrics (see Taxonomy above) to move toward predictive scaling laws. Key findings: Distillation allows a 32B model to nearly match a top-tier teacher on competition math. **KV Cache Efficiency:** ETS achieves $1.8\times$ KV cache reduction vs. REBASE, yielding $1.4\times$ faster inference on H100. **Controllability:** In $k-\epsilon$ tests, >97% of targets reachable with prompt length $\le 10$ tokens and $\epsilon \le 0.05$.

### Length Control and Budget-Aware Reasoning (L1)

Unbounded CoT length causes cost explosion. **L1** [source:wellecks:l1-controlling-how-long-a-reasoning-mode] uses RL to adhere to length constraints in the prompt (e.g., "use up to 1000 tokens") by adding a length penalty to the reward. The model learns to express uncertainty ("Wait..."), backtrack ("Alternatively..."), and self-verify within the budget. This shifts control from *implicit* (model decides when to stop) to *explicit* (user specifies budget). AlphaProof's $r_t=-1$ per tactic is a hard-coded length penalty; L1 makes it a controllable hyperparameter. BG-MCTS [source:arxiv:2602.09574] provides a complementary *search-side* budget adherence mechanism via $\rho$-conditioned selection/widening.

## Cross-Modality: Inference-Time Scaling in Diffusion Models

The principles of test-time search extend beyond LLMs. **Inference-time Scaling of Diffusion Models** [source:arxiv:2505.23614] orchestrates **global search** (BFS/DFS over denoising particles) and **local search** (annealed Langevin MCMC) to optimize a verifier $f(x_0)$.

*   **Global Search (BFS):** Parallel denoising of $N$ particles; scoring via denoised estimate $x_{0|t}$; resampling (Multinomial/SSP) and tempering schedules (Constant/Increase/Infinite) reallocate compute.
*   **Global Search (Adaptive DFS):** Iterative denoising with **adaptive backtracking** (reintroduce noise if verifier score $< \delta_t$).
*   **Local Search:** Samples from $\tilde{p}_{0}(x_{0})\propto p_{0}(x_{0})f(x_{0})^{\lambda}$ via **annealed Langevin MCMC** (ULA):

$$
\boldsymbol{x}^{i+1}=\boldsymbol{x}^{i}+\eta\nabla_{\boldsymbol{x}^{i}}\log\nu(\boldsymbol{x}^{i})+\sqrt{2\eta}\boldsymbol{\epsilon}^{i}
$$

    **Proposition 1:** In continuous limit ($T\to\infty$), training-free guidance with recurrence $\equiv$ Langevin MCMC on annealed distributions.
*   **Double-Verifier Strategy:** Separate verifiers for local/global search to mitigate reward hacking (OOD samples fooling verifier).

**Results:** Image Gen (SDv1.5, $N=8$): Improved BFS **1.087** vs DAS **1.052** vs SVDD **0.775**. SDXL: **1.291** vs DAS **1.265**. DFS outperforms BoN/BFS with $2\times$ less compute. Offline RL (D4RL): TTS **86.1** avg locomotion vs D-QL **86.3**, QGPO **86.6**, DAS **80.2**, TFG **82.1**. Policy Distillation: TTS-finetuned outperforms DPPO (Halfcheetah **51.6** vs **47.8**). Double-Verifier (ImageNet): BFS-double FID **118.2** / Acc **55.9%** vs BFS-single FID **133.3** / Acc **46.5%** [source:arxiv:2505.23614]. **Limitation:** Requires tuning additional hyperparameters; suggests evolutionary search for hyperparameter optimization.

## Case Studies: AlphaProof and AlphaGeometry 2

| Component | AlphaProof | AlphaGeometry 2 |
|-----------|------------|-----------------|
| **Domain** | Algebra, Number Theory | Geometry |
| **Formalism** | Lean 4 | Custom symbolic engine + NL |
| **Policy** | 3B encoder-decoder (proof network) | Gemini-based LM (trained from scratch) |
| **Search** | AND-OR MCTS with learned value net | Neuro-symbolic: LM proposes constructions, engine verifies |
| **Reward** | $r_t=-1$ per tactic; $G_t=\min(\text{subgoal returns})$ | Symbolic engine verification (binary) |
| **Training** | 300B tokens pretrain $\to$ 300K SFT $\to$ RL on 80M problems | 10$\times$ synthetic data vs AlphaGeometry 1 |
| **Test-Time RL** | Generates problem variants, runs focused RL | Knowledge-sharing across search trees |
| **IMO 2024** | 3/5 non-geometry (incl. hardest) | 1/1 geometry in 19 sec |
| **Combined** | **28/42 (Silver)** | |

AlphaGeometry 2's symbolic engine is **2 orders of magnitude faster** than v1, enabling deeper search [source:deepmind:alphageometry-2-new-geometry-reasoning-s]. Both systems required **manual formalization** of IMO problems—a major deployment bottleneck. Combinatorics problems remain unsolved by either system [source:deepmind:alphageometry-2-new-geometry-reasoning-s; nature:olympiad-level-formal-mathematical-reaso].

## Current Status and Trajectory

**Rising**: Test-time search (MCTS, BoN, self-correction with tools) is the default for high-stakes reasoning (math, code, formal proof). The RL-search flywheel (ReST-MCTS, AlphaProof) is the dominant paradigm for pushing frontier capabilities. **Budget-aware search** (BG-MCTS, L1) is emerging as critical for practical deployment. **Inference-time alignment without weight updates** (Controlled Decoding, SAFFRON's MRM) enables modular, configurable steering. **Cross-modality scaling** (diffusion models) validates the generality of search+verifier frameworks.

**Default**: BoN/self-consistency with ORMs is standard for API-based deployment; PRMs are adopted where step-wise control matters (e.g., process supervision for math). **Fading/Unsettled**: Pure intrinsic self-correction (no tools) shows inconsistent gains and is not widely reported as reliable for hard reasoning [source:arxiv:2501.02497; wellecks:l1-controlling-how-long-a-reasoning-mode]. Prompt-space search (optimizing reasoning structure $p$) is largely unexplored beyond fixed templates [source:arxiv:2510.09988]. **Critical gap**: No universal inference scaling law exists to predict compute-optimal search configuration for a given task/model/verifier triplet [source:arxiv:2501.02497; arxiv:2503.24235]. System-2 models struggle to generalize to non-symbolic reasoning (e.g., open-ended analysis, creative writing) [source:arxiv:2501.02497]. **Safety-specific scaling** requires architectural innovations (MRM) to overcome PRM latency bottlenecks [source:arxiv:2506.06444]. **Reward model saturation** (near-binary PRM scores) limits search signal granularity in MCTS [source:arxiv:2602.09574].

## Key Takeaways

- Test-time compute splits into **TTA (System-1 robustness)** and **Test-Time Reasoning (System-2 search)**; only the latter induces deliberate multi-step reasoning.
- **Search strategies form a hierarchy**: repeated sampling (parallel, easy) $\to$ self-correction (sequential, needs extrinsic feedback) $\to$ tree search (structured, needs value function). **Budget-aware variants** (BG-MCTS, L1) add explicit compute control.
- **Reward design must match search algorithm**: BoN needs terminal score; MCTS needs step-wise $Q(s,a)$; RL needs differentiable/low-variance reward. **Structural innovations** (MRM for safety, prefix scorer for CD) can overcome latency/granularity limits of standard PRMs.
- **RL and search are dual optimizers**: RL internalizes, search externalizes. The flywheel (search $\to$ training data $\to$ better policy $\to$ better search) drives frontier progress (STaR, ReST-MCTS, AlphaProof). **Distillation** converts external search compute into internalized capability.
- **Length control is becoming explicit**: L1-style budget-aware RL and BG-MCTS's $\rho$-conditioned search replace implicit EOS prediction; AlphaProof's per-step penalty is a special case.
- **Formal verification (Lean) provides perfect rewards** but requires manual formalization and massive compute (AlphaProof: days per problem).
- **Inference scaling laws are unknown**: compute-optimal tradeoff between model size, token budget, and search width/depth is empirical, not predicted. New metrics (Scalability, Controllability, Reasoning Efficiency) aim to quantify this [source:arxiv:2503.24235].
- **Cross-modality validation**: Diffusion models adopt identical global+local search + verifier framework, confirming generality [source:arxiv:2505.23614].
- **Safety scaling demands efficiency**: SAFFRON's MRM + Trie KV caching shows 3$\times$ TFLOP reduction vs. baselines for same ASR [source:arxiv:2506.06444].

## Related Topics

- [RL for reasoning models](rl-for-reasoning.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [KL regularization in RLHF](kl-regularization.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [DPO variants deep-dive](dpo-variants.md)
- [RLAIF (RL from AI feedback)](rlaif.md)
- [Nash and game-theoretic preference optimization](nash-and-game-theoretic-po.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)
- [Length and format bias](length-and-format-bias.md)
- [The alignment tax](alignment-tax.md)
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md)
- [LLM-as-judge](llm-as-judge.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [Judging bias and contamination](judging-bias-and-contamination.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)

## References
- [source:arxiv:2510.09988] [Unifying Tree Search Algorithm and Reward Design for LLM Reasoning](https://arxiv.org/abs/2510.09988)
- [source:arxiv:2304.01132] [Search-in-the-Chain: Interactively Enhancing Large Language Models with Search](https://arxiv.org/abs/2304.01132)
- [source:arxiv:2403.09629] [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
- [source:wellecks:l1-controlling-how-long-a-reasoning-mode] [L1: Controlling how long a reasoning model thinks with reinforcement learning](https://wellecks.com/data/welleck2025scifm_tutorial.pdf)
- [source:deepmind:alphageometry-2-new-geometry-reasoning-s] [AlphaGeometry 2: New geometry reasoning system](https://deepmind.google/blog/ai-solves-imo-problems-at-silver-medal-level/)
- [source:arxiv:2501.02497] [Test-Time Compute: from System-1 Thinking to System-2 Reasoning](https://arxiv.org/html/2501.02497v2)
- [source:arxiv:2203.14465] [STaR: Self-Taught Reasoner: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- [source:nature:olympiad-level-formal-mathematical-reaso] [Olympiad-level formal mathematical reasoning with reinforcement learning (AlphaProof)](https://www.nature.com/articles/s41586-025-09833-y)
- [source:arxiv:2503.24235] [A Survey on Test-Time Scaling in Large Language Models: What, How, Where, and How Well?](https://arxiv.org/abs/2503.24235)
- [source:arxiv:2602.09574] [Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling of LLMs](https://arxiv.org/abs/2602.09574)
- [source:arxiv:2109.13582] [PPL-MCTS: Constrained Textual Generation Through Discriminator-Guided MCTS Decoding](https://arxiv.org/abs/2109.13582)
- [source:arxiv:2303.05510] [Planning with Large Language Models for Code Generation](https://arxiv.org/abs/2303.05510)
- [source:arxiv:2505.23614] [Inference-time Scaling of Diffusion Models through Classical Search](https://arxiv.org/abs/2505.23614)
- [source:arxiv:2506.06444] [Saffron-1: Safety Inference Scaling](https://arxiv.org/abs/2506.06444)
- [source:arxiv:2310.17022] [Controlled Decoding from Language Models](https://arxiv.org/abs/2310.17022)
