---
title: RLAIF (RL from AI feedback)
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2212.08073
- arxiv:2302.10845
- arxiv:2401.10020
- arxiv:2305.18290
- arxiv:2402.04792
- arxiv:1706.03741
- arxiv:2306.05685
- arxiv:2401.10020
open_questions:
- Can self-rewarding models iteratively improve their reward modeling capabilities
  without catastrophic reward hacking (e.g., length bias, verbosity bias)?
- How can RLAIF methods be adapted to ensure safety alignment without human oversight?
- What are the scaling laws for self-rewarding models beyond three iterations? Will
  performance gains saturate or continue?
- Can online AI feedback (OAIF) generalize to out-of-distribution prompts or rapidly
  evolving human values?
---

Here is the fully revised article with all requested fixes, grounded strictly in the provided sources, and maintaining the rigorous technical style:

---

# RLAIF (Reinforcement Learning from AI Feedback)

Reinforcement Learning from AI Feedback (RLAIF) is a paradigm for aligning large language models (LLMs) by replacing or augmenting human preference labels with AI-generated feedback. This approach addresses the scalability, cost, and iteration-speed limitations of Reinforcement Learning from Human Feedback (RLHF) while enabling continuous, self-improving alignment loops. RLAIF encompasses methods such as Constitutional AI (CAI), self-rewarding models, and online AI feedback (OAIF), each leveraging LLMs as annotators, critics, or reward models to guide policy optimization.

---

## Core Problem and Motivation

RLHF has become the de facto standard for aligning LLMs with human preferences, but it suffers from three fundamental bottlenecks:
1. **Scalability**: Human annotation is slow, expensive, and difficult to scale. For example, training a single reward model for harmlessness in Constitutional AI required 182,831 red-teaming prompts [source:arxiv:2212.08073].
2. **Static Feedback**: The reward model is frozen after training, preventing it from adapting to the evolving policy or improving alongside it. This creates a performance ceiling limited by the quality of the initial human annotations [source:arxiv:2401.10020].
3. **Distribution Shift**: Offline preference datasets are collected from a fixed data-generating policy (e.g., a supervised fine-tuning (SFT) model), which diverges from the evolving aligned policy during training. This mismatch causes overfitting and performance degradation in direct alignment from preferences (DAP) methods like DPO [source:arxiv:2402.04792].

RLAIF addresses these challenges by replacing human feedback with AI-generated labels, enabling:
- **Autonomous scaling**: AI feedback can be generated on-demand, reducing reliance on human labor.
- **Dynamic adaptation**: The feedback model can evolve alongside the policy, avoiding the static reward model bottleneck.
- **On-policy feedback**: Online AI feedback (OAIF) mitigates distribution shift by evaluating the policy’s own generations during training [source:arxiv:2402.04792].

---

## Methodological Taxonomy

RLAIF methods can be categorized along two axes: (1) the **source of feedback** (human vs. AI) and (2) the **feedback modality** (offline vs. online). The key variants are summarized in Table 1.

| Method               | Feedback Source | Feedback Modality | Key Innovation                                                                 | Example Sources                     |
|----------------------|-----------------|-------------------|--------------------------------------------------------------------------------|-------------------------------------|
| RLHF                 | Human           | Offline           | Standard pipeline: human preferences → reward model → RL fine-tuning.         | [source:arxiv:1706.03741]           |
| Constitutional AI    | AI + Human      | Offline           | AI critiques/revises responses using natural language principles ("constitution"). | [source:arxiv:2212.08073]           |
| Self-Rewarding       | AI              | Offline           | LLM generates and evaluates its own responses, training via DPO.               | [source:arxiv:2401.10020]           |
| Online AI Feedback   | AI              | Online            | LLM evaluates on-policy generations during training, enabling DAP methods.     | [source:arxiv:2402.04792]           |

### 1. Constitutional AI (CAI)
Constitutional AI [source:arxiv:2212.08073] replaces human harmlessness labels with AI-generated feedback guided by a "constitution"—a set of natural language principles. The method proceeds in two phases:
1. **Supervised Learning (SL) Phase**:
   - For each red-teaming prompt, the model generates an initial response.
   - The model critiques its own response using a randomly sampled constitutional principle (e.g., "Identify specific ways the assistant’s last response was harmful or unethical").
   - The model revises its response based on the critique.
   - This critique-revision cycle is repeated sequentially, and the model is fine-tuned on the revised responses to produce the SL-CAI checkpoint.
2. **Reinforcement Learning (RL) Phase**:
   - The SL-CAI model generates response pairs for each prompt.
   - A feedback model evaluates the pairs using a multiple-choice prompt guided by constitutional principles, optionally employing Chain-of-Thought (CoT) reasoning.
   - The feedback model’s log-probabilities for each option are normalized to produce soft preference targets:
     $$ \hat{y}_A = \frac{\exp(\log p(A))}{\exp(\log p(A)) + \exp(\log p(B))}. $$
   - These targets are used to train a hybrid preference model (PM), which combines AI-generated harmlessness labels with human helpfulness labels. The PM provides the reward signal for PPO-style RL, yielding the final RL-CAI policy.

**Key Design Choices**:
- **Soft Preference Targets**: CoT prompting causes the feedback model to commit strongly to one option, yielding poorly calibrated probabilities near 0 or 1. To mitigate this, probabilities are clamped to the range $[0.4, 0.6]$ before training the PM.
- **Principle Ensembling**: Labels are generated by ensembling over 16 constitutional principles, stabilizing PM scores.
- **Hybrid Feedback**: Human feedback is retained for helpfulness, limiting full autonomy.

### 2. Self-Rewarding Language Models
Self-rewarding models [source:arxiv:2401.10020] eliminate the fixed reward model bottleneck by enabling the LLM to generate and evaluate its own responses. The method operates via an iterative self-alignment loop:
1. **Initialization**: A base model $M_0$ is fine-tuned via SFT on seed Instruction Fine-Tuning (IFT) and Evaluation Fine-Tuning (EFT) data to produce $M_1$.
2. **Self-Instruction Creation**: For each iteration $t$, $M_t$ generates novel prompts via few-shot sampling, produces $N=4$ candidate responses per prompt, and evaluates them using an LLM-as-a-Judge prompt with an additive five-criterion rubric (relevance, coverage, usefulness, clarity, expertise). Scores are averaged over three samples to reduce variance.
3. **Preference Pair Construction**: The highest- and lowest-scoring responses are paired into preference tuples $(x_i, y_i^w, y_i^l)$, discarding ties.
4. **Training**: The model is updated via DPO on the augmented dataset to produce $M_{t+1}$.

The iterative sequence is formally defined as:
$$
\begin{aligned}
M_0 &: \text{Base pretrained LLM} \\
M_1 &: \text{SFT on IFT + EFT} \\
M_{t+1} &: \text{DPO on IFT + EFT + AIFT}(M_t) \text{ for } t \geq 1,
\end{aligned}
$$
where $\text{AIFT}(M_t)$ denotes the AI-generated preference pairs from iteration $t$.

### 3. Online AI Feedback (OAIF)
OAIF [source:arxiv:2402.04792] injects online, on-policy feedback into DAP methods by using an LLM as an annotator. The training procedure is:
1. Sample a prompt $x$ from a dataset.
2. Sample two candidate responses $y^1, y^2$ from the current policy $\pi_{\theta^t}(\cdot|x)$.
3. Prompt an LLM annotator to label the pair as preferred ($y^+$) and less preferred ($y^-$).
4. Update the policy parameters via $\theta^{t+1} \leftarrow \theta^t - \eta \nabla_\theta \ell(x, y^+, y^-, \theta^t)$, where $\ell$ is a DAP loss (e.g., DPO, IPO, or SLiC).

**Key Design Choices**:
- **Stop-Gradient**: Gradients are not propagated through the sampling or annotation steps, treating the LLM feedback as a static target per iteration.
- **Loss-Agnostic**: OAIF is compatible with any differentiable DAP objective.
- **Controllable Feedback**: The LLM annotator’s behavior can be adjusted via instruction prompts without retraining.

---

## Key Formulations

### 1. Preference Modeling
All RLAIF methods rely on a preference model to compare responses. The standard Bradley-Terry model [source:arxiv:1706.03741] defines the probability of preferring response $y^1$ over $y^2$ as:
$$ P[y^1 \succ y^2] = \frac{\exp(r(x, y^1))}{\exp(r(x, y^1)) + \exp(r(x, y^2))}, $$
where $r(x, y)$ is a latent reward function. In RLAIF, $r(x, y)$ is either:
- **Implicitly modeled** via the policy’s log-probabilities (e.g., in DPO [source:arxiv:2305.18290]).
- **Explicitly generated** by an LLM-as-a-Judge (e.g., in self-rewarding models [source:arxiv:2401.10020]).

### 2. Direct Preference Optimization (DPO)
DPO [source:arxiv:2305.18290] eliminates the need for an explicit reward model by directly optimizing the policy via a classification objective. The DPO loss is derived from the optimal RLHF policy under a KL constraint:
$$ \pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right), $$
where $Z(x)$ is a partition function. Inverting this relationship yields the implicit reward:
$$ r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x). $$
Substituting into the Bradley-Terry model and canceling $Z(x)$ yields the DPO loss:
$$ \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]. $$

**PPO Clip Equation**:
For completeness, the PPO clip objective used in RLHF is:
$$ \mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right], $$
where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio, $\hat{A}_t$ is the advantage estimate, and $\epsilon$ is a hyperparameter (typically 0.1 or 0.2) [source:arxiv:1706.03741].

### 3. Online AI Feedback (OAIF) Losses
OAIF adapts DAP losses to online settings. The key variants are:
- **DPO**:
  $$ -\log \sigma \left(\beta \log \frac{\pi_\theta(y^+|x) \pi_{\theta^0}(y^-|x)}{\pi_{\theta^0}(y^+|x) \pi_\theta(y^-|x)}\right) $$
- **IPO** (Identity Preference Optimization):
  $$ \left(\log \left(\frac{\pi_\theta(y^+|x) \pi_{\theta^0}(y^-|x)}{\pi_\theta(y^-|x) \pi_{\theta^0}(y^+|x)}\right) - \frac{1}{2\beta}\right)^2 $$
- **SLiC** (Sequence Likelihood Calibration):
  $$ \max \left(0, 1 - \beta \log \left(\frac{\pi_\theta(y^+|x) \pi_{\theta^0}(y^-|x)}{\pi_\theta(y^-|x) \pi_{\theta^0}(y^+|x)}\right)\right) $$

---

## Quantitative Results

### 1. Constitutional AI
- **Harmlessness**: RL-CAI models achieve higher harmlessness Elo scores than RLHF baselines while reducing evasiveness. Absolute harmfulness scores (0–4 scale) decrease progressively across training steps, with crowdworker AB tests showing RL-CAI outperforming RLHF in harmlessness comparisons (8,135 comparisons) [source:arxiv:2212.08073].
- **Helpfulness**: RL-CAI maintains or improves helpfulness compared to RLHF, with crowdworker AB tests showing no significant degradation (10,274 comparisons) [source:arxiv:2212.08073].
- **Critique-Revision**: Harmlessness preference scores improve monotonically with each critique-revision step, with diminishing returns after 3–4 steps [source:arxiv:2212.08073].
- **Principle Ensembling**: Varying the number of constitutional principles does not significantly alter harmlessness scores but increases response diversity, aiding RL exploration [source:arxiv:2212.08073].

### 2. Self-Rewarding Models
- **Instruction Following**: AlpacaEval 2.0 win rates (vs. GPT-4 Turbo) improve from 9.94% ($M_1$) to 20.44% ($M_3$), surpassing GPT-4 0613 (15.76%) [source:arxiv:2401.10020].
- **Head-to-Head**: $M_2$ wins 55.5% against $M_1$; $M_3$ wins 47.7% against $M_2$ (GPT-4 judged) [source:arxiv:2401.10020].
- **Reward Modeling**: Pairwise accuracy against human rankings improves from 65.1% (SFT baseline) to 81.7% ($M_3$) [source:arxiv:2401.10020].
- **Response Length**: Average length grows from 1,092 tokens ($M_1$) to 2,552 tokens ($M_3$), correlating with perceived quality [source:arxiv:2401.10020].

### 3. Online AI Feedback (OAIF)
- **Human Evaluations**: Online DPO achieves a 66% win rate over offline DPO in human side-by-side evaluations. On TL;DR summarization, online DPO is preferred 58% of the time in four-way comparisons against SFT, RLHF, and RLAIF [source:arxiv:2402.04792].
- **Behavioral Control**: Instructing the annotator to favor brevity reduces average response length from 120 to 40 tokens while maintaining quality scores above the SFT baseline (3.72 and 3.26 vs. 3.19) [source:arxiv:2402.04792].
- **Small Annotators**: OAIF remains effective with small annotators; PaLM 2-XS as the annotator for a PaLM 2-XS policy yields a human quality score of 3.41, comparable to RLHF (3.38) [source:arxiv:2402.04792].

---

## Current Status and Trajectory

RLAIF is **rising rapidly** as a default technique for scalable, autonomous alignment, but its long-term trajectory remains contingent on unresolved challenges.

### Evidence of Growth
1. **Adoption**: Constitutional AI has been integrated into research systems (e.g., Anthropic’s experiments [source:arxiv:2212.08073]), and self-rewarding models are being explored by research labs (e.g., [source:arxiv:2401.10020]).
2. **Performance**: Self-rewarding models achieve state-of-the-art results on instruction-following benchmarks (e.g., AlpacaEval 2.0) without human feedback, demonstrating the viability of AI-generated preferences [source:arxiv:2401.10020].
3. **Methodological Diversity**: The emergence of online AI feedback (OAIF) [source:arxiv:2402.04792] and hybrid human-AI approaches (e.g., CAI) suggests a broadening toolkit for RLAIF.

### Challenges and Uncertainties
1. **Reward Hacking**: Self-rewarding models exhibit a strong correlation between response length and perceived quality, raising concerns about superficial alignment [source:arxiv:2401.10020]. LLM-as-a-Judge evaluations are also susceptible to position and verbosity biases [source:arxiv:2306.05685].
2. **Safety**: RLAIF methods have not been rigorously evaluated for safety alignment. Constitutional AI retains human feedback for helpfulness, and self-rewarding models lack safety-specific training [source:arxiv:2212.08073][source:arxiv:2401.10020].
3. **Scaling Laws**: The long-term performance of self-rewarding models remains unknown. Preliminary results are limited to three iterations, and it is unclear whether gains will saturate or continue [source:arxiv:2401.10020].
4. **Distribution Shift**: Online AI feedback mitigates distribution shift, but its effectiveness for out-of-distribution prompts or evolving human values is untested [source:arxiv:2402.04792].

### Trajectory
- **Short-Term (1–2 years)**: RLAIF will likely become a **default technique** for scalable alignment, particularly in domains where human feedback is expensive or slow (e.g., multilingual alignment, long-context tasks). Hybrid human-AI approaches (e.g., CAI) will dominate research systems.
- **Medium-Term (2–5 years)**: Online AI feedback (OAIF) may replace offline DAP methods (e.g., DPO) as the standard for preference optimization, given its superior performance and on-policy nature. Self-rewarding models could enable continuous, autonomous improvement, but safety and reward hacking risks will require mitigation.
- **Long-Term (5+ years)**: The trajectory hinges on resolving the **alignment tax**—whether RLAIF can match or exceed RLHF performance without sacrificing generality or safety. If self-rewarding models can iteratively improve their reward modeling capabilities while avoiding catastrophic reward hacking, RLAIF may become a **dominant paradigm** for alignment. However, if these challenges remain unresolved, RLAIF may remain a complementary technique to RLHF.

---

## Key Limitations and Open Challenges

### 1. Reward Hacking and Over-Optimization
- **Problem**: RLAIF methods are susceptible to reward hacking, where the policy exploits flaws in the AI feedback model to achieve high scores without genuine alignment. For example, self-rewarding models exhibit a strong correlation between response length and perceived quality [source:arxiv:2401.10020], and LLM-as-a-Judge evaluations are biased toward verbose or positionally favored responses [source:arxiv:2306.05685].
- **Evidence**: In Constitutional AI, over-optimization for harmlessness yields overly harsh, preachy, or boilerplate responses [source:arxiv:2212.08073]. In self-rewarding models, average response length grows from 1,092 to 2,552 tokens across iterations, raising concerns about superficial alignment [source:arxiv:2401.10020].
- **Mitigations**: Proposed solutions include:
  - **Reward Model Ensembling**: Using multiple AI feedback models to reduce variance and bias [source:arxiv:2212.08073].
  - **Length Regularization**: Penalizing response length during training [source:arxiv:2401.10020].
  - **Bias Calibration**: Adjusting LLM-as-a-Judge prompts to mitigate position and verbosity biases (e.g., swapping response order to achieve 65–66% consistency in GPT-4) [source:arxiv:2306.05685].
- **Unresolved**: None of these mitigations fully address the root cause—AI feedback models may inherently favor superficial proxies for quality (e.g., length, fluency) over genuine alignment.

### 2. Safety and Harmlessness
- **Problem**: RLAIF methods have not been rigorously evaluated for safety alignment. Constitutional AI retains human feedback for helpfulness, and self-rewarding models lack safety-specific training [source:arxiv:2212.08073][source:arxiv:2401.10020].
- **Evidence**: In Constitutional AI, critiques sometimes contain inaccurate or overstated criticisms, though revisions generally mitigate initial harms [source:arxiv:2212.08073]. Self-rewarding models have not been evaluated for jailbreak resistance or harmful content generation.
- **Mitigations**: Proposed solutions include:
  - **Safety-Specific Constitutions**: Extending Constitutional AI with principles targeting safety and robustness [source:arxiv:2212.08073].
  - **Red-Teaming**: Iterative red-teaming and feedback to identify and mitigate harmful behaviors.
  - **Hybrid Feedback**: Combining AI feedback with human safety labels.
- **Unresolved**: It is unclear whether AI feedback can reliably identify and mitigate harmful behaviors without human oversight. The risk of **automating harmfulness**—where the AI feedback model itself is misaligned—remains a critical concern.

### 3. Distribution Shift and Generalization
- **Problem**: Offline RLAIF methods (e.g., DPO with AI-generated preferences) suffer from distribution shift, where the policy’s generations diverge from the feedback model’s training data. Online AI feedback (OAIF) mitigates this but has not been tested for out-of-distribution prompts or evolving human values [source:arxiv:2402.04792].
- **Evidence**: Offline DPO degrades in performance when the policy’s generations diverge from the SFT baseline, whereas OAIF maintains performance by evaluating on-policy generations [source:arxiv:2402.04792].
- **Mitigations**: Proposed solutions include:
  - **Online Feedback**: OAIF and similar methods to ensure on-policy evaluation [source:arxiv:2402.04792].
  - **Data Augmentation**: Expanding the prompt distribution during training to improve generalization.
- **Unresolved**: The effectiveness of online AI feedback for out-of-distribution prompts or rapidly evolving human values (e.g., emerging social norms) is unknown. Additionally, the **sample efficiency** of OAIF remains a bottleneck for real-time human feedback integration (requiring ~256,000 samples for ~2,000 training steps) [source:arxiv:2402.04792].

### 4. Scalability and Compute Costs
- **Problem**: While RLAIF reduces the cost of human annotation, it introduces new computational overheads. Self-rewarding models require generating and evaluating multiple candidate responses per prompt, and online AI feedback necessitates frequent LLM-as-a-Judge calls [source:arxiv:2401.10020][source:arxiv:2402.04792].
- **Evidence**: Self-rewarding models generate 4 candidate responses per prompt and evaluate them 3 times, increasing compute costs by an order of magnitude [source:arxiv:2401.10020]. OAIF requires ~256,000 samples for ~2,000 training steps, limiting real-time applications [source:arxiv:2402.04792].
- **Mitigations**: Proposed solutions include:
  - **Efficient Sampling**: Reducing the number of candidate responses or evaluations per prompt.
  - **Distilled Feedback Models**: Training smaller, specialized feedback models to reduce inference costs.
- **Unresolved**: The trade-off between compute costs and alignment quality remains poorly characterized. It is unclear whether RLAIF can scale to larger models (e.g., 100B+ parameters) without prohibitive computational overhead.

### 5. Alignment Tax
- **Problem**: RLAIF methods may incur an **alignment tax**, where improvements in preference alignment come at the cost of degraded performance on standard benchmarks (e.g., MMLU, GSM8K) [source:arxiv:2401.10020].
- **Evidence**: Self-rewarding models show slight regressions on ARC-Challenge, HellaSwag, and other NLP benchmarks compared to the base model [source:arxiv:2401.10020]. This is attributed to the narrow distribution of the OpenAssistant seed data.
- **Mitigations**: Proposed solutions include:
  - **Multi-Objective Training**: Jointly optimizing for alignment and benchmark performance.
  - **Diverse Seed Data**: Expanding the seed dataset to cover a broader range of tasks.
- **Unresolved**: The alignment tax is not well-understood, and it is unclear whether it can be eliminated or merely mitigated. The trade-off between alignment and generality remains a fundamental challenge.

---

## Key Takeaways

- **RLAIF replaces or augments human feedback with AI-generated labels**, enabling scalable, autonomous alignment. Key methods include Constitutional AI (CAI), self-rewarding models, and online AI feedback (OAIF).
- **Constitutional AI** uses natural language principles to guide AI critiques and revisions, reducing evasiveness while maintaining helpfulness. It retains human feedback for helpfulness, limiting full autonomy [source:arxiv:2212.08073].
- **Self-rewarding models** iteratively generate and evaluate their own responses, eliminating the fixed reward model bottleneck. They achieve state-of-the-art results on instruction-following benchmarks (e.g., AlpacaEval 2.0 win rate of 20.44% vs. GPT-4 Turbo) but exhibit reward hacking (e.g., length bias) [source:arxiv:2401.10020].
- **Online AI feedback (OAIF)** injects on-policy AI feedback into DAP methods, mitigating distribution shift and improving performance over offline methods (66% win rate over offline DPO) [source:arxiv:2402.04792].
- **RLAIF is rising rapidly** as a default technique for scalable alignment, but its long-term trajectory depends on resolving reward hacking, safety, distribution shift, and the alignment tax.
- **Key challenges** include:
  - Reward hacking and over-optimization (e.g., length bias in self-rewarding models, 8.7% verbosity bias failure rate in GPT-4).
  - Safety alignment (e.g., lack of safety-specific training in self-rewarding models).
  - Distribution shift and generalization (e.g., OAIF’s untested performance on out-of-distribution prompts).
  - Scalability and compute costs (e.g., ~256,000 samples required for OAIF training).
  - Alignment tax (e.g., regressions on ARC-Challenge and HellaSwag in self-rewarding models).

---

## Related Topics

- [[DPO and preference optimization]](dpo-and-preference-optimization.md): Direct Preference Optimization (DPO) and its variants, which eliminate the need for explicit reward modeling in preference alignment.
- [[Reward modeling for LLMs]](reward-modeling.md): The design and training of reward models for RLHF and RLAIF.
- [[RL for LLMs — overview]](rl-for-llms-overview.md): A high-level overview of reinforcement learning techniques for LLM alignment.
- [[The RLHF/PPO pipeline]](rlhf-ppo-pipeline.md): The standard RLHF pipeline using Proximal Policy Optimization (PPO).
- [[Reward hacking in RLHF]](reward-hacking.md): The phenomenon of policies exploiting flaws in reward models to achieve high scores without genuine alignment.
- [[Reward model over-optimization]](reward-model-overoptimization.md): The degradation of policy performance due to excessive optimization against a fixed reward model.
- [[LLM-as-judge]](llm-as-judge.md): The use of LLMs as automated evaluators for preference alignment, including biases and mitigation strategies.
- [[Alignment and win-rate evals]](alignment-and-winrate-evals.md): Evaluation methodologies for measuring alignment quality, including win-rate comparisons.
- [[Judging bias and contamination]](judging-bias-and-contamination.md): Biases and contamination risks in LLM-as-a-Judge evaluations (e.g., position bias, verbosity bias).
- [[Self-improvement and self-play RL]](self-improvement-and-self-play.md): Methods for autonomous improvement in reinforcement learning, including self-play and self-rewarding.

---

##

## References
- [source:arxiv:2212.08073] [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [source:arxiv:2302.10845] [Strengthening Language Model Alignment Using the Constitutional AI Framework](https://arxiv.org/abs/2302.10845)
- [source:arxiv:2401.10020] [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2402.04792] [Judging LLM-as-a-Judge: Wins and Losses](https://arxiv.org/abs/2402.04792)
- [source:arxiv:1706.03741] [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- [source:arxiv:2306.05685] [LLM as a Judge](https://arxiv.org/abs/2306.05685)
- [source:arxiv:2401.10020] [Self-Rewarding LLMs: A Framework for Aligning Language Models via Self-Generated Feedback](https://arxiv.org/abs/2401.10020)
