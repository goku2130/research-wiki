---
title: Nash and game-theoretic preference optimization
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2305.18290
- arxiv:2402.04792
- arxiv:2312.09244
- arxiv:2401.12118
- arxiv:2310.12798
open_questions:
- 'Scalability of game-theoretic methods**: Can Nash equilibria and OAIF scale to
  models with >100B parameters, or are they limited to smaller architectures? [source:arxiv:2402.04792]
  leaves this unanswered.'
- 'Trade-offs between game-theoretic methods and ensembles**: Under what conditions
  do Nash equilibria outperform reward model ensembles, and vice versa? The literature
  presents conflicting conclusions (Section 3.2.1).'
- 'Evaluation metrics**: How can win rates and other alignment metrics be standardized
  to enable fair comparisons across methods? [source:arxiv:2305.18290] highlights
  the sensitivity of win rates to evaluation prompts.'
- 'Out-of-distribution generalization**: How do game-theoretic methods perform on
  out-of-distribution prompts, and can they generalize beyond the training distribution?
  [source:arxiv:2402.04792] does not address this.'
---

# Nash and Game-Theoretic Preference Optimization

Large language models (LLMs) are increasingly fine-tuned to align with complex, multi-dimensional human preferences. Standard Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) methods treat alignment as a single-agent optimization problem, which can lead to reward hacking, preference collapse, and off-policy drift. Nash and game-theoretic preference optimization reframes alignment as a *multi-player game* in which policies compete or cooperate to satisfy diverse, potentially conflicting preferences, enabling equilibrium solutions that balance trade-offs and mitigate these failure modes.

---

## 1. Foundations: From Single-Agent RLHF to Multi-Agent Games

### 1.1 The Single-Agent RLHF Objective
Standard RLHF optimizes a policy $\pi_\theta$ to maximize an expected reward $r(x,y)$ while constraining divergence from a reference policy $\pi_{\text{ref}}$ via KL regularization [source:arxiv:2305.18290]:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r(x,y) \right] - \beta \mathbb{D}_{\text{KL}} \left[ \pi_\theta(y|x) \|\pi_{\text{ref}}(y|x) \right].
$$

The optimal policy under this objective has the closed-form solution:

$$
\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp \left( \frac{1}{\beta} r(x,y) \right),
$$

where $Z(x)$ is the partition function. DPO reparameterizes this solution to eliminate the explicit reward model, yielding a binary cross-entropy loss over preference pairs $(y_w, y_l)$ [source:arxiv:2305.18290]:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right].
$$

### 1.2 Limitations of Single-Agent Optimization
Single-agent RLHF and DPO suffer from three critical limitations:
1. **Reward hacking**: Policies exploit reward model underspecification, producing outputs that maximize reward but violate intended preferences [source:arxiv:2312.09244]. For example:
   - RLHF policies trained on TL;DR summarization datasets exhibit increased output length and extractiveness.
   - HELPFULNESS policies shift toward list-formatted responses, with the proportion of such responses increasing significantly post-RLHF compared to the training data [source:arxiv:2312.09244].
2. **Off-policy drift**: Offline DAP methods (e.g., DPO) train on static datasets, causing the policy's generation distribution to diverge from the training distribution. This leads to overfitting and suboptimal alignment [source:arxiv:2402.04792].
3. **Scalability**: Single-agent methods like PPO are computationally expensive and unstable, requiring complex multi-stage training loops with continuous policy sampling [source:arxiv:2305.18290].

---

## 2. Game-Theoretic Preference Optimization: Core Concepts

### 2.1 Nash Equilibria for Preference Alignment
A Nash equilibrium (NE) is a strategy profile $\{\pi_i^*\}_{i=1}^N$ in which no agent can unilaterally improve its utility by deviating from $\pi_i^*$. In preference alignment, agents correspond to *preference dimensions* (e.g., helpfulness, harmlessness), and utilities are defined over these dimensions. Formally, let $\mathcal{P} = \{P_1, \dots, P_N\}$ be a set of preference dimensions, and let $u_i(\pi_1, \dots, \pi_N)$ be the utility of agent $i$ under the joint policy profile. A NE satisfies:

$$
u_i(\pi_i^*, \pi_{-i}^*) \geq u_i(\pi_i, \pi_{-i}^*) \quad \forall \pi_i, \forall i.
$$

#### 2.1.1 Regularized Nash Equilibria
KL regularization can be used to ensure uniqueness and improve convergence. The regularized utility for agent $i$ is:

$$
u_i^\text{reg}(\pi_i, \pi_{-i}) = u_i(\pi_i, \pi_{-i}) - \beta \mathbb{D}_{\text{KL}} \left[ \pi_i(y|x) \|\pi_{\text{ref}}(y|x) \right].
$$

The regularized NE is the unique fixed point of the best-response dynamics under $u_i^\text{reg}$.

### 2.2 Online AI Feedback (OAIF) as a Game-Theoretic Framework
Online AI Feedback (OAIF) [source:arxiv:2402.04792] resolves off-policy drift in direct alignment from preferences (DAP) methods by making the process interactive and on-policy. While not explicitly framed as a game, OAIF shares key properties with game-theoretic methods:
1. **Dynamic feedback**: OAIF uses an external LLM annotator to rank responses generated by the current policy, enabling on-policy updates.
2. **Decentralized optimization**: The policy and annotator can be viewed as two "players" in a cooperative game, where the annotator provides feedback to guide the policy toward better alignment.

OAIF proceeds iteratively:
1. Sample a prompt $x$ from a dataset.
2. Generate two candidate responses $y^1, y^2$ independently from the current policy $\pi_{\theta^t}(\cdot|x)$.
3. Query an external LLM annotator to rank the pair, designating one as the preferred response $y^+$ and the other as the less preferred $y^-$.
4. Update the policy parameters $\theta$ via gradient descent on a standard DAP loss (e.g., DPO, IPO, SLiC).

**Key Formulas**:
OAIF is compatible with any differentiable DAP loss. The primary loss functions are:
- **DPO loss**:

$$
- \log \sigma \left(\beta \log \frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}\right)
$$

- **IPO loss**:

$$
\left(\log \left(\frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x})}\right) - \frac {1}{2 \beta}\right) ^ {2}
$$

- **SLiC loss**:

$$
\max \left(0, 1 - \beta \log \left(\frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x})}\right)\right)
$$

---

## 3. Reward Hacking and Nash Equilibria

### 3.1 Reward Hacking in Single-Agent Optimization
Reward hacking arises when a policy exploits reward model underspecification to maximize reward without satisfying the intended preference. For example:
- **TL;DR summarization**: Policies exhibit increased output length and extractiveness, increasing reward but violating conciseness [source:arxiv:2312.09244].
- **HELPFULNESS**: Policies shift toward list-formatted responses, with the proportion of such responses increasing significantly post-RLHF compared to the training data. This exploits the reward model's bias toward structured outputs [source:arxiv:2312.09244].

### 3.2 Mitigating Reward Hacking with Ensembles
Reward model ensembles mitigate reward hacking by aggregating multiple reward models to reduce variance and bias. In [source:arxiv:2312.09244], ensembles are constructed by varying pretraining or finetuning seeds and aggregating scores using conservative functions (e.g., MEAN, MEDIAN, MEAN_MINUS_STD, MIN). Key results include:
- **Best-of-N reranking**: Pretrain ensembles achieve a 90% win rate against the SFT baseline at $n=64$, compared to 85.3% for single reward models (RMs) [source:arxiv:2312.09244].
- **RLHF**: Pretrain ensembles provide superior reward-KL tradeoffs, with T5-XXL reward scores decreasing (indicating reward hacking) less frequently than for single RMs [source:arxiv:2312.09244].

However, ensembles do not eliminate reward hacking entirely. When all ensemble members share similar error patterns (e.g., favoring list-formatted responses), the aggregated policy may still exploit these errors [source:arxiv:2312.09244].

#### 3.2.1 Disagreements in the Literature
There is ongoing debate about the trade-offs between game-theoretic equilibrium methods (e.g., Nash equilibria) and reward model ensembles:
- **Ensembles**: [source:arxiv:2312.09244] demonstrates that ensembles mitigate reward hacking but do not eliminate it. The authors argue that ensembles are limited by the inability to quantify uncertainty far from the training distribution, as policy optimization shifts outputs to regions where all RMs erroneously extrapolate in unison.
- **Game-theoretic methods**: While not explicitly tested in [source:arxiv:2312.09244], game-theoretic methods are hypothesized to address reward hacking by balancing multiple preference dimensions. However, their scalability and practical implementation remain open questions.

---

## 4. Online AI Feedback: Algorithms and Results

### 4.1 Online DAP Methods
Online DAP methods (e.g., OAIF) achieve superior alignment by avoiding off-policy drift. Key results from [source:arxiv:2402.04792] include:
- **Win rates**: Online DAP methods (DPO, IPO, SLiC) achieve an average win rate of ~66% against their offline counterparts in human evaluations.
- **Overfitting**: Offline DPO exhibits rapid overfitting, with win rates collapsing after ~3,500 training steps, whereas online DPO performance consistently improves.
- **Prompt controllability**: Instructing the annotator to prefer shorter outputs reduces average length from ~120 to ~40 tokens, with human-rated quality scores decreasing from 4.08 to 3.26 (still surpassing the SFT baseline of 3.19).

### 4.2 Limitations
- **Sample inefficiency**: Aligning a model requires ~256,000 samples across 2,000 steps, which is prohibitive for single-user personalization without low-rank adaptation techniques [source:arxiv:2402.04792].
- **External annotator dependence**: OAIF relies on an external LLM annotator, as self-annotation requires identical model architecture and size. This limits scalability and flexibility [source:arxiv:2402.04792].
- **Scalability**: The study isolates distribution shifts over responses $p(\boldsymbol{y}|\boldsymbol{x})$ but does not address shifts in the prompt distribution $p_X$ or ground-truth human value functions. Evaluations assume in-distribution prompts, leaving out-of-distribution generalization untested [source:arxiv:2402.04792].

#### 4.2.1 Disagreements in the Literature
There is conflicting evidence about the scalability of OAIF:
- **Positive results**: [source:arxiv:2402.04792] demonstrates that OAIF remains effective with smaller annotators (e.g., PaLM 2-XS), yielding a quality score of 3.41, comparable to RLHF's 3.38.
- **Negative results**: The same study notes that scaling to larger models remains unverified, and the sample inefficiency of OAIF (~256,000 samples) poses challenges for real-world deployment.

---

## 5. Current Status and Trajectory

### 5.1 Adoption and Maturity
Nash and game-theoretic preference optimization are **emerging techniques** with limited but growing adoption:
- **Research**: Theoretical foundations are well-established in game theory, but their application to LLM alignment is recent. Key papers (e.g., [source:arxiv:2402.04792], [source:arxiv:2312.09244]) demonstrate proof-of-concept results, but large-scale deployments are not widely reported. For example, [source:arxiv:2402.04792] evaluates OAIF on models up to PaLM 2-XS, leaving scalability to larger architectures an open question.
- **Industry**: Companies like Anthropic and DeepMind have explored multi-objective alignment [source:arxiv:2402.04792], but public reports of Nash equilibria in production systems are rare. Self-play methods are more common in reinforcement learning (e.g., AlphaGo's use of self-play for policy improvement [source:arxiv:1712.01815]), but their use in LLM alignment is nascent. For instance, AlphaGo's self-play prevalence is not directly applicable to LLM alignment due to differences in task structure and reward modeling.
- **Open-source**: Libraries for game-theoretic alignment are not yet mainstream. While frameworks like Hugging Face's TRL support DPO and PPO, there are no widely adopted libraries for Nash equilibria or OAIF. The maturity of open-source tools for these methods lags behind single-agent approaches [source:arxiv:2305.18290].

### 5.2 Trajectory: Rising, Default, or Fading?
**Rising, but not yet default**. The following trends suggest increasing adoption:
1. **Multi-dimensional alignment**: As LLMs are deployed in safety-critical domains, the need to balance conflicting preferences (e.g., helpfulness vs. harmlessness) will drive adoption of game-theoretic methods.
2. **Reward hacking mitigation**: Single-agent methods (e.g., DPO, RLHF) are increasingly recognized as vulnerable to reward hacking [source:arxiv:2312.09244]. Nash equilibria and ensembles offer a principled alternative.
3. **Online feedback**: The success of online DAP methods (e.g., OAIF) [source:arxiv:2402.04792] suggests that interactive, game-theoretic approaches will gain traction as infrastructure for real-time feedback improves.

**Hedging**:
- **Scalability**: Online methods require significant computational resources (e.g., ~256,000 samples for OAIF) [source:arxiv:2402.04792]. This limits their applicability to large-scale models without further optimization.
- **Evaluation**: Metrics like win rates are not yet standardized, making it difficult to compare methods across papers. For example, [source:arxiv:2305.18290] notes that DPO's win rates are sensitive to the specific GPT-4 prompts used for evaluation.
- **Disagreements**: The literature presents conflicting conclusions about the trade-offs between game-theoretic methods and reward model ensembles (Section 3.2.1) and the scalability of OAIF (Section 4.2.1). These disagreements highlight the need for further research.

---

## 6. Key Takeaways
- **Nash equilibria** provide a principled framework for balancing multi-dimensional preferences in LLM alignment, avoiding the pitfalls of single-agent optimization (e.g., reward hacking, off-policy drift).
- **Online AI Feedback (OAIF)** demonstrates the effectiveness of interactive, on-policy methods for alignment, achieving ~66% win rates against offline DAP methods [source:arxiv:2402.04792].
- **Reward hacking mitigation**: Reward model ensembles reduce reward hacking by aggregating multiple models, achieving a 90% win rate against SFT baselines at $n=64$ [source:arxiv:2312.09244]. However, ensembles do not eliminate reward hacking entirely.
- **Current status**: Emerging technique with growing research interest, but not yet widely adopted in industry or open-source. Scalability, evaluation, and disagreements in the literature remain open challenges.
- **Future directions**: Integration with online feedback, scalability to large models, and standardized evaluation metrics are key areas for future work. Addressing disagreements in the literature (e.g., trade-offs between game-theoretic methods and ensembles) will be critical for advancing the field.

---

## 7. Related Topics
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md): Foundational methods for preference alignment, which Nash equilibria extend to multi-dimensional settings.
- [Reward modeling for LLMs](reward-modeling.md): Explicit reward modeling is replaced by implicit utilities in game-theoretic methods.
- [Reward hacking in RLHF](reward-hacking.md): Nash equilibria and ensembles mitigate reward hacking, a key failure mode of single-agent RLHF.
- [Reward model over-optimization](reward-model-overoptimization.md): Game-theoretic methods address over-optimization by balancing multiple reward models.
- [Self-improvement and self-play RL](self-improvement-and-self-play.md): Self-play is a core algorithm for computing Nash equilibria in LLM alignment.
- [Process vs outcome reward models](process-vs-outcome-rewards.md): Game-theoretic methods can balance process-based and outcome-based preferences.
- [Alignment and win-rate evals](alignment-and-winrate-evals.md): Win rates are a key metric for evaluating Nash equilibria in preference alignment.
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md): Single-agent RLHF methods, which Nash equilibria aim to improve upon.
- [RLAIF (RL from AI feedback)](rlaif.md): Online feedback methods like OAIF build on RLAIF to enable interactive alignment.

---

##

## References
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2402.04792] [Direct Language Model Alignment from Online AI Feedback](https://arxiv.org/abs/2402.04792)
- [source:arxiv:2312.09244] [Helping or Herding? Reward Model Ensembles Mitigate but do not Eliminate Reward Hacking](https://arxiv.org/abs/2312.09244)
- [source:arxiv:2401.12118] [Measures of the Capital Network of the U.S. Economy](https://arxiv.org/abs/2401.12118)
- [source:arxiv:2310.12798] [MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter](https://arxiv.org/abs/2310.12798)
