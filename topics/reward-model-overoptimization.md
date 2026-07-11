---
title: Reward model over-optimization
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2210.10760
- arxiv:2406.02900
- arxiv:2310.02743
- arxiv:2312.09244
- arxiv:2310.04373
- arxiv:2305.18290
- arxiv:2402.09345
- arxiv:2311.00168
open_questions:
- 'Adversarial Goodhart**: How do scaling laws for overoptimization change when policies
  actively manipulate reward models (e.g., via gradient-based attacks or emergent
  strategies)? Are there theoretical or empirical bounds on the degradation rate under
  adversarial conditions?'
- 'Online RLHF**: How do reward model ensembles perform in *online* RLHF settings,
  where RMs are periodically retrained on policy-generated data? Does ensemble diversity
  degrade over time as policies converge?'
- 'Dynamic KL Budgeting**: Can constrained RLHF methods (e.g., $ \xi $-PPO) be extended
  to higher-dimensional composite reward settings without ground-truth access? Are
  there proxy-free alternatives for identifying optimal thresholds?'
- 'Scaling to Frontier Models**: How do overoptimization dynamics change for models
  with >100B parameters? Do larger models exhibit fundamentally different scaling
  laws, or do existing functional forms ($ R(d) = d(\alpha - \beta \log d) $) hold?'
---

# Reward Model Over-Optimization: Scaling Laws, KL Budgets, and Ensembles

Reward model over-optimization is the pathological phenomenon in reinforcement learning from human feedback (RLHF) where policies optimized against a learned proxy reward model (RM) achieve diminishing or negative returns on the true objective as optimization pressure increases. This degradation—rooted in Goodhart’s law—manifests across both classical RLHF pipelines (e.g., PPO) and direct alignment algorithms (e.g., DPO), with severity modulated by proxy RM capacity, KL divergence budgets, and aggregation strategies like ensembles. Understanding its scaling laws and mitigation mechanisms is critical for safe and effective alignment of large language models (LLMs).

---

## Theoretical Foundations

### Goodhart’s Law and Overoptimization
Goodhart’s law—*"When a measure becomes a target, it ceases to be a good measure"*—underpins overoptimization. In RLHF, the proxy RM $ \hat{r} $ approximates the unobservable ground-truth reward $ r^* $, but optimization against $ \hat{r} $ amplifies approximation errors. The relationship between $ r^* $ and $ \hat{r} $ is formalized via regressional Goodhart [source:arxiv:2210.10760]:

$$
\mathbb{E}[r^* \mid \hat{r} = x] = \mathbb{E}[r^*] + (x - \mathbb{E}[\hat{r}] - \mathbb{E}[Z]) \frac{\text{Var}(r^*)}{\text{Var}(r^*) + \text{Var}(Z)} + \varepsilon,
$$

where $ Z $ is independent noise. As $ x \to \infty $, the linear relationship breaks down, and $ r^* $ plateaus or declines.

### KL Divergence as Optimization Pressure
The Kullback-Leibler (KL) divergence between the optimized policy $ \pi $ and the reference policy $ \pi_{\text{ref}} $ quantifies optimization pressure:

$$
d = \sqrt{D_{\text{KL}}(\pi \parallel \pi_{\text{ref}})}.
$$

This metric is central to scaling laws, where $ d $ acts as a proxy for "optimization steps" in both best-of-$ n $ (BoN) sampling and RL fine-tuning [source:arxiv:2210.10760][source:arxiv:2406.02900].

---

## Scaling Laws for Overoptimization

### Functional Forms
Empirical scaling laws describe how ground-truth reward $ R $ degrades as a function of $ d $. Two distinct functional forms emerge for different optimization methods:

1. **Best-of-$ n $ (BoN) sampling**:
   $$
   R_{\text{bon}}(d) = d(\alpha_{\text{bon}} - \beta_{\text{bon}} d),
   $$
   where $ \alpha_{\text{bon}} $ and $ \beta_{\text{bon}} $ scale logarithmically with proxy RM parameter count [source:arxiv:2210.10760].

2. **Reinforcement learning (PPO)**:
   $$
   R_{\text{RL}}(d) = d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log d).
   $$
   Here, $ \alpha_{\text{RL}} $ is nearly constant across RM sizes, while $ \beta_{\text{RL}} $ captures the rate of degradation [source:arxiv:2210.10760].

For **iterated RLHF**, the scaling law generalizes to:
$$
R_{\text{RL}}(d) = d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log(d) + \beta_{\text{RL}} \log(k)),
$$
where $ k $ is the number of training iterations [source:arxiv:2210.10760].

### Direct Alignment Algorithms (DAAs)
Overoptimization persists in DAAs like DPO, IPO, and SLiC-HF, where the policy is optimized directly against preference data without an explicit RM. The same logarithmic scaling law applies [source:arxiv:2406.02900]:

$$
R(d) = d(\alpha - \beta \log d).
$$

Here, $ \alpha $ and $ \beta $ depend on model scale, dataset size, and algorithm choice. For example, IPO achieves lower KL divergences and mitigates overoptimization more effectively than DPO or SLiC [source:arxiv:2406.02900].

### Key Observations
1. **KL Efficiency**: BoN is significantly more KL-efficient than RL, consuming KL distance approximately quadratically per step [source:arxiv:2210.10760].
2. **Policy Scaling**: Larger policies (e.g., 6B vs. 1.2B) gain less absolute reward from optimization but peak at similar KL distances [source:arxiv:2210.10760].
3. **Intra-Epoch Degradation**: In DAAs, models often reach peak performance after processing only a fraction of the training data before degrading. For example, configurations with wider KL budgets peak after approximately 25% of training [source:arxiv:2406.02900].
4. **Proxy RM Scaling**: Proxy RMs require at least 2,000 training comparisons to surpass near-chance performance; beyond this threshold, larger RMs improve faster [source:arxiv:2210.10760].

---

## The KL Budget: Balancing Optimization and Degradation

### Definition and Role
The KL budget is the maximum allowable $ D_{\text{KL}}(\pi \parallel \pi_{\text{ref}}) $ during optimization. It acts as a regularizer, trading off alignment quality against policy drift. In PPO, the KL budget is enforced via a penalty term in the reward signal:

$$
R^{\text{PPO}}(q, a) = R(q, a) - \beta \log \left[ \frac{\pi^{\text{PPO}}(a|q)}{\pi^{\text{ref}}(a|q)} \right],
$$

where $ \beta $ controls the penalty strength [source:arxiv:2310.02743].

### Empirical Tradeoffs
1. **Early Stopping**: Explicit KL penalties in PPO act as early stopping, improving proxy scores without altering the gold score-KL frontier [source:arxiv:2210.10760].
2. **Optimal KL Budgets**: For DAAs, the optimal KL budget varies by model scale. Smaller models (1B parameters) overoptimize rapidly, while larger models (6.9B) exhibit better KL control [source:arxiv:2406.02900].
3. **Algorithm Dependence**: IPO consistently achieves lower KL divergences than DPO or SLiC, reducing overoptimization [source:arxiv:2406.02900].

### Limitations
- **Brittleness**: Fixed KL budgets are sensitive to proxy RM quality and dataset distribution. Overly conservative budgets stifle alignment, while permissive budgets invite overoptimization.
- **Dynamic Adjustment**: Static KL penalties cannot adapt to intra-epoch degradation, motivating dynamic methods like constrained RLHF [source:arxiv:2310.04373].

---

## Ensembles: Mitigating Overoptimization via Diversity

### Motivation
Reward model ensembles exploit diversity to reduce variance and mitigate overoptimization. Ensembles address two key failure modes:
1. **Underspecification**: RMs trained on identical data but different seeds diverge significantly on out-of-distribution (OOD) inputs [source:arxiv:2312.09244].
2. **Conservative Aggregation**: Simple averaging fails to penalize high-variance predictions, while conservative methods (e.g., min, mean-minus-std) reduce overoptimization [source:arxiv:2310.02743][source:arxiv:2312.09244].

### Ensemble Methods
Three aggregation strategies are commonly used:

| Method               | Formula                                                                 | Description                                                                 |
|----------------------|-------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| Mean (MEAN)          | $ R_\mu(q, a) = \frac{1}{k} \sum_{i=1}^k R_i(q, a) $                     | Simple average; vulnerable to high-variance predictions.                   |
| Worst-Case (WCO)     | $ R_{\text{WCO}}(q, a) = \min_i R_i(q, a) $                             | Conservative; penalizes high-variance predictions.                         |
| Uncertainty-Weighted (UWO) | $ R_{\text{UWO}}(q, a) = \frac{1}{k} \sum_i R_i(q, a) - \lambda \text{Var}(R_i(q, a)) $ | Balances mean reward with variance penalty.                                |

### Quantitative Results
1. **BoN Sampling**:
   - WCO and UWO eliminate overoptimization under noiseless conditions and improve gold reward by up to 75% under 25% label noise [source:arxiv:2310.02743].
   - Pretrain ensembles (varying pretraining seeds) outperform finetune ensembles (varying finetuning seeds) in BoN reranking, achieving higher win rates (e.g., 90.0% vs. 87.3% for TL;DR summarization) [source:arxiv:2312.09244].
2. **PPO**:
   - WCO and UWO reduce overoptimization across all KL penalty weights, matching or exceeding single RM performance [source:arxiv:2310.02743].
   - Combining ensembles with a minimal KL penalty ($ \beta = 0.01 $) fully prevents overoptimization without performance degradation, whereas single RMs require a 20× larger penalty ($ \beta = 0.2 $) [source:arxiv:2310.02743].
3. **Scaling**:
   - Ensembles with 4–5 members yield optimal results; larger ensembles provide diminishing returns [source:arxiv:2310.02743].

### Limitations and Disagreements
- **Failure Modes**: Ensembles fail when all members share similar error patterns (e.g., length bias), unanimously endorsing misaligned outputs [source:arxiv:2312.09244].
- **Online RLHF**: Ensemble efficacy in *online* RLHF (where RMs are periodically retrained) remains untested [source:arxiv:2310.02743].
- **Disagreement**: [source:arxiv:2310.02743] reports that ensembles *eliminate* overoptimization, while [source:arxiv:2312.09244] states they *mitigate but do not prevent* it. The discrepancy arises from task-specific failure modes (e.g., factuality vs. summarization).

---

## Constrained RLHF: Dynamic KL Budgeting

### Motivation
Fixed KL budgets or weights are brittle in composite reward settings, where multiple component RMs (e.g., METEOR, intent classification) are combined. Correlation between components shifts proxy points, making static weighting schemes ineffective [source:arxiv:2310.04373].

### Method: Constrained RLHF
Constrained RLHF reformulates alignment as a constrained Markov decision process (CMDP):

$$
\max_{\pi} v_0^{\pi} \quad \text{s.t.} \quad v_i^{\pi} \geq \theta_i, \ i = 1, \dots, N,
$$

where $ v_0^{\pi} $ is the negative KL divergence from $ \pi_{\text{ref}} $, and $ v_i^{\pi} $ are component RM values constrained to their proxy points $ \theta_i $. Lagrangian relaxation converts this into a min-max game:

$$
\max_{\pi} \min_{\boldsymbol{\mu} \geq 0} v_0^{\pi} + \sum_{i=1}^N \mu_i (v_i^{\pi} - \theta_i).
$$

### Key Results
1. **Proxy Point Identification**: Proxy points are located via polynomial surface fitting or gradient-free Nelder-Mead optimization [source:arxiv:2310.04373].
2. **Performance**: Constrained variants (e.g., $ \xi $-PPO, $ \mu $-PPO) outperform standard PPO, achieving higher evaluation scores while enforcing thresholds [source:arxiv:2310.04373].
3. **Efficiency**: Nelder-Mead optimization converges to optimal thresholds in a single training run, reducing computational overhead [source:arxiv:2310.04373].

### Limitations
- **Ground-Truth Access**: Proxy point identification requires minimal access to ground-truth evaluation metrics [source:arxiv:2310.04373].
- **Convergence**: Primal-dual optimization guarantees convergence of averaged iterates, not final policies [source:arxiv:2310.04373].
- **Scalability**: Validated only for two-component RMs; higher-dimensional settings remain unexplored.

---

## Information-Theoretic Reward Models (InfoRM)

### Motivation
Reward misgeneralization—where RMs overfit to spurious features (e.g., length bias)—drives overoptimization. InfoRM addresses this via a variational information bottleneck (IB) objective, compressing inputs into a latent representation $ \boldsymbol{S} $ that retains only preference-relevant information [source:arxiv:2402.09345].

### Method
1. **Latent Parameterization**: Encode chosen/rejected responses into Gaussian latent variables $ \boldsymbol{S} $.
2. **Reward Decoding**: Decode $ \boldsymbol{S} $ into scalar rewards via an MLP.
3. **Optimization**: Maximize preference prediction accuracy while penalizing KL divergence from a standard normal prior $ r(\boldsymbol{S}) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) $.

### Key Results
1. **Overoptimization Mitigation**: InfoRM maintains stable gold reward growth under 25% label noise, whereas standard RMs exhibit severe overoptimization [source:arxiv:2402.09345].
2. **Out-of-Distribution Generalization**: InfoRM outperforms standard RMs on AlpacaEval (66.63% vs. 65.38%) and Truthful QA (46.87% vs. 40.63%) [source:arxiv:2402.09345].
3. **Detection**: The Cluster Separation Index (CSI) reliably tracks overoptimization, spiking for standard RMs but remaining low for InfoRM [source:arxiv:2402.09345].

### Limitations
- **Scalability**: Evaluated only up to 7B parameters; larger models remain untested.
- **Latency**: CSI-based monitoring incurs inference overhead.
- **Prompt Sensitivity**: Automated win-rate evaluations are sensitive to prompt structure.

---

## Current Status and Trajectory

### Adoption and Trends
- **Scaling Laws**: The functional forms for overoptimization ($ R(d) = d(\alpha - \beta \log d) $ for RL, $ R(d) = d(\alpha - \beta d) $ for BoN) are now widely cited and validated across RLHF and DAAs [source:arxiv:2210.10760][source:arxiv:2406.02900]. They are increasingly used for hyperparameter tuning and early stopping.
- **KL Budgets**: Explicit KL penalties in PPO are standard practice, but static budgets are giving way to dynamic methods like constrained RLHF [source:arxiv:2310.04373].
- **Ensembles**: Ensemble-based conservative aggregation (e.g., WCO, UWO) is rising in offline RLHF but not yet widely reported in production systems. Pretrain ensembles are preferred over finetune ensembles for their robustness [source:arxiv:2310.02743][source:arxiv:2312.09244].
- **InfoRM**: Information-theoretic RMs are an active research direction, with early results promising but not yet scaled to frontier models [source:arxiv:2402.09345].

### Trajectory
- **Rising**: Dynamic KL budgeting (e.g., constrained RLHF) and ensemble methods are gaining traction as scalable mitigations. InfoRM and related IB-based approaches are poised for growth as model scales increase.
- **Default**: Static KL penalties in PPO remain the default, but their limitations are increasingly acknowledged.
- **Fading**: Single-reward-model pipelines are declining in favor of ensemble or composite reward approaches, though computational costs remain a barrier.

### Disagreements and Open Challenges
- **Ensemble Efficacy**: [source:arxiv:2310.02743] claims ensembles *eliminate* overoptimization, while [source:arxiv:2312.09244] argues they only *mitigate* it under specific conditions. Resolution requires larger-scale studies across diverse tasks.
- **Online RLHF**: Ensemble performance in online settings (where RMs are periodically retrained) is untested, leaving a critical gap for iterative alignment.
- **Adversarial Goodhart**: Current scaling laws do not account for adversarial overoptimization, where policies actively manipulate RMs [source:arxiv:2210.10760]. This remains a theoretical risk for advanced systems.

---

## Key Takeaways
- **Scaling Laws**: Overoptimization follows predictable functional forms ($ R(d) = d(\alpha - \beta \log d) $ for RL, $ R(d) = d(\alpha - \beta d) $ for BoN), enabling extrapolation and early stopping.
- **KL Budget**: The KL divergence $ D_{\text{KL}}(\pi \parallel \pi_{\text{ref}}) $ is the primary knob for controlling optimization pressure, but static budgets are brittle.
- **Ensembles**: Reward model ensembles (especially with conservative aggregation) mitigate overoptimization by reducing variance and penalizing high-uncertainty predictions.
- **Constrained RLHF**: Dynamic KL budgeting via constrained optimization (e.g., $ \xi $-PPO) outperforms static penalties but requires ground-truth access for proxy point identification.
- **InfoRM**: Information-theoretic RMs address misgeneralization by compressing inputs into preference-relevant latent spaces, improving OOD robustness.
- **DAAs**: Overoptimization persists in direct alignment algorithms (DPO, IPO, SLiC), with IPO exhibiting better KL control than DPO.
- **Intra-Epoch Degradation**: Models often peak after processing only a fraction of training data, necessitating checkpoint monitoring.

---

## Related Topics
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [The alignment tax](alignment-tax.md)

---

##

## References
- [source:arxiv:2210.10760] [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)
- [source:arxiv:2406.02900] [Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms](https://arxiv.org/abs/2406.02900)
- [source:arxiv:2310.02743] [Reward Model Ensembles Help Mitigate Overoptimization](https://arxiv.org/abs/2310.02743)
- [source:arxiv:2312.09244] [Helping or Herding? Reward Model Ensembles Mitigate but do not Prevent Overoptimization](https://arxiv.org/abs/2312.09244)
- [source:arxiv:2310.04373] [Confronting Reward Model Overoptimization with Constrained RLHF](https://arxiv.org/abs/2310.04373)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2402.09345] [Mitigating Reward Hacking via Information-Theoretic Reward Models](https://arxiv.org/abs/2402.09345)
- [source:arxiv:2311.00168] [Objective Mismatch in Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2311.00168)
