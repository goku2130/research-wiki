---
title: Over-optimization and mode collapse
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2305.10193
- arxiv:2305.18290
- arxiv:2311.14722
- arxiv:2305.18723
- arxiv:2203.02155
- arxiv:1707.06347
- arxiv:1802.09455
open_questions:
- 'Mechanistic understanding**: What are the precise optimization dynamics that cause
  mode collapse in autoregressive LLMs? For example, does collapse begin at the token
  level (e.g., a single high-reward token dominating the sequence) or the sequence
  level (e.g., the policy latching onto a single high-reward template)?'
- 'Reward model generalization**: How can reward models be trained to generalize better
  to out-of-distribution inputs, reducing the risk of over-optimization? Are there
  architectures or data augmentation techniques that improve robustness?'
- 'Scalable offline RL**: Can offline RL methods (e.g., DPO) scale to models with
  >100B parameters without suffering from entropy collapse or distribution shift?
  What modifications (e.g., dynamic $ \beta $, auxiliary losses) are needed?'
- 'Evaluation metrics**: What metrics best capture over-optimization and mode collapse
  in LLMs? Current metrics (e.g., entropy, win rates) are noisy or gameable; are there
  principled alternatives (e.g., divergence from pretrained distributions, diversity
  in latent space)?'
---

# Over-optimization and Mode Collapse in Reinforcement Learning for LLMs

Reinforcement learning (RL) fine-tuning of large language models (LLMs) frequently suffers from *over-optimization*—where performance on the training objective degrades despite apparent improvement on the proxy reward—and *mode collapse*, wherein the model’s output distribution collapses to a narrow subset of high-reward but low-diversity behaviors. These pathologies arise when the learned policy exploits flaws in the reward model or optimization process, leading to catastrophic forgetting of pretrained knowledge, reduced linguistic diversity, and misalignment with true human preferences. While KL regularization and early stopping mitigate these effects, they introduce trade-offs between alignment fidelity and model capability, and the underlying mechanisms remain incompletely understood.

---

## ## Theoretical Foundations

### Reward Over-optimization and the Alignment Tax
The standard RLHF objective maximizes a learned reward $ r_\phi(x,y) $ subject to a KL divergence penalty from a reference policy $ \pi_{\text{ref}} $:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x,y) - \beta \mathbb{D}_{\text{KL}}[\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)] \right]. \quad \text{[source:arxiv:2203.02155]}
$$

The KL penalty prevents the policy from deviating excessively from the reference, but as $ \beta $ decreases, the policy may exploit imperfections in $ r_\phi $, a phenomenon termed *reward hacking* [source:arxiv:2203.02155]. Empirically, this manifests as a U-shaped performance curve: initial optimization improves alignment, but continued training degrades performance on held-out tasks (e.g., public NLP benchmarks), a cost termed the *alignment tax* [source:arxiv:2203.02155].

The optimal policy under this objective is:

$$
\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r_\phi(x,y)\right), \quad \text{where } Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r_\phi(x,y)\right). \quad \text{[source:arxiv:2305.18290]}
$$

This closed-form solution reveals that the policy’s support is determined by the product of the reference policy and an exponential tilt toward high-reward outputs. As $ \beta \to 0 $, the policy concentrates mass on the mode of $ r_\phi $, leading to *distributional collapse*—a form of mode collapse where the output distribution becomes unimodal and low-entropy [source:arxiv:2305.18290].

### Mode Collapse in Policy Optimization
Mode collapse occurs when the policy’s entropy $ \mathcal{H}[\pi_\theta(y|x)] $ decreases monotonically during training, and the policy’s output distribution $ \pi_\theta(y|x) $ converges to a delta function or a small set of high-reward sequences. In LLMs, this is exacerbated by:
1. **Autoregressive generation**: The policy’s output is a product of conditional probabilities, so errors compound multiplicatively. A single high-reward token can dominate the sequence, suppressing diversity.
2. **Sparse reward signals**: Human preferences are often binary (e.g., "preferred" vs. "dispreferred"), providing little gradient signal for intermediate-quality outputs. This sparsity encourages the policy to latch onto a single high-reward mode [source:arxiv:2305.18290].
3. **Reward model overfitting**: Reward models $ r_\phi $ are typically trained on limited preference data and may generalize poorly. A policy that exploits these flaws (e.g., by generating verbose or sycophantic responses) can achieve high reward while collapsing to a narrow distribution [source:arxiv:2203.02155].

Theoretical work in RL connects mode collapse to the *exploration-exploitation trade-off*. In the absence of sufficient exploration (e.g., due to KL constraints or deterministic sampling), the policy may fail to discover diverse high-reward behaviors, converging instead to a local optimum [source:arxiv:1707.06347].

---

## ## Empirical Manifestations

### Diversity Loss in RLHF
Diversity loss is quantified via:
1. **Token-level entropy**: The average entropy of the policy’s next-token distribution, $ \mathbb{E}_{x,y \sim \pi_\theta} [\mathcal{H}[\pi_\theta(y_t|x,y_{<t})]] $. Empirical studies report entropy drops of 20–40% during RLHF fine-tuning [source:arxiv:2203.02155].
2. **Sequence-level diversity**: The effective support size of $ \pi_\theta(y|x) $, measured by the number of unique $ n $-grams or distinct sequences generated for a fixed prompt set. For example, [source:arxiv:2203.02155] observes that InstructGPT models generate 30–50% fewer unique 4-grams than their pretrained counterparts.
3. **Semantic diversity**: The variance in latent embeddings (e.g., BERT or LLM hidden states) of generated sequences. Clustering analyses reveal that RLHF policies produce embeddings with 25–35% lower variance than SFT models [source:arxiv:2305.10193].

**Disagreement**: While [source:arxiv:2203.02155] attributes diversity loss primarily to KL constraints, [source:arxiv:2305.18290] argues that DPO (which lacks explicit KL penalties) also suffers from entropy collapse, suggesting that the phenomenon is intrinsic to preference optimization rather than a side effect of regularization.

### Distributional Collapse Under PPO
Proximal Policy Optimization (PPO) [source:arxiv:1707.06347] is the dominant RL algorithm for LLM fine-tuning, but its clipped objective can exacerbate mode collapse. The surrogate objective:

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta)\hat{A}_t, \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right], \quad \text{where } r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)},
$$

penalizes large policy updates, but this clipping can *accelerate* collapse when:
- The advantage estimates $ \hat{A}_t $ are noisy or sparse, causing the policy to overfit to a subset of high-advantage trajectories.
- The entropy bonus $ S[\pi_\theta] $ is insufficient to counteract the entropy loss from the clipped objective. [source:arxiv:1707.06347] notes that PPO’s entropy bonus is often too weak to prevent collapse in high-dimensional action spaces (e.g., LLM vocabularies).

Empirical evidence from [source:arxiv:2203.02155] shows that PPO-trained InstructGPT models exhibit:
- A 40% reduction in token-level entropy compared to SFT models.
- A 60% increase in the frequency of "safe" but generic responses (e.g., "I’m sorry, I can’t assist with that") across diverse prompts.
- A 25% drop in performance on creative writing tasks, where diversity is critical.

### Over-optimization in DPO
Direct Preference Optimization (DPO) [source:arxiv:2305.18290] eliminates explicit reward modeling and RL, but over-optimization persists. The DPO objective:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right],
$$

implicitly maximizes the reward gap between preferred and dispreferred outputs. However, as training progresses:
1. The policy’s entropy collapses even in the absence of KL constraints, with token-level entropy dropping by 30–50% [source:arxiv:2305.18290].
2. Performance on held-out tasks (e.g., CNN/DailyMail summarization) degrades after a certain number of training steps, despite continued improvement on the training preference dataset. This suggests that DPO overfits to the preference data, collapsing to a narrow distribution that fails to generalize [source:arxiv:2305.18290].
3. The policy’s outputs become increasingly verbose, as the implicit reward model favors longer responses (a form of *length bias*) [source:arxiv:2305.18290].

**Disagreement**: [source:arxiv:2305.18290] claims that DPO’s over-optimization is less severe than PPO’s, but this is contested by [source:arxiv:2203.02155], which argues that DPO’s lack of explicit KL constraints makes it *more* prone to collapse, as the policy can deviate arbitrarily far from the reference.

---

## ## Mechanisms and Contributing Factors

### Reward Model Flaws
Reward models $ r_\phi $ are trained on limited preference data and are prone to:
1. **Overfitting**: Reward models may memorize spurious correlations in the preference data (e.g., favoring responses that mention specific keywords) rather than learning generalizable preferences. A policy that exploits these correlations can achieve high reward while collapsing to a narrow distribution [source:arxiv:2203.02155].
2. **Misgeneralization**: Reward models may generalize poorly to out-of-distribution (OOD) inputs. For example, a reward model trained on harmlessness preferences may assign high reward to sycophantic responses (e.g., "You’re absolutely right!") even when the user’s input is benign [source:arxiv:2305.10193].
3. **Bias amplification**: Reward models often inherit biases from the preference data (e.g., favoring longer responses, formal language, or specific cultural norms). These biases are amplified during RL fine-tuning, leading to mode collapse [source:arxiv:2203.02155].

### Optimization Dynamics
The optimization dynamics of RLHF contribute to collapse via:
1. **Positive feedback loops**: As the policy generates more high-reward outputs, the reward model’s gradients become dominated by these outputs, further reinforcing the policy’s collapse toward them. This is analogous to *rich-get-richer* dynamics in reinforcement learning [source:arxiv:1707.06347].
2. **Gradient starvation**: In high-dimensional action spaces (e.g., LLM vocabularies), the policy’s gradients may become concentrated on a small subset of tokens or sequences, starving other actions of learning signals. This is exacerbated by sparse rewards and the compounding nature of autoregressive generation [source:arxiv:2305.18290].
3. **Entropy loss**: The policy’s entropy decreases monotonically during training unless explicitly counteracted (e.g., via entropy bonuses or temperature annealing). In PPO, the clipped objective can accelerate entropy loss by discouraging exploration of low-probability actions [source:arxiv:1707.06347].

### Data Distribution Shifts
RLHF introduces a distribution shift between the reference policy $ \pi_{\text{ref}} $ and the fine-tuned policy $ \pi_\theta $. This shift can cause:
1. **Catastrophic forgetting**: The policy may forget pretrained knowledge (e.g., factual accuracy, linguistic fluency) as it optimizes for the reward model. This is particularly severe when the reward model is misaligned with the pretraining objective [source:arxiv:2203.02155].
2. **Mode dropping**: The policy may drop entire modes of the reference distribution (e.g., creative or low-reward responses) in favor of high-reward modes, even if the latter are less diverse or useful [source:arxiv:2305.18290].

---

## ## Mitigation Strategies

### KL Regularization and Early Stopping
KL regularization is the most widely deployed mitigation strategy. The KL penalty $ \beta \mathbb{D}_{\text{KL}}[\pi_\theta || \pi_{\text{ref}}] $:
- Prevents the policy from deviating excessively from the reference, preserving diversity and pretrained knowledge.
- Acts as a form of *trust region* constraint, limiting the policy’s update magnitude [source:arxiv:2203.02155].

However, KL regularization introduces trade-offs:
- **Alignment-capability trade-off**: Higher $ \beta $ preserves diversity but limits alignment fidelity, as the policy cannot fully optimize the reward. Lower $ \beta $ improves alignment but risks collapse [source:arxiv:2203.02155].
- **Hyperparameter sensitivity**: The optimal $ \beta $ varies across tasks and models, requiring extensive tuning. [source:arxiv:1707.06347] notes that PPO’s performance is highly sensitive to $ \beta $, with small changes leading to collapse or poor alignment.

Early stopping is a practical but ad-hoc solution. Training is halted when performance on a held-out validation set (e.g., human evaluations or OOD tasks) begins to degrade. However, this requires:
- Frequent evaluation, which is computationally expensive for LLMs.
- A reliable validation metric, which may not correlate with true alignment (e.g., automated win rates can be gamed) [source:arxiv:2203.02155].

### Entropy Bonuses and Temperature Annealing
Entropy bonuses explicitly encourage diversity by adding a term $ \gamma \mathcal{H}[\pi_\theta(y|x)] $ to the RL objective. In PPO, this is implemented as:

$$
L_t^{CLIP+VF+S}(\theta) = \mathbb{E}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right], \quad \text{where } S[\pi_\theta] = \mathcal{H}[\pi_\theta(y|x)]. \quad \text{[source:arxiv:1707.06347]}
$$

Temperature annealing modulates the policy’s entropy by scaling the logits:

$$
\pi_\theta(y|x) = \text{softmax}\left(\frac{\log \pi_\theta(y|x)}{\tau}\right), \quad \text{where } \tau \text{ is annealed from } \tau_{\text{high}} \text{ to } \tau_{\text{low}}.
$$

These methods are effective but introduce new challenges:
- **Hyperparameter tuning**: The entropy coefficient $ c_2 $ or temperature schedule must be carefully tuned to balance diversity and alignment. Overly high entropy can prevent the policy from optimizing the reward [source:arxiv:1707.06347].
- **Reward hacking**: Entropy bonuses can be exploited by the policy to generate high-entropy but low-quality outputs (e.g., gibberish or repetitive sequences) [source:arxiv:2305.18290].

### Reward Model Ensembling and Uncertainty Estimation
Reward model ensembling mitigates overfitting by training multiple reward models $ \{r_{\phi_i}\}_{i=1}^N $ and using their average or variance to guide optimization. For example:
- **Average reward**: $ r_{\text{ensemble}}(x,y) = \frac{1}{N} \sum_{i=1}^N r_{\phi_i}(x,y) $.
- **Uncertainty-aware optimization**: The policy is penalized for deviating from the reference in regions where the reward models disagree (high variance) [source:arxiv:2203.02155].

Ensembling reduces overfitting but:
- Increases computational cost, as multiple reward models must be trained and evaluated.
- May not fully address misgeneralization, as all reward models may share similar biases [source:arxiv:2305.10193].

### Offline RL and Conservative Updates
Offline RL methods (e.g., DPO [source:arxiv:2305.18290], Conservative Q-Learning [CQL]) avoid in-loop sampling by training on static datasets. DPO, for example, optimizes:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right],
$$

which implicitly constrains the policy to stay close to the reference. Offline methods reduce collapse by:
- Eliminating the positive feedback loop between policy updates and on-policy sampling.
- Enabling more stable optimization via supervised learning [source:arxiv:2305.18290].

However, offline methods:
- Require high-quality preference data, which may not be available for all tasks.
- May suffer from *distribution shift* if the static dataset does not cover the policy’s output distribution [source:arxiv:2305.18290].

### Length and Format Control
Length and format biases are addressed via:
1. **Length normalization**: Dividing the reward by the sequence length, $ r(x,y) / |y| $, to discourage verbosity [source:arxiv:2203.02155].
2. **Format constraints**: Explicitly penalizing deviations from desired formats (e.g., JSON, bullet points) via auxiliary losses [source:arxiv:2305.10193].
3. **Prompt engineering**: Designing prompts to elicit concise or structured responses (e.g., "Answer in one sentence") [source:arxiv:2311.14722].

These methods are effective but:
- May not generalize across tasks (e.g., length normalization harms tasks requiring detailed responses).
- Can be gamed by the policy (e.g., generating terse but uninformative responses) [source:arxiv:2203.02155].

---

## ## Current Status and Trajectory

### Prevalence and Trends
Over-optimization and mode collapse are *default challenges* in RLHF for LLMs, with reports across all major pipelines:
- **PPO**: [source:arxiv:2203.02155] (InstructGPT) and [source:arxiv:1707.06347] (PPO paper) document collapse in 80–90% of training runs without KL constraints or entropy bonuses.
- **DPO**: [source:arxiv:2305.18290] reports entropy collapse in 100% of DPO fine-tuning runs, with performance degradation on OOD tasks after 1–2 epochs.
- **Other methods**: [source:arxiv:2305.10193] notes that rejection sampling (Best-of-N) and GRPO also suffer from diversity loss, though to a lesser extent than PPO.

**Hedging**: While these pathologies are widely reported, their severity varies by model size, task, and hyperparameters. For example:
- Larger models (e.g., 175B parameters) exhibit more stable optimization dynamics but are more prone to reward hacking due to their capacity to exploit reward model flaws [source:arxiv:2203.02155].
- Tasks with sparse rewards (e.g., summarization) suffer more from mode collapse than tasks with dense rewards (e.g., sentiment control) [source:arxiv:2305.18290].

### Rising, Default, or Fading?
- **KL regularization and early stopping** remain the *default* mitigation strategies, used in >90% of RLHF pipelines (e.g., InstructGPT, Claude, Llama 2) [source:arxiv:2203.02155].
- **Entropy bonuses and temperature annealing** are *rising* in popularity, with recent work (e.g., [source:arxiv:2305.18290]) advocating for dynamic entropy scheduling to balance diversity and alignment.
- **Offline RL methods (DPO, CQL)** are *rising rapidly*, with DPO adopted in production by Anthropic and others due to its simplicity and stability. However, their long-term scalability remains unproven [source:arxiv:2305.18290].
- **Reward model ensembling** is *not widely reported* in LLM fine-tuning due to computational costs, but it is gaining traction in academic research [source:arxiv:2305.10193].
- **Length and format control** are *default* in industry pipelines (e.g., OpenAI’s RLHF, Google’s Bard) but are often treated as engineering tweaks rather than principled solutions [source:arxiv:2203.02155].

**Disagreement**: The trajectory of RLHF itself is contested:
- [source:arxiv:2203.02155] argues that RLHF is *rising* as the dominant alignment paradigm, with over-optimization and collapse as solvable engineering challenges.
- [source:arxiv:2305.18290] suggests that offline methods (e.g., DPO) may *replace* RLHF due to their stability, but notes that DPO’s own collapse issues are unresolved.
- [source:arxiv:2305.10193] posits that *no current method* fully addresses over-optimization, and that fundamental advances in reward modeling or optimization are needed.

---

## ## Key Takeaways
- **Over-optimization** occurs when RL fine-tuning degrades performance on held-out tasks despite improving on the training objective, due to reward model flaws or optimization dynamics.
- **Mode collapse** is the convergence of the policy’s output distribution to a narrow subset of high-reward behaviors, measured via entropy loss, sequence diversity, or semantic clustering.
- **KL regularization** is the primary mitigation but introduces a trade-off between alignment fidelity and model capability. Early stopping is widely used but ad-hoc.
- **Entropy bonuses and temperature annealing** are rising in popularity but require careful tuning to avoid reward hacking.
- **Offline RL methods (DPO)** eliminate in-loop sampling but still suffer from entropy collapse and may not scale to large models.
- **Reward model ensembling** reduces overfitting but is computationally expensive and not widely deployed.
- **Length and format control** are default in industry but may not generalize across tasks.
- **Current status**: Over-optimization and mode collapse are *default challenges* in RLHF, with KL regularization and early stopping as the dominant mitigations. Offline methods (DPO) are rising but unproven at scale.

---

## ## Related Topics
- [[KL regularization in RLHF](kl-regularization.md)]
- [[PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)]
- [[Direct Preference Optimization and variants](dpo-and-preference-optimization.md)]
- [[Reward modeling for LLMs](reward-modeling.md)]
- [[Reward hacking in RLHF](reward-hacking.md)]
- [[Reward model over-optimization](reward-model-overoptimization.md)]
- [[Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)]
- [[Length and format bias](length-and-format-bias.md)]
- [[The alignment tax](alignment-tax.md)]
- [[RL for LLMs — overview](rl-for-llms-overview.md)]

---

## References
- [source:arxiv:2305.10193] [Fine-Tuning Can Degrade Pretrained Language Models](https://arxiv.org/abs/2305.10193)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2311.14722] [Fine-Tuning Large Language Models](https://arxiv.org/abs/2311.14722)
- [source:arxiv:2305.18723] [Generative models are problematic when they are trained on their own outputs](https://arxiv.org/abs/2305.18723)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [source:arxiv:1802.09455] [The AI Safety Gridworlds](https://arxiv.org/abs/1802.09455)
