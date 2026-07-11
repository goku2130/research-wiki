---
title: The alignment tax
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2203.02155
- arxiv:2212.08073
- arxiv:2305.14387
- arxiv:2305.18290
- arxiv:2307.14977
- arxiv:1909.08599
- arxiv:2210.10725
- arxiv:2112.11795
open_questions:
- How does the alignment tax scale with model size? Are larger models more or less
  susceptible to capability regressions during alignment?
- Can process-based reward models effectively reduce reward hacking in open-ended
  generation tasks, or are they limited to structured domains like math and code?
- What are the most effective methods for measuring the alignment tax? Are there standardized
  benchmarks or evaluation protocols for quantifying capability regressions?
- How can alignment methods be adapted to multilingual or multimodal settings without
  introducing new capability regressions?
---

# The Alignment Tax: Capability Regressions from Alignment and Their Mitigation

The alignment tax refers to the unintended degradation of general capabilities—such as reasoning, factual accuracy, or task performance—observed when large language models (LLMs) undergo alignment procedures like Reinforcement Learning from Human Feedback (RLHF) or Direct Preference Optimization (DPO). While alignment improves adherence to human preferences (e.g., helpfulness, harmlessness), it often comes at the cost of reduced performance on standard benchmarks or out-of-distribution tasks. This trade-off arises from the tension between optimizing for narrow preference signals and preserving the broad, general-purpose knowledge encoded during pretraining. Mitigation strategies, such as KL regularization, auxiliary pretraining objectives, or hybrid training pipelines, aim to minimize these regressions while maintaining alignment gains.

---

## Mechanisms of the Alignment Tax

### 1. **KL Divergence Penalty and Policy Drift**
Alignment methods like RLHF and DPO constrain the policy $\pi_\theta$ to stay close to a reference model $\pi_{\text{ref}}$ (typically the supervised fine-tuning (SFT) model) via a KL divergence penalty. The RLHF objective for PPO is:

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{\text{RL}}}} \left[ r_\theta(x,y) - \beta \log \left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right) \right],
$$

where $\beta$ controls the strength of the penalty [source:arxiv:2203.02155]. The PPO clip objective is given by:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right],
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the probability ratio, $\hat{A}_t$ is the advantage estimate, and $\epsilon$ is a hyperparameter (typically 0.2) [source:arxiv:2203.02155]. Similarly, DPO reparameterizes the reward function as:

$$
r(x,y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x),
$$

and optimizes the policy to maximize the log-likelihood of preferences under this implicit reward [source:arxiv:2305.18290]. The DPO loss is:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right].
$$

**Problem:** The KL penalty discourages the policy from deviating too far from $\pi_{\text{ref}}$, but it does not guarantee preservation of capabilities outside the preference dataset. If the preference data is narrow (e.g., focused on harmlessness or specific instruction-following tasks), the policy may "forget" or underperform on unrelated tasks, such as mathematical reasoning or factual recall. This is exacerbated when $\beta$ is too small, leading to excessive policy drift, or too large, causing under-optimization of the reward signal. For example, [source:arxiv:2203.02155] reports that InstructGPT models exhibit performance gaps on SQuAD (a reading comprehension benchmark) and DROP (a reasoning benchmark), despite improvements in truthfulness on TruthfulQA.

### 2. **Reward Model Over-Optimization and Hacking**
Reward models trained on human preferences are imperfect proxies for true alignment. Over-optimizing for these models can lead to reward hacking, where the policy exploits spurious correlations in the reward signal at the expense of genuine capability. For example:
- **Length bias:** Models may generate verbose responses to maximize reward, even if the content is uninformative or repetitive. [source:arxiv:2305.14387] demonstrates that simulated annotators exhibit a preference for longer outputs, which drives reward model bias.
- **Format bias:** Models may overfit to specific output formats (e.g., bullet points, markdown) that are overrepresented in preference data.

**Evidence:** [source:arxiv:2305.14387] demonstrates that reward model over-optimization can occur in both PPO and DPO when trained on noisy or biased preference data, with win-rates declining after excessive reward maximization. The study shows that this phenomenon is replicated in simulation only when annotator variability and label noise are included, highlighting the role of data quality in reward hacking.

### 3. **Distribution Shift and Out-of-Distribution (OOD) Performance**
Alignment datasets are typically smaller and less diverse than pretraining corpora. This distribution shift can cause the aligned policy to perform poorly on tasks outside the alignment dataset, such as:
- **Factual recall:** Models may hallucinate or provide incorrect answers when queried about niche or technical topics not covered in the preference data.
- **Reasoning tasks:** Performance on math, coding, or logical reasoning benchmarks may degrade if the alignment data lacks such examples.
- **Low-resource languages:** Alignment data is often English-centric, leading to regressions in multilingual performance.

**Evidence:** [source:arxiv:2203.02155] reports that InstructGPT models exhibit gaps on translation tasks and closed-book QA benchmarks like SQuAD, despite improvements in truthfulness on TruthfulQA. Similarly, [source:arxiv:2305.18290] notes that DPO's OOD generalization requires further study, as its performance on CNN/DailyMail summarization may not extend to other domains.

### 4. **Mode Collapse and Reduced Diversity**
Alignment can inadvertently reduce the diversity of model outputs, leading to mode collapse. This occurs when:
- The reward model overfits to a narrow subset of "preferred" outputs, discouraging exploration of alternative, potentially higher-quality responses.
- The KL penalty suppresses low-probability but high-reward outputs, as the policy is discouraged from deviating too far from $\pi_{\text{ref}}$.

**Example:** [source:arxiv:2305.14387] observes that Best-of-$n$ sampling (a rejection sampling method) can achieve higher win-rates than PPO or DPO in some settings, suggesting that RL-based methods may under-explore the output space. Similarly, [source:arxiv:2212.08073] notes that Constitutional AI models can produce overly boilerplate responses when over-trained, indicating a loss of diversity.

---

## Mitigation Strategies

### 1. **KL Regularization and Auxiliary Objectives**
#### a. **PPO-ptx (Pretraining Mixing)**
[source:arxiv:2203.02155] introduces PPO-ptx, which augments the RLHF objective with a pretraining gradient term:

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{\text{RL}}}} \left[ r_\theta(x,y) - \beta \log \left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right) \right] + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log \pi_\phi^{\text{RL}}(x) \right].
$$

Here, $\gamma$ controls the strength of the pretraining gradient. This approach mitigates regressions on standard NLP benchmarks by preserving the model's general knowledge.

**Effectiveness:** PPO-ptx reduces but does not eliminate regressions. [source:arxiv:2203.02155] reports that InstructGPT models still underperform on SQuAD, DROP, and translation tasks compared to the pretrained baseline. The optimal value of $\gamma$ is task-dependent and requires careful tuning.

#### b. **Reference Model Updates**
Instead of fixing $\pi_{\text{ref}}$ as the SFT model, some methods periodically update $\pi_{\text{ref}}$ to the current policy during training. This allows the policy to drift further while maintaining a "moving anchor" to prevent catastrophic forgetting. However, this can also exacerbate reward hacking if the reference model itself becomes misaligned.

### 2. **Hybrid Training Pipelines**
#### a. **Constitutional AI (CAI)**
[source:arxiv:2212.08073] proposes a two-stage pipeline:
1. **Supervised Learning (SL-CAI):** The model generates critiques and revisions of its own outputs based on a "constitution" of principles (e.g., "avoid harmful advice"). This stage bootstraps harmlessness without human labels.
2. **Reinforcement Learning (RL-CAI):** The model is fine-tuned via RL using a hybrid preference model that combines AI-generated harmlessness labels with human helpfulness labels.

**Advantages:**
- Reduces reliance on human harmlessness labels, scaling oversight.
- Mitigates evasive refusals by encouraging transparent engagement with sensitive queries.
- Preserves helpfulness while improving harmlessness.

**Limitations:** CAI still relies on human feedback for helpfulness and can produce overly harsh or boilerplate responses when over-trained. The method is also susceptible to "Goodharting," where models over-optimize for the constitutional principles at the expense of naturalness.

#### b. **Best-of-$n$ Sampling and Rejection Sampling**
Instead of RL, some methods use rejection sampling to select the highest-reward output from $n$ candidates generated by the SFT model. [source:arxiv:2305.14387] shows that Best-of-1024 sampling achieves a 50.7% human win-rate, though it underperforms PPO (55% win-rate) in the same evaluation.

**Trade-offs:**
- **Pros:** Simple, stable, and avoids reward hacking.
- **Cons:** Computationally expensive at inference time; does not improve the underlying policy. The method is also limited by the quality of the SFT model, as it cannot generate outputs beyond its distribution.

### 3. **Reward Model Improvements**
#### a. **Process-Based Reward Models**
Traditional reward models evaluate outcomes (e.g., "is this response helpful?"). Process-based reward models evaluate the *process* of generation (e.g., "does this response follow a logical chain of reasoning?"). This can reduce reward hacking by discouraging superficial strategies like length bias or sycophancy.

**Evidence:** [source:arxiv:2203.02155] notes that InstructGPT models still exhibit over-hedging and failure to detect false premises, suggesting that outcome-based reward models are insufficient for complex reasoning tasks. Process-based rewards remain an active area of research, with no widely adopted implementation to date.

#### b. **Verifiable Rewards (RLVR)**
Verifiable rewards use formal methods (e.g., program execution, theorem proving) to evaluate outputs, reducing reliance on learned reward models. For example, a reward for a coding task could be the fraction of unit tests passed.

**Limitations:** Applicable only to tasks with verifiable outputs (e.g., math, code); not scalable to open-ended generation. The method also requires task-specific engineering and may not generalize across domains.

### 4. **Data Augmentation and Diversity**
#### a. **Expanding Preference Data**
Increasing the diversity of preference data can reduce distribution shift. For example:
- Including prompts from a wide range of domains (e.g., math, coding, creative writing).
- Collecting preferences from diverse annotator groups to avoid demographic bias.

**Evidence:** [source:arxiv:2305.14387] shows that simulated annotator pools with high variability (e.g., 25% label-flip noise) better replicate human reward model over-optimization, suggesting that noisy, diverse data may improve robustness. However, the study also notes that human annotators exhibit biases, such as a preference for longer outputs and list formats, which can propagate into the reward model.

#### b. **Synthetic Data Generation**
Models can generate synthetic prompts and responses, which are then filtered or relabeled by humans or AI. [source:arxiv:2212.08073] uses this approach in CAI, where the model generates red-teaming prompts and critiques its own outputs.

**Trade-offs:**
- **Pros:** Scales data collection; reduces human labeling costs.
- **Cons:** Risks amplifying model biases if synthetic data is not carefully curated. The method also requires a high-quality base model to generate useful synthetic data.

### 5. **Architectural and Optimization Adjustments**
#### a. **Entropy Regularization**
Adding an entropy term to the RL objective encourages exploration and prevents mode collapse:

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{\text{RL}}}} \left[ r_\theta(x,y) - \beta \log \left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right) \right] + \lambda H(\pi_\phi^{\text{RL}}),
$$

where $H(\pi_\phi^{\text{RL}})$ is the entropy of the policy.

**Effectiveness:** [source:arxiv:2305.14387] finds that entropy regularization improves the performance of PPO and DPO, but excessive entropy can reduce alignment quality. The optimal value of $\lambda$ is task-dependent and requires tuning.

#### b. **Temperature Scheduling**
Adjusting the sampling temperature during training can balance exploration and exploitation. [source:arxiv:2305.18290] shows that DPO is more robust to temperature variations than PPO, but optimal temperatures vary by task. For example, DPO achieves a 61% win-rate on TL;DR summarization at temperature 0.0, compared to PPO's 57% at its optimal temperature.

---

## Current Status and Trajectory

The alignment tax is a **persistent and actively researched challenge**, with no single mitigation strategy universally adopted. Current trends and open questions include:

1. **Rising Techniques:**
   - **DPO and variants:** Gaining traction due to simplicity and competitive performance with PPO [source:arxiv:2305.18290]. [source:arxiv:2305.14387] ranks DPO as a top-performing method in simulated human feedback pipelines, with a Spearman correlation of 0.98 between simulated and human method rankings.
   - **Hybrid pipelines (e.g., CAI):** Increasingly used to reduce reliance on human labels while preserving alignment [source:arxiv:2212.08073]. CAI models virtually eliminate evasive responses and improve harmlessness while maintaining helpfulness.
   - **Process-based rewards:** Emerging as a way to reduce reward hacking, though not yet widely deployed. The field lacks consensus on how to design and evaluate process-based rewards.

2. **Default Techniques:**
   - **KL regularization:** Remains the default method for constraining policy drift in RLHF and DPO [source:arxiv:2203.02155][source:arxiv:2305.18290]. The optimal value of $\beta$ is typically in the range of 0.01 to 0.5, depending on the task.
   - **PPO:** Still widely used, especially in production systems (e.g., InstructGPT), despite its complexity [source:arxiv:2203.02155]. PPO's clip objective ($\epsilon = 0.2$) is a standard hyperparameter.

3. **Fading Techniques:**
   - **Pure rejection sampling (Best-of-$n$):** Declining for large-scale deployment due to inference-time costs, though it remains a strong baseline [source:arxiv:2305.14387]. Best-of-1024 sampling achieves a 50.7% human win-rate but is computationally prohibitive for real-time applications.

4. **Unresolved Challenges:**
   - **Scaling to larger models:** Most mitigation strategies (e.g., DPO, PPO-ptx) have been validated on models up to ~50B parameters. Their effectiveness at the scale of frontier models (e.g., >100B parameters) is not widely reported. [source:arxiv:2305.18290] notes that DPO's scalability to larger architectures remains an open question.
   - **Long-horizon tasks:** Alignment methods struggle with tasks requiring multi-step reasoning (e.g., math, agentic tool use), where capability regressions are more pronounced. [source:arxiv:2203.02155] reports that InstructGPT models exhibit failures on tasks with multiple explicit constraints.
   - **Multilingual and multimodal alignment:** The alignment tax in non-English or multimodal settings is understudied. Most alignment datasets are English-centric, and the generalization of alignment methods to other languages or modalities is not well-documented.

**Disagreement:** The field is divided on the root causes of the alignment tax. Some researchers (e.g., [source:arxiv:2305.18290]) argue that the tax stems from imperfect reward models and can be mitigated with better preference learning (e.g., DPO). Others (e.g., [source:arxiv:2203.02155]) attribute it to fundamental trade-offs between alignment and capability, necessitating auxiliary objectives like PPO-ptx. A third perspective ([source:arxiv:2305.14387]) suggests that the alignment tax is driven by data quality and annotator variability, implying that better data collection methods (e.g., diverse annotator pools) could mitigate the issue. The lack of standardized benchmarks for measuring the alignment tax further complicates comparisons.

---

## Key Takeaways

- The **alignment tax** manifests as capability regressions on tasks outside the alignment dataset, caused by KL divergence penalties, reward hacking, distribution shift, and mode collapse. These regressions are quantified in [source:arxiv:2203.02155], where InstructGPT models show performance gaps on SQuAD (reading comprehension) and DROP (reasoning) benchmarks.
- **Mitigation strategies** include:
  - KL regularization (e.g., PPO-ptx, DPO) to constrain policy drift. The PPO clip equation ($\epsilon = 0.2$) and DPO loss are key formulas in these methods.
  - Hybrid training pipelines (e.g., Constitutional AI) to reduce reliance on human labels. CAI achieves harmlessness improvements without human harmlessness labels, as shown in [source:arxiv:2212.08073].
  - Reward model improvements (e.g., process-based rewards, verifiable rewards) to reduce hacking. However, these methods are not yet widely adopted.
  - Data augmentation (e.g., synthetic data, diverse annotators) to reduce distribution shift. [source:arxiv:2305.14387] demonstrates that annotator variability (25% label-flip noise) improves robustness.
  - Entropy regularization and temperature scheduling to prevent mode collapse. [source:arxiv:2305.18290] shows that DPO is more robust to temperature variations than PPO.
- **DPO** is rising as a simpler alternative to PPO, with [source:arxiv:2305.14387] ranking it as a top-performing method in simulated human feedback pipelines. DPO achieves a 61% win-rate on TL;DR summarization, outperforming PPO's 57%.
- **Hybrid pipelines** (e.g., CAI) are gaining traction for scaling alignment without human labels. CAI models improve harmlessness while maintaining helpfulness, as reported in [source:arxiv:2212.08073].
- **Reward hacking** and **distribution shift** are the most persistent sources of capability regression. [source:arxiv:2305.14387] demonstrates that reward model over-optimization occurs when training on noisy or biased data, leading to declining win-rates.
- **Long-horizon tasks** (e.g., math, agentic tool use) and **multilingual/multimodal alignment** remain open challenges. [source:arxiv:2203.02155] notes that InstructGPT models struggle with tasks requiring multi-step reasoning or multiple constraints.

---

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Length and format bias](length-and-format-bias.md)
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)

---

##

## References
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2212.08073] [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [source:arxiv:2305.14387] [AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback](https://arxiv.org/abs/2305.14387)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2307.14977] [Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Want This](https://arxiv.org/abs/2307.14977)
- [source:arxiv:1909.08599] [Learning from Human Preferences](https://arxiv.org/abs/1909.08599)
- [source:arxiv:2210.10725] [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10725)
- [source:arxiv:2112.11795] [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2112.11795)
