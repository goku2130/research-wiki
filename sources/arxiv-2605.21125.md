---
id: arxiv:2605.21125
type: paper
title: 'Advantage Collapse in Group Relative Policy Optimization: Diagnosis and Mitigation'
url: https://arxiv.org/abs/2605.21125
retrieved: '2026-07-10'
maturity: comprehensive
topic: grpo
---

**Core Problem**
Group Relative Policy Optimization (GRPO) suffers from advantage collapse, a failure mode where homogeneous rewards within a sampling group (e.g., all correct or all incorrect) cause within-group reward variance to vanish. This collapses advantages and policy gradients to zero, rendering training batches computationally wasteful. The phenomenon is prevalent in mathematical reasoning tasks, affecting 28–45% of training batches, and remains invisible to standard diagnostics like loss curves or accuracy metrics.

**Diagnostic Metric: Advantage Collapse Rate (ACR)**
To quantify gradient ineffectiveness, the authors introduce ACR, a real-time metric tracking the proportion of groups in a batch with negligible learning signals. ACR leverages reward standard deviations already computed during GRPO training. Early-stage ACR strongly predicts final performance, explaining 62% of the variance in outcomes, enabling practitioners to detect ineffective training configurations before accuracy degradation becomes visible.

**Mitigation Recipe: Adaptive Virtual Sample Policy Optimization (AVSPO)**
AVSPO is a lightweight, plug-and-play extension of GRPO that restores reward diversity without additional model rollouts. The algorithm follows a strict step-by-step recipe:
1. **Monitor:** Compute batch-level ACR at each training iteration.
2. **Adapt Threshold:** Dynamically adjust an intervention threshold based on training stability; decrease the threshold when training stagnates under high ACR to trigger stronger correction.
3. **Inject Virtual Samples:** When ACR exceeds the threshold, generate synthetic virtual reward samples for collapsed groups. The sample count scales adaptively with ACR to balance augmentation strength.
4. **Assign Continuous Rewards:** Assign virtual samples continuous reward values in a controlled range to guarantee non-zero variance, regardless of whether the original group was all-correct or all-incorrect.
5. **Recompute Normalization:** Calculate new group mean and standard deviation using the augmented reward set (real + virtual samples).
6. **Compute Advantages & Update:** Calculate advantages exclusively for real samples using the new normalization statistics, then apply the standard GRPO clipped surrogate objective for policy updates. Virtual samples influence only the normalization baseline, not the policy gradient directly.

**Key Formulas**
The GRPO advantage estimator:
$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$
The ACR diagnostic metric:
$$\text{ACR} = \frac{1}{N} \sum_{j=1}^N \mathbb{I}(\sigma_{r_j} \le \epsilon)$$
The adaptive virtual sample count:
$$N_v = \lceil \text{ACR} \cdot N \cdot \alpha \rceil$$
The AVSPO advantage computation:
$$\hat{A}_i^{\text{AVSPO}} = \frac{r_i - \mu_{r}^{\text{aug}}}{\sigma_{r}^{\text{aug}} + \epsilon}$$
where $\mu_{r}^{\text{aug}}$ and $\sigma_{r}^{\text{aug}}$ denote the mean and standard deviation of the augmented reward set.

**Quantitative Results**
Evaluated across Qwen2.5 models (0.5B to 14B parameters) on six mathematical reasoning benchmarks, AVSPO reduces ACR from 0.28–0.45 (GRPO baseline) to 0.11–0.18, achieving a 58–63% relative reduction in advantage collapse. This correction yields consistent accuracy gains of 4–6 percentage points over vanilla GRPO across all model scales. For example, Qwen2.5-0.5B improves from 16.5% to 21.0% average accuracy, while Qwen2.5-14B advances from 49.9% to 54.5%. Models with higher baseline ACR exhibit larger gains, confirming that restoring batch-level gradient diversity directly correlates with performance. Out-of-domain generalization on MMLU-Pro is maintained.

**Stated Limitations**
The authors note several constraints. First, virtual sample injection introduces a controlled bias relative to original GRPO gradients, though this bias-variance tradeoff is favorable when the alternative is complete gradient stagnation. Second, performance gains diminish on competition-level benchmarks (AMC, AIME), indicating that model capacity, rather than advantage collapse, becomes the primary bottleneck at high difficulty levels. Third, larger models inherently exhibit lower baseline ACR, resulting in comparatively smaller relative improvements. Finally, ACR and AVSPO are specifically designed for group-based policy optimization under binary or verifiable reward structures, and their applicability to continuous or multi-objective reward landscapes remains unverified.
