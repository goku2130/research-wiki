---
title: The alignment tax
maturity: developing
updated: '2026-07-11'
sources:
- forum:current-work-in-ai-alignment
- arxiv:2112.00861
- arxiv:2203.02155
- arxiv:2603.00047
- arxiv:2602.07892
- arxiv:2309.06256
- arxiv:2605.18838
- arxiv:2305.20050
open_questions:
- What is the theoretical lower bound of the alignment tax, and can any method achieve
  the optimal trade-off frontier? [source:arxiv:2309.06256]
- Does the CAPE phase transition ($N_c$) continue to decrease with architectural advances,
  or will irreducible representation entanglement ($\tau_0 > 0$) impose a fundamental
  floor even at infinite scale? [source:arxiv:2603.00047][source:arxiv:2605.18838]
- Can OGPSA's first-order orthogonal projection be extended to higher-order capability
  preservation, and does it scale to >70B models under adaptive black-box attacks?
  [source:arxiv:2602.07892]
- Will process supervision (PRM) generalize beyond mathematical reasoning to mitigate
  alignment tax in open-ended domains, or does its multiplicative step-penalty fundamentally
  limit long-horizon tasks? [source:arxiv:2305.20050]
---

The alignment tax refers to the capability degradation incurred when aligning language models with human preferences through techniques like RLHF, DPO, or safety fine-tuning. While early work treated this trade-off as a necessary cost, recent research has formalized its geometry, identified a scaling-phase transition, and developed mitigation strategies ranging from heterogeneous model averaging to orthogonal gradient projection.

## Conceptual foundations and definitions

Christiano (2020) introduced the alignment tax as the performance penalty incurred when insisting on intent alignment—engineering systems that genuinely try to do what humans want—distinguishing it from adjacent problems of competence, reliability, and governance [source:forum:current-work-in-ai-alignment]. He outlined two high-level strategies: *paying the tax* (accepting capability trade-offs or coordinating developers to enforce alignment standards) or *reducing the tax* through technical innovation, structured as a pipeline that transforms a base algorithm $X$ into an aligned variant $\text{align}(X)$ while preserving utility and scalability. This framework also introduced the outer/inner alignment decomposition: outer alignment specifies the correct objective, while inner alignment ensures the learned model matches that objective, though both mechanisms remain under development [source:forum:current-work-in-ai-alignment].

The InstructGPT paper operationalized the tax as performance regressions on standard NLP benchmarks (DROP, SQuADv2, machine translation) when optimizing for human preference via PPO, and introduced PPO-ptx—mixing pretraining gradients into the RL objective—as a direct mitigation [source:arxiv:2203.02155]. The "General Language Assistant" study similarly measured the tax on coding tasks (HumanEval, QuixBugs), finding negligible degradation at 13B/52B parameters but significant capability loss in smaller models [source:arxiv:2112.00861].

## Empirical evidence of capability regression

| Model / Setting | Alignment Method | Tax Manifestation | Source |
|----------------|------------------|-------------------|--------|
| GPT-3 175B → InstructGPT 175B (PPO-ptx) | RLHF with pretraining mix | Lags behind GPT-3 on DROP, SQuADv2, translation | [source:arxiv:2203.02155] |
| OpenLLaMA-3B, Mistral-7B variants | RSF, DPO, PPO | Forgetting on ARC, Race, PIQA, SQuAD, DROP, WMT Fr-En | [source:arxiv:2309.06256] |
| Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct | SFT, DPO, SFT→DPO sequential | Capability regression on general utility benchmarks | [source:arxiv:2602.07892] |
| Pythia, OPT, Falcon families | Standard alignment | Reasoning–truthfulness anticorrelation ($r=-0.989$ for Pythia) below $N_c$ | [source:arxiv:2605.18838] |

The InstructGPT 1.3B model was preferred over the 175B GPT-3 baseline despite 100× fewer parameters, yet the 175B InstructGPT (PPO-ptx) still underperformed the base GPT-3 on several academic benchmarks [source:arxiv:2203.02155]. The "Mitigating the Alignment Tax of RLHF" paper systematically benchmarked regularization, LoRA, knowledge distillation, reward penalties, and model averaging (MA), finding that MA—simple linear interpolation between pre-RLHF weights $\theta_0$ and post-RLHF weights $\theta$—achieves the most efficient alignment-forgetting Pareto front [source:arxiv:2309.06256]. The CAPE framework revealed that below a critical scale $N_c$, reasoning and truthfulness *strongly anticorrelate* (defining the "alignment tax phase"), with $N_c \approx 0.12\text{B}$ for OPT, $3.5\text{B}$ for Pythia, and $7.0\text{B}$ for Falcon [source:arxiv:2605.18838].

## Theoretical characterizations: geometric and phase-transition views

### Geometric theory of the alignment tax

[source:arxiv:2603.00047] establishes a geometric theory in neural representation space. Assuming a fixed layer with $d$-dimensional hidden states, a unit safety direction $v^*$, and capability directions $c_i = \nabla_h f_i(h)/\|\nabla_h f_i(h)\|$, alignment is modeled as a representation shift $\delta$ constrained by $\|\delta\| \leq B$ (where $B$ originates from the first-order KL penalty in RLHF/DPO). The **alignment tax rate** is defined as

$$
\tau = \|P_C v^*\|^2,
$$

where $P_C$ projects onto the capability subspace $\mathcal{C} = \text{span}(c_1,\dots,c_m)$. For a single capability, the Pareto-optimal tradeoff frontier is

$$
\Delta_S = \Delta_C \cos\alpha + \sin\alpha \sqrt{B^2 - \Delta_C^2}, \quad |\Delta_C| \leq B,
$$

where $\Delta_S$ is safety gain, $\Delta_C$ is capability change, and $\alpha$ is the principal angle between $v^*$ and $\mathcal{C}$. The frontier interpolates between a linear tradeoff ($\alpha=0$) and independent optimization ($\alpha=\pi/2$). Per-task degradation is $\tau_i = \langle v^*, c_i \rangle^2$. Under fixed capability targets $\delta_C^*$, maximum safety is

$$
\Delta_S^{\max} = \langle v^*, \delta_C^* \rangle + \sqrt{B^2 - \|\delta_C^*\|^2} \sqrt{1-\tau},
$$

yielding tax-free safety $\Delta_S^{\text{free}} = B\sqrt{1-\tau}$.

The scaling law decomposes $\tau = \tau_0 + R(d)$, where $\tau_0 = \sum_{i \in I} \gamma_i^2$ is the irreducible tax and the packing residual satisfies $|R(d)| \leq \frac{\tau_0 m \mu + m' \mu^2 + 2\bar{\gamma}|I|\mu + |I|\mu^2}{1-m\mu}$ under bounded coherence $\mu$. Under random feature packing, $|R(d)| = O(m'/d)$; for fixed $m$, $\tau \to \tau_0$ at rate $O(\log d/d)$, while for linear capability growth ($m = \Theta(d)$) the expected incidental tax remains $\Theta(1)$. For LoRA rank $r=8$ in $d=4096$, isotropic perturbation yields $\sim 0.2\%$ capability degradation [source:arxiv:2603.00047].

**Limitations**: The framework assumes the linear representation hypothesis (nonlinear encoding only permits the geometry as a lower bound), treats $v^*$ as fixed (silent on normative specification), relies on first-order KL approximation and small perturbations, conditions on random feature packing with bounded coherence (real models learn structured representations), operates on population-average shifts (excluding input-dependent Jacobian effects), and models safety as a low-dimensional subspace [source:arxiv:2603.00047].

### Phase-transition view: CAPE framework

[source:arxiv:2605.18838] proposes the Capability Coupling Analysis of Phase Emergence (CAPE), modeling the coupling transition as

$$
\gamma_{12}(N, \mathcal{D}) = \gamma_0(\mathcal{D}) \cdot \log_{10}(N/N_c(\mathcal{D})),
$$

where $\gamma_{12} \equiv \Delta B_2/\Delta B_1$ is the local coupling between consecutive model sizes. Below $N_c$, capabilities anticorrelate (alignment tax phase); above $N_c$, they cooperate ($r > +0.78$). Benchmark trajectories follow the discovered ODE:

$$
\frac{dB_i}{d\log_{10}N} = \sum_j c_{ij}B_j + \sum_{j \le k} d_{ijk}B_j B_k.
$$

Dimensionality collapse is quantified via the participation ratio $d_{\text{eff}} = (\sum_{i=1}^5 \lambda_i)^2 / \sum_{i=1}^5 \lambda_i^2$. Width normalization $B_{\text{norm}} = B / (d_{\text{model}}/d_{\text{ref}})$ eliminates the anticorrelation across all tested families (Pythia shifts from $-0.989$ to $+0.963$) [source:arxiv:2605.18838]. Internal analysis reveals 38 of 40 models contain zero competing attention heads (95% CI: 84–99%), indicating the bottleneck resides at the output projection rather than in representational space. Activation steering at the quarter-depth bottleneck layer corrects 60% of misaligned outputs in the tax phase without retraining. At frontier scales, the cooperative regime persists ($r = +0.72$ across 34 models from 10 labs) [source:arxiv:2605.18838].

**Disagreement**: The geometric theory [source:arxiv:2603.00047] treats the tax as a continuous function of subspace alignment ($\tau = \|P_C v^*\|^2$) with a scaling law $\tau = \tau_0 + O(m'/d)$, implying a smooth reduction with width. The CAPE framework [source:arxiv:2605.18838] identifies a sharp phase transition at $N_c$ where coupling flips sign, with width normalization *eliminating* the anticorrelation entirely—suggesting the tax may be an artifact of output-projection bottleneck rather than fundamental representation entanglement. The geometric theory's assumption of fixed $v^*$ and linear representation hypothesis may not capture the dynamical regime change CAPE observes; conversely, CAPE's width-normalization argument (dividing bounded scores by growing denominator) could induce spurious positive correlation, though direct projection-width measurements are claimed to confirm the bottleneck [source:arxiv:2605.18838]. A family spanning 0.1B–10B with no coupling sign change would challenge CAPE; none observed across 16 tested families [source:arxiv:2605.18838].

## Mitigation strategies during RLHF/alignment training

### Pretraining gradient mixing (PPO-ptx)

InstructGPT introduced PPO-ptx, adding a pretraining loss term to the RL objective:

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi_\phi^{\text{RL}}}} \left[ r_\theta(x,y) - \beta \log \frac{\pi_\phi^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)} \right] + \gamma \mathbb{E}_{x\sim D_{\text{pretrain}}} \left[ \log \pi_\phi^{\text{RL}}(x) \right],
$$

where $\gamma$ controls the pretraining loss coefficient [source:arxiv:2203.02155]. This mitigates but does not eliminate regressions on DROP, SQuADv2, and translation.

### Model averaging (MA) and Heterogeneous Model Averaging (HMA)

Standard MA interpolates weights: $\pi_{(1-\alpha)\theta_0 + \alpha\theta}$. [source:arxiv:2309.06256] identifies $\alpha=0.2$ as a robust default. HMA partitions the transformer into $K$ contiguous parts (default $K=3$: layers 1–8, 9–17, 18–26) and assigns per-block ratios $\alpha_k \in [0,1]$, optimizing

$$
\max_{(\alpha_1,\dots,\alpha_K)\in\Omega} \mathbb{E}_x \mathbb{E}_{a\sim\pi_{\theta^{(K)}}(\cdot|x)}[r^*(x,a)], \quad \Omega = \left\{(\alpha_1,\dots,\alpha_K) \mid \frac{1}{K}\sum_k \alpha_k = \alpha, \alpha_k\in[0,1]\right\}.
$$

Theoretically, averaging improves robustness when tasks share feature spaces: $\xi^{(1)}-\xi^{(2)} = F_p\left(\frac{\sqrt{2}(1-p)n}{\sqrt{n+n_o}}\right) - F_p\left((1-p)\sqrt{n}\right) \geq 0$ [source:arxiv:2309.06256]. HMA with $K=3$ consistently outperforms vanilla MA across benchmarks; increasing $K$ to 6 or 9 yields marginal gains but risks overfitting, slightly reducing reading comprehension [source:arxiv:2309.06256].

### Orthogonal Gradient Projection for Safety Alignment (OGPSA)

[source:arxiv:2602.07892] reframes the tax as a heterogeneous continual learning problem: safety-driven gradients project onto parameter directions encoding pre-trained capabilities, causing regression. OGPSA maintains a low-rank general-capability subspace $\mathcal{S}_{\text{gen}}(\theta) := \text{span}\{\mathbf{g}^{(1)}(\theta),\dots,\mathbf{g}^{(M)}(\theta)\}$ estimated from reference datasets $\mathcal{D}_{\text{ref}}^{(i)}$ (helpfulness, truthfulness, etc.). Every $K$ steps, reference gradients are computed and orthonormalized via Gram-Schmidt with redundancy threshold $\delta$ to form $\mathbf{U}_\tau$. At each safety step, the safety gradient $\mathbf{g}_{\text{safe}} = \nabla_\theta \mathcal{L}_{\text{safe}}(\theta_t)$ is projected onto the orthogonal complement:

$$
\tilde{\mathbf{g}}_{\text{safe}} = \mathbf{g}_{\text{safe}} - \mathbf{U}_\tau(\mathbf{U}_\tau^\top \mathbf{g}_{\text{safe}}), \quad \Delta\theta = -\eta \tilde{\mathbf{g}}_{\text{safe}}.
$$

This is the steepest feasible descent direction under orthogonality constraints $\langle \mathbf{g}^{(i)}(\theta), \Delta\theta \rangle = 0$ [source:arxiv:2602.07892].

Empirically, on Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct under SFT, DPO, and sequential SFT→DPO, OGPSA improves the safety–utility trade-off. In sequential SFT→DPO, average performance gain over the instruct baseline increases from 33.98% to 42.74% (Qwen) and 19.74% to 32.98% (Llama). Data efficiency: 100–200 reference samples per dimension vs. 10,000-sample replay baselines. Training time: 2h 56m vs. 4h 5m for general-data mixing. I-GCG attack success drops to 26% (SFT+OGPSA) vs. 32% (standard SFT) [source:arxiv:2602.07892].

**Limitations**: OGPSA provides only first-order local approximation; efficacy depends on reference data diversity (single-direction projections too narrow); minor computational overhead from periodic reference-gradient computation; not evaluated on larger architectures or comprehensive black-box adaptive safety benchmarks; improved utility $\neq$ comprehensive robustness [source:arxiv:2602.07892].

### Preference Model Pre-training (PMP) and context distillation

[source:arxiv:2112.00861] introduces PMP: (1) standard LM pre-training, (2) pre-train preference model on public datasets (Stack Exchange, Reddit, reverted Wikipedia edits), (3) fine-tune on task-specific preference data. Ranked pairs $A > B$ are binarized into two independent comparisons: $\text{GOOD}: A > B$ and $\text{BAD}: B > A$. Binary PMP on mixed public data yields $\sim 0.10$ accuracy gain over baseline fine-tuning at 52B with $<10$k sequence pairs, and transfers better to downstream tasks than ranked PMP [source:arxiv:2112.00861]. Context distillation minimizes $L(\theta) = D_{\text{KL}}(p_0(X|C) \| p_\theta(X))$ to internalize an HHH prompt, though it can slightly degrade human-preferred performance vs. raw prompting [source:arxiv:2112.00861].

## Post-hoc and inference-time mitigations

### Activation steering at the bottleneck layer

[source:arxiv:2605.18838] discovers that 38/40 models have zero competing attention heads (bottleneck at output projection). Adding a truth-direction vector at the quarter-depth probe layer during inference corrects 60% of misaligned outputs in the tax phase without retraining. This zero-retraining intervention exploits the phase structure: below $N_c$, the safety direction is accessible at a specific layer; above $N_c$, the cooperative regime renders such steering unnecessary [source:arxiv:2605.18838].

### Process supervision (PRM) for reasoning reliability

While not a direct alignment tax mitigation, [source:arxiv:2305.20050] shows that process reward models (PRMs) trained on step-level human feedback achieve 78.2% solve rate on MATH (best-of-1860) vs. 72.4% for outcome reward models (ORMs) and 69.6% for majority voting. The PRM score is the product of step-level correctness probabilities: $P(\text{solution}) = \prod_i P(\text{step}_i \text{ is correct})$. Active learning yields 2.6× data efficiency over uniform labeling. However, PRM scoring inherently penalizes longer solutions due to multiplicative reduction, and findings may not generalize beyond mathematical domains [source:arxiv:2305.20050].

## Scaling behavior and the critical scale $N_c$

| Model Family | Critical Scale $N_c$ | Coupling Below $N_c$ | Coupling Above $N_c$ |
|--------------|---------------------|----------------------|----------------------|
| OPT          | $\approx 0.12\text{B}$ | Strong anticorrelation | Cooperative ($r > +0.78$) |
| Pythia       | $\approx 3.5\text{B}$ [2.9B, 13.4B] | $r = -0.989$ ($p=4\times10^{-5}$) | $r > +0.78$ |
| Falcon       | $\approx 7.0\text{B}$ | Strong anticorrelation | Cooperative |

Width normalization eliminates anticorrelation across all families (Pythia: $-0.989 \to +0.963$) [source:arxiv:2605.18838]. The geometric theory predicts $\tau \to \tau_0$ at $O(\log d/d)$ for fixed capability count $m$, but $\Theta(1)$ incidental tax for linear capability growth ($m = \Theta(d)$) [source:arxiv:2603.00047]. The CAPE ODE cross-predicts held-out Llama-2 benchmarks at 5.6% MAE, but fails across phase transitions, requiring phase-specific fitting [source:arxiv:2605.18838]. Frontier scales (34 models from 10 labs) show persistent cooperative regime ($r = +0.72$) [source:arxiv:2605.18838].

**Disagreement on scaling**: The geometric theory [source:arxiv:2603.00047] implies a smooth, predictable reduction in tax rate with width under random packing, with irreducible tax $\tau_0$ determined by true capability-safety entanglement. CAPE [source:arxiv:2605.18838] shows a sharp phase transition at $N_c$ that varies by architecture (0.12B–7B), with width normalization *removing* the tax phase entirely—suggesting the tax may be largely an artifact of output-projection competition rather than deep representation entanglement. The geometric theory's local, first-order analysis may not capture the global dynamical reorganization CAPE identifies; CAPE's width-normalization argument could be confounded by dividing bounded scores by growing denominators, though projection-width measurements are claimed to confirm the bottleneck [source:arxiv:2605.18838].

## Current status and trajectory

The alignment tax has evolved from an acknowledged but informally characterized trade-off (Christiano 2020; InstructGPT 2022) to a formally quantified phenomenon with geometric [source:arxiv:2603.00047] and dynamical [source:arxiv:2605.18838] theories. Mitigation strategies are diversifying:

- **Model averaging (MA/HMA)** is a rising, low-cost post-hoc default: simple weight interpolation with per-layer optimization, validated across RSF, DPO, PPO on OpenLLaMA-3B and Mistral-7B variants [source:arxiv:2309.06256]. Not yet reported as standard in production pipelines but widely applicable.
- **PPO-ptx (pretraining mix)** remains a default in RLHF pipelines but is acknowledged as incomplete—still lags on academic benchmarks [source:arxiv:2203.02155].
- **OGPSA** is a promising but early-stage technique: plug-and-play, data-efficient, improves safety-utility trade-off on 7B/8B models under SFT/DPO/sequential pipelines, but only first-order, untested at scale, and not evaluated against adaptive black-box attacks [source:arxiv:2602.07892].
- **Width normalization / scaling through $N_c$** suggests the tax may be a transient phase: frontier models (34 models from 10 labs) already exhibit cooperative coupling ($r=+0.72$) [source:arxiv:2605.18838]. If $N_c$ continues to decrease with architectural improvements (OPT at 0.12B vs. Falcon at 7B), the tax may become irrelevant for future large models—but this is not widely reported as a consensus, and the geometric theory's irreducible $\tau_0$ implies a fundamental floor.
- **Activation steering** at the bottleneck layer is a zero-retraining inference-time fix effective in the tax phase (60% correction) but untested at scale and phase-dependent [source:arxiv:2605.18838].
- **Process supervision (PRM)** addresses reasoning reliability rather than alignment tax per se, but its superior data efficiency and out-of-distribution generalization (72.9% on AP/AMC vs. 63.8% ORM) make it a rising component of alignment pipelines for reasoning tasks [source:arxiv:2305.20050].

No single mitigation is universally adopted as default; the field is in a **portfolio phase** where multiple complementary strategies (pretraining mix, model averaging, orthogonal gradients, inference-time steering) are combined. The theoretical lower bound of the alignment tax remains undetermined [source:arxiv:2309.06256].

## Key takeaways

- The alignment tax is the capability degradation from aligning LLMs with human preferences via RLHF/DPO/SFT, formally quantified as $\tau = \|P_C v^*\|^2$ in representation space [source:arxiv:2603.00047] and empirically observed as reasoning–truthfulness anticorrelation below a critical scale $N_c$ [source:arxiv:2605.18838].
- **Geometric theory** yields a Pareto frontier $\Delta_S = \Delta_C \cos\alpha + \sin\alpha \sqrt{B^2 - \Delta_C^2}$ and scaling law $\tau = \tau_0 + O(m'/d)$ (fixed $m$) or $\Theta(1)$ (linear $m$) [source:arxiv:2603.00047].
- **CAPE framework** identifies a phase transition at $N_c$ (0.12B–7B across architectures) where coupling flips from anticorrelated to cooperative; width normalization eliminates the tax phase [source:arxiv:2605.18838].
- **Mitigations during training**: PPO-ptx (pretraining mix) [source:arxiv:2203.02155], HMA (heterogeneous model averaging, $\alpha=0.2$, $K=3$) [source:arxiv:2309.06256], OGPSA (orthogonal gradient projection, 100–200 ref samples/dim) [source:arxiv:2602.07892], PMP (preference model pre-training) [source:arxiv:2112.00861].
- **Post-hoc / inference-time**: Standard MA (interpolation), HMA, activation steering at quarter-depth bottleneck (60% correction in tax phase) [source:arxiv:2605.18838].
- **Process supervision (PRM)** outperforms outcome supervision on math reasoning (78.2% vs 72.4% best-of-1860) with 2.6× data efficiency via active learning [source:arxiv:2305.20050].
- The tax appears to be **transient with scale**: frontier models already in cooperative regime ($r=+0.72$ across 34 models) [source:arxiv:2605.18838], but irreducible tax $\tau_0$ may persist [source:arxiv:2603.00047].
- No mitigation fully eliminates the tax; theoretical lower bound unknown [source:arxiv:2309.06256].

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [KL regularization in RLHF](kl-regularization.md)
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [RL for reasoning models](rl-for-reasoning.md)

## References
- [source:forum:current-work-in-ai-alignment] [Current work in AI alignment](https://forum.effectivealtruism.org/posts/63stBTw3WAW6k45dY/paul-christiano-current-work-in-ai-alignment)
- [source:arxiv:2112.00861] [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2603.00047] [What Is the Alignment Tax?](https://arxiv.org/abs/2603.00047)
- [source:arxiv:2602.07892] [Safety Alignment as Continual Learning: Mitigating the Alignment Tax via Orthogonal Gradient Projection](https://arxiv.org/abs/2602.07892)
- [source:arxiv:2309.06256] [Mitigating the Alignment Tax of RLHF](https://arxiv.org/abs/2309.06256)
- [source:arxiv:2605.18838] [Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling](https://arxiv.org/abs/2605.18838)
- [source:arxiv:2305.20050] [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)
