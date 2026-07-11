---
title: Entropy and exploration in RL fine-tuning
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2510.08141
- machinelearning:entropy-preserving-reinforcement-learnin
- arxiv:2604.13902
- emergentmind:exploration-collapse-in-reinforcement-le
- emergentmind:diversity-collapse-in-rl-emergent-mind
- github:awesome-exploration-methods-in-reinforce
open_questions:
- What is the optimal entropy target $\mathcal{H}_0$ as a function of model scale,
  task complexity, and training stage, and can it be annealed dynamically without
  instability?
- Do mass-covering $f$-divergences (forward-KL, JS) in DPH-RL provably maintain support
  coverage in the high-dimensional, non-convex policy manifolds of LLMs, or do they
  merely delay collapse?
- How do perplexity-based disentanglement methods (DiPO) generalize to open-ended
  generation tasks where verifiable rewards are unavailable and PPL–correctness correlation
  breaks down?
- Can exploration bonuses derived from prediction error (RND/ICM) or information gain
  be effectively computed in the token-level action space of LLMs without prohibitive
  overhead?
---

Entropy collapse and diversity loss are central failure modes in RL fine-tuning of LLMs, where policy optimization drives distributions toward deterministic, low-support solutions that degrade multi-sample performance and generalization. Modern methods address this through adaptive entropy control, perplexity-disentanglement, and diversity-preserving objectives that explicitly target the exploration–exploitation trade-off at both token and trajectory levels.

## Entropy Collapse: Mechanisms and Diagnostics

Entropy collapse manifests as a sustained decline in policy entropy $\mathcal{H}(\pi(\cdot|s)) = -\sum_{a \in \mathcal{A}} \pi(a|s) \log \pi(a|s)$ over training iterations, preventing discovery of novel solutions and stagnating policy improvement [source:emergentmind:exploration-collapse-in-reinforcement-le]. In LLM reasoning with verifiable rewards (RLVR), this occurs at two levels: **token-level entropy collapse**, where the policy becomes nearly deterministic and fails to sample alternative generation pathways, and **outcome-level mode collapse**, where the agent converges to a tiny subset of reward-supporting solutions such that $|\mathcal{M}| \ll |\mathcal{S}_{\text{reward}}|$ [source:emergentmind:exploration-collapse-in-reinforcement-le].

The collapse is driven by structural issues in the RL objective rather than hyperparameter choices. Standard expected return maximization creates an "outcome-frequency multiplier" that exerts exponential selective pressure on the most probable outcomes [source:emergentmind:exploration-collapse-in-reinforcement-le]. In GRPO-style updates, softmax dynamics cause **positive sharpening** (increasing logits of sampled tokens) and **negative squeezing** (redistributing mass away from rejected tokens), which can lead to irreversible contraction where gradients vanish once a mode's probability drops below a sampling threshold [source:emergentmind:exploration-collapse-in-reinforcement-le].

Diversity collapse, a closely related phenomenon, is mathematically rooted in **selection bias** (preferring high-base-probability trajectories) and **reinforcement bias** (amplifying those trajectories in proportion to initial likelihood) [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. In continuous policies, unimodal architectures (Gaussian or deterministic) tend to converge to a single mode under distributional shift [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. Key diagnostics include:
- **Representation rank**: Rank deficiency of Gram matrices of behavioral embeddings or vanishing determinant indicates behaviors span only a low-dimensional subspace [source:emergentmind:diversity-collapse-in-rl-emergent-mind].
- **Support shrinkage**: In discrete spaces, RL compresses incorrect trajectories with a reported compression ratio $R_- \approx 0.25\text{--}0.35$ [source:emergentmind:diversity-collapse-in-rl-emergent-mind].
- **Pass@k degradation**: Improvement in single-sample success (Pass@1) often comes at the cost of sharply reduced multi-attempt success (Pass@k) [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

## Exploration Bonuses and Intrinsic Motivation

Classical exploration methods in RL are taxonomized by whether they operate in the **collect phase** (interacting with the environment) or the **train phase** (updating the policy) [source:github:awesome-exploration-methods-in-reinforce]. For LLM fine-tuning, the most relevant categories are:

| Phase | Category | Mechanism | LLM Adaptation |
|-------|----------|-----------|----------------|
| Collect | Action Selection Perturbation | Noise/randomness in action choices | Temperature sampling, nucleus sampling |
| Collect | Action Selection Guidance | Heuristics/signals to guide choices | Process reward models, verifier-guided decoding |
| Collect | State Selection Guidance | Directing agent to novel/high-value states | Prompt selection, curriculum learning |
| Train | Count Based | Visit counts for rarely visited states | N-gram diversity penalties, trajectory deduplication |
| Train | Prediction Based | Prediction errors as intrinsic rewards (RND, ICM) | Model disagreement, uncertainty quantification |
| Train | Entropy Augmented | Entropy terms in objective (e.g., SAC) | Entropy bonuses, KL regularization |
| Train | Information Theory Based | Empowerment, mutual information | Diversity rewards, determinantal point processes |

Recent trends (2025–2026) emphasize **LLM-based exploration**: "exploration hacking" in LLMs, rubric-scaffolded RL for reasoning, and latent exploration decoding for large reasoning models [source:github:awesome-exploration-methods-in-reinforce]. Advanced intrinsic motivation now includes diffusion models for unsupervised RL, spectral Bellman methods, and graph-theoretic intrinsic rewards based on effective resistance [source:github:awesome-exploration-methods-in-reinforce].

## Diversity Preservation Methods

Modern frameworks move beyond global entropy bonuses (which often sacrifice correctness) and pairwise novelty (vulnerable to clustering) toward principled diversity preservation [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Differential Smoothing (DS)
DS modifies rewards to target only correct trajectories: it penalizes high-base-probability correct traces while sharpening mass on incorrect ones to prevent "sharpening" onto a limited set of trajectories [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. DS-GRPO demonstrated improvements up to $+6.7\%$ in Pass@K on real-world mathematical reasoning datasets [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Mode Anchored Reward Augmentation (MARA)
MARA edits the reward landscape to ensure the KL target distribution remains flat across all high-reward modes [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. It maintained near-uniform entropy and Pareto-optimal reward/diversity in drug discovery and creative QA [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Diversity-Preserving Hybrid RL (DPH-RL)
DPH-RL replaces standard reverse-KL regularization with mass-covering $f$-divergences (forward-KL or Jensen–Shannon), which continuously reference the base policy to maintain support across initial modes [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. It successfully prevented Pass@k degradation and catastrophic forgetting in SQL and math tasks [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Diversity via Determinants (DvD)
DvD maximizes the volume of the behavioral manifold by computing the determinant of the agent Gram matrix, ensuring agents are not linearly redundant [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. DvD-TD3 outperformed all baselines in Humanoid-v2 by retaining both forward and backward behaviors [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Proximal Feature Optimization (PFO)
PFO controls feature-rank decay to maintain network plasticity and prevent representation collapse [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### REPO and ADAPO
The entropy-preserving RL framework proposes two algorithm families: **REPO**, which regulates entropy by modifying the advantage function, and **ADAPO**, which uses adaptive asymmetric clipping [source:machinelearning:entropy-preserving-reinforcement-learnin]. Models trained with these methods maintain trajectory diversity, achieve higher final performance, and retain "trainability" in sequential learning tasks [source:machinelearning:entropy-preserving-reinforcement-learnin].

## Temperature and Sampling Strategies

SCOPE-RL identifies a **temperature–sign asymmetry** in entropy dynamics: under high-temperature sampling, positive samples promote entropy growth while negative samples accelerate collapse; under low-temperature sampling, negative samples increase entropy but cause rapid reward collapse [source:arxiv:2510.08141]. SCOPE-RL leverages this via:

1. **Monitor entropy** $\mathcal{H}(\pi_{\theta_{\text{old}}})$ of the previous step.
2. **Adaptive temperature scaling**: $T = \text{clip}(1 + \mathcal{H}_0 - \mathcal{H}(\pi_{\theta_{\text{old}}}), 0.8, 1.2)$ where $\mathcal{H}_0$ is a target entropy threshold [source:arxiv:2510.08141].
3. **Positive sample filtering**: Draw auxiliary samples at temperature $T$, retain only those with $R=1$.
4. **Objective integration**: Augment GRPO with a regularization term from temperature-adjusted positive samples:

$$
\mathcal{J}_{\text{SCOPE}}(\theta) = \mathcal{J}_{\text{GRPO}}(\theta) + \alpha \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G'} \sim \pi_{\theta_{\text{old}}}^T(O|q)} \left[ \frac{1}{G'} \sum_{i=1}^{G'} \mathbf{1}[R(q, o_i) = 1] \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta), \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)) \right]
$$

where $r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ and $\alpha$ is the auxiliary sample ratio [source:arxiv:2510.08141].

With $\alpha = 1/64$ (<2% additional samples), SCOPE-RL achieved consistent gains: Qwen2.5-Math-7B average score $51.60 \to 54.84$ (+3.23), Qwen2.5-7B $47.71 \to 49.81$ (+2.10), Qwen3-4B $67.77 \to 69.73$ (+1.96) [source:arxiv:2510.08141]. Pass@1024 on Qwen2.5-7B improved dramatically: AIME24 $73.3 \to 86.7$, AIME25 $63.3 \to 76.7$ [source:arxiv:2510.08141]. The entropy–performance relationship is non-monotonic: best average performance at $\mathcal{H}_0 = 0.50$ (54.84), outperforming $\mathcal{H}_0 = 0.25$ (53.00) and $\mathcal{H}_0 = 0.75$ (53.45) [source:arxiv:2510.08141].

DiPO introduces **Perplexity Space Disentangling (PSD)** and **Bidirectional Reward Reallocation (BRR)** for fine-grained exploration–exploitation trade-off [source:arxiv:2604.13902]. PSD partitions samples into Exploitation Space (EiS) and Exploration Space (ErS) via a dynamic threshold $\tau^*$ optimized from a cached PPL queue $\mathcal{Q}$:

$$
\tau^* = \arg \min_{\tau} \frac{1}{|\mathcal{Q}|} \sum_{(r_i, p_i) \in \mathcal{Q}} |r_i - \mathbb{I}(p_i < \tau)|
$$

with advantage judgments ensuring PPL correlates with correctness [source:arxiv:2604.13902]. BRR reallocates rewards only for zero-gradient groups: in hard groups (all 0) within EiS, the max-PPL sample gets reward 1 to encourage exploration; in easy groups (all 1) within ErS, the max-PPL sample gets reward 0 to encourage exploitation [source:arxiv:2604.13902]. The objective combines DAPO with reallocated rewards weighted by $\alpha$:

$$
\mathcal{J}_{\text{DiPO}}(\theta) = \mathcal{J}_{\text{DAPO}}(\theta, \mathcal{R}) + \alpha \times \mathcal{J}_{\text{DAPO}}(\theta, \mathcal{R}_r)
$$

Optimal $\alpha = 0.1$; DiPO showed higher robustness than entropy loss (10× $\alpha$ increase caused only 1.43 point drop vs. catastrophic collapse for entropy loss) [source:arxiv:2604.13902]. Results: Qwen3-4B-Base 50.55% vs. DAPO 49.43% and GRPO 48.92%; Qwen3-8B-Base 54.79% vs. 53.23%/53.24%; function calling on BFCLv3: Qwen2.5-3B 55.03% vs. 53.21%, Qwen2.5-7B 62.51% vs. 61.06% [source:arxiv:2604.13902].

Other algorithmic remedies for exploration collapse include:
- **Low-probability Regularization (Lp-Reg)**: Protects "reasoning sparks" via forward-KL and proxy-KL on filtered low-probability tokens; 60.17% average accuracy across five math tasks (+2.66% over standard entropy controls) [source:emergentmind:exploration-collapse-in-reinforcement-le].
- **Inverse Probability Scaling (IPS)**: Weights learning signal by $1/p$ to remove self-reinforcing gradient, converging to reward-proportional stationary distribution; improved recovery rates and coverage in molecule design [source:emergentmind:exploration-collapse-in-reinforcement-le].
- **Unified Entropy Control (UEC-RL)**: High-temperature exploration for difficult prompts + replay-based entropy consolidation; 37.9% relative improvement over GRPO on Geometry3K [source:emergentmind:exploration-collapse-in-reinforcement-le].
- **Dual-Scale Diversity Regularization (DSDR)**: Couples global and local diversity via reward shaping on correct diverse trajectories and token-level entropy; outperformed on Pass@1 and Pass@k with widening gap at higher $k$ [source:emergentmind:exploration-collapse-in-reinforcement-le].
- **Anchored Policy Optimization (APO)**: Support manifold pull + elastic recovery for valid alternatives [source:emergentmind:exploration-collapse-in-reinforcement-le].

## Current Status and Trajectory

Entropy and diversity control in RL fine-tuning is a **rising, actively researched area** with multiple competing frameworks rather than a settled default. SCOPE-RL [source:arxiv:2510.08141] and DiPO [source:arxiv:2604.13902] represent 2025–2026 advances targeting GRPO/DAPO pipelines specifically, while REPO/ADAPO [source:machinelearning:entropy-preserving-reinforcement-learnin] and the diversity-preservation suite (DS, MARA, DPH-RL, DvD, PFO) [source:emergentmind:diversity-collapse-in-rl-emergent-mind] offer broader algorithmic templates. The field has not converged on a single standard: entropy bonuses remain common but are known to cause instability (oscillation between collapse and explosion) [source:arxiv:2510.08141]; asymmetric clipping helps but lacks adaptive control [source:arxiv:2510.08141]; perplexity-based disentanglement (DiPO) shows promise but lacks strict theoretical guarantees [source:arxiv:2604.13902]. The taxonomy of exploration methods continues to expand, with LLM-specific adaptations (latent exploration decoding, rubric-scaffolded RL) appearing in 2025–2026 literature [source:github:awesome-exploration-methods-in-reinforce]. **Not widely reported** is large-scale ablation comparing these methods under identical compute/data budgets; most papers compare against GRPO/DAPO baselines rather than each other. Theoretical convergence guarantees for support-preserving regularization in high-dimensional LLM policy spaces remain **not established** [source:emergentmind:exploration-collapse-in-reinforcement-le; emergentmind:diversity-collapse-in-rl-emergent-mind].

## Key Takeaways

- Entropy collapse in LLM RL fine-tuning is structural: outcome-frequency multipliers and softmax sharpening/squeezing dynamics drive irreversible concentration [source:emergentmind:exploration-collapse-in-reinforcement-le].
- Token-level entropy ($H \to 0$) and outcome-level mode collapse ($|\mathcal{M}| \ll |\mathcal{S}_{\text{reward}}|$) are distinct but coupled failure modes [source:emergentmind:exploration-collapse-in-reinforcement-le].
- Diversity collapse manifests as Pass@1↑ / Pass@k↓ trade-offs, representation rank deficiency, and support shrinkage ($R_- \approx 0.25\text{--}0.35$) [source:emergentmind:diversity-collapse-in-rl-emergent-mind].
- Adaptive temperature control (SCOPE-RL) exploits a temperature–sign asymmetry: high $T$ makes positive samples entropy-increasing, low $T$ makes negative samples entropy-increasing [source:arxiv:2510.08141].
- Perplexity-space disentanglement (DiPO) separates exploitation/exploration spaces via dynamic PPL thresholding and reallocates rewards only in zero-gradient groups [source:arxiv:2604.13902].
- Modern diversity preservation uses mass-covering $f$-divergences (DPH-RL), determinantal volume maximization (DvD), differential smoothing (DS), and mode-anchored rewards (MARA) rather than naive entropy bonuses [source:emergentmind:diversity-collapse-in-rl-emergent-mind].
- REPO/ADAPO modify advantage functions and clipping to preserve entropy, maintaining trainability in sequential tasks [source:machinelearning:entropy-preserving-reinforcement-learnin].
- Non-monotonic entropy–performance curves imply an optimal target entropy ($\mathcal{H}_0 \approx 0.5$ in SCOPE-RL), not maximal entropy [source:arxiv:2510.08141].
- Theoretical analyses rely on tabular softmax, binary rewards, orthogonal gradients, and often only cover $\Delta\mathcal{H} < 0$ regimes [source:arxiv:2510.08141; arxiv:2604.13902].

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Length and format bias](length-and-format-bias.md)

## References
- [source:arxiv:2510.08141] [Entropy Is Controllable in Reinforcement Fine-tuning](https://arxiv.org/html/2510.08141v3)
- [source:machinelearning:entropy-preserving-reinforcement-learnin] [Entropy-Preserving Reinforcement Learning](https://machinelearning.apple.com/research/entropy-preserving-reinforcement-learning)
- [source:arxiv:2604.13902] [DiPO: Disentangled Perplexity Policy Optimization for Fine-grained Fine-tuning](https://arxiv.org/html/2604.13902v1)
- [source:emergentmind:exploration-collapse-in-reinforcement-le] [Exploration Collapse in Reinforcement Learning (Emergent Mind)](https://www.emergentmind.com/topics/exploration-collapse)
- [source:emergentmind:diversity-collapse-in-rl-emergent-mind] [Diversity Collapse in RL (Emergent Mind)](https://www.emergentmind.com/topics/diversity-collapse-in-rl)
- [source:github:awesome-exploration-methods-in-reinforce] [Awesome Exploration Methods in Reinforcement Learning (GitHub)](https://github.com/opendilab/awesome-exploration-rl)
