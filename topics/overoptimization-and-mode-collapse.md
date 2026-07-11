---
title: Over-optimization and mode collapse
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2310.06452
- arxiv:2605.19461
- openaccess:diversegrpo-mitigating-mode-collapse-in-
- lesswrong:mode-collapse-in-rl-may-be-fueled-by-the
- bmva-archive:overcoming-mode-collapse-with-adaptive-m
- researchgate:on-the-algorithmic-bias-of-aligning-larg
- arxiv:2510.01171
open_questions:
- Can distribution-matching objectives (forward KL approximations) be made tractable
  for open-ended RLHF with learned reward models, or are they fundamentally limited
  to verifiable-reward settings?
- Does Verbalized Sampling's diversity recovery generalize to *reasoning* tasks where
  the "correct" distribution is peaked, or does it primarily benefit creative/open-ended
  generation?
- How do typicality bias (reward model) and reverse KL (policy update) *interact*—does
  mitigating one amplify the other, and can they be jointly corrected?
- Is there a unified theoretical framework characterizing the *Pareto frontier* of
  diversity vs. reward across all these mitigation strategies, or are they addressing
  orthogonal collapse modes?
---

Over-optimization in RL fine-tuning drives models toward high-reward but low-entropy policies, systematically collapsing the output distribution onto a narrow set of modes. This phenomenon—mode collapse—manifests as loss of per-input and cross-input diversity, degradation of reasoning chain length, and suppression of minority preferences, and it arises from distinct but interacting mechanisms in the reward model, the policy update rule, and the regularization scheme.

## Mechanisms of Mode Collapse in RL Fine-Tuning

### Reverse KL Minimization and Mode-Seeking Behavior

On-policy RL algorithms such as PPO and GRPO implicitly minimize the reverse KL divergence $D_{\text{KL}}(\pi_\theta \| p^*)$ between the policy and the reward-weighted target distribution $p^* \propto \pi_{\text{ref}} \exp(r/\beta)$ [source:arxiv:2605.19461]. Reverse KL is *mode-seeking*: it forces the policy to cover only the highest-density regions of $p^*$, discarding low-probability but valid modes [source:arxiv:2605.19461]. In contrast, forward KL $D_{\text{KL}}(p^* \| \pi_\theta)$ is *mode-covering* and would preserve the full support of $p^*$, but sampling from $p^*$ is intractable for LLMs [source:arxiv:2605.19461]. This asymmetry explains why GRPO "concentrates its probability mass on the first high-reward trajectory it discovers, prematurely halting exploration of alternative strategies" [source:arxiv:2605.19461].

### Typicality Bias in Reward Models

Reward models trained on human preferences inherit a *typicality bias*: annotators systematically prefer text that is familiar, fluent, and predictable under the pretrained distribution $\pi_{\text{ref}}$ [source:arxiv:2510.01171]. Formally, the learned reward decomposes as $r(x,y) = r_{\text{true}}(x,y) + \alpha \log \pi_{\text{ref}}(y|x) + \epsilon(x)$ with $\alpha > 0$ [source:arxiv:2510.01171]. When this reward is used in a KL-regularized RLHF objective, the optimal policy becomes $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)^\gamma \exp(r_{\text{true}}(x,y)/\beta)$ where $\gamma := 1 + \alpha/\beta > 1$ [source:arxiv:2510.01171]. The exponent $\gamma > 1$ *sharpens* the pretrained distribution, compressing probability mass toward the most typical completions and causing mode collapse even when multiple responses have equal true utility [source:arxiv:2510.01171]. This is a *data-level* driver distinct from algorithmic pathologies.

### Unbounded Logit Growth in Policy Gradient Updates

Standard PPO updates can drive logits of high-reward actions toward infinity because the state-value baseline $v^\pi(s)$ creates a moving target [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the]. In a two-action bandit with rewards $R(a_1)=1, R(a_2)=0.5$, taking the lower-reward action lowers $v^\pi(s)$; subsequently taking the higher-reward action yields a positive advantage $A^\pi(s,a_1)$ relative to the depressed baseline, reinforcing $a_1$ again [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the]. This cycle repeats indefinitely, extracting "an unbounded amount of reinforcement from a single type of experience" and penalizing exploration [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the]. The proposed fix, Action-Conditioned TD Error (ACTDE), replaces $v^\pi(s)$ with $q^\pi(s,a)$ so that "a single reward event provides a finite amount of updating" [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the]. In tabular settings ACTDE converges to a softmax-over-rewards policy (e.g., $\{27\%, 73\%\}$ for rewards 10 and 11) rather than a greedy one-hot policy [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the].

### Discriminator Catastrophic Forgetting in Adversarial Settings

In GANs, mode collapse is driven by *catastrophic forgetting* in the discriminator: as the generator shifts to new modes, the discriminator loses classification accuracy on previously seen modes, creating an oscillatory trajectory where the generator cycles between modes without converging [source:bmva-archive:overcoming-mode-collapse-with-adaptive-m]. The Adaptive Multi Adversarial Training (AMAT) framework detects forgetting by monitoring discriminator scores on a fixed set of *exemplar* samples (one per mode) and spawns new discriminators when $\max(\mathfrak{s}[k]) / \text{avg}(\mathfrak{s}[k]) > \alpha_t$ [source:bmva-archive:overcoming-mode-collapse-with-adaptive-m]. This mechanism—discriminator forgetting causing generator oscillation—has no direct analogue in RLHF but illustrates how *evaluation signal degradation* can induce collapse.

## Manifestations of Diversity Loss

### Per-Input and Cross-Input Diversity Collapse

RLHF reduces both *per-input diversity* (variance of outputs for a fixed prompt) and *cross-input diversity* (variance of outputs across different prompts), the latter providing "empirical evidence of mode collapse, where the model produces similar styles of text regardless of the input" [source:arxiv:2310.06452]. Metrics include Expectation-Adjusted Distinct N-grams (EAD) for syntactic diversity, Sentence-BERT cosine similarity for semantic diversity, and NLI diversity for logical diversity [source:arxiv:2310.06452]. Sweeping the KL penalty coefficient $\beta_{\text{KL}}$ revealed that *higher* penalties resulted in *less* output diversity and generally worse performance, indicating that "the KL penalty cannot be used to trade off diversity for generalisation" [source:arxiv:2310.06452].

### Length Collapse and Reasoning Chain Degradation

In reasoning tasks, GRPO exhibits *length collapse*: response lengths degenerate from $\sim 600$ tokens to $<200$ tokens as training progresses, truncating reasoning chains [source:arxiv:2605.19461]. DMPO maintains robust chains of $\sim 400$ tokens by aligning the group-level policy distribution $q_\theta$ to a Boltzmann target $p$ via MSE loss $\mathcal{L}_{\text{DM}} = \frac{1}{G}\sum_i (p(o_i|\mathcal{O}) - q_\theta(o_i|\mathcal{O}))^2$ [source:arxiv:2605.19461]. The length-normalized log-likelihood $\phi(o_i) = \frac{1}{|o_i|}\sum_t \log \pi_\theta(o_{i,t}|o_{i,<t}, x)$ prevents the exponential decay of probability with trajectory length from biasing the group distribution [source:arxiv:2605.19461].

### Preference Collapse and Minority Viewpoint Suppression

RLHF's KL-regularized optimization systematically suppresses minority viewpoints in the preference data, a phenomenon termed *preference collapse* [source:researchgate:on-the-algorithmic-bias-of-aligning-larg]. Under the Bradley-Terry model, maximizing expected reward without KL regularization leads to collapse that "disproportionately impacts underrepresented groups in the preference data" (Theorem 5.2) [source:researchgate:on-the-algorithmic-bias-of-aligning-larg]. Game-theoretic alignment (NLHF) models alignment as a two-player zero-sum game and uses *mixed strategies* to ensure diversity, achieving Condorcet and Smith consistency where RLHF fails [source:researchgate:on-the-algorithmic-bias-of-aligning-larg]. However, exact *preference matching*—the model output matching a target policy that fully accounts for preference diversity—is provably impossible: "no smooth and learnable mappings of pairwise preferences can guarantee a unique Nash equilibrium that matches a target policy, even when utilizing standard assumptions such as the Bradley-Terry-Luce model" [source:researchgate:on-the-algorithmic-bias-of-aligning-larg].

## Mitigation Strategies

### Distribution Matching and Forward KL Approximation

DMPO approximates forward KL minimization by constructing a *group-conditional* Boltzmann target $p(o_i|\mathcal{O}) = \exp(r(o_i)/\alpha) / \sum_j \exp(r(o_j)/\alpha)$ and aligning the normalized policy distribution $q_\theta$ to it via MSE [source:arxiv:2605.19461]. The combined loss $\mathcal{L}_{\text{DMPO}} = \mathcal{L}_{\text{GRPO}} + \lambda \mathcal{L}_{\text{DM}}$ balances exploitation and exploration [source:arxiv:2605.19461]. On MM-NP-Bench (10 NP-hard combinatorial tasks), DMPO achieves 43.1% Quality Ratio vs. GRPO's 38.4% (+12% relative) and 61.9% Success Rate vs. 55.7% [source:arxiv:2605.19461]. It also improves mathematical reasoning by 2.0% avg across six benchmarks and out-of-domain visual reasoning by 2.3% avg across seven benchmarks [source:arxiv:2605.19461]. Limitation: requires exact, rule-based verifiable rewards; extension to learned/subjective rewards is open [source:arxiv:2605.19461].

### Inference-Time Verbalized Sampling

Verbalized Sampling (VS) is a *training-free* prompting strategy that elicits diverse outputs by asking the model to generate $k$ responses *with their probabilities* (e.g., "Generate 5 jokes about coffee and their corresponding probabilities") [source:arxiv:2510.01171]. Instance-level prompts collapse to a single mode; distribution-level prompts approximate the broader pretrained distribution [source:arxiv:2510.01171]. VS increases semantic diversity by 1.6–2.1$\times$ over direct prompting and improves human evaluation by 25.7% on creative writing [source:arxiv:2510.01171]. On Tulu-3 after DPO, VS retains 66.8% of base model diversity vs. 23.8% for direct prompting [source:arxiv:2510.01171]. For enumerative QA, VS-elicited distributions match pretraining corpora (KL=0.12 for Claude-4-Sonnet) [source:arxiv:2510.01171]. Larger models benefit more (1.5–2$\times$ diversity gains) [source:arxiv:2510.01171]. Trade-off: increased inference latency and token usage; smaller models may lack probability estimation capability [source:arxiv:2510.01171].

### Distributional Creativity Bonuses and Clustering

DiverseGRPO (for image generation) adds an *exploration reward* based on semantic clustering of the generated group [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. Pairwise DreamSim distances $d_{ij}$ form an affinity matrix $A_{ij} = \exp(-d_{ij}^2/2\sigma^2)$; spectral clustering on the normalized Laplacian $L = D^{-1/2}AD^{-1/2}$ yields clusters $C_k$ [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. Images in cluster $C_k$ receive bonus $E_i = \sqrt{N/n_k}$ where $n_k = |C_k|$, so rare clusters get higher reward [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. Final reward $\mathsf{R}_i = Q_i + \beta E_i$ [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. On SD3.5-M/PickScore: DreamSim +18.8%, FID +23.3%, BeyondFID +184.2%, SSIM +25.6% [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. On Flux.1-dev: DreamSim +13.9%, FID +9.1% [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. Reaches same quality in 400 vs. 1,280 iterations (baseline KL=0.01) while reducing mode collapse by 9% [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. Saturation: diversity gains level off after $\beta=5$; increasing SA-Reg steps $K$ has diminishing returns [source:openaccess:diversegrpo-mitigating-mode-collapse-in-].

### Structure-Aware Regularization

DiverseGRPO replaces the uniform KL penalty with a *stage-dependent Wasserstein constraint* concentrated on early denoising steps where diffusion variance $\sigma^2$ is largest and diversity is determined [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]:

$$
\mathcal{L}_{\mathtt{reg}}(t) = \begin{cases} \frac{\|\bar{\mathbf{x}}_{t+\Delta t,\theta} - \bar{\mathbf{x}}_{t+\Delta t,\mathtt{ref}}\|^2}{2}, & t \le K \\ 0, & t > K \end{cases}
$$

This forces semantic coverage early while allowing free optimization for fidelity later [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. The insight—that KL regularization is misaligned with the diffusion process because it is weakest when the diversity budget is most critical—is specific to diffusion but suggests a general principle: *regularization should be allocated where it most affects diversity*.

### Action-Conditioned Value Baselines

ACTDE replaces the state-value baseline $v^\pi(s)$ with an action-value baseline $q^\pi(s,a)$ in the advantage estimator:

$$
A^{\pi*}(s,a) := \mathbb{E}_{s'}[R(s,a,s') + \gamma v^\pi(s')] - q^\pi(s,a)
$$

This mimics reward prediction error and bounds the total reinforcement per experience [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the]. In tabular bandits it converges to softmax-over-rewards; in iterated Prisoner's Dilemma it avoids pure-strategy collapse but shows "unclear convergence properties and may potentially converge to uniform policies" in non-tabular settings [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the]. Sensitivity to whitening, head detachment, and loss choice (PPO vs. ILQL) is high; authors speculate it "might not perform well for RLHF at scale due to the finicky nature of deep RL" [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the].

### Adaptive Multi-Discriminator Frameworks

AMAT spawns discriminators adaptively when forgetting is detected on exemplar samples, assigning responsibility via $\epsilon$-greedy (fake) or uniform (real) sampling, and weighting generator updates by Dirichlet-sampled weights sorted so the most specialized discriminator gets highest weight [source:bmva-archive:overcoming-mode-collapse-with-adaptive-m]. On Stacked MNIST: 1000±0.0 modes covered, KL=0.078±0.01 [source:bmva-archive:overcoming-mode-collapse-with-adaptive-m]. On CIFAR10: BigGAN IS 9.22→9.51, FID 8.94→6.11; SN-GAN IS 8.22→8.34, FID 14.21→13.8; WGAN-GP IS 7.59→7.80, FID 19.2→17.2 [source:bmva-archive:overcoming-mode-collapse-with-adaptive-m]. Per-class FID gains largest for previously weak classes (Frog 56.1%, Truck 42.1%) [source:bmva-archive:overcoming-mode-collapse-with-adaptive-m]. Hyperparameter sensitivity: >7 discriminators degrades performance; warmup $T_t$ too long causes late spawning; pure greedy ($\epsilon=0$) worse than $\epsilon$-greedy; 1-hot weighting worse than soft Dirichlet [source:bmva-archive:overcoming-mode-collapse-with-adaptive-m].

### Game-Theoretic Alignment with Mixed Strategies

NLHF frames alignment as a two-player zero-sum game with payoffs derived from pairwise preferences, using mixed strategies to ensure diversity [source:researchgate:on-the-algorithmic-bias-of-aligning-larg]. It achieves Condorcet consistency (selects candidate beating all others pairwise) and Smith consistency (selects from minimal dominant set) where RLHF fails both [source:researchgate:on-the-algorithmic-bias-of-aligning-larg]. However, the impossibility of exact preference matching means "the framework cannot guarantee a unique equilibrium that perfectly mirrors a diverse target policy" [source:researchgate:on-the-algorithmic-bias-of-aligning-larg].

## Current Status and Trajectory

Mode collapse mitigation is an *active, fragmented research frontier* rather than a settled practice. Distribution-matching methods (DMPO) show strong gains on verifiable-reward reasoning tasks but have not been demonstrated on open-ended RLHF with learned rewards [source:arxiv:2605.19461]. Inference-time prompting (VS) is immediately deployable and scales with model size, but compute cost and small-model reliability limit adoption [source:arxiv:2510.01171]. Creativity bonuses and structure-aware regularization (DiverseGRPO) are established in diffusion-based image generation but untested for LLM token generation [source:openaccess:diversegrpo-mitigating-mode-collapse-in-]. ACTDE remains a theoretical curiosity with "finicky" deep-RL behavior and no large-scale validation [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the]. Game-theoretic alignment (NLHF) offers desirable social-choice properties but faces an impossibility result for exact preference matching [source:researchgate:on-the-algorithmic-bias-of-aligning-larg]. No single approach dominates; the field is exploring *combinations* (e.g., DMPO+VS, or creativity bonuses in GRPO for LLMs) but integration results are not widely reported.

## Key Takeaways

- Mode collapse in RL fine-tuning has at least four distinct mechanistic roots: reverse KL mode-seeking, typicality bias in reward models, unbounded logit growth from state-value baselines, and evaluator forgetting in adversarial setups.
- RLHF reduces both per-input and cross-input diversity; increasing the KL penalty $\beta_{\text{KL}}$ *worsens* diversity, contradicting the intuition that it controls the diversity–quality trade-off [source:arxiv:2310.06452].
- Length collapse (reasoning chain truncation) is a measurable symptom of mode collapse in reasoning models; DMPO's length-normalized group distribution matching mitigates it [source:arxiv:2605.19461].
- Preference collapse suppresses minority viewpoints; game-theoretic mixed strategies (NLHF) improve diversity but cannot achieve exact preference matching [source:researchgate:on-the-algorithmic-bias-of-aligning-larg].
- Training-free inference-time methods (Verbalized Sampling) recover substantial pretrained diversity (66.8% vs 23.8% on Tulu-3) and scale with model size [source:arxiv:2510.01171].
- Diffusion-specific insights (early-step regularization, spectral clustering bonuses) may transfer to LLMs via timestep-analogous token-position regularization, but this is unproven [source:openaccess:diversegrpo-mitigating-mode-collapse-in-].
- ACTDE's bounded-update property is theoretically appealing for tabular cases but shows unstable convergence in function approximation; not ready for production RLHF [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the].

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Length and format bias](length-and-format-bias.md)
- [The alignment tax](alignment-tax.md)
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md)
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [DPO variants deep-dive](dpo-variants.md)
- [RLAIF (RL from AI feedback)](rlaif.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Nash and game-theoretic preference optimization](nash-and-game-theoretic-po.md)
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)

## References
- [source:arxiv:2310.06452] [Understanding the Effects of RLHF on LLM Generalisation and Diversity](https://arxiv.org/abs/2310.06452)
- [source:arxiv:2605.19461] [Beyond Mode Collapse: Distribution Matching for Diverse Reasoning](https://arxiv.org/html/2605.19461v1)
- [source:openaccess:diversegrpo-mitigating-mode-collapse-in-] [DiverseGRPO: Mitigating Mode Collapse in Image Generation via Diversity-Aware GRPO](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_DiverseGRPO_Mitigating_Mode_Collapse_in_Image_Generation_via_Diversity-Aware_GRPO_CVPR_2026_paper.pdf)
- [source:lesswrong:mode-collapse-in-rl-may-be-fueled-by-the] [Mode collapse in RL may be fueled by the update equation](https://www.lesswrong.com/posts/A7RgYuYH4HywNeYWD/mode-collapse-in-rl-may-be-fueled-by-the-update-equation)
- [source:bmva-archive:overcoming-mode-collapse-with-adaptive-m] [Overcoming Mode Collapse with Adaptive Multi Adversarial Training](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0690.pdf)
- [source:researchgate:on-the-algorithmic-bias-of-aligning-larg] [On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference Collapse and Matching Regularization](https://www.researchgate.net/publication/398985488_On_the_Algorithmic_Bias_of_Aligning_Large_Language_Models_with_RLHF_Preference_Collapse_and_Matching_Regularization)
- [source:arxiv:2510.01171] [How to Mitigate Mode Collapse and Unlock LLM Diversity](https://arxiv.org/html/2510.01171v1)
