---
title: Direct Preference Optimization and variants
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2305.18290
- icml:alphadpo-adaptive-reward-margin-for-dire
- mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a
- mesuvash:reward-modeling-and-dpo-learning-what-go
- youtube:orpo-explained-superior-llm-alignment-te
- arxiv:2402.00164
open_questions:
- What is ORPO's actual mechanism, loss function, and memory profile? The provided
  sources do not describe it.
- Does DPO's dominance over PPO with ground-truth rewards hold at frontier model scales
  (70B+ parameters)?
- Can offline preference optimization methods be extended with efficient online exploration
  (e.g., via synthetic data generation or self-play) without reintroducing full RLHF
  complexity?
- How does AlphaDPO's adaptive reference perform on reasoning-heavy benchmarks (math,
  code) where verifiable rewards enable online RL?
---

Direct Preference Optimization (DPO) and its variants eliminate the explicit reward model and reinforcement learning loop of traditional RLHF by reparameterizing the optimal policy in closed form, turning preference learning into a supervised classification or regression problem on static preference data. This family of methods — DPO, IPO, KTO, ORPO, and AlphaDPO — trades the flexibility of online exploration for dramatic computational simplicity, offline training stability, and reduced memory footprint, at the cost of being bounded by the coverage and quality of the fixed preference dataset.

## Why skip the reward model?

Traditional RLHF requires three distinct models (policy, reward model, value network/critic) and an online RL loop where the policy generates samples that are scored by the reward model and optimized via PPO [source:arxiv:2305.18290]. This pipeline incurs 3–5× the compute of supervised fine-tuning (SFT), demands careful hyperparameter tuning for KL penalties and clipping, and suffers from reward hacking where the policy exploits imperfections in the learned reward model [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. DPO's central insight is that under the Bradley–Terry preference model and a KL-constrained reward maximization objective, the optimal policy has a closed-form relationship to the reference policy:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp\!\left(\frac{1}{\beta} r(x, y)\right)
$$

which can be inverted to express the reward *implicitly* as a log-ratio of policy probabilities [source:arxiv:2305.18290]:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)
$$

Substituting this into the Bradley–Terry likelihood yields a loss that depends *only* on the policy $\pi_\theta$ and the frozen reference $\pi_{\mathrm{ref}}$, with no separate reward model:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \right]
$$

The implicit reward is $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)}$ [source:arxiv:2305.18290]. This reduces the alignment pipeline to: collect preference pairs $(x, y_w, y_l)$ from a reference policy (typically the SFT model), then minimize a binary cross-entropy loss on the log-ratio differences. DPO runs at 1.5–2× the cost of SFT and requires only the policy and reference model in memory [source:mesuvash:reward-modeling-and-dpo-learning-what-go].

## DPO: mechanism and behavior

DPO optimizes the same KL-constrained objective as RLHF:

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \big[ r_\phi(x, y) \big] - \beta \, D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x)\big)
$$

but does so *offline* on a fixed dataset $\mathcal{D}$ [source:arxiv:2305.18290]. The reference model $\pi_{\mathrm{ref}}$ is typically the SFT checkpoint used to generate the preference pairs. DPO's gradient on a single pair $(y_w, y_l)$ increases the log-probability of $y_w$ relative to $\pi_{\mathrm{ref}}$ and decreases that of $y_l$, scaled by the sigmoid of the current margin [source:mesuvash:reward-modeling-and-dpo-learning-what-go].

**Empirical results (6B–13B scale):** On TL;DR summarization, DPO achieved ~61% win rate vs. PPO's 57% (both at temperature 0) and 58% human preference over PPO at temperature 0.25 [source:arxiv:2305.18290]. On Anthropic Helpful/Harmless dialogue, DPO was the only efficient method improving over the preferred completions in the dataset, matching Best-of-128 [source:arxiv:2305.18290]. Out-of-distribution generalization to CNN/DailyMail was stronger for DPO (win rate 0.36) than PPO (0.26) [source:arxiv:2305.18290].

**Limitations noted in the original paper:** DPO is an offline algorithm — it cannot explore or correct error types absent from the dataset [source:arxiv:2305.18290]. Its ceiling is bounded by dataset quality and diversity [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. The sigmoid loss saturates, causing vanishing gradients once a pair is correctly ordered, which can lead to overfitting and unbounded growth of the implicit reward gap [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. DPO also strictly requires paired comparisons, incompatible with unary feedback (thumbs-up/down) [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. Scaling beyond 6B parameters was untested in the original work [source:arxiv:2305.18290].

## IPO: bounding the reward gap

Identity Preference Optimization (IPO) addresses DPO's unbounded reward growth by replacing the monotonic sigmoid loss with a squared-error regression targeting a fixed margin [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. Define the log-ratio difference:

$$
h_\theta(x, y_w, y_l) = \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
$$

IPO's loss is:

$$
\mathcal{L}_{\mathrm{IPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( h_\theta - \frac{1}{2\beta} \right)^2 \right]
$$

The gradient $\partial \mathcal{L}_{\mathrm{IPO}} / \partial h_\theta = 2(h_\theta - 1/(2\beta))$ is zero at the target margin $1/(2\beta)$, creating a stable equilibrium: if the margin is too small the gradient pushes it up; if too large it pushes down [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. This prevents the extreme probabilities (chosen → 1, rejected → 0) that DPO's monotonic loss encourages. The target margin $1/(2\beta)$ derives from the KL regularization strength: larger $\beta$ (weaker KL penalty) → smaller target margin; smaller $\beta$ (stronger KL penalty) → larger permitted per-example margin [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. IPO retains DPO's paired-data requirement and reference-model memory cost.

## KTO: learning from unary feedback

Kahneman–Tversky Optimization (KTO) relaxes the paired-comparison requirement, using only binary "desirable/undesirable" labels on individual responses [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. Its loss (simplified) is:

$$
\mathcal{L}_{\mathrm{KTO}}(\theta) = \mathbb{E}_{y_w}\!\left[\sigma\!\left(-r_\theta(x, y_w)\right)\right] + \mathbb{E}_{y_l}\!\left[\sigma\!\left(r_\theta(x, y_l)\right)\right]
$$

where the per-example implicit reward is centered by a batch-wise baseline:

$$
r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} - \mathbb{E}_{y'}\!\left[\beta \log \frac{\pi_\theta(y' \mid x)}{\pi_{\mathrm{ref}}(y' \mid x)}\right]
$$

This centering approximates the normalization constant $Z(x)$ and enables learning from unpaired data [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. KTO is valuable when paired comparisons are unavailable or expensive, but the source does not report quantitative comparisons against DPO/IPO on standard benchmarks.

## ORPO: odds-ratio formulation

The provided sources do not contain a primary description of ORPO (Odds Ratio Preference Optimization). The video source labeled "ORPO Explained" actually describes DeepSeek's GRPO, a PPO variant with group-relative baselines and KL regularization [source:youtube:orpo-explained-superior-llm-alignment-te]. ORPO is mentioned only by name in the variants article as one of four DPO variants addressing DPO's limitations [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a], but its mechanism, loss function, and empirical results are not detailed in the available sources. **No valid source in the provided literature describes ORPO's mechanism or loss function; therefore, it cannot be characterized here.** This is a gap in the provided literature.

## AlphaDPO: adaptive reference and margin

AlphaDPO modifies DPO by introducing an *implicit*, adaptive reference distribution that evolves with the policy [source:icml:alphadpo-adaptive-reward-margin-for-dire]:

$$
\hat{\pi}_{\mathrm{ref}} \propto U(y \mid x) \left( \frac{\pi_\theta}{\pi_{\mathrm{ref}}} \right)^\alpha
$$

where $U(y \mid x)$ is a uniform base distribution and $\alpha$ controls the interpolation between uniform exploration ($\alpha \to 0$) and policy-driven specialization ($\alpha \to 1$) [source:icml:alphadpo-adaptive-reward-margin-for-dire]. This dynamic reference yields instance-adaptive reward margins (unlike SimPO's uniform margin) and theoretically controls sequential KL divergence between iterative updates, improving stability even with poorly calibrated initial references [source:icml:alphadpo-adaptive-reward-margin-for-dire]. Reported results: 58.7% LC win rate on AlpacaEval 2 and 35.7% on Arena-Hard across Mistral2-7B, Llama3-8B, and Gemma2-9B [source:icml:alphadpo-adaptive-reward-margin-for-dire]. The paper positions AlphaDPO as overcoming DPO's static-reference limitation and SimPO's uniform-margin limitation, but does not state AlphaDPO-specific limitations in the abstract.

## Disagreements and open tensions

- **DPO vs. PPO on ground-truth rewards:** The DPO paper reports DPO dominating PPO *even when PPO has access to ground-truth rewards* on sentiment generation [source:arxiv:2305.18290]. This contradicts the conventional wisdom that online RL with a perfect reward should outperform offline methods. The result is at 6B scale; whether it holds at frontier scale is not widely reported.
- **Offline vs. online ceiling:** DPO's authors acknowledge it cannot explore beyond the dataset [source:arxiv:2305.18290], while the AlphaDPO paper claims "robust alignment without requiring multi-stage training" [source:icml:alphadpo-adaptive-reward-margin-for-dire] — but AlphaDPO is also offline. The tension between offline simplicity and online exploration capability remains unresolved in the sources.
- **IPO's target margin derivation:** The IPO article derives $1/(2\beta)$ from the Bradley–Terry model and KL regularization [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a], but does not cite an original IPO paper (e.g., Azar et al. 2023) to confirm this matches the published method. The derivation is presented as self-contained; a primary-source check would settle it.
- **ORPO absence:** No source describes ORPO's mechanism. The variants article lists it as addressing DPO's computational cost (two models in VRAM) [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a], suggesting ORPO may eliminate the reference model, but this is unverified in the provided literature.
- **Reward over-optimization in DPO:** The DPO paper states "it is unclear how reward over-optimization manifests in the DPO setting" [source:arxiv:2305.18290], while the variants article asserts DPO's unbounded reward growth *is* a form of over-optimization [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. These are not direct contradictions but reflect different framings of the same phenomenon.

## Current status and trajectory

DPO and its variants are the **default** offline preference alignment methods as of 2024, widely adopted in open-source LLM alignment (Llama 3, Mistral, Gemma families) due to their 1.5–2× SFT cost, stability, and lack of reward-model training [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. IPO is commonly used as a drop-in replacement for DPO to prevent overfitting; KTO is adopted when only unary feedback is available. AlphaDPO's adaptive reference is a newer proposal (ICML 2024) with strong benchmark numbers but not yet widely reported in production deployments. **ORPO's status cannot be assessed from the provided sources — it is cited as a variant but not described.** The field is moving toward *hybrid* pipelines: offline DPO/IPO for initial alignment, followed by online RL (PPO/GRPO) for reasoning capabilities where verifiable rewards exist [source:youtube:orpo-explained-superior-llm-alignment-te]. Pure offline methods are not fading but are recognized as insufficient for advanced reasoning where exploration and process supervision matter.

## Key takeaways

- DPO eliminates the reward model and RL loop by reparameterizing the optimal KL-constrained policy as an implicit reward $\hat{r}_\theta = \beta \log \frac{\pi_\theta}{\pi_{\mathrm{ref}}}$, yielding a binary classification loss on preference pairs.
- IPO replaces DPO's saturating sigmoid loss with a squared-error loss targeting margin $1/(2\beta)$, creating a stable equilibrium that prevents unbounded reward growth and overconfident predictions.
- KTO enables learning from unary (desirable/undesirable) labels by centering the implicit reward with a batch-wise baseline, removing the paired-comparison requirement.
- AlphaDPO introduces an adaptive implicit reference $\hat{\pi}_{\mathrm{ref}} \propto U (\pi_\theta/\pi_{\mathrm{ref}})^\alpha$ that yields instance-adaptive margins and controls sequential KL divergence.
- All characterized methods (DPO, IPO, KTO, AlphaDPO) are offline, bounded by dataset coverage, and require the reference model in memory. **ORPO cannot be characterized from the provided sources — its mechanism, loss, and memory requirements are undocumented.**
- DPO at 6B–13B scale matched or exceeded PPO on summarization and dialogue, even when PPO used ground-truth rewards; scaling to frontier models is the primary unvalidated claim.

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [KL regularization in RLHF](kl-regularization.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [DPO variants deep-dive](dpo-variants.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)

## References
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:icml:alphadpo-adaptive-reward-margin-for-dire] [AlphaDPO: Adaptive Reward Margin for Direct Preference Optimization](https://icml.cc/virtual/2025/poster/45946)
- [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a] [DPO Variants: IPO, KTO, ORPO & cDPO for LLM Alignment](https://mbrenndoerfer.com/writing/dpo-variants-ipo-kto-orpo-cdpo-llm-alignment)
- [source:mesuvash:reward-modeling-and-dpo-learning-what-go] [Reward Modeling and DPO: Learning What "Good" Means](https://mesuvash.github.io/blog/2026/reward-modeling/)
- [source:youtube:orpo-explained-superior-llm-alignment-te] [ORPO Explained: Superior LLM Alignment Technique vs. DPO/RLHF](https://www.youtube.com/watch?v=32S_xBt2IXw)
- [source:arxiv:2402.00164] [Beyond DPO: A Survey of Preference-Based Alignment Algorithms for LLMs](https://arxiv.org/abs/2402.00164)
