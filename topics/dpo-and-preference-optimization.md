---
title: Direct Preference Optimization and variants
maturity: comprehensive
updated: '2026-07-12'
sources:
- youtube:orpo-explained-superior-llm-alignment-te
- mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a
- arxiv:2402.00164
- arxiv:2305.18290
- icml:alphadpo-adaptive-reward-margin-for-dire
- mesuvash:reward-modeling-and-dpo-learning-what-go
- arxiv:2310.12036
- arxiv:2402.01306
- arxiv:2403.07693
- arxiv:2401.06108
- cameronrwolfe:direct-preference-optimization-dpo-deep-
open_questions:
- Does DPO's advantage over PPO with ground-truth rewards persist at frontier model
  scales (70B+), or is it an artifact of 6B–13B scale?
- Can offline methods like AlphaDPO or IPO truly replace online RL for reasoning tasks,
  or is the hybrid pipeline (offline alignment → online RL) the only viable path?
- What is ORPO's actual mechanism, loss function, and memory profile? The provided
  literature cites it as a variant but does not describe it.
- Does IPO's squared-error loss maintain its stability advantages when scaled from
  bandit experiments to full LLM training on human preference data?
---

Direct Preference Optimization (DPO) and its variants eliminate the explicit reward model and reinforcement learning loop of traditional RLHF by reparameterizing the optimal policy in closed form, turning preference learning into a supervised classification or regression problem on static preference data. This family of methods — DPO, IPO, KTO, ORPO, and AlphaDPO — trades the flexibility of online exploration for dramatic computational simplicity, offline training stability, and reduced memory footprint, at the cost of being bounded by the coverage and quality of the fixed preference dataset.

## Why skip the reward model?

Traditional RLHF requires three distinct models (policy, reward model, value network/critic) and an online RL loop where the policy generates samples that are scored by the reward model and optimized via PPO [source:arxiv:2305.18290]. This pipeline incurs 3–5× the compute of supervised fine-tuning (SFT), demands careful hyperparameter tuning for KL penalties and clipping, and suffers from reward hacking where the policy exploits imperfections in the learned reward model [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. **PPO specifically requires storing four separate model copies in memory: the policy, the reference policy, the reward model, and the value function, leading to high GPU memory requirements** [source:cameronrwolfe:direct-preference-optimization-dpo-deep-]. DPO's central insight is that under the Bradley–Terry preference model and a KL-constrained reward maximization objective, the optimal policy has a closed-form relationship to the reference policy:

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

**The DPO Workflow:**
1. **Initialization:** Begin with a model that has already undergone pretraining and Supervised Fine-Tuning (SFT). This SFT model serves as the reference policy ($\pi_{ref}$) [source:cameronrwolfe:direct-preference-optimization-dpo-deep-].
2. **Preference Data Collection:** Utilize a dataset consisting of triplets: a prompt ($x$), a "chosen" response ($y_w$), and a "rejected" response ($y_l$) [source:cameronrwolfe:direct-preference-optimization-dpo-deep-].
3. **Implicit Reward Optimization:** Instead of training a separate RM, DPO optimizes the policy to maximize the probability of the chosen response relative to the rejected response [source:cameronrwolfe:direct-preference-optimization-dpo-deep-].
4. **Gradient Descent:** The model is updated using basic gradient descent to solve the RLHF objective indirectly, avoiding online sampling and the need for a value function [source:cameronrwolfe:direct-preference-optimization-dpo-deep-].

**Empirical results (6B scale):** On TL;DR summarization, DPO achieved ~61% win rate vs. PPO's 57% (both at temperature 0) [source:arxiv:2305.18290]. On Anthropic Helpful/Harmless dialogue, DPO was the only efficient method improving over the preferred completions in the dataset, matching Best-of-128 [source:arxiv:2305.18290]. Out-of-distribution generalization to CNN/DailyMail was stronger for DPO (win rate 0.36) than PPO (0.26) [source:arxiv:2305.18290].

**Limitations noted in the original paper:** DPO is an offline algorithm — it cannot explore or correct error types absent from the dataset [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. Its ceiling is bounded by dataset quality and diversity [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. The sigmoid loss saturates, causing vanishing gradients once a pair is correctly ordered, which can lead to overfitting and unbounded growth of the implicit reward gap [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. DPO also strictly requires paired comparisons, incompatible with unary feedback (thumbs-up/down) [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. Scaling beyond 6B parameters was untested in the original work [source:arxiv:2305.18290]. **The author notes a common misconception: DPO does not "avoid" reward modeling entirely; rather, it performs reward modeling *implicitly* within the policy** [source:cameronrwolfe:direct-preference-optimization-dpo-deep-]. **PPO-based RLHF "tends to yield the best results in large-scale LLM post-training runs" and remains a common choice in top LLM labs** [source:cameronrwolfe:direct-preference-optimization-dpo-deep-].

## IPO: bounding the reward gap

Identity Preference Optimization (IPO) addresses DPO's unbounded reward growth by replacing the monotonic sigmoid loss with a squared-error regression targeting a fixed margin [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. **The authors propose a general $\Psi$-Preference Optimization ($\Psi$PO) framework that expresses the objective in terms of pairwise preferences rather than pointwise rewards, where RLHF and DPO are special cases with $\Psi(q) = \log(q/(1-q))$** [source:arxiv:2310.12036]. **IPO is the specific instance where $\Psi$ is the identity mapping, ensuring KL-regularization remains effective even with deterministic preferences** [source:arxiv:2310.12036].

Define the log-ratio difference:

$$
h_\theta(x, y_w, y_l) = \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
$$

IPO's loss is:

$$
\mathcal{L}_{\mathrm{IPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( h_\theta - \frac{1}{2\beta} \right)^2 \right]
$$

The gradient $\partial \mathcal{L}_{\mathrm{IPO}} / \partial h_\theta = 2(h_\theta - 1/(2\beta))$ is zero at the target margin $1/(2\beta)$, creating a stable equilibrium: if the margin is too small the gradient pushes it up; if too large it pushes down [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. This prevents the extreme probabilities (chosen → 1, rejected → 0) that DPO's monotonic loss encourages. The target margin $1/(2\beta)$ derives from the KL regularization strength: larger $\beta$ (weaker KL penalty) → smaller target margin; smaller $\beta$ (stronger KL penalty) → larger permitted per-example margin [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. IPO retains DPO's paired-data requirement and reference-model memory cost.

**Implementation Recipe from the original paper:** The authors derive a computationally efficient offline sampled loss function:
1. **Define the log-likelihood ratio gap** $h_\pi$ for a context $x$ and actions $y, y'$:

$$
h _ {\pi} (y, y ^ {\prime}, x) = \log \left(\frac {\pi (y | x) \pi_ {\mathrm{ref}} (y ^ {\prime} | x)}{\pi (y ^ {\prime} | x) \pi_ {\mathrm{ref}} (y | x)}\right)
$$

2. **Initialize** the policy $\pi$ as the reference policy $\pi_{\text{ref}}$.
3. **Minimize the sampled loss** over the preference dataset $\mathcal{D}$:

$$
\underset {(y _ {w}, y _ {l}, x) \sim D} {\mathbb {E}} \left(h _ {\pi} (y _ {w}, y _ {l}, x) - \frac {\tau^ {- 1}}{2}\right) ^ {2}
$$

   This effectively regresses the gap between the policy's log-likelihood ratios and the reference policy's ratios to a constant $\frac{\tau^{-1}}{2}$ [source:arxiv:2310.12036].

**Key Quantitative Results (Bandit Setting):** Tested in a bandit setting with three actions $\{y_a, y_b, y_c\}$ using Adam optimizer (lr=0.01, mini-batch size 9, 18,000 steps) [source:arxiv:2310.12036]:
- **Avoidance of Greedy Policies:** In a total ordering dataset $\mathcal{D}_1 = \{(y_a, y_b), (y_b, y_c), (y_a, y_c)\}$, DPO converged to a deterministic policy ($\pi(y_a)=1$) for all values of $\tau$, ignoring $\pi_{\text{ref}}$. IPO remained close to $\pi_{\text{ref}}$ when $\tau$ was large and only became greedy as $\tau \to 0$ [source:arxiv:2310.12036].
- **Action Preservation:** In scenarios where an action never wins in the dataset, DPO pushed that action's probability to 0 regardless of $\tau$. IPO maintained the action's probability relative to the strength of $\tau$ [source:arxiv:2310.12036].
- **Unobserved Pairs:** With a dataset $\mathcal{D}_3 = \{(y_a, y_b), (y_b, y_a)\}$ where the pair $(y_a, y_c)$ was unobserved, DPO ignored $\pi_{\text{ref}}$ entirely, while IPO's solution scaled gradually with $\tau$ [source:arxiv:2310.12036].

**Stated Limitations:** The authors note that empirical evaluations were conducted on "simple bandit examples" and "minimal experiments." Future work is required to scale these findings to training large language models on human preference data [source:arxiv:2310.12036].

## KTO: learning from unary feedback

Kahneman–Tversky Optimization (KTO) relaxes the paired-comparison requirement, using only binary "desirable/undesirable" labels on individual responses [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. **KTO frames alignment through the lens of prospect theory, proposing that the success of existing alignment methods is due to their nature as Human-Aware Losses (HALOs)—loss functions that implicitly incorporate human cognitive biases such as loss aversion and diminishing sensitivity** [source:arxiv:2402.01306]. Its loss (simplified) is:

$$
\mathcal{L}_{\mathrm{KTO}}(\theta) = \mathbb{E}_{y_w}\!\left[\sigma\!\left(-r_\theta(x, y_w)\right)\right] + \mathbb{E}_{y_l}\!\left[\sigma\!\left(r_\theta(x, y_l)\right)\right]
$$

where the per-example implicit reward is centered by a batch-wise baseline:

$$
r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} - \mathbb{E}_{y'}\!\left[\beta \log \frac{\pi_\theta(y' \mid x)}{\pi_{\mathrm{ref}}(y' \mid x)}\right]
$$

This centering approximates the normalization constant $Z(x)$ and enables learning from unpaired data [source:mesuvash:reward-modeling-and-dpo-learning-what-go].

**Detailed Formulas from the original paper:**
- **Implied Reward:** $r_\theta(x, y) = \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ [source:arxiv:2402.01306]
- **Reference Point Estimation:** $\hat{z}_0 = \max \left(0, \frac{1}{m} \sum_{1 \le i < m} \log \frac{\pi_\theta(y_j|x_i)}{\pi_{\text{ref}}(y_j|x_i)}\right)$ where $j = (i + 1) \mod m$ [source:arxiv:2402.01306]
- **Value Function:** 

$$
v(x, y) = \begin{cases} \lambda_D \sigma(\beta(r_\theta(x, y) - z_0)) & \text{if } y \sim y_{\text{desirable}} | x \\ \lambda_U \sigma(\beta(z_0 - r_\theta(x, y))) & \text{if } y \sim y_{\text{undesirable}} | x \end{cases}
$$

[source:arxiv:2402.01306]
- **Final KTO Loss:** $L_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x, y \sim D} [\lambda_y - v(x, y)]$ where $\beta$ controls risk aversion and $\lambda_y$ controls loss aversion [source:arxiv:2402.01306].

**Key Quantitative Results:**
- **Performance vs. DPO:** KTO matches or exceeds DPO performance across model scales from 1B to 30B parameters. In human evaluations on the OpenAssistant test set, KTO achieved a winrate of $72.9\% \pm 5.3$ compared to DPO's $62.1\% \pm 5.7$ [source:arxiv:2402.01306].
- **Task-Specific Gains:** On GSM8K mathematical reasoning benchmark, swapping DPO for KTO when aligning Zephyr-$\beta$-SFT improved performance by 13.5 points (53.5 vs 40.0) [source:arxiv:2402.01306].
- **Data Efficiency:** KTO can match DPO performance while using up to 90% fewer desirable examples. In a "one-y-per-x" setup (reducing data by 72%), KTO still outperformed DPO and the official Mistral-7B-Instruct [source:arxiv:2402.01306].
- **SFT Independence:** For sufficiently large models (Llama-13B, 30B), KTO can be applied directly to the pretrained model without Supervised Finetuning (SFT) without losing quality, whereas DPO without SFT leads to rambling and hallucinations [source:arxiv:2402.01306].

**Stated Limitations:**
- **Underfitting Risk:** KTO may underfit if the preference data is exceptionally clean (low noise and low intransitivity), as its gradient tends to zero when rewards become extreme [source:arxiv:2402.01306].
- **Hyperparameter Sensitivity:** KTO is highly sensitive to the learning rate, typically requiring a rate 2x to 10x higher than that used for DPO [source:arxiv:2402.01306].
- **Theoretical Gap:** The value function is based on monetary gambles, which may not perfectly represent how humans perceive the quality of generated text [source:arxiv:2402.01306].
- **Reference Model Dependency:** While a reference-free variant exists, it is less performant than the standard KTO [source:arxiv:2402.01306].

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
- **IPO's target margin derivation:** The IPO article derives $1/(2\beta)$ from the Bradley–Terry model and KL regularization [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a], but does not cite an original IPO paper (e.g., Azar et al. 2023) to confirm this matches the published method. The derivation is presented as self-contained; a primary-source check would settle it. **The original IPO paper derives the target margin as $\frac{\tau^{-1}}{2}$ within the $\Psi$PO framework where $\tau$ is the KL regularization coefficient** [source:arxiv:2310.12036], confirming the relationship but using different notation ($\tau$ vs $\beta$).
- **ORPO absence:** No source describes ORPO's mechanism. The variants article mentions ORPO as one of four DPO variants but does not specify which limitation it addresses [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a], suggesting ORPO may eliminate the reference model, but this is unverified in the provided literature.
- **Reward over-optimization in DPO:** The authors note a slight decrease in performance late in training, which may be an instance of reward over-optimization [source:arxiv:2305.18290], while the variants article asserts DPO's unbounded reward growth *is* a form of over-optimization [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a]. These are not direct contradictions but reflect different framings of the same phenomenon.
- **KTO's prospect theory framing vs. DPO's BT model:** KTO explicitly rejects the Bradley-Terry model assumption, arguing that human preferences follow prospect theory (loss aversion, diminishing sensitivity) rather than consistent utility maximization [source:arxiv:2402.01306]. This represents a fundamental theoretical disagreement about the nature of human feedback.
- **IPO's bounded loss vs. KTO's vanishing gradient:** IPO's squared-error loss creates a stable equilibrium at the target margin [source:arxiv:2310.12036], while KTO's loss gradient tends to zero when rewards become extreme, risking underfitting on clean data [source:arxiv:2402.01306]. These are opposing behaviors at the extremes of the reward distribution.

## Current status and trajectory

DPO and its variants are the **default** offline preference alignment methods as of 2024, widely adopted in open-source LLM alignment (Llama 3, Mistral, Gemma families) due to their 1.5–2× SFT cost, stability, and lack of reward-model training [source:mesuvash:reward-modeling-and-dpo-learning-what-go]. IPO is commonly used as a drop-in replacement for DPO to prevent overfitting; KTO is adopted when only unary feedback is available. AlphaDPO's adaptive reference is a newer proposal (ICML 2024) with strong benchmark numbers but not yet widely reported in production deployments. **ORPO's status cannot be assessed from the provided sources — it is cited as a variant but not described.** The field is moving toward *hybrid* pipelines: offline DPO/IPO for initial alignment, followed by online RL (PPO/GRPO) for reasoning capabilities where verifiable rewards exist. Pure offline methods are not fading but are recognized as insufficient for advanced reasoning where exploration and process supervision matter. **The original IPO paper notes that scaling to LLM training on human preference data remains future work** [source:arxiv:2310.12036], while **KTO demonstrates strong scaling to 30B parameters and SFT-free alignment for large models** [source:arxiv:2402.01306].

## Key takeaways

- DPO eliminates the reward model and RL loop by reparameterizing the optimal KL-constrained policy as an implicit reward $\hat{r}_\theta = \beta \log \frac{\pi_\theta}{\pi_{\mathrm{ref}}}$, yielding a binary classification loss on preference pairs.
- IPO replaces DPO's saturating sigmoid loss with a squared-error loss targeting margin $1/(2\beta)$, creating a stable equilibrium that prevents unbounded reward growth and overconfident predictions. **It derives from the $\Psi$PO framework where the identity mapping $\Psi$ bounds the preference probability, ensuring KL regularization remains effective even with deterministic preferences** [source:arxiv:2310.12036].
- KTO enables learning from unary (desirable/undesirable) labels by centering the implicit reward with a batch-wise baseline, removing the paired-comparison requirement. **It is grounded in prospect theory (loss aversion, diminishing sensitivity) rather than the Bradley-Terry model, and matches or exceeds DPO across 1B–30B scales with up to 90% less desirable data** [source:arxiv:2402.01306].
- AlphaDPO introduces an adaptive implicit reference $\hat{\pi}_{\mathrm{ref}} \propto U (\pi_\theta/\pi_{\mathrm{ref}})^\alpha$ that yields instance-adaptive margins and controls sequential KL divergence.
- All characterized methods (DPO, IPO, KTO, AlphaDPO) are offline, bounded by dataset coverage, and require the reference model in memory. **ORPO cannot be characterized from the provided sources — its mechanism, loss, and memory requirements are undocumented.**
- DPO at 6B scale matched or exceeded PPO on summarization and dialogue, even when PPO used ground-truth rewards; scaling to frontier models is the primary unvalidated claim.
- **IPO's bandit experiments demonstrate it avoids greedy policies, preserves actions with zero wins, and gracefully handles unobserved pairs — all failures of DPO** [source:arxiv:2310.12036].
- **KTO achieves SFT-free alignment for large models (13B+) and 13.5-point GSM8K gains over DPO** [source:arxiv:2402.01306].

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
- [source:youtube:orpo-explained-superior-llm-alignment-te] [ORPO Explained: Superior LLM Alignment Technique vs. DPO/RLHF](https://www.youtube.com/watch?v=32S_xBt2IXw)
- [source:mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a] [DPO Variants: IPO, KTO, ORPO & cDPO for LLM Alignment](https://mbrenndoerfer.com/writing/dpo-variants-ipo-kto-orpo-cdpo-llm-alignment)
- [source:arxiv:2402.00164] [Beyond DPO: A Survey of Preference-Based Alignment Algorithms for LLMs](https://arxiv.org/abs/2402.00164)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:icml:alphadpo-adaptive-reward-margin-for-dire] [AlphaDPO: Adaptive Reward Margin for Direct Preference Optimization](https://icml.cc/virtual/2025/poster/45946)
- [source:mesuvash:reward-modeling-and-dpo-learning-what-go] [Reward Modeling and DPO: Learning What "Good" Means](https://mesuvash.github.io/blog/2026/reward-modeling/)
- [source:arxiv:2310.12036] [Identity Preference Optimization (IPO)](https://arxiv.org/abs/2310.12036)
- [source:arxiv:2402.01306] [Kahneman-Tversky Optimization (KTO)](https://arxiv.org/abs/2402.01306)
- [source:arxiv:2403.07693] [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07693)
- [source:arxiv:2401.06108] [From RLHF to Direct Alignment: A Theoretical Unification of Preference Optimization Algorithms](https://arxiv.org/abs/2401.06108)
- [source:cameronrwolfe:direct-preference-optimization-dpo-deep-] [Direct Preference Optimization (DPO) - Deep (Learning) Focus](https://cameronrwolfe.substack.com/p/direct-preference-optimization)
