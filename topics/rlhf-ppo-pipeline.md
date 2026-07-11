---
title: The RLHF/PPO pipeline
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2203.02155
- arxiv:2403.17031
- bair:rethinking-the-role-of-ppo-in-rlhf
- huggingface:navigating-the-rlhf-landscape-from-polic
- medium:inside-the-rlhf-engine-a-deep-dive-into-
- cameronrwolfe:ppo-for-llms-a-guide-for-normal-people
open_questions:
- What is the *minimal* set of PPO hyperparameters that reliably prevents over-optimization
  across model scales (1B–70B) without PPO-ptx?
- Does reward translation invariance (P3O's fix) materially improve *downstream task
  performance* (code, reasoning, tool use) or only proxy reward/KL frontiers on summarization/HH?
- How much of the reported "alignment tax" on academic benchmarks is due to **KL penalty
  strength** vs **pretraining mix coefficient** vs **RM miscalibration**? No source
  disentangles these.
- Can offline preference methods (DPO/IPO/KTO) match PPO's performance on *long-horizon,
  multi-turn, tool-use* tasks where the bandit assumption breaks, or is on-policy
  exploration still necessary?
---

The RLHF/PPO pipeline — Supervised Fine-Tuning (SFT) → Reward Model (RM) → Proximal Policy Optimization (PPO) — is the canonical three-stage recipe introduced by InstructGPT to align language models with human intent [source:arxiv:2203.02155]. This pipeline treats alignment as a reinforcement learning problem where a learned reward model provides the signal, and PPO optimizes the policy under a KL constraint to prevent reward hacking and catastrophic forgetting.

## Supervised Fine-Tuning (SFT)

**Objective and data.** SFT adapts a pre-trained base model to the target distribution of instruction-following behavior by maximizing the likelihood of human-written demonstrations $(x, y)$ [source:arxiv:2203.02155]. The InstructGPT dataset mixes prompts from the OpenAI API with labeler-written prompts covering diverse tasks (generation, brainstorming, QA, etc.) [source:arxiv:2203.02155]. In the TL;DR reproduction, the SFT dataset consists of Reddit posts paired with human summaries, formatted as `"SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"` with a leading space prepended to the completion and an EOS token appended [source:arxiv:2403.17031].

**Training details.** InstructGPT trains for 16 epochs with cosine learning-rate decay and residual dropout 0.2, selecting the final checkpoint by RM score on a validation set [source:arxiv:2203.02155]. The N+ reproduction uses 1 epoch (116,722 episodes), AdamW ($\epsilon=10^{-5}$, lr=$3\times10^{-6}$), cosine scheduler, batch size 128, and **disables dropout** across SFT, RM, and PPO to ensure reproducible log-probabilities for the KL penalty [source:arxiv:2403.17031]. Sequences are right-padded to a fixed shape (e.g., (B, 562) for SFT) [source:arxiv:2403.17031]. The medium article notes the SFT loss is standard cross-entropy: $L_{\text{SFT}} = -\sum_i \log P(y_i \mid x, y_{<i}; \theta_{\text{SFT}})$ [source:medium:inside-the-rlhf-engine-a-deep-dive-into-].

**Disagreement on SFT scope.** InstructGPT uses SFT as the *sole* initialization for both RM and PPO [source:arxiv:2203.02155]. The N+ study initializes the RM from the SFT model with a randomly initialized linear head [source:arxiv:2403.17031], while the HuggingFace overview notes the reference model for KL regularization is typically the SFT model [source:huggingface:navigating-the-rlhf-landscape-from-polic]. No source reports using a *different* model for SFT vs. reference, but the N+ ablation of dropout disabling is a practical detail absent from the original paper.

## Reward Model (RM) Training

**Preference data and loss.** Human labelers rank $K$ model outputs (InstructGPT: $K\in[4,9]$; TL;DR: pairwise comparisons) producing $\binom{K}{2}$ comparisons per prompt [source:arxiv:2203.02155]. The RM $r_\phi(x,y)$ outputs a scalar and is trained with a pairwise ranking loss:

$$
\mathcal{L}_R = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma\bigl(r_\phi(x,y_w) - r_\phi(x,y_l)\bigr) \right]
$$

[source:arxiv:2203.02155][source:arxiv:2403.17031][source:medium:inside-the-rlhf-engine-a-deep-dive-into-]. The N+ implementation writes the equivalent logistic loss $\mathbb{E}[\log(1+e^{r_\phi(x,y_r)-r_\phi(x,y_c)})]$ [source:arxiv:2403.17031].

**Normalization.** Both sources normalize the RM by adding a bias so that a reference set achieves mean score 0. InstructGPT uses *labeler demonstrations* [source:arxiv:2203.02155]; N+ uses *SFT reference summaries* [source:arxiv:2403.17031]. This shift invariance is precisely the issue P3O targets: $r(y|x) + \delta(x)$ is equally valid for the pairwise loss but changes the absolute magnitudes PPO optimizes [source:bair:rethinking-the-role-of-ppo-in-rlhf].

**Architecture and training.** The RM is initialized from the SFT model (N+: with a random linear head) [source:arxiv:2403.17031]. InstructGPT does not specify the head initialization. N+ trains for 1 epoch (92,858 episodes), AdamW ($\epsilon=10^{-5}$, lr=$3\times10^{-6}$), cosine scheduler, batch size 64, with sequences right-padded to (B, 638) [source:arxiv:2403.17031]. **Reward extraction:** N+ takes the scalar from the **EOS token** position; non-EOS tokens typically carry negative logits [source:arxiv:2403.17031]. Left- vs right-padding introduces minor numerical differences (e.g., $-5.4\times10^{-4}$ average reward shift for 6.9B) [source:arxiv:2403.17031].

**Calibration and scaling.** RM validation accuracy scales with model size: 1B (62.8%), 2.8B (66.9%), 6.9B (68.9%) on the TL;DR pref set; similar on CNN/DM [source:arxiv:2403.17031]. Agreement with GPT-3.5 judgments scales similarly (1B: 40.3%, 2.8B: 72.6%, 6.9B: 76.7%) [source:arxiv:2403.17031]. InstructGPT reports 69.6% accuracy on held-out labelers vs 72.4% on training labelers [source:arxiv:2203.02155]. N+ finds RMs are **under-calibrated**: accuracy correlates with score difference but predicted probabilities are overconfident [source:arxiv:2403.17031]. **Disagreement:** N+ reports DPO's *implicit* reward modeling regresses in validation accuracy vs explicit RM, attributing it to loss applied on all tokens vs EOS only, the $\beta$ parameter, and objective differences [source:arxiv:2403.17031]. P3O does not compare explicit vs implicit RM accuracy.

## PPO Fine-Tuning

**RL formulation.** The environment is a contextual bandit: a prompt $x$ is sampled, the policy $\pi_\phi^{\text{RL}}$ generates $y$, and the RM returns $r_\theta(x,y)$ [source:arxiv:2203.02155]. The per-token KL penalty from the SFT policy $\pi^{\text{SFT}}$ is added to the reward:

$$
R(x,y) = r_\theta(x,y) - \beta \, D_{\text{KL}}\bigl(\pi_\phi^{\text{RL}}(y|x) \,\|\, \pi^{\text{SFT}}(y|x)\bigr)
$$

[source:arxiv:2203.02155][source:arxiv:2403.17031]. The objective is $\max_{\pi_\phi} \mathbb{E}_{x\sim\mathcal{D}_{\text{SFT}}, y\sim\pi_\phi}[R(x,y)]$ [source:arxiv:2403.17031].

**PPO-ptx (pretraining mix).** InstructGPT adds a pretraining gradient term to mitigate the alignment tax:

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi_\phi^{\text{RL}}}} \Bigl[ r_\theta(x,y) - \beta \log \frac{\pi_\phi^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)} \Bigr] + \gamma \, \mathbb{E}_{x\sim D_{\text{pretrain}}} \bigl[ \log \pi_\phi^{\text{RL}}(x) \bigr]
$$

where $\gamma$ is the pretraining loss coefficient (0 for plain PPO) [source:arxiv:2203.02155]. N+ does not mention PPO-ptx.

**Value function.** Both sources initialize the value network from the RM. InstructGPT states "the value function is initialized from the RM" [source:arxiv:2203.02155]. N+ clarifies it acts as a **per-token RM** [source:arxiv:2403.17031].

**N+ hyperparameters (most complete public record).**
| Parameter | Value |
|---|---|
| Episodes | 1,000,000 (~8.56 epochs over SFT data) |
| Optimizer | AdamW ($\epsilon=10^{-5}$, lr=$3\times10^{-6}$) |
| Scheduler | Linear |
| Batch size | 512 |
| $\beta$ (KL coeff) | 0.05 |
| $\gamma$ (discount) | 1.0 |
| $\lambda$ (GAE) | 0.95 |
| Mini-batches | 1 |
| PPO epochs $K$ | 4 |
| Clip $\epsilon$ | 0.2 |
| Value clip $\hat{\varepsilon}$ | 0.2 |
| Value coeff $c_1$ | 0.1 |
| Value loss clipping | True |
| Sampling temp | 0.7 |

[source:arxiv:2403.17031]

**Practical tricks (N+).**
- **EOS trick:** If a generation lacks EOS, its reward is set to $-1$ [source:arxiv:2403.17031].
- **Reward whitening:** $\text{whitened} = (v - \text{mean}) \cdot \text{rsqrt}(\text{var} + 10^{-8})$; leads to shorter outputs and lower preference rates, but length-controlled comparisons show similar performance [source:arxiv:2403.17031].
- **Advantage whitening:** `whiten(advantages)` with shifted mean [source:arxiv:2403.17031].
- **Queries left-padded** to 512 tokens for PPO [source:arxiv:2403.17031].

**Policy gradient perspective.** The HuggingFace overview derives the REINFORCE gradient $\nabla_\theta J = \mathbb{E}_{\tau\sim\pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)]$, then introduces rewards-to-go, baselines, and the advantage $A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$ with GAE estimate $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ [source:huggingface:navigating-the-rlhf-landscape-from-polic]. The Cameron Wolfe guide gives the same MDP formulation (state = prompt + generated tokens; action = next token) and the basic policy gradient loss $L(\theta) = -\sum_\tau R(\tau) \sum_t \log \pi_\theta(a_t|s_t)$ [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people].

## Known Issues and Limitations

**Reward hacking / over-optimization.** InstructGPT observes that plain PPO regresses on public NLP benchmarks (alignment tax) and mitigates it with PPO-ptx, but does not eliminate it [source:arxiv:2203.02155]. N+ finds **1B models over-optimize**: KL divergence reaches 50–85 nats and win rate drops below 20% [source:arxiv:2403.17031]. P3O argues the root cause is the **mismatch between comparative RM training and absolute RL optimization**: a reward shift $\delta(x)$ leaves preferences unchanged but alters the PPO objective, causing instability [source:bair:rethinking-the-role-of-ppo-in-rlhf].

**KL–reward trade-off.** P3O shows the KL–reward frontier: DPO > P3O > PPO > SFT in both KL and proxy reward, but **GPT-4 win rates reverse the order for DPO vs P3O** (P3O beats DPO 54.6% despite lower reward) because DPO's higher KL produces lower-quality generations [source:bair:rethinking-the-role-of-ppo-in-rlhf]. InstructGPT does not report KL values; N+ does not compare DPO.

**Labeler and evaluation bias.** InstructGPT acknowledges labelers (~40, English-speaking, US/SE Asia) are not representative; most comparisons have a single labeler [source:arxiv:2203.02155]. N+ uses GPT-3.5 as a judge, noting its own biases [source:arxiv:2403.17031]. P3O uses GPT-4 as judge [source:bair:rethinking-the-role-of-ppo-in-rlhf].

**Computational cost.** InstructGPT: 4.9 PFLOPs-days (SFT) + 60 PFLOPs-days (PPO-ptx) for 175B vs 3,640 for GPT-3 pre-training [source:arxiv:2203.02155]. HuggingFace notes on-policy PPO requires **four large models simultaneously** (Actor, Critic, RM, Reference) and generation is the bottleneck [source:huggingface:navigating-the-rlhf-landscape-from-polic]. Cameron Wolfe emphasizes high variance of basic policy gradients necessitates many trajectories [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people].

**RM calibration.** N+ finds RMs under-calibrated; accuracy varies wildly across data batches (1B: 50.8%–77.1%) and confidence levels [source:arxiv:2403.17031]. InstructGPT does not report calibration.

## Current Status and Trajectory

The **SFT→RM→PPO pipeline remains the foundational reference** for industrial RLHF (InstructGPT, ChatGPT, GPT-4, Claude, Llama 2/3) but **pure PPO is fading in open research** in favor of **offline preference optimization** (DPO, IPO, KTO, SimPO) and **pairwise on-policy methods** (P3O, GRPO) that avoid the RM–RL mismatch and reduce infrastructure burden [source:bair:rethinking-the-role-of-ppo-in-rlhf][source:arxiv:2403.17031]. The N+ reproduction (2024) is valuable precisely because **faithful open-source PPO implementations were scarce** [source:arxiv:2403.17031]. Major labs still use PPO at scale (Llama 3 report cites PPO), but **published details are not widely reported**; the trajectory is toward **simpler, offline, or reward-translation-invariant algorithms** [source:bair:rethinking-the-role-of-ppo-in-rlhf]. **Hedge:** No source claims PPO is abandoned; it remains the production workhorse where compute and engineering capacity exist, but the *research frontier* has moved to alternatives that mitigate its instability and cost.

## Key Takeaways

- The InstructGPT recipe (SFT → pairwise RM → PPO with KL penalty + optional pretraining mix) is the canonical RLHF pipeline [source:arxiv:2203.02155].
- **Critical implementation details** (dropout disabled, EOS-token reward extraction, reward normalization target, left/right padding, value init from RM, EOS trick, whitening) materially affect results and are **not all in the original paper** [source:arxiv:2403.17031].
- **Reward translation invariance** breaks the theoretical link between RM training (comparative) and PPO (absolute); P3O fixes this with a pairwise policy gradient but is not yet widely adopted [source:bair:rethinking-the-role-of-ppo-in-rlhf].
- **Over-optimization** appears at small model scales (1B) as KL explosion and quality collapse; larger models (2.8B+) are more robust [source:arxiv:2403.17031].
- **DPO's implicit RM** regresses in validation accuracy vs explicit RM (N+), and DPO achieves higher proxy reward but higher KL and lower GPT-4 win rate than P3O (P3O) — **optimizing the proxy reward too hard hurts quality** [source:arxiv:2403.17031][source:bair:rethinking-the-role-of-ppo-in-rlhf].
- **Compute cost**: PPO requires 4 large models + rollout generation; offline methods (DPO) need only 2 (policy + ref) and no rollout [source:huggingface:navigating-the-rlhf-landscape-from-polic].
- **Evaluation remains fragile**: human labelers are narrow; LLM judges (GPT-3.5/4) have biases; RM calibration is poor [source:arxiv:2203.02155][source:arxiv:2403.17031][source:bair:rethinking-the-role-of-ppo-in-rlhf].

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)

## References
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2403.17031] [The N+ Implementation Details of RLHF with PPO: A Case ...](https://arxiv.org/abs/2403.17031)
- [source:bair:rethinking-the-role-of-ppo-in-rlhf] [Rethinking the Role of PPO in RLHF](http://bair.berkeley.edu/blog/2023/10/16/p3o/)
- [source:huggingface:navigating-the-rlhf-landscape-from-polic] [Navigating the RLHF Landscape: From Policy Gradients to PPO, GAE, and DPO for LLM Alignment](https://huggingface.co/blog/NormalUhr/rlhf-pipeline)
- [source:medium:inside-the-rlhf-engine-a-deep-dive-into-] [Inside the RLHF Engine: A Deep Dive into SFT, Reward ...](https://medium.com/foundation-models-deep-dive/inside-the-rlhf-engine-a-deep-dive-into-sft-reward-models-and-rl-fine-tuning-60978b291d55)
- [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people] [PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm)
