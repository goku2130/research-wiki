---
title: The RLHF/PPO pipeline
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2403.17031
- arxiv:2203.02155
- medium:inside-the-rlhf-engine-a-deep-dive-into-
- bair:rethinking-the-role-of-ppo-in-rlhf
- cameronrwolfe:ppo-for-llms-a-guide-for-normal-people
- huggingface:navigating-the-rlhf-landscape-from-polic
- arxiv:1706.03741
- arxiv:2305.18290
- cameronrwolfe:ppo-for-llms-a-guide-for-normal-people-c
- arxiv:2404.10719
- interconnects:rlhf-roundup-trying-to-get-good-at-ppo-i
- arxiv:1707.06347
- huggingface:stackllama-a-hands-on-guide-to-rlhf-with
open_questions:
- Can the PPO "recipe" (advantage normalization, large batch, ref-model EMA) be reliably
  reproduced across diverse base models and datasets, or does it require per-model
  tuning that limits open adoption?
- Does DPO's OOD generalization limitation (assigning high probability to responses
  absent from preference data) manifest catastrophically at scale, or is it mitigated
  by larger, more diverse preference datasets?
- How much of PPO's empirical superiority on code tasks (CodeContest, APPS) is due
  to ground-truth verifiable rewards vs. the algorithm itself? Would DPO with the
  same verifiable rewards close the gap?
- Is the reward-translation-invariance fix (P3O, GRPO) necessary for stable PPO at
  scale, or do engineering tricks (whitening, EOS trick, large batch) suffice in practice?
---

The RLHF/PPO pipeline — Supervised Fine-Tuning (SFT) → Reward Model (RM) → Proximal Policy Optimization (PPO) — is the canonical three-stage recipe introduced by InstructGPT to align language models with human intent [source:arxiv:2203.02155]. This pipeline treats alignment as a reinforcement learning problem where a learned reward model provides the signal, and PPO optimizes the policy under a KL constraint to prevent reward hacking and catastrophic forgetting.

## Historical Foundations: Preference Learning Before LLMs

**Deep RL from Human Preferences (Christiano et al. 2017).** The core idea of learning a reward function from human pairwise preferences and optimizing a policy against it predates LLM alignment. Christiano et al. [source:arxiv:1706.03741] demonstrated this loop on Atari games and MuJoCo robotics tasks using **A2C** and **TRPO** as the policy optimizers (not PPO). Key design choices that carry forward:
- **Trajectory segments** (1–2 seconds) rather than single frames for human comparison, because humans need temporal context.
- **Ensemble of reward predictors** with variance-based active query selection to maximize information per human label.
- **Bradley–Terry modeling** of preferences with a 10% uniform-noise term to account for human inconsistency.
- **Online data collection** is critical: offline (fixed-dataset) reward learning fails because the policy exploits the reward model once its occupancy distribution shifts — producing "bizarre behavior" such as avoiding losing in Pong without trying to score [source:arxiv:1706.03741].
- **Sample efficiency**: <1% of agent interactions required human feedback; ~700–1,400 queries sufficed for near-oracle performance on simulated robotics, ~5,500 for Atari.

This work established the **preference → reward model → RL** paradigm that InstructGPT later adapted to language models, replacing TRPO/A2C with PPO and human trajectory comparisons with text completion rankings.

## Supervised Fine-Tuning (SFT)

**Objective and data.** SFT adapts a pre-trained base model to the target distribution of instruction-following behavior by maximizing the likelihood of human-written demonstrations $(x, y)$ [source:arxiv:2203.02155]. The InstructGPT dataset mixes prompts from the OpenAI API with labeler-written prompts covering diverse tasks (generation, brainstorming, QA, etc.) [source:arxiv:2203.02155]. In the TL;DR reproduction, the SFT dataset consists of Reddit posts paired with human summaries, formatted as `"SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:"` with a leading space prepended to the completion and an EOS token appended [source:arxiv:2403.17031].

**Training details.** InstructGPT trains for 16 epochs with cosine learning-rate decay and residual dropout 0.2, selecting the final checkpoint by RM score on a validation set [source:arxiv:2203.02155]. The N+ reproduction uses 1 epoch (116,722 episodes), AdamW ($\epsilon=10^{-5}$, lr=$3\times10^{-6}$), cosine scheduler, batch size 128, and **disables dropout** across SFT, RM, and PPO to ensure reproducible log-probabilities for the KL penalty [source:arxiv:2403.17031]. Sequences are right-padded to a fixed shape (e.g., (B, 562) for SFT) [source:arxiv:2403.17031]. The medium article notes the SFT loss is standard cross-entropy: $L_{\text{SFT}} = -\sum_i \log P(y_i \mid x, y_{<i}; \theta_{\text{SFT}})$ [source:medium:inside-the-rlhf-engine-a-deep-dive-into-].

**Parameter-efficient SFT (StackLLaMA).** For the 7B LLaMA model, StackLLaMA uses **LoRA** ($r=16$, $\alpha=32$, dropout=0.05) with **8-bit quantization** and **packing** (concatenating multiple examples with EOS tokens to fill the context window, eliminating padding) to reduce memory from ~70 GB (bf16 full fine-tune) to ~1.2–1.4 GB per billion parameters [source:huggingface:stackllama-a-hands-on-guide-to-rlhf-with].

**Disagreement on SFT scope.** InstructGPT uses SFT as the *sole* initialization for both RM and PPO [source:arxiv:2203.02155]. The N+ study initializes the RM from the SFT model with a randomly initialized linear head [source:arxiv:2403.17031], while the HuggingFace overview notes the reference model for KL regularization is typically the SFT model [source:huggingface:navigating-the-rlhf-landscape-from-polic]. No source reports using a *different* model for SFT vs. reference, but the N+ ablation of dropout disabling is a practical detail absent from the original paper.

## Reward Model (RM) Training

**Preference data and loss.** Human labelers rank $K$ model outputs (InstructGPT: $K\in[4,9]$; TL;DR: pairwise comparisons) producing $\binom{K}{2}$ comparisons per prompt [source:arxiv:2203.02155]. The RM $r_\phi(x,y)$ outputs a scalar and is trained with a pairwise ranking loss:

$$
\mathcal{L}_R = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma\bigl(r_\phi(x,y_w) - r_\phi(x,y_l)\bigr) \right]
$$

[source:arxiv:2203.02155][source:arxiv:2403.17031][source:medium:inside-the-rlhf-engine-a-deep-dive-into-]. The N+ implementation writes the equivalent logistic loss $\mathbb{E}[\log(1+e^{r_\phi(x,y_r)-r_\phi(x,y_c)})]$ [source:arxiv:2403.17031]. The Christiano et al. 2017 work uses the same Bradley–Terry formulation with an ensemble of predictors and cross-entropy loss against human labels [source:arxiv:1706.03741].

**Normalization.** Both sources normalize the RM by adding a bias so that a reference set achieves mean score 0. InstructGPT uses *labeler demonstrations* [source:arxiv:2203.02155]; N+ uses *SFT reference summaries* [source:arxiv:2403.17031]. This shift invariance is precisely the issue P3O targets: $r(y|x) + \delta(x)$ is equally valid for the pairwise loss but changes the absolute magnitudes PPO optimizes [source:bair:rethinking-the-role-of-ppo-in-rlhf].

**Architecture and training.** The RM is initialized from the SFT model (N+: with a random linear head) [source:arxiv:2403.17031]. InstructGPT does not specify the head initialization. N+ trains for 1 epoch (92,858 episodes), AdamW ($\epsilon=10^{-5}$, lr=$3\times10^{-6}$), cosine scheduler, batch size 64, with sequences right-padded to (B, 638) [source:arxiv:2403.17031]. **Reward extraction:** N+ takes the scalar from the **EOS token** position; non-EOS tokens typically carry negative logits [source:arxiv:2403.17031]. Left- vs right-padding introduces minor numerical differences (e.g., $-5.4\times10^{-4}$ average reward shift for 6.9B) [source:arxiv:2403.17031].

**Parameter-efficient RM (StackLLaMA).** StackLLaMA trains the RM with LoRA ($r=8$, $\alpha=32$, dropout=0.1) on 100k preference pairs, using the same ranking loss formulation [source:huggingface:stackllama-a-hands-on-guide-to-rlhf-with]. The reward model achieved 67% accuracy on held-out pairs.

**Calibration and scaling.** RM validation accuracy scales with model size: 1B (62.8%), 2.8B (66.9%), 6.9B (68.9%) on the TL;DR pref set; similar on CNN/DM [source:arxiv:2403.17031]. Agreement with GPT-3.5 judgments scales similarly (1B: 40.3%, 2.8B: 72.6%, 6.9B: 76.7%) [source:arxiv:2403.17031]. InstructGPT reports 69.6% accuracy on held-out labelers vs 72.4% on training labelers [source:arxiv:2203.02155]. N+ finds RMs are **under-calibrated**: accuracy correlates with score difference but predicted probabilities are overconfident [source:arxiv:2403.17031]. **Disagreement:** N+ reports DPO's *implicit* reward modeling regresses in validation accuracy vs explicit RM, attributing it to loss applied on all tokens vs EOS only, the $\beta$ parameter, and objective differences [source:arxiv:2403.17031]. P3O does not compare explicit vs implicit RM accuracy.

## PPO Fine-Tuning

**RL formulation.** The environment is a contextual bandit: a prompt $x$ is sampled, the policy $\pi_\phi^{\text{RL}}$ generates $y$, and the RM returns $r_\theta(x,y)$ [source:arxiv:2203.02155]. The per-token KL penalty from the SFT policy $\pi^{\text{SFT}}$ is added to the reward:

$$
R(x,y) = r_\theta(x,y) - \beta \, D_{\text{KL}}\bigl(\pi_\phi^{\text{RL}}(y|x) \,\|\, \pi^{\text{SFT}}(y|x)\bigr)
$$

[source:arxiv:2203.02155][source:arxiv:2403.17031]. The objective is $\max_{\pi_\phi} \mathbb{E}_{x\sim\mathcal{D}_{\text{SFT}}, y\sim\pi_\phi}[R(x,y)]$ [source:arxiv:2403.17031].

**PPO algorithm details (Schulman et al. 2017).** PPO was designed to achieve TRPO's stability with first-order optimization [source:arxiv:1707.06347]. Two variants:
- **Clipped surrogate objective** (primary):

$$
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \bigl(r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \bigr) \right]
$$

  where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ and $\epsilon \approx 0.2$. This creates a pessimistic lower bound, removing incentive to move $r_t$ outside $[1-\epsilon, 1+\epsilon]$ when it would improve the objective.
- **Adaptive KL penalty** (alternative):

$$
L^{KLPEN}(\theta) = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot|s_t), \pi_\theta(\cdot|s_t)] \right]
$$

  with $\beta$ adjusted dynamically to target a KL budget $d_{\text{targ}}$.
- **Combined actor-critic loss** (with shared parameters):

$$
L_t^{CLIP+VF+S}(\theta) = \hat{\mathbb{E}}_t \left[ L_t^{CLIP}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

  where $L_t^{VF}$ is squared-error value loss and $S$ is an entropy bonus.

**LLM-to-RL mapping (Cameron Wolfe guide).** The MDP formulation for LLM generation [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people]:
- **Policy $\pi_\theta$**: the LLM itself.
- **Initial state**: input prompt.
- **Action $a_t$**: each token predicted.
- **State $s_t$**: prompt + tokens generated so far.
- **Trajectory**: full completion.
- **Reward $r_t$**: scalar from RM or verifier (typically only at terminal step).
- **Transition**: deterministic (append token).
- **Advantage**: $A(s,a) = Q(s,a) - V(s)$; GAE used in practice.

**PPO-ptx (pretraining mix).** InstructGPT adds a pretraining gradient term to mitigate the alignment tax:

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi_\phi^{\text{RL}}}} \Bigl[ r_\theta(x,y) - \beta \log \frac{\pi_\phi^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)} \Bigr] + \gamma \, \mathbb{E}_{x\sim D_{\text{pretrain}}} \bigl[ \log \pi_\phi^{\text{RL}}(x) \bigr]
$$

where $\gamma$ is the pretraining loss coefficient (0 for plain PPO) [source:arxiv:2203.02155]. N+ does not mention PPO-ptx.

**Value function.** Both sources initialize the value network from the RM. InstructGPT states "the value function is initialized from the RM" [source:arxiv:2203.02155]. N+ clarifies it acts as a **per-token RM** [source:arxiv:2403.17031]. The Cameron Wolfe guide derives the REINFORCE gradient $\nabla_\theta J = \mathbb{E}_{\tau\sim\pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) R(\tau)]$, then introduces rewards-to-go, baselines, and the advantage $A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$ with GAE estimate $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ [source:huggingface:navigating-the-rlhf-landscape-from-polic].

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

**Parameter-efficient PPO (StackLLaMA).** StackLLaMA runs PPO with LoRA on the policy only (SFT model frozen as 8-bit base + reference), KL penalty $\beta$ to prevent reward hacking, trained for 20 hours on 3×8 A100-80GB [source:huggingface:stackllama-a-hands-on-guide-to-rlhf-with]. They observe **reward hacking**: the policy exploits RM imperfections (e.g., generating repetitive code backticks because the RM associates code blocks with high-quality Stack Exchange answers) and **negative KL estimates** due to sampling strategies (EOS suppression, padding) causing instability.

## DPO and the Offline Preference Optimization Alternative

**Direct Preference Optimization (DPO).** DPO eliminates the explicit RM and RL loop by using the analytical mapping between the optimal KL-constrained policy and the reward function [source:arxiv:2305.18290]. The optimal policy satisfies:

$$
\pi_{r}(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

Rearranging gives $r(x, y) = \beta \log \frac{\pi_{r}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)$. Substituting into the Bradley–Terry model cancels the partition function, yielding the **DPO loss**:

$$
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{w} \mid x)}{\pi_{\mathrm{ref}}(y_{w} \mid x)} - \beta \log \frac{\pi_{\theta}(y_{l} \mid x)}{\pi_{\mathrm{ref}}(y_{l} \mid x)} \right) \right]
$$

**DPO recipe:**
1. Collect offline preference dataset $\mathcal{D} = \{x, y_w, y_l\}$.
2. Initialize reference policy $\pi_{\text{ref}}$ (typically the SFT model).
3. Minimize DPO loss — a simple binary cross-entropy on log-ratios — with no rollout generation, no RM, no value network.

**DPO results (Rafailov et al.).** On 6B models: DPO's reward–KL frontier strictly dominates PPO (even PPO with ground-truth rewards) on sentiment generation; 61% GPT-4 win rate vs 57% for PPO on TL;DR summarization; only efficient method improving over preferred completions on Anthropic HH dialogue; better OOD generalization on CNN/DM (0.36 vs 0.26 win rate vs ground-truth summaries) [source:arxiv:2305.18290].

**Stated DPO limitations:** OOD generalization needs more study; unknown if self-labeling works like RLHF; slight late-training performance dip suggests over-optimization; only tested up to 6B; GPT-4 win rates sensitive to evaluation prompt [source:arxiv:2305.18290].

## PPO vs DPO: Empirical and Theoretical Comparison

**Theoretical limitation of DPO.** The "Is DPO Superior to PPO?" paper proves that the policy class induced by PPO is a **proper subset** of that induced by DPO: $\Pi_{\text{PPO}} \subsetneq \Pi_{\text{DPO}}$ [source:arxiv:2404.10719]. Any PPO solution minimizes the DPO objective, but DPO can find solutions that do *not* maximize the RL objective — specifically, DPO may assign high probability to OOD responses absent from the preference dataset, leading to biased policies and unpredictable behavior.

**Critical PPO recipe for SOTA performance.** The same paper identifies three factors that make PPO work at scale [source:arxiv:2404.10719]:
1. **Advantage normalization** — stabilizes training.
2. **Large batch size** — especially critical for code generation.
3. **Reference model EMA** — updating $\pi_{\text{ref}}$ via exponential moving average prevents over-regularization to a stale reference.

**Quantitative comparison (CodeLlama-34B).**
| Task | PPO | DPO / DPO-Iter | SFT / Baselines |
|---|---|---|---|
| CodeContest (10@1k pass) | **22.4%** | 0% (DPO, 1 epoch) | 16.4% (AlphaCode-41B) |
| APPS (pass@5) | **44.4%** | 34.2% (DPO-Iter) | 38.6% |
| HH-RLHF (OpenAssistant reward) | **0.718** | 0.611 (DPO), 0.678 (DPO-Iter) | — |
| HH-RLHF (GPT-4 eval) | 42% win | 30% win | 28% tie |
| SafeRLHF (safety rate) | **99.5%** | 55.4% | — |

PPO consistently outperforms DPO and iterative DPO across dialogue and code generation [source:arxiv:2404.10719]. *Caveat:* Code tasks used ground-truth rewards for PPO and for labeling DPO-Iter pairs, not a learned RM.

**Open-source PPO experience (Interconnects).** The Interconnects team replicated industry-style PPO gains using a Jax implementation on Llama-2-13B with UltraFeedback data [source:interconnects:rlhf-roundup-trying-to-get-good-at-ppo-i]:
- DPO on HH-RLHF: +1.3 score vs instruct baseline.
- DPO on UltraFeedback: +2.9 score.
- PPO on UltraFeedback: **+1.2 additional** over DPO UltraFeedback.
- Llama-3-Instruct post-training adds ~2.5 MMLU points.
- RewardBench top models now >90% accuracy (up from 60–80%).
- **Limitations:** PPO models remained "extremely yappy and a little unhinged" (AlpacaEval length-controlled scores misleading); success more consistent on Llama-2 bases; 70B open models still far below Llama-3-Instruct; larger (70B) RM and reasoning augmentation did *not* yield consistent gains; RewardBench has dataset bugs and uncertain correlation with downstream RLHF.

**P3O perspective.** P3O argues the root cause of PPO instability is the **mismatch between comparative RM training and absolute RL optimization**: a reward shift $\delta(x)$ leaves preferences unchanged but alters the PPO objective [source:bair:rethinking-the-role-of-ppo-in-rlhf]. P3O's pairwise policy gradient is reward-translation-invariant. On the KL–reward frontier: DPO > P3O > PPO > SFT in both KL and proxy reward, but **GPT-4 win rates reverse DPO vs P3O** (P3O beats DPO 54.6% despite lower reward) because DPO's higher KL produces lower-quality generations [source:bair:rethinking-the-role-of-ppo-in-rlhf].

## Known Issues and Limitations

**Reward hacking / over-optimization.** InstructGPT observes that plain PPO regresses on public NLP benchmarks (alignment tax) and mitigates it with PPO-ptx, but does not eliminate it [source:arxiv:2203.02155]. N+ finds **1B models over-optimize**: KL divergence reaches 50–85 nats and win rate drops below 20% [source:arxiv:2403.17031]. P3O argues the root cause is the **mismatch between comparative RM training and absolute RL optimization** [source:bair:rethinking-the-role-of-ppo-in-rlhf]. StackLLaMA observes concrete reward hacking: repetitive code backticks exploited because RM associates code blocks with high scores [source:huggingface:stackllama-a-hands-on-guide-to-rlhf-with].

**KL–reward trade-off.** P3O shows the KL–reward frontier: DPO > P3O > PPO > SFT in both KL and proxy reward, but **GPT-4 win rates reverse the order for DPO vs P3O** (P3O beats DPO 54.6% despite lower reward) because DPO's higher KL produces lower-quality generations [source:bair:rethinking-the-role-of-ppo-in-rlhf]. InstructGPT does not report KL values; N+ does not compare DPO.

**Labeler and evaluation bias.** InstructGPT acknowledges labelers (~40, English-speaking, US/SE Asia) are not representative; most comparisons have a single labeler [source:arxiv:2203.02155]. N+ uses GPT-3.5 as a judge, noting its own biases [source:arxiv:2403.17031]. P3O uses GPT-4 as judge [source:bair:rethinking-the-role-of-ppo-in-rlhf]. Interconnects notes LMSYS Chatbot Arena prompt diversity introduces noise that lowers maximum possible accuracy [source:interconnects:rlhf-roundup-trying-to-get-good-at-ppo-i].

**Computational cost.** InstructGPT: 4.9 PFLOPs-days (SFT) + 60 PFLOPs-days (PPO-ptx) for 175B vs 3,640 for GPT-3 pre-training [source:arxiv:2203.02155]. HuggingFace notes on-policy PPO requires **four large models simultaneously** (Actor, Critic, RM, Reference) and generation is the bottleneck [source:huggingface:navigating-the-rlhf-landscape-from-polic]. Cameron Wolfe emphasizes high variance of basic policy gradients necessitates many trajectories [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people]. StackLLaMA: 20 hours on 3×8 A100-80GB for 7B PPO with LoRA [source:huggingface:stackllama-a-hands-on-guide-to-rlhf-with]. DPO needs only 2 models (policy + ref) and no rollout [source:huggingface:navigating-the-rlhf-landscape-from-polic].

**RM calibration.** N+ finds RMs under-calibrated; accuracy varies wildly across data batches (1B: 50.8%–77.1%) and confidence levels [source:arxiv:2403.17031]. InstructGPT does not report calibration.

**PPO implementation fragility.** The original PPO paper notes adaptive KL penalty performed worse than clipped surrogate in experiments [source:arxiv:1707.06347]. Interconnects describes the open-source PPO stack as "fickle and broken" compared to industry; success more consistent on Llama-2 bases; less success transferring to other bases [source:interconnects:rlhf-roundup-trying-to-get-good-at-ppo-i]. Cameron Wolfe calls PPO "complicated and packed with nuanced implementation details" with high compute/memory overhead [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people].

## Current Status and Trajectory

The **SFT→RM→PPO pipeline remains the foundational reference** for industrial RLHF (InstructGPT, ChatGPT, GPT-4, Claude, Llama 2/3) but **pure PPO is fading in open research** in favor of **offline preference optimization** (DPO, IPO, KTO, SimPO) and **pairwise on-policy methods** (P3O, GRPO) that avoid the RM–RL mismatch and reduce infrastructure burden [source:bair:rethinking-the-role-of-ppo-in-rlhf][source:arxiv:2403.17031]. The N+ reproduction (2024) is valuable precisely because **faithful open-source PPO implementations were scarce** [source:arxiv:2403.17031]. Major labs still use PPO at scale (Llama 3 report cites PPO), but **published details are not widely reported**; the trajectory is toward **simpler, offline, or reward-translation-invariant algorithms** [source:bair:rethinking-the-role-of-ppo-in-rlhf]. **Hedge:** No source claims PPO is abandoned; it remains the production workhorse where compute and engineering capacity exist, but the *research frontier* has moved to alternatives that mitigate its instability and cost.

## Key Takeaways

- The InstructGPT recipe (SFT → pairwise RM → PPO with KL penalty + optional pretraining mix) is the canonical RLHF pipeline [source:arxiv:2203.02155].
- **Critical implementation details** (dropout disabled, EOS-token reward extraction, reward normalization target, left/right padding, value init from RM, EOS trick, whitening) materially affect results and are **not all in the original paper** [source:arxiv:2403.17031].
- **Reward translation invariance** breaks the theoretical link between RM training (comparative) and PPO (absolute); P3O fixes this with a pairwise policy gradient but is not yet widely adopted [source:bair:rethinking-the-role-of-ppo-in-rlhf].
- **Over-optimization** appears at small model scales (1B) as KL explosion and quality collapse; larger models (2.8B+) are more robust [source:arxiv:2403.17031].
- **DPO's implicit RM** regresses in validation accuracy vs explicit RM (N+), and DPO achieves higher proxy reward but higher KL and lower GPT-4 win rate than P3O (P3O) — **optimizing the proxy reward too hard hurts quality** [source:arxiv:2403.17031][source:bair:rethinking-the-role-of-ppo-in-rlhf].
- **PPO can outperform DPO significantly** on code and dialogue when the "PPO recipe" (advantage normalization, large batch, ref-model EMA) is used, but open-source implementations struggle to replicate this [source:arxiv:2404.10719][source:interconnects:rlhf-roundup-trying-to-get-good-at-ppo-i].
- **Compute cost**: PPO requires 4 large models + rollout generation; offline methods (DPO) need only 2 (policy + ref) and no rollout [source:huggingface:navigating-the-rlhf-landscape-from-polic].
- **Evaluation remains fragile**: human labelers are narrow; LLM judges (GPT-3.5/4) have biases; RM calibration is poor [source:arxiv:2203.02155][source:arxiv:2403.17031][source:bair:rethinking-the-role-of-ppo-in-rlhf].
- **Parameter-efficient RLHF** (LoRA + quantization + packing) makes the full pipeline feasible on consumer-grade GPU clusters (StackLLaMA: 7B on 3×8 A100-80GB) [source:huggingface:stackllama-a-hands-on-guide-to-rlhf-with].

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
- [RL for reasoning models](rl-for-reasoning.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)
- [DPO variants deep-dive](dpo-variants.md)
- [RLAIF (RL from AI feedback)](rlaif.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Nash and game-theoretic preference optimization](nash-and-game-theoretic-po.md)
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Length and format bias](length-and-format-bias.md)
- [The alignment tax](alignment-tax.md)
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md)
- [LLM-as-judge](llm-as-judge.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [Judging bias and contamination](judging-bias-and-contamination.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)

## References
- [source:arxiv:2403.17031] [The N+ Implementation Details of RLHF with PPO: A Case ...](https://arxiv.org/abs/2403.17031)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [source:medium:inside-the-rlhf-engine-a-deep-dive-into-] [Inside the RLHF Engine: A Deep Dive into SFT, Reward ...](https://medium.com/foundation-models-deep-dive/inside-the-rlhf-engine-a-deep-dive-into-sft-reward-models-and-rl-fine-tuning-60978b291d55)
- [source:bair:rethinking-the-role-of-ppo-in-rlhf] [Rethinking the Role of PPO in RLHF](http://bair.berkeley.edu/blog/2023/10/16/p3o/)
- [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people] [PPO for LLMs: A Guide for Normal People](https://cameronrwolfe.substack.com/p/ppo-llm)
- [source:huggingface:navigating-the-rlhf-landscape-from-polic] [Navigating the RLHF Landscape: From Policy Gradients to PPO, GAE, and DPO for LLM Alignment](https://huggingface.co/blog/NormalUhr/rlhf-pipeline)
- [source:arxiv:1706.03741] [Deep reinforcement learning from human preferences (Christiano et al. 2017)](https://arxiv.org/abs/1706.03741)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:cameronrwolfe:ppo-for-llms-a-guide-for-normal-people-c] [PPO for LLMs: A Guide for Normal People (Cameron R. Wolfe)](https://cameronrwolfe.substack.com/p/ppo-llm)
- [source:arxiv:2404.10719] [Is DPO Superior to PPO for LLM Alignment? A Theoretical and Empirical Study](https://arxiv.org/abs/2404.10719)
- [source:interconnects:rlhf-roundup-trying-to-get-good-at-ppo-i] [RLHF roundup: Trying to get good at PPO (Interconnects)](https://www.interconnects.ai/p/rlhf-roundup-2024)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [source:huggingface:stackllama-a-hands-on-guide-to-rlhf-with] [StackLLaMA: A hands-on guide to RLHF with LLaMA (Hugging Face)](https://huggingface.co/blog/stackllama)
