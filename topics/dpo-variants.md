---
title: DPO variants deep-dive
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2305.18290
- arxiv:2310.12036
- arxiv:2402.01306
- arxiv:2403.07691
- arxiv:2405.14734
- arxiv:2401.06518
- arxiv:2410.15595
- arxiv:2410.04203
open_questions:
- How does IPO's squared-error loss behave on large LMs with highly imbalanced or
  noisy preference data, and does the support-matching requirement ($\text{Supp}(\mu)=\text{Supp}(\pi_{\mathrm{ref}})$)
  hold in practice?
- Can KTO's prospect-theoretic value function be replaced or calibrated with a more
  text-appropriate utility model to eliminate underfitting on clean preferences without
  sacrificing binary-signal efficiency?
- Does ORPO's monolithic odds-ratio objective scale to >7B parameters and diverse
  instruction distributions, or does the lack of an explicit KL anchor cause drift
  at scale?
- What is the optimal schedule or adaptive rule for SimPO's margin $\gamma$ (and $\beta$)
  across training, and can the reasoning-task degradation be mitigated without sacrificing
  alignment gains?
---

Direct Preference Optimization (DPO) established a reference-model-based, offline preference learning paradigm that avoids explicit reward modeling and RL loops [source:arxiv:2305.18290]. Subsequent variants—IPO, KTO, ORPO, SimPO, and CPO—each relax a different DPO assumption: the Bradley–Terry (BT) link, the need for paired preferences, the separate SFT+alignment stages, the reference-model dependency, and the token-level credit assignment, respectively.

## Identity Preference Optimization (IPO)

### Loss and derivation
IPO arises from the $\Psi$PO framework, which generalizes the KL-regularized preference objective to $\max_\pi \mathbb{E}[\Psi(p^*(y\succ y'\mid x))] - \tau D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})$ [source:arxiv:2310.12036]. Setting $\Psi(q)=q$ (identity mapping) yields an objective that directly maximizes total preference probability minus KL divergence, bypassing the BT assumption $\Psi(q)=\log\frac{q}{1-q}$ used by DPO/RLHF. The optimal policy satisfies a root-finding condition on the log-ratio gap:

$$
h_\pi(y_w,y_l,x) = \log\frac{\pi(y_w\mid x)\pi_{\mathrm{ref}}(y_l\mid x)}{\pi(y_l\mid x)\pi_{\mathrm{ref}}(y_w\mid x)} = \frac{\tau^{-1}}{2}
$$

leading to the **IPO loss** (sampled squared error):

$$
\mathcal{L}_{\mathrm{IPO}} = \mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[ \left( h_\pi(y_w,y_l,x) - \frac{\tau^{-1}}{2} \right)^2 \right]
$$

[source:arxiv:2310.12036].

### Trade-offs
- **Deterministic preferences:** In a two-action bandit with $p^*(y_1\succ y_2)=1$, DPO drives $\pi(y_1)\to1$ for any $\tau$, ignoring the KL penalty; IPO converges to $\pi(y_1)=\sigma(0.5\tau^{-1})$, so $\tau$ retains control [source:arxiv:2310.12036].
- **Unobserved pairs:** DPO can push probabilities of never-winning actions to 0 and never-losing actions to 1 regardless of $\tau$; IPO stays near $\pi_{\mathrm{ref}}$ proportional to $\tau$ [source:arxiv:2310.12036].
- **Support requirement:** IPO's global minimum is unique only if $\text{Supp}(\mu)=\text{Supp}(\pi_{\mathrm{ref}})$; mismatched supports admit multiple solutions [source:arxiv:2310.12036].
- **Scale:** Results are shown on illustrative bandits; the authors explicitly note that scaling to generative LMs is future work [source:arxiv:2310.12036]. The DPO survey confirms IPO's large-scale empirical validation remains limited [source:arxiv:2410.15595].

## Kahneman–Tversky Optimization (KTO)

### Loss and derivation
KTO frames alignment as maximizing *prospect-theoretic* human utility over binary (desirable/undesirable) signals, not paired preferences [source:arxiv:2402.01306]. The implied reward is $r_\theta(x,y)=\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}$. A reference point $z_0$ (estimated via microbatch shift $\hat{z}_0 = \max(0, \frac{1}{m}\sum_{i<m}\log\frac{\pi_\theta(y_j\mid x_i)}{\pi_{\mathrm{ref}}(y_j\mid x_i)})$) separates gains from losses. The **KTO loss** uses a logistic value function:

$$
v(x,y) = \begin{cases}
\lambda_D \sigma(\beta(r_\theta(x,y)-z_0)) & y\sim y_{\mathrm{desirable}}\\
\lambda_U \sigma(\beta(z_0-r_\theta(x,y))) & y\sim y_{\mathrm{undesirable}}
\end{cases}
$$

$$
\mathcal{L}_{\mathrm{KTO}} = \mathbb{E}_{x,y\sim\mathcal{D}}[\lambda_y - v(x,y)]
$$

[source:arxiv:2402.01306].

### Trade-offs
- **Data efficiency:** KTO matches DPO with up to **90% fewer desirable examples** (1:10 desirable:undesirable ratio) and a "one-$y$-per-$x$" setup (72% less data) beat DPO on OpenAssistant (win rate $0.631\pm0.036$ vs $0.600\pm0.037$) [source:arxiv:2402.01306].
- **SFT independence:** At 13B/30B, KTO can skip SFT without rambling/hallucinations; DPO without SFT degrades severely [source:arxiv:2402.01306].
- **Hyperparameter sensitivity:** KTO needs **2–10× higher learning rate** than DPO to compensate for smaller reference-adjusted rewards [source:arxiv:2402.01306].
- **Underfitting risk:** On exceptionally clean (low-noise, low-intransitivity) preferences, KTO's gradient vanishes as rewards become extreme, causing underfitting [source:arxiv:2402.01306].
- **Reference dependency:** A reference-free variant exists but is less performant [source:arxiv:2402.01306].

## Odds Ratio Preference Optimization (ORPO)

### Loss and derivation
ORPO merges SFT and preference alignment into a **single stage** without a reference model [source:arxiv:2403.07691]. It adds an odds-ratio penalty to the NLL loss on the chosen response $y_w$:

$$
\mathcal{L}_{\mathrm{ORPO}} = \mathbb{E}_{(x,y_w,y_l)}\left[ \mathcal{L}_{\mathrm{SFT}} + \lambda \mathcal{L}_{\mathrm{OR}} \right]
$$

where $\mathcal{L}_{\mathrm{SFT}} = -\frac{1}{|y_w|}\sum_{t=1}^{|y_w|}\log\pi_\theta(y_t\mid x,y_{<t})$ and the odds-ratio loss is

$$
\mathcal{L}_{\mathrm{OR}} = -\log\sigma\left( \log\frac{\mathbf{odds}_\theta(y_w\mid x)}{\mathbf{odds}_\theta(y_l\mid x)} \right),\quad
\mathbf{odds}_\theta(y\mid x) = \frac{P_\theta(y\mid x)}{1-P_\theta(y\mid x)},\quad
P_\theta(y\mid x) = \exp\left(\frac{1}{|y|}\sum_{t=1}^{|y|}\log\pi_\theta(y_t\mid x,y_{<t})\right)
$$

[source:arxiv:2403.07691].

### Trade-offs
- **Efficiency:** Eliminates the reference model and separate SFT phase; one forward/backward pass per batch.
- **Benchmarks:** Mistral-7B+ORPO on UltraFeedback reaches **12.20% AlpacaEval 2.0**, **7.32 MT-Bench**, **66.19% IFEval**; Llama-2-7B+ORPO scores **9.44% AlpacaEval 2.0** (vs 7.70% for Llama-2-Chat-13B) [source:arxiv:2403.07691].
- **Reward-model win rates:** On HH-RLHF, ORPO beats SFT (84.0%), PPO (79.4%), DPO (70.9%) per OPT-1.3B RM; on UltraFeedback, beats SFT (80.5%) and PPO (85.8%) [source:arxiv:2403.07691].
- **Scale limit:** Not tested beyond 7B parameters [source:arxiv:2403.07691].
- **Dataset diversity:** Only UltraFeedback and HH-RLHF evaluated; generalization to diverse NLP tasks unverified [source:arxiv:2403.07691].
- **Internal dynamics:** Weight/representation changes not analyzed [source:arxiv:2403.07691].

## Simple Preference Optimization (SimPO)

### Loss and derivation
SimPO removes the reference model *and* aligns the training reward with the generation metric (average log-likelihood) by using a **length-normalized, reference-free reward** with a **target margin** $\gamma$ [source:arxiv:2405.14734]:

$$
r_{\mathrm{SimPO}}(x,y) = \frac{\beta}{|y|}\log\pi_\theta(y\mid x)
$$

$$
\mathcal{L}_{\mathrm{SimPO}} = -\mathbb{E}_{(x,y_w,y_l)}\left[ \log\sigma\left( \frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x) - \frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x) - \gamma \right) \right]
$$

The margin $\gamma>0$ forces $r(y_w) \ge r(y_l) + \gamma$, improving generalization; length normalization ($/|y|$) prevents length exploitation [source:arxiv:2405.14734].

### Trade-offs
- **Performance:** Consistent gains over DPO: up to **+6.4 AlpacaEval 2**, **+7.5 Arena-Hard**. Gemma-2-9B-it-SimPO hits **72.4% LC win rate** (AlpacaEval 2) and **59.1%** (Arena-Hard), ranking 1st among <10B models on Chatbot Arena (Sept 2024) [source:arxiv:2405.14734].
- **Efficiency:** ~20% less runtime, ~10% less peak GPU memory vs vanilla DPO on Llama-3-Base (8×H100) [source:arxiv:2405.14734].
- **Length control:** Spearman $\rho$ between likelihood and length stays near SFT ($\rho=0.34$); without normalization $\rho=0.82$ (strong length exploitation) [source:arxiv:2405.14734].
- **Hyperparameter tuning:** $\gamma$ requires manual tuning [source:arxiv:2405.14734].
- **Reasoning drop:** Preference optimization (including SimPO) can hurt reasoning-heavy tasks (e.g., GSM8K) [source:arxiv:2405.14734].
- **Safety/honesty:** Not explicitly constrained, though competitive on TruthfulQA [source:arxiv:2405.14734].
- **Theory:** Authors call for more rigorous analysis [source:arxiv:2405.14734].

## Contrastive Preference Optimization (CPO)

### Source discrepancy
The provided source [source:arxiv:2401.06518] is titled "Contrastive Preference Optimization: Pushing the Boundaries of LLM Alignment in Large-Scale Settings" but its summary describes **Transitional Grid Maps (TGM) for joint static/dynamic occupancy mapping**—a robotics/SLAM method—with no LLM alignment content. The DPO survey [source:arxiv:2410.15595] and RainbowPO [source:arxiv:2410.04203] do not detail a CPO method under this arXiv ID. Consequently, **no loss formulation, empirical results, or trade-offs for CPO can be extracted from the given sources**. This appears to be a metadata mismatch in the supplied corpus.

## Current status and trajectory

- **IPO:** Theoretical interest is high (addresses BT misspecification), but **large-scale adoption is not widely reported**; the ΨPO paper's bandit-scale experiments and the survey's note on limited scaling [source:arxiv:2410.15595] suggest it remains a **research-stage** method rather than a default.
- **KTO:** **Rising** for data-scarce and SFT-free regimes; the binary-signal advantage is practically valuable, but the 2–10× LR requirement and underfitting on clean data [source:arxiv:2402.01306] mean it is **not a drop-in default**—practitioners must tune aggressively.
- **ORPO:** **Gaining traction for small-to-medium models** (≤7B) due to monolithic efficiency and strong benchmark numbers [source:arxiv:2403.07691]; the 7B ceiling and narrow dataset evaluation mean **scaling to frontier models is unverified**—not yet a general default.
- **SimPO:** **Rapidly rising**; reference-free + length-normalized + margin design yields SOTA on <10B leaderboards [source:arxiv:2405.14734] and RainbowPO confirms length normalization + mixed reference + contextual scaling as the winning combination [source:arxiv:2410.04203]. The need to tune $\gamma$ and reasoning-task drops [source:arxiv:2405.14734] are the main adoption frictions.
- **CPO:** **Status indeterminate** due to source mismatch; no trajectory can be assessed from provided materials.
- **RainbowPO synthesis:** The RainbowPO ablation [source:arxiv:2410.04203] identifies **length normalization ($\eta=1$) as the single most critical component** (removing it drops LC WR 51.66%→45.68%), with mixed reference policy and contextual scaling as secondary. This suggests the field is **converging on a "default stack"** of length-normalized, margin-augmented, mixed-reference objectives—SimPO plus reference mixing—rather than any single XPO variant.

## Key takeaways

- **IPO** fixes DPO's BT-induced overfitting on deterministic preferences via a squared-error loss on the log-ratio gap, but lacks large-scale LM validation and requires matched behavior/reference support [source:arxiv:2310.12036].
- **KTO** enables alignment from binary (desirable/undesirable) signals using prospect-theoretic gains/losses, matching DPO with far fewer desirable examples and allowing SFT-free training at scale, at the cost of higher LR sensitivity and underfitting on clean data [source:arxiv:2402.01306].
- **ORPO** unifies SFT and preference alignment in one reference-free stage via an odds-ratio penalty, achieving strong <7B benchmarks but untested at larger scales and on diverse tasks [source:arxiv:2403.07691].
- **SimPO** replaces the reference-model reward with a length-normalized average log-likelihood plus a target margin $\gamma$, delivering SOTA <10B results with ~20% speedup, but requires $\gamma$ tuning and can degrade reasoning [source:arxiv:2405.14734].
- **CPO** cannot be evaluated from the provided sources due to a metadata/content mismatch [source:arxiv:2401.06518].
- **RainbowPO** demonstrates that the empirically essential ingredients are length normalization, mixed reference policy, and contextual scaling—pointing toward a converging "best practice" stack rather than a single variant dominating [source:arxiv:2410.04203].

## Related topics

- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Length and format bias](length-and-format-bias.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [RL for reasoning models](rl-for-reasoning.md)

## References
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2310.12036] [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036)
- [source:arxiv:2402.01306] [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306)
- [source:arxiv:2403.07691] [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
- [source:arxiv:2405.14734] [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [source:arxiv:2401.06518] [Contrastive Preference Optimization: Pushing the Boundaries of LLM Alignment in Large-Scale Settings](https://arxiv.org/abs/2401.06518)
- [source:arxiv:2410.15595] [A Comprehensive Survey of Direct Preference Optimization](https://arxiv.org/abs/2410.15595)
- [source:arxiv:2410.04203] [Rainbow PO: A Unified Framework for Combining Improvements in Preference Optimization](https://arxiv.org/abs/2410.04203)
