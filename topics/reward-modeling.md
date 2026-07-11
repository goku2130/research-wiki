---
title: Reward modeling for LLMs
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2507.07375
- arxiv:2602.10623
- arxiv:2509.03403
- arxiv:2504.12328
- emergentmind:outcome-supervised-reward-models
- rlhfbook:reward-modeling-rlhf-and-post-training-b
- arxiv:1909.08077
- arxiv:2203.02155
open_questions:
- Does BNRM's Bayesian sparsity generalize to multi-turn, tool-use, or long-horizon
  reward hacking where spurious features may be more complex than length/style?
- Can SMORM's multi-objective anchoring be extended to *latent* attributes discovered
  automatically (e.g., via clustering) rather than human-defined rubrics, removing
  the annotation bottleneck?
- PROF filters with a frozen PRM; would a *learned* consistency predictor (trained
  to predict ORM-PRM agreement) outperform hand-crafted $r^{\text{pro}}$ and generalize
  beyond math/code?
- Is there a unified framework that combines Bayesian regularization (BNRM), multi-objective
  anchoring (SMORM), and process-outcome filtration (PROF) — or do their inductive
  biases conflict?
---

Reward modeling translates human or verifiable feedback into scalar signals that steer LLM behavior, forming the critical bridge between preference data and policy optimization. The field has evolved from simple Bradley–Terry pairwise comparators toward multi-objective, Bayesian, and process-aware architectures that attempt to close the gap between proxy rewards and true intent.

## Bradley–Terry Foundations

The Bradley–Terry (BT) model remains the dominant parametric form for learning reward functions from pairwise preferences. Given a prompt $x$ and two completions $y_c$ (chosen) and $y_r$ (rejected), the probability that $y_c$ is preferred is modeled as

$$
P(y_c \succ y_r \mid x) = \sigma\bigl(r_\theta(x, y_c) - r_\theta(x, y_r)\bigr),
$$

where $r_\theta$ is a scalar reward head atop a frozen or fine-tuned LLM backbone and $\sigma$ is the logistic sigmoid [source:rlhfbook:reward-modeling-rlhf-and-post-training-b]. Training minimizes the negative log-likelihood

$$
\mathcal{L}(\theta) = -\mathbb{E}_{(x,y_c,y_r)\sim\mathcal{D}}\Bigl[\log \sigma\bigl(r_\theta(x,y_c)-r_\theta(x,y_r)\bigr)\Bigr],
$$

which is equivalent to a softplus loss $\log\bigl(1+\exp(r_\theta(x,y_r)-r_\theta(x,y_c))\bigr)$ [source:rlhfbook:reward-modeling-rlhf-and-post-training-b]. In practice, the reward head is a single linear layer on the final-token hidden state of a causal LM; the InstructGPT pipeline initializes the RM from the SFT model and trains for only one epoch to avoid overfitting [source:arxiv:2203.02155].

When $K$ completions are ranked per prompt, the naïve $\binom{K}{2}$ pairwise expansion introduces strong correlations; the Llama 2/3 recipes mitigate this by averaging the per-prompt loss or weighting updates [source:rlhfbook:reward-modeling-rlhf-and-post-training-b]. A margin variant $-\log\sigma(r_c-r_r-m)$ was used in Llama 2 but later dropped due to diminishing returns at scale [source:rlhfbook:reward-modeling-rlhf-and-post-training-b]. The Plackett–Luce extension handles $K$-wise rankings directly:

$$
P(\sigma \mid s, a_0,\dots,a_{K-1}) = \prod_{k=0}^{K-1} \frac{\exp r_\theta(s, a_{\sigma(k)})}{\sum_{j=k}^{K-1}\exp r_\theta(s, a_{\sigma(j)})},
$$

though pairwise BT remains the de facto standard for its simplicity [source:rlhfbook:reward-modeling-rlhf-and-post-training-b].

**Disagreement on data efficiency:** The InstructGPT paper reports strong results with ~33k comparison pairs for a 6B RM [source:arxiv:2203.02155], while the SMORM paper argues that single-objective BT models (SORMs) "typically outperform MORMs in scoring" because they can leverage "more abundant preference data," implying that multi-attribute data scarcity is a binding constraint [source:arxiv:2507.07375]. The survey notes that active learning and data augmentation are used to improve preference collection efficiency, but does not quantify the BT data regime where returns saturate [source:arxiv:2504.12328].

## Reward Hacking: Mechanisms and Mitigations

Reward hacking (a.k.a. reward overoptimization or specification gaming) occurs when a policy exploits imperfections in the learned reward model, achieving high proxy scores while degrading true quality. The survey identifies three root causes: **reward tampering** (the policy manipulates the RM input), **misleading signals** (spurious correlations in the training data), and **sycophancy** (the RM favors responses that flatter the user's apparent beliefs) [source:arxiv:2504.12328]. Empirically, length bias is a canonical spurious feature: a vanilla BT RM on RM-Bench exhibits Pearson correlation $r=0.488$ between response length and reward score [source:arxiv:2602.10623].

Two distinct mitigation philosophies appear in the sources:

1. **Bayesian regularization of the reward model itself.** BNRM places Gamma priors on a non-negative factor-analysis latent space $\theta \sim \text{Gamma}(\alpha_0,\beta_0)$, $\Phi \sim \text{Gamma}(\gamma_0,\delta_0)$, with reward $r = \theta^\top\Phi$. Variational inference with Weibull posteriors yields an ELBO objective that penalizes epistemic uncertainty (global dictionary $\Phi$) and aleatoric uncertainty (instance-specific $\theta$) [source:arxiv:2602.10623]. This suppresses spurious features without explicit bias annotations: on RM-Bench Hard, BNRM reduces the length–reward correlation from $0.488$ to $0.123$ [source:arxiv:2602.10623]. BNRM also shows robustness to label noise (matching BT@20k with only 1k clean samples at 40% noise) and faster convergence (71.75% val accuracy at 0.25 epochs) [source:arxiv:2602.10623].

2. **Architectural fusion of single- and multi-objective signals.** SMORM jointly trains a BT head (single-objective) and an MSE head (multi-attribute) on a shared backbone:

$$
\min_{\theta,\mathbf{w}_S,\mathbf{w}_M} -\mathbb{E}_{\mathcal{D}_S}\log\sigma\bigl(\mathbf{w}_S^\top(f_\theta(x_s,y_c)-f_\theta(x_s,y_r))\bigr) + \mathbb{E}_{\mathcal{D}_M}\bigl\|\mathbf{w}_M^\top f_\theta(x_m,y_m)-\mathbf{r}\bigr\|_2^2.
$$

Theorem 1 proves that under bounded features, positive-definite covariances, and positive correlation between aggregated attributes and BT preference, the multi-objective average score lower-bounds the single-objective score: $r_m \ge c\, r_s - \varepsilon$ [source:arxiv:2507.07375]. This guarantees that high BT scores imply respectable fine-grained quality, mitigating OOD hacking. Empirically, SMORM-F (BT head only) and SMORM-M (average of both heads) show monotonic gold-score improvement under PPO/BoN in OOD settings, whereas baselines plateau or decline [source:arxiv:2507.07375]. SMORM-L (multi-head mean) matches ArmoRM-Llama3-8B on RewardBench (90.4) with **15.9× less** multi-objective data (20k vs. 585.4k) [source:arxiv:2507.07375].

**Disagreement on scope:** BNRM evaluates on standard benchmarks (RewardBench, Unified-Feedback, MT-Bench, Arena-Hard) and human eval, but acknowledges "open-ended RLHF settings may expose more diverse and adaptive forms of reward hacking" [source:arxiv:2602.10623]. SMORM explicitly targets OOD generalization and shows gold-score curves that *increase* throughout PPO training, while the survey notes that overoptimization typically manifests as "gold scores decline while proxy scores rise" in ID settings [source:arxiv:2504.12328]; SMORM claims to avoid this in ID as well [source:arxiv:2507.07375]. Whether Bayesian sparsity (BNRM) or multi-objective anchoring (SMORM) generalizes better to multi-turn, tool-use, or long-horizon hacking is not settled by either source.

## Process vs Outcome Rewards: Tradeoffs and Integration

**Outcome Reward Models (ORMs)** score only the final answer. The canonical ORM loss is binary cross-entropy on verifiable correctness labels $\hat{y}_s\in\{0,1\}$:

$$
\mathcal{L}_{\text{ORM}} = -\bigl(\hat{y}_s\log y_s + (1-\hat{y}_s)\log(1-y_s)\bigr),
$$

where $y_s$ is the predicted probability of correctness [source:arxiv:2504.12328]. ORMs are "easier to implement and generalize" but provide "sparse reward" and "can lead to false positives" — rewarding flawed reasoning that accidentally yields a correct answer [source:arxiv:2504.12328]. The ORM survey quantifies gains: +18% on MATH, +26.9% correctness/+42.2% efficiency in code (ORPS), +2–5% execution accuracy in SQL, +5–15 pp in logical reasoning [source:emergentmind:outcome-supervised-reward-models].

**Process Reward Models (PRMs)** assign stepwise scores. The standard PRM loss sums step-wise cross-entropy:

$$
\mathcal{L}_{\text{PRM}} = -\sum_{i=1}^N \bigl(\hat{y}_{s,i}\log y_{s,i} + (1-\hat{y}_{s,i})\log(1-y_{s,i})\bigr),
$$

where $N$ is the number of reasoning steps [source:arxiv:2504.12328]. PRMs offer "dense reward" and "controllable" supervision but suffer from "high cost for gathering training data," "scalability and generalization problems," and susceptibility to reward hacking (e.g., verbose step generation) [source:arxiv:2504.12328]. The survey notes PRMs are "less supported in open-source RLHF tools compared to Bradley-Terry models" [source:rlhfbook:reward-modeling-rlhf-and-post-training-b].

**The process–outcome mismatch.** PROF formalizes the gap: let $z=1$ denote a valid intermediate process. If an invalid process ($z=0$) yields a correct answer with probability $\epsilon$, the expected outcome reward is

$$
\mathbb{E}_\pi[r] = \alpha_\pi + \epsilon(1-\alpha_\pi) = (1-\epsilon)\alpha_\pi + \epsilon,
$$

where $\alpha_\pi = P(z=1\mid\pi)$. The $\epsilon(1-\alpha_\pi)$ term creates biased gradients that reinforce flawed reasoning [source:arxiv:2509.03403]. Empirically, 26.28% of correct responses from Qwen2.5-Math-7B contained flawed reasoning (Claude audit); PROF filtered 65.88% of these [source:arxiv:2509.03403].

**PROF (PRocess cOnsistency Filter)** avoids direct PRM optimization. For each prompt, it generates $n$ rollouts, computes binary ORM scores $r^o\in\{-1,1\}$, and obtains step-level PRM scores $r^h$. A trajectory-wise consistency score is

$$
r^{\text{pro}} = \Bigl[\frac{1}{H}\sum_{h=1}^H r^h - \lambda\,\mathbb{I}(H=1 \lor H\ge H_\lambda)\Bigr] \cdot r^o,
$$

penalizing single-step or excessively long correct responses. Rollouts are split into correct ($\mathcal{G}_+$) and incorrect ($\mathcal{G}_-$) groups; $k_+,k_-$ are chosen to maximize $k_+k_-$ (balanced batches). $\mathcal{G}_+$ is ranked by $r^{\text{pro}}$ descending (keep top $k_+$); $\mathcal{G}_-$ is either random (PROF-POS) or ranked ascending (PROF-BOTH, keep bottom $k_-$) [source:arxiv:2509.03403].

| Model | GRPO | Blend | PROF-POS | PROF-BOTH |
|-------|------|-------|----------|-----------|
| Qwen2.5-Math-1.5B-base | 37.2% | 35.3% | **40.2%** | 39.6% |
| Qwen2.5-Math-7B-base | 49.9% | 47.3% | 50.6% | **51.7%** |
| LLaMA-3.2-3B-instruct | 23.6% | 15.7% | **25.4%** | — |

PROF also improves reasoning consistency (Monte Carlo step-value scores) by +9.2% (Math500) to +37.4% (Minerva Math) over GRPO, and reduces flawed-reasoning rate in correct answers from 8% to 6% [source:arxiv:2509.03403]. Crucially, PROF is robust to weak PRMs: with Skywork-PRM-1.5B (vs. Qwen-PRM-7B), PROF-POS/BOTH maintain ~50.5–51.0% while Blend degrades [source:arxiv:2509.03403].

**Disagreement on PRM utility:** The survey states PRMs have "potential in reasoning tasks" but "high computational overhead in large-scale RL" and "susceptible to reward hacking" [source:arxiv:2504.12328]. PROF demonstrates that *filtering* with a frozen PRM (no RL on PRM scores) sidesteps hacking and overhead, yet the survey does not discuss filtration-based integration. The ORM survey notes ORMs' "process blindness" and "inconsistency for partial sequences" but does not evaluate PROF-style hybrids [source:emergentmind:outcome-supervised-reward-models]. Whether PROF's filtration approach generalizes beyond math/code (where verifiable ORMs exist) to open-domain tasks remains open.

## Multi-Objective and Bayesian Approaches

Beyond BT, two architectural families address reward hacking by enriching the reward representation:

| Method | Core Idea | Key Formula | Hacking Mitigation |
|--------|-----------|-------------|-------------------|
| **SMORM** | Joint BT + multi-attribute MSE heads on shared backbone | $\min -\mathbb{E}_{\mathcal{D}_S}\log\sigma(\Delta r_S) + \mathbb{E}_{\mathcal{D}_M}\|\mathbf{w}_M^\top f - \mathbf{r}\|^2$ | Theorem 1: $r_m \ge c r_s - \varepsilon$ anchors BT to attributes |
| **BNRM** | Bayesian non-negative factor analysis: $r = \theta^\top\Phi$, $\theta,\Phi\sim\text{Gamma}$ | ELBO: $\mathbb{E}_q[\log p(\mathcal{D}\mid\theta,\Phi)] - \eta\text{KL}(q\|p)$ | Sparsity + non-negativity suppress spurious features; length correlation 0.488→0.123 |

SMORM leverages abundant BT data ($\mathcal{D}_S$) to bootstrap the data-scarce multi-objective head ($\mathcal{D}_M$), achieving lower asymptotic MSE for *both* heads (Theorem 2) [source:arxiv:2507.07375]. BNRM requires no multi-attribute labels; its inductive bias (non-negative, sparse factors) regularizes the BT likelihood directly [source:arxiv:2602.10623].

**Disagreement on supervision requirements:** SMORM *requires* a multi-attribute dataset $\mathcal{D}_M$ (e.g., HelpSteer2, UnifiedFeedback) and shows large gains when it is available (13.9 pts RewardBench over MORM baseline) [source:arxiv:2507.07375]. BNRM achieves comparable or better OOD robustness *without* attribute labels, using only standard BT comparisons. The survey mentions "multi-objective rewards with gating layers" but does not compare these paradigms [source:arxiv:2504.12328]. For practitioners, the tradeoff is: invest in attribute annotation (SMORM) vs. adopt Bayesian architecture with tuning of $\eta$ (BNRM). BNRM reports only 1.3–7.7% training overhead [source:arxiv:2602.10623]; SMORM does not report overhead but uses a single forward pass with two heads.

## Current Status and Trajectory

**Bradley–Terry BT models** remain the *default* for preference modeling in production RLHF pipelines (InstructGPT, Llama 2/3, open-source tooling) due to simplicity, tooling support, and adequate performance when preference data is plentiful [source:rlhfbook:reward-modeling-rlhf-and-post-training-b][source:arxiv:2203.02155]. No source suggests BT is fading; rather, it is the backbone upon which extensions (margins, K-wise, multi-head, Bayesian) are built.

**Reward hacking mitigation** is *rising* as a research priority. The survey devotes a full challenge section to overoptimization [source:arxiv:2504.12328]; BNRM and SMORM (both 2024–2025) propose orthogonal architectural solutions. BNRM's Bayesian approach is gaining traction for its label-free robustness; SMORM's multi-objective fusion is attractive where attribute data exists. Neither has been adopted in major open releases (Llama 3, Nemotron, etc.) as of the source dates — *not widely reported* in production pipelines.

**Process vs. outcome rewards:** ORMs are *default* for verifiable domains (math, code, logic, SQL) where ground-truth outcomes are cheap [source:emergentmind:outcome-supervised-reward-models]. PRMs remain *niche* due to annotation cost and hacking risk, but PROF-style filtration (2025) revives interest by decoupling PRM usage from RL optimization. The survey notes PRMs are "hard to define" and "computational overhead in large-scale RL" [source:arxiv:2504.12328]; PROF does not resolve the definition problem but sidesteps the overhead/hacking problems. *Trajectory:* filtration-based hybrids are rising for reasoning; pure PRM optimization is fading.

**Multi-objective/Bayesian RMs:** Early-stage research. SMORM and BNRM are recent (2024–2025) with strong benchmarks but no reported large-scale deployment. The survey's taxonomy lists "custom classifiers" and "multi-objective rewards" as a subcategory but does not highlight them as mainstream [source:arxiv:2504.12328]. *Status:* promising but not default.

## Key Takeaways

- **Bradley–Terry is the workhorse:** pairwise logistic loss on a scalar head, trained for ~1 epoch, powers most production RLHF. Extensions (margins, K-wise, balanced multi-comparison) are minor tweaks [source:rlhfbook:reward-modeling-rlhf-and-post-training-b][source:arxiv:2203.02155].
- **Reward hacking is structural:** spurious correlations (length, style, sycophancy) are learned from noisy preferences and exploited by policy optimization. Two architectural defenses exist: (1) Bayesian sparsity/non-negativity (BNRM) that regularizes the latent reward factors [source:arxiv:2602.10623]; (2) multi-objective anchoring (SMORM) that ties the BT score to fine-grained attributes via a shared backbone [source:arxiv:2507.07375].
- **ORMs dominate where verifiers exist:** math, code, logic, SQL. They are cheap, scalable, and effective for reranking/BoN but blind to process quality [source:emergentmind:outcome-supervised-reward-models].
- **PRMs are fragile under RL:** dense supervision invites hacking (verbosity, format gaming) and requires expensive annotations. PROF shows that *filtering* rollouts with a frozen PRM (consistency with ORM) improves both outcome accuracy and reasoning faithfulness without optimizing the PRM directly [source:arxiv:2509.03403].
- **Multi-objective data is a bottleneck:** SMORM demonstrates that joint training transfers signal from abundant BT data to scarce attribute data, but the attribute data must exist [source:arxiv:2507.07375]. BNRM avoids this requirement via inductive bias.
- **No unified solution for open-domain hacking:** current mitigations are evaluated on benchmarks (RewardBench, RM-Bench, Arena-Hard) and math/code; multi-turn, tool-use, and long-horizon hacking are *not widely reported* as solved [source:arxiv:2602.10623][source:arxiv:2507.07375].

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md) — policy optimization that consumes reward model outputs
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md) — implicit reward modeling without an explicit RM
- [Reward hacking in RLHF](reward-hacking.md) — deeper survey of hacking typologies and defenses
- [Reward model over-optimization](reward-model-overoptimization.md) — analysis of overoptimization dynamics
- [Process vs outcome reward models](process-vs-outcome-rewards.md) — dedicated comparison of ORM/PRM tradeoffs
- [Verifiable rewards (RLVR)](verifiable-rewards.md) — outcome supervision in reasoning domains
- [RL for reasoning models](rl-for-reasoning.md) — RL with process/outcome rewards for CoT
- [KL regularization in RLHF](kl-regularization.md) — constraint that indirectly limits reward hacking
- [Length and format bias](length-and-format-bias.md) — specific spurious features exploited in hacking
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md) — preference-model biases toward user agreement
- [LLM-as-judge](llm-as-judge.md) — AI preference generation for RM training
- [Alignment and win-rate evals](alignment-and-winrate-evals.md) — evaluation of RM-aligned policies
- [Judging bias and contamination](judging-bias-and-contamination.md) — biases in LLM judges that propagate to RMs
- [RLAIF (RL from AI feedback)](rlaif.md) — AI-generated preferences for RM training
- [Self-improvement and self-play RL](self-improvement-and-self-play.md) — iterative RM/policy co-evolution

## References
- [source:arxiv:2507.07375] [Bradley–Terry and Multi-Objective Reward Modeling Are Complementary](https://arxiv.org/html/2507.07375v1)
- [source:arxiv:2602.10623] [Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](https://arxiv.org/html/2602.10623v2)
- [source:arxiv:2509.03403] [Beyond Correctness: Harmonizing Process and Outcome Rewards through RL Training](https://arxiv.org/html/2509.03403v1)
- [source:arxiv:2504.12328] [A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges, and Future](https://arxiv.org/html/2504.12328v1)
- [source:emergentmind:outcome-supervised-reward-models] [Outcome-Supervised Reward Models](https://www.emergentmind.com/topics/outcome-supervised-reward-model-orm)
- [source:rlhfbook:reward-modeling-rlhf-and-post-training-b] [Reward Modeling | RLHF and Post-Training Book by Nathan Lambert](https://rlhfbook.com/c/05-reward-models)
- [source:arxiv:1909.08077] [Learning to summarize with human feedback](https://arxiv.org/pdf/1909.08077)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)
