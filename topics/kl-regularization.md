---
title: KL regularization in RLHF
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2508.17000
- arxiv:2009.01325
- arxiv:2305.18290
- arxiv:2502.01203
- arxiv:2503.18130
- huggingface:topic-objectives-and-regularization-refe
- arxiv:2510.01555
- arxiv:2411.04625
- rlhfbook:rlhf-and-post-training-book-regularizati
- mbrenndoerfer:kl-divergence-penalty-in-rlhf-theory-imp
- arxiv:2203.02155
- huggingface:rl-llm-wiki-kl-regularization-knowledge-
- arxiv:1909.08593
- openrlhf:openrlhf-kl-penalty-documentation
open_questions:
- Off-policy KL gradient bias magnitude in production GRPO/PPO pipelines?
- Multi-reference RLHF compute/memory tradeoffs at 70B+ scale?
- Forward vs reverse KL empirical diversity comparison at scale?
- BSPO robustness to behavior policy estimation errors?
---

KL regularization is the central mechanism preventing reward over-optimization in RLHF, anchoring the policy to a frozen reference model (typically the SFT checkpoint) via a reverse-KL penalty. This article synthesizes the theoretical foundations—Boltzmann optimal policy, per-token versus sequence-level formulations, and gradient-equivalent estimator families—with recent advances in multi-reference objectives, sample-complexity guarantees, and the critical distinction between KL-as-reward and KL-as-loss implementations.

## Theoretical Foundations

### The KL-regularized objective and Boltzmann optimum

The canonical RLHF objective maximizes expected reward minus a reverse-KL penalty to a reference policy $\pi_{\text{ref}}$ (usually the SFT model) [source:huggingface:topic-objectives-and-regularization-refe]:

$$
\mathcal{J}_{\mathrm{RLHF}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot | x)} \left[ r(x, y) \right] - \beta \, \mathbb{D}_{\mathrm{KL}} \left( \pi_{\theta}(\cdot | x) \parallel \pi_{\text{ref}}(\cdot | x) \right)
$$

where $\beta > 0$ controls the regularization strength. For a fixed reward function, the pointwise optimal policy has the **Boltzmann (Gibbs) form** [source:arxiv:2508.17000; arxiv:2305.18290; arxiv:2502.01203]:

$$
\pi^{*}(y | x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y | x) \, \exp \left( \frac{1}{\beta} r(x, y) \right), \quad Z(x) = \sum_{y} \pi_{\text{ref}}(y | x) \exp \left( \frac{1}{\beta} r(x, y) \right)
$$

This closed-form solution underpins both online RL (where the policy is trained to approach $\pi^{*}$) and offline methods like DPO, which reparameterizes the reward as $r(x,y) = \beta \log \frac{\pi_{\theta}(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$ and substitutes into the Bradley–Terry preference likelihood, yielding a pure classification loss that implicitly enforces the KL constraint [source:arxiv:2305.18290]. The Boltzmann form also appears in early preference learning work [source:arxiv:1909.08593], where the KL penalty coefficient $\beta$ acts as an inverse temperature controlling the sharpness of the policy relative to the reference.

**Core problem — over-optimization.** In RLHF, powerful optimization tools can cause a model to drift too far from the strong, general "reference model" used in previous training stages, leading to **over-optimization** where the model maximizes reward at the expense of out-of-distribution performance [source:rlhfbook:rlhf-and-post-training-book-regularizati]. Manifestations include generation of nonsensical text: repeated tokens, language switching, excessive special characters, or mathematically followable reasoning that leads to incorrect answers [source:rlhfbook:rlhf-and-post-training-book-regularizati]. The KL penalty acts as a "safety tether" keeping the fine-tuned policy close to the reference, preventing the policy from performing an "adversarial search" for high-reward but qualitatively poor outputs [source:mbrenndoerfer:kl-divergence-penalty-in-rlhf-theory-imp].

### Per-token versus sequence-level KL

Early RLHF work applied the KL penalty at the **sequence level** [source:arxiv:2009.01325]:

$$
R(x, y) = r_\theta(x, y) - \beta \log \frac{\pi_\phi^{\text{RL}}(y|x)}{\pi^{\text{SFT}}(y|x)}
$$

InstructGPT and subsequent PPO-based pipelines shifted to **per-token KL-in-reward**, folding the penalty into the reward at each step so the advantage estimator sees it [source:huggingface:topic-objectives-and-regularization-refe; arxiv:2203.02155]:

$$
r_t = r_\phi(x, y) \cdot \mathbb{1}_{t=T} - \beta \log \frac{\pi_\theta(a_t | s_t)}{\pi_{\text{ref}}(a_t | s_t)}
$$

where $s_t = (x, y_{<t})$ and $a_t = y_t$. The sequence-level KL decomposes exactly as the sum of per-token KLs under the chain rule:

$$
\mathbb{D}_{\mathrm{KL}}(\pi_\theta(\cdot|x) \parallel \pi_{\text{ref}}(\cdot|x)) = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\text{ref}}(y_t | x, y_{<t})} \right]
$$

Per-token placement has become the de facto standard in PPO and GRPO because it provides denser gradient signals and integrates naturally with GAE [source:huggingface:topic-objectives-and-regularization-refe]. However, the **gradient equivalence** between KL-in-reward and KL-as-loss holds only under on-policy sampling; off-policy updates require importance-sampling corrections [source:arxiv:2510.01555].

**Practical implementation recipe** [source:rlhfbook:rlhf-and-post-training-book-regularizati; mbrenndoerfer:kl-divergence-penalty-in-rlhf-theory-imp]:
1. **Generate**: Autoregressively sample a full sequence of tokens using the RL model.
2. **Forward Pass**: Run a single pass over the sequence for both the RL model and the reference model to obtain per-token logits.
3. **Log-Probabilities**: Convert logits to log-probabilities using `log_softmax`.
4. **Gather**: Extract the log-probabilities assigned to the tokens that were actually generated.
5. **Approximate KL**: Sum the token log-probabilities to get sequence-level log-probs; the difference between the RL and reference sequence log-probs approximates the KL divergence.

This expectation-based formulation is computationally efficient for large vocabularies (50k–100k tokens) because it only requires evaluating log probabilities for tokens actually generated, rather than summing over the entire vocabulary [source:mbrenndoerfer:kl-divergence-penalty-in-rlhf-theory-imp]. The per-token KL contribution is $\text{KL}_t = \log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{\text{ref}}(y_t | x, y_{<t})$, and the sequence penalty is the sum over $t$ [source:mbrenndoerfer:kl-divergence-penalty-in-rlhf-theory-imp].

**Adaptive control.** Ziegler et al. [source:arxiv:1909.08593] used a log-space proportional controller to vary $\beta$ dynamically, targeting a specific KL value (6–10 nats). This adaptive approach adjusts the tether tightness during training—increasing $\beta$ if the policy drifts dangerously, decreasing it if progress stalls [source:mbrenndoerfer:kl-divergence-penalty-in-rlhf-theory-imp].

## The KL Estimator Zoo: $k_1$, $k_2$, $k_3$ and Gradient Equivalence

Liu et al. [source:arxiv:2510.01555] identify a critical design axis: whether the KL term is placed **in the reward** (as a coefficient multiplying the score function $\nabla_\theta \log \pi_\theta$) or **as a loss** (differentiated directly). They analyze three estimators of the per-token RKL divergence $\delta = \pi_\theta / \pi_{\text{ref}}$:

| Estimator | Formula | Role | Gradient-equivalent coefficient $c(y)$ |
|-----------|---------|------|----------------------------------------|
| $k_1$ | $-\log \delta$ | Unbiased value estimator | $c_1 = \log \delta$ (in reward) / **0** (as loss) |
| $k_2$ | $\frac{1}{2} (\log \delta)^2$ | Biased value estimator | $c_2 = \log \delta$ (as loss) |
| $k_3$ | $\delta - 1 - \log \delta$ | Unbiased, always-positive value estimator | $c_3 = 1 - \delta$ (as loss) |

**Theorem (On-policy gradient equivalence) [source:arxiv:2510.01555]:**  
Under on-policy sampling, the expected gradient of the true RKL objective is:

$$
\nabla_\theta \mathcal{J}_{\mathrm{RKL}} = \mathbb{E}_{x,y \sim \pi_\theta} \left[ \left( \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} \right) \nabla_\theta \log \pi_\theta(y|x) \right]
$$

This **target gradient** is matched exactly by:
- $k_1$ **in reward** (coefficient $c = \log \delta$)
- $k_2$ **as loss** (gradient-equivalent coefficient $c = \log \delta$)

**$k_1$ as loss is vacuous:** Its gradient is $\mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$ (zero-mean score identity), providing **no regularization signal**—only noise [source:arxiv:2510.01555]. Empirically, GRPO with $k_1$-as-loss behaves indistinguishably from a no-KL baseline and fails to constrain drift, especially at 7B scale.

**$k_3$ as loss is a biased first-order approximation:** Its gradient-equivalent coefficient is $c_{3'} = 1 - \delta$, which approximates $-\log \delta$ only near $\delta \approx 1$. The bias expands as:

$$
\mathrm{Bias}(\delta) = (-\log \delta) - (1 - \delta) = \frac{1}{2}(\delta - 1)^2 - \frac{1}{3}(\delta - 1)^3 + O((\delta-1)^4)
$$

This induces **pathological asymmetry** [source:arxiv:2510.01555]:
- **Over-coverage** ($\delta \to 0$): $1-\delta \to 1$ (weak, saturating restoring force) vs. $-\log \delta \to +\infty$ (strong pull back)
- **Under-coverage** ($\delta \to \infty$): $1-\delta \to -\infty$ (explosive push away) vs. $-\log \delta \to -\infty$ logarithmically

Moreover, $\mathrm{Var}[1-\delta] = \chi^2(\pi_{\text{ref}} \parallel \pi_\theta)$, which can be **infinite** if support conditions are violated. In a Gaussian illustration ($p \sim \mathcal{N}(0,1)$, $q \sim \mathcal{N}(0.1, 0.2)$), $k_3$ exhibited sample standard deviation **8.8** vs. $k_1$'s **0.69** [source:arxiv:2510.01555].

**Experimental verdict (GRPO, math reasoning, $\beta=0.5$) [source:arxiv:2510.01555]:**  
- $k_2$-as-loss: **strongest regularization**, lowest reward/length variance, tightest coupling to reference (smallest "Logprob Diff"), highest entropy → best exploration.  
- $k_3$-as-loss: weaker constraints, larger probability gaps, reduced entropy, greater drift/instability despite transient reward gains.  
- $k_1$-as-loss: no regularization; policy drifts further than no-KL baseline at later stages.

**Estimator properties** [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-]:
- $k_1$: Unbiased but high variance; can be negative per token.
- $k_2$: Biased but low variance; always non-negative.
- $k_3$: Unbiased, low variance, always non-negative; recommended default in GRPO.

**Implementation variants** [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-; openrlhf:openrlhf-kl-penalty-documentation]:
- **Placement**: KL can be folded into the **per-token reward** (e.g., PPO, using $k_1$), where it flows through the value function, or added directly to the **loss function** (e.g., GRPO, using $k_2$ or $k_3$), keeping the advantage signal "clean."
- **Accumulation**: Per-token penalties couple regularization to response length, potentially introducing length bias.
- **$\beta$ Control**: $\beta$ can be **fixed** or **adaptive**. Adaptive controllers adjust $\beta$ to maintain a target KL (e.g., 6–10 nats) using a proportional controller [source:arxiv:1909.08593].
- **Reference Management**: While $\pi_{\text{ref}}$ is usually frozen, some recipes (e.g., R1-Zero) refresh the reference model every 400 steps to relax constraints [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-].

**Off-policy correction [source:arxiv:2510.01555; openrlhf:openrlhf-kl-penalty-documentation]:** For any $k_n$-as-loss, compute its gradient-equivalent coefficient $k_{n'} = \frac{\partial k_n}{\partial \log \pi_\theta}$ and apply standard PPO importance sampling/clipping to $k_{n'}$ (either merged with reward or as a separate clipped head). This restores unbiasedness for $k_2$ and $k_3$ under off-policy data. OpenRLHF implements ICEPOP as a hard mask: $\text{vllm\_is} = \exp(\text{old\_log\_probs} - \text{rollout\_log\_probs})$ with clamping thresholds [0.5, 5.0] [source:openrlhf:openrlhf-kl-penalty-documentation].

## Multiple Reference Models

Standard RLHF uses a single $\pi_{\text{ref}}$ (the SFT model). **Multiple reference models** allow ensembling diverse priors (e.g., SFT + larger base model) to improve coverage and reduce overfitting [source:arxiv:2502.01203].

### Reverse KL with multiple references

Objective with weights $\alpha_i \ge 0, \sum \alpha_i = 1$:

$$
\max_\pi \mathbb{E}_{y \sim \pi}[r(x,y)] - \frac{1}{\gamma} \sum_{i=1}^K \alpha_i \, \mathrm{KL}(\pi(\cdot|x) \parallel \pi_{\text{ref},i}(\cdot|x))
$$

**Exact closed-form solution** [source:arxiv:2502.01203]:

$$
\pi^{\gamma}_{\theta^*}(y|x) = \frac{\widehat{\pi}_{\boldsymbol{\alpha},\text{ref}}(y|x)}{\widehat{Z}(x)} \exp(\gamma r_{\theta^*}(x,y))
$$

where the **geometric mixture** reference is:

$$
\widehat{\pi}_{\boldsymbol{\alpha},\text{ref}}(y|x) = \frac{\prod_{i=1}^K \pi_{\text{ref},i}(y|x)^{\alpha_i}}{F_{\boldsymbol{\alpha}}(x)}, \quad F_{\boldsymbol{\alpha}}(x) = \sum_y \prod_{i=1}^K \pi_{\text{ref},i}(y|x)^{\alpha_i}
$$

### Forward KL with multiple references

Objective with weights $\beta_i \ge 0, \sum \beta_i = 1$:

$$
\max_\pi \mathbb{E}_{y \sim \pi}[r(x,y)] - \frac{1}{\gamma} \sum_{i=1}^K \beta_i \, \mathrm{KL}(\pi_{\text{ref},i}(\cdot|x) \parallel \pi(\cdot|x))
$$

**Implicit solution** [source:arxiv:2502.01203]:

$$
\tilde{\pi}^{\gamma}_{\theta^*}(y|x) = \frac{\bar{\pi}_{\boldsymbol{\beta},\text{ref}}(y|x)}{\gamma(\tilde{Z}(x) - r_{\theta^*}(x,y))}, \quad \bar{\pi}_{\boldsymbol{\beta},\text{ref}} = \sum_i \beta_i \pi_{\text{ref},i}
$$

where $\tilde{Z}(x)$ normalizes the policy.

### Sample complexity

| Setting | Sub-optimality gap | Optimality gap |
|---------|-------------------|----------------|
| Reverse KL | $O(1/n)$ | $O(1/\sqrt{n})$ |
| Forward KL | $O(1/\sqrt{n})$ | $O(1/\sqrt{n})$ (with $\gamma = n$) |

Reverse KL enjoys **faster sub-optimality convergence** but both require bounded rewards. The bounds scale with coverage constants $C_{\boldsymbol{\alpha}, \varepsilon_{\text{rkl}}}$ and $e^{R_{\max}}$ [source:arxiv:2502.01203].

**Empirical win rates (UltraFeedback, Qwen 2.5 7B) [source:arxiv:2502.01203]:**
- Base: 8.6%
- SFT: 43.4%
- DPO (SFT ref): 56.4%
- DPO (14B ref): 59.8%
- **DPO (both refs): 66.1%** — substantial gain from multi-reference.

**Alternative regularization: pretraining gradients.** To prevent performance regressions on general NLP tasks, an additional reward can be added to maintain higher probabilities on the pretraining corpus [source:rlhfbook:rlhf-and-post-training-book-regularizati]:

$$
J(\theta) = \mathbb{E}_{(x,y) \sim D\pi_{RL,\theta}} [r_\theta(y|x) - \lambda r_{reg}] + \gamma \mathbb{E}_{x \sim D_{pretrain}} [\log(\pi_{RL,\theta}(x))]
$$

This PPO-ptx approach [source:arxiv:2203.02155] mixes PPO updates with updates that increase the log likelihood of the original pretraining distribution, mitigating the "alignment tax" [source:arxiv:2203.02155].

## Sharp Sample-Complexity Analysis for KL-Regularized Bandits/RLHF

Zhao et al. [source:arxiv:2411.04625] prove that **KL regularization can improve sample complexity from $O(1/\epsilon^2)$ to $O(1/\epsilon)$** under a **Two-Stage Mixed-Policy Sampling (TMPS)** algorithm, provided the reference policy has sufficient coverage.

### Data coverage condition

For a policy $\pi$ and reference $\pi_0$, define $D^2$ as the minimum constant satisfying:

$$
\frac{[R(\theta', x, a) - R(\theta, x, a)]^2}{\mathbb{E}_{x' \sim d_0, a' \sim \pi_0}[(R(\theta', x', a') - R(\theta, x', a'))^2]} \le D^2 \quad \forall (x,a), \theta,\theta'
$$

For RLHF with Bradley–Terry preferences, a centered version is used with a baseline function $b(x)$ [source:arxiv:2411.04625].

### TMPS algorithm (contextual bandits)

1. **Stage 1:** Collect $m$ samples from $\pi_0$, estimate reward $\widehat{\theta}_0$, compute intermediate policy $\pi_{\widehat{\theta}_0}^\eta \propto \pi_0 \exp(\eta R(\widehat{\theta}_0, \cdot))$.
2. **Stage 2:** Collect $n$ samples from $\pi_{\widehat{\theta}_0}^\eta$, re-estimate $\widehat{\theta}$ on combined data, output $\pi_{\widehat{\theta}}^\eta$.

**Sample complexity [source:arxiv:2411.04625]:**  
$m = \Theta(\eta^2 D^2 B^2 \log(N/\delta))$, $n = \Theta((\eta/\epsilon) B^2 \log(N/\delta))$ → total $\mathcal{O}(\max(\eta^2 D^2, \eta/\epsilon) \log N)$.

**Key insight:** The dependence on coverage $D^2$ is **additive** ($\eta^2 D^2 + \eta/\epsilon$), not multiplicative as in prior work. This captures the empirical benefit of KL regularization: a well-covered reference ($\pi_0$ close to optimal) reduces the $\eta^2 D^2$ term.

**Lower bound [source:arxiv:2411.04625]:** $\Omega(\min(\eta \log N / \epsilon, \log N / \epsilon^2))$ — the $\eta/\epsilon$ rate is **minimax optimal** for small $\epsilon$.

**Limitations:** The RLHF bound carries an $e^B$ factor from the logistic link; extension to MDPs is open; local-coverage conditions still yield multiplicative dependence.

## Token-Level Action-Value Perspective: KL-Regularized Q-Learning (KLQ)

KLQ [source:arxiv:2508.17000] reframes online RLHF as a **token-level KL-regularized MDP** and derives an action-value method that avoids PPO's heuristic clipping.

### Token-level KL-regularized MDP

- State $s_t = (x, y_{1:t})$, action $a_t = y_{t+1}$
- Per-step reward: $r_t = -\tau \gamma D_{\mathrm{KL}}(\pi_\theta \parallel \pi_b)(s_{t+1})$ for $t < T$, final reward $r_T = R_\phi(x,y)$
- Objective: maximize discounted KL-regularized return $G_t = \sum_{k=t}^{T-1} \gamma^{k-t} (r_{k+1} - \tau \gamma D_{\mathrm{KL}}(\pi_\theta \parallel \pi_b)(s_{k+1}))$

### Boltzmann optimal Q-function and policy

The optimal Q-function satisfies:

$$
Q^*(s,a) = \tau \log \frac{\pi^*(a|s)}{\pi_b(a|s)} + V^*(s), \quad V^*(s) = \tau \log \sum_a \pi_b(a|s) \exp(Q^*(s,a)/\tau)
$$

This **analytic correspondence** allows parameterizing $Q_\theta(s,a)$ in terms of the policy $\pi_\theta$ and a learned $V_\theta$, enabling initialization from the SFT model [source:arxiv:2508.17000].

### $\lambda$-Return regression

To handle sparse terminal rewards, KLQ uses $\lambda$-returns as regression targets:

$$
G_t^{\lambda,\alpha}[Q_\theta] = \alpha \sum_{k=t}^{T-1} (\lambda \gamma)^{k-t} \delta_k + Q_\theta(s_t, a_t)
$$

where $\delta_t = r_{t+1} + \gamma V_\theta(s_{t+1}) - Q_\theta(s_t, a_t)$ is the TD error. The loss is $\mathcal{L}(\theta) = \mathbb{E}[(Q_\theta(s,a) - \hat{G})^2]$.

### Equivalence to PPO-penalty

KLQ recovers the PPO-penalty objective with a specific $\beta$:

$$
\beta = \tau \left( \frac{1-\alpha}{\alpha} \right)
$$

**Empirical results (TL;DR, Anthropic-HH, 4×A100, 5h/run) [source:arxiv:2508.17000]:**
- KLQ matches PPO on final LM-RLHF objective value.
- **Consistently higher LLM-as-judge win-rate** vs. PPO across KL coefficients, especially at high $\beta$.
- Equal per-update compute cost.
- Limitations: preliminary experiments; no advantage whitening analogue; multi-step reasoning untested.

## Behavior-Supported Regularization: An Alternative to KL

BSPO [source:arxiv:2503.18130] argues that KL regularization penalizes **all** deviations from $\pi_{\text{ref}}$, including high-quality out-of-distribution (OOD) responses. Instead, it uses the **behavior policy** $\beta(a|s)$ (empirical next-token distribution in the reward-model training data) to define "supported" actions.

### Behavior-Supported Bellman Operator

$$
\mathcal{T}_\beta^\pi Q(s,a) = \begin{cases}
\mathcal{T}^\pi Q(s,a) & \text{if } \beta(a|s) > 0 \\
Q_{\min} & \text{otherwise}
\end{cases}
$$

where $Q_{\min} = r_{\min}/(1-\gamma)$. The fixed point $Q_\beta^\pi$ **underestimates returns for unsupported actions**, disadvantaging them without affecting in-distribution values.

### ScoreLM: Joint reward and behavior prediction

A single model predicts both reward $R(x,y;\phi)$ and behavior distribution $\beta(a_t | x, a_{<t}; \phi)$:

$$
\mathcal{L}_{\text{ScoreLM}} = -\mathbb{E}[\log \sigma(R(x,y_w) - R(x,y_l))] - \alpha \mathbb{E}[\log \beta(a_t | x, a_{<t})]
$$

### Integration with PPO

Behavior-supported V-values are predicted via a modified Bellman V-operator, and advantages $A^{\pi_{\theta_k}}$ computed from $V_{\phi_k}$ are plugged into standard PPO loss.

**Results (synthetic gold-reward setup) [source:arxiv:2503.18130]:**
- Proxy RM accuracy: **75.91%** on supported pairs vs. **58.10%** on unsupported pairs.
- BSPO achieves highest gold reward across proxy scales (774M–2.7B); baselines over-optimize.
- BSPO maintains low unsupported-token count; KL penalty fails at smaller KL distances.
- Limitations: synthetic gold reward; assumes well-trained RM; OOD prompts and multi-objective settings unaddressed.

## Implicit Regularization: RL vs. SFT

Research indicates that RL provides implicit regularization—a resistance to catastrophic forgetting—that Supervised Fine-Tuning (SFT) lacks [source:rlhfbook:rlhf-and-post-training-book-regularizati].

**Generalization Results:** In a study using the **V-IRL** visual navigation task (shifting from absolute to relative directions), RL improved OOD per-step accuracy from **80.8% to 91.8%**, while SFT caused the accuracy to collapse from **80.8% to 1.3%** [source:rlhfbook:rlhf-and-post-training-book-regularizati].

**Theoretical Mechanism (Forward vs. Reverse KL):** The difference in behavior is attributed to the direction of KL divergence optimized [source:rlhfbook:rlhf-and-post-training-book-regularizati]:
- **SFT (Forward KL):** Minimizes $KL(\pi^* \| \pi_\theta)$. This is **mode-covering**, meaning it penalizes the model for failing to assign probability to regions where the target has mass. In multimodal distributions, this forces the policy to redistribute mass away from "old" modes (prior knowledge) to cover the new target, causing forgetting.
- **RL (Reverse KL):** Minimizes $KL(\pi_\theta \| \pi^*)$. This is **mode-seeking**, meaning it only penalizes the model in regions where it actually places mass. RL can shift a new mode toward the target without disturbing existing modes, preserving prior knowledge.

**RL's Razor:** This thesis postulates that on-policy methods are inherently biased toward solutions closest to the original policy in KL divergence. Empirical data shows that forgetting is directly proportional to the KL divergence between the trained and initial policies, with a correlation of $R^2 = 0.96$ [source:rlhfbook:rlhf-and-post-training-book-regularizati].

## Historical $\beta$ Values and Placement Conventions

| Method / Paper | $\beta$ | Placement | Notes |
|----------------|---------|-----------|-------|
| Ziegler et al. 2019 [source:arxiv:1909.08593] | adaptive (target 6–10 nats) | sequence-level in reward | Log-space PI controller |
| InstructGPT [source:arxiv:2203.02155] | 0.02 | per-token in reward | Fixed $\beta$; PPO-ptx with pretraining gradients |
| DPO [source:arxiv:2305.18290] | 0.1 (0.5 for TL;DR) | implicit in loss | $\beta$ = inverse temperature |
| GRPO (DeepSeekMath) [source:arxiv:2402.03300] | 0.04 | KL-as-loss ($k_3$) | Unbiased, always-positive estimator |
| DeepSeek-R1 [source:arxiv:2501.12948] | 0.001 | in loss | Verifier reward → much smaller $\beta$ |
| OpenRLHF defaults [source:openrlhf:openrlhf-kl-penalty-documentation] | 0.01 (initial) | reward penalty ($k_1$) or loss ($k_2$, $k_3$) | Clip range $\epsilon=0.2$; IS correction thresholds [0.5, 5.0] |
| Preference RLHF (general) [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-] | 0.02–0.04 | per-token reward or loss | Higher $\beta$ to prevent drift |
| Verifiable-Reward RL (RLVR) [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-] | 0.001 or 0 | in loss | $\beta=0$ in DAPO, Open-Reasoner-Zero |

**Open question [source:huggingface:topic-objectives-and-regularization-refe]:** The cross-recipe trend (verifier rewards → smaller $\beta$) is an inference, not a proven result. Whether reverse KL is the optimal divergence (vs. forward KL, $f$-divergences) remains open.

## Current Status and Trajectory

**Rising:**  
- **Gradient-aware KL implementation** ($k_2$-as-loss, off-policy corrected $k_3$) is gaining traction as the community recognizes that "unbiased value estimation $\neq$ good gradients" [source:arxiv:2510.01555]. GRPO's $k_3$-as-loss is being re-evaluated; $k_2$-as-loss shows superior stability/regularization in controlled experiments.  
- **Multi-reference RLHF** moves from theory to practice: exact closed-form solutions for reverse/forward KL [source:arxiv:2502.01203] enable principled ensembling of diverse priors (SFT + base models), with demonstrated win-rate gains.  
- **Token-level action-value methods** (KLQ [source:arxiv:2508.17000]) provide a theoretically grounded alternative to PPO, matching performance with cleaner foundations and better judge win-rates.  
- **Sharp sample-complexity theory** [source:arxiv:2411.04625] establishes $O(1/\epsilon)$ rates under coverage conditions, validating the empirical benefit of KL regularization.  
- **Behavior-supported regularization** (BSPO [source:arxiv:2503.18130]) challenges the universality of KL by targeting only actions outside the reward-model training distribution, showing promise in synthetic settings.

**Default:**  
- Per-token KL-in-reward (PPO) and KL-as-loss $k_3$ (GRPO) remain workhorses in production pipelines.  
- DPO's implicit KL (fixed $\beta$) dominates offline preference optimization.  
- Adaptive $\beta$ controllers (Ziegler-style) are less common; fixed $\beta$ tuned per reward type is standard.  
- OpenRLHF provides a unified framework supporting PPO, REINFORCE++, RLOO, GRPO/Dr.GRPO with flexible KL placement (reward penalty via $k_1$ or loss term via $k_2$/$k_3$) and off-policy corrections (TIS, ICEPOP, Seq-Mask-TIS) [source:openrlhf:openrlhf-kl-penalty-documentation].

**Fading / Unresolved:**  
- Sequence-level KL penalty (superseded by per-token).  
- Naïve $k_1$-as-loss (proven vacuous [source:arxiv:2510.01555]).  
- Reference-free variants (SimPO/ORPO) — not yet processed in the corpus; open question how much anchor benefit survives without $\pi_{\text{ref}}$ [source:huggingface:topic-objectives-and-regularization-refe].  
- Forward KL and general $f$-divergences — theoretical solutions exist [source:arxiv:2502.01203] but empirical adoption is minimal.  
- Many regularization techniques emerge in literature to stabilize experimental setups but often disappear in subsequent model iterations as setups are simplified; the core KL distance remains the most stable and popular variant [source:rlhfbook:rlhf-and-post-training-book-regularizati].

**Hedge:** The shift from $k_3$ to $k_2$ in GRPO-style pipelines is **not widely reported** in large-scale deployments; most public GRPO implementations still use $k_3$. Multi-reference RLHF has **not been demonstrated at frontier-model scale** (experiments at 7B). KLQ's advantage over PPO on reasoning tasks is **untested**. BSPO assumes the behavior policy $\beta$ is perfectly estimated from RM training data; in practice $\beta$ is learned (ScoreLM) and errors could misclassify ID/OOD actions [source:arxiv:2503.18130].

## Disagreements and Open Tensions

| Issue | Source A | Source B | Tension |
|-------|----------|----------|---------|
| **$k_3$ estimator quality** | John (2020) cited in [source:arxiv:2510.01555] claims $k_3$ is "strictly better" | Liu et al. [source:arxiv:2510.01555] show $k_3$-as-loss has **pathological asymmetry, bias, and potentially infinite variance**; $k_2$-as-loss dominates empirically | The "strictly better" claim holds for **value estimation** (unbiased, positive) but **fails for gradient optimization** — the two objectives diverge. |
| **KL vs. behavior support** | Standard RLHF: KL to SFT reference is universal anchor [source:huggingface:topic-objectives-and-regularization-refe] | BSPO [source:arxiv:2503.18130]: KL penalizes *good* OOD responses; behavior-supported Bellman operator targets only *unsupported* actions | KL is a **blunt instrument**; behavior support is **data-dependent** but requires accurate $\beta$ estimation and assumes RM training data covers desired behaviors. |
| **Reverse vs. forward KL** | Reverse KL (mode-seeking) is standard; exact multi-ref solution exists [source:arxiv:2502.01203] | Forward KL (mean-seeking) has implicit multi-ref solution; sample complexity $O(1/\sqrt{n})$ for both gaps | Reverse KL converges faster in sub-optimality ($O(1/n)$) but is **mode-seeking** — may collapse diversity. Forward KL covers more modes but slower. **No empirical comparison at scale.** |
| **$\beta$ scaling with reward trustworthiness** | InstructGPT: $\beta=0.02$ (preference RM) [source:arxiv:2203.02155] | DeepSeek-R1: $\beta=0.001$ (verifier reward) [source:arxiv:2501.12948] | **Cross-recipe inference**, not controlled experiment. Verifier rewards are sparser but accurate; preference RMs are dense but noisy. Optimal $\beta$ likely depends on reward noise structure, not just "trustworthiness." |
| **Per-token vs. sequence KL gradient equivalence** | Per-token KL-in-reward is standard [source:huggingface:topic-objectives-and-regularization-refe] | Liu et al. [source:arxiv:2510.01555]: equivalence to KL-as-loss holds **only on-policy**; off-policy requires IS correction | Most PPO/GRPO pipelines are **effectively off-policy** (replay buffers, multiple epochs). The practical impact of ignoring IS for KL term is **not quantified**. |
| **Alignment tax recovery** | Increasing $\beta$ (even up to 2.0) does not recover capability loss; only mixing pretraining gradients (PPO-ptx) addresses this [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-] | InstructGPT [source:arxiv:2203.02155] uses PPO-ptx with pretraining gradients to mitigate alignment tax on public NLP benchmarks | Consistent: KL penalty alone cannot recover base capabilities; pretraining gradient mixing is necessary. |
| **Reasoning interference** | Reference-KL can be counterproductive in verifiable-reward reasoning; may suppress exploration necessary for long-CoT discovery [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-] | DeepSeek-R1 uses $\beta=0.001$ with verifier rewards [source:arxiv:2501.12948]; some recipes drop $\beta$ entirely | Low $\beta$ or no KL is viable for verifiable rewards (hard to hack), but dangerous for learned reward models [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-]. |

## Key Takeaways

- **KL regularization is the primary guardrail against reward over-optimization**, implementing a reverse-KL penalty to a frozen reference (usually SFT) that yields a Boltzmann optimal policy $\pi^* \propto \pi_{\text{ref}} \exp(r/\beta)$.
- **Per-token KL placement** (in reward or loss) is now standard; it decomposes exactly to sequence KL and provides denser gradients.
- **Not all KL estimators are equal for optimization**: $k_1$-as-loss gives zero expected gradient (vacuous); $k_3$-as-loss is a biased first-order approximation with pathological asymmetry and unstable variance; **$k_2$-as-loss (or $k_1$-in-reward) matches the true RKL gradient on-policy**.
- **Off-policy corrections are necessary** for KL-as-loss methods: convert to gradient-equivalent coefficient and apply PPO-style clipping/IS.
- **Multiple reference models** admit exact closed-form solutions (geometric mixture for reverse KL, arithmetic for forward KL) and improve win-rates by combining diverse priors.
- **Sample complexity theory** confirms KL regularization can achieve $O(1/\epsilon)$ rates under coverage conditions, with additive (not multiplicative) dependence on coverage.
- **KLQ** provides a token-level action-value alternative to PPO with theoretical grounding, matching performance and exceeding judge win-rates.
- **Behavior-supported regularization** (BSPO) challenges the universality of KL by targeting only actions outside the reward-model training distribution, showing promise in synthetic settings.
- **$\beta$ values vary widely** (0.001–0.5) across recipes; the relationship to reward type (verifier vs. preference) is observed but not rigorously established.
- **RL provides implicit regularization** (resistance to catastrophic forgetting) that SFT lacks, due to reverse KL's mode-seeking vs. forward KL's mode-covering behavior [source:rlhfbook:rlhf-and-post-training-book-regularizati].
- **Pretraining gradient mixing (PPO-ptx)** is necessary to mitigate the alignment tax on general NLP benchmarks; KL penalty alone cannot recover capability loss [source:arxiv:2203.02155; huggingface:rl-llm-wiki-kl-regularization-knowledge-].

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md) — PPO's KL-in-reward implementation and clipping mechanics
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md) — DPO's implicit KL regularization via reward reparameterization
- [GRPO (Group Relative Policy Optimization)](grpo.md) — GRPO's $k_3$-as-loss KL estimator and group normalization
- [Reward modeling for LLMs](reward-modeling.md) — Reward model training and its interaction with KL regularization
- [RL for reasoning models](rl-for-reasoning.md) — KL regularization in long-CoT and verifiable-reward settings
- [Policy gradient methods for LLMs](policy-gradient-methods.md) — Score-function estimators and variance reduction
- [MDP formulation of LLM generation](mdp-formulation.md) — Token-level MDP view underlying KLQ and per-token KL
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md) — End-to-end pipeline including KL coefficient scheduling
- [DPO variants deep-dive](dpo-variants.md) — Extensions of DPO's implicit KL (IPO, KTO, etc.)
- [RLAIF (RL from AI feedback)](rlaif.md) — KL regularization when reward model is an LLM
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md) — KL-constrained sampling without policy updates
- [Nash and game-theoretic preference optimization](nash-and-game-theoretic-po.md) — Multi-objective KL regularization
- [Self-improvement and self-play RL](self-improvement-and-self-play.md) — Evolving reference models during training
- [Process vs outcome reward models](process-vs-outcome-rewards.md) — Per-step vs. terminal rewards and KL placement
- [Reward hacking in RLHF](reward-hacking.md) — Failure modes KL regularization aims to prevent
- [Reward model over-optimization](reward-model-overoptimization.md) — Empirical characterization of over-optimization
- [Verifiable rewards (RLVR)](verifiable-rewards.md) — Low-$\beta$ regimes with ground-truth rewards
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md) — KL's implicit entropy bonus and mode-seeking behavior
- [Length and format bias](length-and-format-bias.md) — KL's interaction with length penalties
- [The alignment tax](alignment-tax.md) — KL as the mechanism trading off reward vs. base capabilities
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md) — Phenomenology KL regularization addresses
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md) — Behavioral pathologies linked to weak KL
- [LLM-as-judge](llm-as-judge.md) — Evaluation methodology for KL-regularized policies
- [Alignment and win-rate evals](alignment-and-winrate-evals.md) — Benchmarks measuring KL-regularization effectiveness
- [Judging bias and contamination](judging-bias-and-contamination.md) — Evaluation artifacts in KL-constrained RLHF
- [Distributed RL training for LLMs](distributed-rl-training.md) — Systems considerations for KL-regularized PPO/GRPO
- [Async and off-policy RL](async-and-off-policy-rl.md) — Off-policy corrections for KL terms
- [Rollout generation infrastructure](rollout-generation-infra.md) — Reference model serving for per-token KL computation
- [RL for math and code](rl-for-math-and-code.md) — KL regularization in verifiable-reward domains
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md) — KL in multi-turn tool-use trajectories
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md) — KL-regularized policy as proposer for search

## Open Questions

- **Off-policy KL gradient bias in practice:** Most GRPO/PPO pipelines run multiple epochs on buffered data (effectively off-policy). The magnitude of gradient bias from ignoring importance sampling on the KL term ($k_2$ or $k_3$ as loss) is **not quantified at scale**. Does the proposed IS correction [source:arxiv:2510.01555] materially improve stability?
- **Multi-reference RLHF at frontier scale:** The 66% win-rate gain at 7B [source:arxiv:2502.01203] is compelling, but **compute/memory cost of evaluating $K$ reference models per token** and the optimal weighting scheme $\alpha_i$ for heterogeneous priors (SFT + base + domain-adapted) are unexplored at 70B+.
- **Forward vs. reverse KL for diversity:** Reverse KL is mode-seeking; forward KL is mean-seeking. Theory gives exact solutions for both [source:arxiv:2502.01203], but **no large-scale empirical comparison** on diversity metrics (distinct-n, entropy, semantic coverage) exists. Is a mixture optimal?
- **Behavior-supported regularization with learned $\beta$:** BSPO [source:arxiv:2503.18130] assumes the behavior policy $\beta$ is perfectly estimated from RM training data. In practice, $\beta$ is learned (ScoreLM) and **errors in $\beta$ could misclassify ID/OOD actions**. How robust is BSPO to $\beta$ estimation error, and can it be combined with KL (hybrid anchor)?
- **$\beta$ scheduling for verifiable rewards:** DeepSeek-R1 uses $\beta=0.001$ with verifier rewards [source:arxiv:2501.12948], but **no systematic study** exists on how $\beta$ should scale with reward sparsity, noise, and verifiability. Is there a theoretical principle linking $\beta$ to reward signal-to-noise ratio?
- **Reference-free regularization limits:** SimPO/ORPO drop $\pi_{\text{ref}}$ entirely [source:huggingface:topic-objectives-and-regularization-refe]. **How much of KL's anti-collapse benefit comes from the reference anchor vs. the entropy bonus?** Can entropy regularization alone suffice when the reward is verifiable?
- **Implicit regularization in RL vs SFT:** The $R^2=0.96$ correlation between forgetting and KL divergence [source:rlhfbook:rlhf-and-post-training-book-regularizati] suggests a strong theoretical link. Can this be formalized into a generalization bound for RLHF?
- **KL penalty vs. pretraining gradients for alignment tax:** PPO-ptx adds pretraining gradients to recover base capabilities [source:arxiv:2203.02155]. Is there a unified framework that interpolates between KL regularization and pretraining gradient mixing?

## References
- [source:arxiv:2508.17000] [KL-Regularised Q-Learning: A Token-level Action-Value perspective on Online RLHF](https://arxiv.org/html/2508.17000v1)
- [source:arxiv:2009.01325] [Simple, Scalable, and Effective Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2009.01325)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2502.01203] [KL-Regularized RLHF with Multiple Reference Models: Exact Solutions and Sample Complexity](https://arxiv.org/abs/2502.01203)
- [source:arxiv:2503.18130] [Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization](https://arxiv.org/html/2503.18130v1)
- [source:huggingface:topic-objectives-and-regularization-refe] [topic: objectives-and-regularization/reference-model-and-kl](https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/commit/a03840159b93b042b5460ac2233c7acf82ca7842)
- [source:arxiv:2510.01555] [Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization](https://arxiv.org/abs/2510.01555)
- [source:arxiv:2411.04625] [Sharp Analysis for KL-Regularized Contextual Bandits and RLHF](https://arxiv.org/abs/2411.04625)
- [source:rlhfbook:rlhf-and-post-training-book-regularizati] [RLHF and Post-Training Book: Regularization](https://rlhfbook.com/c/15-regularization)
- [source:mbrenndoerfer:kl-divergence-penalty-in-rlhf-theory-imp] [KL Divergence Penalty in RLHF: Theory & Implementation](https://mbrenndoerfer.com/writing/kl-divergence-penalty-rlhf-training)
- [source:arxiv:2203.02155] [InstructGPT: Training language models to follow instructions with human feedback (Ouyang et al. 2022)](https://arxiv.org/abs/2203.02155)
- [source:huggingface:rl-llm-wiki-kl-regularization-knowledge-] [RL-LLM Wiki: KL Regularization Knowledge Base](https://huggingface.co/datasets/rl-llm-wiki/knowledge-base/blob/main/topics/foundations/kl-regularization.md)
- [source:arxiv:1909.08593] [The Boltzmann Optimum in RLHF (Ziegler et al. 2019)](https://arxiv.org/abs/1909.08593)
- [source:openrlhf:openrlhf-kl-penalty-documentation] [OpenRLHF: KL Penalty Documentation](https://openrlhf.readthedocs.io/en/latest/agent_training.html)
