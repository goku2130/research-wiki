---
title: PPO for LLM fine-tuning (RLHF)
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2403.19279
- arxiv:1707.06347
- arxiv:2310.00212
- arxiv:1909.08593
- arxiv:2009.01325
- arxiv:1706.03741
- arxiv:2307.04964
- arxiv:2204.05862
- arxiv:2203.02155
- cameronrwolfe:proximal-policy-optimization-ppo-the-key
- spinningup:spinning-up-in-deep-rl-ppo-documentation
- bair:rethinking-the-role-of-ppo-in-rlhf-bair-
open_questions:
- What is the optimal scaling law for KL coefficient $\beta$ with respect to model
  size, reward magnitude, and sequence length? No consensus exists across [source:arxiv:2204.05862]
  ($\beta=0.001$ for 52B), [source:arxiv:1909.08593] (target KL 0.1–1.0 nats), [source:arxiv:2307.04964]
  (fixed $\eta$, unspecified), [source:arxiv:2203.02155] (unspecified).
- Does per-token GAE with $\lambda<1$ materially improve credit assignment over sequence-level
  advantage $r(x,y)-V(x)$ in LLM bandits? [source:arxiv:2307.04964] uses $\lambda=0.9$;
  [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] uses $\lambda=0.97$;
  [source:arxiv:2204.05862] uses sequence-level; [source:arxiv:2203.02155] uses bandit
  formulation.
- Is the entropy bonus $c_2 S[\pi_\theta]$ beneficial or harmful in LLM PPO? [source:arxiv:1707.06347]
  includes it; [source:arxiv:2307.04964] omits it; [source:arxiv:2204.05862] notes
  RLHF decreases entropy; [source:spinningup:spinning-up-in-deep-rl-ppo-documentation]
  omits it.
- Can P3O's pairwise formulation (invariant to reward translation) be combined with
  PPO-ptx's pretraining gradient mixing? [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-]
  and [source:arxiv:2203.02155] address different stability issues.
---

Proximal Policy Optimization (PPO) is the dominant reinforcement learning algorithm for aligning large language models via RLHF, combining a clipped surrogate objective with a KL penalty to stabilize policy updates against a learned reward model. This article provides a rigorous technical synthesis of the policy-gradient foundations, reward model integration, KL regularization mechanics, clipping behavior, and practical modifications that distinguish LLM PPO from its original continuous-control formulation.

## Policy-gradient foundations for LLM fine-tuning

The policy gradient theorem gives the gradient of the expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$ as $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t,a_t)]$ [source:arxiv:1707.06347]. In the LLM RLHF setting, the trajectory $\tau$ is a generated response $y$ conditioned on a prompt $x$, the action space is the vocabulary at each token position, and the reward is sparse — typically a single scalar $r(x,y)$ from a reward model assigned at sequence end [source:arxiv:2009.01325]. This reduces the problem to a contextual bandit: the advantage $A(x,y)$ collapses to $r(x,y) - V(x)$ where $V(x)$ is a learned value function (critic) estimating the expected reward for prompt $x$ [source:arxiv:2307.04964]. The vanilla policy gradient objective $L^{PG}(\theta) = \hat{\mathbb{E}}_t[\log \pi_\theta(a_t|s_t) \hat{A}_t]$ [source:arxiv:1707.06347] becomes, for a full sequence $y = (y_1,\dots,y_T)$,

$$
L^{PG}(\theta) = \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(\cdot|x)}\left[ \sum_{t=1}^T \log \pi_\theta(y_t|x,y_{<t}) \cdot \hat{A}_t \right],
$$

where $\hat{A}_t$ is estimated via Generalized Advantage Estimation (GAE) [source:arxiv:1707.06347; arxiv:2307.04964]. GAE computes $\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$; in the bandit limit $\gamma=1$, $\lambda=1$, and $r_t=0$ for $t<T$, this simplifies to $\hat{A}_t = r(x,y) - V(x)$ for all $t$ [source:arxiv:2307.04964]. The value function $V_\phi(x)$ is trained with a squared-error loss $L^{VF} = (V_\phi(x) - r(x,y))^2$ [source:arxiv:1707.06347; arxiv:2307.04964].

**MDP formulation for LLMs:** [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] maps the RL framework explicitly to the LLM context: the policy $\pi_\theta$ is the LLM itself; the initial state is the input prompt; each action $a_t$ is an individual token predicted by the LLM; the state $s_t$ is the running completion (prompt plus all tokens generated up to time $t$); the trajectory is the entire generated completion from prompt to stop token (e.g., `<eos>`); and the reward is a scalar value provided by a reward model or verifier. This formulation uses a discount factor $\gamma$ to prioritize immediate rewards over distant ones, though in the bandit limit $\gamma=1$.

**Disagreement on GAE necessity:** The original PPO paper [source:arxiv:1707.06347] advocates GAE for variance reduction in multi-step MDPs. For LLM bandits, [source:arxiv:2307.04964] notes GAE reduces to a single advantage per sequence, yet many implementations retain per-token GAE with $\lambda<1$ to propagate the terminal reward backward through tokens, effectively doing credit assignment across positions. [source:arxiv:2204.05862] uses a simpler formulation: the total reward $r_{PM}(y|x)$ is assigned at sequence end and the KL penalty is applied per-token (see below), without explicit per-token advantage decomposition. [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] specifies GAE with $\lambda=0.97$ and $\gamma=0.99$ as default hyperparameters, and the Spinning Up implementation uses GAE for advantage estimation even in discrete action spaces. The practical impact of per-token vs. sequence-level advantage estimation on LLM alignment quality is not widely reported.

**Spinning Up hyperparameters:** [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] reports typical hyperparameters for PPO implementations: clip ratio $\epsilon \in [0.1, 0.3]$ (default 0.2); target KL 0.01 or 0.05 (used for early stopping); discount factor $\gamma=0.99$; GAE lambda $\lambda=0.97$; policy learning rate $3\times10^{-4}$, value function learning rate $10^{-3}$; 50 epochs of interaction with maximum 80 gradient descent steps for both policy and value function per epoch.

## Reward model integration

The reward model $r_\psi(x,y)$ is trained on pairwise human preferences using the Bradley-Terry loss [source:arxiv:1706.03741; arxiv:1909.08593; arxiv:2009.01325; arxiv:2204.05862; arxiv:2307.04964]:

$$
\mathcal{L}_{RM}(\psi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma\bigl(r_\psi(x,y_w) - r_\psi(x,y_l)\bigr) \right],
$$

where $y_w$ is preferred over $y_l$. The RM is typically initialized from the SFT model with a scalar head [source:arxiv:2009.01325; arxiv:2307.04964]. During PPO, the RM score becomes the environment reward. Critical practical details:

- **Reward normalization:** [source:arxiv:1706.03741] normalizes RM outputs to zero mean and constant std per batch. [source:arxiv:2307.04964] adds clipping: $\tilde{r} = \text{clip}\bigl((r - \mu)/\sigma, -\delta, \delta\bigr)$.
- **Reward model overoptimization:** As the policy shifts, the fixed RM becomes inaccurate off-distribution, leading to reward hacking [source:arxiv:2403.19279]. [source:arxiv:2204.05862] observes train/test RM score divergence after ~150k samples, indicating overfitting. [source:arxiv:2403.19279] proposes Reward Learning on Policy (RLP) to retrain the RM on policy-generated samples using synthetic preferences or multi-view learning.
- **Value function initialization:** [source:arxiv:2307.04964] initializes the critic $V_\phi$ from the RM parameters (minus the final head), which accelerates convergence. [source:arxiv:2009.01325] uses a separate Transformer for the value function initialized from the RM.

**InstructGPT RM training:** [source:arxiv:2203.02155] trains the reward model on all $\binom{K}{2}$ comparisons from a single ranking task as a single batch element to prevent overfitting and increase efficiency. The RM loss is:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

**Relative vs. absolute feedback discrepancy:** [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] identifies a fundamental discrepancy: reward models are trained using **relative feedback** (comparisons between response pairs), yet the RL fine-tuning stage (typically PPO) optimizes for a **single, absolute reward**. This inconsistency makes the RL process sensitive to "reward translation"—where shifting a reward function by a constant $\delta(x)$ does not change the preference loss but can significantly mislead an RL algorithm. The policy may be disrupted by the reward scale of a prompt rather than learning the actual relative preferences.

**Disagreement on reward scaling:** [source:arxiv:1706.03741] normalizes rewards to unit variance per batch. [source:arxiv:2307.04964] uses a running mean/std with clipping. [source:arxiv:2204.05862] does not explicitly describe reward normalization in the PPO phase, only that the PM score is used directly with a KL penalty. [source:arxiv:2203.02155] does not specify reward normalization in the PPO phase. The sensitivity of PPO to reward scaling in LLMs is not widely reported but is known to be high in continuous control [source:arxiv:1707.06347]. [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] argues this sensitivity is precisely the problem: PPO is not invariant to reward translation, unlike the Bradley-Terry loss used to train the RM.

## KL penalty: theory and practice

The KL penalty prevents the RL policy $\pi_\theta$ from drifting too far from the reference policy $\pi^{\text{ref}}$ (usually the SFT model), preserving language quality and preventing reward hacking. The modified reward is

$$
r_{\text{total}}(x,y) = r_\psi(x,y) - \beta \, D_{\text{KL}}\bigl(\pi_\theta(\cdot|x) \,\|\, \pi^{\text{ref}}(\cdot|x)\bigr)
$$

[source:arxiv:1909.08593; arxiv:2009.01325; arxiv:2204.05862; arxiv:2307.04964]. Two forms of KL application exist:

1. **Sequence-level KL:** $D_{\text{KL}}(\pi_\theta(y|x) \| \pi^{\text{ref}}(y|x)) = \sum_t \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi^{\text{ref}}(y_t|x,y_{<t})}$ subtracted once per sequence [source:arxiv:1909.08593; arxiv:2009.01325].
2. **Token-level KL:** $-\beta \sum_t \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi^{\text{ref}}(y_t|x,y_{<t})}$ added to the per-token reward, so the advantage at each step includes the KL term [source:arxiv:2307.04964; arxiv:2204.05862].

These are mathematically equivalent for the total return but differ in advantage estimation: token-level KL distributes the penalty across time steps, affecting GAE. [source:arxiv:2307.04964] uses token-level KL in the total reward $r_{\text{total}}(x,y_i) = r(x,y_i) - \eta \text{KL}(\pi_\theta^{\text{RL}}(y_i|x), \pi^{\text{SFT}}(y_i|x))$ per token $y_i$.

**InstructGPT PPO-ptx objective:** [source:arxiv:2203.02155] formulates the PPO-ptx objective as:

$$
\text{objective} (\phi) = \mathbb{E} _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\frac{\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x)}{\pi^ {\mathrm{SFT}} (y \mid x)}\right) \right] + \gamma \mathbb{E} _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]
$$

where $\beta$ is the KL reward coefficient and $\gamma$ is the pretraining loss coefficient. This mixes PPO updates with updates that increase the log likelihood of the original pretraining distribution to mitigate the alignment tax.

**Adaptive KL control:** [source:arxiv:1707.06347] proposes an adaptive $\beta$ for the penalty version (PPO-Penalty): if $d < d_{\text{targ}}/1.5$, $\beta \leftarrow \beta/2$; if $d > 1.5 d_{\text{targ}}$, $\beta \leftarrow 2\beta$, where $d = \hat{\mathbb{E}}_t[\text{KL}]$. [source:arxiv:1909.08593] uses a log-space proportional controller:

$$
e_t = \text{clip}\left(\frac{\text{KL}(\pi_t,\rho) - \text{KL}_{\text{target}}}{\text{KL}_{\text{target}}}, -0.2, 0.2\right), \quad \beta_{t+1} = \beta_t \exp(K_\beta e_t), \quad K_\beta=0.1.
$$

[source:arxiv:2204.05862] uses a fixed $\lambda_{\text{KL}} = 0.001$ (they report $\beta \geq 0$ typically $0.001$). [source:arxiv:2307.04964] uses a fixed $\eta$ (KL reward coefficient) and monitors KL divergence as a stability metric. [source:arxiv:2009.01325] uses a fixed $\beta$. [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] employs **early stopping** based on KL divergence: gradient steps halt if the mean KL-divergence exceeds the `target_kl` threshold (typically 0.01 or 0.05), as clipping alone may not fully prevent the policy from moving too far from the old policy.

**Disagreement on KL coefficient magnitude:** [source:arxiv:2204.05862] uses $\beta=0.001$ for 52B models. [source:arxiv:1909.08593] targets $\text{KL}_{\text{target}}$ around 0.1–1.0 nats (implied by their controller). [source:arxiv:2307.04964] does not specify $\eta$ numerically but notes monitoring KL is crucial. [source:arxiv:2203.02155] does not specify $\beta$ numerically. The optimal $\beta$ scales with model size and reward magnitude; no consensus scaling law is reported. [source:arxiv:1707.06347] found adaptive KL penalty performed worse than clipping on MuJoCo (Table 1: adaptive KL $d_{\text{targ}}=0.01$ scored 0.74 vs clipping $\epsilon=0.2$ scored 0.82), but this was for continuous control without a separate reward model.

**KL-reward tradeoff:** [source:arxiv:2204.05862] observes an approximately linear relationship between $\sqrt{D_{\text{KL}}(\pi\|\pi_0)}$ and PM reward during training (Figures 4, 13). [source:arxiv:2310.00212] shows P3O achieves a better KL-reward frontier than PPO and DPO: on HH, P3O-V1/V2 dominate PPO and DPO respectively, delivering 0.1–0.3 higher reward at same KL. [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] notes that excessive deviation from the reference policy (high KL divergence) can lead the online policy to "cut corners" of the reward model, resulting in incoherent continuations. While P3O provides better KL control than DPO, the balance between reward maximization and maintaining the integrity of the reference policy remains a critical constraint.

## PPO clipping mechanism

The clipped surrogate objective replaces the hard KL constraint of TRPO with a pessimistic bound on the policy ratio $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$ [source:arxiv:1707.06347]:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\Bigl( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \Bigr) \right].
$$

For $\hat{A}_t > 0$, the objective increases with $r_t$ until $1+\epsilon$, then flattens; for $\hat{A}_t < 0$, it decreases until $1-\epsilon$, then flattens. This prevents destructive large updates when optimizing multiple epochs on the same batch [source:arxiv:1707.06347]. In LLMs, the ratio is computed per token:

$$
r_t(\theta) = \frac{\pi_\theta(y_t|x,y_{<t})}{\pi_{\theta_{\text{old}}}(y_t|x,y_{<t})},
$$

and the objective sums over tokens [source:arxiv:2307.04964]. The clipping threshold $\epsilon$ is typically 0.2 [source:arxiv:1707.06347; arxiv:2307.04964]. [source:arxiv:1707.06347] Table 1 shows $\epsilon=0.2$ outperforms 0.1 (0.82 vs 0.76) and 0.3 (0.70) on MuJoCo.

**Spinning Up PPO-Clip details:** [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] provides the precise clipped objective formulation used by OpenAI. The objective function $L$ is:

$$
L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a) \right)
$$

A simplified implementation version uses:

$$
L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a), g(\epsilon, A^{\pi_{\theta_k}}(s,a)) \right)
$$

with $g(\epsilon, A)$ defined as:

$$
g(\epsilon, A) = \begin{cases} (1 + \epsilon) A & A \geq 0 \\ (1 - \epsilon) A & A < 0 \end{cases}
$$

This ensures that if advantage $A$ is positive, the objective hits a ceiling once $\pi_{\theta}(a|s) > (1+\epsilon) \pi_{\theta_k}(a|s)$; if negative, it hits a ceiling once $\pi_{\theta}(a|s) < (1-\epsilon) \pi_{\theta_k}(a|s)$.

**Value function clipping:** [source:arxiv:1707.06347] and [source:arxiv:2307.04964] clip the value function loss: $L^{VF} = \max\bigl( (V_\phi - V^{\text{targ}})^2, (\text{clip}(V_\phi, V_{\text{old}}-\epsilon, V_{\text{old}}+\epsilon) - V^{\text{targ}})^2 \bigr)$. This stabilizes critic training.

**Full PPO objective for LLMs:**

$$
L(\theta) = \hat{\mathbb{E}}_{x,y} \left[ \sum_t L_t^{\text{CLIP}}(\theta) - c_1 L_t^{VF}(\theta) + c_2 S[\pi_\theta](x,y_{<t}) \right]
$$

[source:arxiv:1707.06347; arxiv:2307.04964], where $S$ is an entropy bonus. [source:arxiv:2307.04964] adds a pretraining gradient term (PPO-ptx): $\lambda_{\text{ptx}} \mathbb{E}_{x\sim\mathcal{D}_{\text{pretrain}}}[\log \pi_\theta(x)]$ to mitigate alignment tax.

**P3O clipping variants:** [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] introduces two implementation styles for clipping in Pairwise PPO: **V1 (separate clipping)** and **V2 (joint clipping)**. Both clip the importance sampling ratio and the gradient updates to prevent excessively large updates and better manage the trade-off between KL divergence and reward.

**Disagreement on clipping vs. penalty for LLMs:** The original PPO paper [source:arxiv:1707.06347] found clipping superior to adaptive KL penalty on continuous control. However, all major LLM RLHF papers [source:arxiv:1909.08593; arxiv:2009.01325; arxiv:2204.05862; arxiv:2307.04964] use **both** clipping and a KL penalty simultaneously. The KL penalty acts on the divergence from the *reference* (SFT) model, while clipping constrains the step from the *old* (current iteration) policy. These serve different purposes: clipping ensures per-update stability; KL penalty anchors to the initial supervised model. [source:arxiv:2310.00212] argues PPO's token-wise clipping is mismatched to trajectory-wise reward models (BTL is invariant to constant reward shifts, but PPO is not), motivating their pairwise P3O algorithm.

## PPO variants and practical modifications for LLMs

| Modification | Description | Sources |
|--------------|-------------|---------|
| **PPO-ptx** | Adds pretraining loss $\lambda_{\text{ptx}} \mathbb{E}[\log \pi_\theta(x)]$ to preserve pretrained capabilities | [source:arxiv:2307.04964; arxiv:2203.02155] |
| **Reward normalization & clipping** | Running mean/std + clip to $[-\delta,\delta]$ | [source:arxiv:2307.04964] |
| **Critic initialization from RM** | Initialize $V_\phi$ from RM backbone | [source:arxiv:2307.04964] |
| **Small experience buffer** | Use small batch for environment sampling (e.g., 128) and smaller minibatch for updates (e.g., 32) | [source:arxiv:2307.04964] |
| **Global gradient clipping** | Clip global norm (e.g., 1.0) | [source:arxiv:2307.04964] |
| **Token-level KL in reward** | Distribute KL penalty per token for GAE | [source:arxiv:2307.04964; arxiv:2204.05862] |
| **Adaptive KL controller** | Proportional control on KL target | [source:arxiv:1909.08593] |
| **Early stopping on KL** | Halt gradient steps if mean KL exceeds target_kl | [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] |
| **Reference model = SFT model** | Fixed $\pi^{\text{ref}}$ throughout training | [source:arxiv:1909.08593; arxiv:2009.01325; arxiv:2204.05862] |
| **Multiple PPO epochs per rollout** | Typically 1–4 epochs (K in original PPO) | [source:arxiv:1707.06347; arxiv:2307.04964] |
| **Nucleus sampling for rollouts** | $p=0.9, \tau=0.8$, repetition penalty 1.1 | [source:arxiv:2307.04964] |
| **Pairwise PPO (P3O)** | Optimizes relative preferences directly; invariant to reward translation | [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] |

**Hyperparameters reported in [source:arxiv:2307.04964] (7B/13B models):**
- Policy LR: $5\times10^{-7}$, Critic LR: $1.65\times10^{-6}$, warmup 10%
- Batch size (env): 128, minibatch: 32
- GAE $\lambda=0.9$, $\gamma=1$ (bandit)
- $\epsilon=0.2$, $c_1=0.5$, $c_2=0$ (entropy bonus often omitted)
- Max length 2048, KL coefficient $\eta$ not specified numerically

**Spinning Up hyperparameters:** [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] reports: clip ratio $\epsilon=0.2$; target KL 0.01 or 0.05; $\gamma=0.99$; $\lambda=0.97$; policy LR $3\times10^{-4}$, value LR $10^{-3}$; 50 epochs, max 80 gradient steps per epoch.

**InstructGPT compute efficiency:** [source:arxiv:2203.02155] reports training the 175B PPO-ptx model required 60 petaflops/s-days, compared to 3,640 petaflops/s-days for the original GPT-3 pretraining — alignment is significantly cheaper than pretraining.

**Disagreement on entropy bonus:** [source:arxiv:1707.06347] includes $c_2 S[\pi_\theta]$ in the objective. [source:arxiv:2307.04964] does not mention entropy bonus in PPO-max. [source:arxiv:2204.05862] notes RLHF decreases policy entropy, potentially limiting diversity for online training. [source:arxiv:1706.03741] adds entropy bonus for exploration due to non-stationary reward. [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] does not include entropy bonus in the listed objective. The role of explicit entropy regularization in LLM PPO is not widely reported; implicit entropy control via KL penalty may suffice.

**Resource and complexity considerations:** [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] notes PPO carries high compute and memory overhead, restricting experimentation to those with extensive computational resources. The algorithm is described as complicated and "packed with nuanced implementation details," requiring both deep theoretical understanding and substantial practical domain knowledge. [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] notes PPO can become trapped in local optima as the policy becomes less random to exploit rewards, and policy drift remains a concern mitigated by early stopping.

## Current status and trajectory

PPO remains the **default** RL optimizer for industrial RLHF pipelines (OpenAI, Anthropic, Google, Meta) as of 2024, but its dominance is contested by simpler alternatives. The trajectory shows three phases:

1. **2019–2022: Establishment.** Ziegler et al. [source:arxiv:1909.08593], Stiennon et al. [source:arxiv:2009.01325], Ouyang et al. [source:arxiv:2203.02155] (InstructGPT), and Bai et al. [source:arxiv:2204.05862] established PPO+KL as the canonical RLHF recipe. InstructGPT demonstrated that the 1.3B aligned model was preferred over the 175B GPT-3 baseline, with the 175B InstructGPT preferred over GPT-3 $85 \pm 3\%$ of the time and over few-shot GPT-3 $71 \pm 4\%$ of the time. Hallucination rate dropped from 41% to 21% on closed-domain tasks, and toxicity reduced by ~25%. PPO was chosen for its stability over TRPO and A2C [source:arxiv:1707.06347; arxiv:1706.03741].
2. **2023: Engineering hardening.** Zheng et al. [source:arxiv:2307.04964] documented "Secrets of RLHF" — PPO-max with critic initialization, reward clipping, ptx loss, and small buffers — addressing instability ("pattern collapse") where policies overoptimize the RM. This represents a maturation of practice, not algorithmic innovation.
3. **2023–present: Challenge from offline and pairwise methods.** DPO [source:arxiv:2305.18290] (implied by [source:arxiv:2310.00212] and [source:arxiv:2403.19279]), P3O [source:arxiv:2310.00212], and RLP [source:arxiv:2403.19279] demonstrate competitive or better KL-reward frontiers without standard online RL. [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] shows P3O achieves strictly dominant KL-reward frontiers vs PPO and DPO: on HH dataset, P3O wins 57.0% against PPO and 69.3% against SFT (GPT-4 eval); vs DPO, P3O has 45.4% GPT-4 win rate despite DPO's higher raw reward (DPO exhibits higher KL). [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] notes PPO served as the "default RL algorithm in LLM post-training for years" but alternative algorithms like GRPO have been adopted for specific tasks like LLM reasoning. PPO's sample inefficiency (requires on-policy rollouts, multiple epochs) and infrastructure complexity (distributed rollout generation, critic training, GAE) motivate alternatives. However, PPO remains **necessary for**:
   - Iterative online RLHF where the RM is updated with fresh human data [source:arxiv:2204.05862]
   - Settings requiring per-token credit assignment (e.g., process supervision, tool use) [source:arxiv:2307.04964]
   - Cases where the reward is not purely preference-based (e.g., verifiable rewards, RL for reasoning) [source:arxiv:2307.04964]

**Hedge:** The field has not abandoned PPO; major labs still use it for their strongest models. But the *proportion* of alignment research using PPO is declining as DPO, KTO, and RLAIF gain traction for static datasets. No source reports a large-scale ablation proving PPO's superiority over DPO when both are tuned optimally on the same data and compute budget. [source:arxiv:2310.00212] shows P3O beats PPO on KL-reward frontier, but P3O is still an online RL method. The "PPO vs. offline" debate is not settled by current benchmarks.

**InstructGPT limitations:** [source:arxiv:2203.02155] notes demographic bias (aligned to ~40 contractors, primarily English-speaking in US and Southeast Asia), safety gaps (still generates toxic/biased outputs, fabricates facts), instruction failures (fails to detect false premises, over-hedges, struggles with multiple constraints), and harmful compliance (follows harmful instructions, generating more toxic output than GPT-3 when explicitly prompted to be biased).

## Key takeaways

- PPO for LLMs adapts the clipped surrogate objective $L^{\text{CLIP}}$ to a contextual bandit: per-token ratios, sequence-level sparse reward, GAE reducing to $r(x,y)-V(x)$.
- Two distinct regularizers operate simultaneously: **clipping** ($\epsilon\approx0.2$) constrains the per-update policy ratio $r_t(\theta)$; **KL penalty** ($\beta\sim10^{-3}$ to $10^{-1}$) anchors to the frozen SFT model. They are not interchangeable.
- The reward model is trained with Bradley-Terry loss on pairwise preferences; its score is normalized and clipped before entering PPO. RM staleness under policy shift is a fundamental limitation addressed by RLP [source:arxiv:2403.19279] and iterative online RLHF [source:arxiv:2204.05862].
- Practical stability requires: critic initialized from RM, reward normalization/clipping, small rollout buffers, global gradient clipping, and optionally pretraining gradient mixing (PPO-ptx) [source:arxiv:2307.04964; arxiv:2203.02155].
- Token-level KL penalty (added to per-token reward) is mathematically equivalent to sequence-level KL for the total return but changes advantage estimation; most LLM implementations use token-level [source:arxiv:2307.04964; arxiv:2204.05862].
- PPO's sample inefficiency and infrastructure complexity motivate offline alternatives (DPO, P3O, RLP), but PPO remains the only method supporting fully online, iterative preference collection and per-token credit assignment.
- **InstructGPT results:** 175B InstructGPT preferred over 175B GPT-3 $85\pm3\%$; 1.3B InstructGPT preferred over 175B GPT-3; hallucination rate 41%→21%; toxicity reduced ~25%; alignment cost 60 vs 3,640 petaflops/s-days vs pretraining [source:arxiv:2203.02155].
- **P3O results:** Strictly dominant KL-reward frontiers vs PPO/DPO; 57% win rate vs PPO, 69.3% vs SFT on HH (GPT-4 eval); 45.4% GPT-4 win rate vs DPO despite DPO's higher raw reward [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-].

## Related topics

- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md) — Offline alternative to PPO using the same preference data
- [GRPO (Group Relative Policy Optimization)](grpo.md) — Another PPO variant for LLMs
- [Reward modeling for LLMs](reward-modeling.md) — Details on RM architecture, training, and overoptimization
- [RL for reasoning models](rl-for-reasoning.md) — PPO applied to verifiable rewards (math, code)
- [Policy gradient methods for LLMs](policy-gradient-methods.md) — Broader policy gradient family
- [KL regularization in RLHF](kl-regularization.md) — Deep dive on KL penalty theory and adaptive control
- [MDP formulation of LLM generation](mdp-formulation.md) — Bandit vs. MDP perspectives
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md) — End-to-end system view
- [Reward model over-optimization](reward-model-overoptimization.md) — Pathology PPO exacerbates
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md) — Role of entropy bonus vs. KL
- [Distributed RL training for LLMs](distributed-rl-training.md) — Infrastructure for PPO rollouts and updates
- [Async and off-policy RL](async-and-off-policy-rl.md) — Alternatives to on-policy PPO
- [Rollout generation infrastructure](rollout-generation-infra.md) — Engineering of the environment interaction loop

## References
- [source:arxiv:2403.19279] [Fine-Tuning Language Models with Reward Learning on Policy](https://arxiv.org/abs/2403.19279)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [source:arxiv:2310.00212] [Pairwise Proximal Policy Optimization (P3O)](https://arxiv.org/abs/2310.00212)
- [source:arxiv:1909.08593] [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593)
- [source:arxiv:2009.01325] [Simple, Scalable, and Effective Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2009.01325)
- [source:arxiv:1706.03741] [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- [source:arxiv:2307.04964] [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964)
- [source:arxiv:2204.05862] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] [Proximal Policy Optimization (PPO): The Key to LLM Alignment (Overview)](https://cameronrwolfe.substack.com/p/ppo-llm)
- [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] [Spinning Up in Deep RL: PPO Documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] [Rethinking the Role of PPO in RLHF (BAIR Blog)](http://bair.berkeley.edu/blog/2023/10/16/p3o/)
