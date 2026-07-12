---
title: PPO for LLM fine-tuning (RLHF)
maturity: comprehensive
updated: '2026-07-12'
sources:
- arxiv:2203.02155
- bair:rethinking-the-role-of-ppo-in-rlhf-bair-
- arxiv:2310.00212
- arxiv:2009.01325
- arxiv:2307.04964
- arxiv:2403.19279
- arxiv:2204.05862
- arxiv:1909.08593
- arxiv:1706.03741
- cameronrwolfe:proximal-policy-optimization-ppo-the-key
- spinningup:spinning-up-in-deep-rl-ppo-documentation
- arxiv:1707.06347
- arxiv:2602.21765
- arxiv:2505.17508
- arxiv:2503.11019
- arxiv:2312.11456
- arxiv:2312.12065
- arxiv:2502.13177
open_questions:
- How does the choice of advantage estimator (GAE vs. sequence-level) affect alignment
  quality when the reward model is trained on pairwise preferences?
- What is the optimal KL coefficient scaling law with respect to model size and reward
  magnitude?
- Can the theoretical convergence guarantees for PPO-Clip under neural approximation
  be extended to the LLM RLHF setting with a learned reward model and KL penalty?
- Does the entropy bonus provide measurable benefits in LLM PPO beyond the implicit
  entropy control from the KL penalty?
---

Proximal Policy Optimization (PPO) is the dominant reinforcement learning algorithm for aligning large language models via RLHF, combining a clipped surrogate objective with a KL penalty to stabilize policy updates against a learned reward model. This article provides a rigorous technical synthesis of the policy-gradient foundations, reward model integration, KL regularization mechanics, clipping behavior, and practical modifications that distinguish LLM PPO from its original continuous-control formulation.

## Policy-gradient foundations for LLM fine-tuning

The policy gradient theorem gives the gradient of the expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \gamma^t r_t]$ as $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) A^{\pi_\theta}(s_t,a_t)]$. In the LLM RLHF setting, the trajectory $\tau$ is a generated response $y$ conditioned on a prompt $x$, the action space is the vocabulary at each token position, and the reward is sparse — typically a single scalar $r(x,y)$ from a reward model assigned at sequence end [source:arxiv:2009.01325]. The vanilla policy gradient objective $L^{PG}(\theta) = \hat{\mathbb{E}}_t[\log \pi_\theta(a_t|s_t) \hat{A}_t]$ becomes, for a full sequence $y = (y_1,\dots,y_T)$,

$$
L^{PG}(\theta) = \mathbb{E}_{x\sim\mathcal{D}, y\sim\pi_\theta(\cdot|x)}\left[ \sum_{t=1}^T \log \pi_\theta(y_t|x,y_{<t}) \cdot \hat{A}_t \right],
$$

where $\hat{A}_t$ is estimated via Generalized Advantage Estimation (GAE) [source:arxiv:1707.06347; arxiv:2307.04964]. GAE computes $\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$ with $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ [source:arxiv:1707.06347; arxiv:2307.04964]. The value function $V_\phi(x)$ is trained with a squared-error loss $L^{VF} = (V_\phi(s_t) - \hat{R}_t)^2$ [source:arxiv:1707.06347; arxiv:2307.04964].

**MDP formulation for LLMs:** [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] maps the RL framework explicitly to the LLM context: the policy $\pi_\theta$ is the LLM itself; the initial state is the input prompt; each action $a_t$ is an individual token predicted by the LLM; the state $s_t$ is the running completion (prompt plus all tokens generated up to time $t$); the trajectory is the entire generated completion from prompt to stop token (e.g., `<eos>`); and the reward is a scalar value provided by a reward model or verifier. This formulation uses a discount factor $\gamma$ to prioritize immediate rewards over distant ones.

**Disagreement on GAE necessity:** The original PPO paper [source:arxiv:1707.06347] advocates GAE for variance reduction in multi-step MDPs. For LLM bandits, [source:arxiv:2307.04964] provides the general GAE formula. [source:arxiv:2204.05862] uses a simpler formulation: the total reward $r_{PM}(y|x)$ is assigned at sequence end and the KL penalty is applied per-token (see below). [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] specifies GAE with $\lambda=0.97$ and $\gamma=0.99$ as default hyperparameters, and the Spinning Up implementation uses GAE for advantage estimation even in discrete action spaces. The practical impact of per-token vs. sequence-level advantage estimation on LLM alignment quality is not widely reported.

**Spinning Up hyperparameters:** [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] reports typical hyperparameters for PPO implementations: clip ratio $\epsilon \in [0.1, 0.3]$ (default 0.2); target KL 0.01 or 0.05 (used for early stopping); discount factor $\gamma=0.99$; GAE lambda $\lambda=0.97$; policy learning rate $3\times10^{-4}$, value function learning rate $10^{-3}$; 50 epochs of interaction with maximum 80 gradient descent steps for both policy and value function per epoch.

### Unified KL-regularized policy gradient frameworks

Recent work has developed unified frameworks for KL-regularized policy gradients that clarify the design space of KL direction, normalization, and estimation.

**Regularized Policy Gradient (RPG) framework:** [source:arxiv:2505.17508] proposes the **Regularized Policy Gradient (RPG)** view, a unified derivation specifying the weighting required for various KL variants to ensure the optimized surrogate yields the exact gradient of the intended objective. RPG distinguishes between normalized and unnormalized KL divergences. For unnormalized measures $\pi_{\text{old}}$ with total mass $Z_{\text{old}} = \int \pi_{\text{old}}(x) dx$, the framework defines:
*   **Unnormalized Forward KL (UFKL):** $\text{UKL}(\pi_{\text{old}} \| \pi_\theta) = \int_x \pi_{\text{old}}(x) \log \frac{\pi_{\text{old}}(x)}{\pi_\theta(x)} dx + \int_x (\pi_\theta(x) - \pi_{\text{old}}(x)) dx$
*   **Unnormalized Reverse KL (URKL):** $\text{UKL}(\pi_\theta \| \pi_{\text{old}}) = \int_x \pi_\theta(x) \log \frac{\pi_\theta(x)}{\pi_{\text{old}}(x)} dx + \int_x (\pi_{\text{old}}(x) - \pi_\theta(x)) dx$

The authors prove that the widely used **$k_3$ estimator**, defined as $k_3(y) = y - 1 - \log y$, is equivalent to the unnormalized KL divergence. The RPG framework follows a four-step iterative process: (1) Objective Construction: $J(\theta) = \mathbb{E}_{\pi_\theta}[R(x)] - \beta \cdot \text{Divergence}$; (2) Gradient Derivation using importance weights $w(x) = \pi_\theta(x)/\pi_{\text{old}}(x)$; (3) Surrogate Formulation (fully differentiable or REINFORCE-style with stop-gradient); (4) Policy Update with periodic reference policy refresh. For URKL, the differentiable surrogate is $\mathcal{L}_{\text{URKL}}(\theta) = Z_{\text{old}} \mathbb{E}_{x \sim \tilde{\pi}_{\text{old}}}[-w(x)R(x) + \beta(w(x) \log w(x) - w(x))]$.

**Residual Policy Gradient (RPG) for policy customization:** [source:arxiv:2503.11019] proposes a different **Residual Policy Gradient (RPG)** extending Residual Q-Learning to gradient-based RL for policy customization — adapting a prior policy $\pi$ (trained on basic reward $r$) to new task-specific requirements (add-on reward $r_R$) while preserving original properties. The method derives a **Soft Policy Gradient (SPG)** showing maximizing the entropy-regularized objective is equivalent to standard policy gradient over reformulated reward $r^{\text{SPG}} = r(s_t, a_t) - \alpha \log \pi_\theta(a_t|s_t)$. The customization problem is framed as an augmented MDP $\mathcal{M}^{\text{aug}} = (\mathcal{S}, \mathcal{A}, \omega' \log \pi + r_R, p)$ where $\omega' = \omega \alpha$ balances basic and add-on tasks. The resulting **Residual Policy Gradient** reward for advantage estimation is $r^{\text{RPG}}(s_t, a_t) = r_R(s_t, a_t) + \omega' \log \pi(a_t|s_t) - \hat{\alpha} \log \pi_{\theta}(a_t|s_t)$. The authors rethink the **KL-regularized objective** as a special case where $\omega' = \hat{\alpha} = \beta$: $r^{\mathrm{KL}}(s_{t}, a_{t}) = r_{R}(s_{t}, a_{t}) + \beta \log \pi(a_{t} | s_{t}) - \beta \log \pi_{\theta}(a_{t} | s_{t})$. **Residual PPO** integrates this into PPO by computing advantages using $r^{\text{RPG}}$ and removing the repeated entropy term from the actor loss. Experiments on MuJoCo show Residual PPO matches or exceeds full policy retraining: HalfCheetah $5488.3 \pm 75.6$ vs full policy $5556.2 \pm 57.8$; Ant $5568.3 \pm 852.4$ vs KL PPO $4019.4 \pm 1103.4$.

**PPO-Clip global optimality:** [source:arxiv:2312.12065] establishes the first global convergence proof for PPO-Clip under neural function approximation. The authors reinterpret the PPO-Clip objective through hinge loss: $L_{\text{Hinge}}(\theta) = \frac{1}{|\mathcal{D}|} \sum_{(s, a) \in \mathcal{D}} \text{weight} \times \ell(\text{label}, \text{classifier}, \text{margin})$ where PPO-Clip is a special case with weight $|A^\pi(s, a)|$, classifier $\rho_{s,a}(\theta) - 1$, margin $\epsilon$. For tabular settings, Entropic Mirror Descent (EMDA) yields convergence to optimal $V^{\pi^*}(s)$ with probability one. For neural approximation, they propose a decoupled **Neural PPO-Clip Recipe**: (1) Policy Evaluation: approximate $Q_\omega$ via MSBE with two-layer network; (2) Direct Policy Search: EMDA for $K$ iterations minimizing generalized PPO-Clip objective; (3) Neural Approximation: regression-based update of energy function $f_\theta$; (4) Policy Update: $\pi_{\theta_{t+1}} \propto \exp\{\tau_{t+1}^{-1} f_{\theta_{t+1}}\}$. The closed-form EMDA target policy is $\log \widehat{\pi}_{t+1}(a|s) \propto C_t(s, a) A_{\omega_t}(s, a) + \tau_t^{-1} f_{\theta_t}(s, a)$. They prove a min-iterate convergence rate of $O(T^{-\alpha})$ for $\alpha \in [1/2, 1)$, achieving $O(1/\sqrt{T})$ at $\alpha=1/2$. The clipping range $\epsilon$ only affects the pre-constant, not asymptotic behavior. Assumptions include two-layer networks, sufficient representational capacity, distributional regularity, bounded concentrability, and infinite state-action visitation (tabular).

## Reward model integration

The reward model $r_\psi(x,y)$ is trained on pairwise human preferences using the Bradley-Terry loss [source:arxiv:1706.03741; arxiv:1909.08593; arxiv:2009.01325; arxiv:2204.05862; arxiv:2307.04964]:

$$
\mathcal{L}_{RM}(\psi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}} \left[ \log \sigma\bigl(r_\psi(x,y_w) - r_\psi(x,y_l)\bigr) \right],
$$

where $y_w$ is preferred over $y_l$. The RM is typically initialized from the SFT model with a scalar head [source:arxiv:2009.01325; arxiv:2307.04964]. During PPO, the RM score becomes the environment reward. Critical practical details:

- **Reward normalization:** [source:arxiv:1706.03741] normalizes RM outputs such that reference summaries achieve a mean score of 0. [source:arxiv:2307.04964] adds clipping: $\tilde{r} = \text{clip}\bigl((r - \mu)/\sigma, -\delta, \delta\bigr)$.
- **Reward model overoptimization:** As the policy shifts, the fixed RM becomes inaccurate off-distribution, leading to reward hacking [source:arxiv:2403.19279]. [source:arxiv:2204.05862] observes train/test RM score divergence after ~150k samples, indicating overfitting. [source:arxiv:2403.19279] proposes Reward Learning on Policy (RLP) to retrain the RM on policy-generated samples using synthetic preferences or multi-view learning.
- **Value function initialization:** [source:arxiv:2307.04964] mentions reward model initialization and pre-training for the critic model. [source:arxiv:2009.01325] uses a separate Transformer for the value function initialized from the RM.

**InstructGPT RM training:** [source:arxiv:2203.02155] trains the reward model on all $\binom{K}{2}$ comparisons from a single ranking task as a single batch element to prevent overfitting and increase efficiency. The RM loss is:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

**Relative vs. absolute feedback discrepancy:** [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] identifies a fundamental discrepancy: reward models are trained using **relative feedback** (comparisons between response pairs), yet the RL fine-tuning stage (typically PPO) optimizes for a **single, absolute reward**. This inconsistency makes the RL process sensitive to "reward translation"—where shifting a reward function by a constant $\delta(x)$ does not change the preference loss but can significantly mislead an RL algorithm. The policy may be disrupted by the reward scale of a prompt rather than learning the actual relative preferences.

**Disagreement on reward scaling:** [source:arxiv:1706.03741] normalizes rewards such that reference summaries achieve a mean score of 0. [source:arxiv:2307.04964] uses a running mean/std with clipping. [source:arxiv:2204.05862] does not explicitly describe reward normalization in the PPO phase, only that the PM score is used directly with a KL penalty. [source:arxiv:2203.02155] does not specify reward normalization in the PPO phase. [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] argues this sensitivity is precisely the problem: PPO is not invariant to reward translation, unlike the Bradley-Terry loss used to train the RM.

### Generalization under reward shift and clipped KL

[source:arxiv:2602.21765] develops a generalization theory for RLHF using change-of-measure decomposition and PAC-Bayes tools, bounding the discrepancy between the empirical objective $\widehat{J}_{n,K}^{\phi,\tau}(\theta)$ and the target population objective $J^{\star}(\theta)$. The theory addresses two practical implementation details: **Reward Shift** (reward models trained on earlier/mixed behavior policies but optimized on current policy rollouts, leading to unreliable regions) and **Clipped KL Regularisation** (KL divergence estimated from sampled log-probability ratios and clipped to threshold $\tau$, introducing systematic objective mismatch).

The **Fixed-Policy Generalisation Bound** (Theorem 1) states with probability $\geq 1-\delta$:

$$
\left| \widehat{J}_{n, K}^{\phi, \tau}(\theta)-J^{\star}(\theta) \right| \leq (1+2 \beta \tau)\left(\sqrt{\frac{\log (4 / \delta)}{2 n}}+\sqrt{\frac{\log (4 / \delta)}{2 n K}}\right) + \mathcal{C}(\theta) \sqrt{L_{\text {train }}^{(2)}(\phi)} + \beta \mathbb{E}_{(X, Y) \sim D_{\theta}}\left[\left|\ell_{\theta}(X, Y)-\ell_{\theta}^{\tau}(X, Y)\right|\right]
$$

Three interpretable components:
1. **Sampling Error:** Noise from finite prompts $n$ and rollouts $K$ per prompt.
2. **Reward Shift Error:** Gap between learned reward $\hat{r}_\phi$ and target $r^*$, amplified by distribution shift. **Coverage Coefficient** $\mathcal{C}(\theta) = \sqrt{1 + \chi^2(D_{\theta} \| D_{\text{train}})}$ quantifies shift from training to deployment. **Training Error** $L_{\text{train}}^{(2)}(\phi) = \mathbb{E}_{(X,Y)\sim D_{\text{train}}}[(\hat{r}_{\phi}(X,Y)-r^{*}(X,Y))^{2}]$.
3. **KL Clipping Error:** Bias from clipped log-ratio $\ell_\theta^\tau$ vs exact $\ell_\theta$. $\beta$ is KL regularization strength.

For data-dependent policy selection, the **PAC-Bayes Bound** (Theorem 2) averages over posterior $Q$, adding complexity term $\text{KL}(Q \| P)$ to sampling error.

**Practical implications:**
- **Optimal KL Clipping Threshold ($\tau^*$):** $\tau^*$ should be the $(1 - 2\alpha_{n,K,\delta})$-quantile of $|\ell_\theta(X,Y)|$ under $D_\theta$, where $\alpha_{n,K,\delta} = \sqrt{\frac{\log(4/\delta)}{2n}} + \sqrt{\frac{\log(4/\delta)}{2nK}}$. As evaluation budget increases, $\tau$ should increase to reduce bias.
- **Budget Allocation:** Uniform cost: $K^* = 1$ (prioritize prompt coverage). Prefill/decode cost: $K^* = \max \left\{ 1, \left( \frac{c_{\text{prefill}}}{c_{\text{decode}}} \right)^{2/3} \right\}$. Variance-aware: $K^* \approx \max \left\{ 1, \sqrt{\frac{c_{\text{prefill}}}{c_{\text{decode}}} \cdot \frac{\sigma^2_{\text{rollout}}}{\sigma^2_{\text{prompt}}}} \right\}$.

**Limitations:** Requires absolute continuity $D_\theta \ll D_{\text{train}}$ with finite $\chi^2$ divergence; bounded squared training error; integrable exact log-ratio; heuristic calibration for $\tau$ selection from evaluation sample.

## KL penalty: theory and practice

The KL penalty prevents the RL policy $\pi_\theta$ from drifting too far from the reference policy $\pi^{\text{ref}}$ (usually the SFT model), preserving language quality and preventing reward hacking. The modified reward is

$$
r_{\text{total}}(x,y) = r_\psi(x,y) - \beta \, D_{\text{KL}}\bigl(\pi_\theta(\cdot|x) \,\|\, \pi^{\text{ref}}(\cdot|x)\bigr)
$$

[source:arxiv:1909.08593; arxiv:2009.01325; arxiv:2204.05862; arxiv:2307.04964]. Two forms of KL application exist:

1. **Sequence-level KL:** $D_{\text{KL}}(\pi_\theta(y|x) \| \pi^{\text{ref}}(y|x)) = \sum_t \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi^{\text{ref}}(y_t|x,y_{<t})}$ subtracted once per sequence [source:arxiv:1909.08593; arxiv:2009.01325].
2. **Token-level KL:** $-\beta \sum_t \log \frac{\pi_\theta(y_t|x,y_{<t})}{\pi^{\text{ref}}(y_t|x,y_{<t})}$ added to the per-token reward, so the advantage at each step includes the KL term [source:arxiv:2307.04964].

These are mathematically equivalent for the total return but differ in advantage estimation: token-level KL distributes the penalty across time steps, affecting GAE. [source:arxiv:2307.04964] uses token-level KL in the total reward $r_{\text{total}}(x,y_i) = r(x,y_i) - \eta \text{KL}(\pi_\theta^{\text{RL}}(y_i|x), \pi^{\text{SFT}}(y_i|x))$ per token $y_i$.

**InstructGPT PPO-ptx objective:** [source:arxiv:2203.02155] formulates the PPO-ptx objective as:

$$
\text{objective} (\phi) = \mathbb{E} _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\frac{\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x)}{\pi^ {\mathrm{SFT}} (y \mid x)}\right) \right] + \gamma \mathbb{E} _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]
$$

where $\beta$ is the KL reward coefficient and $\gamma$ is the pretraining loss coefficient. This mixes PPO updates with updates that increase the log likelihood of the original pretraining distribution to mitigate the alignment tax.

**Adaptive KL control:** [source:arxiv:1707.06347] proposes an adaptive $\beta$ for the penalty version (PPO-Penalty): if $d < d_{\text{targ}}/1.5$, $\beta \leftarrow \beta/2$; if $d > 1.5 d_{\text{targ}}$, $\beta \leftarrow 2\beta$, where $d = \hat{\mathbb{E}}_t[\text{KL}]$. [source:arxiv:1909.08593] uses a log-space proportional controller. [source:arxiv:2204.05862] uses a fixed $\lambda_{\text{KL}} = 0.001$ (they report $\beta \geq 0$ typically $0.001$). [source:arxiv:2307.04964] uses a fixed $\eta$ (KL reward coefficient) and monitors KL divergence as a stability metric. [source:arxiv:2009.01325] uses a fixed $\beta$. [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] employs **early stopping** based on KL divergence: gradient steps halt if the mean KL-divergence exceeds the `target_kl` threshold (typically 0.01 or 0.05), as clipping alone may not fully prevent the policy from moving too far from the old policy.

**Disagreement on KL coefficient magnitude:** [source:arxiv:2204.05862] uses $\beta=0.001$. [source:arxiv:1909.08593] uses a log-space proportional controller to target a specific KL value. [source:arxiv:2307.04964] does not specify $\eta$ numerically but notes monitoring KL is crucial. [source:arxiv:2203.02155] does not specify $\beta$ numerically. The optimal $\beta$ scales with model size and reward magnitude; no consensus scaling law is reported. [source:arxiv:1707.06347] found adaptive KL penalty performed worse than clipping on MuJoCo (adaptive KL achieved between 0.68 and 0.74 vs clipping $\epsilon=0.2$ scored 0.82), but this was for continuous control without a separate reward model.

**KL-reward tradeoff:** [source:arxiv:2204.05862] observes an approximately linear relationship between $\sqrt{D_{\text{KL}}(\pi\|\pi_0)}$ and PM reward during training (Figures 4, 13). [source:arxiv:2310.00212] shows P3O achieves a better KL-reward frontier than PPO and DPO: on HH, P3O-V1/V2 dominate PPO and DPO respectively, delivering 0.1–0.3 higher reward at same KL. [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] notes that excessive deviation from the reference policy (high KL divergence) can lead the online policy to "cut corners" of the reward model, resulting in incoherent continuations. While P3O provides better KL control than DPO, the balance between reward maximization and maintaining the integrity of the reference policy remains a critical constraint.

### Instance-level adaptive KL penalty for direct alignment

[source:arxiv:2502.13177] proposes **$\epsilon$-Direct Preference Optimization ($\epsilon$-DPO)**, an instance-level adaptive KL penalty control mechanism for DPO (which shares the KL-regularized objective structure with PPO). Standard DPO uses a static $\beta$; $\epsilon$-DPO adapts $\beta$ per preference pair based on logit monotonicity under $\beta$ perturbation.

**Method:** For $\beta$ and small $\epsilon>0$, define perturbed values $\beta_{\varepsilon}^{-} := \frac{\beta}{1+\varepsilon}$, $\beta_{\varepsilon}^{+} := \frac{\beta}{1-\varepsilon}$. Approximate policies under perturbed $\beta$ by reusing logits: $\pi_{\theta(\beta_{\varepsilon}^{-})} \approx \prod_{i=1}^{n} \text{Softmax}\big((1+\varepsilon)f_{\theta}(x,y_{1:i-1}) - \varepsilon f_{\text{ref}}(x,y_{1:i-1})\big)_{y_{i}}$. Compute log-likelihood ratio $z_{\theta}(x, y^w, y^l) = \log \frac{\pi_\theta(y^w|x)}{\pi_\theta(y^l|x)}$. Determine instance-level $\tilde{\beta}$:
- $\tilde{\beta} = \beta_{\varepsilon}^{-}$ if $z_{\theta(\beta_{\varepsilon}^{-})} > z_{\theta(\beta)} > z_{\theta(\beta_{\varepsilon}^{+})}$
- $\tilde{\beta} = \beta_{\varepsilon}^{+}$ if $z_{\theta(\beta_{\varepsilon}^{-})} < z_{\theta(\beta)} < z_{\theta(\beta_{\varepsilon}^{+})}$
- $\tilde{\beta} = \beta$ otherwise.

Update using DPO loss with instance-specific $\tilde{\beta}$, then update baseline $\beta \leftarrow \mathbb{E}_{x,y^w,y^l}[\tilde{\beta}]$.

**Results:** On Llama-3-8B-Instruct with UltraFeedback: AlpacaEval 2 length-controlled win rate 46.4% (vs DPO 40.3%); Arena-Hard win rate 36.7% (vs DPO 32.6%); MT-Bench 8.0. Computational overhead negligible ($\Delta t \approx 0.0006\text{s}$/step). Superior Pareto frontier between forward KL and win rate vs TR-DPO on Anthropic-HH. **Limitations:** Sensitivity to $\epsilon$ at high levels/early training; downstream math task degradation (GSM8K) consistent with other direct alignment methods.

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

**Full PPO objective for LLMs:**

$$
L(\theta) = \hat{\mathbb{E}}_{x,y} \left[ \sum_t L_t^{\text{CLIP}}(\theta) - c_1 L_t^{VF}(\theta) \right]
$$

[source:arxiv:1707.06347; arxiv:2307.04964]. [source:arxiv:2307.04964] adds a pretraining gradient term (PPO-ptx): $\lambda_{\text{ptx}} \mathbb{E}_{x\sim\mathcal{D}_{\text{pretrain}}}[\log \pi_\theta(x)]$ to mitigate alignment tax.

**P3O clipping variants:** [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] introduces two implementation styles for clipping in Pairwise PPO: **V1 (separate clipping)** and **V2 (joint clipping)**. Both clip the importance sampling ratio and the gradient updates to prevent excessively large updates and better manage the trade-off between KL divergence and reward.

**RPG-Style Clip for off-policy variance reduction:** [source:arxiv:2505.17508] introduces **RPG-Style Clip** adapting Dual-Clip by decomposing the importance weight into ratio $w(x)$ and regularized advantage $\hat{A}_{\text{reg}}(x)$. The REINFORCE-style loss becomes:

$$
\mathcal{L}^{\text{RPG-Clip}}(\theta) = -\mathbb{E}_{x \sim \tilde{\pi}_{\text{old}}}[\mathcal{C}(w(x), \text{SG}(\hat{A}_{\text{reg}}(x))) \log \pi_\theta(x)]
$$

where $\mathcal{C}$ clips $w$ to $[1-\epsilon_1, 1+\epsilon_2]$ for positive advantages and enforces lower bound $c$ for negative advantages. This manages high variance of importance weights in off-policy settings. The clipping parameters $(\epsilon_1, \epsilon_2)$ govern a controllable bias-variance trade-off; principled schedules for these are a future direction.

**Disagreement on clipping vs. penalty for LLMs:** The original PPO paper [source:arxiv:1707.06347] found clipping superior to adaptive KL penalty on continuous control. However, [source:arxiv:2307.04964] explicitly combines both clipping and a KL penalty. The KL penalty acts on the divergence from the *reference* (SFT) model, while clipping constrains the step from the *old* (current iteration) policy. These serve different purposes: clipping ensures per-update stability; KL penalty anchors to the initial supervised model. [source:arxiv:2310.00212] argues PPO's token-wise clipping is mismatched to trajectory-wise reward models (BTL is invariant to constant reward shifts, but PPO is not), motivating their pairwise P3O algorithm.

## PPO variants and practical modifications for LLMs

| Modification | Description | Sources |
|--------------|-------------|---------|
| **PPO-ptx** | Adds pretraining loss $\lambda_{\text{ptx}} \mathbb{E}[\log \pi_\theta(x)]$ to preserve pretrained capabilities | [source:arxiv:2307.04964; arxiv:2203.02155] |
| **Reward normalization & clipping** | Running mean/std + clip to $[-\delta,\delta]$ | [source:arxiv:2307.04964] |
| **Critic initialization from RM** | Initialize $V_\phi$ from RM backbone | [source:arxiv:2307.04964] |
| **Small experience buffer** | Use small batch for environment sampling (e.g., 128) and smaller minibatch for updates (e.g., 32) | [source:arxiv:2307.04964] |
| **Global gradient clipping** | Clip global norm (e.g., 1.0) | [source:arxiv:2307.04964] |
| **Token-level KL in reward** | Distribute KL penalty per token for GAE | [source:arxiv:2307.04964] |
| **Adaptive KL controller** | Proportional control on KL target | [source:arxiv:1909.08593] |
| **Early stopping on KL** | Halt gradient steps if mean KL exceeds target_kl | [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] |
| **Reference model = SFT model** | Fixed $\pi^{\text{ref}}$ throughout training | [source:arxiv:1909.08593; arxiv:2009.01325; arxiv:2204.05862] |
| **Multiple PPO epochs per rollout** | Typically 1–4 epochs (K in original PPO) | [source:arxiv:1707.06347] |
| **Nucleus sampling for rollouts** | $p=0.9, \tau=0.8$, repetition penalty 1.1 | [source:arxiv:2307.04964] |
| **Pairwise PPO (P3O)** | Optimizes relative preferences directly; invariant to reward translation | [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] |
| **Gibbs Sampling from Human Feedback (GSHF)** | Offline learning with pessimism: conservative reward estimation via uncertainty penalties | [source:arxiv:2312.11456] |
| **Online Iterative GSHF** | Non-symmetric structure: Main Agent exploits, Enhancer explores via uncertainty maximization under KL bound | [source:arxiv:2312.11456] |
| **Multi-step RSO** | Progressive rejection sampling: $\pi_0 \to \pi_0 \exp(\frac{1}{\eta_1}r) \to \dots \to \pi_0 \exp(\frac{1}{\eta_N}r)$ | [source:arxiv:2312.11456] |
| **RPG-REINFORCE with RPG-Style Clip** | Unified KL-regularized PG with off-policy importance weighting correction and adaptive clipping | [source:arxiv:2505.17508] |
| **Residual PPO** | Policy customization via augmented MDP reward $r^{\text{RPG}} = r_R + \omega' \log \pi - \hat{\alpha} \log \pi_\theta$ | [source:arxiv:2503.11019] |
| **Neural PPO-Clip Recipe** | Decoupled policy search (EMDA) + neural approximation for provable convergence | [source:arxiv:2312.12065] |

**Hyperparameters reported in [source:arxiv:2307.04964] (7B/13B models):**
- Policy LR: $5\times10^{-7}$, Critic LR: $1.65\times10^{-6}$, warmup 10%
- Batch size (env): 128, minibatch: 32
- GAE $\lambda=0.9$
- Max length 2048, KL coefficient $\eta$ not specified numerically

**Spinning Up hyperparameters:** [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] reports: clip ratio $\epsilon=0.2$; target KL 0.01 or 0.05; $\gamma=0.99$; $\lambda=0.97$; policy LR $3\times10^{-4}$, value LR $10^{-3}$; 50 epochs, max 80 gradient steps per epoch.

**GSHF practical results:** [source:arxiv:2312.11456] Hybrid-GSHF-DPO outperformed DPO and RSO in gold reward and GPT-4 evaluations on HH-RLHF, with higher robustness to OOD data (smaller in-domain vs OOD reward gap). Online-GSHF-DPO on Mistral-7B achieved 34.79% length-control win rate on AlpacaEval-2. Multi-step RSO produced reward-KL curves strictly dominating original RSO. **Limitations:** Length bias observed (avg length 161→263 tokens), suggesting reward models favor wordier responses; uncertainty estimation for deep networks relies on heuristic ensembles; alignment tax persists under strong optimization on imperfect rewards.

**RPG experimental results:** [source:arxiv:2505.17508] On Qwen3-4B and Qwen2.5-7B-Instruct (AIME24, AIME25, AMC23, 8K context): RPG-REINFORCE with RPG-Style Clip improved accuracy by up to **+6 absolute pp** over DAPO baseline on AIME24/25. On AIME25, achieved **52% accuracy**, surpassing official Qwen3-4B-Instruct (47%). More stable training progressions in reward and policy entropy vs GRPO.

**Residual PPO results:** [source:arxiv:2503.11019] MuJoCo: HalfCheetah $5488.3 \pm 75.6$ (vs full policy $5556.2 \pm 57.8$); Ant $5568.3 \pm 852.4$ (vs KL PPO $4019.4 \pm 1103.4$); Hopper $4401.4 \pm 505.3$ (improving on Prior Policy and Greedy PPO). Soft PPO matched/outperformed other entropy-regularized PPO variants (HalfCheetah $5896.5 \pm 371.8$ vs Repeat-Entropy PPO $-38.9 \pm 58.2$). **Limitation:** Decoupling $\omega'$ and $\hat{\alpha}$ can lead to suboptimal results with "excessively large assumption misalignments" (observed in Hopper).

**Disagreement on entropy bonus:** [source:arxiv:1707.06347] includes $c_2 S[\pi_\theta]$ in the objective. [source:arxiv:2307.04964] does not mention entropy bonus in PPO-max. [source:arxiv:2204.05862] notes RLHF decreases policy entropy, potentially limiting diversity for online training. [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] does not include entropy bonus in the listed objective. The role of explicit entropy regularization in LLM PPO is not widely reported; implicit entropy control via KL penalty may suffice.

**Resource and complexity considerations:** [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] notes PPO carries high compute and memory overhead, restricting experimentation to those with extensive computational resources. The algorithm is described as complicated and "packed with nuanced implementation details," requiring both deep theoretical understanding and substantial practical domain knowledge. [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] notes PPO can become trapped in local optima as the policy becomes less random to exploit rewards, and policy drift remains a concern mitigated by early stopping.

## Current status and trajectory

PPO remains the **default** RL optimizer for industrial RLHF pipelines (OpenAI, Anthropic, Google, Meta) as of 2024, but its dominance is contested by simpler alternatives. The trajectory shows three phases:

1. **2019–2022: Establishment.** Ziegler et al. [source:arxiv:1909.08593], Stiennon et al. [source:arxiv:2009.01325], Ouyang et al. [source:arxiv:2203.02155] (InstructGPT), and Bai et al. [source:arxiv:2204.05862] established PPO+KL as the canonical RLHF recipe. InstructGPT demonstrated that the 1.3B aligned model was preferred over the 175B GPT-3 baseline, with the 175B InstructGPT preferred over GPT-3 $85 \pm 3\%$ of the time and over few-shot GPT-3 $71 \pm 4\%$ of the time. Hallucination rate dropped from 41% to 21% on closed-domain tasks, and toxicity reduced by ~25%.
2. **2023: Engineering hardening.** Zheng et al. [source:arxiv:2307.04964] documented "Secrets of RLHF" — PPO-max with critic initialization, reward clipping, ptx loss, and small buffers — addressing instability ("pattern collapse") where policies overoptimize the RM. This represents a maturation of practice, not algorithmic innovation.
3. **2023–present: Challenge from offline and pairwise methods.** DPO (implied by [source:arxiv:2310.00212] and [source:arxiv:2403.19279]), P3O [source:arxiv:2310.00212], and RLP [source:arxiv:2403.19279] demonstrate competitive or better KL-reward frontiers without standard online RL. [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] shows P3O achieves strictly dominant KL-reward frontiers vs PPO and DPO: on HH dataset, P3O wins 57.0% against PPO and 69.3% against SFT (GPT-4 eval); vs DPO, P3O has 45.4% GPT-4 win rate despite DPO's higher raw reward (DPO exhibits higher KL). [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] notes PPO served as the "default RL algorithm in LLM post-training for years" but alternative algorithms like GRPO have been adopted for specific tasks like LLM reasoning. PPO's sample inefficiency (requires on-policy rollouts, multiple epochs) and infrastructure complexity (distributed rollout generation, critic training, GAE) motivate alternatives. However, PPO remains **necessary for**:
   - Iterative online RLHF where the RM is updated with fresh human data [source:arxiv:2204.05862]

**Hedge:** The field has not abandoned PPO; major labs still use it for their strongest models. But the *proportion* of alignment research using PPO is declining as DPO, KTO, and RLAIF gain traction for static datasets. No source reports a large-scale ablation proving PPO's superiority over DPO when both are tuned optimally on the same data and compute budget. [source:arxiv:2310.00212] shows P3O beats PPO on KL-reward frontier, but P3O is still an online RL method. The "PPO vs. offline" debate is not settled by current benchmarks.

**InstructGPT limitations:** [source:arxiv:2203.02155] notes demographic bias (aligned to ~40 contractors, primarily English-speaking in US and Southeast Asia), safety gaps (still generates toxic/biased outputs, fabricates facts), instruction failures (fails to detect false premises, over-hedges, struggles with multiple constraints), and harmful compliance (follows harmful instructions, generating more toxic output than GPT-3 when explicitly prompted to be biased).

**Theoretical maturation:** Recent work has significantly advanced theoretical understanding: [source:arxiv:2312.12065] proves global convergence of PPO-Clip with $O(1/\sqrt{T})$ rate under neural approximation; [source:arxiv:2602.21765] provides PAC-Bayes generalization bounds decomposing reward shift, sampling error, and KL clipping error with practical hyperparameter guidelines; [source:arxiv:2505.17508] unifies KL-regularized PG variants and identifies GRPO's importance-weighting mismatch; [source:arxiv:2503.11019] frames KL-regularization as residual policy gradient for policy customization.

## Key takeaways

- PPO for LLMs adapts the clipped surrogate objective $L^{\text{CLIP}}$ to a contextual bandit: per-token ratios, sequence-level sparse reward, GAE reducing to $r(x,y)-V(x)$.
- Two distinct regularizers operate simultaneously: **clipping** ($\epsilon\approx0.2$) constrains the per-update policy ratio $r_t(\theta)$; **KL penalty** ($\beta\sim10^{-3}$ to $10^{-1}$) anchors to the frozen SFT model. They are not interchangeable.
- The reward model is trained with Bradley-Terry loss on pairwise preferences; its score is normalized and clipped before entering PPO. RM staleness under policy shift is a fundamental limitation addressed by RLP [source:arxiv:2403.19279] and iterative online RLHF [source:arxiv:2204.05862].
- Practical stability requires: critic initialized from RM, reward normalization/clipping, small rollout buffers, global gradient clipping, and optionally pretraining gradient mixing (PPO-ptx) [source:arxiv:2307.04964; arxiv:2203.02155].
- Token-level KL penalty (added to per-token reward) is mathematically equivalent to sequence-level KL for the total return but changes advantage estimation; most LLM implementations use token-level [source:arxiv:2307.04964].
- PPO's sample inefficiency and infrastructure complexity motivate offline alternatives (DPO, P3O, RLP), but PPO remains the only method supporting fully online, iterative preference collection and per-token credit assignment.
- **InstructGPT results:** 175B InstructGPT preferred over 175B GPT-3 $85\pm3\%$; 1.3B InstructGPT preferred over 175B GPT-3; hallucination rate 41%→21%; toxicity reduced ~25% [source:arxiv:2203.02155].
- **P3O results:** Strictly dominant KL-reward frontiers vs PPO/DPO; 57% win rate vs PPO, 69.3% vs SFT on HH (GPT-4 eval); 45.4% GPT-4 win rate vs DPO despite DPO's higher raw reward [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-].
- **Generalization theory:** RLHF generalization error decomposes into sampling error, reward shift error (quantified by coverage coefficient $\mathcal{C}(\theta) = \sqrt{1 + \chi^2(D_{\theta} \| D_{\text{train}})}$), and KL clipping error. Optimal clipping threshold $\tau^*$ is the $(1-2\alpha)$-quantile of log-ratio magnitude; optimal rollouts per prompt $K^* = \max\{1, (c_{\text{prefill}}/c_{\text{decode}})^{2/3}\}$ [source:arxiv:2602.21765].
- **Unified KL frameworks:** RPG [source:arxiv:2505.17508] unifies forward/reverse, normalized/unnormalized KL with correct importance weighting; Residual PG [source:arxiv:2503.11019] reinterprets KL-regularization as policy customization via augmented MDP rewards.
- **Instance-adaptive KL:** $\epsilon$-DPO [source:arxiv:2502.13177] achieves superior KL-reward Pareto frontiers via per-instance $\beta$ adaptation using logit monotonicity under perturbation.
- **Provable convergence:** PPO-Clip attains global optimality with $O(1/\sqrt{T})$ rate; clipping range $\epsilon$ affects only pre-constants [source:arxiv:2312.12065].

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
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [source:bair:rethinking-the-role-of-ppo-in-rlhf-bair-] [Rethinking the Role of PPO in RLHF (BAIR Blog)](http://bair.berkeley.edu/blog/2023/10/16/p3o/)
- [source:arxiv:2310.00212] [Pairwise Proximal Policy Optimization (P3O)](https://arxiv.org/abs/2310.00212)
- [source:arxiv:2009.01325] [Simple, Scalable, and Effective Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2009.01325)
- [source:arxiv:2307.04964] [Secrets of RLHF in Large Language Models Part I: PPO](https://arxiv.org/abs/2307.04964)
- [source:arxiv:2403.19279] [Fine-Tuning Language Models with Reward Learning on Policy](https://arxiv.org/abs/2403.19279)
- [source:arxiv:2204.05862] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [source:arxiv:1909.08593] [The Boltzmann Optimum in RLHF (Ziegler et al. 2019)](https://arxiv.org/abs/1909.08593)
- [source:arxiv:1706.03741] [Deep reinforcement learning from human preferences (Christiano et al. 2017)](https://arxiv.org/abs/1706.03741)
- [source:cameronrwolfe:proximal-policy-optimization-ppo-the-key] [Proximal Policy Optimization (PPO): The Key to LLM Alignment (Overview)](https://cameronrwolfe.substack.com/p/ppo-llm)
- [source:spinningup:spinning-up-in-deep-rl-ppo-documentation] [Spinning Up in Deep RL: PPO Documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)
- [source:arxiv:2602.21765] [Generalisation of RLHF under Reward Shift and Clipped KL Regularisation](https://arxiv.org/abs/2602.21765)
- [source:arxiv:2505.17508] [On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning](https://arxiv.org/abs/2505.17508)
- [source:arxiv:2503.11019] [Residual Policy Gradient: A Reward View of KL-regularized Objective](https://arxiv.org/abs/2503.11019)
- [source:arxiv:2312.11456] [Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-Constraint](https://arxiv.org/abs/2312.11456)
- [source:arxiv:2312.12065] [PPO-Clip Attains Global Optimality: Towards Deeper Understandings of Clipping](https://arxiv.org/abs/2312.12065)
- [source:arxiv:2502.13177] [KL Penalty Control via Perturbation for Direct Preference Optimization](https://arxiv.org/abs/2502.13177)
