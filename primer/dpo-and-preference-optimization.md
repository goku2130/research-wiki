---
title: Direct Preference Optimization and variants
kind: primer
reference: ../topics/dpo-and-preference-optimization.md
updated: '2026-07-12'
---

# Direct Preference Optimization and Variants: A Primer

**Scaffold.** Direct Preference Optimization (DPO) and its descendants (IPO, KTO, AlphaDPO) replace the three-model, online RL loop of classic RLHF with a single supervised loss on static preference pairs. By the end of this note you will understand: (1) how the Bradley–Terry model plus a KL constraint yields a closed-form optimal policy that makes the reward *implicit*; (2) why DPO's loss is just binary cross-entropy on log-ratio differences; (3) how IPO bounds the reward gap, KTO learns from unary labels, and AlphaDPO adapts the reference distribution; and (4) where the offline paradigm hits its ceiling. This connects to PPO/RLHF (the method it replaces), reward modeling (the component it absorbs), and the broader debate on offline vs. online alignment for reasoning.

---

## The Core Mechanism: From RLHF to a Classification Loss

Classic RLHF optimizes  

$$
\max_\pi \mathbb{E}_{x,y\sim\pi}[r_\phi(x,y)] - \beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
$$

with three live models (policy, reward model, value network) and an online PPO loop. The compute is 3–5× SFT; memory requires four model copies.

DPO's insight: under the Bradley–Terry preference model $P(y_w\succ y_l|x)=\sigma(r(x,y_w)-r(x,y_l))$, the KL-constrained optimum has a **closed form**:

$$
\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\mathrm{ref}}(y|x)\exp\!\Big(\frac{1}{\beta}r(x,y)\Big)
$$

Invert this to express the reward *implicitly* as a log-ratio against the reference policy:

$$
r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta\log Z(x)
$$

The partition function $Z(x)$ cancels when we take the difference between a chosen ($y_w$) and rejected ($y_l$) response. Substitute into the Bradley–Terry log-likelihood and the reward model disappears entirely. What remains is a loss depending **only** on the trainable policy $\pi_\theta$ and the frozen reference $\pi_{\mathrm{ref}}$:

$$
\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}_{(x,y_w,y_l)}\Big[ \log\sigma\Big( \beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \Big) \Big]
$$

**Why this works.** The term $\beta\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$ *is* the implicit reward $\hat{r}_\theta(x,y)$. The loss asks: "is the implicit reward of the chosen response higher than that of the rejected one?" It is exactly binary cross-entropy on the margin of log-ratios. No reward model, no value network, no online sampling — just a frozen reference (usually the SFT checkpoint) and a static dataset of $(x,y_w,y_l)$ triplets. Training cost: 1.5–2× SFT; memory: two models.

**Common confusion pre-empted.** DPO does *not* "avoid reward modeling." It performs reward modeling *inside the policy* via the log-ratio. The reference model anchors the scale; without it, the log-ratio is meaningless.

---

## Runnable Check: DPO Loss in 12 Lines

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_logps_chosen, policy_logps_rejected,
             ref_logps_chosen, ref_logps_rejected, beta=0.1):
    """
    policy_logps_*: log-probs from current policy, shape (batch,)
    ref_logps_*:    log-probs from frozen reference, shape (batch,)
    Returns scalar loss.
    """
    # Implicit rewards: beta * (log pi_theta - log pi_ref)
    r_chosen = beta * (policy_logps_chosen - ref_logps_chosen)
    r_rejected = beta * (policy_logps_rejected - ref_logps_rejected)
    # Bradley-Terry margin: log sigma(r_chosen - r_rejected)
    margin = r_chosen - r_rejected
    loss = -F.logsigmoid(margin).mean()
    return loss

# Sanity checks
torch.manual_seed(0)
B = 4
pol_ch = torch.randn(B)
pol_rej = torch.randn(B)
ref_ch = torch.randn(B)
ref_rej = torch.randn(B)

loss = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=0.5)
assert loss.item() > 0, "Loss should be positive"

# If policy matches reference exactly, margin = 0 -> loss = log(2)
loss_ref = dpo_loss(ref_ch, ref_rej, ref_ch, ref_rej, beta=0.5)
assert abs(loss_ref.item() - torch.log(torch.tensor(2.0)).item()) < 1e-4

# If chosen much better than rejected, loss -> 0
pol_ch_big = ref_ch + 10.0
pol_rej_small = ref_rej - 10.0
loss_perfect = dpo_loss(pol_ch_big, pol_rej_small, ref_ch, ref_rej, beta=0.5)
assert loss_perfect.item() < 0.01, "Near-perfect ordering should give near-zero loss"

print("All checks passed. Loss at reference:", loss_ref.item(), "| Perfect:", loss_perfect.item())
```

---

## The Variant Landscape

| Method | Core Change | Data Requirement | Key Equation |
|--------|-------------|------------------|--------------|
| **IPO** | Squared-error loss targeting margin $1/(2\beta)$ | Paired $(y_w,y_l)$ | $\mathcal{L}_{\mathrm{IPO}} = \mathbb{E}[(h_\theta - 1/(2\beta))^2]$ |
| **KTO** | Prospect-theory loss with batch-wise centering | Unary (desirable/undesirable) | $r_\theta(x,y) = \beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}} - \mathbb{E}_{y'}[\beta\log\frac{\pi_\theta}{\pi_{\mathrm{ref}}}]$ |
| **AlphaDPO** | Adaptive implicit reference $\hat{\pi}_{\mathrm{ref}} \propto U(\pi_\theta/\pi_{\mathrm{ref}})^\alpha$ | Paired | Instance-adaptive margin via $\alpha$ |

### IPO: Bounding the Reward Gap
DPO's sigmoid loss saturates: once a pair is correctly ordered, the gradient vanishes, but the implicit reward gap keeps growing (overfitting). IPO replaces the monotonic loss with a **squared-error regression** to a fixed target margin $1/(2\beta)$. The gradient $2(h_\theta - 1/(2\beta))$ is zero *at* the target, creating a stable equilibrium — too small pushes up, too large pushes down. This prevents the extreme probabilities (chosen→1, rejected→0) that DPO encourages. In bandit experiments, IPO avoids greedy collapse, preserves actions that never win, and gracefully handles unobserved pairs — all failure modes of DPO. The target margin derives from the KL coefficient: stronger KL penalty (smaller $\beta$) → larger permitted margin.

### KTO: Learning from Thumbs-Up/Down
KTO drops the paired-comparison requirement entirely. It centers the implicit reward by a batch-wise baseline (approximating $Z(x)$), then applies a **prospect-theory value function** with loss aversion ($\lambda$) and diminishing sensitivity ($\beta$). The loss is $\mathbb{E}_{y_w}[\sigma(-r_\theta)] + \mathbb{E}_{y_l}[\sigma(r_\theta)]$ where $r_\theta$ is the centered log-ratio. This enables alignment from unary feedback — critical for real-world deployment where paired comparisons are expensive. KTO matches or exceeds DPO at 1B–30B scales, gains 13.5 points on GSM8K over DPO, and for models ≥13B can skip SFT entirely. Trade-off: highly learning-rate sensitive (needs 2–10× DPO's LR) and risks underfitting on exceptionally clean data.

### AlphaDPO: Adaptive Reference, Adaptive Margin
AlphaDPO introduces an *implicit* reference that evolves with the policy:

$$
\hat{\pi}_{\mathrm{ref}} \propto U(y|x) \left(\frac{\pi_\theta}{\pi_{\mathrm{ref}}}\right)^\alpha
$$

$\alpha=0$ gives uniform exploration; $\alpha=1$ recovers the static reference. This yields **instance-adaptive margins** (unlike SimPO's uniform margin) and theoretically controls sequential KL divergence between updates, stabilizing training even with poorly calibrated initial references. Reported: 58.7% LC win rate on AlpacaEval 2, 35.7% on Arena-Hard across Mistral2-7B, Llama3-8B, Gemma2-9B.

### ORPO: Undocumented in Sources
The provided literature mentions ORPO by name but contains no mechanism, loss, or results. A source labeled "ORPO Explained" actually describes DeepSeek's GRPO. **ORPO cannot be characterized from the given references.**

---

## Load-Bearing Disagreements

1. **DPO vs. PPO with ground-truth rewards.** The DPO paper reports DPO *beating* PPO on sentiment generation even when PPO has access to the *true* reward function (6B scale). This contradicts the standard intuition that online RL with a perfect reward should dominate offline methods. Whether this holds at frontier scale is untested in the sources.

2. **Prospect theory vs. Bradley–Terry.** KTO explicitly rejects the Bradley–Terry assumption (consistent utility maximization), arguing human preferences follow loss aversion and diminishing sensitivity. DPO/IPO/AlphaDPO all inherit BT. This is a fundamental theoretical split about what human feedback *is*.

3. **Offline ceiling vs. hybrid future.** All characterized methods are offline — bounded by dataset coverage. The AlphaDPO paper claims "robust alignment without multi-stage training," yet remains offline. The field is converging on hybrid pipelines: offline DPO/IPO for initial alignment, then online RL (PPO/GRPO) for reasoning where verifiable rewards exist. Pure offline is not fading but recognized as insufficient for advanced reasoning.

---

**Current status.** DPO/IPO/KTO are the default offline alignment stack in open-source LLM post-training (Llama 3, Mistral, Gemma); AlphaDPO (ICML 2024) shows strong benchmarks but limited production adoption; ORPO remains undocumented in the provided literature.

**Full reference.** See the comprehensive fact-checked article "Direct Preference Optimization and variants" for derivations, scaling data, bandit experiments, and the complete reference list including arXiv:2305.18290 (DPO), arXiv:2310.12036 (IPO), arXiv:2402.01306 (KTO), ICML:AlphaDPO, and associated commentaries.

---
*Full reference (citations, derivations, variants):* [Direct Preference Optimization and variants](../topics/dpo-and-preference-optimization.md)
