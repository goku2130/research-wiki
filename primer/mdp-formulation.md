---
title: MDP formulation of LLM generation
kind: primer
reference: ../topics/mdp-formulation.md
updated: '2026-07-12'
---

# MDP Formulation of LLM Generation: Why Token-Level RL Collapses to Bandits (and When It Doesn't)

**Scaffold.** LLM generation is sequential: each token conditions the next. It's tempting to model this as a token-level Markov Decision Process (MDP) — state = prompt + history, action = next token, reward = outcome. But under the standard post-training setup (outcome supervision, terminal reward only), this MDP *mathematically degenerates* into a contextual bandit where the "action" is the entire completion. This primer teaches why the collapse happens, what it implies for algorithm design (GRPO ≈ filtered SFT), and the two escape hatches that keep token-level MDPs alive: process supervision and learned dense rewards (RTO).

---

## The Token-Level MDP — And Why It Degenerates

**The canonical setup.** At step $t$, the state $s_t = (q, o_{<t})$ concatenates the prompt $q$ and all previously generated tokens. The action $a_t = o_t$ picks the next token from the vocabulary. Transitions are deterministic: $s_{t+1} = (s_t, a_t)$. The episode ends at $\langle \text{eos} \rangle$. The policy $\pi_\theta(a_t|s_t)$ is exactly the LLM's conditional distribution.

**The reward structure breaks the MDP.** In outcome-supervised RL (RLHF, RLVR, GRPO), a scalar reward $R(\tau)$ arrives *only* at the terminal state. All intermediate rewards are zero: $r_t = 0$ for $t < T$. With discount $\gamma=1$ (standard), the return-to-go from *any* state in a trajectory is identical — it's just the terminal reward $R$.

**The collapse.** Because (1) the state encodes the full action history, and (2) the advantage is uniform across tokens ($\hat{A}_{i,t} = \hat{A}_i$), the policy gradient weights every token's log-probability by the *same* scalar. The gradient becomes:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \sum_{t=1}^T \nabla_\theta \log \pi_\theta(o_t | q, o_{<t}) \right] = \mathbb{E}_{o_{1:T} \sim \pi_\theta(\cdot|q)} \left[ R(q, o_{1:T}) \nabla_\theta \log \pi_\theta(o_{1:T} | q) \right]
$$

This is *exactly* the contextual bandit gradient: context $q$, action = full completion $o_{1:T}$, reward $R$. The "MDP" machinery (value functions, GAE, per-step credit assignment) is inert — the bandit formulation isn't an approximation; it's the *effective* formulation.

> **Intuition check.** If every token in a response gets the same credit/blame, you're not doing temporal credit assignment — you're just up/down-weighting whole sequences. That's a bandit.

---

## GRPO = Filtered Iterative SFT (F-ISFT)

The degeneration isn't just theoretical. [arXiv:2505.13697] proves that under the standard assumptions (negligible KL, clipping inactive, uniform advantage), the GRPO objective simplifies to a weighted maximum-likelihood update on *filtered* model generations:

$$
\mathcal{J}(\theta) \propto \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{\hat{A}_i}{|o_i|} \sum_{t=1}^{|o_i|} \frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})} \right]
$$

Splitting responses into positive ($\mathcal{G}^+$) and negative ($\mathcal{G}^-$) advantage sets yields a gradient that **increases log-prob of good completions and decreases log-prob of bad ones**, weighted by advantage and inverse length. This is *Filtered Iterative Supervised Fine-Tuning* (F-ISFT). Empirically, F-ISFT matches GRPO across GSM8K, Countdown, and multiple model families (Qwen, Llama, DeepSeek-Math).

**Implication.** The "RL" in GRPO is largely nominal. The observed length bias (incorrect responses grow *much* longer) falls out of the $1/|o_i|$ scaling and training on negative samples — not improved reasoning.

---

## Two Escape Hatches: When the MDP Stays Non-Degenerate

### 1. Process Supervision (Per-Step Rewards)
If a Process Reward Model (PRM) scores intermediate reasoning steps, rewards $r_t$ become non-zero *before* termination. Return-to-go now varies across $t$, the value function learns non-trivial structure, GAE becomes active, and the token-level MDP regains its distinguishing power. This is the native formulation for PPO with PRMs.

### 2. Token-Wise Reward Learning (RTO)
[arXiv:2404.18922] revives the token-MDP for *generation* by extracting **dense per-token rewards** from a DPO-aligned policy. The insight: a DPO policy $\pi_{\text{dpo}}$ implicitly represents a reward function via the log-ratio against the reference $\pi_{\text{ref}}$. RTO uses this as a per-token reward for PPO:

$$
r_{\text{rto}}(x, y_{1:h}) = \beta_1 \log \frac{\pi_{\text{dpo}}(y_h|x,y_{<h})}{\pi_{\text{ref}}(y_h|x,y_{<h})} - \beta_2 \log \frac{\pi_\theta(y_h|x,y_{<h})}{\pi_{\text{ref}}(y_h|x,y_{<h})} \quad (h < H)
$$

with an optional sentence-level MLE term at the final token. On Llama-3-8B, RTO beats PPO by **7.5 pts** on AlpacaEval 2 and **4.1 pts** on Arena-Hard using **1/8 the data**. Theoretically, token-wise rewards reduce sample complexity from $A^H$ to $A^{\min\{\xi+1, H\}}$ (where $\xi$ is the "reward horizon").

> **Why this works.** RTO converts preference data (which is sequence-level) into a *token-level* reward signal that PPO can actually use for credit assignment. The MDP is no longer degenerate because $r_t \neq 0$ and varies with $t$.

---

## Token-Level MDPs That *Don't* Collapse: Input Refinement (RTLIR)

Not all token-MDPs are for generation. **RTLIR** [AAAI 2025] formulates *input refinement* as a genuine token-MDP:
- **State**: embeddings of input tokens seen so far + keep/delete decisions
- **Action**: binary keep/delete per input token (single left-to-right pass)
- **Rewards**: immediate cosine similarity to target + terminal log-prob ratio
- **Learning**: Q-learning with value decomposition + prioritized replay

Here the horizon is fixed (input length), credits are non-uniform (immediate + terminal), and the policy decides *which tokens to keep* — a bona fide token-level MDP with no bandit equivalent.

---

## Runnable Check: Bandit Gradient = Collapsed MDP Gradient

```python
import torch
import torch.nn.functional as F

# Toy setup: vocab=3, seq_len=4, batch=2
vocab, T, B = 3, 4, 2
logits = torch.randn(B, T, vocab, requires_grad=True)  # policy logits
pi = F.softmax(logits, dim=-1)
tokens = torch.randint(0, vocab, (B, T))                # sampled completions
rewards = torch.tensor([1.0, -0.5])                     # terminal rewards per sequence

# --- Bandit gradient: (R - b) * grad log pi(sequence) ---
log_pi_seq = pi.gather(-1, tokens.unsqueeze(-1)).squeeze(-1).log().sum(dim=1)  # [B]
baseline = rewards.mean()
bandit_grad = ((rewards - baseline).unsqueeze(1) * torch.autograd.grad(log_pi_seq.sum(), logits, retain_graph=True)[0])

# --- Collapsed MDP gradient: sum_t R * grad log pi(token_t) ---
mdp_grad = torch.zeros_like(logits)
for t in range(T):
    log_pi_t = pi[:, t].gather(-1, tokens[:, t:t+1]).squeeze(-1).log()  # [B]
    mdp_grad[:, t] = torch.autograd.grad((rewards * log_pi_t).sum(), logits, retain_graph=True)[0][:, t]

# They match (up to numerical noise)
assert torch.allclose(bandit_grad, mdp_grad, atol=1e-5), "Gradients diverge — collapse fails"
print("✓ Bandit gradient == Collapsed MDP gradient (uniform advantage)")
```

---

## Load-Bearing Disagreements

1. **MDP vs. Bandit as *design choices* vs. *mathematical necessity*.** Some sources (LinkedIn, X posts) present token-MDP (PPO) and bandit (REINFORCE/RLOO) as distinct, co-existing formulations with similar empirical performance. [arXiv:2505.13697] contradicts this: the MDP *as implemented* (terminal reward + uniform advantage) *is* a bandit — the similarity isn't coincidence, it's identity. The bandit formulation isn't an alternative; it's the correct abstraction for outcome supervision.

2. **Is the KL penalty in GRPO load-bearing?** The F-ISFT derivation assumes the KL term is negligible due to clipping. If KL regularization is strong (e.g., large $\beta$), the bandit equivalence breaks — the MDP view retains a per-step constraint that the bandit gradient doesn't capture. This matters for stability but not for the core credit-assignment argument.

---

## Current Status
The token-level MDP for *generation* is fading as a distinct formulation for outcome-supervised post-training; the community converges on the bandit view (RLOO, REINFORCE, DPO-as-bandit). Process supervision (PRMs) and token-wise reward learning (RTO) keep token-MDPs alive for generation. Token-MDPs thrive natively in non-generation tasks (RTLIR for input refinement, tool-use, reasoning-step RL). MDL-based state representation remains a research curiosity.

**Full reference:** See the comprehensive article "MDP formulation of LLM generation" for derivations, extended tables, and all citations.

---
*Full reference (citations, derivations, variants):* [MDP formulation of LLM generation](../topics/mdp-formulation.md)
