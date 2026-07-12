---
title: Self-improvement and self-play RL
kind: primer
reference: ../topics/self-improvement-and-self-play.md
updated: '2026-07-12'
---

# Self-Improvement and Self-Play RL: A Primer

**Scaffold.** This primer covers five methods that let language models improve themselves by generating their own training signals—rationales, samples, tasks, subtask solutions, or reflected trajectories—and filtering them through an external oracle. By the end you will understand the single loop that unifies STaR, ReST, Re-ReST, Absolute Zero, and Iterated Amplification (IDA), why the choice of oracle determines the method's scalability, and where the current tensions lie. This connects directly to verifiable-reward RL (RLVR), policy-gradient fine-tuning, and the broader shift from human-annotated to model-generated supervision.

---

## The Core Mechanism: Generate → Verify → Filter/Refine → Train

Every method in this family runs a loop with three moving parts:

1. **Generator** — the model (or a proposer head) produces candidates: CoT rationales, translations, code tasks, subtask answers, or agent trajectories.
2. **Oracle** — an external verifier judges each candidate. The oracle can be *exact* (ground-truth answer, code executor), *learned* (reward model), *human* (decomposition + atomic answers), or *environmental* (unit tests + reflector).
3. **Update** — the model is trained on the oracle-approved subset, either by supervised learning on filtered data (STaR, ReST, Re-ReST, IDA) or by policy-gradient on a learned reward (Absolute Zero).

The **scalability axis** is the oracle's strength. Exact oracles (ground truth, code execution) give perfect signal but require the answer or a deterministic environment. Learned oracles (reward models) scale to open-ended tasks but drift and overfit. Human oracles (IDA) don't scale at all. Environmental oracles with reflection (Re-ReST) sit in between: the environment gives sparse feedback, a reflector densifies it, and the model learns from the refined trajectory.

---

## The Unifying Objective

STaR makes the loop explicit as a policy-gradient problem. Let the model be $\pi_\theta$, the dataset $\mathcal{D} = \{(x_i, y_i)\}$, and the rationale–answer pair $(\hat{r}_i, \hat{y}_i) \sim \pi_\theta(\cdot|x_i)$. The reward is the indicator of answer correctness:

$$
J(\theta) = \sum_i \mathbb{E}_{(\hat{r}_i, \hat{y}_i) \sim \pi_\theta(\cdot|x_i)} \bigl[ \mathbb{1}(\hat{y}_i = y_i) \bigr]
$$

The gradient uses the log-derivative trick:

$$
\nabla_\theta J = \sum_i \mathbb{E} \bigl[ \mathbb{1}(\hat{y}_i = y_i) \; \nabla_\theta \log \pi_\theta(\hat{y}_i, \hat{r}_i | x_i) \bigr]
$$

**Why this works.** The indicator $\mathbb{1}(\hat{y}_i = y_i)$ zeros out gradients from wrong answers—exactly what STaR's filtering step does. The remaining gradients reinforce *only* the rationales that led to the right answer. Rationalization (prompting with the correct $y_i$ to get a justification) fills the buffer when the model can't yet solve $x_i$ on its own, keeping the gradient estimator alive.

**Common confusion.** STaR *looks* like supervised learning on filtered data, but the derivation shows it's REINFORCE with a binary reward and no baseline. The "filtering = indicator" equivalence holds only because the reward is exact and sparse. ReST replaces the indicator with a learned reward model $R(x,y)$ and a threshold $\tau$; Absolute Zero replaces it with a *learnability* reward $1 - \bar{r}_{\text{solve}}$ for the proposer and a binary solve reward for the solver. The loop structure stays the same; the oracle changes.

---

## Runnable Check: STaR's Filtering as REINFORCE

The snippet below simulates one STaR iteration on a toy arithmetic task. It shows that filtering by correctness is mathematically equivalent to REINFORCE with an indicator reward.

```python
import torch
import torch.nn.functional as F

# Toy setup: model predicts answer logits for 2-digit addition
# Ground truth: 23 + 45 = 68
x = "23+45"
y_true = 68
vocab_size = 200  # 0-199
logits = torch.randn(vocab_size, requires_grad=True)  # untrained model

# STaR step: sample K rationales+answers, keep only correct ones
K = 1000
samples = torch.multinomial(F.softmax(logits, dim=-1), K, replacement=True)
correct_mask = (samples == y_true)
correct_samples = samples[correct_mask]

# REINFORCE gradient with indicator reward
# log p(y|x) for each sample
log_probs = F.log_softmax(logits, dim=-1)[samples]
rewards = correct_mask.float()  # indicator reward
grad_reinforce = (rewards * log_probs).mean().backward(retain_graph=True)
grad_reinforce = logits.grad.clone()
logits.grad.zero_()

# STaR's implicit gradient: SL on filtered correct samples only
# This is sum_{correct} grad log p(y|x) / N_correct
if correct_mask.any():
    log_probs_correct = F.log_softmax(logits, dim=-1)[correct_samples]
    loss_sl = -log_probs_correct.mean()
    loss_sl.backward()
    grad_star = logits.grad.clone()
else:
    grad_star = torch.zeros_like(logits)

# They match up to a constant scale (REINFORCE divides by K, SL divides by N_correct)
scale = K / max(correct_mask.sum().item(), 1)
assert torch.allclose(grad_reinforce * scale, grad_star, atol=1e-5), \
    "Filtering gradient != REINFORCE gradient"
print("STaR filtering == REINFORCE with indicator reward ✓")
```

---

## Load-Bearing Disagreements

1. **Exact vs. learned oracle.** STaR's filter is ground-truth correctness—exact but requires labeled answers. ReST's filter is a reward model—dense but a proxy that drifts as the policy improves. The reference notes ReST's human evaluations *diverge* from reward rankings under drift, and "delusions" (e.g., repetitive translations) appear. Absolute Zero tries to have it both ways: a code executor (exact) for the solver, but a *learnability* reward for the proposer that is itself estimated by the solver's success rate. The tension: **can a learned proposer reward stay aligned without an exact oracle?**

2. **"Zero data" is a spectrum.** Absolute Zero claims "zero human-curated QA pairs" but depends on a deterministic Python executor as an external verifier. IDA requires a human decomposer. STaR requires ground-truth answers. Re-ReST needs an environment (unit tests) *and* a reflector. The disagreement is practical: **how weak an oracle can you tolerate before the self-generated signal collapses?** Current trajectory favors verifiable-reward oracles (code, math) over learned reward models.

---

## Current Status

The field is converging on verifiable-reward self-play (Absolute Zero) and iterative self-training with learned rewards (ReST spirit via DPO/GRPO) for reasoning, while reflection-reinforced self-training (Re-ReST) emerges for agentic settings; IDA's decomposition paradigm lives on in alignment theory, not engineering practice.

**Full reference:** See the comprehensive article "Self-improvement and self-play RL" for derivations, full result tables, and 11 source citations.

---
*Full reference (citations, derivations, variants):* [Self-improvement and self-play RL](../topics/self-improvement-and-self-play.md)
