---
title: Rejection sampling and Best-of-N
kind: primer
reference: ../topics/rejection-sampling-and-bon.md
updated: '2026-07-12'
---

# Rejection Sampling and Best-of-N: A Primer

**Scaffold.** This primer covers two generate-and-score paradigms that share a name but serve different stages of the LLM lifecycle. **Best-of-N (BoN)** is an *inference-time* alignment method: sample $N$ candidates from a frozen policy, score them with a reward model $\hat{r}$, and return the highest-scoring one. **Rejection Sampling (RS)** is a *training-time* data curation pipeline: generate $N$ completions per prompt, score them, and keep the top-$K$ as a supervised fine-tuning (SFT) dataset. Both are simple, require no policy-agnostic wrappers, yet both hit a hard ceiling: **no inference-time algorithm can overcome a fundamentally inaccurate reward model**. By the end you will understand why vanilla BoN overoptimizes, how regularization restores monotonic scaling, how compute-efficient variants simulate large $N$ on limited hardware, and why training for BoN deployment changes the objective.

---

## The Core Mechanism: Inference-Time Best-of-N

**Intuition.** Imagine you have a decent but imperfect language model $\pi_{\text{ref}}$ and a reward model $\hat{r}$ that correlates with human preference. Instead of deploying $\pi_{\text{ref}}$ directly, you spend extra compute at test time: generate $N$ independent answers, let $\hat{r}$ rank them, and serve the winner. This is *rejection sampling at inference time*—you "reject" $N-1$ samples implicitly by keeping only the best.

**The algorithm.** For a prompt $x$:
1. Draw $y_1, \dots, y_N \stackrel{\text{i.i.d.}}{\sim} \pi_{\text{ref}}(\cdot \mid x)$.
2. Compute scores $\hat{r}(x, y_i)$.
3. Return $y^* = \arg\max_i \hat{r}(x, y_i)$.

**Why it works (and why it eventually breaks).** If $\hat{r}$ were the true reward $r^*$, BoN would monotonically improve with $N$: the maximum of $N$ i.i.d. draws from any distribution stochastically increases with $N$. But $\hat{r}$ has error. As $N$ grows, the selected $y^*$ exploits regions where $\hat{r}$ overestimates $r^*$—**reward hacking**. Empirically, win-rates peak at $N=16$–$32$ then decline.

**The fundamental limit.** [source:icml:is-best-of-n-the-best-of-them-coverage-s] formalizes this via two quantities:
- **Coverage** $\mathcal{C}^{\pi^*}(x) := \mathbb{E}_{y \sim \pi^*} \left[ \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$ — how well the base policy can generate high-quality responses.
- **Reward model error** $\varepsilon_{\text{RM}}^2(x) := \mathbb{E}_{y \sim \pi_{\text{ref}}} \left[ (\hat{r}(x, y) - r^*(x, y))^2 \right]$ — MSE of $\hat{r}$ under the base policy.

They prove BoN's regret scales as $\varepsilon_{\text{RM}}^{2/3}(x)$, while the information-theoretic optimum is $\sqrt{\mathcal{C}^{\pi^*}(x) \cdot \varepsilon_{\text{RM}}^2(x)}$. The gap vanishes only for binary rewards (e.g., final-answer correctness), where BoN is near-optimal.

---

## Regularization: Restoring Monotonic Scaling

Two load-bearing approaches prevent the quality peak-and-decline.

### MBR-BoN: Wasserstein Proximity Regularization

**Idea.** Penalize selections that drift far from the "center of mass" of the $N$ samples. This keeps the output in regions where $\pi_{\text{ref}}$ has density and $\hat{r}$ is better calibrated.

**Objective.** Given samples $Y_{\text{ref}} \sim \pi_{\text{ref}}$, select:

$$
y_{\text{MBR-BoN}} = \arg\max_{y \in Y_{\text{ref}}} \hat{r}(x, y) + \beta \frac{1}{N} \sum_{y' \in Y_{\text{ref}}} \cos(\text{emb}(y), \text{emb}(y'))
$$

where $\text{emb}$ uses `all-mpnet-base-v2` and $\beta$ trades off reward vs. proximity. The authors show this minimizes the Wasserstein distance between the output distribution and the empirical reference distribution—arguing Wasserstein is more robust than KL for inference-time regularization because KL requires exponentially many samples to estimate and is oversensitive to minor textual variations.

**Practical note.** Optimal $\beta$ varies widely across reward models (e.g., $0.5$ for SHP-Large/XL vs. $20.0$ for OASST) but is findable with $\sim 10$ development instances. Quadratic $O(N^2)$ utility computation adds $\sim 2$s for $N=128$ on T4.

### InferenceTimePessimism: $\chi^2$-Regularized Rejection Sampling

**Idea.** Implement a theoretically grounded pessimism penalty via a novel rejection sampling scheme that *decouples* the computational budget $N$ from the penalty strength $\beta$. This achieves **scaling-monotonicity**: performance never degrades as $N \to \infty$.

**Algorithm** (simplified):
1. Draw $N$ responses $\widehat{\mathcal{Y}}_N \sim \pi_{\text{ref}}(\cdot \mid x)$.
2. Find $\widehat{\lambda}(x)$ such that $\frac{1}{N} \sum_{y \in \widehat{\mathcal{Y}}_N} \text{relu}\left(\beta^{-1} (\hat{r}(x, y) - \widehat{\lambda}(x))\right) = 1$.
3. Define weights $w(y \mid x) := \text{relu}\left(\beta^{-1} (\hat{r}(x, y) - \widehat{\lambda}(x))\right)$.
4. Draw another $N$ responses; accept $y_i$ with probability $\min\{w(y_i \mid x) / M, 1\}$ where $M = (R_{\max} - \widehat{\lambda}(x)) / \beta$.
5. Return first accepted sample; if none, return a fresh $(N+1)$-th sample.

**Regret bound** (achieves the "skyline" optimum):

$$
J(\pi^{*}; x) - J(\widehat{\pi}_{\text{Pes}}; x) \lesssim \sqrt{\mathcal{C}^{\pi^{*}}(x) \cdot \varepsilon_{\mathrm{RM}}^{2}(x)} + \beta \cdot \mathcal{C}^{\pi^{*}}(x) + \beta^{-1} \cdot \varepsilon_{\mathrm{RM}}^{2}(x)
$$

Setting $\beta \asymp \sqrt{\varepsilon_{\mathrm{RM}}^{2}(x) / \mathcal{C}^{\pi^{*}}(x)}$ yields optimal regret $\sqrt{\mathcal{C}^{\pi^{*}}(x) \cdot \varepsilon_{\mathrm{RM}}^{2}(x)}$.

**Empirical contrast.** On GSM8K/MATH with Phi-3-Mini, vanilla BoN peaks then declines; InferenceTimePessimism improves monotonically with $N$. For binary-reward tasks the gap to BoN narrows.

---

## Compute Efficiency: Simulating Large $N$ on Limited Hardware

Three complementary strategies make large-$N$ BoN practical.

| Method | Core Idea | Compute Reduction |
|--------|-----------|-------------------|
| **Speculative Rejection** | Generate tokens in batches; score partial completions; reject bottom $\alpha$-quantile early; iterate until completion. | Matches BoN rewards requiring 16–32 GPUs on **one GPU** (AlpacaFarm, Llama-3-8B). |
| **TreeBoN** | Tree-search: generate $N$ root segments, iteratively prune to top parents, branch $N_{\text{children}}$ children per parent. Uses DPO implicit reward $r_{\text{partial}} = \sum_k w_k \log \frac{\pi^*(y_k \mid \cdot)}{\pi(y_k \mid \cdot)}$ for partial scoring (traditional RMs are "chaotic" on partial text). | 6.3% of BoN compute ($N=8$ roots vs. $128$) maintains 55% win rate; 9% pass@1 gain on GSM8K. |
| **AdaBoN** | Exploratory phase estimates per-prompt reward distribution; allocation phase gives more samples to prompts with higher marginal returns. | Outperforms uniform allocation at same budget; competitive with 20% larger uniform budget. |

**STARS** takes a different angle: **fixed-horizon synchronous verification**. Instead of dynamic uncertainty-based segmentation (which causes stragglers and miscalibrated confident hallucinations), STARS generates exactly $K$ tokens for all batch requests, pauses *simultaneously* for parallel RM scoring, rejects/rewinds low-scoring trajectories, and repeats. This bounds "compute-at-risk" to $K$ tokens per hallucination and eliminates pipeline bubbles. On HH-RLHF (batch 64), STARS ($K=15$) achieves **185 T/s** (53.5% over CARDS) with only 15.0 tokens/rejection waste vs. CARDS' 45.2. Trade-off: slight alignment quality drop (Llama-7B: 60.2% vs. CARDS 64.5% Win-Tie).

---

## Training-Time Rejection Sampling: Data Curation for SFT

Distinct from inference-time BoN, **RS as a fine-tuning method** builds the SFT dataset:
1. For each of $M$ prompts, generate $N$ completions ($N=10$–$30$, temp $0.7$–$1.0$).
2. Score with $\hat{r}$.
3. Select either (a) top-1 per prompt: $y_{i, \arg\max_j r_{i,j}}$, or (b) top-$K$ overall pairs from the flattened $M \times N$ matrix.
4. Train SFT on selected $(x, y)$ pairs.

Used in Llama 2 Chat and WebGPT. Limitations: heavy dependence on $\hat{r}$ quality; reusing SFT prompts risks overfitting; throughput improved by length-sorted batching for reward inference.

---

## Inference-Aware Training: Optimize for How You Test

If you will deploy with BoN, standard SFT/RL (optimizing single-response quality) is suboptimal. [source:arxiv:2412.15287] derives objectives targeting the BoN policy:

$$
\pi_{\text{bon}}(y \mid x) \propto \pi(y \mid x) \exp(\lambda_N Q_\pi(x, y)), \quad Q_\pi(x, y) = \mathbb{E}_{y' \sim \pi}[\mathbf{1}_{r(x,y) \ge r(x,y')}]
$$

- **BoN-SFT**: Variational approximation maximizing expert likelihood under $\pi_{\text{bon}}$.
- **BoN-RL**: REINFORCE-style gradient sampling from $\pi_{\text{bon}}$.
- **BoN-RLB** (binary rewards): Closed-form gradient with asymmetric weights for "hard" examples:

$$
g_N^+(p) = \frac{N p^{N-1}}{1 - p^N}, \quad g^-(p) = \frac{N p}{1 - p}, \quad p = P_{\text{fail}}(x)
$$

  Positive-only variant BoN-RLB(P) uses $\bar{g}_N^+(p) = \frac{N p^{N-1}(1-p)}{1-p^N}$ when negative data is unavailable.

**Results** (Gemma 2B/9B): BoN-RL-V improves MATH Bo32 from 26.8% → 30.8%; BoN-RL-S improves pass@32 from 60.0% → 67.0%; BoN-RLB(P) improves HumanEval pass@16 from 61.6% → 67.1%. Gains generalize to held-out benchmarks. Caveats: verifier noise causes Type II errors at high $T$/large $N$; generating $N$ samples per gradient step is expensive (mitigated by BoN Distillation); weights explode for hard problems ($p \to 1$), requiring clipping.

---

## Runnable Check: Vanilla BoN vs. MBR-BoN Selection

```python
import numpy as np

def vanilla_bon(rewards: np.ndarray) -> int:
    """Return index of max reward."""
    return int(np.argmax(rewards))

def mbr_bon(rewards: np.ndarray, embeddings: np.ndarray, beta: float) -> int:
    """
    MBR-BoN selection: reward + beta * mean cosine similarity to other samples.
    embeddings: (N, d) normalized to unit length.
    """
    N = rewards.shape[0]
    # Cosine similarity matrix (N, N)
    sim = embeddings @ embeddings.T  # already normalized
    # Mean similarity to *other* samples (exclude self)
    mean_sim = (sim.sum(axis=1) - 1.0) / (N - 1)
    scores = rewards + beta * mean_sim
    return int(np.argmax(scores))

# Synthetic test: 8 samples, reward model overestimates sample 0
np.random.seed(0)
N = 8
true_quality = np.array([0.5, 0.7, 0.6, 0.8, 0.4, 0.9, 0.3, 0.65])
noise = np.random.normal(0, 0.15, N)
rewards = true_quality + noise  # reward model with error

# Embeddings: sample 0 is an outlier (low similarity to others)
emb = np.random.randn(N, 32)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
emb[0] = -emb[1:]  # make sample 0 dissimilar to cluster

bon_idx = vanilla_bon(rewards)
mbr_idx = mbr_bon(rewards, emb, beta=1.0)

print(f"Vanilla BoN picks: {bon_idx} (true quality: {true_quality[bon_idx]:.2f}, reward: {rewards[bon_idx]:.2f})")
print(f"MBR-BoN picks:   {mbr_idx} (true quality: {true_quality[mbr_idx]:.2f}, reward: {rewards[mbr_idx]:.2f})")

# MBR should avoid the overestimated outlier (sample 0) if beta is strong enough
assert mbr_idx != 0 or rewards[0] < rewards.max(), "MBR-BoN should not blindly pick the reward-hacked sample"
```

**What this shows.** Vanilla BoN chases the highest $\hat{r}$ even when it's an outlier (reward hacking). MBR-BoN's proximity term pulls selection toward the cluster of mutually similar samples, where $\hat{r}$ is better calibrated.

---

## Load-Bearing Disagreements / Caveats

1. **MBR-BoN vs. InferenceTimePessimism: proximity vs. pessimism.** MBR-BoN regularizes toward the *empirical sample center* (Wasserstein); InferenceTimePessimism regularizes via a *theoretically derived $\chi^2$ penalty* that is provably regret-optimal and scaling-monotonic. MBR-BoN is simpler (no second sampling pass, no $\lambda$ solve) but lacks formal guarantees; its $\beta$ is reward-model-sensitive. InferenceTimePessimism requires uncertainty estimation ($\varepsilon_{\text{RM}}$) and a second $N$-sample draw, but achieves the optimal regret skyline. **No head-to-head comparison exists on shared benchmarks.**

2. **Fixed-horizon vs. tree-search verification.** STARS eliminates stragglers and bounds hallucination risk with synchronous $K$-token heartbeats, but assumes RMs can score partial text reliably (they often can't). TreeBoN sidesteps this by using DPO implicit rewards for partial scoring, which are *more correlated* with final quality—but requires a DPO model and tree infrastructure. **How STARS' fixed horizon interacts with TreeBoN's tree structure is unexplored.**

---

## Current Status

BoN and its variants are **rising as the default inference-time alignment tool** for reasoning/coding (where verifiable rewards exist); speculative decoding (Speculative Rejection, TreeBoN, AdaBoN) solves the compute bottleneck; MBR-BoN and InferenceTimePessimism solve the overoptimization bottleneck; STARS solves the system-efficiency bottleneck; BoN-aware training (BoN-SFT/RL) is a promising but not yet widely adopted direction. All remain fundamentally capped by reward model accuracy.

**Full reference:** See the linked reference article for 13 source papers covering scaling laws, regularization theory, speculative decoding, tree search, synchronous verification, training-time RS, Monte Carlo foundations, and inference-aware fine-tuning.

---
*Full reference (citations, derivations, variants):* [Rejection sampling and Best-of-N](../topics/rejection-sampling-and-bon.md)
