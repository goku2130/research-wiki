---
title: Rejection sampling and Best-of-N
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2410.20290
- arxiv:2412.15287
- machinelearning:adabon-adaptive-best-of-n-alignment
- magazine:categories-of-inference-time-scaling-for
- aclanthology:regularized-best-of-n-sampling-with-mini
- rlhfbook:rejection-sampling-nathan-lambert-rlhf-b
- icml:is-best-of-n-the-best-of-them-coverage-s
- arxiv:2404.01054
- arxiv:2503.21878
- arxiv:2410.16033
- arxiv:2511.03827
- arxiv:2107.11820
- arxiv:1205.5494
open_questions:
- How do MBR-BoN, InferenceTimePessimism, TreeBoN, and STARS compare head-to-head
  on identical benchmarks and compute budgets?
- Is $\chi^2$ uncertainty estimation practical at scale for InferenceTimePessimism,
  or does it require prohibitively accurate reward model error estimates?
- Does BoN-aware training (BoN-SFT/RL) generalize beyond math/code to open-ended chat
  and creative tasks?
- Can STARS' fixed-horizon verification be combined with TreeBoN's tree search or
  DPO-based implicit rewards for better partial-sequence scoring?
---

Best-of-N (BoN) and rejection sampling (RS) are two distinct but related paradigms for aligning language models: BoN operates at inference time by generating multiple candidates and selecting the highest-scoring one under a reward model, while RS operates at training time by filtering model generations through a reward model to construct a high-quality supervised fine-tuning dataset. Both methods face a fundamental tension between exploitating a proxy reward model and avoiding overoptimization, driving a line of work on regularization, adaptive computation, and inference-aware training.

## Inference-time Best-of-N: mechanics and scaling laws

Standard Best-of-N (BoN) decoding samples $N$ independent responses $y_1, \dots, y_N \sim \pi_{\text{ref}}(\cdot \mid x)$ from a reference policy and returns $y^* = \arg\max_i \hat{r}(x, y_i)$, where $\hat{r}$ is a learned reward model [source:icml:is-best-of-n-the-best-of-them-coverage-s]. The method is simple, requires no weight updates, and is competitive with RLHF post-training for moderate $N$ [source:arxiv:2410.20290]. However, its computational cost scales linearly with $N$: generating $N$ full sequences requires either large batch sizes (risking OOM) or high latency, making $N > 1000$ impractical on typical hardware [source:arxiv:2410.20290].

The quality of BoN as a function of $N$ is not monotonic. Empirically, win-rates against a base model often peak at moderate $N$ (e.g., $N=16$–$32$) and then degrade due to **reward hacking**—the reward model $\hat{r}$ assigns high scores to outputs that exploit its errors rather than reflecting true quality $r^*$ [source:aclanthology:regularized-best-of-n-sampling-with-mini][source:icml:is-best-of-n-the-best-of-them-coverage-s]. Theoretically, [source:icml:is-best-of-n-the-best-of-them-coverage-s] proves that BoN achieves optimal regret only under stringent $L_\infty$-type coverage of the base policy over high-reward responses; under realistic $L_1$ coverage, BoN provably overoptimizes and fails to provide tight guarantees. The information-theoretic limit is governed by the mean-squared error of $\hat{r}$ and the base policy's coverage: no inference-time algorithm can overcome a fundamentally inaccurate reward model [source:icml:is-best-of-n-the-best-of-them-coverage-s].

### Coverage, reward error, and fundamental limits

[source:arxiv:2503.21878] formalizes the trade-off through two key quantities: **coverage** $\mathcal{C}^{\pi^*}(x) := \mathbb{E}_{y \sim \pi^*(\cdot|x)} \left[ \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right]$, measuring the base policy's ability to generate high-quality responses, and **reward model error** $\varepsilon_{\text{BH}}^2(x) := \mathbb{E}_{y \sim \pi_{\text{ref}}(\cdot|x)} \left[ (\hat{r}(x, y) - r^*(x, y))^2 \right]$, the expected squared error under the base policy. They prove that any sample-and-evaluate algorithm requires $N \gtrsim 1/\varepsilon_{\text{RM}}(x)$ queries to achieve optimal regret, and that BoN's regret scales as $\varepsilon_{\text{RM}}^{2/3}(x)$ rather than the optimal $\sqrt{\mathcal{C}^{\pi^*}(x) \cdot \varepsilon_{\text{RM}}^2(x)}$. For binary rewards (e.g., final answer correctness), the gap between $\mathcal{C}^{\pi^*}$ and uniform coverage vanishes, making BoN near-optimal in those specific regimes [source:arxiv:2503.21878].

## Regularized and robust BoN variants

### Minimum Bayes Risk regularization (MBR-BoN)

To mitigate reward hacking, [source:aclanthology:regularized-best-of-n-sampling-with-mini] proposes **MBR-BoN**, which adds a proximity regularizer encouraging the selected response to stay near the center of the sample distribution. Given $N$ samples $Y_{\text{ref}} \sim \pi_{\text{ref}}$, MBR-BoN selects:

$$
y_{\text{MBR-BoN}} = \arg\max_{y \in Y_{\text{ref}}} \hat{r}(x, y) + \beta \frac{1}{N} \sum_{y' \in Y_{\text{ref}}} U(y, y')
$$

where $U(y, y') = \cos(\text{emb}(y), \text{emb}(y'))$ is cosine similarity of embeddings (using `all-mpnet-base-v2`) and $\beta$ controls the trade-off [source:arxiv:2404.01054]. The authors show that maximizing this objective minimizes the **Wasserstein distance** between the output distribution and the empirical reference distribution, arguing that Wasserstein is more robust than KL for inference-time regularization because KL requires exponentially many samples to estimate reliably and is oversensitive to minor textual variations [source:aclanthology:regularized-best-of-n-sampling-with-mini][source:arxiv:2404.01054]. On AlpacaFarm and hh-rlhf with Mistral-7B-SFT-beta and Dolly-v2-3b, MBR-BoN consistently outperforms vanilla BoN and pure MBR across proxy reward models (SHP-Large, SHP-XL, OASST), with optimal $\beta$ varying widely (e.g., $0.5$ for SHP-Large/XL vs. $20.0$ for OASST) but findable with as few as 10 development instances [source:aclanthology:regularized-best-of-n-sampling-with-mini][source:arxiv:2404.01054]. MBR-BoN also improves downstream DPO training when used to generate preference pairs (chosen = MBR-BoN selection; rejected = lowest reward response) [source:aclanthology:regularized-best-of-n-sampling-with-mini][source:arxiv:2404.01054]. The quadratic $O(N^2)$ utility computation adds overhead (~2s for $N=128$ on T4) [source:aclanthology:regularized-best-of-n-sampling-with-mini].

### InferenceTimePessimism: $\chi^2$-regularized rejection sampling

[source:icml:is-best-of-n-the-best-of-them-coverage-s] introduces **InferenceTimePessimism**, a theoretically grounded alternative that implements $\chi^2$-regularization via a novel rejection sampling scheme at inference time. The algorithm decouples the computational budget $N$ from the pessimism penalty strength, achieving **scaling-monotonicity**: performance does not degrade as $N \to \infty$, unlike BoN [source:arxiv:2503.21878].

**Algorithm** [source:arxiv:2503.21878]:
1. Draw $N$ i.i.d. responses $\widehat{\mathcal{Y}}_{N} \sim \pi_{\text{ref}}(\cdot \mid x)$.
2. Compute normalization constant $\widehat{\lambda}(x)$ such that $\frac{1}{N} \sum_{y \in \widehat{\mathcal{Y}}_{N}} \text{relu} \left(\beta^{-1} (\widehat{r}(x, y) - \widehat{\lambda}(x))\right) = 1$.
3. Define importance weights $w(y \mid x) := \text{relu} \left(\beta^{-1} (\widehat{r}(x, y) - \widehat{\lambda}(x))\right)$.
4. Draw another set of $N$ responses; for each $y_i$, sample Bernoulli $\xi_i$ with probability $\min\left\{\frac{w(y_{i}|x)}{M}, 1\right\}$ where $M := \frac{R_{\max} - \widehat{\lambda}(x)}{\beta}$.
5. Return the first $y_i$ where $\xi_i = 1$; if none accepted, return the $(N+1)$-th sample.

**Regret bound** [source:arxiv:2503.21878]:

$$
J(\pi^{*}; x) - J(\widehat{\pi}_{\text{Pes}}; x) \lesssim \sqrt{\mathcal{C}^{\pi^{*}}(x) \cdot \varepsilon_{\mathrm{RM}}^{2}(x)} + \beta \cdot \mathcal{C}^{\pi^{*}}(x) + \beta^{-1} \cdot \varepsilon_{\mathrm{RM}}^{2}(x)
$$

Setting $\beta \asymp \sqrt{\varepsilon_{\mathrm{RM}}^{2}(x) / \mathcal{C}^{\pi^{*}}(x)}$ achieves the optimal "skyline" regret $\sqrt{\mathcal{C}^{\pi^{*}}(x) \cdot \varepsilon_{\mathrm{RM}}^{2}(x)}$.

On GSM8K, MMLU, and MATH with Phi-3-Mini and reward models (OASST, GEMMA-RM, LLAMA-RM, ARMO-RM), InferenceTimePessimism maintains monotonic improvement while BoN peaks and declines; for Phi-3-Mini on MATH using LLAMA-RM, both show ~41% lift but InferenceTimePessimism remains robust to larger $N$ [source:arxiv:2503.21878]. Limitations: the conservative $\varepsilon_{\text{BH}}^2$ metric may overestimate mismatch; binary-reward tasks reduce the gap to BoN; evaluating full proofs rather than final answers would better demonstrate benefits [source:arxiv:2503.21878].

### Adaptive budget allocation (AdaBoN)

Recognizing that prompts vary in "alignment difficulty," **AdaBoN** allocates the inference budget adaptively [source:machinelearning:adabon-adaptive-best-of-n-alignment]. An exploratory phase samples a small number of responses per prompt to estimate the reward distribution; an allocation phase then distributes the remaining budget, giving more samples to prompts where the estimated distribution suggests higher marginal returns. Across 12 LM/RM pairs on AlpacaEval, HH-RLHF, and PKU-SafeRLHF, AdaBoN outperforms uniform allocation at the same total budget and remains competitive against uniform allocation with a 20% larger budget; performance improves with batch size [source:machinelearning:adabon-adaptive-best-of-n-alignment]. No explicit formulas for the allocation rule are provided in the source.

### Speculative Rejection: efficient large-$N$ simulation

**Speculative Rejection** simulates large-$N$ BoN on a single GPU by early-stopping unpromising trajectories [source:arxiv:2410.20290]. Starting with a batch $b_{\text{init}}$ that fits in memory, it generates tokens until near OOM, scores partial responses $s(Y_k^{\le \tau_k})$, computes the $\alpha$-quantile cutoff $r_{\text{cut}} = q_\alpha(\mathcal{R}_{\text{partial}})$, and rejects sequences below $r_{\text{cut}}$. The process iterates until completions. The accepted set is $\mathcal{I}_{\text{accepted}} = \{k : s(Y_k^{\le \tau_k}) \ge r_{\text{cut}}\}$, and the final output is $Y_{k^*}$ with $k^* = \arg\max_{k \in \mathcal{I}} s(Y_k)$. On AlpacaFarm with Llama-3-8B/Mistral-7B, Speculative Rejection ($\alpha=0.5$) on one GPU matches BoN rewards that would require 16–32 GPUs, achieving 66.17% win-rate and 70.01% length-controlled win-rate vs. GPT-4-Turbo, outperforming Bo120 (50%) and Bo3840 (62.89%) [source:arxiv:2410.20290]. For perplexity minimization, it achieves 39.9$\times$ speedup over Bo120 with lower perplexity (1.554 vs. 2.407) [source:arxiv:2410.20290]. Limitations: fixed $\alpha$ is suboptimal for prompts with varying partial-final reward correlation; reward models trained as value functions (predicting final score from partial) would improve early stopping [source:arxiv:2410.20290].

### TreeBoN: speculative tree-search decoding

**TreeBoN** integrates a speculative tree-search strategy into the BoN framework, iteratively branching promising paths and pruning low-quality ones using a weighted implicit reward derived from a DPO policy model rather than a traditional reward model [source:arxiv:2410.16033].

**Algorithm** [source:arxiv:2410.16033]:
1. Divide maximum response length $l_{\max}$ into $N_{\text{layer}}$ equal segments of length $l_i = l_{\max} / N_{\text{layer}}$.
2. **Root generation**: Base policy $\pi_{\text{base}}$ generates $N$ initial candidates of length $l_1$, forming $C_1$.
3. **Iterative expansion** (for layers $i = 1$ to $N_{\text{layer}}-1$):
   - **Partial evaluation**: Compute scores for all $y \in C_i$ using partial reward function $r$.
   - **Pruning**: Select top candidates to form active set $P_i$ of size $N / N_{\text{children}}$.
   - **Branching**: For each parent in $P_i$, sample $N_{\text{children}}$ children of length $l_{i+1}$, forming $C_{i+1}$.
4. **Final selection**: Evaluate complete responses in $C_{N_{\text{layer}}}$ with a reward model; return $\arg\max_{y \in C_{N_{\text{layer}}}} r(y|x)$.

**Implicit reward guidance** [source:arxiv:2410.16033]: To evaluate partial responses $\mathbf{y}_{:K}$, TreeBoN uses a weighted sum of log-likelihood ratios from a DPO model:

$$
r_{\text{partial}}(\mathbf{y}_{:K}|\mathbf{x})=\sum_{k=0}^{K-1}w_{k}\log\frac{\pi^{*}(y_{k}|\mathbf{x},\mathbf{y}_{:k})}{\pi(y_{k}|\mathbf{x},\mathbf{y}_{:k})}, \quad w_k = \frac{1}{|\mathbf{y}_{:k}|}
$$

The authors demonstrate that traditional RMs produce "chaotic" and poorly correlated scores for partial completions, making the DPO-based implicit reward essential for reliable early pruning.

**Results** [source:arxiv:2410.16033]: Against BoN (GPT-4 eval), TreeBoN achieves 65% win rate on TutorEval and ~60% on AlpacaFarm, HH-RLHF, UltraFeedback. With only 6.3% of compute budget ($N=8$ root nodes vs. BoN's $N=128$), TreeBoN maintains 55% win rate. On GSM8K, TreeBoN increases pass@1 by 9% over BoN at 576 tokens. Under identical compute constraints, TreeBoN (63.21% win rate) significantly outperforms SBoN (49.66%) and CARDS (51.01%) at max length 192. Ablation confirms the tree structure itself provides significant advantage over standard BoN even with the same DPO reward. Limitation: the authors note TreeBoN suffers from the difficulty of training.

### STARS: synchronous token alignment with rejection sampling

**STARS (Synchronous Token Alignment for Robust Supervision)** replaces dynamic, uncertainty-based verification with a fixed-horizon supervision schedule to eliminate hardware underutilization and mitigate risks of miscalibrated model confidence [source:arxiv:2511.03827].

**Core problem** [source:arxiv:2511.03827]: Existing methods (e.g., CARDS) use model uncertainty (entropy) to determine segmentation for RM verification. Two failures arise: (1) **Miscalibrated confident hallucinations**—LLMs assign high probability to incorrect/toxic tokens, delaying verification and allowing errors to cascade; (2) **The straggler problem**—dynamic segmentation creates a ragged frontier where the batch waits for the longest segment, causing GPU idle time.

**Method** [source:arxiv:2511.03827]:
1. **Fixed-horizon generation**: Model generates exactly $K$ tokens for all requests in a batch.
2. **Synchronous verification**: Entire batch pauses simultaneously for parallel RM forward pass.
3. **Rejection sampling**: Trajectories accepted/rejected based on RM score; rejected trajectories rewind and re-sample.
4. **Iteration**: This "supervision heartbeat" repeats every $K$ tokens until completion.

This bounds "compute-at-risk" to at most $K$ tokens per hallucination. The batch latency under dynamic batching is $L_{\text{batch}} = \max_i(L_i)$; STARS ensures $L_i = K$ for all $i$, eliminating pipeline bubbles [source:arxiv:2511.03827].

**Results** (HH-RLHF, Llama-7B/Mistral-7B with Llama-7B-RM) [source:arxiv:2511.03827]:
- **Alignment quality** (Win-Tie vs. Vanilla): Llama-7b: STARS 60.2% vs. CARDS 64.5%; Mistral-7b: STARS 64.5% vs. CARDS 69.8%.
- **System efficiency** (Batch size 64): STARS ($K=15$) achieves **185.0 T/s** (53.5% improvement over CARDS 120.5 T/s); STARS ($K=30$) reaches **195.0 T/s**. Rejection waste: STARS ($K=15$) discards **15.0** tokens/rejection vs. CARDS **45.2**.

Limitation: RMs trained on complete responses can be miscalibrated on incomplete text; scoring partial text remains a challenge for reward-guided decoding [source:arxiv:2511.03827].

## Rejection Sampling as a fine-tuning method

Distinct from inference-time BoN, **Rejection Sampling (RS)** is a training-time data curation pipeline [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b]. For each of $M$ prompts, generate $N$ completions ($N=10$–$30$ typical, temperature $0.7$–$1.0$), score with a reward model, and select either (a) the top-1 per prompt or (b) the top-$K$ overall pairs. The selected $(x, y)$ pairs then form an SFT dataset. Formally, with reward matrix $r_{i,j} = R(y_{i,j} \mid x_i)$, top-per-prompt selects $y_{i, \arg\max_j r_{i,j}}$; top-overall flattens and takes top-$K$ [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b]. RS was used in Llama 2 Chat and WebGPT but remains underdocumented regarding canonical SFT hyperparameters [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b]. Key limitations: heavy dependence on reward model quality; reusing SFT prompts risks overfitting; throughput can be improved by length-sorted batching for reward inference [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b].

### Foundations: Rejection Sampling in Monte Carlo methods

The rejection sampling paradigm in LLM alignment traces to classical Monte Carlo methods for parameter estimation [source:arxiv:2107.11820]. Given a target density $\bar{\pi}(\boldsymbol{\theta}|\mathbf{y})$ known up to normalization, RS draws proposals $\boldsymbol{\theta}^{(t)} \sim \bar{q}(\boldsymbol{\theta})$ and accepts if $u \le \frac{\pi(\boldsymbol{\theta}^{(t)})}{Cq(\boldsymbol{\theta}^{(t)})}$ where $C$ satisfies $Cq(\boldsymbol{\theta}) \ge \pi(\boldsymbol{\theta})$. The acceptance probability is $P_A = Z_\pi / CZ_q$; in high dimensions, finding a tight $C$ is difficult and $P_A$ becomes very low, rendering naive RS inefficient [source:arxiv:2107.11820].

**Adaptive Rejection Metropolis Sampling (ARMS)** and its improvements ($\text{A}^2\text{RMS}$, $\text{IA}^2\text{RMS}$) address this for non-log-concave 1D densities by adaptively constructing a piecewise-linear proposal envelope [source:arxiv:1205.5494]. Standard ARMS only adds support points when the proposal overestimates the target ($\pi_t(x) \ge p(x)$), causing the chain to trap in a single mode of multimodal densities. $\text{A}^2\text{RMS}$ and $\text{IA}^2\text{RMS}$ add a second control step that incorporates support points even when $\pi_t(x) < p(x)$, ensuring the proposal converges to the target. On a 3-Gaussian mixture, $\text{IA}^2\text{RMS}$ reduces estimation variance by ~6× and the $L_1$ proposal-target distance $D_{\pi|p}(t)$ by >10× compared to ARMS (0.0609 vs. 3.0020) [source:arxiv:1205.5494]. $\text{A}^2\text{RMS}$ requires discarding the first $K$ samples to ensure convergence; $\text{IA}^2\text{RMS}$ uses an auxiliary variable to maintain independence. These adaptive envelope techniques inform modern speculative decoding methods that dynamically bound partial-sequence rewards.

## Inference-aware fine-tuning for BoN

If a model is destined for BoN deployment, standard SFT/RL (which optimize single-response quality) are suboptimal. [source:arxiv:2412.15287] derives **inference-aware training** objectives that directly optimize the BoN policy $\pi_{\text{bon}}(y \mid x) \propto \pi(y \mid x) \exp(\lambda_N Q_\pi(x, y))$, where $Q_\pi(x, y) = \mathbb{E}_{y' \sim \pi}[\mathbf{1}_{r(x,y) \ge r(x,y')}]$ is the expected win-rate under verifier $r$.

- **BoN-SFT**: Variational approximation maximizes expert likelihood under $\pi_{\text{bon}}$, regularizing for exploration.
- **BoN-RL**: REINFORCE-style gradient sampling from $\pi_{\text{bon}}$ to maximize environment reward $R(x,y)$.
- **BoN-RLB** (binary rewards): Closed-form gradient with asymmetric weights for "hard" examples:

$$
g_N^+(p) = \frac{N p^{N-1}}{1 - p^N}, \quad g^-(p) = \frac{N p}{1 - p}
$$

  where $p = P_{\text{fail}}(x)$ is base failure probability. A positive-only variant BoN-RLB(P) uses $\bar{g}_N^+(p) = \frac{N p^{N-1}(1-p)}{1-p^N}$ when negative data is unavailable [source:arxiv:2412.15287].

On Gemma 2B/9B: BoN-RL-V improves MATH Bo32 from 26.8\% to 30.8\%; BoN-RL-S improves pass@32 from 60.0\% to 67.0\%; BoN-RLB(P) improves HumanEval pass@16 from 61.6\% to 67.1\% [source:arxiv:2412.15287]. Gains generalize to held-out benchmarks (Functional MATH, MathOdyssey) and across temperatures. Limitations: verifier noise causes Type II errors at high $T$ and large $N$; generating $N$ samples per gradient step is expensive (mitigated by BoN Distillation); weights $g_N^+(p)$ explode for hard problems ($p \to 1$), requiring clipping [source:arxiv:2412.15287].

## Position in the inference-time scaling landscape

BoN and RS are core instances of **inference-time scaling** (test-time compute), alongside chain-of-thought, self-consistency, self-refinement, and search over solution paths [source:magazine:categories-of-inference-time-scaling-for]. The central trade-off is accuracy vs. compute/latency: one study reports base accuracy ~15\% rising to ~52\% with combined inference-scaling methods [source:magazine:categories-of-inference-time-scaling-for]. The most effective strategy combines a stronger base model (training-time scaling) with inference-time techniques [source:magazine:categories-of-inference-time-scaling-for].

## Current status and trajectory

BoN and its variants are **rising as a default inference-time alignment tool**, especially for reasoning and coding where verifiable rewards exist. Speculative Rejection, TreeBoN, and AdaBoN address the compute bottleneck, making large-$N$ BoN practical on limited hardware. MBR-BoN and InferenceTimePessimism address the overoptimization bottleneck, with the latter offering stronger theoretical guarantees (regret-optimality, scaling-monotonicity) but requiring uncertainty estimation. STARS introduces a synchronous, fixed-horizon paradigm that eliminates the straggler problem and bounds hallucination risk, trading slight alignment quality for large system throughput gains. Inference-aware training (BoN-SFT/RL) is a newer direction with strong early results on math/code but not yet widely adopted in major open models. Rejection Sampling (training-time) remains a workhorse for data curation (Llama 2, WebGPT) but is underdocumented. The field is converging on **hybrid approaches**: inference-aware training + regularized/adaptive BoN at test time. However, all methods remain fundamentally limited by reward model accuracy—no inference-time algorithm can overcome a broken $\hat{r}$ [source:icml:is-best-of-n-the-best-of-them-coverage-s]. Not widely reported: head-to-head comparisons of MBR-BoN vs. InferenceTimePessimism vs. TreeBoN vs. STARS on the same benchmarks; the practicality of $\chi^2$ uncertainty estimation at scale; whether BoN-aware training generalizes beyond math/code to open-ended chat; and how fixed-horizon verification (STARS) interacts with tree-search (TreeBoN) or implicit rewards.

## Key takeaways

- **BoN vs. RS are distinct**: BoN = inference-time selection; RS = training-time data filtering for SFT. Both use generate-and-score but serve different stages.
- **Vanilla BoN overoptimizes**: Quality peaks then drops as $N$ grows due to reward hacking; this is theoretically inevitable under $L_1$ coverage [source:icml:is-best-of-n-the-best-of-them-coverage-s].
- **Regularization restores monotonicity**: MBR-BoN (Wasserstein proximity) [source:aclanthology:regularized-best-of-n-sampling-with-mini][source:arxiv:2404.01054] and InferenceTimePessimism ($\chi^2$ pessimism) [source:icml:is-best-of-n-the-best-of-them-coverage-s][source:arxiv:2503.21878] prevent degradation; the latter is provably regret-optimal and scaling-monotonic.
- **Compute efficiency is solvable**: Speculative Rejection simulates large $N$ on one GPU via early rejection [source:arxiv:2410.20290]; TreeBoN uses tree-search with DPO implicit rewards for 16× compute reduction [source:arxiv:2410.16033]; AdaBoN allocates budget adaptively per prompt [source:machinelearning:adabon-adaptive-best-of-n-alignment]; STARS uses fixed-horizon synchronous verification for 53\% throughput gain [source:arxiv:2511.03827].
- **Train for how you test**: BoN-aware SFT/RL directly optimize the BoN policy, yielding large gains on math/code (e.g., +4\% MATH Bo32, +5.5\% HumanEval pass@16) [source:arxiv:2412.15287].
- **Reward model quality is the hard ceiling**: All inference-time methods are bounded by $\hat{r}$'s MSE and base policy coverage [source:icml:is-best-of-n-the-best-of-them-coverage-s][source:arxiv:2503.21878].

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [KL regularization in RLHF](kl-regularization.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Length and format bias](length-and-format-bias.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [LLM-as-judge](llm-as-judge.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)

## References
- [source:arxiv:2410.20290] [Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/html/2410.20290v2)
- [source:arxiv:2412.15287] [Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models](https://arxiv.org/pdf/2412.15287)
- [source:machinelearning:adabon-adaptive-best-of-n-alignment] [AdaBoN: Adaptive Best-of-N Alignment](https://machinelearning.apple.com/research/best-of-n)
- [source:magazine:categories-of-inference-time-scaling-for] [Categories of Inference-Time Scaling for Improved LLM Reasoning](https://magazine.sebastianraschka.com/p/categories-of-inference-time-scaling)
- [source:aclanthology:regularized-best-of-n-sampling-with-mini] [Regularized Best-of-N Sampling with Minimum Bayes Risk Decoding](https://aclanthology.org/2025.naacl-long.472.pdf)
- [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b] [Rejection Sampling - Nathan Lambert (RLHF Book)](https://rlhfbook.com/c/09-rejection-sampling)
- [source:icml:is-best-of-n-the-best-of-them-coverage-s] [Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment](https://icml.cc/virtual/2025/poster/45322)
- [source:arxiv:2404.01054] [Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language Model Alignment](https://arxiv.org/abs/2404.01054)
- [source:arxiv:2503.21878] [Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment](https://arxiv.org/abs/2503.21878)
- [source:arxiv:2410.16033] [TreeBoN: Enhancing Inference-Time Alignment with Speculative Tree-Search Decoding](https://arxiv.org/abs/2410.16033)
- [source:arxiv:2511.03827] [STARS: Segment-level Token Alignment with Rejection Sampling in Large Language Models](https://arxiv.org/abs/2511.03827)
- [source:arxiv:2107.11820] [A survey of Monte Carlo methods for parameter estimation](https://arxiv.org/abs/2107.11820)
- [source:arxiv:1205.5494] [Improved Adaptive Rejection Metropolis Sampling Algorithms](https://arxiv.org/abs/1205.5494)
