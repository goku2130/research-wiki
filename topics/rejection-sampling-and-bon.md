---
title: Rejection sampling and Best-of-N
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2410.20290
- arxiv:2412.15287
- rlhfbook:rejection-sampling-nathan-lambert-rlhf-b
- aclanthology:regularized-best-of-n-sampling-with-mini
- machinelearning:adabon-adaptive-best-of-n-alignment
- icml:is-best-of-n-the-best-of-them-coverage-s
- magazine:categories-of-inference-time-scaling-for
open_questions:
- How do MBR-BoN and InferenceTimePessimism compare head-to-head on the same benchmarks,
  reward models, and base policies? No source provides this direct comparison.
- Can InferenceTimePessimism's $\chi^2$ uncertainty estimation be made practical at
  scale (e.g., without large ensembles), or do lighter-weight approximations preserve
  its theoretical guarantees?
- Does BoN-aware training (BoN-SFT/RL) generalize to open-ended instruction following
  and chat, or is its benefit confined to verifiable domains (math/code)?
- What is the optimal interaction between inference-aware training and regularized/adaptive
  BoN at test time—do they compose synergistically or redundantly?
---

Best-of-N (BoN) and rejection sampling (RS) are two distinct but related paradigms for aligning language models: BoN operates at inference time by generating multiple candidates and selecting the highest-scoring one under a reward model, while RS operates at training time by filtering model generations through a reward model to construct a high-quality supervised fine-tuning dataset. Both methods face a fundamental tension between exploitating a proxy reward model and avoiding overoptimization, driving a line of work on regularization, adaptive computation, and inference-aware training.

## Inference-time Best-of-N: mechanics and scaling laws

Standard Best-of-N (BoN) decoding samples $N$ independent responses $y_1, \dots, y_N \sim \pi_{\text{ref}}(\cdot \mid x)$ from a reference policy and returns $y^* = \arg\max_i \hat{r}(x, y_i)$, where $\hat{r}$ is a learned reward model [source:icml:is-best-of-n-the-best-of-them-coverage-s]. The method is simple, requires no weight updates, and is competitive with RLHF post-training for moderate $N$ [source:arxiv:2410.20290]. However, its computational cost scales linearly with $N$: generating $N$ full sequences requires either large batch sizes (risking OOM) or high latency, making $N > 1000$ impractical on typical hardware [source:arxiv:2410.20290].

The quality of BoN as a function of $N$ is not monotonic. Empirically, win-rates against a base model often peak at moderate $N$ (e.g., $N=16$–$32$) and then degrade due to **reward hacking**—the reward model $\hat{r}$ assigns high scores to outputs that exploit its errors rather than reflecting true quality $r^*$ [source:aclanthology:regularized-best-of-n-sampling-with-mini][source:icml:is-best-of-n-the-best-of-them-coverage-s]. Theoretically, [source:icml:is-best-of-n-the-best-of-them-coverage-s] proves that BoN achieves optimal regret only under stringent $L_\infty$-type coverage of the base policy over high-reward responses; under realistic $L_1$ coverage, BoN provably overoptimizes and fails to provide tight guarantees. The information-theoretic limit is governed by the mean-squared error of $\hat{r}$ and the base policy's coverage: no inference-time algorithm can overcome a fundamentally inaccurate reward model [source:icml:is-best-of-n-the-best-of-them-coverage-s].

## Regularized and robust BoN variants

### Minimum Bayes Risk regularization (MBR-BoN)

To mitigate reward hacking, [source:aclanthology:regularized-best-of-n-sampling-with-mini] proposes **MBR-BoN**, which adds a proximity regularizer encouraging the selected response to stay near the center of the sample distribution. Given $N$ samples $Y_{\text{ref}} \sim \pi_{\text{ref}}$, MBR-BoN selects:

$$
y_{\text{MBR-BoN}} = \arg\max_{y \in Y_{\text{ref}}} \hat{r}(x, y) + \beta \frac{1}{N} \sum_{y' \in Y_{\text{ref}}} U(y, y')
$$

where $U(y, y') = \cos(\text{emb}(y), \text{emb}(y'))$ is cosine similarity of embeddings and $\beta$ controls the trade-off. The authors show that maximizing this objective minimizes the Wasserstein distance between the output distribution and the empirical reference distribution, arguing that Wasserstein is more robust than KL for inference-time regularization because KL requires exponentially many samples to estimate reliably and is oversensitive to minor textual variations [source:aclanthology:regularized-best-of-n-sampling-with-mini]. On AlpacaFarm and hh-rlhf with Mistral-7B, MBR-BoN consistently outperforms vanilla BoN and pure MBR across proxy reward models (SHP-Large, SHP-XL, OASST), with optimal $\beta$ varying widely (e.g., $0.5$ for SHP-Large vs. $20.0$ for OASST) but findable with as few as 10 development instances [source:aclanthology:regularized-best-of-n-sampling-with-mini]. MBR-BoN also improves downstream DPO training when used to generate preference pairs [source:aclanthology:regularized-best-of-n-sampling-with-mini]. The quadratic $O(N^2)$ utility computation adds overhead (~2s for $N=128$ on T4) [source:aclanthology:regularized-best-of-n-sampling-with-mini].

### InferenceTimePessimism: $\chi^2$-regularized rejection sampling

[source:icml:is-best-of-n-the-best-of-them-coverage-s] introduces **InferenceTimePessimism**, a theoretically grounded alternative that implements $\chi^2$-regularization via a novel rejection sampling scheme at inference time. The algorithm decouples the computational budget $N$ from the pessimism penalty strength, achieving **scaling-monotonicity**: performance does not degrade as $N \to \infty$, unlike BoN. On GSM8K with OASST reward model, InferenceTimePessimism maintains monotonic improvement while BoN peaks and declines; it is proven regret-optimal, matching the "skyline" of best achievable reward given $\hat{r}$ [source:icml:is-best-of-n-the-best-of-them-coverage-s]. The method requires estimating uncertainty (e.g., via ensembles or variance of $\hat{r}$), which adds implementation complexity not required by MBR-BoN.

### Adaptive budget allocation (AdaBoN)

Recognizing that prompts vary in "alignment difficulty," **AdaBoN** allocates the inference budget adaptively [source:machinelearning:adabon-adaptive-best-of-n-alignment]. An exploratory phase samples a small number of responses per prompt to estimate the reward distribution; an allocation phase then distributes the remaining budget, giving more samples to prompts where the estimated distribution suggests higher marginal returns. Across 12 LM/RM pairs on AlpacaEval, HH-RLHF, and PKU-SafeRLHF, AdaBoN outperforms uniform allocation at the same total budget and remains competitive against uniform allocation with a 20% larger budget; performance improves with batch size [source:machinelearning:adabon-adaptive-best-of-n-alignment]. No explicit formulas for the allocation rule are provided in the source.

### Speculative Rejection: efficient large-$N$ simulation

**Speculative Rejection** simulates large-$N$ BoN on a single GPU by early-stopping unpromising trajectories [source:arxiv:2410.20290]. Starting with a batch $b_{\text{init}}$ that fits in memory, it generates tokens until near OOM, scores partial responses $s(Y_k^{\le \tau_k})$, computes the $\alpha$-quantile cutoff $r_{\text{cut}} = q_\alpha(\mathcal{R}_{\text{partial}})$, and rejects sequences below $r_{\text{cut}}$. The process iterates until completions. The accepted set is $\mathcal{I}_{\text{accepted}} = \{k : s(Y_k^{\le \tau_k}) \ge r_{\text{cut}}\}$, and the final output is $Y_{k^*}$ with $k^* = \arg\max_{k \in \mathcal{I}} s(Y_k)$. On AlpacaFarm with Llama-3-8B/Mistral-7B, Speculative Rejection ($\alpha=0.5$) on one GPU matches BoN rewards that would require 16–32 GPUs, achieving 66.17% win-rate and 70.01% length-controlled win-rate vs. GPT-4-Turbo, outperforming Bo120 (50%) and Bo3840 (62.89%) [source:arxiv:2410.20290]. For perplexity minimization, it achieves 39.9$\times$ speedup over Bo120 with lower perplexity (1.554 vs. 2.407) [source:arxiv:2410.20290]. Limitations: fixed $\alpha$ is suboptimal for prompts with varying partial-final reward correlation; reward models trained as value functions (predicting final score from partial) would improve early stopping [source:arxiv:2410.20290].

## Rejection Sampling as a fine-tuning method

Distinct from inference-time BoN, **Rejection Sampling (RS)** is a training-time data curation pipeline [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b]. For each of $M$ prompts, generate $N$ completions ($N=10$–$30$ typical, temperature $0.7$–$1.0$), score with a reward model, and select either (a) the top-1 per prompt or (b) the top-$K$ overall pairs. The selected $(x, y)$ pairs then form an SFT dataset. Formally, with reward matrix $r_{i,j} = R(y_{i,j} \mid x_i)$, top-per-prompt selects $y_{i, \arg\max_j r_{i,j}}$; top-overall flattens and takes top-$K$ [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b]. RS was used in Llama 2 Chat and WebGPT but remains underdocumented regarding canonical SFT hyperparameters [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b]. Key limitations: heavy dependence on reward model quality; reusing SFT prompts risks overfitting; throughput can be improved by length-sorted batching for reward inference [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b].

## Inference-aware fine-tuning for BoN

If a model is destined for BoN deployment, standard SFT/RL (which optimize single-response quality) are suboptimal. [source:arxiv:2412.15287] derives **inference-aware training** objectives that directly optimize the BoN policy $\pi_{\text{bon}}(y \mid x) \propto \pi(y \mid x) \exp(\lambda_N Q_\pi(x, y))$, where $Q_\pi(x, y) = \mathbb{E}_{y' \sim \pi}[\mathbf{1}_{r(x,y) \ge r(x,y')}]$ is the expected win-rate under verifier $r$.

- **BoN-SFT**: Variational approximation maximizes expert likelihood under $\pi_{\text{bon}}$, regularizing for exploration.
- **BoN-RL**: REINFORCE-style gradient sampling from $\pi_{\text{bon}}$ to maximize environment reward $R(x,y)$.
- **BoN-RLB** (binary rewards): Closed-form gradient with asymmetric weights for "hard" examples:

$$
g_N^+(p) = \frac{N p^{N-1}}{1 - p^N}, \quad g^-(p) = \frac{N p}{1 - p}
$$

  where $p = P_{\text{fail}}(x)$ is base failure probability. A positive-only variant BoN-RLB(P) uses $\bar{g}_N^+(p) = \frac{N p^{N-1}(1-p)}{1-p^N}$ when negative data is unavailable [source:arxiv:2412.15287].

On Gemma 2B/9B: BoN-RL-V improves MATH Bo32 from 26.8% to 30.8%; BoN-RL-S improves pass@32 from 60.0% to 67.0%; BoN-RLB(P) improves HumanEval pass@16 from 61.6% to 67.1% [source:arxiv:2412.15287]. Gains generalize to held-out benchmarks (Functional MATH, MathOdyssey) and across temperatures. Limitations: verifier noise causes Type II errors at high $T$ and large $N$; generating $N$ samples per gradient step is expensive (mitigated by BoN Distillation); weights $g_N^+(p)$ explode for hard problems ($p \to 1$), requiring clipping [source:arxiv:2412.15287].

## Position in the inference-time scaling landscape

BoN and RS are core instances of **inference-time scaling** (test-time compute), alongside chain-of-thought, self-consistency, self-refinement, and search over solution paths [source:magazine:categories-of-inference-time-scaling-for]. The central trade-off is accuracy vs. compute/latency: one study reports base accuracy ~15% rising to ~52% with combined inference-scaling methods [source:magazine:categories-of-inference-time-scaling-for]. The most effective strategy combines a stronger base model (training-time scaling) with inference-time techniques [source:magazine:categories-of-inference-time-scaling-for].

## Current status and trajectory

BoN and its variants are **rising as a default inference-time alignment tool**, especially for reasoning and coding where verifiable rewards exist. Speculative Rejection and AdaBoN address the compute bottleneck, making large-$N$ BoN practical on limited hardware. MBR-BoN and InferenceTimePessimism address the overoptimization bottleneck, with the latter offering stronger theoretical guarantees (regret-optimality, scaling-monotonicity) but requiring uncertainty estimation. Inference-aware training (BoN-SFT/RL) is a newer direction with strong early results on math/code but not yet widely adopted in major open models. Rejection Sampling (training-time) remains a workhorse for data curation (Llama 2, WebGPT) but is underdocumented. The field is converging on **hybrid approaches**: inference-aware training + regularized/adaptive BoN at test time. However, all methods remain fundamentally limited by reward model accuracy—no inference-time algorithm can overcome a broken $\hat{r}$ [source:icml:is-best-of-n-the-best-of-them-coverage-s]. Not widely reported: head-to-head comparisons of MBR-BoN vs. InferenceTimePessimism on the same benchmarks; the practicality of $\chi^2$ uncertainty estimation at scale; and whether BoN-aware training generalizes beyond math/code to open-ended chat.

## Key takeaways

- **BoN vs. RS are distinct**: BoN = inference-time selection; RS = training-time data filtering for SFT. Both use generate-and-score but serve different stages.
- **Vanilla BoN overoptimizes**: Quality peaks then drops as $N$ grows due to reward hacking; this is theoretically inevitable under $L_1$ coverage [source:icml:is-best-of-n-the-best-of-them-coverage-s].
- **Regularization restores monotonicity**: MBR-BoN (Wasserstein proximity) [source:aclanthology:regularized-best-of-n-sampling-with-mini] and InferenceTimePessimism ($\chi^2$ pessimism) [source:icml:is-best-of-n-the-best-of-them-coverage-s] prevent degradation; the latter is provably regret-optimal and scaling-monotonic.
- **Compute efficiency is solvable**: Speculative Rejection simulates large $N$ on one GPU via early rejection [source:arxiv:2410.20290]; AdaBoN allocates budget adaptively per prompt [source:machinelearning:adabon-adaptive-best-of-n-alignment].
- **Train for how you test**: BoN-aware SFT/RL directly optimize the BoN policy, yielding large gains on math/code (e.g., +4% MATH Bo32, +5.5% HumanEval pass@16) [source:arxiv:2412.15287].
- **Reward model quality is the hard ceiling**: All inference-time methods are bounded by $\hat{r}$'s MSE and base policy coverage [source:icml:is-best-of-n-the-best-of-them-coverage-s].

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

## References
- [source:arxiv:2410.20290] [Fast Best-of-N Decoding via Speculative Rejection](https://arxiv.org/html/2410.20290v2)
- [source:arxiv:2412.15287] [Inference-Aware Fine-Tuning for Best-of-N Sampling in Large Language Models](https://arxiv.org/pdf/2412.15287)
- [source:rlhfbook:rejection-sampling-nathan-lambert-rlhf-b] [Rejection Sampling - Nathan Lambert (RLHF Book)](https://rlhfbook.com/c/09-rejection-sampling)
- [source:aclanthology:regularized-best-of-n-sampling-with-mini] [Regularized Best-of-N Sampling with Minimum Bayes Risk Decoding](https://aclanthology.org/2025.naacl-long.472.pdf)
- [source:machinelearning:adabon-adaptive-best-of-n-alignment] [AdaBoN: Adaptive Best-of-N Alignment](https://machinelearning.apple.com/research/best-of-n)
- [source:icml:is-best-of-n-the-best-of-them-coverage-s] [Is Best-of-N the Best of Them? Coverage, Scaling, and Optimality in Inference-Time Alignment](https://icml.cc/virtual/2025/poster/45322)
- [source:magazine:categories-of-inference-time-scaling-for] [Categories of Inference-Time Scaling for Improved LLM Reasoning](https://magazine.sebastianraschka.com/p/categories-of-inference-time-scaling)
