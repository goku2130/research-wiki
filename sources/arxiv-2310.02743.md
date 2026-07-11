---
id: arxiv:2310.02743
type: paper
title: Reward Model Ensembles Help Mitigate Overoptimization
url: https://arxiv.org/abs/2310.02743
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-model-overoptimization
---

The core problem addressed is *overoptimization* in Reinforcement Learning from Human Feedback (RLHF). Learned proxy reward models are imperfect approximations of true human preferences (modeled as a "gold" reward model). When policies are optimized against these proxies, performance initially improves but subsequently regresses relative to the true reward. While scaling proxy reward models mitigates this, it is computationally prohibitive and often infeasible.

The proposed method follows a structured recipe: (1) Train an ensemble of $k$ proxy reward models $\{R_1, ..., R_k\}$ using identical data and hyperparameters but distinct random seeds to induce diversity. (2) During policy optimization, aggregate ensemble outputs using one of three objectives. (3) Guide policy optimization via either Best-of-$n$ (BoN) sampling or Proximal Policy Optimization (PPO). (4) Optionally apply a KL penalty during PPO to regularize policy deviation from the initial model.

The key mathematical formulations define the optimization objectives and regularization. The BoN degree of optimization is quantified as:

$$
\mathrm{KL}_{\text{bon}} = \log n - \frac{n-1}{n}
$$

PPO incorporates a KL penalty into the reward signal:

$$
R^{\mathrm{PPO}}(q, a) = R(q, a) - \beta \log \left[ \frac{\pi^{\mathrm{PPO}}(a|q)}{\pi^{\mathrm{init}}(a|q)} \right]
$$

Ensemble aggregation methods are defined as:

$$
R_\mu(q, a) := \frac{1}{k} \sum_{i=1}^{k} R_i(q, a)
$$

$$
R_{\mathrm{WCO}}(q, a) := \min_{i} R_{i}(q, a)
$$

$$
R_{\mathrm{UWO}}(q, a) := \frac{1}{k} \sum_{i} R_{i}(q, a) - \lambda \frac{1}{k} \sum_{i} \left( R_{i}(q, a) - \frac{1}{k} \sum_{i} R_{i}(q, a) \right)^{2}
$$

where $q$ and $a$ denote prompts and responses, respectively.

Quantitative results demonstrate that ensemble-based conservative optimization effectively eliminates overoptimization. Under both noiseless conditions and 25% label noise (simulating human annotator disagreement), WCO and UWO prevent overoptimization in BoN sampling, improving final gold reward performance by up to $\sim 30\%$ without noise and up to $\sim 75\%$ with noise compared to single reward models. For PPO, WCO and UWO consistently reduce overoptimization and match or exceed single reward model performance across all KL penalty weights. Notably, combining these methods with a minimal KL penalty of $\beta = 0.01$ fully prevents overoptimization without performance degradation. In contrast, single reward models require a $20\times$ larger KL penalty ($\beta = 0.2$) to eliminate overoptimization, incurring substantial performance costs. The performance gains from ensembling are orthogonal to scaling reward model size or training dataset size. Hyperparameter analysis indicates that ensemble cardinalities of 4–5 members yield optimal results, and UWO is robust to reasonable variations in $\lambda$.

The study acknowledges several limitations. The methodology is evaluated exclusively within an *offline* RLHF framework where the reward model remains fixed during policy optimization; its efficacy in *online* RLHF, where reward models are periodically retrained, remains untested. Additionally, the approach relies on a fixed compute budget that constrained BoN evaluation to a maximum of $n_{max} = 12,500$ samples. Conservative methods like WCO may occasionally incur performance penalties due to their highly risk-averse nature. Finally, while results are robust across model scales (7M to 1.3B parameters) and dataset sizes, the authors note that replication on larger-scale language models and diverse RLHF datasets is necessary to confirm generalizability.
