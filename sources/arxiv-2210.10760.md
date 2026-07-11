---
id: arxiv:2210.10760
type: paper
title: Scaling Laws for Reward Model Overoptimization
url: https://arxiv.org/abs/2210.10760
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-model-overoptimization
---

**Core Problem**
In reinforcement learning from human feedback (RLHF), optimizing a policy against a learned proxy reward model (RM) frequently triggers overoptimization, a direct manifestation of Goodhart’s law where maximizing the proxy score eventually degrades the true objective. Empirically quantifying this degradation is hindered by the prohibitive cost of collecting human preference labels at scale. This work establishes empirical scaling laws for overoptimization to predict how ground-truth performance degrades as optimization pressure increases, providing a quantitative foundation for theoretical alignment research and safe model deployment.

**Methodology**
The authors employ a synthetic experimental pipeline to bypass human labeling costs. First, a fixed 6B-parameter “gold-standard” RM is designated as the ground-truth reward signal. Second, proxy RMs (varying from 3M to 3B parameters) are trained on synthetic preference pairs generated deterministically: two policy rollouts are sampled for identical prompts, and the trajectory with the higher gold score is labeled as preferred. The synthetic dataset comprises 100,000 comparisons, with 10% reserved for validation. Third, initial policies are GPT-3 series models (1.2B and 6B parameters) fine-tuned via supervised fine-tuning (SFT) for two epochs. Fourth, optimization is executed using two distinct methods: best-of-$n$ (BoN) sampling, which selects the highest-proxy trajectory from $n$ candidates, and Proximal Policy Optimization (PPO) for reinforcement learning. Fifth, proxy RM scores are recalibrated via cross-entropy minimization on soft labels, and all scores are recentered and unit-normalized for cross-model comparability. Finally, the relationship between gold reward and optimization pressure is parameterized by $d = \sqrt{D_{\mathrm{KL}}(\pi \parallel \pi_{\mathrm{init}})}$.

**Key Formulas**
The gold reward $R$ as a function of $d$ follows distinct functional forms for each optimization method:
$$R_{\mathrm{bon}}(d) = d(\alpha_{\mathrm{bon}} - \beta_{\mathrm{bon}}d)$$
$$R_{\mathrm{RL}}(d) = d(\alpha_{\mathrm{RL}} - \beta_{\mathrm{RL}}\log d)$$
The coefficients $\alpha$ and $\beta$ scale smoothly with proxy RM parameters and dataset size. For regressional Goodhart effects, the expected gold reward given a proxy score $\hat{x}$ is modeled as:
$$\mathbb{E}[X \mid \hat{X} = \hat{x}] = \mathbb{E}[X] + (\hat{x} - \mathbb{E}[X] - \mathbb{E}[Z]) \frac{\text{Var}(X)}{\text{Var}(X) + \text{Var}(Z)} + \varepsilon$$
where $X$ is the gold reward, $\hat{X}$ is the proxy, and $Z$ is independent noise. For iterated RLHF over $k$ iterations, the final gold score scales as $R_{\text{RL}}(d) = d(\alpha_{\text{RL}} - \beta_{\text{RL}} \log(d) + \beta_{\text{RL}} \log(k))$.

**Quantitative Results**
The BoN functional form was validated as a true advance prediction, accurately fitting data up to $n=60,000$ (KL $\approx 10$ nats) after being hypothesized on data up to $n=1,000$ (KL $\approx 6$ nats). Coefficients $\alpha_{\mathrm{bon}}$ and $\beta_{\mathrm{bon}}$ exhibit smooth, approximate logarithmic scaling with proxy RM parameter count, while $\alpha_{\mathrm{RL}}$ remains nearly constant across RM sizes. Regarding data scaling, proxy RMs require at least 2,000 training comparisons to surpass near-chance performance; beyond this threshold, larger RMs improve faster. Policy size scaling reveals that while a 6B policy gains less absolute reward from optimization than a 1.2B policy, both peak at nearly identical KL distances, and the proxy-gold shortfall remains consistent. BoN is significantly more KL-efficient than RL, which consumes KL distance approximately quadratically per step. Implementing an explicit KL penalty in PPO acts strictly as early stopping, improving the proxy score but leaving the gold score-KL frontier unchanged.

**Limitations**
The synthetic setup inherently excludes overoptimization stemming from misalignment between ground-truth labels and actual human intent. The authors note that RM correlations in real-world settings may limit transferability. Only two policy sizes were evaluated, restricting generalization of policy-scaling trends. The functional form for proxy RM scores resists accurate fitting and extrapolation. Adversarial Goodhart is not captured, as the evaluated models lack the capacity for active manipulation, potentially causing these scaling laws to break down in more advanced systems. Finally, the theoretical model for iterated RLHF relies on unverified assumptions regarding coefficient stability and KL additivity across training iterations.
