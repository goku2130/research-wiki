---
id: arxiv:2308.08998
type: paper
title: Reinforced Self-Training (ReST) for Language Modeling
url: https://arxiv.org/abs/2308.08998
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

The core problem addressed is the computational inefficiency and reward-hacking vulnerability of online reinforcement learning from human feedback (RLHF) methods, alongside the dataset-quality limitations of static offline reinforcement learning. The authors propose Reinforced Self-Training (ReST), a growing batch reinforcement learning algorithm that decouples data generation from policy optimization to align large language models with human preferences efficiently.

The ReST recipe operates in two alternating loops. First, the **Grow** step samples multiple output sequences $y$ from the current policy $\pi_\theta$ for each context $x$ in the original dataset $\mathcal{D}$, forming an augmented dataset $\mathcal{D}_g$ that is scored by a reward model $R(x,y)$. Second, the **Improve** step filters $\mathcal{D}_g$ using a reward threshold $\tau$ and fine-tunes the policy on the retained samples. This Improve step is repeated iteratively with progressively increasing thresholds $\tau_1 < \tau_2 < \dots$, allowing the model to amortize the expensive dataset generation cost across multiple optimization passes while gradually shifting focus to higher-quality synthetic data.

Mathematically, the autoregressive policy is defined as $\pi_\theta(y \mid x) = \prod_{t=1}^T \pi_\theta(y_t \mid y_{1:t-1}, x)$. The initial policy is trained via negative log-likelihood (NLL): $\mathcal{L}_{\text{NLL}}(\theta) = -\mathbb{E}_{(x,y)\sim\mathcal{D}} \left[ \sum_{t=1}^T \log \pi_\theta(y_t \mid y_{1:t-1}, x) \right]$. The augmented dataset is constructed as $\mathcal{D}_g = \{ (x^i, y^i)|_{i=1}^{N_g} \text{ s.t. } x^i \sim \mathcal{D}, y^i \sim \pi_\theta(y|x^i) \} \cup \mathcal{D}$. Policy improvement utilizes a threshold-based filtering function $F(x, y; \tau) = \mathbb{1}_{R(x, y) > \tau}$ and optimizes the objective $J(\theta) = \mathbb{E}_{(x, y) \sim \mathcal{D}_g} [ F(x, y; \tau) \mathcal{L}(x, y; \theta) ]$, where $\mathcal{L}$ can be the NLL loss or an offline RL loss.

Quantitative evaluations on machine translation benchmarks demonstrate ReST's efficacy. On IWSLT 2014 De-En, the supervised baseline (BC) achieved an average reward of 70.9. ReST with one Grow and four Improve steps (G=1, I=4) reached 77.8, while two Grow steps with three Improve steps (G=2, I=3) achieved 83.1. A second Grow step alone yielded a 5.3-point reward increase over the first. Compared to online PPO, which scored 71.6 with equivalent data, ReST significantly outperformed it without the BLEU score degradation observed in online methods. Human evaluations confirmed that all ReST variants outperformed the supervised baseline, though reward model rankings did not perfectly align with human preference rankings.

The authors identify several limitations. First, reward models serve as imperfect proxies for human preferences, leading to discrepancies between automated reward scores and human evaluations, particularly as policies diverge from the behavior model. Second, repeated Grow steps increase the risk of overfitting to the reward model. Third, the deterministic nature of language modeling limits the performance gains of complex offline RL losses, with standard BC loss empirically outperforming advanced alternatives. Finally, the Grow step relies on simple sampling, which restricts exploration; the authors note that incorporating strategies like Monte Carlo Tree Search could mitigate this limitation.
