---
id: arxiv:2009.01325
type: paper
title: Learning to summarize with human feedback
url: https://arxiv.org/abs/2009.01325
retrieved: '2026-07-10'
maturity: comprehensive
topic: ppo-for-llms
---

**Core Problem**
Training and evaluating abstractive summarization models traditionally relies on supervised fine-tuning to maximize the likelihood of human reference summaries and automatic metrics like ROUGE. This pipeline suffers from a fundamental misalignment with actual human-preferred quality. The maximum-likelihood objective fails to distinguish between critical errors (e.g., hallucination) and trivial ones, incentivizes probability mass on low-quality references, and suffers from distributional shift during sampling. Furthermore, ROUGE correlates poorly with human judgments, making it an unreliable proxy for summary quality.

**Methodology**
The authors propose a batch-based reinforcement learning pipeline aligned with human preferences. The process iterates through three core steps:
1. **Data Collection:** Summaries are sampled from multiple sources (current policy, initial supervised policy, human references, and baselines). Pairs of summaries for identical prompts are sent to human labelers, who select the preferred summary.
2. **Reward Modeling:** A reward model (RM) is trained via supervised learning to predict the log odds that a given summary is preferred. The RM uses a Transformer decoder with a randomly initialized linear head outputting a scalar score.
3. **Policy Optimization:** The RM’s output is treated as a reward signal to fine-tune a generative policy using Proximal Policy Optimization (PPO). A separate value network is employed to stabilize training. The pipeline can be repeated iteratively by collecting new human comparisons from the improved policy.

**Key Formulas**
The reward model loss minimizes the negative log odds of the preferred summary’s score relative to the rejected one:
$$\mathcal{L}_{RM}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} [\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))]$$
where $x$ is the prompt, $y_w$ and $y_l$ are the preferred and less-preferred summaries, $r_\theta$ is the RM’s scalar output, and $\mathcal{D}$ is the human judgment dataset. The RL reward function combines the RM output with a KL divergence penalty to constrain policy deviation:
$$R(x, y) = r_\theta(x, y) - \beta \log \frac{\pi_\phi(y|x)}{\pi_{ref}(y|x)}$$
where $\pi_\phi$ is the RL policy, $\pi_{ref}$ is the initial supervised policy, and $\beta$ controls the penalty strength. This KL term acts as an entropy bonus to encourage exploration and prevents the policy from collapsing or generating outputs outside the RM’s training distribution.

**Quantitative Results**
Experiments utilize a filtered TL;DR dataset (~123k posts) and a human feedback dataset containing 64,832 summary comparisons. Labeler-researcher agreement reached 77% ± 2%, closely matching researcher-researcher agreement (73% ± 4%). On TL;DR, the 1.3B human feedback (HF) model achieved a 61% raw preference score against human references, significantly outperforming a 6.7B supervised baseline (43%). After controlling for summary length, the 6.7B HF model remained preferred ~65% of the time. In 7-point Likert quality assessments, the 6.7B PPO model achieved perfect (7/7) overall scores 45% of the time, compared to 20% for the supervised baseline and 23% for references. The models also demonstrated strong domain transfer: 6.7B HF models trained on Reddit generated CNN/DM news summaries that nearly matched the quality of models explicitly fine-tuned on CNN/DM, despite producing roughly half the number of tokens.

**Stated Limitations**
The authors identify several constraints. The reward model is an imperfect proxy for human preference due to limited capacity and a narrow training distribution. Over-optimizing against the RM causes true human preferences to degrade and eventually become anti-correlated with RM scores, a phenomenon also observed when optimizing automatic metrics like ROUGE. Summary length acts as a confounding variable, as models naturally learned to generate longer outputs; while length-controlled evaluations still favor the HF models, the dependency remains. Finally, cross-dataset transfer evaluations on CNN/DM are complicated by significant length mismatches between the transferred models and domain-specific baselines, making direct token-for-token comparisons difficult.
