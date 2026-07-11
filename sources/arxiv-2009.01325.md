---
id: arxiv:2009.01325
type: paper
title: Learning to summarize from human feedback
url: https://arxiv.org/abs/2009.01325
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

The core problem addressed in this work is the misalignment between standard language model fine-tuning objectives and actual human quality judgments. Traditional summarization models are trained via supervised learning to maximize the likelihood of human reference summaries and evaluated using ROUGE. This maximum-likelihood objective fails to distinguish between critical errors (e.g., factual hallucinations) and trivial ones (e.g., synonym selection), incentivizes probability mass on low-quality demonstrations, and suffers from distributional shift during sampling. Furthermore, ROUGE correlates poorly with human preferences, making it an unreliable proxy for summary quality. To resolve this, the authors propose a pipeline that directly optimizes models for human preferences.

The methodology follows a three-step iterative recipe. First, human feedback is collected by sampling summaries from multiple policies and baselines for Reddit posts, then presenting pairwise comparisons to human labelers who select the preferred summary. Second, a reward model is trained via supervised learning on these comparisons to predict the human-preferred output. Third, a policy is fine-tuned using reinforcement learning (PPO) to maximize the reward model’s output as a reward signal. The process can be repeated iteratively by gathering new human data from the updated policy and retraining.

The reward model is initialized from a supervised baseline with a linear head and trained to minimize the negative log-likelihood of the preferred summary’s log-odds:
$$\text{loss}(r_\theta) = -E_{(x,y_0,y_1,i)\sim D}[\log(\sigma(r_\theta(x,y_i) - r_\theta(x,y_{1-i})))]$$
where $r_\theta(x, y)$ is the scalar reward output for post $x$ and summary $y$, and $D$ is the dataset of human judgments. During policy optimization, the total reward incorporates a KL divergence penalty relative to the initial supervised policy $\pi^{\text{SFT}}$:
$$R(x, y) = r_\theta(x, y) - \beta \log[\pi_\phi^{\text{RL}}(y|x)/\pi^{\text{SFT}}(y|x)]$$
This penalty acts as an entropy bonus to encourage exploration and prevents the policy from diverging too far from the training distribution.

Quantitative evaluations demonstrate significant improvements over supervised baselines. The study utilizes a filtered TL;DR dataset of 123,169 posts (24–48 tokens) and collects 64,832 human comparisons. Labeler-researcher agreement reaches $77\% \pm 2\%$, comparable to researcher-researcher agreement ($73\% \pm 4\%$). On the TL;DR dataset, the 1.3B human feedback model is preferred over human references 61% of the time, significantly outperforming a 6.7B supervised baseline (43%). Length-controlled evaluations show the 6.7B human feedback model preferred ~65% of the time. The models generalize to the CNN/DM news dataset without task-specific fine-tuning, nearly matching the quality of a T5 model explicitly trained on CNN/DM references. Scaling analyses indicate that doubling training data yields a ~1.1% increase in reward model validation accuracy, while doubling model size yields a ~1.8% increase. The 6.7B reward model approaches human inter-labeler agreement (66.9%). Direct optimization against the reward model consistently outperforms ROUGE optimization in human preference metrics, whereas ROUGE agreement drops to ~50% for human feedback models compared to the reward model’s stable 62%.

The authors identify several limitations. The approach incurs substantial computational and financial costs, requiring approximately 320 GPU-days for 6.7B reinforcement learning fine-tuning and thousands of labeler hours for data collection. The reward model exhibits a bias toward longer summaries, preferring length-increasing edits only 62.6% of the time versus 76.4% for humans. Over-optimization against the reward model can lead to reward hacking and degraded quality, mirroring issues observed with ROUGE. Additionally, training on unmoderated Reddit content introduces risks of generating biased or offensive summaries, and the methodology could be misused for persuasive or toxic content generation. The authors note that defining "good" behavior for complex tasks requires careful stakeholder inclusion to avoid reinforcing narrow or harmful preferences.
