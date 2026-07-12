---
id: arxiv:2009.01325
type: paper
title: Simple, Scalable, and Effective Reinforcement Learning from Human Feedback
url: https://arxiv.org/abs/2009.01325
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

The authors address the problem of training and evaluating summarization models, noting that traditional methods like training on human reference summaries and evaluating with ROUGE are imperfect proxies for actual summary quality. They propose a method to significantly improve summary quality by training a model to optimize for human preferences.

**Core Problem:**
The core problem is the misalignment between standard fine-tuning objectives (e.g., maximizing log probability of human-written text) and the desired outcome of generating high-quality outputs as determined by humans. This misalignment leads to issues such as models making important errors (e.g., fabricating facts) or producing undesirable artifacts (e.g., repetition) when optimizing for metrics like ROUGE, which often correlate poorly with human judgment.

**Method/Recipe Step by Step:**
The method, adapted to a batch setting, involves three iterative steps:

1.  **Collect samples and human comparisons:**
    *   Summaries are sampled from various sources, including the current policy, an initial supervised policy, original human reference summaries, and other baselines for a given Reddit post.
    *   Human evaluators are presented with pairs of these summaries and asked to select the preferred one.
    *   The authors transitioned to an offline setting for collecting large batches of comparison data.
    *   They maintained a hands-on relationship with labelers, providing detailed instructions, answering questions, and offering regular feedback to ensure high agreement with researcher judgments.

2.  **Learn a reward model (RM) from human comparisons:**
    *   A Transformer decoder model, initialized from a supervised baseline, is trained with a randomly initialized linear head to output a scalar value.
    *   This reward model is trained to predict the log odds that one summary ($y_i$) is preferred over another ($y_{1-i}$) given a post ($x$) and a dataset of human judgments ($D$).
    *   The RM's output is normalized such that reference summaries achieve a mean score of 0.

3.  **Optimize a policy against the reward model:**
    *   A policy, also a Transformer decoder initialized from a supervised model, is fine-tuned using reinforcement learning (specifically, the PPO algorithm).
    *   The scalar output of the reward model ($r_\theta(x, y)$) is used as the reward for the entire generated summary.
    *   A KL divergence penalty term is included in the reward function to prevent the policy from deviating too far from the initial supervised model and to encourage exploration.
    *   A separate Transformer network is used for the PPO value function, initialized with the reward model's parameters.

**Key Formulas in LaTeX:**

1.  **Reward Model Loss:**

$$
\text{loss}(r_\theta) = -E_{(x,y_0,y_1,i)\sim D}[\log(\sigma(r_\theta(x,y_i) - r_\theta(x,y_{1-i})))]
$$

    where $r_\theta(x, y)$ is the scalar output of the reward model for post $x$ and summary $y$ with parameters $\theta$, and $D$ is the dataset of human judgments.

2.  **Reinforcement Learning Reward Function:**

$$
R(x, y) = r_\theta(x, y) - \beta \log[\pi_\phi^{\text{RL}}(y|x)/\pi^{\text{SFT}}(y|x)]
$$

    where $r_\theta(x, y)$ is the reward model's output, $\beta$ is a KL coefficient, $\pi_\phi^{\text{RL}}(y|x)$ is the learned RL policy with parameters $\phi$, and $\pi^{\text{SFT}}(y|x)$ is the original supervised model.

**Key Quantitative Results and Numbers:**

*   **Human Preference Scores (TL;DR dataset):**
    *   1.3B human feedback model: 61% preference over human reference summaries.
    *   6.7B human feedback model: Significantly outperforms the 1.3B model and human reference summaries.
    *   1.3B human feedback model significantly outperforms a 6.7B supervised model (61% vs. 43% raw preference score against reference summaries).
*   **Summary Length Control:** After controlling for length, the preference of human feedback models vs. reference summaries drops by ~5%, but the 6.7B model summaries are still preferred ~65% of the time.
*   **Summary Quality Axes (TL;DR dataset):** Human feedback models outperform supervised baselines across coverage, accuracy, coherence, and overall quality, particularly in coverage. The 6.7B PPO model achieves a 7/7 overall score 45% of the time (compared to 20% for 6.7B supervised baseline and 23% for reference summaries).
*   **Transfer to CNN/DM:** The 6.7B human feedback model, trained on TL;DR, performs almost as well as a 6.7B model fine-tuned on CNN/DM reference summaries, despite generating much shorter summaries (about half as many tokens on average).
*   **Reward Model Scaling:**
    *   Doubling training data leads to ~1.1% increase in reward model validation accuracy.
    *   Doubling model size leads to ~1.8% increase in reward model validation accuracy.
    *   The 6.7B reward model nearly matches the inter-labeler agreement value of 66.9% on CNN/DM, agreeing with labeler preferences 66.5% of the time.
*   **Reward Model Sensitivity:**
    *   RMs prefer human-edited summaries (minimal edits for improvement) 79.4% (1.3B) and 82.8% (6.7B) of the time, compared to 84.1% for human evaluators.
    *   RMs reliably select original summaries over perturbed ones with reversed participant roles (92.9% for 1.3B, 97.2% for 6.7B).
    *   RMs are biased towards longer summaries: 6.7B RM prefers improving edits that make summaries shorter only 62.6% of the time (vs. 76.4% for humans).
*   **Automatic Metrics Comparison:**
    *   Learned reward models consistently outperform ROUGE, summary length, amount of copying, and log probability in predicting human preferences.
    *   ROUGE agreement with labelers drops from ~57% (supervised baselines) to ~50% (human feedback models).
    *   Log probability agreement with humans drops to ≤50% on comparisons between human feedback model samples.
*   **Human Data Quality:**
    *   Labelers agree with researchers 77% ± 2% of the time.
    *   Researchers agree with each other 73% ± 4% of the time.
    *   Labelers agree with each other 72% of the time in the training corpus. Taking the modal label from 3 labelers increases agreement with researchers from 72% to 77%.

**Stated Limitations:**

*   **Cost and Time:** The method is expensive and time-consuming. Fine-tuning the 6.7B model with RL required approximately 320 GPU-days. Data collection involved thousands of labeler hours and significant researcher time. This prevented collecting an equivalent amount of high-quality human demonstrations for supervised baselines.
*   **Over-optimization:** Optimizing too strongly against the reward model can lead to overfitting, where the reward model becomes anti-correlated with true human preferences.
*   **Labeler Drift:** The criteria used by human labelers to evaluate summaries might gradually shift over time, potentially leading to a regression in labeler-researcher agreement.
*   **Dataset Specificity:** The TL;DR dataset is heavily skewed towards relationship advice (about two-thirds of the dataset), raising concerns about the generality of models trained solely on this data, although transfer performance to CNN/DM was strong.
*   **Bias and Harmful Content:** The TL;DR dataset, being user-submitted with minimal moderation, often contains offensive content or harmful social biases. This means models trained on it can generate biased or offensive summaries.
*   **Job Automation:** Improving ML algorithms to perform tasks previously done by humans could lead to significant job loss.
