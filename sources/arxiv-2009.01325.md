---
id: arxiv:2009.01325
type: paper
title: Learning to summarize with human feedback
url: https://arxiv.org/abs/2009.01325
retrieved: '2026-07-11'
maturity: comprehensive
topic: ppo-for-llms
---

This paper introduces a method to significantly improve summary quality by training language models with human feedback, addressing the limitations of traditional supervised learning and ROUGE metrics.

**Core Problem:**
Traditional summarization models are trained to predict human reference summaries and evaluated using ROUGE, which are often poor proxies for actual summary quality. This leads to a misalignment between the training objective (maximizing likelihood of human-written text) and the desired outcome (generating high-quality summaries as judged by humans). Issues include:
1.  Lack of distinction between important and unimportant errors.
2.  Models incentivized to reproduce low-quality human demonstrations.
3.  Performance degradation due to distributional shift during sampling.
4.  ROUGE's poor correlation with human judgments.

**Method/Recipe Step by Step:**

The method, adapted to a batch setting, involves three iterative steps:

1.  **Collect Samples and Human Comparisons (Step 1):**
    *   Summaries are sampled from various sources (current policy, initial policy, reference summaries, baselines) for a given Reddit post.
    *   Human evaluators are presented with pairs of these summaries and asked to select the preferred one.
    *   Labelers provide "naive interpretations" of summaries before seeing the original post to identify ambiguities.
    *   Labelers indicate their confidence in preference on a 9-point scale.
    *   Quality control measures include detailed instructions, immediate feedback for labelers, continuous monitoring of labeler-researcher agreement, and researcher comparison calibrations.

2.  **Learn a Reward Model (RM) from Human Comparisons (Step 2):**
    *   A reward model is trained using supervised learning to predict the human-preferred summary.
    *   The RM is initialized from a supervised baseline model, with a randomly initialized linear head outputting a scalar value.
    *   The model is trained to predict the log odds that one summary is better than another, given a post and two candidate summaries.
    *   After training, RM outputs are normalized so that reference summaries have a mean score of 0.

3.  **Optimize a Policy Against the Reward Model (Step 3):**
    *   The logit output of the reward model is treated as a reward signal.
    *   A summarization policy is fine-tuned using reinforcement learning (RL), specifically the Proximal Policy Optimization (PPO) algorithm.
    *   Each time step in PPO corresponds to generating a BPE token, and the reward is given for the entire generated summary.
    *   The policy is initialized from a supervised model fine-tuned on the Reddit TL;DR dataset.
    *   A KL divergence penalty term is added to the reward function to encourage exploration and prevent the policy from deviating too much from the initial supervised model.
    *   A separate Transformer network is used for the PPO value function, initialized with the reward model's parameters.

**Key Formulas in LaTeX:**

1.  **Reward Model Loss:**

$$
\text{loss}(r_\theta) = -E_{(x,y_0,y_1,i)\sim D}[\log(\sigma(r_\theta(x,y_i) - r_\theta(x,y_{1-i})))]
$$

    where $r_\theta(x, y)$ is the scalar output of the reward model for post $x$ and summary $y$ with parameters $\theta$, and $D$ is the dataset of human judgments.

2.  **Full Reward Function for Policy Optimization:**

$$
R(x, y) = r_\theta(x, y) - \beta \log[\pi_\phi^{\text{RL}}(y|x)/\pi^{\text{SFT}}(y|x)]
$$

    where $r_\theta(x, y)$ is the reward model output, $\pi_\phi^{\text{RL}}$ is the learned RL policy with parameters $\phi$, $\pi^{\text{SFT}}$ is the initial supervised model, and $\beta$ is the KL coefficient.

**Key Quantitative Results and Numbers:**

*   **TL;DR Dataset Performance:**
    *   1.3B human feedback model significantly outperforms a 6.7B supervised model (61% vs. 43% raw preference score against reference summaries).
    *   6.7B human feedback model further outperforms the 1.3B model.
    *   Both human feedback models are preferred over the original human demonstrations in the dataset.
    *   After controlling for length, the preference of 6.7B human feedback model summaries over reference summaries drops by ~5%, but still preferred ~65% of the time.
    *   6.7B PPO model summaries achieve a 7/7 overall quality score 45% of the time (compared to 20% for 6.7B supervised baseline and 23% for reference summaries).
*   **Transfer to CNN/DM:**
    *   Reddit-trained human feedback models generate high-quality summaries of CNN/DM news articles without news-specific fine-tuning.
    *   The 6.7B human feedback model performs almost as well as a 6.7B model fine-tuned on CNN/DM reference summaries, despite generating much shorter summaries.
*   **Human Agreement Rates:**
    *   Labelers agree with researchers $77\% \pm 2\%$ of the time.
    *   Researchers agree with each other $73\% \pm 4\%$ of the time.
    *   Labelers agree with each other 72% of the time in the training corpus.
*   **Reward Model Scaling:**
    *   Doubling training data leads to a ~1.1% increase in RM validation accuracy.
    *   Doubling model size leads to a ~1.8% increase in RM validation accuracy.
    *   The 6.7B reward model nearly matches the inter-labeler agreement value of 66.9% on CNN/DM.
    *   RMs prefer human-edited summaries over original ones 79.4% (1.3B) and 82.8% (6.7B) of the time, compared to 84.1% for human evaluators.
    *   RMs reliably select original summaries over role-reversed perturbed summaries (92.9% for 1.3B, 97.2% for 6.7B).
    *   RMs show a bias towards longer summaries: 6.7B RM prefers improving edits that make summaries shorter only 62.6% of the time (vs. 76.4% for humans).
*   **Automatic Metrics Comparison:**
    *   Learned reward models consistently outperform ROUGE, summary length, copying amount, and log probability as predictors of human preferences.
    *   ROUGE agreement with labelers drops from ~57% (supervised baseline samples) to ~50% (human feedback model samples).
    *   Log probability agreement with humans drops to $\le$50% for human feedback model comparisons.
    *   Optimizing ROUGE peaks sooner and at a substantially lower quality rate than optimizing reward models.
*   **Training Costs:** Fine-tuning the 6.7B model with RL required approximately 320 GPU-days.

**Stated Limitations:**

*   **Time and Cost:** Producing the final models is expensive, requiring thousands of labeler hours and significant researcher time.
*   **Lack of Baselines:** Due to cost, an equivalent amount of high-quality human demonstrations for supervised baselines was not collected.
*   **Labeler Drift:** Potential for labeler criteria to shift over time, though monitoring indicated relative stability.
*   **Over-optimization:** Optimizing too strongly against the reward model can lead to overfitting, where the reward model becomes anti-correlated with true human preferences.
*   **Domain Specificity:** The TL;DR dataset is heavily skewed towards relationship advice (two-thirds of posts), raising concerns about generality, though transfer to CNN/DM suggests some generalization.
*   **Bias and Harmful Content:** The Reddit TL;DR dataset contains offensive or biased content, meaning models trained on it can generate biased or offensive summaries. Thorough study of potential harms is recommended before deployment.
*   **Societal Impact:** Improved automation capabilities could lead to job displacement.
*   **Defining "Good" Behavior:** For complex tasks, defining "good" model behavior where humans might disagree requires careful consideration and inclusion of diverse perspectives.
