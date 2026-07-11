---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

**Core Problem**
Aligning large language models (LLMs) with human preferences typically relies on Reinforcement Learning from Human Feedback (RLHF), a multi-stage pipeline that first fits a reward model to preference data and then fine-tunes the policy via reinforcement learning (e.g., PPO) to maximize reward while constraining KL divergence from a reference model. This process is computationally expensive, unstable, and requires significant hyperparameter tuning and in-loop sampling during training.

**Method and Recipe**
Direct Preference Optimization (DPO) reformulates the RLHF objective into a single-stage supervised learning problem by leveraging a change of variables that maps reward functions directly to optimal policies. The DPO training recipe proceeds as follows:
1. **Data Collection:** Sample response pairs $(y_1, y_2)$ from a reference policy $\pi_{\text{ref}}$ (typically a supervised fine-tuned model) for prompts $x$, and annotate them with human preferences to construct an offline dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^N$.
2. **Implicit Reward Parameterization:** Instead of training a separate reward network, DPO implicitly defines the reward as the log-probability ratio of the current policy $\pi_\theta$ relative to $\pi_{\text{ref}}$, scaled by a temperature parameter $\beta$: $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$.
3. **Policy Optimization:** Optimize $\pi_\theta$ by minimizing a binary cross-entropy loss that increases the relative log-probability of preferred completions $y_w$ over dispreferred completions $y_l$. The update is dynamically weighted by the implicit reward's confidence, preventing naive probability ratio objectives from causing model degeneration.
4. **Training:** Perform standard gradient descent on the loss function without sampling from the model or employing RL algorithms, significantly simplifying implementation and reducing computational overhead.

**Key Formulas**
DPO is derived from the Bradley-Terry preference model, which defines the probability of human preference as:
\[
p^*(y_1 \succ y_2 \mid x) = \frac{\exp(r^*(x, y_1))}{\exp(r^*(x, y_1)) + \exp(r^*(x, y_2))}.
\]
The optimal policy under the standard KL-constrained reward maximization objective takes the form:
\[
\pi_r(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp \left( \frac{1}{\beta} r(x, y) \right),
\]
where $Z(x)$ is a partition function. By algebraically rearranging this expression to solve for the reward $r(x,y)$ and substituting it back into the preference model, the partition function cancels out. This yields the DPO objective, which optimizes the policy directly:
\[
\mathcal{L}_{\mathrm{DPO}}(\pi_{\theta}; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_{w}, y_{l}) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_{\theta}(y_{w} \mid x)}{\pi_{\mathrm{ref}}(y_{w} \mid x)} - \beta \log \frac{\pi_{\theta}(y_{l} \mid x)}{\pi_{\mathrm{ref}}(y_{l} \mid x)} \right) \right].
\]

**Quantitative Results**
Experiments across three tasks demonstrate DPO's efficacy:
- **Controlled Sentiment Generation:** DPO produces the most efficient reward-KL frontier, strictly dominating PPO and even PPO with ground-truth rewards (PPO-GT) across all tested KL constraints.
- **TL;DR Summarization (GPT-J):** DPO achieves a ~61% win rate against reference summaries at temperature 0.0, outperforming PPO's 57% at its optimal temperature. DPO also exhibits greater robustness to sampling temperature variations.
- **Single-Turn Dialogue (Anthropic HH, Pythia-2.8B):** DPO is the only computationally efficient method to improve over the dataset's preferred completions, matching the performance of the computationally intensive "Best of 128" baseline.
- **Out-of-Distribution Generalization:** On the CNN/DailyMail dataset, DPO maintains higher GPT-4 win rates versus ground-truth summaries (0.36 at temp 0, 0.31 at temp 0.25) compared to PPO (0.26, 0.23).
- **Evaluation Validity:** Human studies confirm that GPT-4 judgments correlate strongly with human preferences, with agreement rates ranging from 65% to 87%.

**Stated Limitations**
The authors identify several limitations and directions for future work. DPO's out-of-distribution generalization relative to explicit reward learning requires more comprehensive study. The manifestation of reward over-optimization within the DPO framework remains unclear, particularly regarding minor performance decreases observed during training. The method has only been evaluated on models up to 6B parameters, leaving scaling to larger architectures unexplored. Additionally, automated evaluation win rates are sensitive to prompt formulation, necessitating better methods for eliciting high-quality judgments from LLMs. Finally, the application of DPO to generative models in modalities beyond text has not been investigated.
