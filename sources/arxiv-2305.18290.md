---
id: arxiv:2305.18290
type: paper
title: 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model'
url: https://arxiv.org/abs/2305.18290
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

The core problem addressed is the alignment of large-scale unsupervised language models with human preferences. Standard Reinforcement Learning from Human Feedback (RLHF) pipelines are computationally intensive, unstable, and require complex multi-stage training: first fitting an explicit reward model to preference data, then fine-tuning the language model via reinforcement learning (e.g., PPO) to maximize the learned reward while constraining KL divergence from a reference policy. This process demands continuous sampling from the policy during training, significant hyperparameter tuning, and incurs high computational costs.

Direct Preference Optimization (DPO) resolves this by eliminating explicit reward modeling and reinforcement learning. The method leverages a change of variables to express the optimal policy of the KL-constrained reward maximization objective in closed form, enabling direct policy optimization via a simple classification loss. The DPO recipe proceeds as follows: (1) initialize a reference policy $\pi_{\text{ref}}$, typically a supervised fine-tuned (SFT) model; (2) construct an offline dataset $\mathcal{D} = \{(x^{(i)}, y_w^{(i)}, y_l^{(i)})\}_{i=1}^N$ containing prompts and human preferences over paired completions; (3) optimize a parameterized policy $\pi_\theta$ by minimizing the DPO loss, which acts as a maximum likelihood objective over the preference distribution; and (4) train using standard supervised fine-tuning without in-loop sampling or extensive hyperparameter search.

The theoretical foundation begins with the standard RLHF objective:
\[
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} [r_\phi(x,y)] - \beta \mathbb{D}_{\text{KL}}[\pi_\theta(y|x) || \pi_{\text{ref}}(y|x)].
\]
The optimal policy under this objective takes the form $\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r(x,y))$, where $Z(x)$ is a partition function. By rearranging this relationship, the reward function can be reparameterized as $r(x,y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$. Substituting this reparameterization into the Bradley-Terry preference model causes the partition function to cancel, yielding a preference probability dependent solely on the policy. This derivation produces the DPO objective:
\[
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right].
\]
The gradient of this loss increases the likelihood of preferred completions and decreases dispreferred ones, weighted dynamically by how incorrectly the implicit reward model orders the completions.

Quantitative evaluations demonstrate DPO's efficacy across multiple benchmarks. In controlled sentiment generation, DPO produces the most efficient reward-KL frontier, strictly dominating PPO across all tested KL targets and outperforming PPO even when the latter accesses ground-truth rewards. On the Reddit TL;DR summarization dataset using GPT-J, DPO achieves a ~61% win rate against reference summaries at temperature 0.0, surpassing PPO's 57% at its optimal temperature, while exhibiting greater robustness to sampling temperature variations. For single-turn dialogue on the Anthropic Helpful and Harmless dataset (Pythia-2.8B), DPO is the only computationally efficient method to improve upon the dataset's preferred completions, matching the performance of a computationally expensive "Best of 128" baseline. Furthermore, DPO generalizes well to out-of-distribution inputs (CNN/DailyMail), achieving win rates of 0.36 and 0.31 at temperatures 0 and 0.25, respectively, compared to PPO's 0.26 and 0.23. Human evaluation studies confirm that GPT-4 automated judgments correlate strongly with human preferences, with agreement rates comparable to inter-annotator reliability.

The authors acknowledge several limitations. The out-of-distribution generalization properties of DPO relative to explicit reward learning require further investigation. It remains unclear how reward over-optimization manifests within the DPO framework, and whether observed performance dips indicate this phenomenon. The methodology has only been evaluated on models up to 6B parameters, leaving scaling to larger architectures as future work. Additionally, the authors note that automated evaluation win rates are sensitive to prompt formulation, necessitating better methods for eliciting high-quality judgments from automated evaluators. Finally, the potential for leveraging self-labeling from DPO policies to utilize unlabeled prompts remains an open direction.
