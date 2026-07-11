---
id: rlhfbook:direct-alignment-algorithms
type: web
title: Direct-Alignment Algorithms
url: https://rlhfbook.com/c/08-direct-alignment
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

Direct-Alignment Algorithms (DAAs) address the computational and implementation complexity of traditional Reinforcement Learning from Human Feedback (RLHF). Standard RLHF requires training an intermediate reward model and employing reinforcement learning optimizers to solve a constrained preference objective. DAAs bypass these components, directly optimizing the policy to maximize the same RLHF objective using only offline preference pairs $(x, y_c, y_r)$, where $x$ denotes prompts and $y_c, y_r$ denote chosen and rejected completions. This approach significantly reduces compute overhead and simplifies experimentation while targeting identical alignment goals.

The canonical DAA, Direct Preference Optimization (DPO), operates by replacing explicit reward modeling with an implicit reward defined as a scaled log-probability ratio relative to a frozen reference model $\pi_{\text{ref}}$. The optimization targets the constrained RLHF objective:

$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}}\mathbb{E}_{y \sim \pi(y|x)} \left[r_\theta(x, y)\right] - \beta \mathcal{D}_{\text{KL}}\left(\pi(y|x) \| \pi_{\text{ref}}(y|x)\right)
$$

The derivation proceeds sequentially by expanding the KL-divergence term, introducing a partition function $Z(x)$, and recognizing the resulting expression as a KL-divergence between the target policy and an unnormalized Gibbs distribution. Minimizing this divergence yields the optimal policy in closed form:

$$
\pi^*(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp\left(\frac{1}{\beta}r(x,y)\right)
$$

Substituting this optimal policy into a Bradley-Terry preference model and solving for the reward yields the implicit reward formulation:

$$
r(x, y) = \beta \log \frac{\pi_r(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

The resulting DPO loss function directly differentiates the negative log-likelihood of observed preferences:

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_c, y_r) \sim \mathcal{D}}\left[ \log \sigma\left( \beta \log \frac{\pi_{\theta}(y_c \mid x)}{\pi_{\text{ref}}(y_c \mid x)} - \beta \log \frac{\pi_{\theta}(y_r \mid x)}{\pi_{\text{ref}}(y_r \mid x)} \right) \right]
$$

The gradient update scales the difference in log-probability gradients for chosen and rejected responses by a weighting factor $w = \sigma\!\left(r_{\theta}(x, y_r) - r_{\theta}(x, y_c)\right)$ and the temperature parameter $\beta$:

$$
\nabla_{\theta}\mathcal{L}_{\text{DPO}}(\pi_{\theta}; \pi_{\text{ref}}) = -\beta \mathbb{E}_{(x, y_c, y_r)\sim \mathcal{D}}\left[ w \cdot \left(\nabla_{\theta}\log \pi_{\theta}(y_c \mid x) - \nabla_{\theta}\log \pi_{\theta}(y_r \mid x)\right) \right]
$$

Practically, implementation computes log-probability gaps for the policy and reference model, calculates logits as their difference, and applies a negative log-sigmoid loss. Caching reference model log-probabilities prior to training reduces peak GPU memory usage by 50%. Since its release, DPO and its variants have been adopted in prominent models including Zephyr-$\beta$, Llama 3 Instruct, Tülu 2 and 3, and Nemotron 4 340B, though initial deployments required surprisingly low learning rates.

Despite its simplicity, DPO exhibits several documented limitations. The algorithm treats all preference pairs with equal weight, ignoring richer reward margins or multi-class preference signals. Optimization exclusively widens the probability gap between chosen and rejected responses, which can inadvertently reduce the absolute probabilities of both outputs, a phenomenon termed preference displacement that may increase unaddressed token probabilities. Furthermore, DPO relies entirely on offline completions from previous or external models, limiting the freshness of the training signal. The hyperparameter $\beta$ explicitly fixes the KL divergence target, requiring careful tuning that depends heavily on the base model and dataset distribution. To mitigate these issues, variants such as REBEL (incorporating reward margins), cDPO and IPO (addressing label noise and overfitting), ORPO (removing the reference model requirement), and SimPO (modifying log-probability averaging) have been proposed. Online adaptations like Online DPO and D2PO further alleviate offline data constraints by generating fresh completions and dynamically relabeling preferences during training.
