---
id: arxiv:2406.02900
type: paper
title: Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms
url: https://arxiv.org/abs/2406.02900
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-model-overoptimization
---

The paper investigates the phenomenon of reward over-optimization, or "reward hacking," within Direct Alignment Algorithms (DAAs) such as Direct Preference Optimization (DPO), Identity Preference Optimization (IPO), and Sequence Likelihood Calibration with Human Feedback (SLiC-HF). While DAAs bypass the explicit reward modeling stage of classical Reinforcement Learning from Human Feedback (RLHF) by directly optimizing the policy via preference data, they still exhibit severe degradation in true model quality as optimization intensifies. This over-optimization manifests as a hump-shaped performance curve relative to the Kullback-Leibler (KL) divergence budget, where increasing the KL constraint initially improves alignment but subsequently causes performance to deteriorate. Notably, this degradation often occurs intra-epoch, with models frequently reaching peak performance after processing only a fraction of the training data before diverging from the reference policy.

To formalize this behavior, the authors unify various DAA objectives under a single framework and execute a systematic empirical recipe. The procedure begins with Supervised Fine-Tuning (SFT) on the Reddit TL;DR summarization dataset (92K preferred-dispreferred pairs), followed by direct alignment training using DPO, IPO, or SLiC loss formulations. The experimental design evaluates three Pythia model scales (1B, 2.8B, and 6.9B parameters) across seven $\beta$ hyperparameters representing different KL budgets. Model performance is quantified via GPT-4 win rates against dataset summaries. The methodology further incorporates intra-epoch checkpoint tracking, length-correlation regression, implicit reward accuracy metrics, and a deterministic Tree MDP to isolate out-of-distribution (OOD) trajectory allocation.

The theoretical foundation relies on several key equations. The Bradley-Terry model characterizes preference probabilities as $p(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))$, where $\sigma$ is the logistic function. The general DAA objective is formulated as:

$$
\mathcal{L}_{\text{DAA}}(\pi_{\theta}; \pi_{\text{ref}}) = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[g\left(\beta \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right],
$$

where $g$ denotes a convex loss function (e.g., negative log-sigmoid for DPO, quadratic for IPO). The authors demonstrate that reward over-optimization in DAAs follows a scaling law analogous to classical RLHF:

$$
R(d) = d(\alpha - \beta \log d),
$$

where $d = \sqrt{\mathbb{D}_{\text{KL}}(\pi \| \pi_{\text{ref}})}$, and $\alpha, \beta$ are constants dependent on dataset size and parameter count. This formulation halves the root mean squared error (RMSE) compared to quadratic fits between KL divergence and win rates. Additionally, the expected implicit reward is shown to approximate the forward KL divergence: $\mathbb{E}[\log \frac{\pi_{\theta}(y_w|x)}{\pi_{\text{ref}}(y_w|x)}] \approx -\mathbb{D}_{\text{KL}}[\pi_{\text{ref}} \| \pi_{\theta}]$.

Quantitative results reveal distinct scaling and algorithmic dependencies. The 1B model rapidly achieves high KL values and exhibits immediate over-optimization across all objectives, whereas the 6.9B model demonstrates superior KL control and reduced degradation. IPO consistently achieves lower KL divergences and mitigates over-optimization more effectively than DPO and SLiC. Intra-epoch analysis shows that configurations with wider KL budgets peak after approximately 25% of training before degrading. Furthermore, implicit reward classification accuracy shows negligible correlation with downstream win rates for DPO and SLiC, indicating that fitting preference data does not guarantee policy improvement. Length extrapolation analysis confirms that weaker models or those under strict KL budgets disproportionately optimize for sequence length. Theoretical analysis via Proposition 1 proves that finite preference datasets render the DAA loss non-strictly convex, creating a non-trivial null space that permits infinite optimal policies allocating substantial probability mass to OOD trajectories.

The authors explicitly state several limitations. The study is constrained by computational budgets, preventing evaluation at larger model scales. The analysis assumes the existence of an underlying human preference model, acknowledging that no perfect preference representation exists. The work characterizes over-optimization dynamics but does not propose mitigation strategies. Additionally, due to compute limitations, experiments were not repeated across multiple seeds, precluding error bars or statistical significance testing. The focus remains strictly on offline DAA training, leaving online sampling dynamics unaddressed.
