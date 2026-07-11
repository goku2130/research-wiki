---
id: arxiv:2402.09345
type: paper
title: Mitigating Reward Hacking via Information-Theoretic Reward Models
url: https://arxiv.org/abs/2402.09345
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-model-overoptimization
---

**Core Problem**
Reward hacking, or reward overoptimization, is a critical instability in reinforcement learning from human feedback (RLHF). It stems from reward misgeneralization, where proxy reward models (RMs) overfit to spurious, preference-irrelevant features in training data (e.g., response length bias). During RL optimization, the policy model exploits these artifacts, causing proxy rewards to rise while true human preference scores decline, resulting in outputs that are overly verbose, excessively cautious, or semantically hollow.

**Method/Recipe Step by Step**
InfoRM addresses misgeneralization by reformulating reward modeling through a variational information bottleneck (IB) objective. The framework compresses the input into a latent representation $\boldsymbol{S}$ that retains only human-preference-relevant information. The implementation proceeds as follows:
1. **Encoding:** Pass chosen ($\boldsymbol{x}^w$) and rejected ($\boldsymbol{x}^l$) responses through an LLM backbone encoder $f_\phi(\cdot)$ to extract hidden states.
2. **Latent Parameterization:** Compute the mean $f_\phi^\mu(\boldsymbol{x})$ and log-variance $f_\phi^\sigma(\boldsymbol{x})$ of a multivariate Gaussian with diagonal covariance.
3. **Reparameterization Sampling:** Draw independent noise $\boldsymbol{\epsilon}^w, \boldsymbol{\epsilon}^l \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$ and compute latent vectors $\boldsymbol{s} = h_\phi(\boldsymbol{x}, \boldsymbol{\epsilon}) = f_\phi^\mu(\boldsymbol{x}) + f_\phi^\sigma(\boldsymbol{x})\boldsymbol{\epsilon}$.
4. **Reward Decoding:** Feed $\boldsymbol{s}$ into a separate MLP decoder $g_\psi(\cdot)$ to produce scalar reward predictions.
5. **Optimization:** Minimize a variational lower bound that maximizes preference prediction accuracy while penalizing the KL divergence between the latent posterior and a standard normal prior $r(\boldsymbol{S}) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$.
6. **Detection Metric:** Post-training, cluster the IB latent space using DBSCAN and compute the Cluster Separation Index (CSI) to quantify outlier deviations, signaling overoptimization.

**Key Formulas**
The theoretical IB objective balances preference utility and input irrelevance:
$$\max_{\boldsymbol{\theta}} I(\boldsymbol{S}; Y) - \beta I(\boldsymbol{X}; \boldsymbol{S}|Y)$$
The practical training loss approximates the variational lower bound:
$$L_{\text{preference}} = \log \sigma \left( g_{\psi}(h_{\phi}(\boldsymbol{x}^w,\boldsymbol{\epsilon}^w)) - g_{\psi}(h_{\phi}(\boldsymbol{x}^l,\boldsymbol{\epsilon}^l)) \right)$$
$$L_{\text{bottleneck}} = \text{KL} \left[ p_{\phi}(\boldsymbol{S}|\boldsymbol{x}^w), r(\boldsymbol{S}) \right] + \text{KL} \left[ p_{\phi}(\boldsymbol{S}|\boldsymbol{x}^l), r(\boldsymbol{S}) \right]$$
The CSI detection metric is calculated as:
$$\mathrm{CSI} = \sum_{i=1}^n |C_i| \cdot \min_{\mathbf{s} \in S} \|\mathbf{c}_i - \mathbf{s}\|$$
where $C_i$ denotes DBSCAN clusters, $\mathbf{c}_i$ their geometric centroids, and $S$ the set of SFT model outputs.

**Key Quantitative Results**
Simulated RLHF experiments with 25% label noise demonstrate that InfoRM maintains stable gold reward growth, whereas standard RMs exhibit severe overoptimization. Real-world GPT-4 evaluations confirm consistent superiority across benchmarks: on Anthropic-Helpful, InfoRM achieves a 57.0% win rate versus 54.5% for standard RMs; on Anthropic-Harmless, 57.1% versus 54.2%; on AlpacaFarm, 48.9% versus 45.1%; and on TL;DR Summary, 73.1% versus 70.4%. InfoRM also shows stronger out-of-distribution generalization on the Flan dataset. On reward model benchmarks, InfoRM achieves 66.63% accuracy on AlpacaEval and 46.87% on Truthful QA (MC), outperforming standard RMs (65.38% and 40.63%, respectively). The CSI metric reliably tracks overoptimization, showing abrupt spikes for standard RMs (e.g., between 600–700 training steps) while remaining consistently low for InfoRM.

**Stated Limitations**
The authors identify three primary limitations. First, scaling InfoRM to state-of-the-art models significantly larger than the evaluated 7B parameters remains unexplored. Second, the CSI-based monitoring mechanism exhibits inference latency and requires evaluation on test datasets, highlighting a need for real-time, lightweight detection metrics. Third, automated win-rate evaluations via GPT-4 are sensitive to prompt structure, indicating that future work must optimize prompt engineering to ensure consistent and reliable judgment quality.
