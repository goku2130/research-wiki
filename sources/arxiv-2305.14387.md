---
id: arxiv:2305.14387
type: paper
title: 'AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback'
url: https://arxiv.org/abs/2305.14387
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-and-winrate-evals
---

**Core Problem**
Training instruction-following large language models (LLMs) via learning from pairwise feedback (LPF) is hindered by three major barriers: the prohibitive cost of human data annotation, the absence of trustworthy automated evaluation protocols, and a lack of validated reference implementations for LPF algorithms. These gaps obscure the development workflow and impede reproducible research.

**Method/Recipe**
AlpacaFarm addresses these challenges through a structured simulation pipeline:
1. **Data Partitioning:** The 52k Alpaca dataset is split into supervised fine-tuning (SFT; 10k), pairwise preference (10k), unlabeled (20k), and validation (2k) subsets.
2. **Simulated Pairwise Feedback:** Human annotators are proxied using oracle API LLMs (GPT-4, ChatGPT, Davinci003). A pool of 13 distinct simulated annotators is constructed by varying base models, prompt formats, batch sizes, and in-context examples. To replicate inter- and intra-annotator variability, the training simulator injects 25% random label noise.
3. **Automatic Evaluation:** An 805-instruction evaluation set is compiled by combining five open-source datasets, guided by real-world Alpaca Demo interactions. Model performance is measured as win-rate against a Davinci003 reference using the noise-free evaluator $p_{\text{sim}}^{\text{eval}}$.
4. **Reference Implementations:** Seven LPF methods are implemented, all initializing from an SFT checkpoint: Binary FeedME, Binary Reward Conditioning, Direct Preference Optimization (DPO), Best-of-$n$ sampling ($n=1024$), Expert Iteration, Proximal Policy Optimization (PPO), and Quark.
5. **End-to-End Validation:** Eleven models/methods are trained and evaluated in simulation. Their performance rankings are directly compared against identical methods trained on actual human feedback to verify simulator fidelity.

**Key Formulas**
Surrogate reward models are trained by maximizing the Bradley–Terry likelihood:
$$\text{maximize}_\phi \sum_j \log \frac{\exp(\hat{R}_\phi(x^{(j)}, y_z^{(j)}))}{\exp(\hat{R}_\phi(x^{(j)}, y_0^{(j)})) + \exp(\hat{R}_\phi(x^{(j)}, y_1^{(j)}))}.$$
DPO optimizes an implicit reward via:
$$\mathbb{E}_{(x, y_0, y_1, z) \sim \mathcal{D}_{\text{pairwise}}} \left[ \log \sigma \left( \beta \log \frac{p_\theta(y_z \mid x)}{p_{\text{SFT}}(y_z \mid x)} - \beta \log \frac{p_\theta(y_{1-z} \mid x)}{p_{\text{SFT}}(y_{1-z} \mid x)} \right) \right].$$
PPO maximizes a KL-regularized objective:
$$\mathbb{E}_{x \sim p(x), y \sim p_\theta(y|x)} \left[ \hat{R}_\phi(x, y) - \beta \log \frac{p_\theta(y \mid x)}{p_{\text{SFT}}(y \mid x)} \right].$$

**Key Quantitative Results**
The simulated annotators achieve a 65% agreement rate with human majority votes, closely matching the 66% human-human agreement rate, while costing only $6 per 1,000 annotations (50× cheaper than crowdworkers). End-to-end validation yields a Spearman correlation of 0.98 between simulated and human method rankings. PPO trained on human feedback achieves a 55% win-rate against Davinci003, up from 44% for the SFT baseline. Best-of-1024 achieves 50.7% (human) and 45.0% (simulation). The evaluation protocol correlates strongly with real-world Alpaca Demo interactions ($r^2 = 0.97$). Crucially, the simulator replicates reward over-optimization dynamics when annotator variability is included, whereas deterministic single-prompt LLM feedback fails to exhibit this phenomenon.

**Stated Limitations**
The framework is validated only on relatively simple, single-turn instructions and exclusively uses LLaMA 7B as a base model. Human validation relies on a small pool of crowdworkers who exhibit biases toward longer outputs and list formats. The simulator assumes access to a powerful oracle LLM, which may not be available in production settings. Furthermore, optimal hyperparameters (e.g., KL regularization coefficients) differ between simulated and human feedback training, and simulated annotators retain specific biases, such as preferring first-seen outputs or same-model generations. Despite these constraints, AlpacaFarm provides a validated, cost-effective environment for rapid LPF method development and transfer to real-world human feedback pipelines.
