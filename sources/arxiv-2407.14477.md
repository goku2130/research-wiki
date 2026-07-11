---
id: arxiv:2407.14477
type: paper
title: Data-Centric Human Preference with Rationales for Direct Preference Alignment
url: https://arxiv.org/abs/2407.14477
retrieved: '2026-07-11'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Data-Centric Human Preference with Rationales for Direct Preference Alignment

### Core Problem
Standard preference datasets used for aligning Large Language Models (LLMs) typically consist of a prompt and a pair of responses (one preferred, one rejected). The authors argue that these datasets are "opaque" because they lack explicit information regarding *why* one response is preferred over another. This ambiguity can lead to data inefficiency, the learning of spurious correlations (such as a bias toward longer responses), and a requirement for prohibitively expensive human annotation to achieve robust alignment.

### Method
The authors propose a data-centric framework that augments standard preference pairs with **rationales**—explanations that detail the reasoning behind the human preference.

**Step-by-Step Recipe:**
1. **Dataset Augmentation:** A standard pairwise preference dataset $\mathcal{D} = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}\}_{i=1}^{N}$ (where $x$ is the context, $y_w$ is the winning response, and $y_l$ is the losing response) is enriched into $\mathcal{D}' = \{x^{(i)}, y_w^{(i)}, y_l^{(i)}, r^{(i)}\}_{i=1}^{N}$ by adding machine-generated rationales $r$.
2. **Formulation of Rationale-Enriched Preference:** The preference function is extended to $p^*(y_w \succ y_l, r | x)$. Using the chain rule, this is decomposed into the pairwise preference term and the probability of the rationale given the context and preference: $p^*(r | x, y_w \succ y_l) = \pi_\theta(r | x, y_w \succ y_l)$.
3. **Loss Function Adaptation (RDPO):** The Direct Preference Optimization (DPO) loss is modified to include a rationale prediction term. The resulting Rationale-DPO (RDPO) objective is:

$$
\mathcal{L}_{\mathsf{RDPO}}(\pi_{\theta};\pi_{\mathrm{ref}})=-\mathbb{E}_{(x,y_{\mathrm{W}},y_{\mathrm{L}},r)\sim\mathcal{D}'}\left[\log\sigma\left(\beta\log\frac{\pi_{\theta}(y_{\mathrm{W}}|x)}{\pi_{\mathrm{ref}}(y_{\mathrm{W}}|x)}-\beta\log\frac{\pi_{\theta}(y_{\mathrm{L}}|x)}{\pi_{\mathrm{ref}}(y_{\mathrm{L}}|x)}\right)+\gamma\log\pi_{\theta}(r|x,y_{\mathrm{W}}\succ y_{\mathrm{L}})\right]
$$

where $\beta$ is the divergence hyperparameter and $\gamma$ weights the impact of the rationales.
4. **Versatility:** This framework is similarly adapted to other algorithms, creating RORPO (Rationale-ORPO) and RSimPO (Rationale-SimPO).

### Key Quantitative Results
*   **Annotation Efficiency:** On the Orca dataset, RDPO achieved a 60% win rate against the SFT baseline using only 3,000 training samples, whereas standard DPO required 9,000 samples to reach the same rate. RDPO eventually reached a win rate above 66%.
*   **Direct Comparison:** RDPO outperformed DPO with a win rate exceeding 60% on both the Orca and UltraFeedback datasets.
*   **Ablation (Rationale-Only):** Training with only the rationale loss (setting DPO alignment loss to zero) achieved a win rate of over 61% against SFT on the Orca dataset, demonstrating that rationales inherently encode preference signals.
*   **AlpacaEval 2.0 (Length-Controlled Win Rate - LC WR):**
    *   **Mistral-7B-Instruct-v0.2:** RDPO achieved 22.42% LC WR, compared to 19.52% for DPO and 17.11% for the original model.
    *   **Llama-3-8B-Instruct:** RDPO achieved 27.55% LC WR, compared to 26.02% for DPO and 22.92% for the original model.
*   **RSimPO Performance:** RSimPO achieved win rates over 65% against the SimPO model on both the UltraFeedback and Anthropic HH datasets.

### Theoretical Insights
Using information theory, the authors analyze the conditional mutual information $I(Z; g(R)|S)$ between true preferences $Z$ and rationale-implied preferences $g(R)$ given the input-response pair $S$. They find that rationales are most beneficial when reward modeling based on query-response pairs alone is biased or overconfident. Furthermore, they derive generalization bounds suggesting that training with rationales reduces sample complexity provided the rationales do not contain excessive irrelevant information (small $\eta_1$).

### Stated Limitations
*   **Model Scale:** The authors have only successfully trained models up to 8 billion parameters; the impact on larger models remains unexplored.
*   **Dataset Scope:** The current method is designed for pairwise preferences; it needs to be extended to handle unpaired responses (e.g., for use with KTO).
*   **Reward Modeling:** The specific impact of rationales on training explicit reward models (as opposed to direct alignment) requires further study to distinguish them from existing "critique" literature.
