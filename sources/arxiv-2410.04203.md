---
id: arxiv:2410.04203
type: paper
title: 'Rainbow PO: A Unified Framework for Combining Improvements in Preference Optimization'
url: https://arxiv.org/abs/2410.04203
retrieved: '2026-07-12'
maturity: comprehensive
topic: dpo-variants
---

# Rainbow PO: A Unified Framework for Combining Improvements in Preference Optimization

### Core Problem
The Direct Preference Optimization (DPO) family has seen a proliferation of variants (collectively termed "XPOs") that introduce various mathematical modifications to the loss objective to improve alignment. However, there is a lack of understanding regarding which specific components actually contribute to performance gains, and fair comparisons are scarce due to inconsistent experimental setups. The authors seek to demystify these contributions by decomposing XPOs into orthogonal components and determining if they can be combined into a single, superior framework.

### Method and Recipe
The authors categorized existing XPO improvements into seven broad directions: length normalization, link function, margin/home advantage, reference policy, contextual scaling, rejection sampling optimization (RSO), and supervised fine-tuning (SFT) loss. Through empirical validation, they found that length normalization, mixing reference policies, and contextual scaling were the most effective.

**RainbowPO** is constructed by integrating these three essential components:

1.  **Length Normalization:** To mitigate verbosity and length bias in LLM-as-a-judge evaluations, the reward is normalized by the response length $|y|$.
2.  **Mixing Reference Policy:** Instead of using only the SFT policy ($\pi_{\text{ref}}$) or a reference-free margin (as in SimPO), RainbowPO uses a blended policy $\pi_{\alpha, \gamma}$ that exponentially mixes the SFT policy with a hypothetical optimal policy $\pi_{\gamma}$ using a mixing coefficient $\alpha \in [0, 1]$.
3.  **Contextual Scaling:** A scaling factor $\phi(x)$ is introduced to account for the dispersion or uncertainty of preference pairs for different prompts, based on the Mallows ranking model.
4.  **Preference Dataset Formulation:** The framework utilizes a modified version of RSO. Instead of true reward values, it computes percentile rewards (ranking rewards) over a generation set to stabilize the formulation of the preference dataset $\mathcal{D}_{RS}$.

### Key Formulas
The unified **RainbowPO** objective is defined as:

$$
\mathcal{L}_{\mathtt{RainbowPO}}\left(\pi_{\theta};\pi_{\text{ref}}\right):=-\mathbb{E}_{\left(x,y_{w},y_{l}\right)\sim\mathcal{D}}f\left[\phi(x)\left(\frac{\beta}{|y_{w}|^{\eta}}\text{l o g}\frac{\pi_{\theta}\left(y_{w}\mid x\right)}{\pi_{\alpha,\gamma}\left(y_{w}\mid x\right)}-\frac{\beta}{|y_{l}|^{\eta}}\text{l o g}\frac{\pi_{\theta}\left(y_{l}\mid x\right)}{\pi_{\alpha,\gamma}\left(y_{l}\mid x\right)}\right)\right]
$$

where $\eta \in \{0, 1\}$ (set to 1 for length normalization) and $f$ is the link function (typically $-\log \sigma(\cdot)$).

The **mixing reference policy** is defined as:

$$
\pi_{\alpha,\gamma}(y\ |\ x)\propto\pi_{\text{r e f}}^{\alpha}(y\ |\ x)\cdot\pi_{\gamma}^{1-\alpha}(y\ |\ x)
$$

The practical implementation of the loss with the mixing policy is:

$$
\mathcal{L}_{\mathrm{L N-D P O}}\big(\pi_{\theta};\pi_{\alpha,\gamma}\big) = -\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\text{l o g}\sigma\left(\beta\text{l o g}\frac{\pi_{\theta}\left(y_{w}\ |\ x\right)^{1/|y_{w}|}}{\pi_{\theta}\left(y_{l}\ |\ x\right)^{1/|y_{l}|}}-\alpha\beta\text{l o g}\frac{\pi_{\mathsf{r e f}}\ (y_{w}\ |\ x)^{1/|y_{w}|}}{\pi_{\mathsf{r e f}}\ (y_{l}\ |\ x)^{1/|y_{l}|}}-(1-\alpha)\gamma\right)
$$

### Key Quantitative Results
Experiments were conducted using **Llama3-8B-Instruct** and evaluated via **AlpacaEval2**.

*   **Performance Gain:** RainbowPO improved the Length Controlled Win Rate (LC WR) from **22.92%** (base model) to **51.66%** when using GPT-4 as the judge.
*   **Comparison to Baselines:** In a 3-epoch training comparison, RainbowPO (51.66% LC WR) outperformed SimPO (48.40% LC WR) and DPO (43.65% LC WR).
*   **Cross-Judge Validation:** Using Llama3-70B as the judge, RainbowPO achieved an LC WR of **63.94%**.
*   **Verbosity Control:** RainbowPO reduced the average response length from **2169** tokens (base model) to **1878** tokens.
*   **Ablation Insights:** Length normalization was identified as the most critical component; removing it dropped the LC WR from 51.66% to 45.68%.

### Stated Limitations
*   **Model Scope:** Evaluations were limited to the Llama3-8B-Instruct base model; generalizability to other architectures (e.g., Gemma, Mistral) was not tested.
*   **Metric Scope:** The study focused primarily on instruction-following via AlpacaEval2; other metrics like MT-bench or Arena-Hard were not utilized.
*   **Unexplored XPO Ideas:** The framework did not incorporate dynamic reference policy updates (e.g., sDPO) or methods for handling noisy preferences.
*   **Resource Constraints:** Due to time and computing limits, the authors did not benchmark the algorithm on reasoning or Chain-of-Thought (COT) capabilities.
