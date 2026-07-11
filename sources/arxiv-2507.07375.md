---
id: arxiv:2507.07375
type: paper
title: Bradley–Terry and Multi-Objective Reward Modeling Are Complementary
url: https://arxiv.org/html/2507.07375v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-modeling
---

The authors address the problem of reward hacking in Reinforcement Learning from Human Feedback (RLHF) for Large Language Models (LLMs), particularly in challenging out-of-distribution (OOD) settings. Existing methods struggle with OOD generalization, and while multi-objective reward models (MORMs) show promise by incorporating fine-grained attribute scores, their performance is often limited by data availability. Single-objective reward models (SORMs), trained on more abundant preference data, typically outperform MORMs in scoring. The core problem is how to efficiently mitigate reward hacking in OOD settings using fine-grained attribute scores without requiring additional costly multi-attribute preference data.

The proposed method, **Joint Single and Multi-Objective Reward Model (SMORM)**, unifies Bradley-Terry (BT) single-objective and multi-objective regression-based reward functions by jointly training them using a shared embedding space.

**Method/Recipe Step-by-Step:**

1.  **Shared Backbone:** A pre-trained decoder-only LLM (e.g., gemma-2B-it, Mistral-7B-Instruct-v0.2) is used as a feature extractor $f_{\theta}$. The concatenation of prompt and response, $x \oplus y$, is passed through the decoder layers, and the hidden state of the final decoder layer forms a $d$-dimensional feature vector.
2.  **Dual Heads:** Two linear heads are attached to the shared feature extractor:
    *   A single-objective head with weights $\mathbf{w}_{S} \in \mathbb{R}^{d \times 1}$, which outputs a scalar reward score (e.g., for chosen/rejected preferences).
    *   A multi-objective head with weights $\mathbf{w}_{M} \in \mathbb{R}^{d \times \widetilde{k}}$, which produces a $\widetilde{k}$-dimensional vector of attribute scores (e.g., helpfulness, coherence, verbosity).
3.  **Joint Training Loss:** SMORM is trained using a combined loss function that optimizes both heads simultaneously:

$$
\text{min}_{\theta,\mathbf{w}_{S},\mathbf{w}_{M}}\ -\mathbb{E}_{\mathcal{D}_{S}}\left[\text{log}\sigma(\mathbf{w}_{S}^{\top}((f_{\theta}(x_{s},y_{c})-f_{\theta}(x_{s},y_{r}))))\right]+\mathbb{E}_{\mathcal{D}_{M}}\left\|\mathbf{w}_{M}^{\top}f_{\theta}(x_{m},y_{m})-\mathbf{r}\right\|_{2}^{2}
$$

    where:
    *   The first term is the Bradley-Terry loss for the single-objective head, trained on a chosen-rejected preference dataset $\mathcal{D}_{S}=\{x_{s},y_{c},y_{r}\}$. $\sigma(\cdot)$ is the sigmoid function.
    *   The second term is the Mean Squared Error (MSE) loss for the multi-objective head, trained on a multi-attribute preference dataset $\mathcal{D}_{M}=\left\{x_{m},y_{m},\mathbf{r}\right\}$, where $\mathbf{r}$ is the vector of ground-truth attribute scores.
4.  **Inference Strategies:** SMORM supports multiple inference strategies:
    *   **SMORM-F:** Uses only the single-objective head to produce the reward score.
    *   **SMORM-L:** Computes the mean of the scores from the multi-objective head.
    *   **SMORM-M:** Averages the scores from both the single-objective and multi-objective heads.

**Key Formulas in LaTeX:**

*   **Bradley-Terry Loss (Single-Objective):**

$$
\mathcal{L}_{\text{reward}}(\theta)=-\mathbb{E}_{(x,y_{c},y_{r})\sim\mathcal{D}_{x}}\left[\text{log}\left(\sigma\left(r_{\theta}(x,y_{c})-r_{\theta}(x,y_{r})\right)\right)\right]
$$

    where $r_{\theta}(x,y) = \mathbf{w}_{S}^{\top}f_{\theta}(x,y)$.
*   **Multi-Objective Regression Loss:**

$$
\mathcal{L}(\psi)=\mathbb{E}_{(x,y,\mathbf{r})\sim\mathcal{D}_{M}}\left\|R_{\psi}(x,y)-\mathbf{r}\right\|_{2}^{2}
$$

    where $R_{\psi}(x,y) = \mathbf{w}_{M}^{\top}f_{\theta}(x,y)$.
*   **SMORM Joint Loss:**

$$
\text{min}_{\theta,\mathbf{w}_{S},\mathbf{w}_{M}}\ -\mathbb{E}_{\mathcal{D}_{S}}\left[\text{log}\sigma(\mathbf{w}_{S}^{\top}((f_{\theta}(x_{s},y_{c})-f_{\theta}(x_{s},y_{r}))))\right]+\mathbb{E}_{\mathcal{D}_{M}}\left\|\mathbf{w}_{M}^{\top}f_{\theta}(x_{m},y_{m})-\mathbf{r}\right\|_{2}^{2}
$$

*   **Implicit Multi-Attribute Effect (Theorem 1):** For a reward model trained under SMORM, with bounded features, positive-definite covariances, and positive correlation between aggregated fine-grained attribute scores and single-objective preference, there exist constants $c = \frac{\mathbf{1}^{\top}\alpha}{K\left(\mu_{S}^{\top}\ \Sigma_{S}^{-1}\ \mu_{S}\right)}$ and $\varepsilon \geq 0$ such that for every pair $(x,y)$:

$$
r_{m}(x,y)\;=\;\textstyle\frac{1}{K}\sum_{i=1}^{K}w_{M,i}^{\top}f_{\theta}(x,y)\;\;\geq\;\;c\left(w_{S}^{\top}f_{\theta}(x,y)\right)\;-\;\varepsilon=c r_{s}(x,y)\;-\;\varepsilon.
$$

    This implies that a high single-objective score $r_s$ ensures a respectable level of fine-grained quality $r_m$.
*   **Pairwise Preference Error to MSE Loss (Lemma 1):**

$$
\mathbb{E}_{\mathcal{D}_{S}}\left|\mathbb{P}(y_{A}\succ y_{B})-\mathbb{P}^{*}(y_{A}\succ y_{B})\right|\leq\frac{1}{4}\mathbb{E}_{\mathcal{D}_{S}}\left(\sqrt{2\mathit{M S E}(r_{s})}\right)
$$

    And for multi-objective:

$$
\mathbb{E}_{\mathcal{D}_{M}}\left|e_{m}-e_{m}^{\star}\right|\leq\mathbb{E}_{\mathcal{D}_{M}}\left(\sqrt{2\mathit{M S E}(r_{m})}\right)
$$

    where $MSE(r_s) = (r_s(y) - g_s(y))^2$.
*   **Asymptotic MSE Reduction (Theorem 2):** SMORM yields lower asymptotic MSE for both single- and multi-objective heads compared to training either head alone:

$$
\mathrm {M S E} _ {S} ^ {S M O R M} < \mathrm {M S E} _ {S} ^ {s i n g l e}, \quad \mathrm {M S E} _ {M} ^ {S M O R M} < \mathrm {M S E} _ {M} ^ {m u l t i}
$$

**Key Quantitative Results and Numbers:**

*   **OOD Reward Hacking Mitigation:** In PPO and BoN experiments under OOD settings (gemma-2B-it base model), SMORM-F and SMORM-M significantly outperform baselines (Baseline Classifier, ODIN, Baseline SM, GRM) in terms of gold scores, which measure true quality. For instance, in Figure 2, SMORM-F and SMORM-M show consistent increases in gold scores as proxy scores increase, while baselines often plateau or decrease.
*   **Multi-Objective Performance Improvement:** SMORM significantly improves the performance of its multi-objective head. Using Mistral-7B-Instruct as the base model and 40k samples from UnifiedFeedback, SMORM-L achieves **13.9** points higher on RewardBench and **12.4** points higher on RM-Bench compared to a baseline MORM.
*   **Outperforming Larger Models:** SMORM-L (7B model) trained on 20K $\mathcal{D}_M$ (HelpSteer2) outperforms a 70B baseline model (Llama-3-70B-RM) on RewardBench (89.0 vs. 88.8 Avg score). An 8B SMORM-L model matches the performance of ArmoRM-Llama3-8B-v0.1 (90.4 Avg score) despite the latter being trained on **15.9x** more multi-objective data (20K vs. 585.4K).
*   **In-Distribution Performance:** In ID PPO and BoN experiments, SMORM-F and SMORM-M show consistent increases in gold scores throughout training, unlike baselines that exhibit reward overoptimization (gold scores decline while proxy scores rise).

**Stated Limitations:**

*   The theoretical analysis (Theorem 1 and 2) relies on assumptions such as bounded features, positive-definite covariances, and positive correlation between aggregated fine-grained attribute scores and single-objective preference. While these are generally justified in real-world annotation settings, their strict adherence is assumed.
*   The paper notes that the conflicting objectives of reward modeling and text generation in methods like GRM can cause training instability and sensitivity to balancing weights, indicating a limitation of such approaches.
*   Prior research predominantly focuses on and evaluates in-distribution scenarios, meaning state-of-the-art methods struggle in more challenging out-of-distribution (OOD) settings, which SMORM aims to address.
*   The performance of multi-objective reward models is constrained by the limited availability of large-scale, high-quality annotated data. This limitation arises from either low-quality LLM-as-Judge annotations or high-quality human annotations that are difficult to scale.
