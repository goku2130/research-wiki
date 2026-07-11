---
id: mbrenndoerfer:dpo-variants-ipo-kto-orpo-cdpo-for-llm-a
type: web
title: 'DPO Variants: IPO, KTO, ORPO & cDPO for LLM Alignment'
url: https://mbrenndoerfer.com/writing/dpo-variants-ipo-kto-orpo-cdpo-llm-alignment
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-and-preference-optimization
---

This article introduces four variants of Direct Preference Optimization (DPO)—Identity Preference Optimization (IPO), Kahneman-Tversky Optimization (KTO), Odds Ratio Preference Optimization (ORPO), and Conservative DPO (cDPO)—each designed to address specific practical limitations of the original DPO algorithm for Large Language Model (LLM) alignment.

**Core Problem:**
The original DPO, while simpler than RLHF, exhibited several practical issues:
1.  **Unbounded Reward Growth:** DPO's loss function could lead to overconfident predictions and extreme probabilities, as the implicit reward gap between chosen and rejected responses could grow without bound. This caused models to assign probabilities approaching 1 to chosen responses and 0 to rejected ones, even when human preferences are not absolute.
2.  **Paired Data Requirement:** DPO necessitates head-to-head comparisons (chosen vs. rejected pairs), which is often not how feedback is collected (e.g., thumbs-up/thumbs-down on individual responses).
3.  **Computational Cost:** DPO requires keeping two copies of a large model (policy and reference) in VRAM simultaneously, posing a challenge for limited GPU memory.
4.  **Annotation Noise:** Crowd-sourced preference data can contain incorrect labels, leading to erratic model behavior.

**Method/Recipe Step by Step:**

The article focuses on IPO in detail:

**Identity Preference Optimization (IPO):**
IPO addresses the unbounded reward growth problem by reformulating preference learning as a regression problem with a specific target.

1.  **Problem Identification (DPO Overfitting):** The DPO loss function, $\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w\|x)}{\pi_{\text{ref}}(y_w\|x)} - \log \frac{\pi_\theta(y_l\|x)}{\pi_{\text{ref}}(y_l\|x)} \right) \right) \right]$, incentivizes the model to make the log-ratio difference as large as possible. The negative log-sigmoid approaches 0 as its argument grows, meaning the gradient always pushes to increase the preference gap, even when it's already large. This leads to extreme probabilities and potential overfitting to non-absolute human preferences.
2.  **IPO's Squared Error Objective:** IPO replaces DPO's monotonic loss with a squared-error objective that targets a fixed margin.
    *   Define the log-ratio difference: $h_\theta(x, y_w, y_l) = \log \frac{\pi_\theta(y_w\|x)}{\pi_{\text{ref}}(y_w\|x)} - \log \frac{\pi_\theta(y_l\|x)}{\pi_{\text{ref}}(y_l\|x)}$.
    *   The IPO loss function is: $\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( h_\theta - \frac{1}{2\beta} \right)^2 \right]$.
    *   This loss penalizes deviations from the target margin $\frac{1}{2\beta}$, whether the preference is too weak or too strong.
3.  **Gradient Behavior:** The gradient of the IPO loss with respect to $h_\theta$ is $\frac{\partial \mathcal{L}_{\text{IPO}}}{\partial h_\theta} = 2 \left( h_\theta - \frac{1}{2\beta} \right)$. This gradient is zero when $h_\theta = \frac{1}{2\beta}$, creating a stable equilibrium. If $h_\theta < \frac{1}{2\beta}$, the gradient is negative, pushing $h_\theta$ up. If $h_\theta > \frac{1}{2\beta}$, the gradient is positive, pushing $h_\theta$ down. This self-correcting behavior prevents unbounded growth.
4.  **Interpreting the Target Margin:** The target $\frac{1}{2\beta}$ is derived from the regularization strength $\beta$.
    *   A large $\beta$ (weak KL penalty) results in a small target margin $\frac{1}{2\beta}$.
    *   A small $\beta$ (strong KL penalty) results in a large target margin $\frac{1}{2\beta}$.
    This reflects that stronger KL constraints (small $\beta$) permit larger per-example margins because the overall deviation is tightly controlled. The target margin also aligns with the Bradley-Terry model, representing a moderate preference probability rather than absolute certainty.

**Key Formulas (in LaTeX):**

*   **DPO Loss Function:**
    $\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \left( \log \frac{\pi_\theta(y_w\|x)}{\pi_{\text{ref}}(y_w\|x)} - \log \frac{\pi_\theta(y_l\|x)}{\pi_{\text{ref}}(y_l\|x)} \right) \right) \right]$
*   **Log-Ratio Difference (Implicit Reward):**
    $h_\theta(x, y_w, y_l) = \log \frac{\pi_\theta(y_w\|x)}{\pi_{\text{ref}}(y_w\|x)} - \log \frac{\pi_\theta(y_l\|x)}{\pi_{\text{ref}}(y_l\|x)}$
*   **IPO Loss Function:**
    $\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( h_\theta - \frac{1}{2\beta} \right)^2 \right]$
*   **Gradient of IPO Loss w.r.t. $h_\theta$:**
    $\frac{\partial \mathcal{L}_{\text{IPO}}}{\partial h_\theta} = 2 \left( h_\theta - \frac{1}{2\beta} \right)$
*   **DPO Gradient Magnitude w.r.t. $h_\theta$:**
    $\left\| \frac{\partial \mathcal{L}_{\text{DPO}}}{\partial h_\theta} \right\| = \beta \sigma(-\beta h_\theta)$
*   **IPO Gradient Magnitude w.r.t. $h_\theta$:**
    $\left\| \frac{\partial \mathcal{L}_{\text{IPO}}}{\partial h_\theta} \right\| = 2 \left\| h_\theta - \frac{1}{2\beta} \right\|$

**Key Quantitative Results and Numbers:**
The article describes the conceptual behavior and mathematical properties of IPO but does not present specific numerical results or performance metrics from experiments. It highlights that the IPO gradient is zero when $h_\theta = \frac{1}{2\beta}$, establishing a stable equilibrium, and that the gradient's strength is proportional to the distance from this target.

**Stated Limitations:**
The article details the limitations of the *original DPO* that motivated the development of its variants:
1.  **Unbounded Reward Growth:** DPO's loss function lacks a natural stopping point, leading to extreme, overconfident predictions and potential numerical instability as log probabilities approach $-\infty$.
2.  **Paired Data Requirement:** DPO strictly requires pairwise comparisons (chosen vs. rejected), making it incompatible with simpler feedback formats like thumbs-up/thumbs-down.
3.  **Computational Overhead:** The need to keep both the policy and a frozen reference model in VRAM simultaneously can be computationally expensive, especially for large models and limited GPU resources.
4.  **Sensitivity to Annotation Noise:** DPO can be sensitive to errors in human preference data, leading to erratic model behavior.

IPO specifically addresses the unbounded reward growth and overfitting problem of DPO. The article does not state specific limitations of IPO itself, but rather presents it as a solution to a DPO limitation.
