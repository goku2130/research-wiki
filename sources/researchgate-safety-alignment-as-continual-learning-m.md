---
id: researchgate:safety-alignment-as-continual-learning-m
type: web
title: 'Safety Alignment as Continual Learning: Mitigating the Alignment Tax via Orthogonal
  Gradient Projection'
url: https://www.researchgate.net/publication/400603496_Safety_Alignment_as_Continual_Learning_Mitigating_the_Alignment_Tax_via_Orthogonal_Gradient_Projection
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

# Summary: Safety Alignment as Continual Learning

### Core Problem
The authors address the "alignment tax," a phenomenon where safety post-training (e.g., Supervised Fine-Tuning (SFT) or Direct Preference Optimization (DPO)) reduces a Large Language Model's (LLM) general utility in areas such as reasoning and coding. The researchers argue that this tax is a manifestation of catastrophic forgetting within a **heterogeneous continual learning (HCL)** framework. Specifically, the sequential nature of alignment—moving from pre-training to SFT and then to preference optimization—introduces shifts in both data distributions and optimization objectives. This creates a "stability-plasticity dilemma" where safety-driven gradient updates overwrite the parameter subspaces critical for maintaining pre-trained general competencies.

### Method: Orthogonal Gradient Projection for Safety Alignment (OGPSA)
OGPSA is a lightweight, plug-and-play geometric procedure designed to decouple safety acquisition from capability retention. Rather than using soft constraints like KL penalties or computationally expensive data replay, OGPSA explicitly constrains safety updates to avoid interfering with capability-preserving directions in the parameter space.

**The OGPSA recipe is as follows:**
1. **Reference Set Selection:** A small, representative subset of general-purpose data is identified to serve as a proxy for the model's core competencies.
2. **Subspace Estimation:** The method estimates a low-rank "capability subspace" by analyzing gradients derived from this reference set. This subspace captures the directions in the parameter space essential for stabilizing general abilities.
3. **Gradient Projection:** During the safety alignment phase (SFT or DPO), the safety-related gradient is computed. This gradient is then projected onto the **orthogonal complement** of the previously learned capability subspace.
4. **Parameter Update:** The model is updated using only the projected gradient, effectively filtering out components that would risk overwriting prior knowledge.

### Key Formulas
The authors formalize the alignment tax ($\Delta \text{tax}$) as the difference in evaluation metrics ($\Phi$) on a general evaluation suite ($D_{\text{eval}}$) between the pre-trained model ($\theta_{\text{pre}}$) and the safety-aligned model ($\theta_{\text{safe}}$):

$$
\Delta \text{tax} = \Phi(\theta_{\text{pre}}; D_{\text{eval}}) - \Phi(\theta_{\text{safe}}; D_{\text{eval}})
$$

The method operates on the principle of first-order orthogonal projection to ensure that the safety update $\Delta \theta$ does not project onto the capability subspace.

### Key Quantitative Results
OGPSA was tested across SFT, DPO, and sequential SFT $\rightarrow$ DPO pipelines, consistently improving the safety-utility Pareto frontier. The most notable results were observed using **Qwen2.5-7B-Instruct** in the sequential **SFT $\rightarrow$ DPO** setting:

*   **SimpleQA:** General capability recovered from **0.53%** (standard baseline) to **3.03%** with OGPSA.
*   **IFEval:** General capability improved from **51.94%** (standard baseline) to **63.96%** with OGPSA.

The authors note that OGPSA preserves strong safety performance while recovering these general capabilities, outperforming naive tuning, model merging, LoRA, and general data replay.

### Stated Limitations
While the provided text emphasizes the advantages of OGPSA over traditional continual learning (CL) methods—which often assume homogeneous optimization objectives—it does not explicitly list specific failure modes or limitations of the OGPSA method itself. It does, however, contrast OGPSA against existing "soft constraints" (like KL penalties), noting that those heuristic approaches fail to explicitly remove the specific components of safety updates that interfere with capability-preserving directions.
