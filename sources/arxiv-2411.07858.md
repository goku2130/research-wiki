---
id: arxiv:2411.07858
type: paper
title: 'Verbosity ≠ Veracity: Demystify Verbosity Compensation Behavior of Large Language
  Models'
url: https://arxiv.org/abs/2411.07858
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

# Verbosity $\neq$ Veracity: Demystifying Verbosity Compensation in LLMs

### Core Problem
The authors identify a behavior in Large Language Models (LLMs) termed **Verbosity Compensation (VC)**. VC occurs when an LLM, despite being prompted to respond concisely, generates excessive, compressible tokens (e.g., repeating the question, introducing ambiguity, or excessive enumeration) as a way to "compensate" for uncertainty. This behavior is analogous to human hesitation under uncertainty. VC is problematic because it increases server costs and latency, confuses users, and—critically—is often negatively correlated with veracity, meaning verbose responses are frequently less accurate than concise ones.

### Method and Recipe
The researchers analyzed 14 LLMs across five knowledge and reasoning-based QA datasets (Qasper, LongBench, NarrativeQA, NQ30, and MMLU).

**1. Dataset Construction and VC Detection**
*   **Filtering:** To isolate VC, the authors selected samples where the ground truth answer $y$ contains fewer than $n=4$ words.
*   **Detection:** A response $r$ is flagged as exhibiting VC if its length $|r| > 3$ tokens, given the gold answer is $\le 3$ tokens.

**2. Performance Evaluation**
The authors used **recall** to measure performance to avoid biasing results toward longer responses:

$$
\text{recall}(y,r) = \frac{|r \cap y|}{|y|}
$$

They defined the **Performance Difference ($\Delta$)** as the average recall of concise responses minus the average recall of verbose responses:

$$
\Delta(D) = \frac{\sum_{(x,y) \in D} (1 - V(x,y,r)) \times \text{recall}(y,r)}{\sum_{(x,y) \in D} (1 - V(x,y,r))} - \frac{\sum_{(x,y) \in D} V(x,y,r) \times \text{recall}(y,r)}{\sum_{(x,y) \in D} V(x,y,r)}
$$

where $V(x,y,r)=1$ if VC is detected. To account for varying absolute performance across datasets, they introduced the **Relative Performance Difference ($\delta$):**

$$
\delta(D) = \frac{\Delta(D)}{\frac{\sum_{(x,y) \in D} \text{recall}(y,r)}{|D|}}
$$

**3. Uncertainty Quantification**
Uncertainty was measured using **perplexity** for open-source models and the **sum of eigenvalues of the graph Laplacian** for closed-source models.

**4. Mitigation: Cascade Model Selection (CaSel)**
The authors proposed a cascade algorithm to reduce VC:
1.  Order a set of LLMs from weakest to strongest.
2.  Prompt the weakest model for a response.
3.  If the verbosity detector $V(x,y,r) = 1$, discard the response and repeat the process with the next strongest model.
4.  Return the first response that is concise.

A modified **Routing Algorithm** was also developed, using probabilities $p_c$ (for concise responses) and $p_v$ (for verbose responses) to balance performance and API costs.

### Key Quantitative Results
*   **Pervasiveness:** VC was found across all models and datasets. Mistral-7b exhibited the highest average VC frequency (74.19%), while Llama3-70b had the lowest (13.62%). GPT-4o showed a VC frequency of 50.40% on certain benchmarks.
*   **Performance Gap:** Verbose responses generally performed worse. Llama3-70b exhibited a notable performance drop of 27.61% on the Qasper dataset when responding verbosely.
*   **Mitigation Efficacy:** The CaSel algorithm significantly reduced VC. For the Mistral model on the Qasper dataset, VC frequency dropped from 63.81% to 16.16%.
*   **Uncertainty Correlation:** Verbose responses exhibited higher uncertainty across all five datasets.

### Stated Limitations
The authors note that the absolute performance of LLMs can "twist" the performance difference $\Delta$, as datasets with lower overall performance have less room for a large $\Delta$. This necessitated the creation of the relative metric $\delta$ for fair comparison. Additionally, the study highlights that current LLMs struggle to naturally disentangle verbosity from veracity, as the performance gap does not diminish as model capability increases.
