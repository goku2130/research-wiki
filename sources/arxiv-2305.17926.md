---
id: arxiv:2305.17926
type: paper
title: Large Language Models are not Fair Evaluators
url: https://arxiv.org/abs/2305.17926
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

The source identifies a critical flaw in the prevailing "LLM-as-a-judge" paradigm: severe positional bias. When evaluating pairwise responses, advanced models like GPT-4 and ChatGPT systematically favor candidates based solely on their presentation order, even when prompts explicitly instruct the evaluator to ignore sequence. This bias allows evaluation outcomes to be easily manipulated; for instance, swapping the order of Vicuna-13B and ChatGPT responses causes ChatGPT to act as an evaluator where Vicuna-13B appears superior in 66 out of 80 tested queries.

To rectify this, the authors propose a three-stage calibration framework. First, Multiple Evidence Calibration (MEC) modifies the evaluation template to require the LLM to generate explanatory evidence before assigning numerical scores. This leverages autoregressive generation to ground ratings in reasoning. The process is repeated $k$ times to yield multiple independent scores. Second, Balanced Position Calibration (BPC) mitigates positional preference by executing MEC twice per query, swapping the positions of the two candidate responses. The final calibrated score for each response is computed by averaging the $2k$ generated scores across both positions. Third, Human-in-the-Loop Calibration (HITLC) introduces a selective human review mechanism. Using the evaluation outcomes from MEC and BPC, the framework calculates a Balanced Position Diversity Entropy (BPDE) score to quantify evaluation uncertainty. Examples exceeding a predefined threshold $\beta$ are flagged for human annotation, with final decisions resolved via majority voting.

The methodology relies on three core mathematical formulations. The positional sensitivity is quantified via the Conflict Rate:
\[ \text{Conflict Rate} = \frac{\sum_{i=1}^{N} \mathbb{I}(\mathbf{ER}_i^{r12} \neq \mathbf{ER}_i^{r21})}{N} \]
where $\mathbb{I}$ is an indicator function comparing evaluation results $\mathbf{ER}$ from swapped positions. The BPC calibrated score $CS_R$ for response $R$ is:
\[ CS_R = \sum_{i=1}^{k} \frac{S_R^i + S_R^{\prime i}}{2k}, \quad R \in \{r1, r2\} \]
where $S_R^i$ and $S_R^{\prime i}$ denote scores at the first and second positions. HITLC utilizes BPDE to guide human intervention:
\[ \mathrm{BPDE} = \sum_{\mathbf{er} \in \{\text{win}, \text{tie}, \text{lose}\}} -\mathbf{p}_{\mathbf{er}} \log \mathbf{p}_{\mathbf{er}} \]
with outcome probabilities $\mathbf{p}_{\mathbf{er}} = \frac{\sum_{i=1}^{k} \mathbb{I}(\mathbf{ER}_i = \mathbf{er}) + \mathbb{I}(\mathbf{ER}_i^{\prime} = \mathbf{er})}{2k}$.

Empirical validation on 80 manually annotated Vicuna Benchmark queries yields significant quantitative improvements. Vanilla evaluators exhibit high conflict rates: GPT-4 shows 46.3% and 5.0%, while ChatGPT reaches 82.5% and 52.5% across different model pairings. Win rates fluctuate drastically by position; for example, Vicuna-13B’s win rate against ChatGPT drops from 82.5% to 2.5% when moved from the second to the first position. Applying MEC and BPC to ChatGPT increases alignment accuracy with human judgments from 44.4% to 58.7% (+14.3%) and raises the kappa coefficient from 0.06 to 0.31. Integrating HITLC with a 20% human annotation threshold ($\beta=20\%$) pushes ChatGPT’s accuracy to 71.3% and kappa to 0.52, matching the human average (71.7% accuracy, 0.54 kappa) while reducing annotation costs by 39% (from $30.00 to $18.30). Optimal performance for MEC is observed at $k=3$ with sampling temperatures between 0.6 and 1.0.

The authors note several limitations. Positional bias is highly dependent on the intrinsic quality gap between responses; it severely impacts comparisons with small score differences ($\le 1$) but diminishes when gaps are large ($\ge 3$). The BPDE metric requires a manually tuned threshold $\beta$ to balance cost and accuracy. Furthermore, the calibration framework currently targets pairwise comparison templates and incurs higher API costs as $k$ increases, though HITLC mitigates this by limiting human labor. The study also acknowledges that while explicit prompt instructions to ignore order are ineffective, the proposed calibration strategies provide a robust, cost-efficient alternative for reliable automated evaluation.
