---
id: aligned:distinguishing-three-alignment-taxes-by-
type: web
title: Distinguishing three alignment taxes - by Jan Leike
url: https://aligned.substack.com/p/three-alignment-taxes
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

# Summary: Distinguishing Three Alignment Taxes

### Core Problem
The central problem addressed is the "alignment tax"—the additional costs incurred when aligning an AI system. These taxes are undesirable because they hinder the adoption of alignment techniques. In competitive markets, companies may avoid alignment if the resulting taxes lead to less performant models, higher investment costs, or delayed deployment, as these factors result in lost market share and commercial opportunity costs.

### Taxonomy of Alignment Taxes
The author distinguishes between three specific types of alignment taxes:

1.  **Performance Taxes:** Regressions in a model's capability compared to an unaligned baseline.
2.  **Development Taxes:** The resources expended to align the model, including researcher time, compute costs, and compensation for human feedback.
3.  **Time-to-deployment Taxes:** The total wall-clock time required to transform a pretrained model into a sufficiently aligned model.

### Quantification and Methodology

#### Performance Tax Calculation
While traditionally measured by benchmark score reductions, the author proposes a more direct monetary quantification based on inference-time compute. If an unaligned model has performance $Z$ on capability $X$, and an aligned model has performance $Z' < Z$, the tax is measured by the extra compute required to restore performance to $Z$.

If the aligned model requires $T\%$ more inference-time compute to return to performance $Z$, the alignment tax is $T\%$. 

**Examples:**
*   Running a "best-of-2" sampling strategy corresponds to a $100\%$ alignment tax.
*   Running "best-of-4" for $10\%$ of all tasks corresponds to a $4 \times 10\% = 40\%$ alignment tax.

#### Development Tax Components
Development taxes include the creation of RLHF codebases, management of human labelers, compute, and researcher effort. The author notes that these costs are often independent of model size; larger models may actually justify higher development spending.

#### Time-to-Deployment Pipeline
The process for producing aligned models (e.g., InstructGPT, ChatGPT, Sparrow) follows a sequential recipe:
1.  Collecting prompts.
2.  Collecting demonstrations.
3.  Supervised fine-tuning (SFT).
4.  Collecting comparisons.
5.  Training reward models.
6.  RL fine-tuning.
7.  Human evaluations.

### Key Quantitative Results
*   **Development Costs:** The author estimates the total development costs of InstructGPT were approximately $5\text{--}20\%$ of the development cost of GPT-3.
*   **Deployment Timelines:** The alignment pipeline for GPT-3 took approximately $9$ months. With improved infrastructure and reusable data/code, this has been reduced to approximately $3$ months.
*   **Market Sensitivity:** In competitive API markets, even a $10\%$ performance tax can be prohibitive due to low switching costs for customers.

### Contextual Impact
The significance of these taxes varies by context:
*   **Competitive Markets:** Performance taxes are critical. For example, the author suggests that DALL·E 2's conservative safety mitigations (a performance tax) may have contributed to the wider adoption of competitors like Stable Diffusion and Midjourney.
*   **Automated Alignment Research:** Performance taxes are lower priority because the goal is progress rather than market competition. However, time-to-deployment taxes remain critical; if alignment progress lags behind capability growth, AI progress may need to be paused.

### Stated Limitations
The author identifies a critical flaw in current optimization: existing training loops cannot align models that perform tasks too difficult for humans to evaluate. Consequently, reducing the time-to-deployment of current pipelines will not suffice for future, more capable models. The necessary infrastructure for AI-assisted evaluation is still under development. Additionally, if performance taxes are not managed, they may compound from one generation of AI to the next.
