---
id: mbrenndoerfer:position-bias-in-llm-judges-measurement-
type: web
title: 'Position Bias in LLM Judges: Measurement and Mitigation'
url: https://mbrenndoerfer.com/writing/position-bias-in-llm-judges
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

# Position Bias in LLM Judges: Measurement and Mitigation

### Core Problem
Position bias in LLM-as-Judge frameworks is the tendency of a judge model to systematically prefer candidate responses based on their position in the prompt (e.g., first or last) rather than their actual quality. This structural failure distorts evaluation results and compromises model comparisons. The bias is driven by two primary mechanisms: **recency bias** (weighting recent tokens more heavily) and **primacy bias** (weighting the first item as a reference point). These are further exacerbated by the "U-curve" attention pattern, where transformer models attend more strongly to the beginning and end of a sequence while underweighting the middle ("lost in the middle" phenomenon).

### Measurement Methodology
To quantify position bias, the author proposes a "swap" methodology where evaluations are run in both possible orderings for every pair of responses.

**Step-by-Step Recipe:**
1. **Dual Evaluation:** For each comparison pair, submit the responses in two orders: (Response A, Response B) and (Response B, Response A).
2. **Relabeling:** Relabel the verdicts from the second run to match the original identities of A and B.
3. **Consistency Analysis:** Compare the verdicts to determine if the judge's choice changed based solely on the order.
4. **Directional Analysis:** Calculate the preference rate for specific positions to determine if the bias is systematic (directional) or stochastic (noise).

### Key Formulas
The author defines three primary metrics for position bias and one for verbosity bias:

**Swap Consistency (SC):** The fraction of comparisons where the judge gives the same relative verdict regardless of order.

$$
SC = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left[v_1^{(i)} = v_2^{(i)}\right]
$$

*Where $N$ is the total pairs, $v_1$ is the verdict with A first, and $v_2$ is the verdict with B first.*

**Position Preference Rate (PPR):** The frequency with which the judge favors the response in the first (A) slot.

$$
PPR(A) = \frac{\text{count of times position-A is preferred}}{N}
$$

**First-Position Bias (FPB):** The directional preference among only the inconsistent pairs.

$$
FPB = \frac{n_{A \to A}}{n_{A \to A} + n_{B \to B}}
$$

*Where $n_{A \to A}$ is the count of cases where the judge picked position A in both orderings.*

**Verbosity Bias Regression:** To measure if length ($\ell_i$) influences the judge score ($s_i$) independently of human-rated quality ($q_i$):

$$
s_i = \alpha + \beta_1 q_i + \beta_2 \ell_i + \epsilon_i
$$

*Verbosity bias is present if $\beta_2 > 0$ with statistical significance.*

### Key Quantitative Results
* **GPT-4 Performance:** When used as a judge, GPT-4 exhibited a preference for the first response in approximately 60% to 70% of cases where it changed its verdict upon swapping.
* **General Consistency:** In practice, swap consistency rates often range between 0.7 and 0.8, indicating that 20% to 30% of judgments flip based on presentation order.
* **Verbosity Impact:** Some judge models show a $\beta_2/\beta_1$ ratio as high as 0.4, suggesting that response length can contribute nearly as much to the final verdict as actual quality.

### Related Biases and Limitations
The author identifies two compounding biases:
* **Verbosity Bias:** The tendency to score longer responses higher, potentially because length acts as a proxy for effort or allows the response to hit more internal "checklist" items.
* **Sycophancy:** The tendency to agree with the prompt's endorsed position or exhibit **model self-preference**, where a judge favors responses from its own model family.

**Limitations and Mitigation:**
While prompt engineering can dampen bias—such as using neutral labels (A/B) instead of ordinal numbers (1/2), reducing the distance between responses, and placing evaluation criteria before the responses—these methods cannot eliminate bias entirely. Furthermore, smaller models typically exhibit stronger and less consistent position biases than larger, instruction-tuned models due to a heavier reliance on shallow heuristics.
