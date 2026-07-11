---
id: arxiv:2604.16790
type: paper
title: 'Bias in the Loop: Auditing LLM-as-a-Judge for Software Engineering'
url: https://arxiv.org/html/2604.16790v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

# Summary: Bias in the Loop: Auditing LLM-as-a-Judge for Software Engineering

### Core Problem
Large Language Models (LLMs) are increasingly used as "judges" to evaluate code artifacts (generation, repair, and test generation) when human review or executable tests are unavailable. However, this practice lacks a principled account of reliability. The authors identify a critical vulnerability: LLM judges are highly sensitive to non-semantic prompt perturbations—such as the order of candidates or the tone of the prompt—which can shift verdicts even when the underlying code remains unchanged. This sensitivity threatens the validity and reproducibility of software engineering (SE) evaluations.

### Method/Recipe
The researchers employed a "measurement-first" lens to audit LLM judges using the following step-by-step approach:

1.  **Dataset Selection**: They used `CodeJudgeBench`, consisting of triplets (Instruction, Good Response, Bad Response) across three tasks: **Code Generation (CodeGen)**, **Code Repair (CodeRepair)**, and **Unit Test Generation (TestGen)**.
2.  **Model Selection**: Three judges were evaluated: **Qwen3-4B** (open-source), **Qwen2.5-Coder-3B** (code-specialized open-source), and **GPT** (closed-source API).
3.  **Bias Injection (RQ1)**: The authors designed a suite of 12 explicit prompt-injected biases tailored for code, including:
    *   **Position**: Swapping the order of candidates A and B.
    *   **Content/Framing**: Authority, Bandwagon, Chain-of-Thought (CoT), Distraction, Diversity, Final-only, Model-name, Refined, Self-enhance, Sentiment, and Verbosity.
4.  **Evaluation Protocol**: 
    *   **Pairwise Comparison**: The judge chooses between Assistant A and Assistant B.
    *   **Fixed Evidence**: Only the prompt is modified; the code snippets remain identical.
    *   **Position Swapping**: Every item is evaluated in both original and swapped orders to isolate position effects.
5.  **Consistency Testing (RQ2)**: To measure test-retest reliability, the same case was presented twice to the same judge under identical settings with session isolation.
6.  **Confidence Analysis**: Token-level confidence was extracted using TokenScope to analyze the probability assigned to the final decision token.

### Key Formulas
The study uses **Micro-accuracy** to measure bias sensitivity:

$$
\text{A c c}(b)=\frac{1}{|D||\mathcal{P}|}\sum_{i\in D}\sum_{p\in\mathcal{P}}\mathbb{I}\Big[\hat{y}_{b}^{(i,p)}=y^{(i,p)}\Big]
$$

Where $D$ is the set of items, $\mathcal{P}$ is the set of presentations (original and swap), $y$ is the gold position, and $\hat{y}_b$ is the judge's prediction under bias condition $b$.

**Consistency Rate (CR)** is used to quantify reliability:

$$
\mathrm{C R}=\frac{1}{|D|}\sum_{i\in D}\mathbb{I}\Big(\hat{y}^{(i,1)}=\hat{y}^{(i,2)}\Big)
$$

Where $\hat{y}^{(i,1)}$ and $\hat{y}^{(i,2)}$ are the verdicts from two independent runs.

### Key Quantitative Results
*   **Bias Sensitivity**: Biases act as "positional priors." CoT, Authority, Refined, and Sentiment strongly push judges toward candidate A. Conversely, Verbosity consistently pushes judges toward candidate B.
*   **Model-Specific Findings**:
    *   **Qwen2.5-Coder-3B**: Exhibited a strong baseline preference for position A; when the gold answer was at position B, no-bias accuracy fell below chance (38.03%–49.27%). Under "A-correct" settings, Sentiment and Refined biases nearly saturated accuracy, but collapsed when the gold answer moved to B.
    *   **GPT**: Showed a strong no-bias baseline (~80% accuracy), but remained sensitive. TestGen was the most fragile task; for example, the Model-name bias dropped TestGen overall accuracy from 77.46% to 62.51%.
*   **Task Fragility**: The impact of bias scales with task difficulty: $\text{TestGen} > \text{CodeGen} > \text{CodeRepair}$.
*   **Consistency vs. Correctness**: For Qwen2.5-Coder-3B, baseline CR for TestGen was low (50.36%), but injecting Sentiment or Refined biases inflated CR to $\sim 86\%$. The authors note this is a "dangerous illusion of reliability," where the model anchors on the bias rather than the code.

### Stated Limitations
The authors highlight that high consistency (CR) does not imply correctness; rather, it can reflect a systematic bias where the model relies on superficial heuristics to break ties in hard cases. Additionally, due to API costs, GPT results were reported primarily as accuracy differences relative to the no-bias baseline rather than full position-stratified tables.
