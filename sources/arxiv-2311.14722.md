---
id: arxiv:2311.14722
type: paper
title: Fine-Tuning Large Language Models
url: https://arxiv.org/abs/2311.14722
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

**Core Problem**
Financial question answering demands complex, multi-hop numerical reasoning and precise arithmetic calculations over unstructured text and tabular data. Large language models (LLMs) frequently fail at accurate mathematical computation, and adapting them to financial domains typically relies on few-shot prompting. Few-shot approaches are highly sensitive to sample selection and ordering, consume excessive input tokens, and require labor-intensive curation of hand-crafted examples. The core challenge is to design a zero-shot prompting framework that extracts domain-specific reasoning from LLMs while bypassing their inherent arithmetic limitations.

**Methodology**
The authors propose two zero-shot prompting architectures: ZS-FinPYT and ZS-FinDSL. Both leverage four prompt engineering principles—signifiers, memetic proxies, behavioral constraints, and meta-prompting—to guide the LLM toward generating executable code rather than direct textual answers. 

The ZS-FinPYT pipeline operates via a single prompt. First, a signifier explicitly defines the task by instructing the model to read a financial passage and table. Second, a memetic proxy (`#Python`) and behavioral constraints direct the model to write valid Python code, avoid non-executable statements, and store the final result in a variable named `ans`. The generated script is then executed externally using Python’s `exec()` function, ensuring precise computation.

The ZS-FinDSL pipeline employs a dual-prompt sequence. The initial reasoning extraction prompt uses a meta-prompting trigger to generate step-by-step calculations. The subsequent program extraction prompt constrains the output to a strict domain-specific language (DSL) format. The model must map the reasoning to a structured format containing only permitted operations (`add`, `subtract`, `multiply`, `divide`, `exponent`, `greater-than`, `max`, `min`). This DSL is parsed and executed by a dedicated interpreter, effectively decoupling logical reasoning from numerical evaluation.

**Key Formulas**
To accommodate formatting variations in LLM outputs (e.g., "7 million" versus "7,000,000"), evaluation employs a relative tolerance metric. A generated answer ($\hat{a}$) is considered correct if it satisfies:
\[
\operatorname{abs} (\hat {a} - \tilde {a}) \leq \operatorname{rel} _ {-} \operatorname{tol} * \max (\operatorname{abs} (\hat {a}), \operatorname{abs} (\tilde {a}))
\]
where $\tilde{a}$ represents the gold answer and the relative tolerance (`rel_tol`) is fixed at 0.001.

**Quantitative Results**
Evaluated across FinQA, ConvFinQA, and TATQA (arithmetic subset) using text-davinci-003, gpt-3.5-turbo, and gpt-4, both proposed methods significantly outperformed zero-shot baselines (ZS-STD and ZS-CoT). Compared to ZS-STD, ZS-FinPYT improved accuracy by 4.5% to 47%, while ZS-FinDSL improved by 5.22% to 38.72%. Against ZS-CoT, ZS-FinPYT achieved gains of 3% to 33.22% for text-davinci-003 and gpt-3.5-turbo, and 1.5% to 3.5% for gpt-4. The highest recorded accuracy was 93.00% on TATQA using gpt-4 with ZS-FinPYT. Notably, the zero-shot approaches performed within 10% of few-shot Program of Thoughts (PoT) baselines, occasionally surpassing them. Performance breakdowns indicated that table-only questions and single-step programs yielded superior accuracy, whereas multi-hop reasoning and numerical questions presented greater difficulty.

**Stated Limitations**
The study is strictly limited to the GPT-x model family, meaning prompt efficacy may not generalize to other LLM architectures without modification. The authors acknowledge unexplained reasoning failures and program extraction errors, particularly in ZS-FinDSL where correct reasoning occasionally fails to translate into valid DSL syntax. Furthermore, the current prompting framework does not mitigate overconfident hallucination; when underlying reasoning is flawed, the system may still output a high-confidence but incorrect explanation, posing challenges for real-world deployment.
