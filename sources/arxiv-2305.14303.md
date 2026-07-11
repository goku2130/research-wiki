---
id: arxiv:2305.14303
type: paper
title: Mathematical Verification with Large Language Models
url: https://arxiv.org/abs/2305.14303
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

The source addresses a critical gap in table-to-text generation, which traditionally focuses on generic summaries or single-sentence conversions, thereby overlooking real-world user information needs. The authors define a new task, query-focused table summarization, requiring models to perform human-like reasoning and analysis to generate tailored, paragraph-long summaries grounded in specific user queries.

To operationalize this task, the authors construct QTSUMM and propose REFACTOR, a retrieval-and-reasoning framework. The dataset construction follows a strict pipeline: source tables are filtered from LOGICNLG and ToTTO for appropriate size, non-string columns, and flat structure. Annotators then generate 2–3 answerable user queries per table, deliberately avoiding short-answer questions. A second annotator writes paragraph-long summaries that must comprehensively address the query using only table-grounded information, requiring complex reasoning across multiple regions. The pipeline incorporates multi-round validation, expert annotators (17 STEM graduate students), and de-biasing protocols such as random row highlighting and query diversity monitoring. REFACTOR improves generation through a step-by-step recipe: (1) it executes six predefined reasoning operations (conjunction, counting, ordering, comparison, and numerical operations) via programmatic template instantiation to produce candidate facts; (2) it ranks these facts using a QA encoding model based on cosine similarity with the query embedding; (3) it selects the top-$n$ facts and concatenates them to the model input to guide the final summary generation.

The task objective is formalized as maximizing the conditional likelihood of the summary tokens:
\[
\boldsymbol {Y} = \operatorname{argmax} \prod_ {i = 1} ^ {n} P (y _ {i} | y _ {< i}, \boldsymbol {Q}, \boldsymbol {T}; \theta)
\]
The number of facts to retrieve is calculated as:
\[
n = \max\left(\sqrt{\frac{row_{num} \times column_{num}}{2}}, 5\right)
\]

QTSUMM comprises 2,934 tables and 7,111 query-summary pairs. Tables average 11.8 rows and 6.6 columns, with summaries averaging 68.0 words. Evaluations across text generation, table-to-text, and large language model baselines reveal significant performance gaps. REFACTOR consistently improves baseline outputs in human evaluations, particularly in faithfulness and comprehensiveness. For example, GPT-4’s faithfulness score increased by 0.16, and Llama-2-70B’s improved by 0.28 on a 5-point Likert scale. Automated metrics also show gains, such as BART’s BLEU score increasing by 0.3 and A3CU score by 0.6. Human evaluation of REFACTOR’s intermediate outputs indicates that 56.4% of generated facts (528 out of 937) are deemed relevant. Error analysis of 200 REFACTOR cases identifies parsing difficulties (24 cases), ranking failures for complex queries (17 cases), and unsupported reasoning operations (13 cases) as primary failure modes.

The authors explicitly state several limitations. Baseline models are constrained by maximum token limits (e.g., 1024 tokens), preventing them from processing large tables exceeding approximately 300 cells. REFACTOR’s template-based fact generation may fail to capture all crucial insights for highly complex queries, and its QA-based ranking struggles with nuanced user intents like correlation analysis. Furthermore, the study highlights a persistent mismatch between automated metrics (e.g., BLEU, ROUGE) and human judgments, underscoring the need for more explainable and reliable automated evaluation systems for this task.
