---
id: arxiv:2310.08491
type: paper
title: 'Prometheus: Inducing Fine-grained Evaluation Capability in Language Models'
url: https://arxiv.org/abs/2310.08491
retrieved: '2026-07-11'
maturity: comprehensive
topic: llm-as-judge
---

**Core Problem**
Evaluating long-form LLM outputs using proprietary models like GPT-4 has become standard but introduces critical bottlenecks: closed-source opacity, uncontrolled versioning, and prohibitive API costs. Concurrently, existing open-source evaluators rely on coarse-grained, single-dimensional metrics (e.g., helpfulness/harmlessness) and lack the capacity to assess text against user-defined, fine-grained score rubrics. The authors address the need for an open-source, reproducible, and cost-effective evaluator that can generalize across thousands of customized assessment criteria.

**Methodology & Recipe**
The authors introduce PROMETHEUS, a 13B open-source evaluator fine-tuned on a newly constructed dataset, FEEDBACK COLLECTION. The dataset construction follows a strict four-step pipeline: (1) curating 50 initial seed score rubrics; (2) expanding to 1,000 rubrics via GPT-4 in-context learning and paraphrasing across 10 iterative rounds to ensure lexical diversity; (3) generating 20,000 instructions semantically aligned with each rubric; and (4) producing 100,000 response-feedback pairs (scores 1–5) using GPT-4. To mitigate decision and length biases, the dataset enforces uniform response lengths per score tier and an equal distribution across the 1–5 rating scale. Each training instance comprises four input components (instruction, response to evaluate, customized score rubric, and a reference answer scoring 5) and two output components (detailed feedback rationale, followed by an integer score). The model is fine-tuned sequentially to generate feedback and then the score, separated by a `[RESULT]` delimiter, using Llama-2-Chat as the base. The source does not provide explicit mathematical loss functions; evaluation relies on standard correlation metrics (Pearson, Kendall-Tau, Spearman) for absolute grading and accuracy for ranking grading.

**Key Quantitative Results**
PROMETHEUS demonstrates strong alignment with human and proprietary benchmarks. When evaluated across 45 customized rubrics, PROMETHEUS 13B achieves a Pearson correlation of 0.897 with human annotators, matching GPT-4 (0.882) and significantly surpassing GPT-3.5-Turbo (0.392). In pairwise human preference tests, PROMETHEUS’s feedback is preferred over GPT-4’s in 58.62% of cases and over GPT-3.5-Turbo’s in 79.57%. Across 1,222 rubrics spanning four benchmarks, PROMETHEUS 13B outperforms its base model by +0.420 (seen rubrics) and +0.397 (unseen rubrics) in Pearson correlation, while also exceeding Llama-2-Chat 70B and GPT-3.5-Turbo. In ranking-based human preference tasks, PROMETHEUS 13B achieves 79.19% average accuracy on the HHH Alignment dataset and 57.72% on MT Bench Human Judgment, outperforming specialized reward models like StanfordNLP and Almost Reward Model.

**Stated Limitations**
Despite its performance, the authors identify several constraints. Human annotators frequently reject PROMETHEUS’s feedback for being overly critical, contrasting with GPT-4’s more neutral tone. While PROMETHEUS matches GPT-4 on the FEEDBACK BENCH, it consistently lags behind GPT-4 on external benchmarks (Vicuna, MT Bench, Flask Eval), likely due to distributional mismatches between the training data and evaluation prompts. Furthermore, the authors note that directly fine-tuning on a target task’s feedback dataset yields superior task-specific performance, highlighting a trade-off between generalization and domain specialization. The model’s context window also restricts the inclusion of additional reference materials (e.g., negative-score reference answers), and while length bias was mitigated during training, the authors acknowledge that adversarial testing for verbosity bias remains an open area for future work.
