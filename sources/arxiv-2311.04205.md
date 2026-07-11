---
id: arxiv:2311.04205
type: paper
title: Length-Controlled AlpacaEval with a Machine Learning Judge
url: https://arxiv.org/abs/2311.04205
retrieved: '2026-07-11'
maturity: comprehensive
topic: length-and-format-bias
---

The paper addresses a fundamental disconnect in human-LLM interaction: prompt ambiguity and divergent "frames of thought" cause LLMs to misinterpret seemingly clear queries, yielding incorrect responses. While prompt quality dictates output fidelity, systematic methods for aligning human intent with LLM comprehension remain underdeveloped. The authors identify that benchmark datasets contain ambiguities imperceptible to humans but detrimental to models, leading to systematic errors in zero-shot settings.

To resolve this, the authors propose Rephrase and Respond (RaR), an unsupervised, training-free prompting strategy that enhances semantic clarity by having the LLM rearticulate the query before answering. The single-step RaR recipe employs a unified prompt:
$$P_{\text{RaR}} = \text{"}\{q\}\text{"} \n \text{Rephrase and expand the question, and respond.}$$
A two-step variant decouples rephrasing and response generation to leverage stronger models for weaker ones. The rephrasing step uses:
$$P_{\text{rep}} = \text{"}\{q\}\text{"} \n \text{Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.}$$
The responding step then feeds both versions to the target model:
$$P_{\text{resp}} = \text{(original) }\{q\} \text{ (rephrased) }\{q_{\text{rep}}\} \text{ Use your answer for the rephrased question to answer the original question.}$$
The methodology supports multiple phrasing variations (e.g., "Reword and elaborate...") without performance degradation. This approach is plug-and-play, requires no gradient updates, and is shown to be complementary to Chain-of-Thought (CoT) prompting.

Empirical evaluations across ten zero-shot benchmarks demonstrate RaR’s efficacy. On GPT-4, RaR increased average accuracy from 64.95% to 89.77%. Notable gains include Last Letter Concatenation (2) rising from 52.05% to 99.09%, and Last Letter Concatenation (4) surging from 21.36% to 86.82%. In contrast to zero-shot CoT, which degraded performance on the Chinese Idiom task (32.38% baseline vs. 31.43% CoT), RaR achieved 35.24%. On StereoSet, RaR improved the Language Modeling Score to 97.73% (vs. 84.09% baseline) and the Fair Score to 42.27% (vs. 6.82% baseline). When combined with flawed few-shot CoT examples, RaR mitigated performance collapse, lifting accuracy from 52.27% to 95.45%. Cross-model experiments confirm that while all architectures benefit, GPT-4 achieves the largest gains. Furthermore, GPT-4’s rephrased queries significantly boosted Vicuna-13b’s performance on tasks where Vicuna’s self-rephrasing failed (e.g., Last Letter (2): 5.45% self-rephrased vs. 10.45% GPT-4-rephrased). The authors also observe that iterative self-rephrasing by GPT-4 converges to consistent clarifications by the third iteration, though the queries become increasingly elaborate.

The authors delineate several limitations. RaR does not universally guarantee improvement; certain tasks like Sports yielded marginal or slightly diminished results for GPT-3.5 and Vicuna. Weaker models exhibit poor self-rephrasing capabilities, occasionally distorting original intent or introducing irrelevant constraints. The method also introduces computational overhead due to the additional generation step. Additionally, while RaR complements CoT, zero-shot CoT remains prone to hallucination snowballing and bias amplification on complex reasoning tasks. The authors conclude that RaR effectively bridges the human-LLM interpretability gap but emphasize that rigorous human-LLM review of task design remains necessary to ensure intention clarity.
