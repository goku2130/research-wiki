---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-10'
maturity: comprehensive
topic: ppo-for-llms
---

**Core Problem**
Large language models (LMs) trained via next-token prediction exhibit fundamental misalignment with user intent. The standard language modeling objective diverges from the desired goal of following instructions helpfully, honestly, and harmlessly, resulting in unintended behaviors such as generating untruthful, toxic, biased, or unhelpful outputs. Aligning these models requires shifting the training objective from raw text prediction to explicit human preference optimization.

**Methodology**
The authors align GPT-3 using Reinforcement Learning from Human Feedback (RLHF) through a structured three-step fine-tuning pipeline. 
1. **Supervised Fine-Tuning (SFT):** A baseline policy is trained on a dataset of human demonstrations. This dataset comprises approximately 13,000 prompts sourced from labeler-written instructions and prompts submitted to the OpenAI API, with personally identifiable information filtered out.
2. **Reward Model Training:** A reward model (RM) is trained on a separate dataset of approximately 33,000 human-labeled pairwise comparisons. Contractors rank model outputs for identical prompts, and the RM learns to predict which output aligns with human preference.
3. **Policy Optimization via PPO:** The SFT policy is fine-tuned to maximize the RM's scalar reward output using the Proximal Policy Optimization (PPO) algorithm. The pipeline can be iterated by continuously collecting new comparison data on the current best policy to refine the RM and subsequent policy. The final models, termed InstructGPT, are trained at three scales: 1.3B, 6B, and 175B parameters.

*Note on Formulas:* The provided text outlines the methodology algorithmically without explicit mathematical formulations. The optimization process conceptually involves training a reward model to predict human preference rankings and subsequently maximizing this scalar reward signal via the PPO algorithm, without presenting closed-form equations in the excerpt.

**Quantitative Results**
Human evaluations on held-out API prompts demonstrate significant alignment improvements. Outputs from the 1.3B InstructGPT model are preferred over the 175B GPT-3 baseline despite utilizing over 100× fewer parameters. The 175B InstructGPT model is preferred over standard 175B GPT-3 in 85.3% of pairwise comparisons, and over few-shot 175B GPT-3 in 71.4% of comparisons. Regarding truthfulness, InstructGPT generates truthful and informative responses approximately twice as frequently as GPT-3 on the TruthfulQA benchmark. On closed-domain tasks, hallucination rates drop from 41% in GPT-3 to 21% in InstructGPT. Toxicity is reduced by approximately 25% when models are prompted to be respectful, though no significant improvements are observed on bias benchmarks like Winogender and CrowSPairs. To mitigate performance regressions on public NLP datasets (e.g., SQuAD, DROP, HellaSwag), the authors introduce PPO-ptx, which mixes PPO updates with pretraining log-likelihood maximization, preserving labeler preference scores while reducing the alignment tax.

**Stated Limitations**
The authors acknowledge several constraints. The alignment procedure optimizes behavior according to the specific preferences of their contractor labelers and researchers, rather than a generalized notion of human values. InstructGPT models still commit simple errors, including failing to follow instructions, fabricating facts, providing overly hedged responses to straightforward questions, and missing false premises. Furthermore, the models exhibit performance regressions on standard NLP benchmarks unless explicitly mitigated, and they do not significantly reduce bias. Generalization to broader user demographics and inputs where human preferences diverge requires further investigation. Finally, while InstructGPT demonstrates promising cross-task generalization to domains like code summarization and non-English instructions, these capabilities remain limited by the scarcity of direct supervision signals in those areas during fine-tuning.
