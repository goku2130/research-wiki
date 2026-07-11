---
id: arxiv:2305.20050
type: paper
title: Let's Verify Step by Step
url: https://arxiv.org/abs/2305.20050
retrieved: '2026-07-10'
maturity: comprehensive
topic: reward-modeling
---

Large language models frequently commit logical errors during multi-step reasoning, necessitating reliable reward models to guide search or reinforcement learning. The field relies on two primary training paradigms: outcome supervision (ORM), which evaluates only the final answer, and process supervision (PRM), which evaluates each intermediate reasoning step. Given the high cost of human feedback, rigorously comparing these methods and optimizing data collection is critical for advancing reasoning reliability.

The authors conduct a two-tiered investigation using a fixed generator model finetuned to output newline-delimited, step-by-step solutions to MATH dataset problems. The training recipe proceeds as follows: (1) A generator model is finetuned on filtered, correct MATH solutions to enforce a structured step-by-step format. (2) For process supervision, human annotators label each step as positive, negative, or neutral, yielding PRM800K (800,000 step-level labels across 75,000 solutions and 12,000 problems). To maximize annotation value, an active learning strategy surfaces "convincing wrong-answer" solutions—those achieving an incorrect final answer but receiving high scores from the current PRM. (3) For outcome supervision, models are trained on uniformly sampled solutions graded automatically by final-answer correctness. (4) To isolate confounding factors like dataset bias and false-positive grading, small-scale ablations use the large-scale PRM as a synthetic oracle to supervise smaller models trained on identical datasets. (5) Evaluation employs best-of-N search, where the reward model selects the highest-ranked solution from N uniformly sampled candidates, which is then auto-graded.

The PRM predicts step correctness via a single-token output, trained by maximizing log-likelihood. At inference, the aggregate score for a complete solution is computed as the product of individual step correctness probabilities:

$$
P(\text{solution}) = \prod_{i=1}^{K} P(\text{step}_i \text{ is correct})
$$

where $K$ is the number of steps. This formulation ensures that any single incorrect step drastically reduces the overall solution score.

The process-supervised model achieves a 78% solve rate on a representative subset of the MATH test set using best-of-1860 search, significantly outperforming both the outcome-supervised baseline and majority voting across all N values. The performance gap widens as N increases, indicating superior search capabilities. Active learning improves data efficiency by approximately 2.6$\times$ compared to uniform labeling. Out-of-distribution evaluation on held-out AP and AMC exams demonstrates robust generalization, with the PRM achieving 45% on AP Calculus, 60% on AP Chemistry, 45% on AP Physics, and 84% on AMC10/12.

The large-scale comparison suffers from non-comparable training sets: the ORM is trained on 100 uniformly sampled solutions per problem, while the PRM relies on an actively learned, answer-incorrect-biased dataset. Automatic final-answer grading introduces false positives for ORMs, where models reach correct answers via flawed reasoning. Iterative retraining of the PRM during active learning caused unexplained instability. The study intentionally excludes reinforcement learning fine-tuning of the generator, focusing solely on reward model training. Furthermore, the negative alignment tax observed in mathematics may not generalize to other domains, and potential test set contamination from online sources remains a concern.
