---
id: arxiv:2305.20050
type: paper
title: Let's Verify Step by Step
url: https://arxiv.org/abs/2305.20050
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

**Core Problem**
Large language models frequently exhibit logical errors and hallucinations during multi-step reasoning, undermining reliability. Training robust reward models is essential for mitigating these failures, yet the optimal supervision paradigm remains debated. This work investigates whether outcome supervision (ORM), which evaluates only final answers, or process supervision (PRM), which evaluates intermediate reasoning steps, yields more reliable models for complex mathematical reasoning, while addressing the high cost of human feedback.

**Methodology**
The authors implement a sequential training and evaluation pipeline:
1. **Generator Preparation:** A generator model is trained to output newline-delimited, step-by-step solutions by few-shot generating MATH problems, filtering for correct final answers, and finetuning the base model on the resulting dataset.
2. **Mathematical Pretraining:** All base models undergo an additional pretraining phase on approximately 1.5B math-relevant tokens (MathMix) to enhance reasoning capabilities.
3. **Active Data Collection:** Human labelers assign positive, negative, or neutral labels to each reasoning step. To maximize feedback value, the pipeline employs active learning: a lightweight selector PRM scores 1,000 candidate solutions per problem, and the top-K most "convincing wrong-answer" solutions are surfaced to labelers. The PRM is iteratively updated during collection.
4. **Reward Model Training:** ORMs are trained on uniformly sampled solutions using automatic final-answer grading. PRMs are trained via next-token prediction to maximize the log-likelihood of step-level correctness labels.
5. **Inference & Evaluation:** At test time, the PRM computes a solution score as the product of step-level correctness probabilities. Models are evaluated via best-of-N search over uniformly sampled generator solutions, selecting the highest-ranked output for automatic grading.

**Key Formulas**
The solution-level score for a PRM is defined as the joint probability of all steps being correct:

$$
P(\text{solution}) = \prod_{i} P(\text{step}_i \text{ is correct})
$$

For synthetic supervision experiments, a step is classified as incorrect if the large-scale PRM assigns a negative label with probability exceeding a threshold:

$$
P(\text{negative}) > 0.2
$$

**Quantitative Results**
On a representative 500-problem subset of the MATH test set, the large-scale PRM achieves a 78.2% solve rate under best-of-1860 search, significantly outperforming the ORM at 72.4% and majority voting at 69.6%. The performance gap widens as the search budget increases. Active learning yields a 2.6× improvement in data efficiency compared to uniform labeling. Out-of-distribution evaluation on recent AP and AMC STEM exams shows the PRM solving 72.9% of problems (aggregate), compared to 63.8% for the ORM and 61.3% for majority voting. The released PRM800K dataset contains 800,000 step-level human feedback labels across 75,000 solutions.

**Stated Limitations**
The large-scale ORM and PRM training sets are not directly comparable; the ORM uses 100 uniform samples per problem while the PRM relies on an actively learned, answer-incorrect-biased dataset. Automatic grading introduces false positives for ORMs when models reach correct answers via flawed reasoning. Test set contamination remains possible due to online MATH problem postings, though decontamination heuristics and OOD results mitigate this concern. The PRM scoring function inherently penalizes longer solutions due to the multiplicative reduction. Additionally, iterative retraining of the active learning selector PRM introduced instability, and the authors note that findings may not generalize beyond mathematical domains.
