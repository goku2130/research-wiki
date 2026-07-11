---
id: arxiv:2409.12917
type: paper
title: Training Language Models to Self-Correct via Reinforcement Learning
url: https://arxiv.org/abs/2409.12917
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

The paper addresses the persistent failure of modern large language models (LLMs) to perform intrinsic self-correction, defined as the ability to revise their own erroneous responses without external feedback or oracle supervision. Prior fine-tuning approaches relying on self-generated correction traces consistently suffer from two primary failure modes: distribution shift, where corrections trained on base-model errors fail under the model’s own mistakes, and behavior collapse, where models learn to produce optimal first-attempt responses followed by superficial or no modifications. To overcome these limitations, the authors propose Self-Correction via Reinforcement Learning (SCoRe), a multi-turn online RL framework that trains a single model entirely on self-generated data.

SCoRe employs a two-stage training recipe designed to decouple first- and second-attempt behaviors and explicitly incentivize corrective progress. In Stage I, the policy is initialized by optimizing second-attempt accuracy while strictly constraining the first-attempt distribution to match the base model via a KL-divergence penalty. This prevents early coupling of the two turns. In Stage II, the model undergoes joint multi-turn RL optimization over both attempts. Crucially, Stage II incorporates reward shaping to prevent collapse to a non-correcting strategy. A progress bonus is added to the second-attempt reward: $\hat{b}(y_2|y_1, y^*) := \alpha \cdot (\hat{r}(y_2, y^*) - \hat{r}(y_1, y^*))$, where $\alpha > 1.0$. This bonus heavily rewards transitions that flip an incorrect response to correct while penalizing transitions that incorrectly modify a correct response.

The formal objectives underpinning SCoRe are defined as follows. The general multi-turn objective maximizes cumulative correctness over $l+1$ turns:
\[
\max _ {\pi_ {\theta}} \mathbb {E} _ {x, y ^ {*} \sim \mathcal {D}, \hat {y} _ {l + 1} \sim \pi_ {\theta} (\cdot | [ x, \hat {y} _ {1: l}, p _ {1: l} ])} \left[ \sum_ {i = 1} ^ {l + 1} \hat {r} \left(\hat {y} _ {i}, y ^ {*}\right) \right].
\]
Stage I optimizes second-attempt reward with a strict first-turn KL constraint:
\[
\max _ {\theta} \mathbb {E} _ {x _ {1}, y _ {1} \sim \pi_ {\theta} (\cdot | x), y _ {2} \sim \pi_ {\theta} (\cdot | [ x _ {1}, p _ {1} ])} \left[ \widehat {r} (y _ {2}, y ^ {*}) - \beta_ {2} D _ {K L} \left(\pi_ {\theta} (\cdot | x _ {1}) | | \pi_ {\text { ref }} (\cdot | x _ {1})\right) \right].
\]
Stage II jointly optimizes both turns with standard KL regularization:
\[
\max _ {\theta} \mathbb {E} _ {x _ {1}, y _ {1} \sim \pi_ {\theta} (\cdot | x), y _ {2} \sim \pi_ {\theta} (\cdot | [ x _ {1}, p _ {1} ])} \left[ \sum_ {i = 1} ^ {2} \widehat {r} (y _ {i}, y ^ {*}) - \beta_ {1} D _ {K L} \left(\pi_ {\theta} (\cdot | x _ {i}) | | \pi_ {\text { ref }} (\cdot | x _ {i})\right) \right].
\]

Empirical evaluations on Gemini 1.5 Flash and Gemini 1.0 Pro demonstrate SCoRe’s efficacy. On the MATH benchmark, SCoRe achieves a 15.6% absolute improvement in intrinsic self-correction ($\Delta(t1,t2)$) over the base model, raising accuracy from 52.6% to 60.0% at turn one and 41.4% to 64.4% at turn two. The fraction of corrected initial errors ($\Delta^{i\to c}$) reaches 5.8%, while erroneous modifications of correct answers ($\Delta^{c\to i}$) drop to 1.4%. On HumanEval, SCoRe yields a 9.1% absolute self-correction gain, improving accuracy from 53.7% to 64.6% across two turns, with $\Delta^{i\to c}$ at 15.2% and $\Delta^{c\to i}$ at 3.0%. Furthermore, inference-time compute scaling experiments show that generating $K$ parallel samples followed by one sequential self-correction turn is more compute-efficient than sampling $2K$ solutions in parallel, yielding a 10.5% accuracy gain versus 7.4%.

The authors acknowledge several limitations. SCoRe was trained exclusively for a single round of iterative self-correction (two attempts total) due to infrastructure constraints, meaning performance on subsequent correction rounds may degrade. The two-stage architecture currently requires separate training runs, and the authors note that unifying these stages would streamline deployment. Additionally, the results suggest that teaching complex meta-strategies like self-correction inherently exceeds standard supervised fine-tuning capabilities, necessitating explicit regularization mechanisms such as progress-based reward shaping to avoid degenerate policy modes.
