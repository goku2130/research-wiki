---
id: arxiv:2410.02197
type: paper
title: 'Beyond Bradley-Terry Models: A General Preference Model for Language Model
  Alignment'
url: https://arxiv.org/abs/2410.02197
retrieved: '2026-07-10'
maturity: comprehensive
topic: reward-modeling
---

**Core Problem**
Aligning language models with human preferences traditionally relies on scalar reward functions, most notably the Bradley-Terry (BT) model. While computationally efficient, the BT model’s scalar assumption enforces a total ordering that cannot capture context-dependent or intransitive (cyclic) human judgments. Conversely, supervised pairwise preference models can express complex cyclic structures but suffer from quadratic computational scaling ($O(N^2)$) when evaluating candidate sets, alongside asymmetric positional biases inherent to causal attention and template-based concatenation.

**Method and Recipe**
The authors propose the General Preference embedding Model (GPM) and General Preference Optimization (GPO) to bridge expressiveness and efficiency. The implementation follows a structured recipe:
1. **Response Embedding:** Each response $y$ to a prompt $x$ is mapped to a latent preference embedding vector $e_y$ using an eigenvector embedding head applied to the language model’s final hidden state.
2. **Context-Dependent Scaling:** An eigenvalue scale gate computes scaling factors $\lambda$ based solely on the prompt $x$, modulating the importance of different preference dimensions (e.g., helpfulness, safety).
3. **Skew-Symmetric Interaction:** Preferences are modeled via a skew-symmetric operator $M$, constructed as a block-diagonal matrix with $2 \times 2$ blocks of the form $\begin{pmatrix} 0 & \lambda \\ -\lambda & 0 \end{pmatrix}$. The preference score between responses $y_1$ and $y_2$ is computed as the inner product $s(y_1, y_2) = \langle e_{y_1}, M e_{y_2} \rangle$.
4. **Probability Conversion:** The score is converted to a preference probability using the logistic function: $P(y_1 \succ y_2) = \sigma(s(y_1, y_2))$.
5. **Optimization (GPO):** For alignment, GPO replaces scalar rewards with the continuous preference score. It iteratively updates the policy by maximizing the expected preference score against an opponent policy, regularized by KL divergence to the reference model. The empirical win rate is estimated by averaging scores over sampled responses.

**Key Formulas**
The mathematical foundation rests on three core equations:
- Preference Score: $s(y_1, y_2) = \langle e_{y_1}, M e_{y_2} \rangle$
- Preference Probability: $P(y_1 \succ y_2) = \frac{1}{1 + \exp(-s(y_1, y_2))}$
- GPO Objective: $\max_{\pi} \mathbb{E}_{p(x)} \left[ \mathbb{E}_{y \sim \pi} [s(x, y)] - \beta D_{\text{KL}}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)) \right]$
- Empirical Score Estimation: $\hat{s}(x, y) = \frac{1}{K} \sum_{i=1}^K s(x, y_i)$

**Key Quantitative Results**
GPM demonstrates superior expressiveness and alignment performance. On the CyclicPreference dataset, GPM achieves 100% accuracy across four cyclic settings, whereas BT reward models score between 50.0% and 62.9% (performing near random guessing). On the RewardBench benchmark, GPM consistently outperforms BT models. Using Gemma-2B-it, GPM (embedding dimension 6) achieves an average score of 82.29%, surpassing the BT model’s 74.85% by 7.44 points. With Llama-3.1-8B-Instruct, GPM (dimension 8) reaches 91.90% versus the BT model’s 90.56% (+1.34). In downstream alignment via SPPO, GPO with GPM improves win rates over BT-based SPPO. For the 2B model at iteration 3, GPO achieves a win rate of 48.25% compared to 42.12% for BT. For the 8B model at iteration 3, GPO reports a win rate of 41.54% (noted with a +2.98 margin in the source table), while the BT baseline scores 41.64%.

**Stated Limitations**
The source explicitly notes that the theoretical convergence of GPO (Theorem 5.1) relies on the realizability assumption and requires the preference score to be bounded within $[-1, 1]$. Additionally, training stability and score boundedness necessitate explicit normalization of the embedding vectors to unit length. The method’s expressiveness, while theoretically capable of representing any real skew-symmetric preference matrix, depends on the latent space dimensionality and the quality of the prompt-conditioned scaling mechanism.
