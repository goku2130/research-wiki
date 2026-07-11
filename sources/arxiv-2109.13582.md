---
id: arxiv:2109.13582
type: paper
title: 'PPL-MCTS: Constrained Textual Generation Through Discriminator-Guided MCTS
  Decoding'
url: https://arxiv.org/abs/2109.13582
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# PPL-MCTS: Constrained Textual Generation Through Discriminator-Guided MCTS Decoding

### Core Problem
The authors address the challenge of controlling text generation in Large Language Models (LMs) to satisfy specific constraints (e.g., sentiment polarity, emotion, or style) without fine-tuning the LM. Traditional fine-tuning is computationally expensive, requires separate models for every constraint, and is often impossible for massive models like GPT-3. While discriminator-guided methods exist, they often rely on "myopic" decoding strategies that make short-term decisions, which can lead to errors as the sequence grows.

### Method
The proposed **PPL-MCTS (Plug and Play Language - Monte Carlo Tree Search)** treats text generation as a tree exploration process. It uses a pre-trained LM to provide token probabilities and a separate discriminator to score how well a completed sequence satisfies the constraint $c$.

#### MCTS Decoding Recipe
Each iteration of the decoding process consists of four steps:
1.  **Selection**: The algorithm recursively selects child nodes from the root to an unexpanded node. Selection is guided by the **Polynomial Upper Confidence Trees (PUCT)** formula to balance exploitation of high-scoring paths and exploration of new ones:

$$
PUCT(i)=\frac{s_{i}}{n_{i}}+c_{puct} p_{\theta}(x_{i}\mid x_{1:t-1})\frac{\sqrt{N_{i}}}{1+n_{i}}
$$

    where $s_i$ is the aggregated score, $n_i$ is the number of simulations through node $i$, $N_i$ is the number of simulations through the parent, and $c_{puct}$ is an exploration constant.
2.  **Expansion**: If the selected node is not terminal, the LM is used to generate its children.
3.  **Simulation (Roll-out)**: A child is sampled according to $p_\theta(x_i \mid x_{1:t-1})$, and a random walk (or other pattern) is used to reach a terminal node.
4.  **Backpropagation**: The final score of the terminal sequence, $p(x \mid c)$, is backpropagated to update the aggregated scores $s_i$ of all parent nodes using an averaging strategy.

#### Model Improvements
*   **Constraint Strength**: To control the trade-off between the LM's likelihood and the constraint, the authors introduce a parameter $\alpha \in [0, 1]$:

$$
p(x \mid c) \propto p_{D}(c \mid x)^{\alpha} p_{\theta}(x)^{1-\alpha}
$$

*   **Repetition Penalty**: To prevent repetitive patterns, a scalar factor $I(i)$ is applied to the temperature $\tau$ of the logits $z$:

$$
p_{\theta}^{'}(x_{i}\mid x_{1:t-1})=\frac{\exp{(z_{i}/(\tau\cdot I(i)))}}{\sum_{v}\exp{(z_{v}/(\tau\cdot I(v)))}}
$$

    where $I(i) > 1$ if the token $x_i$ is already present in the sequence.

### Key Quantitative Results
The method was evaluated on three datasets: `amazon_polarity` (English), `CLS` (French), and `emotion` (Tweets), using accuracy (oracle discriminator), oracle perplexity, and Self-BLEU (diversity).

*   **Performance**: PPL-MCTS achieved state-of-the-art results across all three datasets, remaining competitive with task-specifically trained models like **GeDi-Classloss**.
*   **Comparison**: In the `amazon_polarity` dataset, PPL-MCTS achieved an accuracy of $0.97$ and a 5-Self-BLEU of $0.63$, compared to GeDi-Classloss's $0.96$ accuracy and $0.6$ Self-BLEU.
*   **Human Evaluation (CLS dataset)**: PPL-MCTS and GeDi-Classloss performed similarly in polarity (4.43 vs 4.46) and readability (4.05 vs 4.19) on a 1–5 scale. Both significantly outperformed PPLM (Polarity: 3.74, Readability: 3.12).
*   **Re-ranking Baselines**: The "Sampling-Argmax" method achieved the highest accuracy (e.g., $0.99$ on `amazon_polarity`) and highest diversity (lowest Self-BLEU), but suffered from very high oracle perplexity (16.5), indicating poor readability.
*   **Roll-out Impact**: On the `CLS` dataset, accuracy improved quickly as roll-out size increased, plateauing after only 5 tokens.

### Limitations
*   **Computational Cost**: The roll-out phase is expensive for long sequences. While reducing roll-out size or removing it entirely reduces cost, it diminishes the "long-term vision" of the search.
*   **Readability Trade-off**: Simple re-ranking methods that maximize constraint satisfaction often produce texts with high perplexity (low quality), highlighting the need for the balanced exploration provided by MCTS.
