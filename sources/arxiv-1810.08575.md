---
id: arxiv:1810.08575
type: paper
title: Supervising strong learners by amplifying weak experts (Christiano et al.)
url: https://arxiv.org/abs/1810.08575
retrieved: '2026-07-12'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# Supervising Strong Learners by Amplifying Weak Experts

### Core Problem
The authors address the challenge of training machine learning systems for tasks that are "beyond human scale," where the objective is too complex for a human to directly demonstrate or evaluate. In such cases, practitioners often rely on easier-to-specify proxy objectives. However, aggressively optimizing these proxies can lead to pathological behavior (Goodhart's Law), where the agent maximizes the proxy without achieving the actual goal. The core problem is the lack of a meaningful training signal for tasks that exceed human cognitive or observational capacities.

### Method: Iterated Amplification
The proposed solution is **Iterated Amplification**, a framework where a human expert $H$ trains an agent $X$ by using multiple copies of $X$ as assistants. Instead of $H$ solving a problem alone, the composite system $\text{Amplify}^{\text{H}}(X)$ allows $H$ to decompose a complex problem into simpler subproblems, which $X$ then solves. $X$ is subsequently trained to predict the output of this amplified system.

To reduce the burden on the human, the authors introduce a **human predictor** $H'$, which is trained to imitate $H$'s coordination and decomposition strategies. $X$ then learns from $\text{Amplify}^{\text{H}'}(X)$.

#### Step-by-Step Recipe
The training process runs four parallel processes:
1. **Data Collection:** Sample a question $Q$ from a distribution $\mathcal{D}$. The expert $H$ identifies a sequence of $k$ subquestions $Q_1, \dots, Q_k$. The agent $X$ computes subanswers $A_1, \dots, A_k$. Finally, $H$ computes the final answer $A$. This is recorded as a transcript:

$$
\tau = (Q, Q_1, A_1, \ldots, Q_k, A_k, A)
$$

2. **Human Modeling:** Train the predictor $H'$ to predict the decisions made by $H$ (the subquestions $Q_i$ and the final answer $A$) within the transcripts.
3. **Target Generation:** Sample $Q \sim \mathcal{D}$ and use the predictor $H'$ to generate answers via $\text{Amplify}^{\text{H}'}(X)$, resulting in $(Q, A)$ pairs.
4. **Agent Training:** Train $X$ using supervised learning on these $(Q, A)$ pairs.

#### Model Architecture
$X$ is implemented as a Transformer-based encoder-decoder. The encoder processes the context (a set of facts), and the decoder generates answers. The layer transformation is defined as:

$$
y \leftarrow \text{LayerNorm } x + \text{Attention } x
$$

$$
z \leftarrow \text{LayerNorm } y + \text{BatchNorm MLP } y
$$

The model uses an autoregressive MLP for symbol generation, incorporating a pointer network mechanism to copy symbols from the context.

### Key Quantitative Results
The method was tested on five toy algorithmic tasks: permutation powering, expression evaluation, union find, shortest path, and wildcard search.

*   **Performance:** Iterated Amplification solved all five tasks effectively, achieving results comparable to supervised learning from ground truth data, albeit with a "modest slowdown."
*   **Sample Complexity:** While supervised learning from ground truth required tens of millions of examples, Iterated Amplification required only tens of thousands of examples to learn the decompositions. Specifically, the number of oracle calls to $H$ ranged from 6,000 (expression evaluation) to 24,000 (shortest path).
*   **Computational Overhead:** Training required roughly twice the total computation per question compared to standard supervised learning because of the overhead of running $\text{Amplify}^{\text{H}'}(X)$ to generate targets.

### Stated Limitations
The authors acknowledge several simplifications in their experimental setup:
*   **Oracle Substitution:** A hand-coded algorithm replaced the human expert $H$, meaning the study does not prove humans can decompose real-world tasks or that "messy" human decompositions are learnable.
*   **Domain Constraints:** Experiments were limited to combinatorial domains where algorithmic ground truth exists for evaluation.
*   **Learning Objective:** $X$ was trained via supervised learning; the authors suggest that in real-world applications, $\text{Amplify}^{\text{H}}(X)$ might instead be used to learn a reward function for reinforcement learning.
*   **Distribution Dependency:** The success of the method depends on the question distribution $\mathcal{D}$ being broad enough to cover all necessary subquestions generated during the amplification process.
