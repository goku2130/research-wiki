---
id: rlhfbook:rejection-sampling-nathan-lambert-rlhf-b
type: web
title: Rejection Sampling - Nathan Lambert (RLHF Book)
url: https://rlhfbook.com/c/09-rejection-sampling
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

# Rejection Sampling (RS)

Rejection Sampling (RS) is a versatile method used in preference fine-tuning to curate high-quality candidate completions for model optimization. In the context of language models, the core problem RS addresses is the inability to sample directly from a target distribution of high-quality completions. To solve this, RS uses the current model as a sampling distribution and a trained reward model as a heuristic filter to identify and isolate the highest-quality outputs for further training.

### Methodological Process

The rejection sampling pipeline consists of the following steps:

1.  **Prompt and Reward Model Selection**: A set of prompts is selected (often by reusing prompts from the initial supervised fine-tuning (SFT) stage, though this may risk overfitting). A pre-trained reward model is also required.
2.  **Completion Generation**: The model to be optimized generates $N$ candidate completions for each of the $M$ selected prompts. This stage involves tuning hyperparameters such as sampling temperature, top-p, and max sequence length.
3.  **Reward Scoring**: Every prompt-completion pair is passed through the reward model to assign a scalar score.
4.  **Selection of Top Completions**: The best completions are filtered based on their rewards using one of two primary selection functions:
    *   **Top Per Prompt**: The completion with the maximum reward is selected for every prompt.
    *   **Top Overall Pairs**: The reward matrix is flattened, and the top $K$ completions across the entire dataset are selected.
5.  **Instruction Fine-Tuning**: The starting checkpoint is fine-tuned using standard SFT loss on the selected high-reward completions.

### Key Formulas

Let $X = [x_1, x_2, \dots, x_M]$ be a vector of $M$ prompts. For each prompt $x_i$, $N$ completions are generated, forming a matrix $Y$ where $y_{i,j}$ is the $j$-th completion for the $i$-th prompt.

The reward model $R$ produces a reward matrix $R$ where each element $r_{i,j}$ is computed as:

$$
r_{i,j} = R(y_{i,j} \mid x_i)
$$

**Top Per Prompt Selection**:
The selection function $S(R)$ identifies the index of the maximum reward for each row:

$$
S(R) = [\arg\max_j r_{1,j}, \arg\max_j r_{2,j}, \dots, \arg\max_j r_{M,j}]
$$

The chosen completions are then:

$$
Y_{chosen} = [y_{1, S(R)_1}, y_{2, S(R)_2}, \dots, y_{M, S(R)_M}]
$$

**Top Overall Pairs Selection**:
The reward matrix is flattened into a vector $R_{flat}$ of length $M \times N$. The selection function $S_K$ identifies the indices of the $K$ highest values:

$$
S_K(R_{flat}) = \text{argsort}(R_{flat})[-K:]
$$

### Implementation Details and Quantitative Parameters

The effectiveness of RS is highly dependent on the following hyperparameters:
*   **Sampling Temperature**: Typically set between $0.7$ and $1.0$ to ensure diversity in generated completions.
*   **Completions per Prompt**: Successful implementations typically generate $10$ to $30$ or more completions per prompt; using too few can result in noisy or biased training.
*   **Efficiency**: Throughput can be improved by sorting tokenized completions by length during batch reward model inference to minimize padding tokens.

### Limitations and Distinctions

**Limitations**:
*   **Documentation Gap**: The author notes that RS is widely used (e.g., in Llama 2 Chat, WebGPT) but remains underdocumented regarding canonical implementations and specific SFT settings.
*   **Reward Model Dependency**: The final result is heavily dependent on the quality and accuracy of the reward model.
*   **Overfitting**: Reusing all prompts from the initial SFT stage can lead to overfitting.

**Distinction from Best-of-N (BoN)**:
While both RS and BoN use the generate-and-score procedure, BoN is a sampling technique used at inference time to find the best completion for a prompt. Unlike RS, BoN does **not** involve fine-tuning the underlying model.
