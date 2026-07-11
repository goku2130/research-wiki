---
id: cameronrwolfe:demystifying-reasoning-models-by-cameron
type: web
title: Demystifying Reasoning Models - by Cameron R. Wolfe, Ph.D.
url: https://cameronrwolfe.substack.com/p/demystifying-reasoning-models
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

Reasoning models represent a new paradigm in Large Language Model (LLM) research, differing from standard LLMs by incorporating a variable "thinking" phase before generating a final answer. This "thinking" involves generating internal, long Chains of Thought (CoT), also referred to as reasoning traces or trajectories, which are hidden from the user but summarized in the final output.

### Core Problem

The core problem addressed by reasoning models is the limitation of standard LLMs in solving complex, verifiable tasks (e.g., advanced mathematics, coding, scientific questions) that require decomposition, error detection, and exploration of alternative solutions. While traditional LLMs rely heavily on scaling laws (larger models, more data) and alignment techniques (SFT, RLHF), reasoning models aim to achieve significant performance improvements by explicitly modeling and optimizing the reasoning process itself.

### Method/Recipe Step by Step

1.  **Pretraining**: Similar to standard LLMs, reasoning models undergo pretraining on vast amounts of raw textual data from the internet.
2.  **Alignment**: Models are aligned to produce human-preferable outputs using supervised finetuning (SFT) and reinforcement learning from human feedback (RLHF).
3.  **Reasoning Training (Reinforcement Learning)**: This is the distinguishing step. Reasoning models are trained using large-scale reinforcement learning (RL) to develop effective "thinking" capabilities. This training enables them to:
    *   Decompose complex problems into smaller, manageable parts.
    *   Critique their own partial solutions and identify errors.
    *   Explore multiple alternative solutions.
    *   Generate long Chains of Thought (CoT) as internal reasoning steps.
4.  **Inference-Time Compute Control**: During inference, the length of the generated CoT can be varied to control the computational effort. More complex problems can be allocated more compute by generating longer CoTs, while simpler problems require shorter CoTs, saving computational resources.
5.  **Output Generation**: After the internal reasoning process, the model provides a final answer, often accompanied by a model-written summary of its internal CoT, rather than the full trace.

### Key Formulas (Implicit)

The text does not provide explicit mathematical formulas. However, it implicitly refers to:

*   **LLM Scaling Laws**: The principle that "we get better results by pretraining larger models on more data." This suggests a relationship between model size, data quantity, and performance, though no specific formula is given.
*   **Reinforcement Learning (RL)**: The training of reasoning models involves "large-scale reinforcement learning (RL)." This implies the use of an objective function to maximize a reward signal, typical in RL, but the specific formulation is not detailed.
*   **Majority Vote**: For evaluation, a "majority vote among 64 parallel output samples" is mentioned, suggesting an ensemble method where:

$$
\text{Final Answer} = \text{mode}(\text{Output}_1, \text{Output}_2, \dots, \text{Output}_{64})
$$

    where $\text{mode}$ represents the most frequent answer among the parallel samples.

### Key Quantitative Results and Numbers

*   **o1-preview vs. GPT-4o**:
    *   **AIME 2024**: o1-preview (single sample) outperforms GPT-4o. o1-preview (64 samples) further improves performance.
    *   **Codeforces**: o1-preview (single sample) outperforms GPT-4o. o1-preview (64 samples) further improves performance.
    *   **GPQA Diamond**: o1-preview (single sample) outperforms GPT-4o. o1-preview (64 samples) further improves performance.
*   **o1 vs. GPT-4o on AIME problems**:
    *   GPT-4o solved **12%** of AIME problems.
    *   o1 solves **74% to 93%** of AIME problems, depending on inference settings.
*   **o1 performance**:
    *   Places among the top **500 students** in the US on AIME 2024.
    *   Ranks within the **11th percentile** of competitive human programmers on Codeforces.
*   **o3 performance**:
    *   **ARC-AGI benchmark**: Achieves **87.5%** score, exceeding human-level performance of 85%. GPT-4o achieves **5%** accuracy on this benchmark.
    *   **SWE-Bench Verified**: Achieves **71.7%** accuracy.
    *   **Codeforces**: Achieves an Elo score of **2727**, ranking among the top 200 competitive programmers.
    *   **EpochAI's FrontierMath**: Achieves **25.2%** accuracy, improving upon the previous state-of-the-art of **2.0%**.
*   **o3-mini performance**:
    *   **Cost reduction**: **80%** cheaper than the full o1 model.
    *   **Efficiency**: Delivers responses **24% faster** than o1-mini, with an average response time of **7.7 seconds** compared to **10.16 seconds**.
    *   **Reasoning Effort Settings**:
        *   **Low effort**: Matches o1-mini performance.
        *   **High effort**: Exceeds performance of all other OpenAI reasoning models (including full o1).
    *   **Human Preference**: Scores higher in human preference studies compared to prior reasoning models.
*   **Google Gemini-2.0 Flash Thinking**: Lags behind the performance of o1 and o3-mini.
*   **Grok-3 Reasoning Beta**:
    *   Exceeds o3-mini with high reasoning efforts.
    *   Achieves **96%** accuracy on AIME’24, close to o3's **97%**.
*   **GPQA benchmark**: Experts (PhD holders) reach **65%** accuracy; highly skilled non-experts reach **34%** accuracy.

### Stated Limitations

*   **Initial Reasoning Models' Capabilities**: Early reasoning models were "less capable than standard LLMs in many ways" initially, despite improving reasoning capabilities significantly.
*   **Closed Models**: OpenAI's o1, o1-mini, o3, and o3-mini are "closed models," meaning there is no public information about their internal workings or how they are trained beyond general statements.
*   **Benchmark Saturation**: Benchmarks like GSM8K are "completely saturated" by reasoning models, making them ineffective for meaningfully evaluating the best reasoning models.
*   **Verification of o3 Results**: The public did not have access to the full o3 model to verify its reported results at the time of writing.
