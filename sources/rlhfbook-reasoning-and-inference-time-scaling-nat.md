---
id: rlhfbook:reasoning-and-inference-time-scaling-nat
type: web
title: Reasoning and Inference-Time Scaling - Nathan Lambert
url: https://rlhfbook.com/c/07-reasoning
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

The provided text discusses the emergence and significance of **Reasoning Models** and **Inference-Time Scaling** in language model performance, particularly through the application of **Reinforcement Learning with Verifiable Rewards (RLVR)**.

### Core Problem
The core problem addressed is the need to significantly improve language model performance in domains requiring precise reasoning, such as mathematics, coding, and science, by leveraging more computation during inference. Traditional Reinforcement Learning from Human Feedback (RLHF) struggles with subjective evaluation, making it less suitable for tasks with objectively verifiable correct answers.

### Method/Recipe Step by Step
The method, RLVR, is a form of reinforcement learning that proceeds similarly to RLHF but replaces the subjective reward model with an objective scoring function based on verifiable outcomes.

1.  **Initial Model Training**: Models undergo self-supervised learning on vast internet data (the "bulk of the cake"), followed by supervised fine-tuning (SFT) for instruction following (the "icing").
2.  **RLVR Training Cycle**: This is the "cherry on top" and involves an iterative process:
    *   **Sample Multiple Answers**: For a given question or prompt, the model generates multiple potential responses.
    *   **Verification Function Application**: A verification function (scoring function) is applied to each generated response.
        *   **Mathematics**: For math problems, the verification function checks if an `extracted_answer` (e.g., using `\boxed{}` notation or "The answer is:") matches the ground truth. A positive reward (e.g., 1) is given for correctness, and 0 otherwise.
        *   **Code Generation**: For coding tasks, unit tests serve as verification functions. If all unit tests pass, a reward (e.g., 1) is given; if any fail, the reward is 0. Partial credit proportional to passed tests can also be used.
    *   **Gradient Steps**: The model takes gradient steps to increase the likelihood of generating correct answers.
    *   **Repeat**: This cycle is repeated, revisiting the same data hundreds or thousands of times, allowing the model to learn and reinforce positive behaviors.

This process enables models to "think" extensively before answering, generating "thinking tokens" that represent reasoning chains, which are strongly correlated with improved downstream performance.

### Key Formulas in LaTeX
The text describes the reward mechanism for RLVR but does not provide explicit mathematical formulas in LaTeX. However, the reward function can be conceptualized as:

*   **For Mathematics**:
    $R(\text{response}) = \begin{cases} 1 & \text{if extracted\_answer}(\text{response}) == \text{ground\_truth\_answer} \\ 0 & \text{otherwise} \end{cases}$

*   **For Code Generation**:
    $R(\text{response}) = \begin{cases} 1 & \text{if all unit tests pass for code in response} \\ 0 & \text{otherwise} \end{cases}$
    (or a proportional reward for partial credit).

### Key Quantitative Results and Numbers
*   **Timeline**: Inference-time scaling and reasoning models enabled "a massive step in language model performance at the end of 2024, through 2025, and into the future."
*   **Training Epochs**: RLVR involves "hundreds or thousands of epochs over the same few data points" compared to "1 or 2 epochs of loss updates" for standard instruction tuning.
*   **Model Examples**:
    *   OpenAI's o1 (released 2024)
    *   DeepSeek R1 (released 2025-01-22)
    *   Kimi 1.5 (released 2025-01-22)
    *   Open-Reasoner-Zero (released 2025-03-31)
    *   Seed-Thinking 1.5 (released 2025-04-10)
    *   Phi-4 Reasoning (released 2025-04-30, a 14B model)
    *   Llama-Nemotron (released 2025-05-02)
    *   INTELLECT-2 (released 2025-05-12)
    *   Xiaomi MiMo (released 2025-05-12)
    *   Qwen 3 (released 2025-05-14)
    *   Hunyuan-TurboS (released 2025-05-21)
    *   Skywork OR-1 (released 2025-05-28)

### Stated Limitations
*   **Prerequisite Capabilities**: RL training for reasoning became viable only with leading models from approximately 2024 onwards, indicating that a certain level of underlying model capability was necessary.
*   **Generalization**: While improvements on training questions generalize to unseen questions and "some" domains, the extent of this generalization is not fully quantified.
*   **Computational Cost**: The reasoning stage in models like DeepSeek R1 can take "thousands of tokens" before producing an answer, implying significant computational cost during inference. Long-context language models are a prerequisite for advanced reasoning behavior.
*   **Scope of Inference-Time Scaling**: While RL training is a "short path" to inference-time scaling, other methods exist and will continue to be explored. The correlation between increased token generation and downstream performance is crucial; otherwise, it is "just wasted energy."
*   **Prior Work Limitations**: Earlier reasoning research (e.g., STaR, TRICE) did not scale to the same level as DeepSeek R1 and subsequent models, or they required sacrifices in overall model performance for specialized abilities.
