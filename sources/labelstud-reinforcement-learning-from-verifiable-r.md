---
id: labelstud:reinforcement-learning-from-verifiable-r
type: web
title: Reinforcement Learning from Verifiable Rewards
url: https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

The article "Reinforcement Learning from Verifiable Rewards" introduces Verifiable Rewards as a training strategy for Large Language Models (LLMs), notably employed by models like DeepSeek R1 and Tülu 3.

**Core Problem:**
Traditional neural reward functions in Reinforcement Learning from Human Feedback (RLHF) can suffer from biases and reward hacking. There is a need for a more direct, bias-free, and robust method to provide learning signals to LLMs, especially for precision-critical tasks.

**Method/Recipe Step by Step:**
Verifiable Rewards are defined as simple functions that provide a binary ground truth signal (typically "1" for correct, "0" for incorrect) indicating whether a model's output meets a predefined correctness criterion. This approach offers several benefits:

1.  **Ground Truth Alignment:** The binary nature ensures a direct, bias-free objective connection to ground truth, suitable for tasks like mathematical problem-solving and code execution.
2.  **Ease of Design and Evaluation:** Subject matter experts can establish clear correctness criteria without deep machine learning expertise. Automated evaluation is facilitated, minimizing reliance on human judgment.
3.  **Robustness Against Reward Hacking:** Reliance on strict, rule-based evaluations rather than learned approximations reduces the LLM's ability to "hack" the system. No partial credit is given for superficially meeting criteria.

**Types of Verifiable Rewards:**

*   **Mathematical Correctness:** Checks the accuracy of numerical or symbolic computations.
    *   *Example:* For GSM8k dataset, an LLM generates a step-by-step solution. A string-matching algorithm compares the final result with the ground truth.
        *   Score 1: Completely correct.
        *   Score 0.1: Correct format but incorrect answer.
        *   Score 0: Incorrect format.
*   **Code Execution:** Derived from executing generated code and comparing the outcome to an expected result (e.g., unit tests, exceptions).
    *   *Example:* In multi-turn code synthesis:
        *   Reward +1: End of episode and all tests pass.
        *   Reward -1: End of episode and any test fails.
        *   Reward -0.2: No valid code generated.
*   **Instruction-Following and Formatting Rewards:** Evaluates adherence to instructions or guidelines, often via simple string-matching.
    *   *Example:* Checking if keywords are included, if the response is in a specific language, or if reasoning is within `<think>` and `</think>` tags.
    *   *Python Function Example:*
        ```python
        def strict_format_reward_func(completions, **kwargs) -> list[float]:
           pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
           responses = [completion[0]["content"] for completion in completions]
           matches = [re.match(pattern, r) for r in responses]
           return [1.0 if match else 0.0 for match in matches]
        ```
*   **Additional Categories:** Factual accuracy, logical consistency, and compliance with regulations.

**How to Design a Verifiable Reward Function:**

1.  **Collect and Curate Ground Truth Data:**
    *   Leverage expert knowledge to define correct responses.
    *   Use high-quality, reliable, and decontaminated datasets.
    *   Ensure diverse sampling.
2.  **Determine a Rule-Based Reward Function Based on Ground Truth:**
    *   Identify tasks with verifiable outcomes (e.g., mathematical proofs, code execution).
    *   Use deterministic rules (e.g., exact match, passing test cases).
    *   Prioritize rule-based over model-driven rewards.
    *   Implement multi-level scoring for partial correctness.
3.  **Validate the Reward Model Based on Generated Examples:**
    *   Run controlled tests to measure how well the function distinguishes correct from incorrect responses.
    *   Evaluate for robustness against formatting issues.
    *   A/B test with RL agents.
4.  **Integrate it in the RL Process:**
    *   Train policies using Reinforcement Learning with Verifiable Rewards (RLVR).
    *   Optimize with Proximal Policy Optimization (PPO) to balance reward maximization and controlled model divergence.
    *   Monitor reward outputs to prevent exploitation of loopholes.

**Key Quantitative Results and Numbers:**
The article does not present specific quantitative results or numbers from experiments but refers to the successful employment of verifiable rewards by models such as DeepSeek R1 and Tülu 3. It provides examples of scoring mechanisms:
*   Mathematical Correctness: 1 point for completely correct, 0.1 for correct format, 0 for incorrect format.
*   Code Execution: +1 for all tests pass, -1 for any test fails, -0.2 for no valid code generated.
*   Instruction Following: 1.0 if pattern matches, 0.0 otherwise.

**Stated Limitations:**
The article does not explicitly state limitations of verifiable rewards. However, the design process implies a potential challenge in that it "requires expert knowledge, domain expertise, and structured data interfaces," suggesting that tasks lacking these might be difficult to apply verifiable rewards to effectively. The need for "carefully designing verifiable criteria" for high-diversity RL environments also hints at the complexity involved in broader applications.
