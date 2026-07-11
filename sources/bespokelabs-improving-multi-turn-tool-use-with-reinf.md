---
id: bespokelabs:improving-multi-turn-tool-use-with-reinf
type: web
title: Improving Multi-Turn Tool Use with Reinforcement Learning
url: https://bespokelabs.ai/blog/improving-multi-turn-tool-use-with-reinforcement-learning
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

# Improving Multi-Turn Tool Use with Reinforcement Learning

### Core Problem
The primary challenge addressed is the orchestration of multiple tools in multi-turn interactions. While single-tool invocation is common, agents often struggle to combine outputs from multiple tools as inputs for subsequent calls to complete complex workflows. The authors argue that manual prompt engineering is unscalable due to the proliferation of edge cases, and supervised finetuning (SFT) is bottlenecked by the high cost of human demonstrations or the limited capabilities of existing teacher models.

### Method and Training Recipe
The researchers employed Group Relative Policy Optimization (GRPO) to enable a model to self-discover tool-use strategies through trial-and-error. They used **Qwen2.5-7B-Instruct** as the base model and trained it on a subset of the **BFCL (Berkeley Function Calling Leaderboard) v3 multi-turn-base category**.

**Step-by-Step Recipe:**
1. **Data Selection:** 200 questions were taken from the BFCL v3 multi-turn-base category, utilizing a 50-50 train-test split. To accelerate training, only the first query of each example was used.
2. **Reward Design:** A simple binary reward function was implemented:
   - Reward = $1$ if the agent passes the BFCL evaluation check (comprising both state-based and response-based checks).
   - Reward = $0$ otherwise.
3. **Optimization:** The model was trained for 1,600 steps (100 epochs) using the following hyperparameters:
   - **Hardware:** 4x H200 141GB GPUs (3 for training, 1 for vLLM inference).
   - **Batching:** $\text{per\_device\_batch\_size} = 1$ with $\text{gradient\_accumulation\_step} = 4$.
   - **Learning Rate:** Constant $1 \times 10^{-6}$ with 20 warmup steps.
   - **Gradient Clipping:** $\text{max\_grad\_norm} = 0.2$.
   - **KL Divergence:** $\beta = 0.001$, with the reference model updated every $M = 100$ steps.
   - **Policy Steps:** $\mu = 2$ (one on-policy step and one off-policy step per batch).
   - **Filtering:** "Overlong Filtering" was applied to mask the loss for rollouts exceeding the maximum completion length.

### Key Quantitative Results
Using only 100 training samples, the researchers achieved a **23% improvement** in tool-use performance on the evaluation set, with accuracy increasing from **55% to 78%**. Qualitatively, the agent shifted from attempting single, incorrect tool calls to successfully planning and executing sequences (e.g., using `get_nearest_airport_by_city` $\rightarrow$ `get_flight_cost` $\rightarrow$ `book_flight`).

### Findings and Limitations
The authors identified several critical instabilities and design choices during experimentation:

*   **Completion Length Blowup:** A recurring failure mode where the agent produced increasingly long strings of gibberish and random characters, leading to accuracy collapse.
    *   **Ineffective Mitigations:** Dr.GRPO, gibberish penalties (via GPT-4o-mini), and the complete removal of the KL divergence penalty all failed to prevent or even accelerated the collapse.
    *   **Effective Mitigation:** The combination of Overlong Filtering and a small KL weight ($\beta = 0.001$) provided the most stability.
*   **Reward Hacking:** Complex reward functions—incorporating tool execution counts and format checks (reasoning tags)—led to worse stability. The "less is more" approach, using only a correctness-based reward, yielded the best performance.
*   **Reference Model Dynamics:** Updating the reference model every 100 steps boosted performance. The authors posit that a stronger policy model requires a stronger reference model as a regulator to prevent the KL term from pulling the model back toward its initial, lower-performing state.
