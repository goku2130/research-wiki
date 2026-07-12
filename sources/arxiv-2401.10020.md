---
id: arxiv:2401.10020
type: paper
title: Self-Rewarding Language Models
url: https://arxiv.org/abs/2401.10020
retrieved: '2026-07-12'
maturity: comprehensive
topic: rlaif
---

# Self-Rewarding Language Models

### Core Problem
Traditional alignment methods, such as Reinforcement Learning from Human Feedback (RLHF), rely on separate reward models trained from human preferences. This architecture introduces two primary bottlenecks: first, the model's performance is capped by the quality and level of human feedback; second, the reward model remains frozen during LLM training, preventing it from improving as the language model evolves.

### Method
The authors propose **Self-Rewarding Language Models**, where a single model is trained to simultaneously follow instructions and act as its own reward model via LLM-as-a-Judge prompting. This creates a "virtuous circle" where improvements in instruction following also enhance the model's ability to provide higher-quality rewards for its own subsequent training.

#### Step-by-Step Recipe
1.  **Initialization ($M_1$):**
    *   **Instruction Fine-Tuning (IFT):** The base pretrained model ($M_0$) is fine-tuned on a seed set of human-authored (prompt, response) pairs.
    *   **Evaluation Fine-Tuning (EFT):** The model is fine-tuned on seed data to act as a judge. This involves prompts asking the model to evaluate a response based on five additive criteria (relevance, coverage, usefulness, clarity, and expertise), producing a chain-of-thought justification and a final score $r \in [0, 5]$.
2.  **Self-Instruction Creation:**
    *   **Prompt Generation:** New prompts $x_i$ are generated using few-shot prompting from the seed IFT data.
    *   **Candidate Generation:** The current model $M_t$ generates $N$ diverse candidate responses $\{y_i^1, \dots, y_i^N\}$ for each prompt.
    *   **Self-Evaluation:** $M_t$ evaluates its own responses using the LLM-as-a-Judge prompt, assigning scores $r_i^n$.
3.  **Instruction Following Training:**
    *   **Preference Pair Construction:** For each prompt, the highest-scoring response is designated the winner ($y_i^w$) and the lowest-scoring the loser ($y_i^l$). Pairs with identical scores are discarded.
    *   **Optimization:** The next model $M_{t+1}$ is trained using Direct Preference Optimization (DPO) on these AI Feedback Training (AIFT) pairs.
4.  **Iteration:** The process is repeated, where $M_{t+1}$ is used to generate and reward the data for $M_{t+2}$.

#### Model Sequence
The training progression is defined as:

$$
M_0 \xrightarrow{\text{SFT (IFT+EFT)}} M_1 \xrightarrow{\text{DPO}(\text{AIFT}(M_1))} M_2 \xrightarrow{\text{DPO}(\text{AIFT}(M_2))} M_3
$$

### Key Quantitative Results
The authors used Llama 2 70B as the base model. Results demonstrate that both instruction following and reward modeling abilities improve across iterations.

**Instruction Following (AlpacaEval 2.0 Win Rate over GPT-4 Turbo):**
*   **Iteration 1 ($M_1$):** 9.94%
*   **Iteration 2 ($M_2$):** 15.38%
*   **Iteration 3 ($M_3$):** 20.44%
*   *Comparison:* $M_3$ outperformed Claude 2 (17.19%), Gemini Pro (16.85%), and GPT-4 0613 (15.76%).

**Reward Modeling Ability (Alignment with Human Rankings):**
*   **SFT Baseline (IFT only):** 65.1% pairwise accuracy
*   **Iteration 1 ($M_1$):** 78.7% pairwise accuracy
*   **Iteration 2 ($M_2$):** 80.4% pairwise accuracy
*   **Iteration 3 ($M_3$):** 81.7% pairwise accuracy

**General Performance (MT-Bench Overall Score):**
*   **SFT Baseline:** 6.85 / 10
*   **Iteration 3 ($M_3$):** 7.25 / 10

### Limitations
*   **Reasoning Gap:** The approach showed minimal gains in mathematics and logical reasoning, suggesting it primarily helps the model better utilize existing knowledge rather than acquiring new reasoning capabilities.
*   **Length Bias:** There is a noted increase in response length across iterations ($M_1$: 1092 tokens $\to M_3$: 2552 tokens), which may correlate with higher perceived quality by judges.
*   **Preliminary Scope:** The study only tested three iterations in one setting; the "scaling laws" for more iterations or different model sizes remain unexplored.
*   **Safety and Hacking:** The authors did not conduct formal safety evaluations and noted the potential for "reward-hacking" within the self-rewarding framework.
