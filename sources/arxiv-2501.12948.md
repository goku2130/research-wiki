---
id: arxiv:2501.12948
type: paper
title: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
  Learning'
url: https://arxiv.org/abs/2501.12948
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

### Core Problem
Traditional Large Language Model (LLM) reasoning capabilities heavily rely on Supervised Fine-Tuning (SFT) using human-annotated demonstrations. This approach is limited by the scalability of human labeling, the introduction of human cognitive biases, and a performance ceiling imposed by the quality of human exemplars, which prevents models from exploring superior, non-human reasoning pathways.

### Method and Recipe
The researchers developed two models: **DeepSeek-R1-Zero** (pure RL) and **DeepSeek-R1** (multi-stage pipeline).

#### 1. DeepSeek-R1-Zero (Pure RL)
DeepSeek-R1-Zero was trained on DeepSeek-V3-Base using pure reinforcement learning without an initial SFT phase.
*   **Algorithm:** Group Relative Policy Optimization (GRPO), which eliminates the need for a value model by estimating advantages from group scores.
*   **Reward Design:** A rule-based reward system was used to avoid "reward hacking" associated with neural reward models.
    *   **Accuracy Reward:** Verifies correctness via deterministic results (math) or compilers (code).
    *   **Format Reward:** Incentivizes the model to encapsulate reasoning within `<think>` and `</think>` tags.
*   **Process:** The model was incentivized to generate long chains-of-thought (CoT), leading to the emergent development of self-reflection and verification (the "aha moment").

#### 2. DeepSeek-R1 (Multi-stage Pipeline)
To address R1-Zero's poor readability and language mixing, DeepSeek-R1 followed a four-stage process:
1.  **Cold-Start:** SFT on a small set of high-quality, human-aligned reasoning trajectories.
2.  **First RL Stage:** GRPO training to improve reasoning and language consistency.
3.  **Rejection Sampling & SFT:** A large-scale SFT phase incorporating 600k reasoning samples (via rejection sampling) and 200k non-reasoning samples to enhance general writing and tool-use capabilities.
4.  **Second RL Stage:** A final alignment stage using a combination of rule-based rewards (for reasoning) and model-based rewards (for helpfulness and harmlessness).

### Key Formulas
The GRPO objective maximizes the following:

$$
\mathcal {J} _ {G R P O} (\theta) = \mathbb {E} [ q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q) ] \frac {1}{G} \sum _ {i = 1} ^ {G} \left(\min \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta_ {o l d}} (o _ {i} | q)} A _ {i}, \text{clip} \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta_ {o l d}} (o _ {i} | q)}, 1 - \varepsilon , 1 + \varepsilon\right) A _ {i}\right) - \beta \mathbb {D} _ {K L} (\pi_ {\theta} | | \pi_ {r e f})\right)
$$

The KL divergence is calculated as:

$$
\mathbb {D} _ {K L} \left(\pi_ {\theta} | | \pi_ {r e f}\right) = \frac {\pi_ {r e f} (o _ {i} | q)}{\pi_ {\theta} (o _ {i} | q)} - \log \frac {\pi_ {r e f} (o _ {i} | q)}{\pi_ {\theta} (o _ {i} | q)} - 1
$$

The advantage $A_i$ is computed relative to the group:

$$
A _ {i} = \frac {r _ {i} - \text { mean } (\{r _ {1} , r _ {2} , \cdots , r _ {G} \})}{\text { std } (\{r _ {1} , r _ {2} , \cdots , r _ {G} \})}
$$

The rule-based reward is:

$$
Reward_{\text{rule}} = Reward_{\text{acc}} + Reward_{\text{format}}
$$

The language consistency reward is:

$$
Reward_{language} = \frac{Num(Words_{target})}{Num(Words)}
$$

### Key Quantitative Results
*   **DeepSeek-R1-Zero:** On the AIME 2024 benchmark, Pass@1 accuracy increased from **15.6% to 77.9%** during RL training, reaching **86.7%** with self-consistency decoding.
*   **DeepSeek-R1 (Final):**
    *   **AIME 2024 (Pass@1):** 79.8%
    *   **MATH-500 (Pass@1):** 97.3%
    *   **Codeforces (Rating):** 2029
    *   **General Alignment:** The final RL stage improved AlpacaEval 2.0 by **25%** and ArenaHard by **17%**.

### Stated Limitations
*   **Output & Tools:** Suboptimal structural output capabilities and an inability to use external tools (e.g., search engines, calculators).
*   **Efficiency:** Occasional "overthinking" (excessive token use) for simple questions.
*   **Language:** Language mixing persists when handling queries in languages other than English or Chinese.
*   **Prompting:** High sensitivity to prompts; few-shot prompting degrades performance, making zero-shot the recommended setting.
*   **Software Engineering:** Limited improvement over DeepSeek-V3 in software engineering tasks due to the high computational cost of RL evaluations.
*   **Reward Hacking:** Pure RL is susceptible to reward hacking when relying on neural reward models rather than rule-based verifiers.
