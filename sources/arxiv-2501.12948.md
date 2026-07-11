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
Traditional Large Language Model (LLM) reasoning capabilities rely heavily on Supervised Fine-Tuning (SFT) using human-annotated demonstrations. This approach is limited by the scalability of human labeling, the introduction of human cognitive biases, and a performance ceiling capped by the quality of human exemplars, which prevents models from exploring superior, non-human-like reasoning pathways.

### Method/Recipe
The researchers developed two models: **DeepSeek-R1-Zero** (pure RL) and **DeepSeek-R1** (a multi-stage pipeline).

#### 1. DeepSeek-R1-Zero (Pure RL)
DeepSeek-R1-Zero was trained directly from the DeepSeek-V3-Base model using reinforcement learning without an initial SFT phase.
*   **Algorithm:** Group Relative Policy Optimization (GRPO), which eliminates the need for a value model by estimating advantages from group scores.
*   **Reward Design:** A rule-based reward system was used to avoid "reward hacking" common in neural reward models.
    *   **Accuracy Reward:** Verifies correctness via deterministic results (e.g., math boxes) or compiler test cases (coding).
    *   **Format Reward:** Incentivizes the model to encapsulate reasoning within `<think>` and `</think>` tags.
*   **Training Process:** The model was trained for 10,400 steps. It naturally developed "aha moments" (self-correction) and increased its thinking time (response length) as performance improved.

#### 2. DeepSeek-R1 (Multi-stage Pipeline)
To address R1-Zero's poor readability and language mixing, DeepSeek-R1 followed a structured pipeline:
1.  **Cold-Start:** Fine-tuning on a small set of high-quality, human-aligned long Chain-of-Thought (CoT) data.
2.  **First RL Stage:** RL training to improve reasoning and language consistency.
3.  **Rejection Sampling & SFT:** Generating correct reasoning trajectories via rejection sampling and combining them with non-reasoning data (writing, factual QA) for SFT.
4.  **Second RL Stage:** A final RL phase using a mix of rule-based rewards (for reasoning) and model-based rewards (for helpfulness and harmlessness).

### Key Formulas
**GRPO Objective:**

$$
\mathcal {J} _ {G R P O} (\theta) = \mathbb {E} [ q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q) ] \frac {1}{G} \sum _ {i = 1} ^ {G} \left(\min \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta_ {o l d}} (o _ {i} | q)} A _ {i}, \text{clip} \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta_ {o l d}} (o _ {i} | q)}, 1 - \varepsilon , 1 + \varepsilon\right) A _ {i}\right) - \beta \mathbb {D} _ {K L} (\pi_ {\theta} | | \pi_ {r e f})\right)
$$

**KL Divergence Estimator:**

$$
\mathbb {D} _ {K L} \left(\pi_ {\theta} | | \pi_ {r e f}\right) = \frac {\pi_ {r e f} (o _ {i} | q)}{\pi_ {\theta} (o _ {i} | q)} - \log \frac {\pi_ {r e f} (o _ {i} | q)}{\pi_ {\theta} (o _ {i} | q)} - 1
$$

**Advantage Calculation:**

$$
A _ {i} = \frac {r _ {i} - \text { mean } (\{r _ {1} , r _ {2} , \cdots , r _ {G} \})}{\text { std } (\{r _ {1} , r _ {2} , \cdots , r _ {G} \})}
$$

**Rule-based Reward:**

$$
Reward_{\text{rule}} = Reward_{\text{acc}} + Reward_{\text{format}}
$$

**Language Consistency Reward:**

$$
Reward_{\text{language}} = \frac{Num(Words_{\text{target}})}{Num(Words)}
$$

### Key Quantitative Results
*   **DeepSeek-R1-Zero:** On the AIME 2024 benchmark, Pass@1 accuracy increased from **15.6% to 77.9%** (reaching **86.7%** with self-consistency decoding).
*   **DeepSeek-R1 Final Performance:**
    *   **AIME 2024 (Pass@1):** 79.8%
    *   **MATH-500 (Pass@1):** 97.3%
    *   **Codeforces Rating:** 2029
    *   **General Alignment:** Final RL stage improved AlpacaEval 2.0 by **25%** and ArenaHard by **17%**.
*   **SFT Dataset:** Approximately **800k samples** (600k reasoning, 200k non-reasoning).

### Stated Limitations
*   **Tool Use & Structure:** Suboptimal structural output capabilities and an inability to use external tools (e.g., calculators, search engines).
*   **Token Efficiency:** Occasional "overthinking" (excessive reasoning) on simple questions.
*   **Language Mixing:** Tendency to mix languages or default to English/Chinese when handling other languages.
*   **Prompt Sensitivity:** Performance degrades with few-shot prompting; zero-shot is recommended.
*   **Software Engineering:** Limited gains over DeepSeek-V3 due to the high computational cost of RL evaluations in this domain.
*   **Reward Hacking:** Pure RL is difficult to scale for non-verifiable tasks (like creative writing) where reliable rule-based reward models cannot be constructed.
