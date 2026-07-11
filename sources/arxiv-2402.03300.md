---
id: arxiv:2402.03300
type: paper
title: 'DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language
  Models'
url: https://arxiv.org/abs/2402.03300
retrieved: '2026-07-11'
maturity: comprehensive
topic: grpo
---

The DeepSeekMath paper introduces DeepSeekMath 7B, a language model specialized in mathematical reasoning, and Group Relative Policy Optimization (GRPO), a novel reinforcement learning (RL) algorithm.

**Core Problem:**
Mathematical reasoning is challenging for language models due to its complex and structured nature. Existing open-source models significantly lag behind proprietary models like GPT-4 and Gemini-Ultra in mathematical performance. Traditional PPO-based RL algorithms for improving LLMs are resource-intensive due to the need for a separate critic model.

**Method/Recipe Step-by-Step:**

1.  **Base Model Initialization:** DeepSeekMath 7B is initialized from DeepSeek-Coder-Base-v1.5 7B. The authors note that starting from a code-trained model is more effective than a general LLM.
2.  **Mathematical Pre-training (DeepSeekMath-Base 7B):**
    *   **Data Collection (DeepSeekMath Corpus):** An iterative pipeline is used to gather 120B math-related tokens from Common Crawl.
        *   **Initial Classifier Training:** A fastText model is trained using OpenWebMath instances as positive examples and diverse Common Crawl pages as negative examples.
        *   **First Iteration Mining:** The classifier mines mathematical web pages from a URL-deduplicated Common Crawl (40B HTML pages). Top-ranking pages (initially 40B tokens) are retained.
        *   **Iterative Refinement:** Math-related domains are identified (over 10% of pages collected in the first iteration). Human annotation refines URLs within these domains, and uncollected pages from these URLs are added to the seed corpus. The classifier is updated with this enriched dataset. This process is repeated for four iterations, yielding 35.5M mathematical web pages, totaling 120B tokens.
    *   **Data Decontamination:** Web pages containing 10-gram strings (or exact matches for 3-gram to 9-gram strings) identical to any sub-string from English (GSM8K, MATH) and Chinese (CMATH, AGIEval) mathematical benchmarks are filtered out.
    *   **Pre-training Data Distribution:** The 500B tokens for pre-training DeepSeekMath-Base 7B consist of: 56% DeepSeekMath Corpus, 4% AlgebraicStack, 10% arXiv, 20% Github code, and 10% natural language data from Common Crawl (English and Chinese).
    *   **Training Settings:** AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight\_decay = 0.1), multi-step learning rate schedule (peak after 2,000 warmup steps, decreases to 31.6% after 80% training, 10.0% after 90% training). Max learning rate: 4.2e-4. Batch size: 10M tokens.
3.  **Supervised Fine-Tuning (DeepSeekMath-Instruct 7B):**
    *   **Data Curation:** A mathematical instruction-tuning dataset (776K examples) is created, covering English and Chinese problems with Chain-of-Thought (CoT), Program-of-Thought (PoT), and tool-integrated reasoning formats.
        *   English datasets include annotated GSM8K and MATH problems, and subsets of MathInstruct and Lila-OOD.
        *   Chinese datasets include K-12 mathematical problems with CoT and tool-integrated solutions.
    *   **Training Settings:** Training examples are concatenated to a max context length of 4K tokens. Model trained for 500 steps with a batch size of 256 and a constant learning rate of 5e-5.
4.  **Reinforcement Learning (DeepSeekMath-RL 7B) with Group Relative Policy Optimization (GRPO):**
    *   **GRPO Algorithm:** A variant of PPO that eliminates the need for a critic model.
        *   For each question $q$, a group of $G$ outputs $\{o_1, \dots, o_G\}$ are sampled from the old policy $\pi_{\theta_{old}}$.
        *   Rewards $r = \{r_1, \dots, r_G\}$ are computed for these outputs using a reward model.
        *   Rewards are normalized by subtracting the group average and dividing by the group standard deviation.
        *   **Outcome Supervision:** The advantage $\hat{A}_{i,t}$ for all tokens in output $o_i$ is set to the normalized reward $\widetilde{r}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$.
        *   **Process Supervision:** For each step $j$ in output $o_i$, a process reward model provides $r_i^{index(j)}$. These are normalized $\widetilde{r}_i^{index(j)} = \frac{r_i^{index(j)} - \text{mean}(\mathbf{R})}{\text{std}(\mathbf{R})}$. The advantage $\hat{A}_{i,t}$ is the sum of normalized rewards from following steps: $\hat{A}_{i,t} = \sum_{index(j) \geq t} \widetilde{r}_i^{index(j)}$.
        *   The policy model is optimized by maximizing the GRPO objective, which includes a direct KL divergence penalty from a reference model.
    *   **Iterative GRPO:** The reward model is continually trained using a replay mechanism (10% historical data). The reference model is updated to the current policy model, and the policy model is continually trained with the new reward model.
    *   **Training Settings:** RL is based on DeepSeekMath-Instruct 7B. Training data: chain-of-thought GSM8K and MATH questions from SFT data (approx. 144K questions). Initial reward model trained on DeepSeekMath-Base 7B with LR 2e-5. GRPO policy model LR 1e-6. KL coefficient $\beta = 0.04$. 64 outputs sampled per question. Max length 1024. Training batch size 1024. Policy model updated once per exploration stage.

**Key Formulas (in LaTeX):**

*   **PPO Surrogate Objective (Equation 1):**

$$
\mathcal {J} _ {P P O} (\theta) = \mathbb {E} [ q \sim P (Q), o \sim \pi_ {\theta_ {o l d}} (O | q) ] \frac {1}{| o |} \sum_ {t = 1} ^ {| o |} \min \left[ \frac {\pi_ {\theta} (o _ {t} | q , o _ {<   t})}{\pi_ {\theta_ {o l d}} (o _ {t} | q , o _ {<   t})} A _ {t}, \text{clip} \left(\frac {\pi_ {\theta} (o _ {t} | q , o _ {<   t})}{\pi_ {\theta_ {o l d}} (o _ {t} | q , o _ {<   t})}, 1 - \varepsilon , 1 + \varepsilon\right) A _ {t} \right]
$$

*   **PPO Reward with KL Penalty (Equation 2):**

$$
r _ {t} = r _ {\varphi} (q, o _ {\leq t}) - \beta \log \frac {\pi_ {\theta} (o _ {t} | q , o _ {<   t})}{\pi_ {r e f} (o _ {t} | q , o _ {<   t})}
$$

*   **GRPO Objective (Equation 3):**

$$
\begin{array}{l} \mathcal {J} _ {G R P O} (\theta) = \mathbb {E} [ q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q) ] \\ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| o _ {i} |} \sum_ {t = 1} ^ {| o _ {i} |} \left\{\min \left[ \frac {\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta_ {o l d}} (o _ {i , t} | q , o _ {i , <   t})} \hat {A} _ {i, t}, \text{clip} \left(\frac {\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta_ {o l d}} (o _ {i , t} | q , o _ {i , <   t})}, 1 - \varepsilon , 1 + \varepsilon\right) \hat {A} _ {i, t} \right] - \beta \mathbb {D} _ {K L} [ \pi_ {\theta} | | \pi_ {r e f} ] \right\} \\ \end{array}
$$

*   **Unbiased KL Divergence Estimator (Equation 4):**

$$
\mathbb {D} _ {K L} \left[ \pi_ {\theta} | | \pi_ {r e f} \right] = \frac {\pi_ {r e f} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})} - \log \frac {\pi_ {r e f} (o _ {i , t} | q , o _ {i , <   t})}{\pi_ {\theta} (o _ {i , t} | q , o _ {i , <   t})} - 1
$$

*   **General Gradient Equation for Training Methods (Equation 5):**

$$
\nabla_ {\theta} \mathcal {J} _ {\mathcal {A}} (\theta) = \mathbb {E} [ \underbrace {(q , o) \sim \mathcal {D}} _ {\text {Data Source}} ] \left(\frac {1}{| o |} \sum_ {t = 1} ^ {| o |} \underbrace {G C _ {\mathcal {A}} (q , o , t , \pi_ {r f})} _ {\text {Gradient Coefficient}} \nabla_ {\theta} \log \pi_ {\theta} (o _ {t} | q, o _ {<   t})\right)
$$

**Key Quantitative Results and Numbers:**

*   **DeepSeekMath 7B (Top1 accuracy on MATH benchmark):** 51.7% (without external toolkits and voting techniques).
*   **DeepSeekMath 7B (Self-consistency over 64 samples on MATH):** 60.9%.
*   **DeepSeekMath-Base 7B (on GSM8K):** 64.2%.
*   **DeepSeekMath-Base 7B (on MATH):** 36.2%.
*   **DeepSeekMath-Base 7B (outperforms Minerva 540B on MATH):** 36.2% vs 33.6%.
*   **DeepSeekMath-Instruct 7B (on MATH, Chain-of-Thought):** 46.8%.
*   **DeepSeekMath-RL 7B (on MATH, Chain-of-Thought):** 51.7%.
*   **DeepSeekMath-Instruct 7B (on GSM8K, Chain-of-Thought):** 82.9%.
*   **DeepSeekMath-RL 7B (on GSM8K, Chain-of-Thought):** 88.2%.
*   **DeepSeekMath-Instruct 7B (on MATH, Tool-Integrated Reasoning):** 57.4%.
*   **DeepSeekMath-RL 7B (on MATH, Tool-Integrated Reasoning):** 58.8%.
*   **DeepSeekMath-RL 7B (on CMATH, Chain-of-Thought):** 88.8% (vs DeepSeekMath-Instruct 7B at 84.6%).
*   **DeepSeekMath Corpus size:** 120B tokens (7x Minerva, 9x OpenWebMath).
*   **Code training benefits:** DeepSeek-LLM 1.3B trained with "Code Training for 400B Tokens → Math Training for 150B Tokens" achieved 21.9% on GSM8K (w/o tool use) and 17.4% on GSM8K+Python (w/ tool use), outperforming "General Training" (19.1% and 14.3% respectively).

**Stated Limitations:**

*   **Geometry and Theorem-Proving:** DeepSeekMath's capability in geometry and theorem-proving is relatively weaker than closed models, potentially due to data selection bias in pre-training and fine-tuning.
*   **Model Scale and Few-Shot Capability:** Restricted by its 7B model scale, DeepSeekMath is worse than GPT-4 in few-shot capability. It shows similar performance in zero-shot and few-shot evaluations, unlike GPT-4 which improves with few-shot inputs.
*   **ArXiv Papers' Impact:** The conclusion that arXiv papers are ineffective in improving mathematical reasoning has limitations. The study did not investigate:
    *   The impact of arXiv tokens on specific math-related tasks not included (e.g., informalization of theorems).
    *   The effect of arXiv tokens when combined with other data types.
    *   Whether benefits of arXiv papers would manifest at a larger model scale.
*   **RL Pipeline Improvement:** The current RL pipeline only improves Maj@K performance, suggesting potential for improvement in data source (unlabeled questions with sampled outputs) and advanced sampling strategies (e.g., tree-search methods).
*   **Reward Model Reliability:** All current methods fully trust the reward function signal, which is problematic as reward signals can be unreliable (e.g., PRM800K dataset has ~20% incorrect annotations). This highlights a need for RL algorithms robust against noisy reward signals.
