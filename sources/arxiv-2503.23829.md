---
id: arxiv:2503.23829
type: paper
title: Expanding RL with Verifiable Rewards Across Diverse Domains
url: https://arxiv.org/abs/2503.23829
retrieved: '2026-07-11'
maturity: comprehensive
topic: verifiable-rewards
---

# Summary: Expanding RL with Verifiable Rewards Across Diverse Domains

### Core Problem
Reinforcement Learning with Verifiable Rewards (RLVR) has successfully improved Large Language Model (LLM) performance in structured domains like mathematics and coding, where rule-based verifiers can easily determine correctness via binary signals (correct/incorrect). However, extending RLVR to broader, unstructured domains—such as medicine, chemistry, psychology, economics, and education—is challenging because reference answers are typically free-form, making rule-based verification ineffective. The authors note that only 60.3% of mathematical problems and 45.4% of complex multi-domain queries possess the single-term numerical answers required for rule-based verification.

### Method
The authors propose a framework that replaces rule-based binary rewards with model-based soft rewards derived from a generative verifier.

**1. Generative Reward Estimation**
The system utilizes a generative LLM $\pi_{\phi}$ as a verifier. The model is prompted to output a binary judgment $c \in \{0, 1\}$ based on the prompt $x$, the expert reference answer $a$, and the model's response $y$.
*   **Binary Reward:** $r_{\phi}(x, a, y_i) = 1(c_i = 1)$.
*   **Soft Reward:** To provide more granular signals and handle ambiguity, the reward is calculated using the probability of the indicative token:

$$
r_{\phi}(x, a, y_i) = \begin{cases} \pi_{\phi}(1 | x, a, y_{i}^{T}) & \text{if } c_i = 1, \\ 1 - \pi_{\phi}(0 | x, a, y_{i}^{T}) & \text{if } c_i = 0, \\ 0 & \text{otherwise.} \end{cases}
$$

**2. Reward Model (RM) Training**
To balance efficiency and performance, the authors train a compact 7B-scale reward model (RM-7B) without extensive domain-specific human annotation:
*   **Distillation:** A larger teacher model (Qwen2.5-72B-Instruct) provides binary judgments for response samples collected during the RL exploration phase.
*   **SFT:** The RM-7B is fine-tuned via supervised learning on these distilled labels (160k samples).

**3. RL Optimization**
The policy $\pi_{\theta}$ is optimized using policy gradient algorithms (REINFORCE, RLOO, or REINFORCE++). The objective function is:

$$
J(\theta) = \mathbb{E}_{(x,a)\sim D} \mathbb{E}_{y_{i}\sim \pi_{\theta}(|x|)} \left[ r_{\phi}(x, a, y_i) \right]
$$

The gradient is computed as:

$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{(x,a)\sim D} \mathbb{E}_{y_{i}\sim \pi_{\theta}(|x|)} \left[ r_{\phi}(x, a, y_i) \nabla_{\theta} \log \pi_{\theta}(y_i | x) \right]
$$

To ensure stability, the authors apply **z-score normalization**:

$$
\bar{r}(x, a, y_i) = \frac{r(x, a, y_i) - \mu_r}{\sigma_r}
$$

A KL divergence penalty is also added to the reward to mitigate bias:

$$
\bar{r}(x, a, y_i) \leftarrow \bar{r}(x, a, y_i) - \beta \log \left( \frac{\pi_0(y_i | x)}{\pi_{\text{ref}}(y_i | x)} \right)
$$

### Key Quantitative Results
*   **Verifier Consistency:** Binary judgments showed high agreement between Qwen2.5-72B and GPT-4o, with Cohen’s Kappa ($\kappa$) exceeding 0.86 for mathematics and 0.88 for college-level problems.
*   **Performance Gains:** The RLVR framework using RM-7B outperformed state-of-the-art models including Qwen2.5-72B-Instruct and DeepSeek-R1-Distill-Qwen-32B, achieving accuracy boosts of up to 8.0% in free-form reasoning tasks.
*   **Scalability:** While rule-based rewards showed unstable performance and eventual degradation as training data increased to 100k, the RM-7B showed consistent improvement.
*   **Out-of-Distribution (OOD) Generalization:** RM-7B significantly outperformed rule-based rewards on the NaturalReasoning (39.8% vs 29.4%) and WebInstruct (44.0% vs 33.9%) benchmarks.

### Limitations
The authors identify several limitations and open questions:
*   **Lack of CoT in Verifier:** The reward model is instructed to output only 0 or 1 without Chain-of-Thought (CoT) reasoning; the necessity of rationales for assessing semantic equivalence remains unexplored.
*   **No Format Constraints:** The method does not employ format-based rewards or constraints, which are common in other RLVR studies to facilitate parsing.
*   **Process Supervision:** The study does not address how to assign rewards to intermediate steps when direct supervision for those steps is unavailable.
