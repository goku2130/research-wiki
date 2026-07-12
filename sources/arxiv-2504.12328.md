---
id: arxiv:2504.12328
type: paper
title: 'A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges,
  and Future'
url: https://arxiv.org/html/2504.12328v1
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-modeling
---

Reward Models (RMs) are crucial for aligning Large Language Models (LLMs) with human preferences by acting as proxies to provide reward signals for training, particularly in Reinforcement Learning from Human Feedback (RLHF).

**Core Problem:**
The core problem RMs address is ensuring LLMs' behavior adheres to human values and preferences, as scaling LLM parameters or training data alone does not guarantee alignment. Larger models can still produce undesirable outputs like hallucinations or harmful content. RMs aim to reduce human involvement in this alignment process by learning to approximate human preferences.

**Method/Recipe Step by Step:**
The process of developing and utilizing RMs involves three main stages: Preference Collection, Reward Modeling, and Usage.

1.  **Preference Collection (§2.1):**
    *   **Human Preference (§2.1.1):**
        *   Human annotators label pairs of trajectories (policy model-environment interactions) or response pairs from LLMs/humans given prompts.
        *   **Efficiency:** Methods include active learning (e.g., information gain, entropy-based sampling), data augmentation, and sequential pairwise comparison.
        *   **Quality:** Improved through demonstrations, active annotator selection, user-friendly interfaces, fine-grained goals/rules, and selecting diverse batch samples or online collection to prevent distribution shift.
    *   **AI Preference (§2.1.2):**
        *   LLMs generate preference labels, often combined with human-generated labels (e.g., RLAIF).
        *   Off-the-shelf LLMs can directly provide rewards during RL to address out-of-distribution issues.
        *   Instruction templates are used to elicit preferences (e.g., UltraFeedback).
        *   Human-defined principles can instruct RMs.
        *   AI preferences can be integrated with human preferences, such as LLMs generating synthetic critiques or combining LLM responses with human-annotated negative samples.

2.  **Reward Modeling (§2.2):**
    *   **Reward Model Type Level (§2.2.1):**
        *   **Discriminative Reward:**
            *   **Sequence Classifiers (Figure 3(a)):** A base model with an MLP-based reward head outputs a scalar reward for a single response. Examples include conditional RMs and augmenting Bradley-Terry (BT) models with absolute rewards.
            *   **Custom Classifiers (Figure 3(b)):** Take comparison pairs as input or output multiple scores. Methods include comparing candidate pairs with scoring functions, optimizing ensembles of metrics, and leveraging multi-objective rewards with gating layers.
        *   **Generative Reward (Figure 3(c)):**
            *   Leverage LLMs' generative capabilities to provide preference scores.
            *   Use general or specialized LLMs as judges to rate responses or generate better options.
            *   Extract next-token probabilities of answer indicators as scores.
            *   Rewrite responses under minimum editing constraints to obtain token-level scores.
            *   Can be optimized using Self-Instruct and integrated with CoT or RAG.
        *   **Implicit Reward (Figure 3(d)):**
            *   Methods like DPO and $\beta$-DPO directly optimize policies without an explicit RM.
            *   Focus on robustness improvements through data sampling/selection, token-level optimization, reference-free methods, and self-play optimization.
    *   **Reward Granularity Level (§2.2.2):**
        *   **Outcome-level Reward Model (ORM):** Predicts the probability of a completion leading to a correct answer.
            *   **Training:** Typically uses a cross-entropy loss:
                $L_{ORM} = -(\hat{y}_s \log y_s + (1 - \hat{y}_s) \log (1 - y_s))$
                where $y_s$ is the predicted probability and $\hat{y}_s$ is the correctness label for solution $s$.
            *   **Advantages:** Potential in flexible tasks, ease of implementation.
            *   **Disadvantages:** Can lead to false positives, sparse reward.
        *   **Process-level Reward Model (PRM):** Assigns a score to each step in the reasoning process.
            *   **Training:** Uses a standard classification loss function:
                $L_{PRM} = -\sum_{i=1}^{N} (\hat{y}_{s,i} \log y_{s,i} + (1 - \hat{y}_{s,i}) \log (1 - y_{s,i}))$
                where $y_{s,i}$ is the prediction score for step $i$, $\hat{y}_{s,i}$ is its correctness label, and $N$ is the total number of reasoning steps.
            *   **Methods for PRM Training Data:** Human-annotated stepwise labels, Monte Carlo (MC) method for estimating step quality (Q-value), Monte Carlo Tree Search (MCTS), and contrastive learning with stepwise discriminators.
            *   **Advantages:** Potential in reasoning tasks, dense reward, controllable.
            *   **Disadvantages:** High cost for gathering training data, value estimation yields inferior performance, hard to define process reward, scalability and generalization problems, susceptible to reward hacking, high computational overhead in large-scale RL.

3.  **Usage (§2.3):**
    *   **Data Selection:** RMs rank LLM-generated responses, and high-reward data is used for fine-tuning LLMs (e.g., RAFT, RRHF, ReST).
    *   **Policy Training:** RMs provide feedback signals to shape LLM decision-making. Strategies to mitigate low robustness (out-of-distribution generalization, mismatched human judgment) include length-controlled reward setting, causal reward modeling, Bayesian methods, and ensemble RMs.
    *   **Inference:** RMs rank multiple outputs. PRMs are used to evaluate progress and improve reasoning in tree search frameworks (e.g., Tree-PLV, ReST-MCTS*). RMs can also evaluate intermediate decoding steps to dynamically invoke more powerful models.

**Key Quantitative Results and Numbers:**
The survey does not present specific quantitative results or numbers from individual studies but rather discusses general trends and findings. For example, it notes that ORMs tend to be better than PRMs in tasks with flexible processes due to ease of implementation and generalizability, while PRMs show potential in reasoning tasks.

**Stated Limitations (Challenges):**
*   **Preference Collection (§5):**
    *   Potential biases between researchers' and annotators' preferences.
    *   Variations in annotator expertise leading to noisy data, especially in knowledge-intensive tasks.
    *   Inconsistencies between sparse feedback (ratings, rankings) and expensive dense feedback.
*   **Training (§5.2):**
    *   **Overoptimization (Reward Hacking):** RMs can be excessively optimized to narrow evaluation metrics, leading to performance degradation in RL policies. Causes include reward tampering, misleading signals, and sycophancy.
*   **Bias in Evaluation (§5.3):**
    *   RMs used as judges can introduce intrinsic biases towards superficial text quality.
    *   Biases towards specific format patterns in top-ranking RMs and popular benchmarks.
    *   Evaluator biases related to length, concreteness, and empty references.
    *   Preference leakage from the relevance between synthetic data generators and RMs.
*   **RM Evaluation (Benchmarks §4):**
    *   High cost of evaluating RMs by training a full RL policy.
    *   Difficulty in assessing RMs independently due to entanglement with policy performance.
    *   RMs' sensitivity to changes in input style, domain, or format, requiring dynamic, multi-faceted testing.
    *   PRMs face challenges in manual annotation being expensive and not scalable, automated annotation producing unsatisfactory results, difficulty in defining process rewards, and computational overhead in large-scale RL.
