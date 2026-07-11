---
id: arxiv:2411.00418
type: paper
title: Self-Evolved Reward Learning for LLMs
url: https://arxiv.org/abs/2411.00418
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

# Self-Evolved Reward Learning (SER) for LLMs

### Core Problem
Reinforcement Learning from Human Feedback (RLHF) is essential for aligning Large Language Models (LLMs) with human preferences. However, the process is bottlenecked by the requirement for massive amounts of high-quality, human-annotated preference data, which is expensive to acquire and difficult to scale. While Reinforcement Learning from AI Feedback (RLAIF) mitigates this, it often relies on the heuristic assumption that an external, stronger LLM can provide high-quality feedback.

### Method: Self-Evolved Reward Learning (SER)
SER is an iterative "feedback-then-train" framework where a Reward Model (RM) generates its own training data to improve itself, reducing dependency on human labels. The process consists of four primary steps:

**1. Self-Labeling**
The RM is initialized as a "seed" model trained on a small fraction (15%) of human-annotated data. It then predicts reward scores for unlabeled data:

$$
r_{i}=RM(Q_{i},A_{i})
$$

**2. Identifying Learning Status and Data Selection**
The system assesses the RM's current capability to determine the filtering strategy. The learning status $\mathcal{S}$ is determined by predicted probabilities $p$ and thresholds $\tau_{\text{high}}=0.55, \tau_{\text{low}}=0.45, \tau_{\Delta}=0.3$:

$$
\mathcal{S}=\begin{cases}{\mathsf{Status}_{1},}&{\mathsf{if}\;(p_{i}^{1}>\tau_{\mathsf{high}}\:\mathsf{and}\;p_{i}^{2}<\tau_{\mathsf{low}})\:\mathsf{or}\;(p_{i}^{1}<\tau_{\mathsf{low}}\:\mathsf{and}\;p_{i}^{2}>\tau_{\mathsf{high}}),}\\ {\mathsf{Status}_{2},}&{\mathsf{else}\:\mathsf{if}\;\Delta_{i}\geq\tau_{\Delta},}\\ {\mathsf{Stop},}&{\mathsf{otherwise}.}\\ \end{cases}
$$

where $\Delta_{i}=|p_{i}^{1}-p_{i}^{2}|$. 
*   **Status 1 (Easier Task):** Focuses on distinguishing clearly "good" vs. "bad" answers.
*   **Status 2 (Harder Task):** Focuses on amplifying subtle differences between answers of similar quality.

**3. Retraining the Reward Model**
High-confidence data is filtered based on the status. For Status 1, samples are selected where $RM(Q, A^1) > \tau_{\text{high}}$ and $RM(Q, A^2) < \tau_{\text{low}}$. For Status 2, samples are selected where $|RM(Q, A^1) - RM(Q, A^2)| > \tau_{\Delta}$. The RM is then retrained using a pairwise loss function:

$$
\mathcal{L}_{\mathtt{pair}}=\frac{1}{|D_{\mathtt{filtered}}|}\sum_{(Q_{j},A_{j}^{1},A_{j}^{2})\in D_{\mathtt{filtered}}}\max(0,\Delta-(RM(Q_{j},A_{j}^{1})-RM(Q_{j},A_{j}^{2})))
$$

The training set for iteration $n$ is the union of newly filtered data and data from iteration $n-1$.

**4. LLM Training via RL**
Once the RM converges, it is used as the reward signal $r = RM(Q, A)$ to optimize the LLM policy $\pi_\phi$ using a modified Proximal Policy Optimization (PPO) framework to maximize the expected reward.

### Key Quantitative Results
*   **Data Efficiency:** SER achieves performance comparable to models trained on full human-labeled datasets using only **15% of the seed data**.
*   **Performance Gains:** Compared to seed models (Loop 0), SER provided an average accuracy increase of **7.88%**. In data-rich scenarios (Stack Overflow), the improvement was **2.4%**.
*   **Comparison to Full Datasets:** The average performance difference between SER and models trained on the full human-labeled dataset was only **0.3%**. In specific cases, SER exceeded the full dataset baseline (e.g., Mistral 7B on the Summarize dataset by **1.93%**).
*   **Cost Reduction:** 
    *   Human labeling cost: $\approx \$0.668$ per sample.
    *   SER (15\% Human-Labeled) cost: $\approx \$0.100$ per sample.
    *   LLM-Labeled + SER cost: $\approx \$0.002$ per sample.
*   **PPO Results:** SER-guided PPO models consistently outperformed SFT baselines in win rates on the HH-RLHF and StackOverflow datasets.

### Stated Limitations
The authors note that the data filtering strategies are currently **empirical** and that a more robust, autonomous method for identifying learning statuses is needed. Additionally, while theoretical convergence proofs are provided in the appendix, they acknowledge that a more **rigorous theoretical analysis** of the overall effectiveness is still required. Finally, due to the high computational cost of PPO, the authors did not integrate LLM response generation into every iteration of the self-evolution loop.
