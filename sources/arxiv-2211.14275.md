---
id: arxiv:2211.14275
type: paper
title: Outcome-supervised Verifiers for Planning in Complex Reasoning Tasks
url: https://arxiv.org/abs/2211.14275
retrieved: '2026-07-12'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# Summary: Outcome-supervised Verifiers for Planning in Complex Reasoning Tasks

### Core Problem
The authors investigate the optimal way to supervise language models (LMs) performing complex reasoning tasks: **outcome-based supervision** (rewarding only the final correct answer) versus **process-based supervision** (rewarding each individual reasoning step). The primary goal is to determine if outcome-based approaches can minimize "trace error"—instances where a model reaches the correct final answer through flawed reasoning—which is a critical concern for AI safety and educational applications.

### Method
The study uses the GSM8K dataset (grade school math word problems) and a 70B parameter base LM. The researchers compared several training and decoding configurations:

1.  **Supervised Fine-Tuning (SFT):** The model is fine-tuned to maximize the log-likelihood of human-provided reasoning traces (process-based).
2.  **Reward Models (RMs):**
    *   **Outcome-supervised RM (ORM):** Trained to predict a binary label based on whether the final answer of a full sample is correct.
    *   **Process-supervised RM (PRM):** Trained on human-annotated binary labels for every individual reasoning step.
3.  **Decoding Strategies:** The model generates $K=96$ samples. The best sample is selected via:
    *   **Majority Voting:** Selecting the most common final answer.
    *   **RM-weighted Decoding (Verifier-Voting):** Weighting samples by the RM's estimated correctness probability. The final answer $f^*$ and the best sample $y^*$ are selected as:

$$
f^{*}=\mathbf{a r g\_m a x}_{f}\sum_{y_{i}:\mathbf{f i n a l\_a n s}(y_{i})=f}\mathbf{r m\_p r o b}(y_{i})
$$

$$
y^{*}=\mathbf{a r g\,m a x}_{y:\mathrm{f i n a l~a n s}(y)=f^{*}}\mathbf{r m\_p r o b}(y_{i})
$$

4.  **RL via Expert Iteration:**
    *   **Final-Answer RL:** Expert samples are those with correct final answers.
    *   **ORM-RL:** Expert samples are those with the highest ORM score.
    *   **PRM-RL:** Each step is treated as an episode; the step with the highest PRM score is selected.
5.  **Distillation:** The policy is further refined by SFT on the expert samples generated during RL.

### Key Quantitative Results
The best performing model (SFT + ORM-RL with ORM reranking) achieved a **final-answer error rate of 12.7%** (down from the previous best of 16.8%) and a **trace error rate of 3.4%** (down from 14.0%).

*   **Outcome vs. Process Performance:** Outcome-based and process-based approaches yielded similar final-answer error rates (e.g., 23.5% for Few-shot+Final-Answer RL vs. 22.3% for SFT without RMs). However, process-based supervision significantly reduced trace error (11.4% for SFT vs. 19.8% for Few-shot+Final-Answer RL).
*   **ORM Emulation:** A key finding was that ORMs, despite being trained only on final outcomes, produced predictions that agreed more closely with process-based (PRM) labels than with the outcome labels themselves.
*   **Selective Prediction:** By allowing the model to abstain from answering 30% of the questions, the final-answer error rate dropped from 14.1% to **2.7%**.
*   **OOD Generalization:** On the MATH pre-algebra dataset, the SFT+ORM-RL model achieved a final-answer error of **63.2%**, outperforming GPT-3 (92.3%) but trailing Minerva (29%).

### Stated Limitations
The authors note that their results may be domain-specific to mathematics. In math, incorrect reasoning steps are unlikely to lead to correct final answers, which likely explains why ORMs can effectively emulate process-based feedback. In other domains, "undesirable behaviors" (such as manipulation) might actually help a model achieve a highly-rated outcome, meaning outcome-based supervision would be less likely to induce a correct internal process. Additionally, process-based supervision is constrained by the high cost of human annotation.
