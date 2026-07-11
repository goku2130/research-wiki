---
id: arxiv:2511.22570
type: paper
title: 'DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning'
url: https://arxiv.org/html/2511.22570v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

The paper "DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning" addresses the limitations of current large language models (LLMs) in mathematical reasoning, particularly in theorem proving, where rewarding based on final answers is insufficient. The core problem is that correct final answers do not guarantee correct reasoning, and many mathematical tasks require rigorous step-by-step derivation rather than numerical answers, making traditional final answer rewards inapplicable. This leads to LLMs producing mathematically invalid or logically inconsistent proofs and exhibiting high false-positive rates in verification.

The proposed method, DeepSeekMath-V2, aims to achieve self-verifiable mathematical reasoning by training an accurate and faithful LLM-based verifier for theorem proving and then using this verifier to train a proof generator. The process involves several key steps:

1.  **Proof Verification Training**:
    *   **Curating Cold Start RL Data**: Problems requiring proofs are crawled from Art of Problem Solving (AoPS) contests (17,503 problems). Candidate proofs are generated using a variant of DeepSeek-V3.2-Exp-Thinking, which iteratively refines its outputs. Mathematical experts then score these proofs (0 for fundamentally flawed, 0.5 for minor errors, 1 for complete and rigorous) according to high-level rubrics, creating an initial RL dataset $\mathcal{D}_{\nu} = \{(X_i, Y_i, A_i)\}$, where $X_i$ is the problem, $Y_i$ is the proof, and $A_i \in \{0, 0.5, 1\}$ is the expert-annotated score.
    *   **RL Objective for Verifier**: A verifier $c_{\varphi}(\cdot|X, Y, \mathcal{I}_{\nu})$ is trained using reinforcement learning to produce a proof analysis (summary of issues and a score $A'_i$). The RL objective maximizes a reward function with two components:
        *   **Format Reward ($R_{\text{format}}$)**: An indicator function that ensures the output contains specific key phrases ("Here is my evaluation of the solution:" and a score within \boxed{} following "Based on my evaluation, the final overall score should be:").
        *   **Score Reward ($R_{\text{score}}$)**: Rewards based on the proximity between the predicted score $A'_i$ and the annotated score $A_i$: $R_{\text{score}}(A'_i, A_i) = 1 - |A'_i - A_i|$.
        The objective is:
        $$ \max_{c_{\varphi}} \mathbb{E}_{(X_i, Y_i, A_i) \sim \mathcal{D}_{\nu}, (Z'_i, A'_i) \sim c_{\varphi}(\cdot|X_i, Y_i)} [R_{\text{format}}(Z'_i) \cdot R_{\text{score}}(A'_i, A_i)] $$
        where $Z'_i$ is the verifier's final response.

2.  **Meta-Verification**:
    *   To address the issue of verifiers hallucinating non-existent issues while predicting correct scores, a meta-verifier is introduced. This secondary evaluation process assesses whether identified issues truly exist and logically justify the predicted score.
    *   **Meta-Verifier Training**:
        1.  An initial verifier $c_{\varphi}$ is obtained.
        2.  Mathematical experts score the quality of the verifier's responses according to meta-verification rubrics $\mathcal{I}_{m\nu}$, creating dataset $\mathcal{D}_{m\nu} = \{(X_i, Y_i, Z_i, \bar{A}_i)\}$, where $Z_i$ is the verifier's analysis and $\bar{A}_i \in \{0, 0.5, 1\}$ is the expert-annotated quality score.
        3.  A dedicated meta-verifier $c_{\eta}(\cdot|X, Y, Z, \mathcal{I}_{m\nu})$ is trained using an RL objective similar to the verifier training, with format and score rewards.
    *   **Enhanced Verifier Training**: The meta-verifier's feedback is integrated into the verifier's reward function:
        $$ R_V = R_{\text{format}} \cdot R_{\text{score}} \cdot R_{\text{meta}} $$
        where $R_{\text{meta}}$ is the quality score from the meta-verifier. The enhanced verifier is trained on both $\mathcal{D}_{\nu}$ and $\mathcal{D}_{m\nu}$.

3.  **Proof Generation Training**:
    *   A proof generator $c_{\theta}(\cdot|X)$ is trained with the verifier $c_{\varphi}$ serving as a generative reward model. The objective is:
        $$ \max_{\pi_{\theta}} \mathbb{E}_{X_i \sim \mathcal{D}_p, Y_i \sim \pi_{\theta}(\cdot|X_i)} [R_Y] $$
        where $R_Y$ is the reward for the generated proof.
    *   **Self-Verification in Generator**: The generator is prompted to produce a proof $Y$ followed by a self-analysis $Z$ (in the same format as the verifier). The verifier $c_{\varphi}$ assesses both: the proof $Y$ receives score $s = A$, and the self-analysis $Z$ receives a meta-verification score $ms = \bar{A}$. The combined reward function is:
        $$ R = R_{\text{format}}(Y, Z) \cdot (\alpha \cdot R_Y + \beta \cdot R_Z) $$
        where $R_Y = s$ and $R_Z = R_{\text{score}}(s', s) \cdot R_{\text{meta}}(Z)$. Here, $s'$ is the generator's self-predicted score, and $\alpha=0.76$, $\beta=0.24$. This incentivizes faithful self-evaluation and iterative refinement.

4.  **Synergy and Automated Labeling**:
    *   As the generator improves, it produces new proofs that challenge the verifier. These hard-to-verify proofs become valuable training data for enhancing the verifier.
    *   **Automated Labeling Process**:
        1.  For each proof, generate $k$ independent verification analyses.
        2.  For analyses reporting issues (scores 0 or 0.5), generate $m$ meta-verification assessments to validate the identified problems. An analysis is deemed valid if the majority of meta-assessments confirm its findings.
        3.  For each proof, examine analyses that assign the lowest score. If at least 9 such analyses are deemed valid, the proof is labeled with that lowest score. If no legitimate issues are identified across all verification attempts, the proof is labeled with 1. Otherwise, the proof is discarded or routed to human experts.

**Key Quantitative Results and Numbers:**

*   **Verifier Improvement**: On a validation split of $\mathcal{D}_{\nu}$, the average quality score of the verifier's proof analyses (as evaluated by the meta-verifier) improved from **0.85 to 0.96**, while maintaining the same accuracy in proof score prediction.
*   **One-Shot Generation (CNML-Level Problems)**: DeepSeekMath-V2 consistently outperforms GPT-5-Thinking-High and Gemini 2.5-Pro across all categories (algebra, geometry, number theory, combinatorics, inequality). For example, on algebra problems, DeepSeekMath-V2 achieved a mean proof score of **0.60**, compared to **0.54** for GPT-5-Thinking-High and **0.17** for Gemini 2.5-Pro (Figure 1).
*   **Sequential Refinement with Self-Verification**:
    *   **Pass@1**: Improves substantially as maximum sequential attempts increase (Figure 2).
    *   **Best@32**: Self-selected best proofs achieve significantly higher verification scores than the thread average, demonstrating accurate self-assessment.
*   **High-Compute Search (Competition Problems)**:
    *   **IMO 2025**: Solved **5 of 6 problems** (P1, P2, P3, P4, P5) with 83.3% points, achieving gold-level performance.
    *   **CMO 2024**: Solved **4 problems** (P1, P2, P4, P5) and partial credit on another (P6), with 73.8% points, achieving gold-level performance.
    *   **Putnam 2024**: Solved **11 of 12 problems completely** (A1-B4, B5, B6) and the remaining problem with minor errors, scoring **118/120**, surpassing the highest human score of 90.
    *   **IMO-ProofBench**: Outperforms DeepMind's DeepThink (IMO Gold) on the basic set and remains competitive on the advanced set (Figure 3).

**Stated Limitations:**

*   The approach of rewarding correct final answers faces fundamental limitations: correct answers do not guarantee correct reasoning, and it is inapplicable to theorem proving tasks requiring rigorous derivation.
*   LLMs trained with final answer rewards still frequently produce mathematically invalid or logically inconsistent natural-language proofs and exhibit high false-positive rates in verifying proof validity.
*   When prompted to both generate and analyze its own proof in one shot, the generator tends to claim correctness even when external verifiers easily identify flaws, indicating a failure to evaluate its own work with the same rigor as a dedicated verifier.
*   For challenging problems, models often cannot generate comprehensive and rigorous proofs in a single attempt within the 128K token limit.
*   The hardest IMO-level problems remain challenging for the model, even with scaled test-time compute.
