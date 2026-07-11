---
id: arxiv:2511.22570
type: paper
title: 'DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning'
url: https://arxiv.org/abs/2511.22570
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

# DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning

DeepSeekMath-V2 addresses the fundamental limitations of reinforcement learning (RL) in mathematical reasoning, specifically the reliance on final-answer rewards. While effective for quantitative tasks, final-answer rewards are unreliable proxies for reasoning correctness and are inapplicable to theorem proving, where rigorous step-by-step derivation is the primary objective. This creates a "generation-verification gap" where models can produce proofs but cannot reliably verify their validity or identify logical flaws.

### Methodology

The authors propose a synergistic framework that iteratively improves a proof generator and a proof verifier.

#### 1. Proof Verification Training
The system first trains a verifier $\pi_{\varphi}$ to evaluate proofs based on high-level rubrics ($\mathcal{I}_{\upsilon}$), assigning scores of $1$ (rigorous), $0.5$ (sound logic with minor errors), or $0$ (fundamentally flawed). 
*   **Cold Start:** An initial dataset $\mathcal{D}_{\upsilon}$ was created by crawling 17,503 problems from Art of Problem Solving (AoPS) and having experts score candidate proofs.
*   **RL Objective:** The verifier is trained using Group Relative Policy Optimization (GRPO) with a reward function combining a format reward ($R_{\text{format}}$) and a score reward:

$$
R_{\text{score}}(A', A) = 1 - |A' - A|
$$

    where $A'$ is the predicted score and $A$ is the annotated score.

#### 2. Meta-Verification
To prevent the verifier from hallucinating issues to justify a low score, a secondary **meta-verifier** is trained to assess the quality of the verifier's analysis. The meta-verifier evaluates whether identified defects actually exist and if the score matches the findings. The enhanced verifier reward becomes:

$$
R_V = R_{\text{format}} \cdot R_{\text{score}} \cdot R_{\text{meta}}
$$

where $R_{\text{meta}}$ is the quality score provided by the meta-verifier.

#### 3. Proof Generation Training
The proof generator $\pi_{\theta}$ is trained using the verifier as the reward model. To instill self-awareness, the generator is prompted to produce both a proof $Y$ and a self-analysis $Z$. The reward function is:

$$
R = R_{\text{format}}(Y, Z) \cdot (\alpha \cdot R_Y + \beta \cdot R_Z)
$$

where $\alpha=0.76$ and $\beta=0.24$. $R_Y$ is the verifier's score of the proof, and $R_Z$ is a combination of the accuracy of the generator's self-assessment and the meta-verification score of the self-analysis:

$$
R_Z = R_{\text{score}}(s', s) \cdot R_{\text{meta}}(Z)
$$

This structure incentivizes the generator to faithfully acknowledge errors rather than falsely claiming correctness.

#### 4. Automated Labeling Pipeline
To scale without human experts, the authors implemented an automated labeling process:
1.  Generate multiple independent verification analyses for a proof.
2.  Use the meta-verifier to validate analyses that report issues.
3.  If at least 9 analyses are deemed valid and assign a low score, the proof is labeled with that score; otherwise, if no issues are found across all attempts, it is labeled $1$.

### Key Quantitative Results

DeepSeekMath-V2 demonstrates state-of-the-art performance in natural-language theorem proving, particularly when scaling test-time compute:

*   **Putnam 2024:** Achieved a score of **118/120**, significantly exceeding the highest human participant score of 90.
*   **IMO 2025:** Solved 5 out of 6 problems, achieving a score of **83.3%** (gold-level).
*   **CMO 2024:** Solved 4 problems fully and received partial credit on another, totaling **73.8%** (gold-level).
*   **Verifier Quality:** The average quality score of the verifier's analyses (as measured by the meta-verifier) improved from **0.85 to 0.96**.

### Limitations

Despite these gains, the authors note that the most challenging IMO-level problems remain difficult for the model to solve fully.
