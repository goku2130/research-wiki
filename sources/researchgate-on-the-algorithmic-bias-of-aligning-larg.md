---
id: researchgate:on-the-algorithmic-bias-of-aligning-larg
type: web
title: 'On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference
  Collapse and Matching Regularization'
url: https://www.researchgate.net/publication/398985488_On_the_Algorithmic_Bias_of_Aligning_Large_Language_Models_with_RLHF_Preference_Collapse_and_Matching_Regularization
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Summary: On the Algorithmic Bias of Aligning Large Language Models with RLHF

## Core Problem
The central problem addressed is the **algorithmic bias** inherent in aligning Large Language Models (LLMs) using Reinforcement Learning from Human Feedback (RLHF). The authors identify a phenomenon termed **preference collapse** (or mode collapse), where the optimization process—specifically the use of KL-regularized optimization—systematically suppresses minority viewpoints and underrepresented preferences in the training data. This results in a model that amplifies dominant human preferences while disregarding the diversity of human perspectives, effectively creating a "mode-seeking" behavior that reduces the diversity of the model's output compared to its pre-trained state.

## Method and Theoretical Framework
The research scrutinizes RLHF through the lenses of **social choice theory** and **diversity principles**. To address the failures of RLHF, the authors explore game-theoretic alignment frameworks, specifically **Nash Learning from Human Feedback (NLHF)**.

The methodology for evaluating and improving alignment involves:
1.  **Modeling Alignment as a Game:** NLHF models the alignment process as a two-player zero-sum game.
2.  **Payoff Analysis:** The study systematically examines different choices of payoffs based on pairwise human preferences to determine which can yield desirable alignment properties.
3.  **Consistency Testing:** The framework establishes necessary and sufficient conditions for:
    *   **Condorcet consistency**: Ensuring the model selects the candidate that beats all others in pairwise comparisons.
    *   **Smith consistency**: Ensuring the model selects from the smallest set of candidates who beat all candidates outside that set.
    *   **Diversity**: Utilizing mixed strategies to ensure the model does not collapse into a single dominant mode.

## Key Theoretical Results
The authors provide a theoretical foundation for the robustness of game-theoretic alignment, contrasting it with the failures of RLHF:
*   **RLHF Failure:** RLHF is shown to fail both social choice theory considerations and diversity requirements.
*   **NLHF Advantage:** In contrast, NLHF is proved to possess these desirable properties, particularly in its ability to ensure diversity through the implementation of mixed strategies.
*   **Preference Collapse (Theorem 5.2):** The research demonstrates that without KL regularization, maximizing expected reward under the **Bradley-Terry model** leads to preference collapse, which disproportionately impacts underrepresented groups in the preference data.

## Quantitative Results and Limitations
While the provided text does not list specific numerical performance benchmarks for the primary paper, it establishes a critical **fundamental limitation** regarding "preference matching."

**Preference Matching Impossibility:**
The authors prove that it is impossible to achieve exact preference matching—defined as the model output exactly matching a target policy that fully accounts for the diversity of human preference. Specifically, they show that no smooth and learnable mappings of pairwise preferences can guarantee a unique Nash equilibrium that matches a target policy, even when utilizing standard assumptions such as the **Bradley-Terry-Luce model**.

## Stated Limitations
The primary limitation identified is the inherent impossibility of preference matching within game-theoretic alignment. While mixed strategies can ensure a level of diversity, the framework cannot guarantee a unique equilibrium that perfectly mirrors a diverse target policy.
