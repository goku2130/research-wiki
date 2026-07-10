---
id: icml:demystifying-long-chain-of-thought-reaso
type: web
title: Demystifying Long Chain-of-Thought Reasoning
url: https://icml.cc/virtual/2025/poster/45449
retrieved: '2026-07-10'
maturity: comprehensive
topic: rl-for-reasoning
---

**Core Problem**
The advancement of reasoning capabilities in large language models (LLMs) has increasingly relied on scaling inference compute, with long chains-of-thought (CoTs) serving as a proven mechanism to enable structured reasoning strategies such as backtracking and error correction. Despite the demonstrated utility of reinforcement learning (RL) in cultivating these extended reasoning trajectories, the precise conditions governing the emergence of long CoTs remain poorly understood. Furthermore, RL training for reasoning tasks demands meticulous architectural and algorithmic design choices, creating a persistent gap between theoretical scaling potential and stable, practical implementation.

**Methodological Protocol**
To systematically disentangle the mechanics underlying long CoT generation, the authors conducted an empirical investigation structured around three sequential experimental phases. First, the protocol evaluates supervised fine-tuning (SFT) as a foundational step, testing whether explicit exposure to step-by-step reasoning examples is strictly necessary or merely beneficial. Second, the methodology scales training compute while continuously monitoring the emergence and stability of extended reasoning trajectories, isolating compute as a variable against reward design. Third, the protocol implements verifiable reward signals paired with large-scale, web-extracted datasets, applying rigorous filtering mechanisms to manage data noise. This phased approach specifically targets out-of-distribution (OOD) reasoning benchmarks, with a concentrated focus on STEM problem-solving domains to assess generalization beyond training distributions.

**Key Findings & Technical Insights**
The empirical analysis yielded three principal technical insights. First, while SFT is not an absolute prerequisite for developing long CoT capabilities, its inclusion substantially simplifies the RL training landscape and improves overall sample efficiency. Second, the emergence of robust reasoning capabilities correlates positively with increased training compute; however, this progression is not deterministic. Consequently, reward shaping emerges as a critical stabilizing mechanism to prevent training collapse and ensure consistent growth in reasoning trajectory length. Third, the scalability of verifiable reward signals proves essential for RL success. The study demonstrates that leveraging large-scale, noisy datasets extracted from the web can be highly effective when paired with rigorous filtering mechanisms. This approach shows particular promise for enhancing OOD reasoning performance, enabling models to generalize to unfamiliar, complex tasks in scientific and engineering contexts.

**Absence of Mathematical Formulations and Quantitative Metrics**
The provided source material does not specify mathematical formulations, algorithmic pseudocode, or explicit numerical benchmarks. Consequently, no key formulas in LaTeX or quantitative results can be extracted for this summary. The authors’ technical contributions are presented at a conceptual and empirical level, emphasizing training dynamics, reward architecture, and data curation strategies over closed-form expressions or reported performance metrics.

**Stated Limitations & Training Constraints**
The study explicitly identifies several constraints inherent to scaling long CoT reasoning. Traditional training methodologies frequently fail to extend reasoning trajectories in a stable manner, often resulting in repetitive or degenerate outputs. The emergence of long CoTs is not guaranteed even with substantial increases in training compute, underscoring the necessity of deliberate reward design to stabilize trajectory length. Additionally, while web-extracted datasets offer scale, they introduce significant noise that requires sophisticated filtering pipelines to prevent the degradation of model reasoning quality. These limitations highlight the delicate balance between data scale, reward verifiability, and training stability required to operationalize advanced reasoning in LLMs.

**Conclusion**
By systematically dissecting the interplay between SFT, compute scaling, reward shaping, and data curation, this work provides a structured framework for optimizing long CoT reasoning in LLMs. The findings establish that stable reasoning emergence depends less on compute alone and more on carefully engineered reward signals and filtering mechanisms, offering actionable guidance for future RL-based reasoning model development.
