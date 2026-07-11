---
id: arxiv:2310.03714
type: paper
title: A Survey of Reinforcement Learning for Large Language Models
url: https://arxiv.org/abs/2310.03714
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

# DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines

### Core Problem
The authors argue that current Large Language Model (LM) pipelines are developed using "hard-coded prompt templates"—lengthy strings of instructions and demonstrations crafted through manual trial and error. This approach is described as brittle and unscalable, akin to hand-tuning weights for a classifier, as specific prompts often fail to generalize across different LMs, data domains, or pipeline architectures.

### Method
DSPy introduces a programming model that abstracts LM pipelines as text transformation graphs, moving away from string manipulation toward a modular, compiled approach. The framework consists of three primary abstractions:

1.  **Signatures**: Declarative specifications of a task's input and output behavior (e.g., `question -> answer`). A signature is defined as a tuple of input fields and output fields, abstracting *what* the transformation does rather than *how* the LM should be prompted.
2.  **Modules**: Parameterized components that implement signatures. These act like neural network layers and can be composed into complex pipelines. Built-in modules include `Predict`, `ChainOfThought`, `ProgramOfThought`, and `ReAct`.
3.  **Teleprompters**: Optimizers that take a program, a training set, and a metric to produce an optimized version of the program by automatically generating and selecting the best prompts or demonstrations.

**The Compilation Process:**
The DSPy compiler optimizes pipelines through three stages:
*   **Stage 1: Candidate Generation**: The compiler identifies all `Predict` modules and uses a "teacher" program (or zero-shot version) to simulate the pipeline. It employs rejection sampling to collect successful traces—input-output pairs that satisfy the validation metric—to serve as few-shot demonstrations.
*   **Stage 2: Parameter Optimization**: The compiler selects the best candidates (demonstrations or instructions) using hyperparameter tuning algorithms such as random search or Tree-structured Parzen Estimators (TPE). Alternatively, `BootstrapFinetune` can be used to update the weights of smaller LMs.
*   **Stage 3: Higher-Order Program Optimization**: The compiler can modify the program's control flow, such as creating ensembles that run multiple versions of a program in parallel and reduce predictions via majority voting.

### Key Quantitative Results
DSPy was evaluated on math word problems (GSM8K) and multi-hop question answering (HotPotQA).

**GSM8K Results:**
*   **GPT-3.5**: Accuracy improved from 24.0% (vanilla) to 44.0% (bootstrap), 64.7% (bootstrap $\times 2$), and reached 88.3% using a `ChainOfThought` ensemble.
*   **Llama2-13b-chat**: Accuracy improved from 7.0% (vanilla) to 28.0% (bootstrap) and reached 49.0% using a `ThoughtReflection` ensemble.
*   **General Gains**: DSPy self-bootstrapped pipelines outperformed standard few-shot prompting by over 25% for GPT-3.5 and 65% for Llama2-13b. They outperformed expert-created demonstrations by 5–46% (GPT-3.5) and 16–40% (Llama2-13b).

**HotPotQA Results:**
*   **GPT-3.5**: A `multihop` program improved from 36.9% (few-shot) to 48.7% (bootstrap) and 54.7% (ensemble) in answer exact match (Ans).
*   **Llama2-13b-chat**: The `multihop` program improved from 34.7% (few-shot) to 42.0% (bootstrap).
*   **Efficiency**: Programs compiled to a 770M-parameter T5-Large were found to be competitive with proprietary GPT-3.5 models using expert-written prompt chains.

### Limitations
The authors note that LMs can be "highly unreliable," although they are efficient at searching solution spaces for multi-stage designs. The current iteration of the compiler focuses primarily on the optimization of demonstrations. Additionally, certain features, such as specifying strict data types (e.g., `bool` or `int`) for signature fields, are described as works in progress. The authors also suggest that higher-order optimization is currently limited to simple ensembles and could be expanded to include dynamic test-time bootstrapping or automatic backtracking.
