---
id: openai:iterated-distillation-and-amplification-
type: web
title: Iterated Distillation and Amplification (IDA) - OpenAI Blog
url: https://openai.com/index/learning-complex-goals-with-iterated-amplification/
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# Iterated Distillation and Amplification (IDA)

### Core Problem
The primary challenge addressed by Iterated Amplification is the "training signal" problem for tasks that are beyond human scale. In standard machine learning, training requires a signal—such as labels in supervised learning or rewards in reinforcement learning—to evaluate performance. However, many complex real-world tasks (e.g., managing the security of a large computer network or designing a complex transit system) are too intricate for a human to perform or judge directly. Consequently, humans cannot provide the necessary training signals for an AI to learn these high-level goals, leading to a gap where the AI cannot be trained on the full complexity of the desired behavior.

### Method
Iterated amplification generates a training signal for complex tasks by leveraging the assumption that while a human cannot solve a large task, they can:
1. Decompose a complex task into smaller, manageable sub-components.
2. Solve very small, basic instances of the task.

The process follows an iterative "amplification" recipe:

1. **Initial Training:** The system samples the smallest possible subtasks. Humans provide demonstrations for these small tasks, which are used to train the initial AI system.
2. **Task Decomposition and Solving:** The system samples slightly larger tasks. Humans are asked to break these tasks down into the smaller components identified in the previous step. The AI systems trained in Step 1 are then used to solve these individual components.
3. **Distillation:** The solutions to these larger tasks—obtained through the combination of human decomposition and AI execution—are used as a new training signal. The AI is trained to solve these second-level tasks directly, without requiring human decomposition.
4. **Iteration:** This process is repeated. The system continues to sample increasingly composite tasks, using the AI's ability to solve previous-level tasks to generate training signals for the next, more complex level.

The goal is to eventually produce a fully automated system capable of solving highly composite tasks despite the absence of an initial direct training signal for those tasks.

### Key Formulas
The provided source does not list specific mathematical formulas.

### Quantitative Results
The researchers tested the method on five toy algorithmic domains:
* Permutation powering
* Sequential assignments
* Wildcard search
* Shortest path
* Union find

In these experiments, the researchers simulated the "human" element by using algorithmic solutions for sub-pieces while pretending the direct solution for the full task was unknown. The results indicated that iterated amplification performed **competitively** with direct supervised learning. The objective was to match the performance of supervised learning while being "handicapped" by the lack of direct ground truth labels.

### Stated Limitations
The authors acknowledge several limitations regarding the current state of the research:
* **Early Stage:** The technique is in its preliminary stages.
* **Domain Scope:** Experiments have only been conducted on simple toy algorithmic domains rather than real-world, beyond-human-scale tasks.
* **Human Integration:** To avoid the complications of using actual humans as training signals during the prototype phase, the experiments used algorithmic signals to simulate human input; actual human-in-the-loop testing had not yet been performed at the time of the report.
