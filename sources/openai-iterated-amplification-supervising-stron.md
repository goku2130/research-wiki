---
id: openai:iterated-amplification-supervising-stron
type: web
title: 'Iterated Amplification: Supervising strong learners by amplifying weak experts
  (OpenAI blog/paper context)'
url: https://openai.com/index/learning-complex-goals-with-iterated-amplification/
retrieved: '2026-07-12'
maturity: comprehensive
topic: self-improvement-and-self-play
---

# Iterated Amplification: Supervising Strong Learners by Amplifying Weak Experts

### Core Problem
The primary challenge addressed is the generation of training signals for AI systems tasked with goals that are "beyond human scale." While many tasks can be trained via algorithmic signals (e.g., game scores) or human judgment/demonstration, some tasks are too complex for a human to perform or evaluate directly (e.g., managing the security of a massive computer network). In such cases, there is no direct training signal available to supervise the learner, and providing an incorrect signal can lead to dangerous or unintended behaviors.

### Method
Iterated amplification is a technique designed to build a training signal for complex tasks from the ground up, based on the assumption that while a human cannot solve the whole task, they can:
1. Perform very small instances of the task.
2. Decompose a larger piece of the task into smaller, manageable components.

The process follows a recursive "recipe" to amplify a weak expert (the human) into a strong learner (the AI):

1. **Initial Training:** Sample small sub-tasks that humans are capable of solving. Train the AI system to perform these small tasks using human demonstrations.
2. **Task Decomposition:** Sample slightly larger tasks. Instead of solving them directly, a human is used to break the task down into the smaller components identified in the previous step.
3. **Composite Solving:** The AI systems trained in Step 1 solve these smaller components. The human coordinates the assembly of these solutions to solve the larger task.
4. **Direct Training:** The solutions obtained through this human-assisted decomposition process are used as a new training signal. The AI is then trained to solve these second-level tasks directly, without requiring human help.
5. **Iteration:** This process is repeated iteratively, using the AI's ability to solve increasingly complex composite tasks to generate training signals for even larger, more complex tasks.

### Key Formulas
The provided text does not contain specific mathematical formulas.

### Quantitative Results
The researchers tested the method on five toy algorithmic tasks:
*   Permutation powering
*   Sequential assignments
*   Wildcard search
*   Shortest path
*   Union find

In these experiments, the researchers simulated the human element by using an algorithmic training signal to provide the "pieces" of the solution. The results indicated that iterated amplification performed **competitively** with direct supervised learning. The goal of the experiment was to match the performance of supervised learning while being "handicapped" by the lack of direct ground-truth labels.

### Stated Limitations
The authors highlight several limitations regarding the current state of the research:
*   **Early Stage:** The technique is in its preliminary stages.
*   **Domain Restriction:** Experiments have only been conducted on simple, toy algorithmic domains rather than real-world, beyond-human-scale tasks.
*   **Simulation of Humans:** To avoid the complications of using actual human subjects in a prototype, the researchers used algorithmic signals to simulate human decomposition and demonstrations.
*   **Learning Scope:** The prototype focused exclusively on supervised learning, though the authors suggest future versions may incorporate reinforcement learning and actual human feedback.
