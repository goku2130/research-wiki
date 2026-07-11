---
id: arxiv:2505.13427
type: paper
title: 'MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level
  Supervision'
url: https://arxiv.org/abs/2505.13427
retrieved: '2026-07-11'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision

## Core Problem
Multimodal Large Language Models (MLLMs) often struggle with complex, multi-step mathematical reasoning. While they may occasionally reach a correct final answer, they frequently produce logically inconsistent intermediate steps or "false positives," where a correct answer is derived from flawed reasoning. This is primarily attributed to a lack of fine-grained, step-level supervision during training, as most existing reward models (Outcome Reward Models) only provide scalar feedback on the final result.

## Method
The authors propose **MM-PRM**, a process reward model trained via a fully automated, three-stage scalable framework.

### 1. Policy Model Construction (MM-Policy)
To generate high-quality reasoning trajectories, the authors developed **MM-Policy**. 
* **Data:** A corpus of over 5.1 million mathematical examples was curated from diverse sources (e.g., R-CoT, MAVIS, NuminaMath). 
* **Refinement:** Solutions were restructured into a modular Chain-of-Thought (CoT) schema using Qwen2.5-72B-Instruct, marking steps with `<step></step>` and conclusions with `<answer></answer>`.
* **Training:** InternVL2.5-8B was fine-tuned on this data, updating only the language module while keeping the vision encoder frozen.

### 2. Process Supervision Data Generation
The framework generates step-level labels without human intervention using a Monte Carlo Tree Search (MCTS) pipeline based on OmegaPRM.
* **Seed Data:** The authors introduced **MM-K12**, a dataset of 10,000 curated multimodal math problems with verifiable answers.
* **Pipeline:** For each problem, MM-Policy generates multiple candidate solutions. The MCTS engine performs hierarchical rollouts from partial prefixes. 
* **Labeling:** Binary search is used to pinpoint the first erroneous step. A step is considered correct if its downstream completions frequently lead to the correct final answer. This process generated over 700,000 step-level annotations.

### 3. Process Reward Model (PRM) Training
MM-PRM is trained as a classifier to evaluate the correctness of each reasoning step.
* **Labeling Strategy:** Instead of binary "hard" labels, the model uses **soft labels**, taking the continuous Monte Carlo (MC) scores as targets to preserve uncertainty and reduce noise.
* **Model Design:** A special marker token $\sigma$ (instantiated as `<prm>`) is interleaved after each reasoning step.
* **Formulas:**
    * **Soft Label:** For a step $x_t$ in path $x$, the target is $\hat{y}_{t}=\mathrm{MC}(x_{<t})\in[0,1]$.
    * **Predicted Probability:** The probability $p^{(i)}$ at the $i$-th occurrence of $\sigma$ is computed via softmax of the logits for "Yes" ($z_{\mathrm{Yes}}^{(i)}$) and "No" ($z_{\mathrm{No}}^{(i)}$):

$$
p^{(i)}=\frac{\exp(z_{\mathrm{Yes}}^{(i)})}{\exp(z_{\mathrm{Yes}}^{(i)})+\exp(z_{\mathrm{No}}^{(i)})}
$$

    * **Objective:** The model minimizes the cross-entropy loss:

$$
\mathcal{L}_{\mathtt{PRM}}=-\sum_{i=1}^{T}\left[\hat{y}^{(i)}\cdot\log p^{(i)}+(1-\hat{y}^{(i)})\cdot\log(1-p^{(i)})\right]
$$

## Key Quantitative Results
MM-PRM was evaluated using a Best-of-N (BoN) inference setup ($N=16$), where the PRM reranks candidate paths.

### Performance Gains (Accuracy)
| Model | MM-K12 (In-domain) | OlympiadBench | MathVista | MathVerse | MathVision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MM-Policy** $\rightarrow$ **+MM-PRM** | 33.92% $\rightarrow$ 42.80% | 15.41% $\rightarrow$ 24.00% | 62.93% $\rightarrow$ 67.60% | 42.99% $\rightarrow$ 46.27% | 21.74% $\rightarrow$ 27.11% |
| **InternVL2.5-8B** $\rightarrow$ **+MM-PRM** | 27.01% $\rightarrow$ 37.80% | 5.23% $\rightarrow$ 15.33% | 56.43% $\rightarrow$ 63.50% | 36.26% $\rightarrow$ 42.56% | 10.04% $\rightarrow$ 19.41% |
| **InternVL2.5-78B** $\rightarrow$ **+MM-PRM** | 42.24% $\rightarrow$ 48.80% | 30.98% $\rightarrow$ 34.67% | 69.48% $\rightarrow$ 73.20% | 50.18% $\rightarrow$ 54.47% | 31.50% $\rightarrow$ 33.26% |

### Training Insights
* **Soft vs. Hard Labels:** Soft labels significantly outperformed hard labels. Using the "Average" aggregator on MM-K12, soft labels achieved **43%** accuracy compared to **34.4%** for hard labels.
* **Learning Rate:** Performance peaked at a small learning rate of $4\times10^{-6}$.
* **Path Diversity:** Increasing $N$ (number of candidates) consistently improved accuracy, particularly for harder tasks like OlympiadBench.

## Limitations
1. **Model Scale:** Training was conducted only on the InternVL series with 8B parameters; the authors did not explore larger models or different model families.
2. **Data Diversity:** The seed data was limited to K-12 mathematics, meaning the PRM may lack exposure to advanced mathematical domains or diverse visual formats outside standard educational settings.
