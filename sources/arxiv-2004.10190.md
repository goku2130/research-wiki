---
id: arxiv:2004.10190
type: paper
title: 'Never Stop Learning: The Effectiveness of Fine-Tuning in Robotic Reinforcement
  Learning'
url: https://arxiv.org/abs/2004.10190
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

# Summary: Never Stop Learning: The Effectiveness of Fine-Tuning in Robotic Reinforcement Learning

### Core Problem
Most robotic learning systems are deployed as fixed policies and are not adapted after deployment. While fine-tuning is common in computer vision and NLP, there is limited research on adapting motor skills themselves. The authors investigate whether a vision-based robotic manipulation policy can be efficiently adapted to "off-distribution" variations—such as changes in background, lighting, object appearance, and robot morphology—using minimal new data.

### Method
The authors propose a simple offline fine-tuning procedure using off-policy reinforcement learning (RL). The process follows these steps:

1.  **Pre-training**: A "base policy" is trained using the QT-Opt algorithm. This involves an offline stage using 580,000 real grasp attempts across 1,000 diverse objects, followed by an online stage of 28,000 grasp attempts.
2.  **Data Collection**: The pre-trained policy is used to collect an exploration dataset of grasp attempts on a specific target (challenge) task.
3.  **Initialization**: The off-policy RL algorithm is initialized with the parameters of the pre-trained base policy.
4.  **Dataset Integration**: Both the base task dataset (saved from pre-training) and the new target task dataset are used as data sources (replay buffers).
5.  **Policy Update**: The policy is updated for 500,000 gradient steps. Training examples are sampled with equal probability from both the base and target datasets.
6.  **Hyperparameter Adjustment**: A reduced learning rate of $10^{-4}$ is used, which represents 25% of the learning rate used during pre-training.
7.  **Evaluation**: The resulting fine-tuned policy is evaluated on the target task.

### Key Quantitative Results
The base policy achieved an 86% success rate on the baseline grasping task. To test robustness, the authors introduced five "Challenge Tasks" that significantly degraded the base policy's performance:

| Challenge Task | Base Policy Success | $\Delta$ (vs. Baseline) | Fine-Tuned Success (Best) | $\Delta$ (Improvement) |
| :--- | :---: | :---: | :---: | :---: |
| Checkerboard Backing | 50% | $-36\%$ | 90% | $+40\%$ |
| Harsh Lighting | 31% | $-55\%$ | 63% | $+31\%$ |
| Extend Gripper 1cm | 76% | $-10\%$ | 93% | $+18\%$ |
| Offset Gripper 10cm | 47% | $-39\%$ | 98% | $+55\%$ |
| Transparent Bottles | 49% | $-37\%$ | 66% | $+17\%$ |

**Key Findings:**
*   **Sample Efficiency**: Fine-tuning required less than 0.2% of the data necessary to learn the task from scratch. Some tasks (e.g., gripper offset) showed substantial gains with as few as 25 exploration grasps.
*   **Comparison**: The method consistently outperformed training from "Scratch" (0% to 37% success) and initializing with "ImageNet" features (0% to 47% success).
*   **Continual Learning**: In a lineage of repeated fine-tuning across multiple tasks, the performance penalty compared to single-step fine-tuning was minimal (typically between 4% and 7% lower), while the "Transparent Bottles" task actually improved by 8%.

### Stated Limitations
The authors identify a significant drawback of offline fine-tuning: the lack of a built-in evaluation step to determine when to stop training. Because the method operates in a low-data regime, repeatedly updating the network leads to **overfitting**. 

Empirical evidence showed that after approximately 500,000 gradient steps, performance on the target task (e.g., Offset Gripper 10cm) dropped precipitously, eventually falling below the performance of the original base policy. The authors note that the point at which this overfitting begins is unstable and not easily predictable, complicating the deployment of pure offline fine-tuning.
