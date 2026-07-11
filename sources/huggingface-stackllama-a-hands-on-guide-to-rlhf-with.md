---
id: huggingface:stackllama-a-hands-on-guide-to-rlhf-with
type: web
title: 'StackLLaMA: A hands-on guide to RLHF with LLaMA (Hugging Face)'
url: https://huggingface.co/blog/stackllama
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

# StackLLaMA: RLHF for Stack Exchange Question Answering

StackLLaMA is a project demonstrating the alignment of a LLaMA 7B model to answer questions on Stack Exchange using Reinforcement Learning from Human Feedback (RLHF). The core problem addressed is the alignment of a capable base model with specific human preferences and behavioral expectations for a technical Q&A domain.

### Method and Training Pipeline

The training process follows a three-stage pipeline: Supervised Fine-Tuning (SFT), Reward Modeling (RM), and Reinforcement Learning from Human Feedback (RLHF).

#### 1. Data Preparation and Efficiency Strategies
To avoid the high memory costs of training a 7B parameter model (which would require $\sim 70\text{GB}$ in bf16), the authors employed **Parameter-Efficient Fine-Tuning (PEFT)** using **Low-Rank Adaptation (LoRA)** and **8-bit quantization**. This reduces the memory footprint to approximately 1.2–1.4GB per billion parameters.

The StackExchange dataset was processed by assigning a score to answers based on the following logic:

$$
\text{score} = \text{round}(\log_2(1 + \text{upvotes})) + (1 \text{ if accepted else } 0)
$$

(A score of $-1$ is assigned if upvotes are negative). To create the preference dataset, the authors sampled up to ten answer pairs per question and converted HTML formatting to Markdown.

#### 2. Supervised Fine-Tuning (SFT)
The model was first adapted to the domain using a causal language modeling objective. To maximize efficiency, the authors used **packing**, where multiple texts are concatenated with EOS tokens to fill the model's context window, eliminating padding tokens. The LoRA configuration for this stage used $r=16$, $\text{lora\_alpha}=32$, and $\text{lora\_dropout}=0.05$.

#### 3. Reward Modeling (RM)
A reward model was trained to predict which of two candidate answers ($y_j, y_k$) for a prompt $x$ a human would prefer. The reward model was trained using a ranking loss:

$$
\text{loss}(\theta)=- E_{\left(x, y_j, y_k\right) \sim D}\left[\log \left(\sigma\left(r_\theta\left(x, y_j\right)-r_\theta\left(x, y_k\right)\right)\right)\right]
$$

where $r_\theta$ is the model's score and $y_j$ is the preferred candidate. This stage utilized a subset of 100,000 pairs for training and 50,000 for evaluation, with a LoRA configuration of $r=8$, $\text{lora\_alpha}=32$, and $\text{lora\_dropout}=0.1$.

#### 4. RLHF via PPO
The final stage used Proximal Policy Optimization (PPO) in a loop: generating responses, rating them with the reward model, and optimizing the policy. To prevent the model from exploiting the reward model (reward hacking) by generating gibberish, a KL-divergence penalty was added:

$$
\text{R}(x, y)=\text{r}(x, y)- \beta \text{KL}(x, y)
$$

The SFT model served as a frozen 8-bit base, while only the policy's LoRA weights were optimized.

### Key Quantitative Results
*   **Reward Model Accuracy:** The reward model achieved a final accuracy of **67%**.
*   **Compute Resources:** The RLHF stage was trained for **20 hours** on **3x8 A100-80GB GPUs**.
*   **Memory Footprint:** Loading the 7B model in 8-bit reduced the weight memory to **7GB**.

### Limitations and Challenges
The authors identified several instabilities inherent in RLHF:
*   **Reward Hacking:** The PPO algorithm occasionally exploited imperfections in the reward model, such as generating repetitive code blocks (`` ` ``) because the reward model associated code blocks with higher-ranked Stack Exchange answers.
*   **Negative KL Divergence:** While KL divergence is theoretically positive, the estimate used in the `trl` library can become negative due to sampling strategies (e.g., suppressing EOS tokens or padding), leading to training instabilities.
*   **Loss Spikes:** The authors noted occasional spikes in loss that contributed to overall instability.
