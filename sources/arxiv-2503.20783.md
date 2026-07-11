---
id: arxiv:2503.20783
type: paper
title: 'Understanding R1-Zero-Like Training: A Critical Perspective'
url: https://arxiv.org/abs/2503.20783
retrieved: '2026-07-10'
maturity: comprehensive
topic: rl-for-reasoning
---

**Core Problem**
R1-Zero-like training applies reinforcement learning (RL) directly to base large language models (LLMs) without supervised fine-tuning (SFT), demonstrating that reasoning capabilities can scale with response length. However, this paradigm's effectiveness is confounded by two hidden factors: pretraining biases in base models and optimization biases in RL algorithms. The authors find that widely used base models like Qwen2.5 already exhibit chat-like behavior and self-reflection patterns without prompt templates, effectively functioning as SFT models and blurring the attribution of performance gains to pure RL. Additionally, the Group Relative Policy Optimization (GRPO) algorithm introduces unintended length and difficulty biases that artificially inflate response lengths, particularly for incorrect outputs. This phenomenon is frequently misinterpreted as emergent long chain-of-thought reasoning, when it may instead be an artifact of the optimization objective.

**Method and Recipe Step by Step**
The study first evaluates base models (Qwen2.5, Llama-3.1, DeepSeek-V3-Base) across mathematical benchmarks using three prompt configurations (R1 template, Qwen-Math template, and no template) to measure their exploratory policy capabilities and answering rates. Next, the authors mathematically decompose the GRPO objective to isolate the sources of bias. To correct this, they propose Dr. GRPO, which removes the response-length division and question-level standard deviation normalization from the loss function. Practically, this requires replacing the dynamic `mask.sum(axis=dim)` term with a constant generation budget in the loss normalization step, recovering an unbiased Monte Carlo advantage estimate. Finally, the authors implement a minimalist R1-Zero recipe: online RL-tuning of the Qwen2.5-Math-7B model using Dr. GRPO on MATH dataset levels 3–5 questions, guided by the Qwen-Math template, and evaluated on standard mathematical benchmarks.

**Key Formulas**
The training objective is framed as an entropy-regularized policy optimization:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=1}^T r_t - \beta \log \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{ref}}(a_t|s_t)} \right]
$$

where $r_t$ is the trajectory return and $\pi_{\text{ref}}$ is a reference policy. Standard PPO optimizes a surrogate objective:

$$
J_{\text{PPO}}(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta_{\text{old}}}} \left[ \sum_{t=1}^T \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)} \hat{A}_t, \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]
$$

GRPO modifies this by introducing two biases in its loss formulation:

$$
\mathcal{L}_{\text{GRPO}} \propto \frac{r_i - \mu_i}{\sigma_i} \cdot \frac{1}{|\tau_i|}
$$

where $|\tau_i|$ is the response length and $\sigma_i$ is the question-level standard deviation of returns. Dr. GRPO eliminates the $1/|\tau_i|$ and $\sigma_i$ terms to restore the unbiased PPO objective.

**Key Quantitative Results**
Removing templates drastically improves Qwen2.5-Math-7B performance, achieving a 38.2% average accuracy across five benchmarks, compared to 23.8% with 4-shot prompting. The minimalist R1-Zero recipe achieves state-of-the-art accuracy on AIME 2024 using only a 7B base model and hours of A100 GPU compute. Dr. GRPO successfully curbs the artificial length inflation seen in vanilla GRPO, substantially reducing the token count of incorrect responses while maintaining reasoning performance. Domain-specific continual pretraining on Llama-3.2-3B significantly raises the RL performance ceiling compared to vanilla Llama, and training on simpler GSM-8K questions with the Qwen-Math template nearly doubles test accuracy on harder benchmarks.

**Stated Limitations**
The authors note that base models like Qwen2.5 are inherently SFT-like due to pretraining on concatenated question-answer pairs, making it difficult to isolate pure RL gains. Self-reflection behaviors ("Aha moments") are not positively correlated with higher post-RL accuracy. The GRPO length bias can create a misperception that long chain-of-thought reasoning emerges naturally from RL scaling. Additionally, the minimalist recipe's success depends heavily on base model-template alignment; mismatched combinations require broader question coverage to achieve comparable performance. The study is primarily focused on mathematical reasoning, and the generalizability of these optimization insights to other domains remains unverified.
