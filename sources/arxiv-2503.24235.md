---
id: arxiv:2503.24235
type: paper
title: 'A Survey on Test-Time Scaling in Large Language Models: What, How, Where,
  and How Well?'
url: https://arxiv.org/abs/2503.24235
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

# Summary: A Survey on Test-Time Scaling in Large Language Models

## Core Problem
As the efficacy of scaling pre-training data and parameters diminishes, research has shifted toward **Test-Time Scaling (TTS)**—also known as "test-time computing." The core problem is how to effectively allocate additional computation during the inference phase to elicit higher-order reasoning and problem-solving capabilities in Large Language Models (LLMs). Despite a surge in techniques (e.g., OpenAI's o1, DeepSeek-R1), the field lacked a unified, systemic framework to categorize methods, compare their effectiveness, and identify consistent scaling laws.

## The TTS Framework (Methodology)
The authors propose a hierarchical, four-axis taxonomy to organize TTS research:

### 1. What to Scale (Scaling Paradigms)
*   **Parallel Scaling:** Generating multiple candidate responses in parallel and aggregating them.

$$
\text{Set of solutions } S = \{s_{m,i} \mid m \le M, i \le k_m\}, \text{ Final answer } \hat{s} = A(s_{1,1}, \dots, s_{M,k_M})
$$

*   **Sequential Scaling:** Iteratively updating intermediate states $n_t$ based on the problem context $p$.

$$
n_{t+1} = R(n_t, p)
$$

*   **Hybrid Scaling:** Combining parallel expansion ($E$) and sequential filtering ($S$).

$$
F_{t+1} = S(E(s)) \text{ for } s \in F_t
$$

*   **Internal Scaling:** Training the model to autonomously determine computation allocation via internal parameters.

$$
z_{t+1} = f_\theta(z_t), \text{ stop}(z_t) = \pi_\theta(z_t)
$$

### 2. How to Scale (Implementation)
*   **Tuning-based:** 
    *   *Supervised Fine-Tuning (SFT):* Includes imitation of long Chain-of-Thought (CoT) traces, distillation from stronger models, and SFT warmup.
    *   *Reinforcement Learning (RL):* Divided into reward model-free (e.g., rule-based rewards in DeepSeek R1) and reward model-based (e.g., PPO, GRPO).
*   **Inference-based:** 
    *   *Stimulation:* Using prompts (e.g., "think step by step"), decoding strategies (e.g., filler tokens), latent strategies (hidden space reasoning), or mixture-of-model strategies.
    *   *Verification:* Outcome verification (scoring final answers) and Process verification (step-level Process Reward Models or PRMs).
    *   *Search:* Utilizing Tree-of-Thoughts, MCTS, or graph search.
    *   *Aggregation:* Selection (e.g., Majority Voting, Best-of-N) or Fusion (merging multiple samples).

### 3. Where to Scale (Application Domains)
*   **Reasoning-intensive:** Mathematics, Programming/Code, Game Playing, and Scientific Reasoning.
*   **Agentic Tasks:** Scaling via design choice (multi-agent), environment feedback, or simulated environments.
*   **Others:** General benchmarks, knowledge-intensive tasks, and evaluation tasks (LLM-as-a-judge).

### 4. How Well to Scale (Evaluation)
*   **Performance:** $\text{Pass@1}$ (first attempt) and $\text{Pass@k}$ (coverage).

$$
\text{P a s s@k}=\frac{1}{n}\sum_{i=1}^{n}\left(1-\frac{\binom{N-C_{i}}{k}}{\binom{N}{k}}\right)
$$

*   **Efficiency:** Measured by token cost, FLOPs, and KV cache size. Reasoning efficiency $\eta$ is defined as:

$$
\eta(\mathcal{M})=\mathbb{E}_{\mathcal{T}\sim p(\mathcal{T})}\left[\frac{Q(\mathcal{M},\mathcal{D})}{\mathcal{C}(\mathcal{M},\mathcal{D})}\right]
$$

*   **Reasoning Quality:** Underthinking score ($\xi_{UT}$) and Process efficiency ($\xi_P$).

$$
\xi_{\mathrm{U T}}=\frac{1}{N_{\mathrm{i n c}}}\sum_{i=1}^{N_{\mathrm{i n c}}}\left(1-\frac{\hat{T}_{i}}{T_{i}}\right), \quad \xi_P = \frac{1}{N} \sum_{i=1}^N \frac{D_i}{T_i}
$$

*   **Controllability:** Adherence to compute budgets via the Control metric:

$$
\text{Control} = \frac{1}{|A|} \sum_{a \in A} I(a_{\min} \le a \le a_{\max})
$$

*   **Scalability:** The average slope of performance gains relative to compute.

$$
\text{S c a l i n g}=\frac{1}{\binom{|\mathcal{A}|}{2}}\sum_{a,b\in{\mathcal{A}}\atop b>a}\frac{f(b)-f(a)}{b-a}
$$

## Key Quantitative Results
*   **Distillation:** A 32B model trained on curated samples from a top-tier reasoner solved competition-level math problems nearly as well as the teacher model.
*   **KV Cache Efficiency:** ETS achieves up to $1.8\times$ KV cache reduction compared to REBASE, resulting in $1.4\times$ faster inference on NVIDIA H100 GPUs.
*   **Controllability:** In $k-\epsilon$ controllability tests, over 97% of targets were reachable with a prompt length $\le 10$ tokens and an error tolerance $\epsilon \le 0.05$.

## Stated Limitations
*   **Parallel Scaling:** Suffers from diminishing returns once solution coverage reaches saturation.
*   **Sequential Scaling:** Prone to error accumulation and challenges in maintaining coherence over long horizons.
*   **Internal Scaling:** Lacks interpretability and controllability; risks "logical drift" or hallucination without external guidance.
*   **General:** The authors note there is "no free lunch," as the optimal scaling strategy depends heavily on the hardness and openness of the specific problem.
