---
id: cameronrwolfe:reinforce-easy-online-rl-for-llms
type: web
title: 'REINFORCE: Easy Online RL for LLMs'
url: https://cameronrwolfe.substack.com/p/reinforce
retrieved: '2026-07-12'
maturity: comprehensive
topic: mdp-formulation
---

# Summary: REINFORCE for Large Language Models

### Core Problem
Online Reinforcement Learning (RL) is critical for LLM alignment (RLHF) and training large reasoning models (LRMs). However, the industry-standard algorithm, Proximal Policy Optimization (PPO), is computationally expensive and complex to implement. PPO requires maintaining four separate copies of the LLM in memory, involves sensitive hyperparameter tuning, and can be unstable to orchestrate. Consequently, many practitioners opt for simpler offline or RL-free alternatives like Supervised Finetuning (SFT) or Direct Preference Optimization (DPO), potentially sacrificing the benefits of online RL.

### Method
The source proposes using simpler policy gradient algorithms—specifically **REINFORCE** and **REINFORCE Leave-One-Out (RLOO)**—which provide the benefits of online RL without the overhead of PPO's actor-critic framework.

#### RL Formulation for LLMs
While RL can be modeled as a Markov Decision Process (MDP) where each token is an action, REINFORCE and RLOO typically adopt a **bandit formulation**. In this setup, the entire generated completion is treated as a single action that receives a single outcome reward after the stop token (e.g., `<eos>`) is produced.

#### The REINFORCE Recipe
REINFORCE is an implementation of the Vanilla Policy Gradient (VPG) that reduces variance by using a baseline instead of a separate learned value model (critic). The step-by-step training process is as follows:
1. **Generation**: Generate a completion for each prompt in a batch using the current policy $\pi_\theta$.
2. **Log-Probability Tracking**: Store the log probabilities of the tokens generated in each completion.
3. **Reward Assignment**: Assign a reward $R(\tau)$ to each completion using a reward model or a deterministic verifier.
4. **Baseline Calculation**: Compute a baseline $b$ by taking the average of rewards (either as a moving average across training or the average of the current batch).
5. **Advantage Estimation**: Compute the advantage by subtracting the baseline from the reward: $R(\tau) - b$.
6. **Gradient Update**: Compute the sum of log probabilities multiplied by the advantage for each completion, then average these across the batch to form a Monte Carlo estimate for the policy update.

### Key Formulas
The general objective is to maximize expected cumulative reward while minimizing the KL divergence between the current policy and a reference policy.

The **Vanilla Policy Gradient (VPG)** is expressed as:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t | s_t) \Psi_t \right]
$$

Where $\Psi_t$ can be the total reward $R(\tau)$, reward-to-go, or an advantage function.

The **REINFORCE** policy gradient estimate for a batch of $N$ trajectories is:

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_{i,t} | s_{i,t}) \right) (R(\tau_i) - b)
$$

### Key Quantitative Results
The source does not provide specific numerical benchmarks but cites research indicating that:
* REINFORCE and RLOO achieve performance similar to PPO.
* Despite the theoretical increase in gradient variance compared to actor-critic methods, this increase does not negatively impact LLM training in practice.

### Stated Limitations
The primary theoretical limitation of REINFORCE is that it produces **higher-variance policy gradients** compared to actor-critic algorithms like PPO, which use a dedicated value function to estimate the advantage more precisely. Additionally, basic policy gradients lack inherent protection against large, unstable policy updates (trust regions).
