---
id: arxiv:2405.15245
type: paper
title: 'TDPO: Harnessing Token-level Reward Guidance for Enhancing Direct Preference
  Optimization'
url: https://arxiv.org/abs/2405.15245
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

# Co-Trojan: Cooperative Backdoor Attack in Decentralized Reinforcement Learning

## Core Problem
In decentralized reinforcement learning (RL), agents improve their policies by sharing and aggregating knowledge. This openness introduces a security vulnerability: malicious agents can share poisoned policies to inject backdoors into benign agents. Traditional Single Backdoor Policy Attacks (SBPA) are often easily detected because a policy containing a full backdoor deviates significantly from a benign optimal policy. The core problem addressed is how to inject a functional backdoor into a benign agent's policy in a covert manner that evades detection.

## Method: Co-Trojan
The proposed method, **Co-Trojan**, utilizes a cooperative strategy where a global backdoor is decomposed into multiple fragments. Instead of one agent sharing a complete backdoor, multiple malicious agents each share a partial component. The full backdoor is only assembled once a benign agent aggregates all the poisoned policies.

### Step-by-Step Recipe
1. **Safe Subspace Identification**: The adversary identifies the optimal policy $\pi^*$ and calculates the state covariance matrix $\Sigma = \mathbb{E}_{s \sim d_{M}^{\pi^{*}}}[ss^{T}]$. Through eigen-decomposition, the state space is divided into a "safe subspace" $E$ (the top $d$ eigenspace) and its orthogonal complement $E^\perp$.
2. **Trigger Decomposition**: The global trigger function $f$ is decomposed into $N$ uncorrelated local trigger functions $\{f_i\}_{i=1}^N$. Each local trigger $f_i$ is designed to operate within a specific subspace $E_i^\perp \subset E^\perp$, where $E_i^\perp = \text{span}(\{u_{i}\}_{i=t_{i-1}+1}^{t_{i}})$.
3. **Local Poisoned Training**: Each malicious agent $i$ trains a local policy $\pi_i^\dagger$ that incorporates only its specific trigger fragment $f_i$. Because each policy only contains a fragment of the backdoor, it remains closer to the benign policy and is harder to detect.
4. **Knowledge Sharing and Aggregation**: Malicious agents share their fragmented policies with benign agents. Benign agents aggregate these policies using a policy-based sharing method:

$$
\pi(a|s) = \frac{1}{N} \sum_{i=1}^N \pi_i(a|s)
$$

5. **Backdoor Assembly**: Once the benign agent aggregates the fragmented policies, the global backdoor policy $\pi_g^\dagger$ is assembled, effectively approximating the target global backdoor policy $\pi^\dagger$.

## Key Formulas
The local training objective for each agent is to maximize the cumulative reward:

$$
\pi_i^* = \arg \max_{\pi_i} \mathbb{E}_{A \sim \pi_i(\cdot|s_t)} \left[ \sum_{t=0}^{\infty} \gamma^t R_i(s_t, \pi_i(s_t)) \right]
$$

The global trigger function $f$ is the average of local triggers:

$$
f = \frac{1}{N} \sum_{i=1}^N f_i
$$

The resulting aggregated backdoor policy $\pi_g^\dagger$ is defined as:

$$
\pi_g^\dagger = \frac{1}{N} \sum_{i=1}^N \pi_i(s_t + f_i(s_{0:t}))
$$

The theoretical correctness is supported by the proof that the aggregated policy approximates the target policy $\pi(s_t + f(s_{0:t}))$. Given an $L$-Lipschitz smooth policy and a bounded expectation $B$ for triggered states, the error is bounded by:

$$
\left\| \pi(s_t + f(s_{0:t})) - \frac{1}{N} \sum_{i=1}^{N} \pi(s_t + f_i(s_{0:t})) \right\| \leq 2LB
$$

## Key Quantitative Results
The method was evaluated using the Parallel Advantage Actor-Critic (PAAC) implementation on two Atari environments: **Breakout** and **Seaquest**.

*   **Effectiveness**: In both environments, agents using Co-Trojan exhibited significantly lower scores in triggered environments compared to clean environments, confirming the successful injection of the backdoor.
*   **Accuracy**: The attack performance of Co-Trojan was found to be comparable to the standard (non-cooperative) TrojDRL attack, demonstrating that decomposing the trigger does not sacrifice the attack's impact.
*   **Robustness**: The attack remained effective across three different poisoning conditions: **strong targeted poison**, **weak targeted poison**, and **untargeted poison**.

## Stated Limitations
The authors note that this research focuses exclusively on the implementation and theoretical guarantee of the attack. They state that the development of robust defense mechanisms against such decentralized cooperative backdoor attacks is a subject for future work.
