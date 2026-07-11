---
title: RL for reasoning models
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2501.12948
- arxiv:2506.14245
- arxiv:2412.14135
- arxiv:2604.18639
- arxiv:2406.06592
- arxiv:2605.05226
- arxiv:2502.10867
- arxiv:2604.17312
open_questions:
- How can the "overthinking" phenomenon (token inefficiency on simple tasks) be mitigated
  without degrading performance on complex tasks? [source:arxiv:2501.12948]
- Can internalized process supervision (IOP) scale to models significantly larger
  than 32B parameters without becoming unstable? [source:arxiv:2605.05226]
- Is there a formal theoretical guarantee for the generalization of RLVR-trained models
  to unseen distributions, beyond empirical observation? [source:arxiv:2506.14245]
---

Reinforcement learning (RL) for reasoning models shifts the training paradigm from next-token prediction to the optimization of deliberative, multi-step trajectories. This approach leverages verifiable rewards and compute-scaling—both at training and inference—to move beyond the "intelligence upper bound" imposed by static human demonstrations [source:arxiv:2502.10867].

## Core Training Recipes and Pipelines

Modern reasoning models typically employ a multi-stage pipeline to transition from base language capabilities to autonomous "System 2" reasoning.

### Multi-Stage RL Pipelines
The DeepSeek-R1 framework utilizes a four-stage process: (1) cold-start SFT with human-aligned CoT data; (2) an initial RL stage using Group Relative Policy Optimization (GRPO) with language consistency rewards; (3) rejection sampling and SFT on a mix of reasoning and general data; and (4) a final RL stage combining rule-based rewards for verifiable tasks and model-based rewards for safety and helpfulness [source:arxiv:2501.12948]. 

Alternatively, the "EasyRL" approach focuses on data efficiency via a self-evolving trajectory: (1) knowledge transfer using a small set of easy labeled samples; (2) a divide-and-conquer strategy that pseudo-labels difficult unlabeled data based on output uncertainty; and (3) difficulty-progressive self-training that iteratively exposes the model to harder samples [source:arxiv:2604.18639].

### The Scaling Roadmap
A generalized roadmap for reproducing o1-style capabilities emphasizes the integration of four components:
1. **Policy Initialization:** Injecting reasoning behaviors (e.g., self-correction, task decomposition) via SFT [source:arxiv:2412.14135].
2. **Reward Design:** Moving from sparse outcome rewards to dense process rewards [source:arxiv:2412.14135].
3. **Search:** Using MCTS or Beam Search to generate high-quality trajectories during both training and inference [source:arxiv:2412.14135].
4. **Learning:** Updating the policy via policy gradients or behavior cloning based on search-generated data [source:arxiv:2412.14135].

## Reward Mechanisms and Credit Assignment

A central tension in reasoning RL is the trade-off between the simplicity of outcome-based rewards and the precision of process-based rewards.

### Outcome-Based and Verifiable Rewards (RLVR)
Verifiable rewards provide binary feedback based on the correctness of the final answer [source:arxiv:2506.14245]. DeepSeek-R1 employs rule-based rewards for accuracy and format to incentivize reasoning without human labeling [source:arxiv:2501.12948]. 

There is a significant theoretical disagreement regarding the effect of RL with Verifiable Rewards (RLVR). Some argue that RLVR merely improves sampling efficiency by reweighting paths already present in the base model; however, empirical evidence using the `CoT-Pass@K` metric suggests that RLVR fundamentally extends the reasoning boundary, generating higher-quality trajectories that are not present in the base model's sampling distribution [source:arxiv:2506.14245].

### Process-Based Supervision (PRMs)
Process Reward Models (PRMs) provide granular, step-level feedback to mitigate the "credit assignment" problem in long-chain reasoning [source:arxiv:2406.06592].
* **Automated Generation:** OmegaPRM uses a divide-and-conquer MCTS algorithm and binary search to locate the first incorrect step in a trajectory, reducing error localization complexity from $O(kM)$ to $O(k \log M)$ [source:arxiv:2406.06592].
* **Value Iteration:** PRMs can be optimized via the Bellman equation to predict cumulative rewards:
  $$V_{\theta}(s) = r(s) + \gamma \max_{a} V_{\theta}(a + s)$$
  where $V_{\theta}(s)$ is the state value and $r(s)$ is the immediate reward [source:arxiv:2502.10867].

### Internalizing Supervision (IOP)
The IOP framework attempts to convert sparse outcome feedback into token-level signals without external PRMs. It treats the model as both a policy generator $\pi_\theta$ and a repair mode $\rho_\theta$. By comparing a failed trajectory $y_i$ with a repaired version $\bar{y}_i$ (derived from a correct reference anchor), it creates a bilateral token-level difference mask. Policy updates are then gated to restrict gradients to these "active" tokens [source:arxiv:2605.05226].

## Optimization and Search Algorithms

### Group Relative Policy Optimization (GRPO)
GRPO is a dominant algorithm for reasoning models, removing the need for a separate value function (critic) by using group-relative advantages. The objective is defined as:

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right) \right]
$$

The advantage $A_i$ is computed via group normalization: $A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$ [source:arxiv:2501.12948][source:arxiv:2502.10867].

### Search and Inference Scaling
Reasoning is modeled as a Markov Decision Process (MDP) where the state $s_t$ is the question and previous reasoning steps [source:arxiv:2502.10867]. To scale performance, models utilize:
* **Test-Time Compute:** Non-autoregressive decoding (Beam Search, MCTS) guided by a PRM [source:arxiv:2502.10867].
* **Reward Shaping:** Transforming sparse rewards into dense ones using potential functions: $F(s_t, a_t) = r(s_t, a_t) + \gamma\phi(s_{t+1}) - \phi(s_t)$ [source:arxiv:2412.14135].

## Current status and trajectory

Techniques for RL-based reasoning are currently **rising and becoming the default** for frontier reasoning models. The transition from pure SFT to RL-driven "Native CoT" is well-documented, with models like DeepSeek-R1 and o1 demonstrating that RL can unlock superhuman performance in mathematics and coding [source:arxiv:2501.12948][source:arxiv:2502.10867]. 

However, the trajectory is bifurcated: 
1. **Verifiable Domains:** RLVR and GRPO are highly effective and widely adopted in math and code [source:arxiv:2604.18639].
2. **Non-Verifiable Domains:** The application of these techniques to creative writing or open-ended research is **not widely reported** and remains a major challenge due to the lack of rule-based rewards, leading to high risks of reward hacking [source:arxiv:2501.12948][source:arxiv:2604.17312].

## Key takeaways
* **Boundary Extension:** RLVR does not just re-weight paths; it can expand the model's inherent reasoning capabilities [source:arxiv:2506.14245].
* **Efficiency Gains:** Automated process supervision (OmegaPRM) can be 75x more efficient than brute-force Monte Carlo for generating PRM data [source:arxiv:2406.06592].
* **GRPO Advantage:** GRPO reduces computational overhead by replacing the critic model with group-based normalization [source:arxiv:2501.12948].
* **Bottleneck:** The primary limitation across all frameworks is the reliance on verifiable ground truth, making the methodology difficult to export to subjective tasks [source:arxiv:2604.18639][source:arxiv:2604.17312].

## Related topics
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Reward modeling for LLMs](reward-modeling.md)

## References
- [source:arxiv:2501.12948] [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [source:arxiv:2506.14245] [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245)
- [source:arxiv:2412.14135] [Scaling of Search and Learning: A Roadmap to Reproduce o1 from Reinforcement Learning Perspective](https://arxiv.org/abs/2412.14135)
- [source:arxiv:2604.18639] [Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement Learning](https://arxiv.org/abs/2604.18639)
- [source:arxiv:2406.06592] [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)
- [source:arxiv:2605.05226] [Internalizing Outcome Supervision into Process Supervision: A New Paradigm for Reinforcement Learning for Reasoning](https://arxiv.org/abs/2605.05226)
- [source:arxiv:2502.10867] [A Tutorial on LLM Reasoning: Relevant Methods behind ChatGPT o1](https://arxiv.org/abs/2502.10867)
- [source:arxiv:2604.17312] [A Survey of Reinforcement Learning for Large Language Models Under Data Scarcity Challenges and Solutions](https://arxiv.org/abs/2604.17312)
