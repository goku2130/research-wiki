---
title: RL for reasoning models
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2506.14245
- magazine:the-state-of-reinforcement-learning-for-
- interconnects:deepseek-r1-s-recipe-to-replicate-o1-and
- github:a-survey-of-reinforcement-learning-for-l
- cameronrwolfe:demystifying-reasoning-models-by-cameron
- rlhfbook:reasoning-and-inference-time-scaling-nat
open_questions:
- What is the minimal base-model capability (pre-training scale, context length, instruction
  tuning) required for RLVR to successfully induce long-CoT reasoning from scratch
  without cold-start SFT?
- How should the verifiable-reward training set be composed (math vs. code vs. STEM,
  difficulty distribution, answer-format constraints) to maximize out-of-domain generalization
  — e.g., to free-form reasoning, agentic tasks, or non-STEM domains?
- Can the implicit CoT incentivization of GRPO be strengthened or made more sample-efficient
  by incorporating lightweight process signals (e.g., format rewards, step-level verification)
  without reintroducing reward-model over-optimization?
- What are the scaling laws for RLVR compute (RL steps × group size × model size)
  versus inference-time compute (Pass@K, majority voting, tree search), and where
  is the Pareto frontier for a given inference budget?
---

Reinforcement learning with verifiable rewards (RLVR) has emerged as the dominant paradigm for training large reasoning models (LRMs), replacing subjective human-preference reward models with objective, executable verification functions for math and code. This shift enables massive-scale RL on verifiable domains, producing models that generate long chains-of-thought (CoT) and achieve inference-time scaling — where performance improves monotonically with test-time compute.

## Foundational Framework: RLVR and Verifiable Rewards

RLVR formulates LLM generation as a Markov decision process where the reward signal $R(y)$ is a deterministic function of the final answer $a$ extracted from a response $y = (c, a)$ comprising a chain-of-thought $c$ and an answer $a$ [source:arxiv:2506.14245]. For mathematics, $R(y) = \mathbb{1}[\text{extracted\_answer}(y) = \text{ground\_truth}]$; for code, $R(y) = \mathbb{1}[\text{all\_unit\_tests\_pass}(y)]$ or a proportional score [source:rlhfbook:reasoning-and-inference-time-scaling-nat]. This replaces the learned reward model of classical RLHF with a ground-truth verifier, eliminating reward-model over-optimization and enabling "hundreds or thousands of epochs over the same few data points" [source:rlhfbook:reasoning-and-inference-time-scaling-nat].

The verifiable-reward setup creates a **logic prior**: correct CoTs are more likely to yield correct answers than incorrect CoTs, i.e.,

$$
P(\mathcal{I}_{\mathrm{Ans}}(a)=1 \mid \mathcal{I}_{\mathrm{CoT}}(c)=1) = \alpha > \beta = P(\mathcal{I}_{\mathrm{Ans}}(a)=1 \mid \mathcal{I}_{\mathrm{CoT}}(c)=0)
$$

[source:arxiv:2506.14245]. This inequality is the theoretical linchpin: even though the reward observes only answer correctness, the policy gradient receives a stronger positive signal from trajectories with correct reasoning.

## Training Recipes: From o1 to DeepSeek R1

OpenAI's o1 (2024) and o3 (2024) demonstrated that massive RL compute — o3 used **10× more training compute than o1** — combined with verifiable rewards yields dramatic gains: o1 solves **74–93%** of AIME problems vs. GPT-4o's **12%**, and o3 achieves **87.5%** on ARC-AGI (human-level 85%) and **25.2%** on FrontierMath (prior SOTA 2.0%) [source:cameronrwolfe:demystifying-reasoning-models-by-cameron]. However, OpenAI's recipe remains closed.

DeepSeek R1 (2025) published a replicable four-stage pipeline [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and]:

1. **Cold-start SFT**: Fine-tune the base model on ~"a few thousand" filtered long-CoT completions from an intermediate R1-Zero model, using few-shot prompting, reflection/verification prompts, and human annotation to enforce readable formatting (e.g., `` tags).
2. **Large-scale RL (R1-Zero style)**: Train with **GRPO** on verifiable reasoning problems "until convergence" — **1000s of RL steps** (vs. 100s for Tülu 3, ~50 for larger models) [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and]. Reward = accuracy bonus (correct answer) + format reward (ad`) + language-consistency reward (match question language).
3. **Rejection sampling for general abilities**: Generate 800k completions (600k reasoning, 200k chat), rank with reward models (including LLM-as-judge for non-verifiable prompts), and SFT the base model on the top-ranked data.
4. **Final RL for general use**: Mix verifiable-domain prompts with standard RLHF preference prompts; use multiple reward models from DeepSeek-V3 pipeline.

**Disagreement on cold-start necessity**: DeepSeek reports R1-Zero (pure RL, no cold-start) exhibits "minor usability issues" like language switching, implying cold-start SFT is essential for readability [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and]. However, [source:arxiv:2506.14245] shows that **base models without cold-start** (Qwen2.5-32B → DAPO-Qwen-32B via GRPO) still achieve significant CoT-Pass@K gains on AIME 2024/2025, suggesting cold-start may accelerate but is not strictly necessary for reasoning emergence. The survey [source:github:a-survey-of-reinforcement-learning-for-l] notes base model requirements for direct RL remain unclear.

## Algorithmic Foundations: GRPO and Policy Optimization

Group Relative Policy Optimization (GRPO) is the workhorse algorithm for RLVR [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2506.14245]. For a prompt $q$, GRPO samples a group of $G$ responses $\{y_i\}_{i=1}^G$, computes rewards $R(y_i)$, and estimates the advantage via Monte Carlo normalization:

$$
\hat{A}(y_i) = \frac{R(y_i) - \mu_{\mathbf{Y}}}{\sigma_{\mathbf{Y}}}, \quad
\mu_{\mathbf{Y}} = \frac{1}{G}\sum_{j=1}^G R(y_j), \quad
\sigma_{\mathbf{Y}} = \sqrt{\frac{1}{G}\sum_{j=1}^G (R(y_j) - \mu_{\mathbf{Y}})^2}
$$

[source:arxiv:2506.14245]. The policy gradient update is

$$
\nabla_\theta J(\theta) \approx \frac{1}{G}\sum_{i=1}^G \hat{A}(y_i) \nabla_\theta \log \pi_\theta(y_i \mid q)
$$

[source:arxiv:2506.14245]. GRPO eliminates the critic/value network of PPO, reducing memory and compute — critical for the 1000s-of-steps training regime [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and]. The survey [source:github:a-survey-of-reinforcement-learning-for-l] categorizes GRPO under "Critic-Free Algorithms" in its policy-optimization taxonomy.

**KL regularization**: Both PPO and GRPO typically add a KL penalty $\beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ to the loss to prevent deviation from the reference (SFT) model [source:magazine:the-state-of-reinforcement-learning-for-; arxiv:2506.14245]. The magnitude of $\beta$ controls the alignment tax [source:alignment-tax.md] and exploration-exploitation trade-off [source:entropy-and-exploration.md].

## Theoretical Analysis: Why Verifiable Rewards Incentivize Correct Reasoning

[source:arxiv:2506.14245] proves that under the logic prior ($\alpha > \beta$), GRPO's expected advantage favors correct CoTs:

$$
\mathbb{E}[\hat{A}(y_i) \mid \mathcal{I}_{\mathrm{CoT}}(c_i)=1] > 0, \quad
\mathbb{E}[\hat{A}(y_i) \mid \mathcal{I}_{\mathrm{CoT}}(c_i)=0] < 0
$$

(Theorem 1). In the idealized limit of large $G$ and deterministic rewards ($\alpha=1, \beta=0$), the conditional advantages converge to

$$
\mathbb{E}[\hat{A} \mid \text{correct CoT}] \to \sqrt{\frac{1-p_c}{p_c}}, \quad
\mathbb{E}[\hat{A} \mid \text{incorrect CoT}] \to -\sqrt{\frac{p_c}{1-p_c}}
$$

where $p_c = P(\mathcal{I}_{\mathrm{CoT}}=1)$ [source:arxiv:2506.14245]. This shows RLVR **implicitly upweights correct reasoning trajectories** even without process supervision.

**Disagreement on distilled models**: [source:arxiv:2506.14245] reports that for **distilled LLMs** (e.g., DeepSeek-R1-Distill-Qwen-7B) in math, RLVR primarily improves sampling efficiency (Pass@1 rises but Pass@K and CoT-Pass@K gaps vanish for large $K$), suggesting distilled models already internalize major reasoning patterns. In contrast, **base models** show persistent CoT-Pass@K gaps up to $K=1024$, indicating genuine reasoning acquisition. The survey [source:github:a-survey-of-reinforcement-learning-for-l] does not distinguish these regimes.

## Empirical Evidence: Pass@K, CoT-Pass@K, and Generalization

**Pass@K on code**: AceReason-Nemotron-7B (post-RLVR) outperforms DeepSeek-R1-Distill-Qwen-7B (pre-RLVR) on LiveCodeBench across $K$ up to 1024; Skywork-OR1-7B shows consistent gaps especially on medium/hard problems [source:arxiv:2506.14245].

**CoT-Pass@K on math**: DAPO-Qwen-32B (post-RLVR) exhibits a **significant CoT-Pass@K gap over Qwen2.5-32B (base) on AIME 2024/2025 at all $K \le 1024$**, with the gap "particularly pronounced on AIME 2025" [source:arxiv:2506.14245]. This metric requires both final answer and intermediate CoT to be correct, verified via an LLM-as-CoT-judge (DeepSeek-R1-0528-Qwen3-8B) with multi-verification ($n=3$) and majority-vote aggregation [source:arxiv:2506.14245].

**Training dynamics**: On fully optimized training questions, $P(CA)^{(q)} \to 1.0$ (fraction of correct answers), while $P(CC|CA)^{(q)}$ (fraction of correct CoTs among correct answers) reaches median **~0.7** [source:arxiv:2506.14245]. Generalization to unseen test sets (AIME 2024) appears within **first 20 training steps** [source:arxiv:2506.14245].

**SFT transfer**: SFT on CoTs generated by DAPO-Qwen-32B matches the Pass@1 of the RL model, confirming high-quality CoTs are **extractable and transferable** [source:arxiv:2506.14245].

**Limitations noted**: 
- LLM-as-CoT-judge is not infallible [source:arxiv:2506.14245].
- No formal generalization guarantee; empirical only [source:arxiv:2506.14245].
- Domain mismatch: DAPO (integer-answer math) shows no gain on Minerva (physics, free-form) [source:arxiv:2506.14245].
- Benchmark contamination risk on static benchmarks [source:arxiv:2506.14245].
- R1-Zero usability issues (language mixing) persist without cold-start [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and].
- Infrastructure details (multi-model memory, generation/verification/loss scheduling) undisclosed [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and].

## Current status and trajectory

RLVR is the **default and rising** paradigm for frontier reasoning models (o1/o3, DeepSeek R1, Kimi 1.5, Qwen 3, Nemotron, Skywork, Phi-4 Reasoning, Llama-Nemotron, MiMo, Hunyuan-TurboS, all 2024–2025) [source:rlhfbook:reasoning-and-inference-time-scaling-nat; github:a-survey-of-reinforcement-learning-for-l]. The field is converging on **GRPO + verifiable rewards + long-CoT cold-start + multi-stage RL/SFT** as the canonical recipe. However, critical details remain **not widely reported**: optimal data composition (verifiable vs. general), reward-model choices for Stage 3/4, KL schedules, and infrastructure for 1000s of RL steps [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and]. The survey [source:github:a-survey-of-reinforcement-learning-for-l] explicitly solicits missing work, indicating the literature is still rapidly expanding. Fading: pure PPO with learned reward models for reasoning; rising: critic-free methods (GRPO, DAPO), process-reward variants [source:process-vs-outcome-rewards.md], and test-time compute integration [source:test-time-and-rl-interplay.md].

## Key takeaways

- **RLVR replaces learned reward models with executable verifiers** (unit tests, answer extraction), enabling massive-scale RL on math/code without reward hacking [source:rlhfbook:reasoning-and-inference-time-scaling-nat; arxiv:2506.14245].
- **GRPO is the de facto algorithm**: critic-free, group-normalized advantages, compatible with 1000s of RL steps [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2506.14245].
- **Logic prior ($\alpha > \beta$) drives implicit CoT improvement**: Theorem 1 proves GRPO upweights correct reasoning trajectories even with outcome-only rewards [source:arxiv:2506.14245].
- **Base models acquire new reasoning** (CoT-Pass@K gaps persist to $K=1024$); distilled models mainly gain sampling efficiency (gaps vanish) [source:arxiv:2506.14245].
- **Cold-start SFT on synthetic long-CoT** improves readability and stabilizes RL but may not be strictly necessary for reasoning emergence [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2506.14245].
- **Multi-stage pipelines (RL → rejection sampling → RL)** integrate general abilities while preserving reasoning [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and].
- **Open challenges**: infrastructure for large-scale RL, optimal data mixes, generalization guarantees, domain transfer, and benchmark contamination [source:arxiv:2506.14245; interconnects:deepseek-r1-s-recipe-to-replicate-o1-and].

## Related topics

- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Length and format bias](length-and-format-bias.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [RL for LLMs — overview](rl-for-llms-overview.md)
- [RLAIF (RL from AI feedback)](rlaif.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)
- [The alignment tax](alignment-tax.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md)
- [LLM-as-judge](llm-as-judge.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [Judging bias and contamination](judging-bias-and-contamination.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)
- [Nash and game-theoretic preference optimization](nash-and-game-theoretic-po.md)
- [DPO variants deep-dive](dpo-variants.md)

## References
- [source:arxiv:2506.14245] [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs - arXiv](https://arxiv.org/html/2506.14245v2)
- [source:magazine:the-state-of-reinforcement-learning-for-] [The State of Reinforcement Learning for LLM Reasoning - Ahead of AI](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and] [DeepSeek R1's recipe to replicate o1 and the future of reasoning LMs](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1)
- [source:github:a-survey-of-reinforcement-learning-for-l] [A Survey of Reinforcement Learning for Large Reasoning Models](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)
- [source:cameronrwolfe:demystifying-reasoning-models-by-cameron] [Demystifying Reasoning Models - by Cameron R. Wolfe, Ph.D.](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models)
- [source:rlhfbook:reasoning-and-inference-time-scaling-nat] [Reasoning and Inference-Time Scaling - Nathan Lambert](https://rlhfbook.com/c/07-reasoning)
