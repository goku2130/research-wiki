---
title: RL for reasoning models
maturity: comprehensive
updated: '2026-07-12'
sources:
- cameronrwolfe:demystifying-reasoning-models-by-cameron
- interconnects:deepseek-r1-s-recipe-to-replicate-o1-and
- rlhfbook:reasoning-and-inference-time-scaling-nat
- arxiv:2506.14245
- github:a-survey-of-reinforcement-learning-for-l
- magazine:the-state-of-reinforcement-learning-for-
- arxiv:2501.12948
- arxiv:2203.14465
- arxiv:2403.09629
- cameronrwolfe:demystifying-reasoning-models-cameron-r-
- arxiv:2505.17746
open_questions:
- What are the minimal base-model capabilities required for direct RL to succeed without
  cold-start SFT?
- How should verifiable-reward RL be balanced with general preference RL in multi-stage
  pipelines to avoid catastrophic forgetting of reasoning?
- Can the logic-prior theory ($\alpha > \beta$) be extended to non-verifiable domains
  via learned process reward models?
- What infrastructure and data-composition recipes enable stable 1000s-of-steps GRPO
  training at scale?
---

Reinforcement learning with verifiable rewards (RLVR) has emerged as the dominant paradigm for training large reasoning models (LRMs), replacing subjective human-preference reward models with objective, executable verification functions for math and code. This shift enables massive-scale RL on verifiable domains, producing models that generate long chains-of-thought (CoT) and achieve inference-time scaling — where performance improves monotonically with test-time compute.

## Foundational Framework: RLVR and Verifiable Rewards

RLVR formulates LLM generation as a Markov decision process where the reward signal $R(y)$ is a deterministic function of the final answer $a$ extracted from a response $y = (c, a)$ comprising a chain-of-thought $c$ and an answer $a$ [source:arxiv:2506.14245]. For mathematics, $R(y) = \mathbb{1}[\text{extracted\_answer}(y) = \text{ground\_truth}]$; for code, $R(y) = \mathbb{1}[\text{all\_unit\_tests\_pass}(y)]$ or a proportional score [source:rlhfbook:reasoning-and-inference-time-scaling-nat]. This replaces the learned reward model of classical RLHF with a ground-truth verifier, eliminating reward-model over-optimization and enabling "hundreds or thousands of epochs over the same few data points" [source:rlhfbook:reasoning-and-inference-time-scaling-nat].

The verifiable-reward setup creates a **logic prior**: correct CoTs are more likely to yield correct answers than incorrect CoTs, i.e.,

$$
P(\mathcal{I}_{\mathrm{Ans}}(a)=1 \mid \mathcal{I}_{\mathrm{CoT}}(c)=1) = \alpha > \beta = P(\mathcal{I}_{\mathrm{Ans}}(a)=1 \mid \mathcal{I}_{\mathrm{CoT}}(c)=0)
$$

[source:arxiv:2506.14245]. This inequality is the theoretical linchpin: even though the reward observes only answer correctness, the policy gradient receives a stronger positive signal from trajectories with correct reasoning.

### Self-Taught Reasoning Precursors: STaR and Quiet-STaR

Before RLVR became dominant, bootstrapping methods demonstrated that models could improve reasoning by learning from their own generated rationales. **STaR (Self-Taught Reasoner)** [source:arxiv:2203.14465] introduced an iterative loop: (1) generate rationales for problems using few-shot CoT prompting, (2) filter rationales that yield correct answers, (3) for failed problems, provide the ground-truth answer as a hint and ask the model to "rationalize" backward to produce a valid rationale, (4) fine-tune on the combined correct rationales, and (5) repeat. STaR frames this as approximating the policy gradient objective with expected reward $J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot | x_i)} [\mathbb{1}(\hat{y}_i = y_i)]$ and gradient $\nabla J = \sum_i \mathbb{E}[\mathbb{1}(\hat{y}_i = y_i) \nabla \log p_M(\hat{y}_i, \hat{r}_i \mid x_i)]$ [source:arxiv:2203.14465]. On GPT-J (6B), STaR with rationalization reached **72.5%** on CommonsenseQA (vs. 36.6% few-shot CoT), **89.5%** on multi-digit arithmetic (vs. 76.3% direct fine-tuning), and **10.7%** on GSM8K (vs. 3.0% few-shot). Limitations include requiring baseline few-shot performance above chance, vulnerability to spurious rationales in high-chance settings (e.g., binary tasks), and non-trivial hint design for rationalization [source:arxiv:2203.14465].

**Quiet-STaR** [source:arxiv:2403.09629] generalizes STaR to unstructured text, enabling LMs to "think" at every token position. It uses a **Think-Talk-Learn** cycle: (1) **Think** — generate $r$ parallel rationales of length $t$ at each token using learned `<startofthought>`/`<endofthought>` tokens and a diagonal attention mask to isolate counterfactual rationale paths; (2) **Talk** — a shallow MLP mixing head interpolates between base LM logits and rationale-conditioned logits via weight $w$; (3) **Learn** — REINFORCE rewards rationales that increase likelihood of future $n_{true}$ tokens (non-myopic loss). On Mistral 7B, Quiet-STaR trained on OpenWebMath improved GSM8K from **5.9% → 10.9%** and CommonsenseQA from **36.3% → 47.2%** zero-shot; with CoT majority voting (maj@8), GSM8K rose from **40.6% → 47.7%** [source:arxiv:2403.09629]. Limitations include substantial inference overhead, testing only at 7B scale, fixed rationale lengths, and no faithfulness guarantee for generated rationales [source:arxiv:2403.09629].

**Fast Quiet-STaR** [source:arxiv:2505.17746] compresses Quiet-STaR's explicit thought tokens via curriculum learning and RL internalization. A three-stage curriculum progressively reduces thought tokens (16→12→8) and ahead tokens (8→4→4). An NTP phase then uses RL (REINFORCE) to match the predictive quality of the thinking model without generating thought tokens, with reward $r_j = \mathcal{L}_{\text{Fast Quiet-STaR}} - \mathcal{L}_{\text{NTP}}$. On Mistral 7B, Fast Quiet-STaR NTP improved average reasoning accuracy by **9%** (Qwen2.5 7B: **5.7%**) at **same inference latency** as the base model; it outperformed Quiet-STaR (16 tokens) by **1.8%** at **41.3%** of its inference time, and achieved **6%** of Quiet-STaR's end-to-end generation time. With CoT maj@6 on GSM8K, accuracy rose from **43.3% → 52.4%**. Training required only **0.5M tokens** and **54 minutes on 8 H800s** [source:arxiv:2505.17746]. Limitations: evaluated primarily on math/logic; method is specific to Quiet-STaR framework [source:arxiv:2505.17746].

## Training Recipes: From o1 to DeepSeek R1

OpenAI's o1 (2024) and o3 (2024) demonstrated that massive RL compute — o3 used **10× more training compute than o1** — combined with verifiable rewards yields dramatic gains: o1 solves **74–93%** of AIME problems vs. GPT-4o's **12%**, and o3 achieves **87.5%** on ARC-AGI (human-level 85%) and **25.2%** on FrontierMath (prior SOTA 2.0%) [source:cameronrwolfe:demystifying-reasoning-models-cameron-r-; cameronrwolfe:demystifying-reasoning-models-by-cameron]. However, OpenAI's recipe remains closed.

DeepSeek R1 (2025) published a replicable four-stage pipeline [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2501.12948]:

1. **Cold-start SFT**: Fine-tune the base model on ~"a few thousand" filtered long-CoT completions from an intermediate R1-Zero model, using few-shot prompting, reflection/verification prompts, and human annotation to enforce readable formatting (e.g., `` tags).
2. **Large-scale RL (R1-Zero style)**: Train with **GRPO** on verifiable reasoning problems "until convergence" — **1000s of RL steps** (vs. 100s for Tülu 3, ~50 for larger models) [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and]. Reward = accuracy bonus (correct answer) + format reward (ad`) + language-consistency reward (match question language).
3. **Rejection sampling for general abilities**: Generate 800k completions (600k reasoning, 200k chat), rank with reward models (including LLM-as-judge for non-verifiable prompts), and SFT the base model on the top-ranked data.
4. **Final RL for general use**: Mix verifiable-domain prompts with standard RLHF preference prompts; use multiple reward models from DeepSeek-V3 pipeline.

**DeepSeek-R1-Zero (Pure RL)**: The paper also introduces **DeepSeek-R1-Zero**, trained directly from DeepSeek-V3-Base using pure RL **without any SFT cold-start** [source:arxiv:2501.12948]. Using GRPO with a rule-based reward system (accuracy reward via deterministic math/code verification + format reward for `` tags), R1-Zero achieved **15.6% → 77.9% Pass@1 on AIME 2024** (reaching **86.7%** with self-consistency) over 10,400 training steps. The model naturally developed "aha moments" (self-correction) and increased thinking time (response length) as performance improved. However, R1-Zero exhibited poor readability and language mixing, motivating the multi-stage R1 pipeline [source:arxiv:2501.12948].

**Disagreement on cold-start necessity**: DeepSeek reports R1-Zero (pure RL, no cold-start) exhibits "minor usability issues" like language switching, implying cold-start SFT is essential for readability [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2501.12948]. However, [source:arxiv:2506.14245] shows that **base models without cold-start** (Qwen2.5-32B → DAPO-Qwen-32B via GRPO) still achieve significant CoT-Pass@K gains on AIME 2024/2025, suggesting cold-start may accelerate but is not strictly necessary for reasoning emergence. [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and] notes base model requirements for direct RL remain unclear.

## Algorithmic Foundations: GRPO and Policy Optimization

Group Relative Policy Optimization (GRPO) is the workhorse algorithm for RLVR [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2506.14245; arxiv:2501.12948]. For a prompt $q$, GRPO samples a group of $G$ responses $\{y_i\}_{i=1}^G$, computes rewards $R(y_i)$, and estimates the advantage via Monte Carlo normalization:

$$
\hat{A}(y_i) = \frac{R(y_i) - \mu_{\mathbf{Y}}}{\sigma_{\mathbf{Y}}}, \quad
\mu_{\mathbf{Y}} = \frac{1}{G}\sum_{j=1}^G R(y_j), \quad
\sigma_{\mathbf{Y}} = \sqrt{\frac{1}{G}\sum_{j=1}^G (R(y_j) - \mu_{\mathbf{Y}})^2}
$$

[source:arxiv:2506.14245; arxiv:2501.12948]. The policy gradient update is

$$
\nabla_\theta J(\theta) \approx \frac{1}{G}\sum_{i=1}^G \hat{A}(y_i) \nabla_\theta \log \pi_\theta(y_i \mid q)
$$

[source:arxiv:2506.14245]. The full GRPO objective includes a KL penalty term [source:arxiv:2501.12948]:

$$
\mathcal {J} _ {G R P O} (\theta) = \mathbb {E} [ q \sim P (Q), \{o _ {i} \} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {o l d}} (O | q) ] \frac {1}{G} \sum _ {i = 1} ^ {G} \left(\min \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta_ {o l d}} (o _ {i} | q)} A _ {i}, \text{clip} \left(\frac {\pi_ {\theta} (o _ {i} | q)}{\pi_ {\theta_ {o l d}} (o _ {i} | q)}, 1 - \varepsilon , 1 + \varepsilon\right) A _ {i}\right) - \beta \mathbb {D} _ {K L} (\pi_ {\theta} | | \pi_ {r e f})\right)
$$

with KL divergence estimator:

$$
\mathbb {D} _ {K L} \left(\pi_ {\theta} | | \pi_ {r e f}\right) = \frac {\pi_ {r e f} (o _ {i} | q)}{\pi_ {\theta} (o _ {i} | q)} - \log \frac {\pi_ {r e f} (o _ {i} | q)}{\pi_ {\theta} (o _ {i} | q)} - 1
$$

[source:arxiv:2501.12948]. GRPO eliminates the critic/value network of PPO, reducing memory and compute — critical for the 1000s-of-steps training regime [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and]. GRPO is categorized as a critic-free method in policy-optimization taxonomies.

**KL regularization**: Both PPO and GRPO typically add a KL penalty $\beta \cdot \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ to the loss to prevent deviation from the reference (SFT) model [source:magazine:the-state-of-reinforcement-learning-for-; arxiv:2506.14245; arxiv:2501.12948]. The magnitude of $\beta$ controls the alignment tax [source:alignment-tax.md] and exploration-exploitation trade-off [source:entropy-and-exploration.md].

**Rule-based reward design**: DeepSeek-R1 uses a composite rule-based reward to avoid reward hacking from neural reward models [source:arxiv:2501.12948]:

$$
Reward_{\text{rule}} = Reward_{\text{acc}} + Reward_{\text{format}}
$$

with an additional language consistency reward:

$$
Reward_{\text{language}} = \frac{Num(Words_{\text{target}})}{Num(Words)}
$$

[source:arxiv:2501.12948].

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

**STaR's theoretical framing**: STaR [source:arxiv:2203.14465] derives its objective as the expected reward over the dataset $J(M, X, Y) = \sum_i \mathbb{E}_{\hat{r}_i, \hat{y}_i \sim p_M(\cdot | x_i)} [\mathbb{1}(\hat{y}_i = y_i)]$, with gradient $\nabla J = \sum_i \mathbb{E}[\mathbb{1}(\hat{y}_i = y_i) \nabla \log p_M(\hat{y}_i, \hat{r}_i | x_i)]$. STaR approximates this by greedy decoding and multiple gradient steps on filtered correct rationales, effectively performing a low-variance policy gradient update.

**Disagreement on distilled models**: [source:arxiv:2506.14245] reports that for **distilled LLMs** (e.g., DeepSeek-R1-Distill-Qwen-7B) in math, RLVR primarily improves sampling efficiency (Pass@1 rises but Pass@K and CoT-Pass@K gaps vanish for large $K$), suggesting distilled models already internalize major reasoning patterns. In contrast, **base models** show persistent CoT-Pass@K gaps up to $K=1024$, indicating genuine reasoning acquisition.

## Empirical Evidence: Pass@K, CoT-Pass@K, and Generalization

**Pass@K on code**: AceReason-Nemotron-7B (post-RLVR) outperforms DeepSeek-R1-Distill-Qwen-7B (pre-RLVR) on LiveCodeBench across $K$ up to 1024; Skywork-OR1-7B shows consistent gaps especially on medium/hard problems [source:arxiv:2506.14245].

**CoT-Pass@K on math**: DAPO-Qwen-32B (post-RLVR) exhibits a **significant CoT-Pass@K gap over Qwen2.5-32B (base) on AIME 2024/2025 at all $K \le 1024$**, with the gap "particularly pronounced on AIME 2025" [source:arxiv:2506.14245]. This metric requires both final answer and intermediate CoT to be correct, verified via an LLM-as-CoT-judge (DeepSeek-R1-0528-Qwen3-8B) with multi-verification ($n=3$) and majority-vote aggregation [source:arxiv:2506.14245].

**DeepSeek-R1 final performance** [source:arxiv:2501.12948]:
- **AIME 2024 (Pass@1): 79.8%**
- **MATH-500 (Pass@1): 97.3%**
- **Codeforces Rating: 2029**
- **General Alignment**: Final RL stage improved AlpacaEval 2.0 by **25%** and ArenaHard by **17%**
- **SFT Dataset**: ~800k samples (600k reasoning, 200k non-reasoning)

**Training dynamics**: On fully optimized training questions, $P(CA)^{(q)} \to 1.0$ (fraction of correct answers), while $P(CC|CA)^{(q)}$ (fraction of correct CoTs among correct answers) reaches median **~0.7** [source:arxiv:2506.14245]. Generalization to unseen test sets (AIME 2024) appears within **first 20 training steps** [source:arxiv:2506.14245].

**SFT transfer**: SFT on CoTs generated by DAPO-Qwen-32B matches the Pass@1 of the RL model, confirming high-quality CoTs are **extractable and transferable** [source:arxiv:2506.14245].

**Quiet-STaR / Fast Quiet-STaR results** [source:arxiv:2403.09629; arxiv:2505.17746]:
- Quiet-STaR on Mistral 7B: GSM8K **5.9% → 10.9%** (OpenWebMath), **5.9% → 8.1%** (C4); CommonsenseQA **36.3% → 47.2%** (OpenWebMath), **36.3% → 42.6%** (C4); CoT maj@8 GSM8K **40.6% → 47.7%**
- Fast Quiet-STaR NTP: **+9% avg accuracy gain on Mistral 7B, +5.7% on Qwen2.5 7B at same latency; 41.3% inference time of Quiet-STaR (16 tokens) with +1.8% accuracy; 6% end-to-end generation time; GSM8K CoT maj@6 **43.3% → 52.4%**

**Limitations noted**: 
- LLM-as-CoT-judge is not infallible [source:arxiv:2506.14245].
- No formal generalization guarantee; empirical only [source:arxiv:2506.14245].
- Domain mismatch: DAPO (integer-answer math) shows no gain on Minerva (physics, free-form) [source:arxiv:2506.14245].
- Benchmark contamination risk on static benchmarks [source:arxiv:2506.14245].
- R1-Zero usability issues (language mixing) persist without cold-start [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2501.12948].
- Infrastructure details (multi-model memory, generation/verification/loss scheduling) undisclosed [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and].
- STaR requires baseline few-shot performance above chance; fails on GPT-2 even for arithmetic [source:arxiv:2203.14465].
- Quiet-STaR: computational overhead, tested only at 7B, fixed rationale lengths, no faithfulness guarantee [source:arxiv:2403.09629].
- Fast Quiet-STaR: evaluated primarily on math/logic; specific to Quiet-STaR framework [source:arxiv:2505.17746].
- DeepSeek-R1 limitations: suboptimal tool use/structured output, occasional overthinking on simple questions, language mixing on non-English/Chinese, prompt sensitivity (zero-shot recommended), limited software engineering gains due to RL evaluation cost, reward hacking difficult to scale for non-verifiable tasks [source:arxiv:2501.12948].

## Current status and trajectory

RLVR is the **default and rising** paradigm for frontier reasoning models (o1/o3, DeepSeek R1, Kimi 1.5, Qwen 3, Nemotron, Skywork, Phi-4 Reasoning, Llama-Nemotron, MiMo, Hunyuan-TurboS, all 2024–2025) [source:rlhfbook:reasoning-and-inference-time-scaling-nat; github:a-survey-of-reinforcement-learning-for-l; cameronrwolfe:demystifying-reasoning-models-cameron-r-]. The field is converging on **GRPO + verifiable rewards + long-CoT cold-start + multi-stage RL/SFT** as the canonical recipe. However, critical details remain **not widely reported**: optimal data composition (verifiable vs. general), reward-model choices for Stage 3/4, KL schedules, and infrastructure for 1000s of RL steps [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and]. The survey [source:github:a-survey-of-reinforcement-learning-for-l] identifies future directions, indicating the literature is still rapidly expanding.

**Alternative reasoning paradigms** (STaR, Quiet-STaR, Fast Quiet-STaR) demonstrate that reasoning can also be bootstrapped from self-generated rationales without verifiable rewards, though they currently lag RLVR on competitive benchmarks. Fast Quiet-STaR's success in internalizing reasoning into standard NTP via RL suggests a path toward efficient inference-time reasoning without explicit CoT tokens [source:arxiv:2505.17746].

**OpenAI o3 series** (2024) further pushes the frontier: **ARC-AGI 87.5%** (human 85%), **Codeforces Elo 2727** (top 200 globally), **SWE-Bench Verified 71.7%**, **FrontierMath 25.2%** (prior SOTA 2.0%), with o3-mini **24% faster** than o1-mini (7.7s vs 10.16s avg) [source:cameronrwolfe:demystifying-reasoning-models-cameron-r-]. **Grok-3 Reasoning Beta** achieves **96% AIME '24** (vs o3 97%), while **Gemini 2.0 Flash Thinking** maintains 1M context but lags on verifiable tasks [source:cameronrwolfe:demystifying-reasoning-models-cameron-r-]. Fading: pure PPO with learned reward models for reasoning; rising: critic-free methods (GRPO, DAPO), process-reward variants [source:process-vs-outcome-rewards.md], test-time compute integration [source:test-time-and-rl-interplay.md], and self-taught reasoning compression [source:arxiv:2505.17746].

## Key takeaways

- **RLVR replaces learned reward models with executable verifiers** (unit tests, answer extraction), enabling massive-scale RL on math/code without reward hacking [source:rlhfbook:reasoning-and-inference-time-scaling-nat; arxiv:2506.14245].
- **GRPO is the de facto algorithm**: critic-free, group-normalized advantages, compatible with 1000s of RL steps [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2506.14245; arxiv:2501.12948].
- **Logic prior ($\alpha > \beta$) drives implicit CoT improvement**: Theorem 1 proves GRPO upweights correct reasoning trajectories even with outcome-only rewards [source:arxiv:2506.14245].
- **Base models acquire new reasoning** (CoT-Pass@K gaps persist to $K=1024$); distilled models mainly gain sampling efficiency (gaps vanish) [source:arxiv:2506.14245].
- **Cold-start SFT on synthetic long-CoT** improves readability and stabilizes RL but may not be strictly necessary for reasoning emergence [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2506.14245; arxiv:2501.12948].
- **Multi-stage pipelines (RL → rejection sampling → RL)** integrate general abilities while preserving reasoning [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2501.12948].
- **Self-taught reasoning (STaR, Quiet-STaR)** bootstraps reasoning from self-generated rationales without verifiable rewards; Fast Quiet-STaR compresses explicit thought tokens into NTP via curriculum + RL [source:arxiv:2203.14465; arxiv:2403.09629; arxiv:2505.17746].
- **Open challenges**: infrastructure for large-scale RL, optimal data mixes, generalization guarantees, domain transfer, benchmark contamination, tool use/structured output, language mixing, overthinking, and scaling verifiable rewards beyond math/code [source:arxiv:2506.14245; interconnects:deepseek-r1-s-recipe-to-replicate-o1-and; arxiv:2501.12948; arxiv:2403.09629; arxiv:2505.17746].

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
- [source:cameronrwolfe:demystifying-reasoning-models-by-cameron] [Demystifying Reasoning Models - by Cameron R. Wolfe, Ph.D.](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models)
- [source:interconnects:deepseek-r1-s-recipe-to-replicate-o1-and] [DeepSeek R1's recipe to replicate o1 and the future of reasoning LMs](https://www.interconnects.ai/p/deepseek-r1-recipe-for-o1)
- [source:rlhfbook:reasoning-and-inference-time-scaling-nat] [Reasoning and Inference-Time Scaling - Nathan Lambert](https://rlhfbook.com/c/07-reasoning)
- [source:arxiv:2506.14245] [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs - arXiv](https://arxiv.org/html/2506.14245v2)
- [source:github:a-survey-of-reinforcement-learning-for-l] [A Survey of Reinforcement Learning for Large Reasoning Models](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)
- [source:magazine:the-state-of-reinforcement-learning-for-] [The State of Reinforcement Learning for LLM Reasoning - Ahead of AI](https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training)
- [source:arxiv:2501.12948] [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [source:arxiv:2203.14465] [STaR: Self-Taught Reasoner: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- [source:arxiv:2403.09629] [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
- [source:cameronrwolfe:demystifying-reasoning-models-cameron-r-] [Demystifying Reasoning Models (Cameron R. Wolfe)](https://cameronrwolfe.substack.com/p/demystifying-reasoning-models)
- [source:arxiv:2505.17746] [Fast Quiet-STaR: Thinking Without Thought Tokens](https://arxiv.org/abs/2505.17746)
