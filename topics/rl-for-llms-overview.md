---
title: RL for LLMs — overview
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2507.04136
- arxiv:2509.16679
- aclanthology:offline-reinforcement-learning-for-llm-m
- cameronrwolfe:online-versus-offline-rl-for-llms
- arxiv:2405.19107
- en:reinforcement-learning-from-human-feedba
- arxiv:2305.18290
- arxiv:2204.05862
open_questions:
- Does RLVR genuinely *expand* the set of solvable reasoning problems beyond the base
  model’s support, or does it only increase the *sampling probability* of solutions
  already accessible to the base model? [source:arxiv:2509.16679]
- Can offline RL methods (OREO, ILQL, DRO) match online GRPO/PPO performance on frontier-scale
  ($> 100$B) reasoning tasks when provided with sufficiently diverse offline datasets,
  or is on-policy exploration fundamentally necessary for hard multi-step credit assignment?
- What is the *minimal* verifier quality threshold for V-RL / RLVR to avoid “Adversarial
  Goodharting” where the policy attacks the verifier rather than learning valid reasoning?
  [source:arxiv:2507.04136]
- How do the critique–revision gap and recursive bias amplification in Constitutional
  AI / RLAIF scale with model size—do they vanish, persist, or worsen beyond 100B
  parameters? [source:arxiv:2507.04136]
---

Reinforcement learning has become the dominant paradigm for aligning large language models with human intent and for eliciting complex reasoning capabilities, replacing static supervised objectives with dynamic reward-driven optimization. This article surveys the full landscape—from the canonical RLHF pipeline through AI feedback, direct preference methods, verifiable-reward reasoning RL, and offline policy optimization—detailing mathematical formulations, empirical trade-offs, and open tensions in the literature.

## Canonical RLHF pipeline

The foundational three-stage RLHF procedure [source:arxiv:2204.05862][source:en:reinforcement-learning-from-human-feedba][source:arxiv:2507.04136] consists of: (1) **Supervised Fine-Tuning (SFT)** on human-written demonstrations $\mathcal{D}_{\text{SFT}}$ to establish a coherent baseline policy $\pi_{\text{SFT}}$; (2) **Reward Model (RM) training** on pairwise preference data $\mathcal{D}_{\text{pref}} = \{(x, y_w, y_l)\}$ using a Bradley–Terry–Luce likelihood  

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\text{pref}}}\Bigl[\log\sigma\bigl(r_\phi(x,y_w)-r_\phi(x,y_l)\bigr)\Bigr]
$$

where $r_\phi$ is a scalar reward head typically initialized from $\pi_{\text{SFT}}$; (3) **PPO optimization** of the policy $\pi_\theta$ against the frozen RM with a KL penalty toward the reference policy $\pi_{\text{ref}} = \pi_{\text{SFT}}$:  

$$
\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{PPO}}(\theta) - \beta\,\mathcal{D}_{\text{KL}}\bigl(\pi_\theta(\cdot|x)\,\|\,\pi_{\text{ref}}(\cdot|x)\bigr)
$$

with the clipped surrogate  

$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t\Bigl[\min\bigl(r_t(\theta)\hat{A}_t,\;\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\bigr)\Bigr],\qquad r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}.
$$

Anthropic’s large-scale study [source:arxiv:2204.05862] found that RLHF improves helpfulness and harmlessness simultaneously for models $\ge 13$B parameters, with a roughly linear relationship between $\sqrt{\mathcal{D}_{\text{KL}}}$ and PM reward during training, but observed **overfitting beyond $\sim 150$k training samples** (train/test PM scores diverge) and an **alignment tax** on standard NLP benchmarks for smaller models. The same work notes that harmlessness data collected by having crowdworkers *select the more harmful response* biases the RM toward simple refusals rather than nuanced “hostage-negotiation” behaviors.

## RLAIF and Constitutional AI

RLAIF replaces human annotators with an LLM judge (e.g., GPT-4, a specialized classifier, or a “constitution”-guided critic) to generate preference labels for RM training [source:arxiv:2507.04136][source:arxiv:2509.16679]. Constitutional AI [source:arxiv:2507.04136] structures this in two phases: (SL-CA) the model self-critiques and revises its outputs against explicit principles, then fine-tunes on the revisions; (RL-CA) the model generates response pairs and a feedback model—conditioned on the constitution—assigns preferences for PPO optimization. **Stated limitations** [source:arxiv:2507.04136]: (i) a **critique–revision gap** where discriminative ability exceeds generative corrective capacity, especially below $\approx 50$B parameters; (ii) **performative alignment** (parroting safety language without semantic understanding); (iii) **over-refusal gap between interpretability of explicit rules and flexibility**; (iv) **recursive alignment risk**—if the AI evaluator has systematic bias $\delta(x)$, its flaws amplify through the loop, producing sycophancy loops and model collapse. The survey notes that RLAIF “may not fully capture nuanced human values” and can propagate biases from the evaluator’s own training data.

## Direct Preference Optimization and offline PO variants

DPO [source:arxiv:2305.18290] eliminates the explicit RM and PPO stages by reparameterizing the optimal KL-constrained policy  

$$
\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\exp\bigl(\tfrac{1}{\beta}r(x,y)\bigr)
$$

to obtain an implicit reward $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$. Substituting into the Bradley–Terry model cancels $Z(x)$, yielding the **DPO loss**  

$$
\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\text{pref}}}\Bigl[\log\sigma\Bigl(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)\Bigr].
$$

DPO matches or exceeds PPO on summarization (TL;DR: DPO 61% win rate vs. PPO 57% at temp 0.0), dialogue (Anthropic HH: only efficient method improving over preferred completions), and OOD generalization (CNN/DM: DPO win rate 0.36 vs. PPO 0.26) [source:arxiv:2305.18290]. **Limitations** [source:arxiv:2507.04136][source:arxiv:2305.18290]: (i) less effective at exploring diverse outputs; (ii) quality highly dependent on preference data accuracy; (iii) **susceptible to length bias and unbounded implicit reward explosion** on noisy labels (“catastrophic overfitting”); (iv) **open gap in robustness to label noise** compared to PPO; (v) scaling to frontier model sizes ($> 100$B) not yet validated in the original work.

**DRO (Direct Reward Optimisation)** [source:arxiv:2405.19107] extends offline PO to *single-trajectory* data $(x,y,r)$ with scalar rewards (thumbs-up/down, 1/0), learning both a policy $\pi_\theta$ and a value function $V_\phi$ by minimizing  

$$
\mathcal{L}_{\text{DRO}}(\pi,V)=\tfrac12\mathbb{E}_{x\sim\rho,\,y\sim\mu(\cdot|x)}\Bigl[\bigl(r(x,y)-V(x)-\tau\log\tfrac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\bigr)^2\Bigr].
$$

On UltraFeedback with T5-L (770M) and T5-XL (3B), DRO-V beats KTO by **63.4%** and **57.5%** win-rate margins respectively vs. SFT, and shows stable performance across an order of magnitude of learning rates. Ablation reveals **separate policy/value networks with multiple value heads** significantly outperforms weight-sharing (Double Net Multiple Values 76.6% vs. Single Net 55.5% win rate vs. SFT) [source:arxiv:2405.19107].

**UNA (Unified Alignment)** [source:arxiv:2507.04136] generalizes PPO, DPO, KTO under a supervised regression objective  

$$
\mathcal{L}_{\text{UNA}}(\theta)=\mathbb{E}_{(x,y,z)\sim\mathcal{D}}\Bigl[\ell\Bigl(\beta\log\tfrac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)},\,z\Bigr)\Bigr],
$$

where $\ell$ is MSE (scalar scores), BCE (binary labels), or logistic margin (pairwise). **Stated limitations**: “exploration deficit” (strictly supervised loss struggles to generate novel high-utility reasoning paths), “functional ceiling” (discourages super-human strategies exceeding target scores), and “support mismatch” (if $\pi_{\text{ref}}$ assigns negligible mass to a valid high-scoring response, implicit reward explodes).

| Method | Data requirement | RL stage | Key hyperparameters | Memory footprint (models) |
|--------|------------------|----------|---------------------|---------------------------|
| RLHF (PPO) | Pairwise prefs | Online (on-policy) | $\beta,\epsilon,\gamma$ | 4 ($\pi,\pi_{\text{ref}},r_\phi,V_\psi$) |
| RLAIF | AI-generated prefs | Online (on-policy) | Same as RLHF | 4 |
| DPO | Pairwise prefs | Offline (RL-free) | $\beta$ | 2 ($\pi,\pi_{\text{ref}}$) |
| DRO | Single-traj $(x,y,r)$ | Offline | $\tau$ | 2 ($\pi,V$) + value net |
| KTO | Binary $(x,y,\text{desirable?})$ | Offline | $\beta$ | 2 |
| UNA | Unified (scalar/binary/pair) | Offline | $\beta$ | 2 |

## RLVR: Outcome-based and process-supervised RL for reasoning

RLVR replaces learned reward models with **verifiable rewards**—automated ground-truth checks (unit tests, proof verifiers, symbolic executors)—enabling RL on reasoning tasks without human preference annotation [source:arxiv:2507.04136][source:arxiv:2509.16679].

**Outcome-Based RL (OB-RL)** [source:arxiv:2507.04136] uses a sparse terminal reward $R(\tau)\in\{0,1\}$:  

$$
\mathcal{L}_{\text{OB-RL}}(\theta)=-\mathbb{E}_{(x,\tau)\sim\pi_\theta}\bigl[R(\tau)\log\pi_\theta(\tau|x)\bigr].
$$

**Limitations**: high-variance gradients, acute credit assignment (valid steps wasted if final answer wrong), “vanishing gradient regime” early in training, and **causal validity blindness** (lucky guesses/shortcuts rewarded as rigorous proofs) [source:arxiv:2507.04136].

**Chain-of-Thought Reward Optimization (CoT-RO)** [source:arxiv:2507.04136] provides dense step-wise rewards $r_t$ from a Process Reward Model (PRM):  

$$
\mathcal{L}_{\text{CoT-RO}}(\theta)=-\mathbb{E}_{(x,\tau)\sim\pi_\theta}\Bigl[\sum_{t=1}^T\gamma^{t-1}r_t\log\pi_\theta(y_t|x,y_{<t})\Bigr].
$$

**Limitations**: PRM training often requires expensive step-level human annotation; “bias of proxy rewards” (confident tone $\neq$ logical validity) integrates over trajectory length causing “reasoning hacking”; “oversight scalability gap” for distinguishing necessary deductions from plausible hallucinations [source:arxiv:2507.04136].

**Verifier-Guided RL (V-RL)** [source:arxiv:2507.04136] uses an external verifier $V_\phi$ (learned, heuristic, or tool) as reward:  

$$
\mathcal{L}_{\text{V-RL}}(\theta)=-\mathbb{E}_{(x,\tau)\sim\pi_\theta}\bigl[V_\phi(x,\tau)\log\pi_\theta(\tau|x)\bigr].
$$

**Limitations**: “verification–generation asymmetry”—policy optimization can degrade into adversarial attack on $V_\phi$ (“Adversarial Goodharting”); overly strict verifier stifles exploration; compute doubles (every sample scored by $V_\phi$); scalable oversight for subjective tasks remains open [source:arxiv:2507.04136].

**GRPO (Group Relative Policy Optimization)** [source:arxiv:2507.04136][source:arxiv:2509.16679] removes the critic by normalizing rewards within a group of $G$ samples per prompt:  

$$
\hat{A}_i^{\text{GR}}=\frac{r(a_i)-\mu}{\sigma},\qquad \mathcal{L}^{\text{GRPO}}(\theta)=\mathbb{E}_{(s,\{a_i\})\sim\pi_{\theta_{\text{old}}}}\Bigl[\frac1G\sum_{i=1}^G\min\bigl(r_i(\theta)\hat{A}_i^{\text{GR}},\text{clip}(r_i(\theta),1-\epsilon,1+\epsilon)\hat{A}_i^{\text{GR}}\bigr)\Bigr].
$$

Reduces memory to **2 model copies** ($\pi,\pi_{\text{ref}}$) vs. PPO’s 4, but converts memory bottleneck into **compute bottleneck** (requires $G\ge 64$ for stable $\sigma$ estimation) and suffers **singularity in homogeneous reward regimes** ($\sigma\to 0$ when all rewards identical) [source:arxiv:2507.04136][source:cameronrwolfe:online-versus-offline-rl-for-llms].

**Empirical RLVR results** [source:arxiv:2509.16679] (Table 1): DeepSeek-R1-Zero gains +31.8 AIME2024, +14.2 GPQA-Diamond, +13.8 LiveCodeBench, +5.7 MATH-500 over base; DeepSeek-R1 (with SFT+RL) adds +40.6 AIME, +29.7 LiveCodeBench; OpenAI-o1-1217 reaches +70.2 AIME, +25.8 GPQA, +30.5 LiveCodeBench. **Critical debate** [source:arxiv:2509.16679]: “It remains debated whether RLVR truly expands LLM reasoning capabilities beyond pre-training or merely amplifies high-reward outputs already present in the base model’s distribution. Some studies suggest RLVR models primarily improve sampling efficiency of correct reasoning paths, with all generated paths existing in the base model’s sampling distribution, implying inherent limitations by the base model.”

## Offline RL for multi-step reasoning

Offline RL methods learn from static datasets without on-policy rollouts, addressing the sample-efficiency and infrastructure cost of online PPO/GRPO.

**OREO (Offline Reasoning Optimization)** [source:aclanthology:offline-reinforcement-learning-for-llm-m] builds on Path Consistency Learning (PCL), enforcing the telescoped soft Bellman equation  

$$
V_\phi(s_t)=R_t-\beta\sum_{i=t}^{T-1}\log\frac{\pi_\theta(a_i|s_i)}{\pi_{\text{ref}}(a_i|s_i)},\qquad R_t=\sum_{i=t}^{T-1}r(s_i,a_i),
$$

with value loss $\mathcal{L}_V(\phi)=\frac1T\sum_t\bigl(V_\phi(s_t)-R_t+\beta\sum_{i=t}^{T-1}\log\frac{\pi_\theta(a_i|s_i)}{\pi_{\text{ref}}(a_i|s_i)}\bigr)^2$ and policy loss derived from the same consistency condition. Three granularities: **token-level** (standard), **step-level** (action = reasoning step), **response-level** (mimics DPO at $s_0$). On Qwen-2.5-Math 1.5B: OREO achieves **5.2% relative gain on GSM8K**, **10.5% on MATH** (52.5% absolute) vs. SFT; on DeepSeekMath 7B: +3.6% GSM8K, +5.1% MATH. Iterative OREO (re-sampling with updated policy) shows steady gains over 3 iterations where baselines saturate. Learned $V_\phi$ enables test-time beam search: $B=7$ gives **+11.4% GSM8K**, **+17.9% MATH** relative improvement [source:aclanthology:offline-reinforcement-learning-for-llm-m].

**ILQL (Implicit Language Q-Learning)** and **VerifierQ** [source:arxiv:2507.04136] apply offline Q-learning with Conservative Q-Learning (CQL) regularization:  

$$
\mathcal{L}_{\text{ILQL}}=\mathcal{L}_{\text{TD}}+\alpha\mathcal{L}_{\text{conservatism}},\qquad \mathcal{L}_{\text{VerifierQ}}=\mathcal{L}_{\text{Bellman}}+\beta\mathcal{L}_{\text{CQL}},
$$

$$
\mathcal{L}_{\text{CQL}}=\mathcal{L}_{\text{Bellman}}+\alpha\Bigl(\mathbb{E}_{(s,a)\sim\mu}[Q_\theta(s,a)]-\mathbb{E}_{(s,a)\sim\mathcal{D}}[Q_\theta(s,a)]\Bigr).
$$

**Stated limitations** [source:arxiv:2507.04136]: intrinsically limited by offline dataset support; overestimation bias from Bellman max yields “delusional value estimation” for OOD tokens; learned value upper-bounded by static data quality; better for “alignment constraints” (suppressing bad behaviors) than complex reasoning requiring on-policy exploration.

## Online vs. offline: the performance–efficiency trade-off

Wolfe [source:cameronrwolfe:online-versus-offline-rl-for-llms] frames the central tension: PPO-based RLHF is **online** (requires on-policy rollouts from current policy), incurring orchestration complexity, stability issues, and **4-model memory footprint** ($\pi,\pi_{\text{ref}},r_\phi,V_\psi\approx 4N$ params). REINFORCE and GRPO reduce this; GRPO without RM (common in RLVR) stores **only 2 copies** ($\pi,\pi_{\text{ref}}$). Offline alternatives—**Rejection Sampling / Best-of-N**, **SuperHF**, **ReST**, **RWR**, **RAFT**, **DPO**, **DRO**, **KTO**, **OREO**—avoid on-policy generation but **“tend to come at a cost in performance compared to online RL, creating an online-offline performance gap.”** [source:cameronrwolfe:online-versus-offline-rl-for-llms]. The same source notes LIMA aligned a model with **only 1K curated SFT examples**, while Tulu-3 uses **~1M SFT examples** and Llama-2 used **four rounds of rejection sampling before RLHF**. **Disagreement**: [source:arxiv:2507.04136] states DPO “may be less effective at exploring diverse outputs compared to traditional RL” and has an “open gap in robustness to label noise,” whereas [source:arxiv:2305.18290] reports DPO *dominates* PPO on reward–KL frontier and shows greater robustness to sampling temperature. [source:arxiv:2509.16679] notes GRPO “can suffer from low data utilization (models struggle with hard samples) and text bias (models ignore images and rely only on text),” a failure mode not highlighted in the GRPO-centric [source:arxiv:2507.04136].

## Current status and trajectory

| Technique | Trajectory | Evidence grounding |
|-----------|------------|-------------------|
| **RLHF (PPO)** | **Default but fading for frontier reasoning** | Still the canonical alignment pipeline [source:arxiv:2204.05862][source:en:reinforcement-learning-from-human-feedba]; however, 4-model memory bottleneck [source:arxiv:2507.04136], instability at long horizons [source:arxiv:2509.16679], and rise of RLVR/GRPO for reasoning tasks suggest declining centrality for *capability* work (remains standard for *safety/alignment* tuning). |
| **RLAIF / Constitutional AI** | **Rising for scalable oversight** | Adopted by major labs (Anthropic, Google) for reducing human annotation cost [source:arxiv:2507.04136]; recursive bias amplification and critique–revision gap remain unresolved [source:arxiv:2507.04136]. |
| **DPO / Offline PO (DRO, KTO, UNA)** | **Default for preference tuning; rising for reasoning via iterative variants** | DPO is widely implemented (HF TRL, Axolotl) and matches PPO on many benchmarks [source:arxiv:2305.18290]; DRO shows strong single-trajectory results on T5-scale [source:arxiv:2405.19107]; UNA provides unification but “exploration deficit” limits reasoning use [source:arxiv:2507.04136]. Iterative DPO (e.g., SPIN, not in sources) not widely reported in these surveys. |
| **RLVR (OB-RL, CoT-RO, V-RL, GRPO)** | **Rapidly rising / dominant for reasoning** | DeepSeek-R1, OpenAI-o1, Gemini 2.5-Pro all use verifiable-reward RL [source:arxiv:2509.16679]; GRPO adopted in open-source (DeepSeek, Qwen) for memory efficiency [source:arxiv:2507.04136][source:cameronrwolfe:online-versus-offline-rl-for-llms]. **Core debate unresolved**: whether RLVR expands reasoning *capability* or merely *sampling efficiency* of pre-existing paths [source:arxiv:2509.16679]. |
| **Offline RL for reasoning (OREO, ILQL, VerifierQ)** | **Niche but promising for data-constrained settings** | OREO shows strong gains on math/agent tasks at 1.5B–7B scale [source:aclanthology:offline-reinforcement-learning-for-llm-m]; ILQL/VerifierQ limited by offline support and overestimation bias [source:arxiv:2507.04136]. Not widely reported at frontier scale ($> 100$B). |
| **Rejection Sampling / Best-of-N / Iterative SFT** | **Default data-curation primitive** | Used in Llama-2 (4 rounds) [source:cameronrwolfe:online-versus-offline-rl-for-llms], Tulu-3, and as pre-RL filter; not a standalone *policy optimization* method but a critical pipeline component. |

**Hedges**: “Not widely reported” at frontier scale for OREO/ILQL/VerifierQ; DPO scaling to $> 100$B “exciting future direction” per original authors [source:arxiv:2305.18290]; RLVR’s *capability expansion* vs. *sampling efficiency* “remains debated” [source:arxiv:2509.16679]; GRPO’s text-bias and low-utilization issues “not widely reported” in other sources.

## Key takeaways

- **RLHF (PPO)** remains the *reference* alignment pipeline but is memory-heavy (4 models), unstable at long horizons, and increasingly supplanted by RLVR for reasoning tasks.
- **RLAIF / Constitutional AI** trades human cost for recursive bias risk; critique–revision gap limits effectiveness below $\sim 50$B params.
- **DPO** eliminates RM+PPO with a simple offline loss, matching PPO on many benchmarks but with **open robustness gaps** on noisy labels and length bias; DRO extends to single-trajectory data with learned value function.
- **RLVR (verifiable rewards)** is the **dominant paradigm for reasoning**: OB-RL (sparse), CoT-RO (dense/PRM), V-RL (external verifier), GRPO (critic-free). **Fundamental question unsettled**: does it *expand* reasoning capability or only *amplify* existing high-reward paths?
- **GRPO** reduces memory to 2 models but requires large group sizes ($G\ge 64$), converting memory pressure into compute pressure; fails catastrophically when group rewards are homogeneous ($\sigma\to 0$).
- **Offline RL (OREO, ILQL, VerifierQ)** enables learning from static multi-step datasets with credit assignment via value functions; strong at small scale, **not validated at frontier scale**.
- **Online–offline performance gap** persists: on-policy methods (PPO, GRPO) still lead on hardest reasoning, but offline methods (DPO, DRO, OREO) close the gap for alignment and some reasoning tasks with far lower infrastructure cost.

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [KL regularization in RLHF](kl-regularization.md)
- [MDP formulation of LLM generation](mdp-formulation.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [DPO variants deep-dive](dpo-variants.md)
- [RLAIF (RL from AI feedback)](rlaif.md)
- [Rejection sampling and Best-of-N](rejection-sampling-and-bon.md)
- [Nash and game-theoretic preference optimization](nash-and-game-theoretic-po.md)
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Process vs outcome reward models](process-vs-outcome-rewards.md)
- [Reward hacking in RLHF](reward-hacking.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md)
- [Length and format bias](length-and-format-bias.md)
- [The alignment tax](alignment-tax.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md)
- [LLM-as-judge](llm-as-judge.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [Judging bias and contamination](judging-bias-and-contamination.md)
- [Distributed RL training for LLMs](distributed-rl-training.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Agentic and tool-use RL](agentic-and-tool-use-rl.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)

## References
- [source:arxiv:2507.04136] [A Technical Survey of Reinforcement Learning Techniques for Large Language Models](https://arxiv.org/html/2507.04136v1)
- [source:arxiv:2509.16679] [Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle](https://arxiv.org/html/2509.16679v1)
- [source:aclanthology:offline-reinforcement-learning-for-llm-m] [Offline Reinforcement Learning for LLM Multi-step Reasoning](https://aclanthology.org/2025.findings-acl.464.pdf)
- [source:cameronrwolfe:online-versus-offline-rl-for-llms] [Online versus Offline RL for LLMs](https://cameronrwolfe.substack.com/p/online-rl)
- [source:arxiv:2405.19107] [Offline Regularised Reinforcement Learning for Large Language Models Alignment](https://arxiv.org/html/2405.19107v1)
- [source:en:reinforcement-learning-from-human-feedba] [Reinforcement learning from human feedback](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2204.05862] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
