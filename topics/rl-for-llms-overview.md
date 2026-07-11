---
title: RL for LLMs — overview
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2305.18290
- arxiv:2405.19107
- arxiv:2509.16679
- aclanthology:offline-reinforcement-learning-for-llm-m
- en:reinforcement-learning-from-human-feedba
- arxiv:2204.05862
- arxiv:2507.04136
- cameronrwolfe:online-versus-offline-rl-for-llms
- arxiv:2203.02155
- arxiv:2212.08073
- arxiv:2309.03409
- arxiv:2402.03300
- arxiv:2310.03714
- arxiv:2404.14367
- arxiv:2403.07691
open_questions:
- Does RLVR genuinely expand the *reasoning capability frontier* of LLMs, or does
  it only improve *sampling efficiency* of reasoning paths already present in the
  base model's distribution? [source:arxiv:2509.16679]
- Can offline PO methods (DPO, DRO, ORPO, OREO) close the online-offline performance
  gap at frontier scale ($>100$B), or is on-policy exploration fundamentally necessary
  for hardest reasoning tasks?
- How do the "critique–revision gap" and "recursive bias amplification" in Constitutional
  AI / RLAIF scale with model size and constitution complexity? [source:arxiv:2212.08073][source:arxiv:2507.04136]
- What are the statistical properties of negative gradients in contrastive PO, and
  can they be regularized to improve robustness to label noise? [source:arxiv:2404.14367]
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

Anthropic's large-scale study [source:arxiv:2204.05862] found that RLHF improves helpfulness and harmlessness simultaneously for models $\ge 13$B parameters, with a roughly linear relationship between $\sqrt{\mathcal{D}_{\text{KL}}}$ and PM reward during training, but observed **overfitting beyond $\sim 150$k training samples** (train/test PM scores diverge) and an **alignment tax** on standard NLP benchmarks for smaller models. The same work notes that harmlessness data collected by having crowdworkers *select the more harmful response* biases the RM toward simple refusals rather than nuanced "hostage-negotiation" behaviors.

**InstructGPT (OpenAI, 2022)** [source:arxiv:2203.02155] established the canonical RLHF recipe at scale on GPT-3. The three-stage pipeline—**SFT on 13k demonstrations**, **RM trained on 33k pairwise comparisons** (using all $\binom{K}{2}$ pairs per ranking as a batch element), and **PPO with a KL penalty**—yielded the **PPO-ptx** objective that mixes the RL reward with a pretraining log-likelihood term to mitigate the alignment tax:

$$
\text{objective} (\phi) = \mathbb{E}_{(x, y) \sim D_{\pi_{\phi}^{\mathrm{RL}}}} \left[ r_{\theta} (x, y) - \beta \log \left(\pi_{\phi}^{\mathrm{RL}} (y \mid x) / \pi^{\mathrm{SFT}} (y \mid x)\right) \right] + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log \left(\pi_{\phi}^{\mathrm{RL}} (x)\right) \right].
$$

**Key quantitative results** [source:arxiv:2203.02155]: labelers preferred 175B InstructGPT over 175B GPT-3 **$85 \pm 3\%$** of the time and over few-shot GPT-3 **$71 \pm 4\%$**; notably, the **1.3B InstructGPT was preferred over 175B GPT-3**. On TruthfulQA, PPO models were truthful/informative **~2× as often** as GPT-3; hallucination rate on closed-domain tasks dropped from **41% → 21%**; toxicity when prompted to be respectful fell **~25%**. InstructGPT outperformed FLAN and T0 (public-instruction-tuned baselines) by **$78 \pm 4\%$** and **$79 \pm 4\%$** win rates respectively. **Stated limitations**: models still fail on false-premise instructions, "over-hedge" on simple questions, struggle with multiple explicit constraints, and can generate *more* toxic content than GPT-3 if explicitly instructed to be biased. Alignment reflects preferences of **~40 contractors (primarily English-speaking)** and API customers, not the global population.

## RLAIF and Constitutional AI

RLAIF replaces human annotators with an LLM judge (e.g., GPT-4, a specialized classifier, or a "constitution"-guided critic) to generate preference labels for RM training [source:arxiv:2507.04136][source:arxiv:2509.16679]. Constitutional AI [source:arxiv:2507.04136] structures this in two phases: (SL-CA) the model self-critiques and revises its outputs against explicit principles, then fine-tunes on the revisions; (RL-CA) the model generates response pairs and a feedback model—conditioned on the constitution—assigns preferences for PPO optimization. **Stated limitations** [source:arxiv:2507.04136]: (i) a **critique–revision gap** where discriminative ability exceeds generative corrective capacity, especially below $\approx 50$B parameters; (ii) **performative alignment** (parroting safety language without semantic understanding); (iii) **over-refusal gap between interpretability of explicit rules and flexibility**; (iv) **recursive alignment risk**—if the AI evaluator has systematic bias $\delta(x)$, its flaws amplify through the loop, producing sycophancy loops and model collapse. The survey notes that RLAIF "may not fully capture nuanced human values" and can propagate biases from the evaluator's own training data.

**Constitutional AI (Anthropic, 2022)** [source:arxiv:2212.08073] provides the full two-stage recipe: **SL-CAI** uses a helpful-only model to generate responses to 182,831 red-teaming prompts, then iteratively critiques/revises them using randomly sampled constitutional principles (e.g., "Identify specific ways in which the assistant's last response is harmful..."). The revised responses (plus 135,296 helpfulness prompts) form the SFT dataset. **RL-CAI** trains a Preference Model (PM) on **AI-generated harmlessness labels** (from a feedback model evaluating response pairs via multiple-choice with optional CoT; probabilities clamped to 40–60% to avoid overconfidence) combined with **human helpfulness labels**. The SL-CAI model is then PPO-finetuned against this PM. **Results**: RL-CAI models were preferred over prior RLHF models for harmlessness while remaining more helpful than SL-CAI; pretrained models >52B with CoT reached human-competitive binary accuracy on harmlessness discrimination. **Limitations**: over-training causes "Goodharting" (overly harsh/boilerplate refusals); critiques sometimes inaccurate though revisions still improve harmlessness; human feedback still required for helpfulness/instruction-following.

## Direct Preference Optimization and offline PO variants

DPO [source:arxiv:2305.18290] eliminates the explicit RM and PPO stages by reparameterizing the optimal KL-constrained policy  

$$
\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\exp\bigl(\tfrac{1}{\beta}r(x,y)\bigr)
$$

to obtain an implicit reward $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$. Substituting into the Bradley–Terry model cancels $Z(x)$, yielding the **DPO loss**  

$$
\mathcal{L}_{\text{DPO}}(\theta)=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}_{\text{pref}}}\Bigl[\log\sigma\Bigl(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Bigr)\Bigr].
$$

DPO matches or exceeds PPO on summarization (TL;DR: DPO 61% win rate vs. PPO 57% at temp 0.0), dialogue (Anthropic HH: only efficient method improving over preferred completions), and OOD generalization (CNN/DM: DPO win rate 0.36 vs. PPO 0.26) [source:arxiv:2305.18290]. **Limitations** [source:arxiv:2507.04136][source:arxiv:2305.18290]: (i) less effective at exploring diverse outputs; (ii) quality highly dependent on preference data accuracy; (iii) **susceptible to length bias and unbounded implicit reward explosion** on noisy labels ("catastrophic overfitting"); (iv) **open gap in robustness to label noise** compared to PPO; (v) scaling to frontier model sizes ($> 100$B) not yet validated in the original work.

**DRO (Direct Reward Optimisation)** [source:arxiv:2405.19107] extends offline PO to *single-trajectory* data $(x,y,r)$ with scalar rewards (thumbs-up/down, 1/0), learning both a policy $\pi_\theta$ and a value function $V_\phi$ by minimizing  

$$
\mathcal{L}_{\text{DRO}}(\pi,V)=\tfrac12\mathbb{E}_{x\sim\rho,\,y\sim\mu(\cdot|x)}\Bigl[\bigl(r(x,y)-V(x)-\tau\log\tfrac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\bigr)^2\Bigr].
$$

On UltraFeedback with T5-L (770M) and T5-XL (3B), DRO-V beats KTO by **63.4%** and **57.5%** win-rate margins respectively vs. SFT, and shows stable performance across an order of magnitude of learning rates. Ablation reveals **separate policy/value networks with multiple value heads** significantly outperforms weight-sharing (Double Net Multiple Values 76.6% vs. Single Net 55.5% win rate vs. SFT) [source:arxiv:2405.19107].

**UNA (Unified Alignment)** [source:arxiv:2507.04136] generalizes PPO, DPO, KTO under a supervised regression objective  

$$
\mathcal{L}_{\text{UNA}}(\theta)=\mathbb{E}_{(x,y,z)\sim\mathcal{D}}\Bigl[\ell\Bigl(\beta\log\tfrac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)},\,z\Bigr)\Bigr],
$$

where $\ell$ is MSE (scalar scores), BCE (binary labels), or logistic margin (pairwise). **Stated limitations**: "exploration deficit" (strictly supervised loss struggles to generate novel high-utility reasoning paths), "functional ceiling" (discourages super-human strategies exceeding target scores), and "support mismatch" (if $\pi_{\text{ref}}$ assigns negligible mass to a valid high-scoring response, implicit reward explodes).

**ORPO (Odds Ratio Preference Optimization)** [source:arxiv:2403.07691] is a **monolithic** alignment algorithm that integrates preference optimization directly into SFT, eliminating both the separate SFT stage and the reference model. It appends an odds-ratio penalty to the NLL loss on chosen responses:

$$
\mathcal{L}_{\text{ORPO}} = \mathbb{E}_{(x, y_w, y_l)} \left[ \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}} \right], \quad
\mathcal{L}_{\text{OR}} = -\log \sigma \left(\log \frac{\mathbf{odds}_\theta(y_w|x)}{\mathbf{odds}_\theta(y_l|x)}\right),
\quad \mathbf{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}.
$$

ORPO requires **half the forward passes** of DPO/RLHF (no frozen reference model). **Results**: Mistral-ORPO-$\beta$ (7B) achieves **12.20% AlpacaEval 2.0**, **7.32 MT-Bench**, **66.19% IFEval (loose)**; Llama-2-7B-ORPO reaches **81.26% AlpacaEval 1.0**, **9.44% AlpacaEval 2.0** (surpassing RLHF-trained Llama-2 Chat 7B/13B); Phi-2-2.7B-ORPO hits **71.80% / 6.35%**; OPT models show win rates vs. SFT up to **84%**, vs. PPO **79.4%**, vs. DPO **70.9%**. **Limitations**: not tested >7B; limited algorithm comparison; domain generalizability and internal weight dynamics understudied.

**"Is RLHF Dead?" (2024)** [source:arxiv:2404.14367] systematically compares on-policy RL vs. offline contrastive methods across three axes: **on-policy sampling**, **sample reuse** (inner steps $T$), and **negative gradients** (pushing down dispreferred responses). Using didactic bandits, synthetic LLM tasks (Min/Mode/Skew Length), and full-scale Pythia-1.4B/Mistral-7B on AlpacaFarm/UltraFeedback, they find: **on-policy sampling** significantly helps when $r^*$ and $\pi_{\text{ref}}$ are misaligned (Min/Skew Length) but not when aligned (Mode Length); **negative gradients** (DPO, IPO) consistently beat maximum-likelihood methods (Pref-FT, RWR) in misaligned regimes; **combining both** (on-policy DPO) yields best results, outperforming on-policy RL (PPO/REINFORCE) and offline contrastive methods; moderate sample reuse ($T>1$) improves sample efficiency but excessive reuse hurts exploration in non-RL methods (PPO more robust due to off-policy correction). They prove on-policy RL and contrastive methods with negative gradients are **mode-seeking (reverse KL)**, enabling aggressive probability mass relocation vs. mode-covering supervised objectives (forward KL). **Limitations**: no statistical guarantees; negative-gradient properties underexplored; coverage analysis ignores pretraining distribution; RM quality/parameterization not studied.

| Method | Data requirement | RL stage | Key hyperparameters | Memory footprint (models) |
|--------|------------------|----------|---------------------|---------------------------|
| RLHF (PPO) | Pairwise prefs | Online (on-policy) | $\beta,\epsilon,\gamma$ | 4 ($\pi,\pi_{\text{ref}},r_\phi,V_\psi$) |
| RLAIF | AI-generated prefs | Online (on-policy) | Same as RLHF | 4 |
| DPO | Pairwise prefs | Offline (RL-free) | $\beta$ | 2 ($\pi,\pi_{\text{ref}}$) |
| DRO | Single-traj $(x,y,r)$ | Offline | $\tau$ | 2 ($\pi,V$) + value net |
| KTO | Binary $(x,y,\text{desirable?})$ | Offline | $\beta$ | 2 |
| UNA | Unified (scalar/binary/pair) | Offline | $\beta$ | 2 |
| ORPO | Pairwise prefs | Offline (monolithic SFT+PO) | $\lambda$ | 1 ($\pi$ only) |

## RLVR: Outcome-based and process-supervised RL for reasoning

RLVR replaces learned reward models with **verifiable rewards**—automated ground-truth checks (unit tests, proof verifiers, symbolic executors)—enabling RL on reasoning tasks without human preference annotation [source:arxiv:2507.04136][source:arxiv:2509.16679].

**Outcome-Based RL (OB-RL)** [source:arxiv:2507.04136] uses a sparse terminal reward $R(\tau)\in\{0,1\}$:  

$$
\mathcal{L}_{\text{OB-RL}}(\theta)=-\mathbb{E}_{(x,\tau)\sim\pi_\theta}\bigl[R(\tau)\log\pi_\theta(\tau|x)\bigr].
$$

**Limitations**: high-variance gradients, acute credit assignment (valid steps wasted if final answer wrong), "vanishing gradient regime" early in training, and **causal validity blindness** (lucky guesses/shortcuts rewarded as rigorous proofs) [source:arxiv:2507.04136].

**Chain-of-Thought Reward Optimization (CoT-RO)** [source:arxiv:2507.04136] provides dense step-wise rewards $r_t$ from a Process Reward Model (PRM):  

$$
\mathcal{L}_{\text{CoT-RO}}(\theta)=-\mathbb{E}_{(x,\tau)\sim\pi_\theta}\Bigl[\sum_{t=1}^T\gamma^{t-1}r_t\log\pi_\theta(y_t|x,y_{<t})\Bigr].
$$

**Limitations**: PRM training often requires expensive step-level human annotation; "bias of proxy rewards" (confident tone $\neq$ logical validity) integrates over trajectory length causing "reasoning hacking"; "oversight scalability gap" for distinguishing necessary deductions from plausible hallucinations [source:arxiv:2507.04136].

**Verifier-Guided RL (V-RL)** [source:arxiv:2507.04136] uses an external verifier $V_\phi$ (learned, heuristic, or tool) as reward:  

$$
\mathcal{L}_{\text{V-RL}}(\theta)=-\mathbb{E}_{(x,\tau)\sim\pi_\theta}\bigl[V_\phi(x,\tau)\log\pi_\theta(\tau|x)\bigr].
$$

**Limitations**: "verification–generation asymmetry"—policy optimization can degrade into adversarial attack on $V_\phi$ ("Adversarial Goodharting"); overly strict verifier stifles exploration; compute doubles (every sample scored by $V_\phi$); scalable oversight for subjective tasks remains open [source:arxiv:2507.04136].

**GRPO (Group Relative Policy Optimization)** [source:arxiv:2507.04136][source:arxiv:2509.16679] removes the critic by normalizing rewards within a group of $G$ samples per prompt:  

$$
\hat{A}_i^{\text{GR}}=\frac{r(a_i)-\mu}{\sigma},\qquad \mathcal{L}^{\text{GRPO}}(\theta)=\mathbb{E}_{(s,\{a_i\})\sim\pi_{\theta_{\text{old}}}}\Bigl[\frac1G\sum_{i=1}^G\min\bigl(r_i(\theta)\hat{A}_i^{\text{GR}},\text{clip}(r_i(\theta),1-\epsilon,1+\epsilon)\hat{A}_i^{\text{GR}}\bigr)\Bigr].
$$

Reduces memory to **2 model copies** ($\pi,\pi_{\text{ref}}$) vs. PPO's 4, but converts memory bottleneck into **compute bottleneck** (requires $G\ge 64$ for stable $\sigma$ estimation) and suffers **singularity in homogeneous reward regimes** ($\sigma\to 0$ when all rewards identical) [source:arxiv:2507.04136][source:cameronrwolfe:online-versus-offline-rl-for-llms].

**DeepSeekMath (2024)** [source:arxiv:2402.03300] demonstrates GRPO at scale for mathematical reasoning. The pipeline: **120B-token math corpus** (iterative fastText classification + human annotation on Common Crawl) → continued pretraining from **DeepSeek-Coder-Base 7B** (500B tokens: 56% math, 20% code, 10% arXiv, 10% NL, 4% AlgebraicStack) → **SFT on 776K CoT/PoT/tool-integrated examples** → **GRPO** with group-size normalization and unbiased KL estimator:

$$
\mathbb{D}_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}] = \frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - \log\frac{\pi_{\text{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_\theta(o_{i,t}|q,o_{i,<t})} - 1.
$$

**Results**: DeepSeekMath-Base 7B achieves **36.2% MATH**, **64.2% GSM8K** (beating Minerva 540B's 33.6% MATH); DeepSeekMath-Instruct 7B reaches **46.8% MATH**, **82.9% GSM8K**; GRPO pushes to **51.7% MATH**, **88.2% GSM8K** (60.9% MATH with self-consistency@64). Code pretraining benefits both tool-use and non-tool-use math; arXiv pretraining showed no notable gain. **Weaknesses**: geometry/theorem-proving; few-shot gains limited vs. GPT-4.

**OPRO (Optimization by PROmpting)** [source:arxiv:2309.03409] uses an LLM as a black-box optimizer to search prompt space via a meta-prompt containing problem description, optimization trajectory (past prompts + scores), and meta-instructions. On **GSM8K**, OPRO-found prompt *"Take a deep breath and work on this problem step-by-step"* achieves **80.2%** (PaLM 2-L scorer) vs. **71.8%** for *"Let's think step by step"* (+8.4%). On **Big-Bench Hard**, OPRO beats human prompts by up to **50%**, improving >5% on 19/23 tasks. **Limitations**: context window restricts problem scale; bumpy loss landscapes cause local optima; optimizer LLMs hallucinate values; training accuracies 5–20% above test (overfitting) though rankings transfer.

**Empirical RLVR results** [source:arxiv:2509.16679] (Table 1): DeepSeek-R1-Zero gains **+31.8 AIME2024, +14.2 GPQA-Diamond, +13.8 LiveCodeBench, +5.7 MATH-500** over base; DeepSeek-R1 (with SFT+RL) adds **+40.6 AIME, +29.7 LiveCodeBench**; OpenAI-o1-1217 reaches **+70.2 AIME, +25.8 GPQA, +30.5 LiveCodeBench**. **Critical debate** [source:arxiv:2509.16679]: "It remains debated whether RLVR truly expands LLM reasoning capabilities beyond pre-training or merely amplifies high-reward outputs already present in the base model's distribution. Some studies suggest RLVR models primarily improve sampling efficiency of correct reasoning paths, with all generated paths existing in the base model's sampling distribution, implying inherent limitations by the base model."

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

**Stated limitations** [source:arxiv:2507.04136]: intrinsically limited by offline dataset support; overestimation bias from Bellman max yields "delusional value estimation" for OOD tokens; learned value upper-bounded by static data quality; better for "alignment constraints" (suppressing bad behaviors) than complex reasoning requiring on-policy exploration.

**DSPy** [source:arxiv:2310.03714] frames LM pipeline optimization as a compilation problem: **Signatures** (declarative I/O specs), **Modules** (parameterized components like `ChainOfThought`, `ReAct`), and **Teleprompters** (optimizers that bootstrap demonstrations/instructions via rejection sampling, hyperparameter search (TPE), or finetuning). The compiler runs three stages: candidate generation (simulate with teacher, collect successful traces), parameter optimization (select best demos/instructions), and higher-order program optimization (ensembles, control-flow changes). **Results**: GSM8K GPT-3.5: 24% (vanilla) → 44% (bootstrap) → 64.7% (bootstrap×2) → **88.3% (CoT ensemble)**; Llama2-13B: 7% → 28% → **49% (ThoughtReflection ensemble)**. HotPotQA GPT-3.5: 36.9% → 48.7% → **54.7% (ensemble)**. Compiled 770M T5-Large competitive with GPT-3.5 on expert prompt chains. **Limitations**: LMs "highly unreliable"; compiler focuses on demonstration optimization; strict typing for signatures WIP; higher-order optimization limited to simple ensembles.

## Online vs. offline: the performance–efficiency trade-off

Wolfe [source:cameronrwolfe:online-versus-offline-rl-for-llms] frames the central tension: PPO-based RLHF is **online** (requires on-policy rollouts from current policy), incurring orchestration complexity, stability issues, and **4-model memory footprint** ($\pi,\pi_{\text{ref}},r_\phi,V_\psi\approx 4N$ params). REINFORCE and GRPO reduce this; GRPO without RM (common in RLVR) stores **only 2 copies** ($\pi,\pi_{\text{ref}}$). Offline alternatives—**Rejection Sampling / Best-of-N**, **SuperHF**, **ReST**, **RWR**, **RAFT**, **DPO**, **DRO**, **KTO**, **OREO**—avoid on-policy generation but **"tend to come at a cost in performance compared to online RL, creating an online-offline performance gap."** [source:cameronrwolfe:online-versus-offline-rl-for-llms]. The same source notes LIMA aligned a model with **only 1K curated SFT examples**, while Tulu-3 uses **~1M SFT examples** and Llama-2 used **four rounds of rejection sampling before RLHF**. **Disagreement**: [source:arxiv:2507.04136] states DPO "may be less effective at exploring diverse outputs compared to traditional RL" and has an "open gap in robustness to label noise," whereas [source:arxiv:2305.18290] reports DPO *dominates* PPO on reward–KL frontier and shows greater robustness to sampling temperature. [source:arxiv:2509.16679] notes GRPO "can suffer from low data utilization (models struggle with hard samples) and text bias (models ignore images and rely only on text)," a failure mode not highlighted in the GRPO-centric [source:arxiv:2507.04136]. [source:arxiv:2404.14367] finds **on-policy DPO (combining on-policy sampling + negative gradients) outperforms both on-policy RL and offline DPO**, challenging the strict online/offline dichotomy.

## Current status and trajectory

| Technique | Trajectory | Evidence grounding |
|-----------|------------|-------------------|
| **RLHF (PPO)** | **Default but fading for frontier reasoning** | Still the canonical alignment pipeline [source:arxiv:2204.05862][source:en:reinforcement-learning-from-human-feedba]; however, 4-model memory bottleneck [source:arxiv:2507.04136], instability at long horizons [source:arxiv:2509.16679], and rise of RLVR/GRPO for reasoning tasks suggest declining centrality for *capability* work (remains standard for *safety/alignment* tuning). |
| **RLAIF / Constitutional AI** | **Rising for scalable oversight** | Adopted by major labs (Anthropic, Google) for reducing human annotation cost [source:arxiv:2507.04136]; recursive bias amplification and critique–revision gap remain unresolved [source:arxiv:2507.04136][source:arxiv:2212.08073]. |
| **DPO / Offline PO (DRO, KTO, UNA, ORPO)** | **Default for preference tuning; rising for reasoning via iterative variants** | DPO widely implemented (HF TRL, Axolotl) and matches PPO on many benchmarks [source:arxiv:2305.18290]; DRO shows strong single-trajectory results on T5-scale [source:arxiv:2405.19107]; UNA provides unification but "exploration deficit" limits reasoning use [source:arxiv:2507.04136]; ORPO eliminates reference model and SFT stage, strong at 7B [source:arxiv:2403.07691]. Iterative DPO (e.g., SPIN, not in sources) not widely reported in these surveys. |
| **RLVR (OB-RL, CoT-RO, V-RL, GRPO)** | **Rapidly rising / dominant for reasoning** | DeepSeek-R1, OpenAI-o1, Gemini 2.5-Pro all use verifiable-reward RL [source:arxiv:2509.16679]; GRPO adopted in open-source (DeepSeek, Qwen) for memory efficiency [source:arxiv:2507.04136][source:cameronrwolfe:online-versus-offline-rl-for-llms]. **Core debate unresolved**: whether RLVR expands reasoning *capability* or only *sampling efficiency* of pre-existing paths [source:arxiv:2509.16679]. DeepSeekMath demonstrates GRPO at 7B math SOTA [source:arxiv:2402.03300]; OPRO shows LLM-as-optimizer for prompt discovery [source:arxiv:2309.03409]. |
| **Offline RL for reasoning (OREO, ILQL, VerifierQ, DSPy)** | **Niche but promising for data-constrained settings** | OREO shows strong gains on math/agent tasks at 1.5B–7B scale [source:aclanthology:offline-reinforcement-learning-for-llm-m]; ILQL/VerifierQ limited by offline support and overestimation bias [source:arxiv:2507.04136]; DSPy compiles declarative pipelines with bootstrap optimization [source:arxiv:2310.03714]. Not widely reported at frontier scale ($> 100$B). |
| **Rejection Sampling / Best-of-N / Iterative SFT** | **Default data-curation primitive** | Used in Llama-2 (4 rounds) [source:cameronrwolfe:online-versus-offline-rl-for-llms], Tulu-3, and as pre-RL filter; not a standalone *policy optimization* method but a critical pipeline component. |

**Hedges**: "Not widely reported" at frontier scale for OREO/ILQL/VerifierQ/DSPy; DPO scaling to $> 100$B "exciting future direction" per original authors [source:arxiv:2305.18290]; RLVR's *capability expansion* vs. *sampling efficiency* "remains debated" [source:arxiv:2509.16679]; GRPO's text-bias and low-utilization issues "not widely reported" in other sources [source:arxiv:2509.16679]; ORPO untested >7B [source:arxiv:2403.07691]; "Is RLHF Dead?" lacks statistical guarantees and doesn't study RM quality effects [source:arxiv:2404.14367].

## Key takeaways

- **RLHF (PPO)** remains the *reference* alignment pipeline but is memory-heavy (4 models), unstable at long horizons, and increasingly supplanted by RLVR for reasoning tasks.
- **RLAIF / Constitutional AI** trades human cost for recursive bias risk; critique–revision gap limits effectiveness below $\sim 50$B params [source:arxiv:2212.08073].
- **DPO** eliminates RM+PPO with a simple offline loss, matching PPO on many benchmarks but with **open robustness gaps** on noisy labels and length bias; **DRO** extends to single-trajectory data with learned value function; **ORPO** unifies SFT+PO in one reference-model-free loss, strong at 7B scale [source:arxiv:2403.07691]; **"Is RLHF Dead?"** shows on-policy sampling + negative gradients (on-policy DPO) beats both pure online RL and pure offline contrastive methods [source:arxiv:2404.14367].
- **RLVR (verifiable rewards)** is the **dominant paradigm for reasoning**: OB-RL (sparse), CoT-RO (dense/PRM), V-RL (external verifier), GRPO (critic-free). **Fundamental question unsettled**: does it *expand* reasoning capability or only *amplify* existing high-reward paths? [source:arxiv:2509.16679]
- **GRPO** reduces memory to 2 models but requires large group sizes ($G\ge 64$), converting memory pressure into compute pressure; fails catastrophically when group rewards are homogeneous ($\sigma\to 0$) [source:arxiv:2402.03300].
- **DeepSeekMath** demonstrates that code-centric pretraining + math corpus + GRPO achieves 7B math SOTA (51.7% MATH, 88.2% GSM8K) [source:arxiv:2402.03300].
- **OPRO** shows LLMs can optimize prompts autonomously, finding "Take a deep breath..." (+8.4% GSM8K) and beating human prompts by up to 50% on BBH [source:arxiv:2309.03409].
- **Offline RL (OREO, ILQL, VerifierQ, DSPy)** enables learning from static multi-step datasets with credit assignment via value functions or compilation; strong at small scale, **not validated at frontier scale**.
- **Online–offline performance gap** persists: on-policy methods (PPO, GRPO) still lead on hardest reasoning, but offline methods (DPO, DRO, OREO, ORPO) close the gap for alignment and some reasoning tasks with far lower infrastructure cost.

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
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2405.19107] [Offline Regularised Reinforcement Learning for Large Language Models Alignment](https://arxiv.org/html/2405.19107v1)
- [source:arxiv:2509.16679] [Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle](https://arxiv.org/html/2509.16679v1)
- [source:aclanthology:offline-reinforcement-learning-for-llm-m] [Offline Reinforcement Learning for LLM Multi-step Reasoning](https://aclanthology.org/2025.findings-acl.464.pdf)
- [source:en:reinforcement-learning-from-human-feedba] [Reinforcement learning from human feedback](https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback)
- [source:arxiv:2204.05862] [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)
- [source:arxiv:2507.04136] [A Technical Survey of Reinforcement Learning Techniques for Large Language Models](https://arxiv.org/html/2507.04136v1)
- [source:cameronrwolfe:online-versus-offline-rl-for-llms] [Online versus Offline RL for LLMs](https://cameronrwolfe.substack.com/p/online-rl)
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [source:arxiv:2212.08073] [Constitutional AI: Harmlessness from AI Feedback (RLAIF)](https://arxiv.org/abs/2212.08073)
- [source:arxiv:2309.03409] [Large Language Models as Optimizers (RLVR/Reasoning context)](https://arxiv.org/abs/2309.03409)
- [source:arxiv:2402.03300] [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (RLVR/GRPO)](https://arxiv.org/abs/2402.03300)
- [source:arxiv:2310.03714] [A Survey of Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2310.03714)
- [source:arxiv:2404.14367] [Is RLHF Dead? A Survey of Offline Preference Optimization for LLMs](https://arxiv.org/abs/2404.14367)
- [source:arxiv:2403.07691] [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691)
