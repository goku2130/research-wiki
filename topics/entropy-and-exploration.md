---
title: Entropy and exploration in RL fine-tuning
maturity: comprehensive
updated: '2026-07-11'
sources:
- arxiv:2604.13902
- machinelearning:entropy-preserving-reinforcement-learnin
- arxiv:2510.08141
- github:awesome-exploration-methods-in-reinforce
- emergentmind:diversity-collapse-in-rl-emergent-mind
- emergentmind:exploration-collapse-in-reinforcement-le
- arxiv:2306.03236
- arxiv:2602.12375
- arxiv:1911.09615
- arxiv:2109.08134
- arxiv:2110.05457
- arxiv:2004.10190
- arxiv:1906.02138
- arxiv:2010.04816
open_questions:
- How do global vs. episodic exploration bonuses map to LLM fine-tuning where "contexts"
  are prompts, reasoning paths, or tasks? Can multiplicative combination of token-level
  (episodic) and trajectory-level (global) bonuses improve RLVR?
- Can VBE-style first-visit optimism via ensemble value errors be adapted to LLM policy
  optimization, where "value" is estimated by a reward model or verifier?
- Does the mellowmax operator (MEMEC) offer advantages over adaptive temperature sampling
  (SCOPE-RL) for LLM exploration, particularly its non-expansion guarantee?
- The batch RL regularization unification (discount factor ↔ Bayesian prior ↔ ε-greedy)
  suggests KL regularization, entropy bonuses, and sampling strategies in LLMs may
  be mathematically equivalent transition-matrix smoothing. Can this be formalized
  for LLM policy spaces?
---

Entropy collapse and diversity loss are central failure modes in RL fine-tuning of LLMs, where policy optimization drives distributions toward deterministic, low-support solutions that degrade multi-sample performance and generalization. Modern methods address this through adaptive entropy control, perplexity-disentanglement, and diversity-preserving objectives that explicitly target the exploration–exploitation trade-off at both token and trajectory levels.

## Entropy Collapse: Mechanisms and Diagnostics

Entropy collapse manifests as a sustained decline in policy entropy $\mathcal{H}(\pi(\cdot|s)) = -\sum_{a \in \mathcal{A}} \pi(a|s) \log \pi(a|s)$ over training iterations, preventing discovery of novel solutions and stagnating policy improvement [source:emergentmind:exploration-collapse-in-reinforcement-le]. In LLM reasoning with verifiable rewards (RLVR), this occurs at two levels: **token-level entropy collapse**, where the policy becomes nearly deterministic and fails to sample alternative generation pathways, and **outcome-level mode collapse**, where the agent converges to a tiny subset of reward-supporting solutions such that $|\mathcal{M}| \ll |\mathcal{S}_{\text{reward}}|$ [source:emergentmind:exploration-collapse-in-reinforcement-le].

The collapse is driven by structural issues in the RL objective rather than hyperparameter choices. Standard expected return maximization creates an "outcome-frequency multiplier" that exerts exponential selective pressure on the most probable outcomes [source:emergentmind:exploration-collapse-in-reinforcement-le]. In GRPO-style updates, softmax dynamics cause **positive sharpening** (increasing logits of sampled tokens) and **negative squeezing** (redistributing mass away from rejected tokens), which can lead to irreversible contraction where gradients vanish once a mode's probability drops below a sampling threshold [source:emergentmind:exploration-collapse-in-reinforcement-le].

Diversity collapse, a closely related phenomenon, is mathematically rooted in **selection bias** (preferring high-base-probability trajectories) and **reinforcement bias** (amplifying those trajectories in proportion to initial likelihood) [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. In continuous policies, unimodal architectures (Gaussian or deterministic) tend to converge to a single mode under distributional shift [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. Key diagnostics include:
- **Representation rank**: Rank deficiency of Gram matrices of behavioral embeddings or vanishing determinant indicates behaviors span only a low-dimensional subspace [source:emergentmind:diversity-collapse-in-rl-emergent-mind].
- **Support shrinkage**: In discrete spaces, RL compresses incorrect trajectories with a reported compression ratio $R_- \approx 0.25\text{--}0.35$ [source:emergentmind:diversity-collapse-in-rl-emergent-mind].
- **Pass@k degradation**: Improvement in single-sample success (Pass@1) often comes at the cost of sharply reduced multi-attempt success (Pass@k) [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Policy Divergence and Meta-Learning Perspectives

In personalized meta-reinforcement learning settings where entities share state/action spaces but possess unique transition dynamics $P^i$, standard meta-learning approaches like MAML (seeking a single optimal initialization) may be sub-optimal [source:arxiv:2010.04816]. **Cluster-Adaptive Meta-Learning (CAML)** addresses this by maintaining multiple representative policy initializations (medoids) rather than a single global initialization. CAML estimates policy divergence using occupancy measures $\rho_{\pi}(s,a)=\pi(a|s)\sum_{t=1}^{T}P(S_{t}=s|\pi)$ and quantifies similarity via symmetric Jensen-Shannon divergence $D_{\mathrm{JS}}(\rho_{s}^{i},\rho_{s}^{j})$. Policies are clustered via $K$-medoids on the pairwise JS divergence distance matrix, yielding $k$ medoid policies as representative initializations. At test time, the agent selects the most appropriate medoid via a multi-armed bandit over $K$ few-shot rollouts, then fine-tunes with Vanilla Policy Gradient. In a 2D continuous gridworld with 24 entity types, CAML consistently outperformed Reptile, Joint Training, Random Initialization, and Unmatched Pretraining baselines in few-shot adaptation ($K=10$ rollouts), with only environment-matched pretraining matching or exceeding it [source:arxiv:2010.04816]. Limitations include preliminary results on a specific testbed and reliance on trajectory-derived occupancy measures for environment similarity.

## Exploration Bonuses and Intrinsic Motivation

Classical exploration methods in RL are taxonomized by whether they operate in the **collect phase** (interacting with the environment) or the **train phase** (updating the policy) [source:github:awesome-exploration-methods-in-reinforce]. For LLM fine-tuning, the most relevant categories are:

| Phase | Category | Mechanism | LLM Adaptation |
|-------|----------|-----------|----------------|
| Collect | Action Selection Perturbation | Noise/randomness in action choices | Temperature sampling, nucleus sampling |
| Collect | Action Selection Guidance | Heuristics/signals to guide choices | Process reward models, verifier-guided decoding |
| Collect | State Selection Guidance | Directing agent to novel/high-value states | Prompt selection, curriculum learning |
| Train | Count Based | Visit counts for rarely visited states | N-gram diversity penalties, trajectory deduplication |
| Train | Prediction Based | Prediction errors as intrinsic rewards (RND, ICM) | Model disagreement, uncertainty quantification |
| Train | Entropy Augmented | Entropy terms in objective (e.g., SAC) | Entropy bonuses, KL regularization |
| Train | Information Theory Based | Empowerment, mutual information | Diversity rewards, determinantal point processes |

Recent trends (2025–2026) emphasize **LLM-based exploration**: "exploration hacking" in LLMs, rubric-scaffolded RL for reasoning, and latent exploration decoding for large reasoning models [source:github:awesome-exploration-methods-in-reinforce]. Advanced intrinsic motivation now includes diffusion models for unsupervised RL, spectral Bellman methods, and graph-theoretic intrinsic rewards based on effective resistance [source:github:awesome-exploration-methods-in-reinforce].

### Global vs. Episodic Bonuses in Contextual MDPs

In Contextual Markov Decision Processes (CMDPs) where environments differ across episodes, the choice between **global novelty bonuses** (computed over entire training history) and **episodic novelty bonuses** (computed per episode) critically affects exploration efficacy [source:arxiv:2306.03236]. The authors propose a framework based on the variance of the optimal value function in representation space $\mathcal{Z}$ across contexts. Let $\psi: \mathcal{S} \to \mathcal{Z}$ be a feature extractor and $V_{\psi,c}^{\star}(z)=\inf_{s\in\psi^{-1}(z)}V^{\star}(s)$ the value function over $\mathcal{Z}$ for context $c$. The framework posits:
1. **Global bonuses fail** when $V_{\psi,c}^{\star}$ varies significantly across contexts: a region visited in one episode may be high-value later, but the agent avoids it because the global bonus is exhausted.
2. **Episodic bonuses succeed** when $V_{\psi,c}^{\star}$ varies significantly, provided the agent can cover $\mathcal{Z}$ within a single episode.
3. **Global bonuses succeed** when $V_{\psi,c}^{\star}$ is stable across contexts: visiting a high-value region once benefits all contexts.

**Bonus definitions**: Random Network Distillation (RND) global bonus $b_{\mathrm{RND}}(s_{t})=\|f(s_{t})-\bar{f}(s_{t})\|_{2}^{2}$; Elliptical Episodic Bonus (E3B) $b_{\mathrm{E3B}}(s_{t})=\phi(s_{t})^{\top}[\sum_{i=t_{0}}^{t-1}\phi(s_{i})\phi(s_{i})^{\top}+\lambda I]^{-1}\phi(s_{t})$. **Multiplicative combination** outperforms additive: $b_{\mathrm{combined}}(s_{t})=\mathbb{I}[N_{e}(\psi(s_{t}))=1]\cdot\frac{1}{\sqrt{N(\psi(s_{t}))}}$ [source:arxiv:2306.03236].

**Key results**: In `MultiRoom` with positional encodings, global bonus performance dropped from $0.99 \pm 0.00$ ($|C|=1$) to $0.00 \pm 0.00$ ($|C|=\infty$), while episodic bonus remained robust at $0.87 \pm 0.10$ ($|C|=\infty$). In pixel environments: **Habitat** favored episodic (E3B); **Montezuma's Revenge** favored global (RND). On **MiniHack** (16 tasks), multiplicative E3B+RND/NovelD achieved SOTA median/IQM performance over E3B alone [source:arxiv:2306.03236]. Limitations: multiplicative combination doesn't always match the single best bonus per task (e.g., worse than episodic alone in Habitat); only simple additive/multiplicative combinations explored, not adaptive methods.

### Value Bonuses using Ensemble Errors (VBE)

**VBE** directly estimates a value bonus $b(s,a)$ layered atop any value-based RL algorithm (e.g., Double DQN), providing **first-visit optimism** without the computational burden of Bootstrap DQN (BDQN) [source:arxiv:2602.12375]. VBE maintains an ensemble of $k$ random action-value functions (RQFs). The optimistic behavior policy selects actions via $\pi_b(s) = \arg\max_{a} q_w(s,a) + c \cdot b(s,a)$ where $b(s,a) = \max_{i \in [k]} |f_{w_i}(s,a) - f_i(s,a)|$ (max absolute ensemble error). A random reward $r_i = f_i(s,a) - \gamma f_i(s', \arg\max_{a'} q_w(s',a'))$ is constructed consistent with the target RQF $f_i$, and a randomly selected predictor RQF $f_{w_i}$ is updated via TD learning on $r_i$ per step [source:arxiv:2602.12375].

**Results**: In tabular **Deepsea**, VBE covered the entire state space at all grid sizes; BDQN failed on larger grids; ACB/RND failed entirely (lack first-visit optimism). In classic control (Sparse Mountain Car, Puddle World, River Swim, Deepsea), VBE learned faster with best final performance; ACB/RND failed in Mountain Car. On **Atari**, VBE scaled to images, matching or exceeding baselines in Breakout, Pong, Qbert, Pitfall; in Gravitar, RND initially stronger but eventually matched [source:arxiv:2602.12375]. Limitations: no convergence theory for changing behavior/target policies (though bonuses empirically converge to zero); sensitivity to ensemble size $k$ and bonus scale $c$ (larger $k,c$ help hard exploration like River Swim but hurt easier tasks like Puddle World).

### Maximum Entropy Mellowmax Episodic Control (MEMEC)

**MEMEC** integrates Episodic Control (Model-Free EC or Neural EC) with a maximum entropy mellowmax policy for principled exploration, replacing naive $\epsilon$-greedy [source:arxiv:1911.09615]. MFEC uses a fixed-size table per action storing highest returns; NEC uses a CNN for embeddings and a Differentiable Neural Dictionary (DND) for $k$-nearest neighbor $Q$-value retrieval. The mellowmax operator $\text{mm}_{\omega}(Q)=\frac{\log(\frac{1}{n}\sum_{i=1}^{n}e^{\omega Q_{i}})}{\omega}$ (non-expansion in infinity norm) and Boltzmann operator $\text{boltz}_{\beta}(Q)=\frac{\sum Q_{i}e^{\beta Q_{i}}}{\sum e^{\beta Q_{i}}}$ define the policy $\pi_{mm}(a|s)=\frac{e^{\beta Q(s,a)}}{\sum_{a} e^{\beta Q(s,a)}}$. Temperature $\beta$ is solved via root-finding (Brent's method) on $\sum_{a} e^{\beta(Q(s,a)-\text{mm}_{\omega}(Q))}(Q(s,a)-\text{mm}_{\omega}(Q))=0$ [source:arxiv:1911.09615].

**Atari results**: MFEC-MEMEC significantly outperformed $\epsilon$-greedy MFEC and D3QN: Q*Bert $11610 \pm 898$ vs $3896 \pm 710$ / $1480 \pm 271$; Ms. Pac-Man $6968 \pm 787$ vs $4178 \pm 510$ / $1851 \pm 98$; Space Invaders $1029 \pm 157$ vs $672 \pm 13$ / $756 \pm 30$; Bowling $68 \pm 7$ vs $62 \pm 8$ / $22 \pm 16$. Pong was worse ($7 \pm 4$ vs $17 \pm 2$). In Acrobot, MEMEC achieved higher rewards and faster learning. Gridworlds favored MFEC with softmax; NEC generally struggled. Limitation: $\omega$ sensitivity varies across domains ($\omega=25$ Pong, $\omega=60$ Ms. Pac-Man), requiring prohibitive grid search in high dimensions [source:arxiv:1911.09615].

### Unreliable Intrinsic Rewards in Multi-Agent RL

In MARL, intrinsic rewards face three challenges: exponential joint-action spaces, credit assignment difficulty, and partial observability reducing count reliability [source:arxiv:1906.02138]. **Centrally-Assisted Exploration (ICQL)** uses a centralized agent only during training to drive exploration, while decentralized agents (IQL) learn the final policy from a shared replay buffer. The central agent uses a COMA-style critic $q_{\psi}^{a}(s_t, \mathbf{u}_t^{-a}, u_{t-1}^{a})$ with **local maximization (lmax)**: initialize joint actions with decentralized agents' greedy actions, iteratively update each $u_t^a$ to maximize $q_{\psi}^{a}$. A $Q(\lambda)$ target $G_{t}^{\lambda} := r_{t} + (1-\lambda)\gamma \text{lmax}_{\bar{u}} q_{\psi'}^{a} + \lambda\gamma G_{t+1}^{\lambda}$ accelerates intrinsic reward transport. The **collaborative intrinsic reward** uses state-based counts $N_t(s_{t+1})$, gives all agents the same reward (max uncertainty across agents), and tracks parameter evolution via exponentially decaying average of inverted matrix $\mathbf{C}_t$. Training samples 50% episodes from central agent, 50% from decentralized agents; central agent discarded after training [source:arxiv:1906.02138].

**Results**: In a 41x10 partially observable predator-prey gridworld (4 agents, valley prey reward 5, mountain prey reward 10), ICQL learned faster than IQL and IQL+Intrinsic, and converged to optimal; IQL+Intrinsic accelerated initially but failed to converge due to distracting incentives. Hyperparameters: $\gamma=0.99$, lr $0.0005$, batch 32, $\sigma=1$, $\alpha=0.0002$, $b_t=0.01$ [source:arxiv:1906.02138]. Limitation: intrinsic rewards are "blessing and curse"; centralized training required; $N_t(s_{t+1})$ proxy is heuristic.

## Diversity Preservation Methods

Modern frameworks move beyond global entropy bonuses (which often sacrifice correctness) and pairwise novelty (vulnerable to clustering) toward principled diversity preservation [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Differential Smoothing (DS)
DS modifies rewards to target only correct trajectories: it penalizes high-base-probability correct traces while sharpening mass on incorrect ones to prevent "sharpening" onto a limited set of trajectories [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. DS-GRPO demonstrated improvements up to $+6.7\%$ in Pass@K on real-world mathematical reasoning datasets [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Mode Anchored Reward Augmentation (MARA)
MARA edits the reward landscape to ensure the KL target distribution remains flat across all high-reward modes [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. It maintained near-uniform entropy and Pareto-optimal reward/diversity in drug discovery and creative QA [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Diversity-Preserving Hybrid RL (DPH-RL)
DPH-RL replaces standard reverse-KL regularization with mass-covering $f$-divergences (forward-KL or Jensen–Shannon), which continuously reference the base policy to maintain support across initial modes [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. It successfully prevented Pass@k degradation and catastrophic forgetting in SQL and math tasks [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Diversity via Determinants (DvD)
DvD maximizes the volume of the behavioral manifold by computing the determinant of the agent Gram matrix, ensuring agents are not linearly redundant [source:emergentmind:diversity-collapse-in-rl-emergent-mind]. DvD-TD3 outperformed all baselines in Humanoid-v2 by retaining both forward and backward behaviors [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### Proximal Feature Optimization (PFO)
PFO controls feature-rank decay to maintain network plasticity and prevent representation collapse [source:emergentmind:diversity-collapse-in-rl-emergent-mind].

### REPO and ADAPO
The entropy-preserving RL framework proposes two algorithm families: **REPO**, which regulates entropy by modifying the advantage function, and **ADAPO**, which uses adaptive asymmetric clipping [source:machinelearning:entropy-preserving-reinforcement-learnin]. Models trained with these methods maintain trajectory diversity, achieve higher final performance, and retain "trainability" in sequential learning tasks [source:machinelearning:entropy-preserving-reinforcement-learnin].

### Regularization Unification in Batch RL

In batch RL, limited exploration leads to inaccurate model estimation and overfitting. Three regularization methods—**Bayesian Prior (Dirichlet)**, **Discount Regularization**, and **Planning over $\epsilon$-greedy Policies**—are unified as a weighted average of the MLE transition matrix $\hat{T}_{MLE}$ and a regularization matrix $T_{reg}$: $\hat{T}(a) = (1 - \epsilon)\hat{T}_{MLE}(a) + \epsilon T_{reg}(a)$ [source:arxiv:2109.08134]. 
- **Bayesian Prior**: $\epsilon = \frac{\sum \alpha_i}{\sum c_i + \sum \alpha_i}$, $T_{reg} = T_{prior\_mean}$.
- **Discount Regularization** (lower $\gamma_l < \gamma$): $\epsilon = \frac{\gamma - \gamma_l}{\gamma}$, $T_{reg} = T_{zeros}$.
- **$\epsilon$-greedy Planning**: $\epsilon$ is the exploration rate, $T_{reg} = \frac{1}{|\mathcal{A}|} \sum_{a'} \hat{T}_{MLE}(a')$.

**Theorem 1**: An MDP with discount $\gamma_l$ has identical optimal policy to one with discount $\gamma$ and transition $(1-\epsilon)T + \epsilon T_{unif}$ (uniform transitions). Hypotheses: uniform priors suit "dense worlds"; discount regularization balances rewards across timescales; $\epsilon$-greedy planning avoids catastrophic outcomes via stochasticity. Empirically, under uniform random data: $\epsilon$-greedy planning best in Cliff Walk (catastrophic); uniform priors effective in Interconnected Grid (dense). Data collection policy significantly shifts relative performance; with optimal-policy data, regularization ceases to help in "Two Goals" MDP. Lower policy loss doesn't consistently correspond to lower transition MSE. Data size less impactful than MDP structure and data policy [source:arxiv:2109.08134]. Limitations: uniformity assumptions restrictive; uneven exploration inconclusive; further formal characterization needed.

### Real-World Fine-Tuning and Offline Adaptation

**REDQ (Randomized Ensembled Double Q-learning)** enables sample-efficient real-world fine-tuning for legged robots [source:arxiv:2110.05457]. Policies pre-trained in simulation (motion imitation with pose/velocity rewards) are fine-tuned online using REDQ's ensemble of Q-functions $Q_{\theta} = \{Q_{\theta^k}\}_{k=1}^{N}$ with random subset target sampling to reduce overestimation. Replay buffers reset on environment shift; recovery policy (trained to stand from random orientations) enables autonomous reset. Onboard IMU/contact sensors via Kalman filter estimate state. **Results**: Fine-tuning outperformed Rapid Motor Adaptation (RMA) and latent space search in unexpected environments (rugged heightfields, low friction); REDQ superior to SAC; robot fine-tuned pacing on lawn, side-stepping on carpet/doormat/memory foam; recovery policy 100% successful, faster than manufacturer's controller on foam; ~2 hours operation for consistent improvement. Limitations: workspace drift requires human repositioning; separate fine-tuning per environment, not continuous lifelong learning [source:arxiv:2110.05457].

**Offline fine-tuning** of vision-based manipulation policies adapts to off-distribution variations (background, lighting, object appearance, morphology) with minimal data [source:arxiv:2004.10190]. Base policy (QT-Opt, 580k real grasps + 28k online) fine-tuned via off-policy RL initialized from base, using combined base+target replay buffers (equal sampling), reduced LR $10^{-4}$ (25% of pre-training), 500k gradient steps. **Results**: Base 86% success on baseline; challenge tasks showed large drops recovered by fine-tuning: Checkerboard Backing 50%→90% (+40%), Harsh Lighting 31%→63% (+31%), Extend Gripper 1cm 76%→93% (+18%), Offset Gripper 10cm 47%→98% (+55%), Transparent Bottles 49%→66% (+17%). Sample efficiency: <0.2% of scratch data; Offset Gripper gained substantially with 25 exploration grasps. Outperformed Scratch (0-37%) and ImageNet init (0-47%). Continual fine-tuning lineage penalty minimal (4-7% lower), Transparent Bottles improved 8%. **Critical limitation**: no built-in evaluation stopping criterion; overfitting after ~500k steps causes performance to drop below base policy, with unstable/unpredictable onset [source:arxiv:2004.10190].

## Temperature and Sampling Strategies

SCOPE-RL identifies a **temperature–sign asymmetry** in entropy dynamics: under high-temperature sampling, positive samples promote entropy growth while negative samples accelerate collapse; under low-temperature sampling, negative samples increase entropy but cause rapid reward collapse [source:arxiv:2510.08141]. SCOPE-RL leverages this via:

1. **Monitor entropy** $\mathcal{H}(\pi_{\theta_{\text{old}}})$ of the previous step.
2. **Adaptive temperature scaling**: $T = \text{clip}(1 + \mathcal{H}_0 - \mathcal{H}(\pi_{\theta_{\text{old}}}), 0.8, 1.2)$ where $\mathcal{H}_0$ is a target entropy threshold [source:arxiv:2510.08141].
3. **Positive sample filtering**: Draw auxiliary samples at temperature $T$, retain only those with $R=1$.
4. **Objective integration**: Augment GRPO with a regularization term from temperature-adjusted positive samples:

$$
\mathcal{J}_{\text{SCOPE}}(\theta) = \mathcal{J}_{\text{GRPO}}(\theta) + \alpha \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^{G'} \sim \pi_{\theta_{\text{old}}}^T(O|q)} \left[ \frac{1}{G'} \sum_{i=1}^{G'} \mathbf{1}[R(q, o_i) = 1] \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min(r_{i,t}(\theta), \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon)) \right]
$$

where $r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ and $\alpha$ is the auxiliary sample ratio [source:arxiv:2510.08141].

With $\alpha = 1/64$ (<2% additional samples), SCOPE-RL achieved consistent gains: Qwen2.5-Math-7B average score $51.60 \to 54.84$ (+3.23), Qwen2.5-7B $47.71 \to 49.81$ (+2.10), Qwen3-4B $67.77 \to 69.73$ (+1.96) [source:arxiv:2510.08141]. Pass@1024 on Qwen2.5-7B improved dramatically: AIME24 $73.3 \to 86.7$, AIME25 $63.3 \to 76.7$ [source:arxiv:2510.08141]. The entropy–performance relationship is non-monotonic: best average performance at $\mathcal{H}_0 = 0.50$ (54.84), outperforming $\mathcal{H}_0 = 0.25$ (53.00) and $\mathcal{H}_0 = 0.75$ (53.45) [source:arxiv:2510.08141].

DiPO introduces **Perplexity Space Disentangling (PSD)** and **Bidirectional Reward Reallocation (BRR)** for fine-grained exploration–exploitation trade-off [source:arxiv:2604.13902]. PSD partitions samples into Exploitation Space (EiS) and Exploration Space (ErS) via a dynamic threshold $\tau^*$ optimized from a cached PPL queue $\mathcal{Q}$:

$$
\tau^* = \arg \min_{\tau} \frac{1}{|\mathcal{Q}|} \sum_{(r_i, p_i) \in \mathcal{Q}} |r_i - \mathbb{I}(p_i < \tau)|
$$

with advantage judgments ensuring PPL correlates with correctness [source:arxiv:2604.13902]. BRR reallocates rewards only for zero-gradient groups: in hard groups (all 0) within EiS, the max-PPL sample gets reward 1 to encourage exploration; in easy groups (all 1) within ErS, the max-PPL sample gets reward 0 to encourage exploitation [source:arxiv:2604.13902]. The objective combines DAPO with reallocated rewards weighted by $\alpha$:

$$
\mathcal{J}_{\text{DiPO}}(\theta) = \mathcal{J}_{\text{DAPO}}(\theta, \mathcal{R}) + \alpha \times \mathcal{J}_{\text{DAPO}}(\theta, \mathcal{R}_r)
$$

Optimal $\alpha = 0.1$; DiPO showed higher robustness than entropy loss (10× $\alpha$ increase caused only 1.43 point drop vs. catastrophic collapse for entropy loss) [source:arxiv:2604.13902]. Results: Qwen3-4B-Base 50.55% vs. DAPO 49.43% and GRPO 48.92%; Qwen3-8B-Base 54.79% vs. 53.23%/53.24%; function calling on BFCLv3: Qwen2.5-3B 55.03% vs. 53.21%, Qwen2.5-7B 62.51% vs. 61.06% [source:arxiv:2604.13902].

Other algorithmic remedies for exploration collapse include:
- **Low-probability Regularization (Lp-Reg)**: Protects "reasoning sparks" via forward-KL and proxy-KL on filtered low-probability tokens; 60.17% average accuracy across five math tasks (+2.66% over standard entropy controls) [source:emergentmind:exploration-collapse-in-reinforcement-le].
- **Inverse Probability Scaling (IPS)**: Weights learning signal by $1/p$ to remove self-reinforcing gradient, converging to reward-proportional stationary distribution; improved recovery rates and coverage in molecule design [source:emergentmind:exploration-collapse-in-reinforcement-le].
- **Unified Entropy Control (UEC-RL)**: High-temperature exploration for difficult prompts + replay-based entropy consolidation; 37.9% relative improvement over GRPO on Geometry3K [source:emergentmind:exploration-collapse-in-reinforcement-le].
- **Dual-Scale Diversity Regularization (DSDR)**: Couples global and local diversity via reward shaping on correct diverse trajectories and token-level entropy; outperformed on Pass@1 and Pass@k with widening gap at higher $k$ [source:emergentmind:exploration-collapse-in-reinforcement-le].
- **Anchored Policy Optimization (APO)**: Support manifold pull + elastic recovery for valid alternatives [source:emergentmind:exploration-collapse-in-reinforcement-le].

## Current Status and Trajectory

Entropy and diversity control in RL fine-tuning is a **rising, actively researched area** with multiple competing frameworks rather than a settled default. SCOPE-RL [source:arxiv:2510.08141] and DiPO [source:arxiv:2604.13902] represent 2025–2026 advances targeting GRPO/DAPO pipelines specifically, while REPO/ADAPO [source:machinelearning:entropy-preserving-reinforcement-learnin] and the diversity-preservation suite (DS, MARA, DPH-RL, DvD, PFO) [source:emergentmind:diversity-collapse-in-rl-emergent-mind] offer broader algorithmic templates. The field has not converged on a single standard: entropy bonuses remain common but are known to cause instability (oscillation between collapse and explosion) [source:arxiv:2510.08141]; asymmetric clipping helps but lacks adaptive control [source:arxiv:2510.08141]; perplexity-based disentanglement (DiPO) shows promise but lacks strict theoretical guarantees [source:arxiv:2604.13902]. The taxonomy of exploration methods continues to expand, with LLM-specific adaptations (latent exploration decoding, rubric-scaffolded RL) appearing in 2025–2026 literature [source:github:awesome-exploration-methods-in-reinforce]. **Not widely reported** is large-scale ablation comparing these methods under identical compute/data budgets; most papers compare against GRPO/DAPO baselines rather than each other. Theoretical convergence guarantees for support-preserving regularization in high-dimensional LLM policy spaces remain **not established** [source:emergentmind:exploration-collapse-in-reinforcement-le; emergentmind:diversity-collapse-in-rl-emergent-mind].

**New findings from 2019–2026 RL literature** contextualize LLM fine-tuning challenges: 
- Global vs. episodic bonus trade-offs in CMDPs [source:arxiv:2306.03236] mirror the token-level vs. trajectory-level exploration tension in LLMs, where "contexts" correspond to prompts or reasoning paths.
- VBE's first-visit optimism via ensemble errors [source:arxiv:2602.12375] offers a template for LLM exploration bonuses that don't require retroactive reward signals.
- MEMEC's mellowmax operator [source:arxiv:1911.09615] provides a theoretically grounded alternative to temperature sampling, with non-expansion guarantees.
- Batch RL regularization unification [source:arxiv:2109.08134] reveals that discount regularization, uniform priors, and $\epsilon$-greedy planning are mathematically equivalent transition-matrix smoothing—suggesting KL regularization, entropy bonuses, and sampling strategies in LLMs may be more deeply connected than currently treated.
- REDQ's ensemble-based sample efficiency [source:arxiv:2110.05457] and offline fine-tuning's <0.2% data requirement [source:arxiv:2004.10190] demonstrate that pre-trained policies can adapt rapidly with minimal on-policy data, but offline fine-tuning lacks reliable stopping criteria.
- ICQL's centralized exploration with decentralized execution [source:arxiv:1906.02138] and CAML's policy-divergence-based meta-learning [source:arxiv:2010.04816] suggest architectures where a "critic" or "router" guides exploration across diverse prompts/tasks while the generator policy remains deployable alone.

## Key Takeaways

- Entropy collapse in LLM RL fine-tuning is structural: outcome-frequency multipliers and softmax sharpening/squeezing dynamics drive irreversible concentration [source:emergentmind:exploration-collapse-in-reinforcement-le].
- Token-level entropy ($H \to 0$) and outcome-level mode collapse ($|\mathcal{M}| \ll |\mathcal{S}_{\text{reward}}|$) are distinct but coupled failure modes [source:emergentmind:exploration-collapse-in-reinforcement-le].
- Diversity collapse manifests as Pass@1↑ / Pass@k↓ trade-offs, representation rank deficiency, and support shrinkage ($R_- \approx 0.25\text{--}0.35$) [source:emergentmind:diversity-collapse-in-rl-emergent-mind].
- Adaptive temperature control (SCOPE-RL) exploits a temperature–sign asymmetry: high $T$ makes positive samples entropy-increasing, low $T$ makes negative samples entropy-increasing [source:arxiv:2510.08141].
- Perplexity-space disentanglement (DiPO) separates exploitation/exploration spaces via dynamic PPL thresholding and reallocates rewards only in zero-gradient groups [source:arxiv:2604.13902].
- Modern diversity preservation uses mass-covering $f$-divergences (DPH-RL), determinantal volume maximization (DvD), differential smoothing (DS), and mode-anchored rewards (MARA) rather than naive entropy bonuses [source:emergentmind:diversity-collapse-in-rl-emergent-mind].
- REPO/ADAPO modify advantage functions and clipping to preserve entropy, maintaining trainability in sequential tasks [source:machinelearning:entropy-preserving-reinforcement-learnin].
- Non-monotonic entropy–performance curves imply an optimal target entropy ($\mathcal{H}_0 \approx 0.5$ in SCOPE-RL), not maximal entropy [source:arxiv:2510.08141].
- Theoretical analyses rely on tabular softmax, binary rewards, orthogonal gradients, and often only cover $\Delta\mathcal{H} < 0$ regimes [source:arxiv:2510.08141; arxiv:2604.13902].
- **Global vs. episodic bonuses**: In contextual settings (varying prompts/tasks), global bonuses fail when optimal value varies across contexts; episodic bonuses succeed if single-episode coverage is feasible; multiplicative combination is more robust than additive [source:arxiv:2306.03236].
- **First-visit optimism** via ensemble value errors (VBE) enables directed exploration without retroactive reward signals, scaling to high-dimensional observations [source:arxiv:2602.12375].
- **Mellowmax exploration** (MEMEC) provides non-expansion guarantees and outperforms $\epsilon$-greedy in sparse-reward Atari games, but requires per-domain $\omega$ tuning [source:arxiv:1911.09615].
- **Batch RL regularization unification** shows discount factor reduction, Bayesian priors, and $\epsilon$-greedy planning are equivalent transition-matrix smoothing operations—suggesting KL, entropy, and sampling controls in LLMs may share a common mathematical structure [source:arxiv:2109.08134].
- **Real-world fine-tuning** (REDQ) and **offline adaptation** demonstrate pre-trained policies adapt with <0.2% data, but offline methods lack reliable early-stopping, risking catastrophic overfitting [source:arxiv:2110.05457; arxiv:2004.10190].
- **Centralized exploration with decentralized execution** (ICQL) and **policy-divergence meta-learning** (CAML) offer architectures for multi-prompt/task exploration where a guide policy is discarded after training [source:arxiv:1906.02138; arxiv:2010.04816].

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [The RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Length and format bias](length-and-format-bias.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)
- [RL for math and code](rl-for-math-and-code.md)
- [Test-time compute and RL interplay](test-time-and-rl-interplay.md)

## References
- [source:arxiv:2604.13902] [DiPO: Disentangled Perplexity Policy Optimization for Fine-grained Fine-tuning](https://arxiv.org/html/2604.13902v1)
- [source:machinelearning:entropy-preserving-reinforcement-learnin] [Entropy-Preserving Reinforcement Learning](https://machinelearning.apple.com/research/entropy-preserving-reinforcement-learning)
- [source:arxiv:2510.08141] [Entropy Is Controllable in Reinforcement Fine-tuning](https://arxiv.org/html/2510.08141v3)
- [source:github:awesome-exploration-methods-in-reinforce] [Awesome Exploration Methods in Reinforcement Learning (GitHub)](https://github.com/opendilab/awesome-exploration-rl)
- [source:emergentmind:diversity-collapse-in-rl-emergent-mind] [Diversity Collapse in RL (Emergent Mind)](https://www.emergentmind.com/topics/diversity-collapse-in-rl)
- [source:emergentmind:exploration-collapse-in-reinforcement-le] [Exploration Collapse in Reinforcement Learning (Emergent Mind)](https://www.emergentmind.com/topics/exploration-collapse)
- [source:arxiv:2306.03236] [A Study of Global and Episodic Bonuses for Exploration in Contextual MDPs](https://arxiv.org/abs/2306.03236)
- [source:arxiv:2602.12375] [Value Bonuses using Ensemble Errors for Exploration in Reinforcement Learning](https://arxiv.org/abs/2602.12375)
- [source:arxiv:1911.09615] [Sample-Efficient Reinforcement Learning with Maximum Entropy Mellowmax Episodic Control](https://arxiv.org/abs/1911.09615)
- [source:arxiv:2109.08134] [Comparison and Unification of Three Regularization Methods in Batch Reinforcement Learning](https://arxiv.org/abs/2109.08134)
- [source:arxiv:2110.05457] [Legged Robots that Keep on Learning: Fine-Tuning Locomotion Policies in the Real World](https://arxiv.org/abs/2110.05457)
- [source:arxiv:2004.10190] [Never Stop Learning: The Effectiveness of Fine-Tuning in Robotic Reinforcement Learning](https://arxiv.org/abs/2004.10190)
- [source:arxiv:1906.02138] [Exploration with Unreliable Intrinsic Reward in Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1906.02138)
- [source:arxiv:2010.04816] [Characterizing Policy Divergence for Personalized Meta-Reinforcement Learning](https://arxiv.org/abs/2010.04816)
