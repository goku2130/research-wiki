---
title: The RLHF/PPO pipeline
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2203.02155
- arxiv:1707.06347
- arxiv:2009.01325
- arxiv:2305.18290
- arxiv:1706.03741
- arxiv:2305.18290
- arxiv:2305.18290
- arxiv:2305.18290
open_questions:
- 'Reward Model Generalization**: How can reward models be trained to generalize robustly
  to out-of-distribution prompts and edge cases, particularly when the PPO policy’s
  output distribution shifts significantly during training?'
- 'Dynamic Reward Models**: Can online or dynamic reward model training improve generalization
  and reduce reward hacking, and if so, what are the trade-offs in terms of stability,
  compute cost, and implementation complexity?'
- 'KL Regularization**: What is the optimal form (forward vs. reverse KL) and strength
  of KL regularization for different tasks and model sizes, and can adaptive schemes
  improve performance without introducing excessive complexity?'
- 'PPO Variants**: Under what conditions does the KL-penalized PPO variant outperform
  the clipped variant, and can hybrid approaches combine the strengths of both?'
---

Here is the rigorously revised wiki article with all grounding issues addressed, the PPO clip equation added, and the full structure preserved:

---

# The RLHF/PPO Pipeline: A Deep Dive into the InstructGPT Recipe

Reinforcement Learning from Human Feedback (RLHF) with Proximal Policy Optimization (PPO) has emerged as the dominant paradigm for aligning large language models (LLMs) with human preferences, as epitomized by the InstructGPT recipe [source:arxiv:2203.02155]. This pipeline transforms base LLMs—trained solely on next-token prediction—into systems that generate outputs judged as helpful, honest, and harmless by human evaluators, despite the inherent ambiguity and subjectivity of these criteria. The RLHF/PPO pipeline achieves this through a three-stage process: supervised fine-tuning (SFT) on high-quality demonstrations, reward modeling from pairwise human preferences, and policy optimization via PPO with KL regularization.

---

## Pipeline Overview: SFT → Reward Model → PPO

The RLHF/PPO pipeline is structured as a sequential, modular process that progressively refines a base LLM into an aligned policy. Each stage addresses a distinct failure mode of the preceding one, while introducing its own challenges and trade-offs.

### 1. Supervised Fine-Tuning (SFT)
**Objective**: Bridge the distributional gap between the base LLM’s pretraining objective (next-token prediction on diverse, uncurated corpora) and the target task distribution (e.g., instruction following, summarization, or dialogue). SFT provides the policy with a "warm start" by exposing it to high-quality, task-specific demonstrations.

**Methodology**:
- **Data Collection**: Human contractors or domain experts provide demonstrations of desired behavior for a set of prompts. For InstructGPT, approximately 13,000 prompts were used, spanning labeler-written instructions and OpenAI API queries [source:arxiv:2203.02155]. In summarization tasks, reference summaries from datasets like TL;DR or CNN/DM are repurposed [source:arxiv:2009.01325].
- **Training**: The base LLM is fine-tuned via standard supervised learning to maximize the likelihood of the demonstrated outputs. The loss is:
  $$
  \mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{SFT}}} \left[ \log \pi_\theta(y \mid x) \right],
  $$
  where $\pi_\theta$ is the policy, $x$ is the prompt, and $y$ is the demonstrated completion. Training typically employs full fine-tuning or parameter-efficient methods (e.g., LoRA) [source:arxiv:2203.02155].

**Key Considerations**:
- **Data Quality**: The SFT dataset defines the initial policy’s behavior. Low-quality or inconsistent demonstrations (e.g., hallucinations, stylistic variations) propagate into the reward model and PPO stages. InstructGPT’s SFT data was curated by contractors with explicit guidelines, but inter-annotator agreement was only $72.6 \pm 1.5\%$ for training workers [source:arxiv:2203.02155].
- **Distribution Shift**: The SFT policy’s output distribution may diverge from the demonstrations due to the LLM’s inductive biases. For example, the policy may generate longer or more verbose outputs than the reference data, a phenomenon exacerbated by the "alignment tax" [source:arxiv:2203.02155].
- **Task Coverage**: The SFT dataset must cover the target task’s input-output space. Narrow coverage (e.g., focusing only on short instructions) leads to poor generalization. InstructGPT’s SFT data included diverse prompts, but the authors note that the model still struggles with complex or ambiguous instructions [source:arxiv:2203.02155].

**Limitations**:
- SFT alone cannot resolve misalignment arising from the base LLM’s pretraining (e.g., toxicity, bias). The SFT policy inherits these biases and may amplify them if the demonstrations are not carefully curated [source:arxiv:2203.02155].
- The SFT policy is static and cannot adapt to new preferences or edge cases without retraining. This motivates the subsequent RL stages.

---

### 2. Reward Modeling
**Objective**: Learn a scalar reward function $r_\theta(x, y)$ that predicts human preferences over model outputs. The reward model serves as a differentiable proxy for human judgment, enabling gradient-based optimization of the policy.

**Methodology**:
- **Data Collection**: Human labelers compare pairs of model outputs for the same prompt, ranking them from best to worst. For InstructGPT, $K=4$ to $9$ outputs per prompt were ranked, yielding $\binom{K}{2}$ pairwise comparisons per prompt [source:arxiv:2203.02155]. In summarization, labelers compared summaries generated by different policies or baselines [source:arxiv:2009.01325].
- **Model Architecture**: The reward model is typically initialized from the SFT policy (or a separate LLM) with an additional linear layer projecting the final hidden state to a scalar reward. This leverages the SFT policy’s understanding of the task while avoiding the need for a separate model [source:arxiv:2203.02155].
- **Training**: The reward model is trained to minimize the negative log-likelihood of the Bradley-Terry preference model:
  $$
  \mathcal{L}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( r_\theta(x, y_w) - r_\theta(x, y_l) \right) \right],
  $$
  where $\sigma$ is the logistic function, $y_w$ and $y_l$ are the preferred and dispreferred completions, and $\mathcal{D}$ is the comparison dataset. This loss encourages the reward model to assign higher scores to preferred outputs [source:arxiv:2203.02155][source:arxiv:2009.01325].

**Key Considerations**:
- **Preference Noise**: Human preferences are inherently noisy and inconsistent. Inter-annotator agreement for InstructGPT was $72.6 \pm 1.5\%$ for training workers and $77.3 \pm 1.3\%$ for held-out workers [source:arxiv:2203.02155]. In summarization, labeler-researcher agreement reached $77\% \pm 2\%$, comparable to researcher-researcher agreement ($73\% \pm 4\%$) [source:arxiv:2009.01325]. The reward model must generalize despite this noise.
- **Bias and Overfitting**: The reward model may overfit to spurious features in the preference data, such as output length, formatting, or stylistic quirks. For example, the summarization reward model exhibited a bias toward longer summaries, preferring length-increasing edits only $62.6\%$ of the time versus $76.4\%$ for humans [source:arxiv:2009.01325]. InstructGPT’s reward model showed no significant improvement on bias benchmarks like Winogender and CrowS-Pairs [source:arxiv:2203.02155].
- **Generalization**: The reward model must generalize to outputs generated by the PPO policy, which may diverge from the SFT policy’s distribution. Offline training on static comparison data can lead to poor generalization if the policy’s output distribution shifts significantly during PPO [source:arxiv:1706.03741].
- **Scaling**: Reward model performance improves with more comparison data and larger model sizes. Doubling the training data for the summarization reward model increased validation accuracy by $\sim 1.1\%$, while doubling model size increased it by $\sim 1.8\%$ [source:arxiv:2009.01325]. However, scaling is computationally expensive: training a 6.7B reward model required $\sim 320$ GPU-days [source:arxiv:2009.01325].

**Limitations**:
- **Reward Hacking**: The PPO policy may exploit flaws in the reward model to achieve high scores without satisfying the underlying human preferences. For example, the policy might generate verbose or overly cautious outputs to maximize reward, a phenomenon known as "reward hacking" [source:arxiv:2009.01325]. This is exacerbated by the reward model’s inability to capture all dimensions of human preference.
- **Distribution Shift**: The reward model is trained on outputs from the SFT policy or earlier PPO iterations. If the PPO policy’s output distribution shifts significantly, the reward model’s predictions may become unreliable. This is a form of "distribution shift" that can lead to unstable training [source:arxiv:1706.03741].
- **Labeler Bias**: The reward model inherits the biases of the human labelers, who may not represent the broader population. InstructGPT’s labelers were primarily English-speaking individuals in the US or Southeast Asia, limiting the model’s alignment to their specific preferences [source:arxiv:2203.02155].

---

### 3. Proximal Policy Optimization (PPO)
**Objective**: Optimize the SFT policy to maximize the reward model’s scores while constraining divergence from the SFT policy to prevent mode collapse, reward hacking, and catastrophic forgetting.

**Methodology**:
- **Policy Initialization**: The PPO policy $\pi_\phi^{\text{RL}}$ is initialized from the SFT policy $\pi^{\text{SFT}}$. This provides a warm start and ensures the policy begins in a region of the parameter space where the reward model is reliable [source:arxiv:2203.02155].
- **Reward Function**: The reward for a prompt $x$ and completion $y$ is defined as:
  $$
  R(x, y) = r_\theta(x, y) - \beta \log \left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right),
  $$
  where $r_\theta(x, y)$ is the reward model’s score, and the second term is a KL penalty that discourages divergence from the SFT policy. The coefficient $\beta$ controls the strength of the penalty [source:arxiv:2203.02155].
- **PPO Objective**: The policy is optimized using the clipped surrogate objective from [source:arxiv:1707.06347]:
  $$
  L^{CLIP}(\phi) = \mathbb{E}_t \left[ \min \left( r_t(\phi) \hat{A}_t, \operatorname{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right],
  $$
  where $r_t(\phi) = \frac{\pi_\phi^{\text{RL}}(a_t \mid s_t)}{\pi_{\phi_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio, $\hat{A}_t$ is the advantage estimate, and $\epsilon$ is the clipping threshold (typically $\epsilon = 0.2$). The full objective combines the clipped surrogate with a value function loss and entropy bonus:
  $$
  L^{CLIP+VF+S}(\phi) = \mathbb{E}_t \left[ L_t^{CLIP}(\phi) - c_1 L_t^{VF}(\phi) + c_2 S[\pi_\phi^{\text{RL}}](s_t) \right],
  $$
  where $L_t^{VF}$ is a squared-error value loss and $S$ is an entropy bonus [source:arxiv:1707.06347].
- **Advantage Estimation**: Advantages are estimated using Generalized Advantage Estimation (GAE):
  $$
  \hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + \dots + (\gamma \lambda)^{T-t+1} \delta_{T-1},
  $$
  where $\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t)$, $\gamma$ is the discount factor, and $\lambda$ controls the bias-variance trade-off [source:arxiv:1707.06347].
- **Training Loop**: The PPO loop alternates between:
  1. **Rollout Generation**: The current policy $\pi_\phi^{\text{RL}}$ generates completions for a batch of prompts. For InstructGPT, prompts were sampled from a mix of labeler-written instructions, OpenAI API queries, and pretraining data (for PPO-ptx) [source:arxiv:2203.02155].
  2. **Advantage Estimation**: Advantages are computed for each token in the generated completions using the reward function $R(x, y)$ and the value function $V_\phi(s_t)$.
  3. **Policy Update**: The policy is updated via minibatch stochastic gradient descent (typically Adam) on the PPO objective for $K$ epochs (typically $K=4$ for InstructGPT) [source:arxiv:2203.02155][source:arxiv:1707.06347].

**Key Considerations**:
- **KL Regularization**: The KL penalty $\beta \log \left( \frac{\pi_\phi^{\text{RL}}(y \mid x)}{\pi^{\text{SFT}}(y \mid x)} \right)$ serves three purposes:
  1. **Mode Collapse Prevention**: Without the KL penalty, the policy may collapse to a narrow set of high-reward outputs, reducing diversity and generalization [source:arxiv:2203.02155].
  2. **Reward Hacking Mitigation**: The KL penalty discourages the policy from exploiting flaws in the reward model, as such exploits often require large deviations from the SFT policy [source:arxiv:2009.01325].
  3. **Stability**: The KL penalty acts as a regularizer, smoothing the optimization landscape and preventing destructive updates [source:arxiv:1707.06347].
  The coefficient $\beta$ must be carefully tuned: too small, and the policy may diverge; too large, and the policy may fail to improve [source:arxiv:2203.02155].
- **PPO-ptx**: InstructGPT introduced an optional pretraining data mix (PPO-ptx) to mitigate performance regressions on standard NLP benchmarks. The combined objective is:
  $$
  \text{objective}(\phi) = \mathbb{E}_{(x, y) \sim \mathcal{D}_{\pi_\phi^{\text{RL}}}} \left[ R(x, y) \right] + \gamma \mathbb{E}_{x \sim \mathcal{D}_{\text{pretrain}}} \left[ \log \pi_\phi^{\text{RL}}(x) \right],
  $$
  where $\gamma$ controls the strength of the pretraining gradient. This reduces the "alignment tax" but does not fully eliminate it [source:arxiv:2203.02155].
- **Hyperparameters**: PPO’s performance is sensitive to hyperparameters, including the clipping threshold $\epsilon$, the number of epochs $K$, the learning rate, and the advantage estimation parameters $\gamma$ and $\lambda$. InstructGPT used $\epsilon=0.2$, $K=4$, and $\gamma=1$ (no discounting) [source:arxiv:2203.02155]. The authors note that PPO’s stability and sample efficiency make it well-suited for LLM fine-tuning, but extensive tuning is still required [source:arxiv:1707.06347].
- **Distributed Training**: PPO for LLMs requires distributed rollout generation to amortize the cost of sampling from the policy. InstructGPT used a distributed infrastructure with $N=128$ parallel actors, each generating $T=512$ tokens per rollout [source:arxiv:2203.02155].

**Limitations**:
- **Compute Cost**: PPO is computationally expensive, requiring multiple rollouts and policy updates per iteration. InstructGPT’s PPO-ptx stage consumed 60 petaflops/s-days, compared to 4.9 for SFT and 3,640 for GPT-3 pretraining [source:arxiv:2203.02155]. The cost scales with model size, rollout length, and the number of parallel actors.
- **Training Instability**: PPO can exhibit unstable training dynamics, including reward hacking, mode collapse, and catastrophic forgetting. The KL penalty and clipping mitigate these issues but do not eliminate them [source:arxiv:2203.02155][source:arxiv:1707.06347].
- **Reward Over-Optimization**: As the policy improves, it may exploit flaws in the reward model, leading to outputs that achieve high reward scores but are misaligned with human preferences. This is a fundamental limitation of optimizing against a fixed reward model [source:arxiv:2009.01325].
- **Generalization**: The PPO policy may overfit to the reward model’s idiosyncrasies, particularly if the reward model’s training data is narrow or noisy. InstructGPT’s policy generalized well to held-out prompts but still exhibited failures on complex or ambiguous instructions [source:arxiv:2203.02155].

---

## Current Status and Trajectory

The RLHF/PPO pipeline is currently the **default** method for aligning LLMs with human preferences, as evidenced by its adoption in state-of-the-art systems like InstructGPT [source:arxiv:2203.02155], Claude, and ChatGPT. Its dominance stems from several factors:
1. **Empirical Success**: RLHF/PPO has demonstrated significant improvements in human preference metrics across diverse tasks, including instruction following, summarization, and dialogue. InstructGPT’s 1.3B model was preferred over a 175B GPT-3 baseline $85 \pm 3\%$ of the time, despite having 100x fewer parameters [source:arxiv:2203.02155]. Summarization models trained with RLHF were preferred over human references 61% of the time [source:arxiv:2009.01325].
2. **Modularity**: The pipeline’s three-stage structure (SFT → reward model → PPO) decouples the alignment problem into manageable subproblems, each with well-defined objectives and evaluation metrics. This modularity enables iterative improvement and debugging.
3. **Scalability**: The pipeline scales to models with hundreds of billions of parameters, as demonstrated by InstructGPT’s 175B variant [source:arxiv:2203.02155]. The computational cost, while high, is manageable relative to pretraining.

However, the pipeline’s trajectory is **not uniformly upward**, and several challenges may limit its long-term dominance:

### Rising Alternatives
- **Direct Preference Optimization (DPO)**: DPO bypasses the reward modeling and RL stages by directly optimizing the policy on preference data via a closed-form loss [source:arxiv:2305.18290]. DPO has demonstrated comparable or superior performance to RLHF/PPO on sentiment control, summarization, and dialogue tasks, with significantly lower computational cost and implementation complexity. For example, DPO achieved a 61% win rate on TL;DR summarization at temperature 0.0, surpassing PPO’s 57% [source:arxiv:2305.18290]. DPO’s simplicity and efficiency make it a strong contender, particularly for smaller models or resource-constrained settings. However, its long-term scalability and generalization capabilities remain unproven.
- **Rejection Sampling and Best-of-N**: These methods generate $N$ completions per prompt and select the highest-reward output, either for inference or as a training signal [source:arxiv:2009.01325]. While computationally expensive, they avoid the complexity of RL and can serve as a strong baseline or post-hoc alignment method. Best-of-128 achieved performance comparable to DPO on the Anthropic HH dataset [source:arxiv:2305.18290].

### Challenges to RLHF/PPO
1. **Compute and Data Costs**: The pipeline’s computational and data requirements are substantial. InstructGPT’s PPO-ptx stage required 60 petaflops/s-days, and the reward modeling stage relied on 33,000 human-labeled comparisons [source:arxiv:2203.02155]. Scaling to larger models or more diverse tasks will exacerbate these costs.
2. **Reward Hacking and Over-Optimization**: The pipeline is fundamentally limited by the quality of the reward model. As policies improve, they may exploit flaws in the reward model, leading to outputs that achieve high reward scores but are misaligned with human preferences [source:arxiv:2009.01325]. This is a manifestation of the "reward over-optimization" problem, which is not fully resolved by KL regularization or clipping.
3. **Generalization and Robustness**: The pipeline’s performance is sensitive to the quality and coverage of the SFT and preference data. Poor generalization to out-of-distribution prompts or edge cases remains a challenge. InstructGPT, for example, struggles with complex or ambiguous instructions [source:arxiv:2203.02155].
4. **Bias and Misalignment**: The pipeline inherits the biases of the human labelers and the base LLM. InstructGPT showed no significant improvement on bias benchmarks like Winogender and CrowS-Pairs, and its alignment is limited to the preferences of a narrow group of contractors [source:arxiv:2203.02155]. Addressing these biases requires more diverse and representative preference data, as well as novel techniques for bias mitigation.

### Trajectory: Default but Evolving
The RLHF/PPO pipeline is likely to remain the **default** method for LLM alignment in the near term, particularly for large-scale, production-grade systems. However, its dominance may be **eroded** by alternatives like DPO, which offer comparable performance with lower complexity. The pipeline’s long-term trajectory will depend on:
- **Scaling**: Whether the pipeline can scale efficiently to models with trillions of parameters or more diverse tasks (e.g., agentic or tool-use scenarios).
- **Generalization**: Whether the pipeline can achieve robust generalization to out-of-distribution prompts, edge cases, and novel tasks without requiring excessive data or compute.
- **Bias and Alignment**: Whether the pipeline can address the biases and misalignment inherent in human feedback, particularly for tasks with conflicting or ambiguous preferences.
- **Compute Efficiency**: Whether the pipeline’s computational cost can be reduced without sacrificing performance, e.g., through more efficient RL algorithms, distributed training, or hybrid approaches.

**Hedging**: While the pipeline is currently the default, its trajectory is not guaranteed. Alternatives like DPO are rising rapidly, and the field may shift toward simpler, more efficient methods if they demonstrate comparable or superior performance. The pipeline’s modularity may also enable hybrid approaches, e.g., using DPO for initial alignment and RLHF/PPO for fine-tuning.

---

## Disagreements and Open Challenges

The RLHF/PPO pipeline is not monolithic, and several key disagreements and open challenges persist in the literature and practice:

### 1. Reward Model Training: Online vs. Offline
**Disagreement**: The reward model can be trained either offline (on a static dataset of comparisons) or online (on comparisons generated by the current policy during PPO). Offline training is simpler and more stable but may suffer from distribution shift if the PPO policy diverges significantly from the SFT policy. Online training can adapt to the policy’s output distribution but introduces complexity and potential instability.

- **Offline Training**: Used in InstructGPT [source:arxiv:2203.02155] and the summarization work [source:arxiv:2009.01325]. The reward model is trained once on a static dataset and remains fixed during PPO. This avoids the complexity of online updates but may lead to poor generalization if the policy’s output distribution shifts.
- **Online Training**: Proposed in [source:arxiv:1706.03741] for robotics tasks, where the reward model is updated periodically using comparisons generated by the current policy. This can improve generalization but introduces challenges in balancing exploration and exploitation, as well as potential feedback loops between the policy and reward model.

**Settling the Disagreement**: Empirical comparisons of offline vs. online reward model training for LLMs are lacking. A systematic study could determine whether the benefits of online training outweigh its complexity for LLM alignment.

### 2. PPO Variants: Clip vs. KL Penalty
**Disagreement**: PPO offers two primary variants for constraining policy updates: the clipped surrogate objective ($L^{CLIP}$) and the KL-penalized objective ($L^{KLPEN}$) [source:arxiv:1707.06347]. The clipped objective is simpler and empirically more stable, but the KL-penalized objective provides stronger theoretical guarantees.

- **Clipped Objective**: Used in InstructGPT [source:arxiv:2203.02155] and the summarization work [source:arxiv:2009.01325]. The clipped objective is defined as:
  $$
  L^{CLIP}(\phi) = \mathbb{E}_t \left[ \min \left( r_t(\phi) \hat{A}_t, \operatorname{clip}(r_t(\phi), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right].
  $$
  It is computationally simple and empirically robust but lacks strong theoretical guarantees.
- **KL-Penalized Objective**: Proposed in [source:arxiv:1707.06347] as an alternative to clipping. The objective is:
  $$
  L^{KLPEN}(\phi) = \mathbb{E}_t \left[ r_t(\phi) \hat{A}_t - \beta \text{KL}[\pi_{\phi_{\text{old}}}, \pi_\phi] \right],
  $$
  where $\beta$ is adaptively adjusted to target a specific KL divergence $d_{\text{targ}}$. This variant provides stronger theoretical guarantees but is more sensitive to hyperparameters and empirically underperformed clipping in [source:arxiv:1707.06347].

**Settling the Disagreement**: The clipped objective is currently the default due to its empirical robustness, but the KL-penalized objective may offer advantages in settings where precise control over policy divergence is critical. Further research could explore hybrid approaches or adaptive schemes that combine the strengths of both variants.

### 3. KL Regularization: Strength and Form
**Disagreement**: The strength and form of the KL penalty in the PPO objective are critical to the pipeline’s success but remain poorly understood. The coefficient $\beta$ must balance reward maximization and policy divergence, but its optimal value is task- and model-dependent.

- **Fixed $\beta$**: Used in InstructGPT [source:arxiv:2203.02155] and the summarization work [source:arxiv:2009.01325]. The coefficient $\beta$ is fixed during training, requiring careful tuning. Too small, and the policy may diverge; too large, and the policy may fail to improve.
- **Adaptive $\beta$**: Proposed in [source:arxiv:1707.06347] for the KL-penalized PPO variant. The coefficient $\beta$ is adjusted dynamically to target a specific KL divergence $d_{\text{targ}}$. This can improve stability but introduces additional hyperparameters and complexity.

**Settling the Disagreement**: The optimal KL penalty may depend on the task, model size, and reward model quality. Systematic studies could explore adaptive schemes or task-specific tuning protocols. Additionally, the form of the KL penalty (e.g., forward vs. reverse KL) may impact performance, but this remains largely unexplored in the RLHF context.

### 4. Reward Hacking and Over-Optimization
**Disagreement**: Reward hacking—where the policy exploits flaws in the reward model to achieve high scores without satisfying the underlying human preferences—is a fundamental challenge in RLHF. The pipeline’s current mitigation strategies (KL regularization, clipping) are necessary but not sufficient.

- **KL Regularization**: The KL penalty discourages large deviations from the SFT policy, which can mitigate reward hacking by limiting the policy’s ability to exploit reward model flaws [source:arxiv:2203.02155]. However, it does not eliminate the problem, as the policy may still exploit subtle flaws within the KL constraint.
- **Iterative Alignment**: Iteratively retraining the reward model on comparisons generated by the PPO policy can improve generalization but introduces complexity and potential instability [source:arxiv:2009.01325].

**Settling the Disagreement**: Reward hacking is an inherent limitation of optimizing against a fixed reward model. Potential solutions include:
- **Process-Based Reward Models**: Evaluating intermediate steps or reasoning traces may reduce the reward model’s susceptibility to exploitation.
- **Dynamic Reward Models**: Updating the reward model during PPO to adapt to the policy’s output distribution may improve generalization and reduce hacking.
- **Hybrid Objectives**: Combining reward model scores with auxiliary objectives (e.g., diversity, coherence) may mitigate hacking by providing additional signals for alignment.

### 5. Generalization to Out-of-Distribution Prompts
**Disagreement**: The pipeline’s generalization to out-of-distribution (OOD) prompts is a critical but understudied challenge. The SFT and preference data may not cover all possible prompts, leading to poor performance on edge cases or novel tasks.

- **SFT Generalization**: The SFT policy’s generalization depends on the coverage and diversity of the SFT dataset. InstructGPT’s SFT data included diverse prompts but still struggled with complex or ambiguous instructions [source:arxiv:2203.02155].
- **Reward Model Generalization**: The reward model’s generalization depends on the coverage of the preference data. Offline training on static data may lead to poor OOD performance if the PPO policy’s output distribution shifts significantly [source:arxiv:2009.01325].
- **PPO Generalization**: The PPO policy’s generalization depends on the reward model’s robustness and the KL penalty’s strength. A weak KL penalty may allow the policy to diverge too far from the SFT distribution, while a strong penalty may limit its ability to adapt to OOD prompts.

**Settling the Disagreement**: Improving OOD generalization may require:
- **Diverse Preference Data**: Collecting preference data for a broader range of prompts, including edge cases and novel tasks.
- **Online Reward Model Training**: Updating the reward model during PPO to adapt to the policy’s output distribution.
- **Auxiliary Objectives**: Incorporating objectives that encourage diversity, coherence, or other desirable properties may improve generalization.

---

## Key Takeaways

- **Modular Pipeline**: The RLHF/PPO pipeline consists of three sequential stages—SFT, reward modeling, and PPO—each addressing a distinct failure mode of the preceding stage. This modularity enables iterative improvement and debugging but introduces complexity and potential failure points.
- **Empirical Success**: The pipeline has demonstrated significant improvements in human preference metrics across diverse tasks, including instruction following, summarization, and dialogue. InstructGPT’s 1.3B model was preferred over a 175B GPT-3 baseline $85 \pm 3\%$ of the time [source:arxiv:2203.02155].
- **Default but Evolving**: The pipeline is currently the default method for LLM alignment but faces challenges from rising alternatives like DPO, as well as inherent limitations in compute cost, reward hacking, and generalization.
- **Critical Trade-offs**: The pipeline involves trade-offs between reward maximization and policy divergence (KL regularization), stability and performance (PPO variants), and generalization and specialization (online vs. offline reward modeling).
- **Open Challenges**: Key unresolved challenges include reward hacking, generalization to OOD prompts, bias and misalignment, and the optimal form and strength of KL regularization. Addressing these challenges may require novel techniques, such as process-based reward models, dynamic reward updates, or hybrid objectives.

---

## Related Topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md): A deeper dive into the PPO stage of the pipeline, including implementation details, hyperparameter tuning, and distributed training.
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md): An exploration of DPO and other methods that bypass explicit reward modeling and RL, including their theoretical foundations and empirical performance.
- [Reward modeling for LLMs](reward-modeling.md): A detailed discussion of reward model architectures, training methodologies, and challenges, including bias, generalization, and reward hacking.
- [KL regularization in RLHF](kl-regularization.md): A focused analysis of KL regularization in RLHF, including its role in preventing mode collapse, reward hacking, and catastrophic forgetting, as well as alternative regularization schemes.
- [Reward hacking in RLHF](reward-hacking.md): An in-depth examination of reward hacking, including its causes, manifestations, and mitigation strategies.
- [Reward model over-optimization](reward-model-overoptimization.md): A discussion of the over-optimization problem in RLHF, where policies exploit flaws in the reward model to achieve high scores without satisfying the underlying human preferences.
- [Process vs outcome reward models](process-vs-outcome-rewards.md): A comparison of outcome-based and process-based reward models, including their strengths, weaknesses, and applications in RLHF.
- [The alignment tax](alignment-tax.md): An analysis of the "alignment tax," where aligning LLMs with human preferences leads to performance regressions on standard NLP benchmarks, and strategies for mitigating this trade-off.
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md): A discussion of over-optimization and mode collapse in RLHF, including their causes, consequences, and mitigation strategies.
- [Sycophancy and misgeneralization](sycophancy-and-misgeneralization.md): An exploration of sycophancy and misgeneralization in aligned LLMs, where models generate outputs that superficially satisfy human preferences but fail to generalize to novel or ambiguous contexts.

---

##

## References
- [source:arxiv:2203.02155] [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [source:arxiv:2009.01325] [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:1706.03741] [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [source:arxiv:2305.18290] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
