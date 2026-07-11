---
id: arxiv:2605.05226
type: paper
title: 'Internalizing Outcome Supervision into Process Supervision: A New Paradigm
  for Reinforcement Learning for Reasoning'
url: https://arxiv.org/abs/2605.05226
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

**Core Problem.** Reasoning reinforcement learning (RL) primarily relies on outcome supervision, which provides only sparse, sequence-level binary feedback. This granularity is insufficient for fine-grained credit assignment in long-chain reasoning, as uniformly penalizing failed trajectories conflates correct intermediate steps with the actual sources of failure. While exogenous process supervision (e.g., human annotations, process reward models) addresses this, it incurs high construction costs and scales poorly. The core challenge is converting end-of-sequence outcome feedback into precise, token-level learning signals without external supervision.

**Method/Recipe.** The proposed IOP (Internalizing Outcome Supervision into Process Supervision) framework reframes reasoning RL as a self-correction paradigm where a single shared-parameter model $\theta$ acts as both a policy generator $\pi_\theta$ and a repair mode $\rho_\theta$. The training proceeds in two stages. Stage 1 performs cold-start supervised fine-tuning (SFT) on 500 manually crafted error-repair pairs to initialize $\rho_\theta$. Stage 2 executes joint RL optimization iteratively: (1) sample $G'$ candidate trajectories from the policy; (2) partition them into correct ($\mathcal{G}_{\text{cor}}$) and error ($\mathcal{G}_{\text{err}}$) sets using a reward threshold $\tau_r=0.5$, where correct samples serve exclusively as reference anchors $a$; (3) generate $G_{\text{rep}}$ repair candidates for each error trajectory conditioned on $x, y, a$; (4) filter candidates via an external audit model to prevent reward hacking, then select the best repair $\bar{y}_i$ by maximizing correctness and minimizing normalized edit distance; (5) compute bilateral token-level difference masks between $y_i$ and $\bar{y}_i$ using Levenshtein alignment, then apply verification-based adaptive truncation to retain only the first $K^*$ difference tokens; (6) apply these masks as gating signals to restrict policy updates to active tokens; and (7) jointly update $\theta$ using both the gated policy objective and the standard repair objective.

**Key Formulas.** The repair scoring function balances task correctness and edit cost:

$$
s_{\text{rep}}(x, y_i, a, \bar{y}_i^{(j)}) = h_i^{(j)} \cdot \left( r_{\text{task}}(x, \bar{y}_i^{(j)}) - \lambda_{\text{edit}} \bar{\Delta}_{\text{edit}}(y_i, \bar{y}_i^{(j)}) \right).
$$

Adaptive truncation determines the optimal window $K^*$ by grafting edits and verifying correctness:

$$
K^* = \begin{cases} K, & r_{\text{task}}(x, \hat{\mathbf{y}}^{(K)}) = 1, \\ 2K, & r_{\text{task}}(x, \hat{\mathbf{y}}^{(K)}) = 0 \wedge r_{\text{task}}(x, \hat{\mathbf{y}}^{(2K)}) = 1, \\ |\mathcal{S}_\infty|, & \text{otherwise.} \end{cases}
$$

Gating restricts the sequence-level likelihood ratio to active tokens:

$$
w_z^{\text{gate}}(\theta) = \exp\left(\frac{1}{\sum_{t'=1}^{T_z} g_{z,t'}} \sum_{t=1}^{T_z} g_{z,t} \log \frac{\pi_\theta(z_t \mid x, z_{<t})}{\pi_{\theta_{\text{old}}}(z_t \mid x, z_{<t})}\right).
$$

The final joint objective combines the gated clipped policy loss and the repair loss:

$$
\mathcal{J}_{\text{IOP}}(\theta) = \mathcal{J}_{\text{IOP-GSPO}}(\theta; \mathcal{B}_{\text{pol}}) + \lambda_{\text{rep}} \mathcal{J}_{\text{rep}}(\theta; \mathcal{B}_{\text{rep}}).
$$


**Quantitative Results.** Evaluated on Qwen3-32B and Qwen3-Next architectures, IOP-GSPO consistently outperforms the GSPO baseline by +4.9–6.9% across AIME25, LiveCodeBench, and HMMT25, with the largest gains on long-chain reasoning (HMMT25: +7.8%/+6.8%). The method achieves approximately 2.3$\times$ sample efficiency, reaching GSPO’s final performance in ~350 steps versus 800. Training dynamics show a synergistic loop: repair success rate rises from 38.4% to 73.1%, policy accuracy improves from 52.1% to 83.5%, and the active token ratio declines from 24.6% to 9.8%, indicating automated signal refinement. Ablations confirm that excluding correct samples from gradient updates is critical to prevent entropy collapse, and the audit gate prevents ~8% of reward-hacking repairs from degrading performance. Computational overhead remains low, adding <6% GPU hours for a 3.7–7.8% accuracy gain.

**Limitations.** The approach has three primary constraints. First, minimum edit distance does not guarantee causal minimality; paraphrasing or equivalent transformations may cause difference masks to misalign with true error sources. Second, the pipeline requires at least one correct trajectory per prompt as a reference anchor, making it vulnerable when the base model is too weak to generate reliable references or repairs. Third, empirical validation is restricted to mathematical and code reasoning benchmarks on models up to 32B parameters, leaving larger-scale models, long-context reasoning, tool use, and multi-turn agent scenarios untested.
