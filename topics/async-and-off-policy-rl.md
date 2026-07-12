---
title: Async and off-policy RL
maturity: comprehensive
updated: '2026-07-12'
sources:
- intellisys:stellaris-staleness-aware-distributed-re
- arxiv:2007.12085
- arxiv:2512.06547
- arxiv:1602.01783
- arxiv:1802.01561
- arxiv:1507.02646
- arxiv:2601.12784
- arxiv:2511.05589
- arxiv:2606.24143
- arxiv:1810.12558
- arxiv:2409.03915
- arxiv:1906.08850
open_questions:
- Can PSIS $\hat{k}$ diagnostics be integrated into async RL training loops for dynamic
  truncation or buffer management?
- What is the bias-variance decomposition for *staleness distributions* (not just
  mean staleness) in LLM-scale async RL?
- How do communication delays in distributed LLM training affect the convergence guarantees
  of async SA theory?
- Can multi-sample MC support caching in AsyncOPD scale to multi-node clusters with
  efficient teacher-score transfer?
---

Async and off-policy reinforcement learning addresses the fundamental tension between sample efficiency and hardware utilization: decoupling environment interaction (rollouts) from gradient computation enables massive throughput but introduces policy-lag, where the behavior policy generating data diverges from the target policy being optimized. This divergence necessitates off-policy corrections—primarily importance sampling (IS)—whose design dictates the bias-variance trade-off, stability, and ultimate scalability of distributed RL systems.

## The decoupled actor-learner paradigm and the policy-lag problem

Classical on-policy algorithms (e.g., A3C [source:arxiv:1602.01783]) run multiple actor-learners on a shared CPU parameter server, synchronizing gradients after every $t_{\text{max}}$ steps. This avoids experience replay but limits throughput to CPU speeds and forces synchronization. IMPALA [source:arxiv:1802.01561] broke this coupling: actors pull the latest target policy $\pi$ from a centralized GPU learner, run $n$-step trajectories under their local behavior policy $\mu$, and push trajectories (states, actions, rewards, $\mu(a|x)$, initial LSTM states) into a queue. The learner consumes minibatches asynchronously, parallelizing time-independent forward passes (e.g., convolutions over all steps) via XLA/cuDNN. This achieves **250,000 frames/sec**—over **30× A3C**—but creates *policy-lag*: $\mu$ is stale relative to $\pi$, making data off-policy. If uncorrected, this off-policy lag can lead to instability and reduced data efficiency [source:arxiv:1802.01561].

## V-trace: truncated importance sampling for actor-critic

IMPALA's V-trace [source:arxiv:1802.01561] corrects the $n$-step value target using per-timestep truncated IS ratios:

$$
\rho_t = \min\!\left(\bar{\rho},\; \frac{\pi(a_t|x_t)}{\mu(a_t|x_t)}\right), \qquad
c_i = \min\!\left(\bar{c},\; \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)}\right)
$$

The V-trace target for $V(x_s)$ is

$$
v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \Bigl(\prod_{i=s}^{t-1} c_i\Bigr)\, \delta V_t,
\quad
\delta V_t = \rho_t\bigl(r_t + \gamma V(x_{t+1}) - V(x_t)\bigr).
$$

The policy gradient uses $\rho_\pi \nabla \log \pi (r_s + \gamma v_{s+1} - V(x_s))$. The truncation $\bar{c}$ reduces variance without changing the fixed point; $\bar{\rho}$ *does* change the fixed point: with finite $\bar{\rho}$, V-trace converges to $V^{\pi_\rho}$ where $\pi_\rho$ lies between $\mu$ and $\pi$, not $V^\pi$ [source:arxiv:1802.01561]. This bias-variance trade-off is explicit: $\bar{\rho}=1$ (no IS correction) yields low-variance but biased updates toward $\mu$; $\bar{\rho}=\infty$ is unbiased but high-variance. IMPALA used finite truncation values in practice, accepting bias for stability at scale.

**Disagreement on truncation strategy**: A3C [source:arxiv:1602.01783] avoids IS entirely by synchronizing frequently (shared RMSProp, 16-thread CPU), accepting lower throughput. IMPALA [source:arxiv:1802.01561] embraces asynchrony and truncates aggressively. Stellaris [source:intellisys:stellaris-staleness-aware-distributed-re] argues that in *multi-learner* async settings, per-learner IS ratios are insufficient because each learner's $\pi_{\theta_i}$ drifts differently; they propose a **global IS truncation** across all learners:

$$
R' := \min\!\Bigl(\bigl|\min_i \tfrac{\pi_{\theta_i}}{\mu_\theta}\bigr|,\; \rho\Bigr),
$$

using the *minimum* ratio across learners to prevent any single learner's wild update from corrupting the aggregate. This is more conservative than IMPALA's per-trajectory truncation and reflects a different failure mode: cross-learner policy drift rather than actor-learner lag.

## Advanced importance sampling diagnostics and stabilization

Beyond simple truncation, modern off-policy RL employs sophisticated IS diagnostics and adaptive stabilization techniques to manage the heavy-tailed ratio distributions that arise in high-dimensional policy spaces.

### Pareto Smoothed Importance Sampling (PSIS)

PSIS [source:arxiv:1507.02646] stabilizes self-normalized IS estimators by replacing the largest importance ratios with expected order statistics from a fitted Generalized Pareto Distribution (GPD). The method sorts ratios $r_s$, retains the bulk $S-M$ weights (with $M = \lfloor \min(0.2S, 3\sqrt{S}) \rfloor$), fits a GPD to the $M$ largest ratios using the Zhang-Stephens approximate Bayesian method, and replaces the tail with inverse-CDF quantiles $w_{S-M+z} = \min(F^{-1}((z-1/2)/M), \max_s r_s)$. The fitted shape parameter $\hat{k}$ serves as a diagnostic:
- $\hat{k} < 0.5$: standard $\sim S^{1/2}$ convergence (CLT regime)
- $0.5 < \hat{k} < 0.7$: slowed convergence $\sim S^{1-k}$, practically usable with caution
- $\hat{k} > 0.7$: bias dominates, variance-based MCSE estimates fail
- $\hat{k} > 1$: mean of ratio distribution likely does not exist

The approximate effective sample size is $\text{ESS}_{\hat{k}} \approx S / 10^{\hat{k}/(1-\hat{k})}$, and the minimum sample size to control RMSE scales as $10^{1/(1-\hat{k})}$. In high-dimensional tests up to $D=1024$, $\hat{k}$ correctly diagnosed ESS collapse even when ratios were bounded, identifying cases where sample size was insufficient to reach the asymptotic regime [source:arxiv:1507.02646].

### Relative Importance Sampling (RIS)

RIS [source:arxiv:1810.12558] proposes a "smooth" alternative to hard truncation. The RIS ratio introduces a smoothness parameter $\beta \in [0,1]$:

$$
\mu_{\beta} = \frac{e^{\pi(a|s)}}{\beta e^{\pi(a|s)} + (1-\beta)e^{b(a|s)}}
$$

As $\beta \to 1$, the variance of the RIS estimator decreases toward zero. Implemented in RIS-off-PAC (standard gradient) and RIS-off-PNAC (natural gradient), the method achieved superior average rewards on continuous control benchmarks (CartPole, Humanoid-v2, Pendulum, MountainCar) compared to A3C, PPO, and SAC, with statistical significance confirmed via Kruskal-Wallis tests ($p < 0.05$) [source:arxiv:1810.12558]. The variance expression $V_{\beta}(\hat{\mu}_{\beta}) = \frac{2\gamma(1-\gamma)(1-\beta)}{[\beta\pi(A|S)+(1-\beta)b(A|S)]^2}$ makes the bias-variance trade-off explicit and tunable via $\beta$.

### Importance Weighted Moment Matching (IWMM)

IWMM [source:arxiv:1906.08850] adapts the proposal distribution implicitly via affine transformations of existing samples rather than parametric refitting. It targets the optimal SNIS proposal $g_{\text{SNIS}}^{\text{opt}}(\theta) \propto p(\theta)|h(\theta) - \mathbb{E}_p[h]|$ using a split-proposal approximation, then applies iterative moment-matching transformations $T_1$ (mean), $T_2$ (marginal variance), $T_3$ (full covariance) guided by the PSIS $\hat{k}$ diagnostic. The method stops when $\hat{k} \le 0.7$ or $T_3$ fails to reduce $\hat{k}$ further. Evaluated on Bayesian LOO-CV, IWMM achieved reliable estimates ($\hat{k} < 0.7$) with $S=1,000$ draws in Poisson regression where parametric AMIS failed at $S=64,000$, reduced problematic folds from 34.8 to 16.2 in a 3,075-parameter ovarian cancer model, and cut compute time from 27,477s (AMIS) to 1,558s [source:arxiv:1906.08850]. Limitations include dependence on initial proposal quality and the inability of simple affine transforms to handle differing tail thicknesses or multimodality.

## Staleness quantification and adaptive control in LLM-scale async RL

At LLM scale (billions of parameters), the proximal policy $\pi_{\text{prox}}$ in decoupled PPO—used as a trust-region anchor separating off-policy correction from the CLIP constraint—requires a full forward pass per minibatch, costing **4–8 seconds** [source:arxiv:2512.06547]. A-3PO eliminates this by approximating $\pi_{\text{prox}}$ via staleness-aware log-linear interpolation. Let $d = v(\pi_\theta) - v(\pi_{\text{behav}})$ be the version gap (number of learner updates since the rollout). Define

$$
\alpha = \begin{cases} 1, & d=0 \\ 1/d, & d\ge 1 \end{cases}, \qquad
\log \pi_{\text{prox}} = \alpha \log \pi_{\text{behav}} + (1-\alpha)\log \pi_\theta.
$$

This satisfies a **sandwich property**: $\min\{\pi_{\text{behav}},\pi_\theta\} \le \pi_{\text{prox}} \le \max\{\pi_{\text{behav}},\pi_\theta\}$. The decoupled CLIP objective becomes

$$
L = \mathbb{E}_t\Bigl[ \tfrac{\pi_{\text{prox}}}{\pi_{\text{behav}}} \min\Bigl( \tfrac{\pi_\theta}{\pi_{\text{prox}}}\hat{A}_t,\; \text{clip}\bigl(\tfrac{\pi_\theta}{\pi_{\text{prox}}},1\pm\epsilon\bigr)\hat{A}_t \Bigr) \Bigr].
$$

The effective importance ratio $r = \pi_\theta/\pi_{\text{prox}} = (\pi_\theta/\pi_{\text{behav}})^\alpha$ is *contractive*: as staleness $d$ grows ($\alpha\to0$), $r\to 1$, automatically damping updates from very stale data. A-3PO reduces proximal-policy compute from **4–8 s → 0.0012 s** (>**3,000×**), yielding **1.8×** overall training speedup vs. synchronous GRPO on Qwen3-8B (14.54 h vs. 26.15 h) with matching AIME24 pass@1 (**66.67%**) and better MATH500 (**66.60% vs. 46.80%**) [source:arxiv:2512.06547].

Stellaris [source:intellisys:stellaris-staleness-aware-distributed-re] takes a complementary approach: rather than approximating a proximal policy, it *delays gradient aggregation* until the average staleness $\bar{\delta}$ in the queue falls below a dynamic threshold $\beta_k = \delta_{\max} \cdot d^k$ ($d\in(0,1]$ decaying over rounds $k$). It further modulates each gradient's learning rate by its staleness $\delta_c$:

$$
\alpha_c = \frac{\alpha_0}{\sqrt{\delta_c}}\;\text{if}\;\delta_c > v, \qquad
g_c = \frac{1}{H_c}\sum_{j=1}^{H_c} \frac{\alpha_0}{\sqrt{\delta_j}} g_{i,j}.
$$

This *staleness-aware aggregation* achieved **2.2× higher rewards** and **41% cost reduction** vs. baselines on MuJoCo/Atari using serverless learners, with **<5% system overhead** [source:intellisys:stellaris-staleness-aware-distributed-re]. The two methods differ in philosophy: A-3PO *approximates* the ideal proximal policy to keep throughput high; Stellaris *throttles* aggregation to bound staleness, trading latency for stability.

### Modern async LLM RL systems: StaleFlow, CoPRIS, and AsyncOPD

Recent work has introduced fully disaggregated architectures for LLM post-training where rollout, reward, and training phases are decoupled onto separate resources, with explicit staleness control and concurrency management.

#### StaleFlow: Staleness-constrained rollout coordination

StaleFlow [source:arxiv:2601.12784] introduces a **virtual staleness buffer** abstraction to enforce a user-defined staleness bound $\eta$. Each trajectory carries a version identifier $V_{\text{traj}}$ (oldest tolerated model version during generation), and a trajectory in buffer version $V_{\text{buf}}$ must satisfy $V_{\text{traj}} + \eta \ge V_{\text{buf}}$. The system uses `Reserve` (placeholder for in-flight trajectories) and `Occupy` (finalize rewarded trajectory) primitives, with buffers categorized as *Waiting*, *Ready*, or *Stuck*. A centralized coordinator executes a **Snapshot-Command Cycle** using a speculative state to avoid acting on outdated information, issuing `Pull`, `Route`, and `Interrupt` commands. Routing uses a multi-level queue by $V_{\text{traj}}$ and a waterfall model maximizing marginal throughput gain $\Delta \mathcal{T}_i$ estimated via a cost model with profiled coefficients $k_1\text{--}k_5$. Migration proactively interrupts trajectories if waiting queues exceed $\varphi_{\text{wait}}$ or throughput gaps exceed $\psi_{\text{throughput}}$. On a 128-GPU H20 cluster with Qwen3-30B-A3B and Qwen2.5-32B, StaleFlow achieves **1.42–2.68×** throughput over VeRL (sync) and **1.42×** over VeRL-Async (strict staleness), preserving convergence for $\eta \in [1,3]$ but collapsing at $\eta=10$. System overhead (TS, PS, coordination) is **<3%** [source:arxiv:2601.12784].

#### CoPRIS: Concurrency-controlled partial rollout with IS

CoPRIS [source:arxiv:2511.05589] addresses the "long-tail problem" of variable response lengths in LLM rollouts by maintaining fixed concurrency $N'$ in inference engines and terminating rollout once batch size $B$ is collected. Unfinished trajectories are buffered with their log-probabilities $\mathbf{L}_i^{(k)}$ from each generation stage $k$, then prioritized for resumption. Cross-stage importance sampling corrects the distribution mismatch: the concatenated log-probability $\mathbf{L}_i = \text{concat}(\mathbf{L}_i^{(1)}, \dots, \mathbf{L}_i^{(K)})$ yields per-token ratios $r_{i,t}(\theta) = \exp(\mathbf{L}_{i,t}^{(\theta)} - \mathbf{L}_{i,t})$ for reweighting. On math reasoning benchmarks (AIME, AMC, MinervaMath, OlympiadBench) with Distill-Qwen (1.5B, 7B) and Qwen3-8B, CoPRIS achieves **1.58–1.94×** end-to-end speedup over synchronous veRL, with near-linear scaling from **1.27× at 8K tokens to 2.26× at 40K tokens** and consistent **1.57–1.85×** across 1.5B–14B models. Distill-Qwen-7B pass@1 improves from 52.66% to **53.68%**. Limitations include entropy sensitivity (high-entropy models benefit more from off-policy exploration) and static concurrency $N'$ potentially triggering memory recomputation for larger models [source:arxiv:2511.05589].

#### AsyncOPD: Asynchronous on-policy distillation with cached teacher support

AsyncOPD [source:arxiv:2606.24143] studies staleness in on-policy distillation (OPD) under a **teacher-cache constraint**: full-vocabulary teacher logits are too expensive to transfer asynchronously, so only a finite action set is scored. This creates support mismatch where the current student policy needs teacher scores for uncached actions. The paper compares forward-KL (teacher-weighted) vs. reverse-KL (student-weighted) objectives, finding forward-KL more robust to staleness. For reverse-KL, it proposes an **importance-sampling identity** recomputing the advantage $A_\theta$ under the current student: $D_R(\theta;s) = -\mathbb{E}_{a\sim p_{\text{old}}}[\rho_\theta(a,s) A_\theta(a,s)]$ with $\rho_\theta = p_\theta/p_{\text{old}}$. To reduce variance of Monte Carlo (MC) estimation with sparse cached supports, it introduces **multi-sample MC** caching $m$ local samples per timestep, averaging $\widehat{L}_m^{\text{MC}} = -\frac{1}{m}\sum_{i=1}^m \rho_\theta(a_i,s) \text{sg}(A_\theta(a_i,s))$. On Qwen3-Base (1.7B, 4B, 8B), AsyncOPD achieves **1.6–3.8×** throughput over synchronous training with comparable accuracy. Multi-sample MC with $m=64$ reduces local next-token variance to **1.49%** of one-sample MC. One-sample MC with IS outperforms stale sparse top-$k$ by recovering missing teacher scores. Limitations: dense full-vocabulary KL not evaluated due to transfer cost; experiments limited to single 8-GPU node [source:arxiv:2606.24143].

## Theoretical foundations: asynchronous stochastic approximation

The convergence of async RL algorithms is grounded in asynchronous stochastic approximation (SA) theory. For average-reward RL in MDPs/SMDPs, the lack of contraction properties in the Bellman operator requires specialized analysis [source:arxiv:2409.03915]. The framework considers component-wise updates $x_{n+1}(i) = x_n(i) + \alpha_{\nu(n,i)}(h_i(x_n) + M_{n+1}(i) + \epsilon_{n+1}(i))$ with diminishing stepsizes, martingale noise $M$, and biased noise $\epsilon$ satisfying $\|\epsilon_{n+1}\| \le \delta_{n+1}(1+\|x_n\|)$, $\delta_{n+1} \to 0$ a.s. Stability is proved via stopping-time techniques constructing auxiliary processes tracking scaled ODEs $\dot{x} = \lambda(t)h_c(x)$. Convergence to a compact connected internally chain transitive invariant set of $\dot{x}=h(x)$ is guaranteed; unique convergence requires $A > L_h$ (Class-1 stepsizes $\alpha_n = 1/(An)$) with asynchrony condition $\gamma A > L_h$, or $A > L_h$ (Class-2 $\alpha_n = 1/(An\ln n)$), plus noise condition $\mu_\delta < -L_h$ [source:arxiv:2409.03915]. Limitations: finite-dimensional $\mathbb{R}^d$ only; communication delays in distributed settings not yet modeled.

## Throughput-stability trade-offs: empirical landscape

| Method | Setting | Throughput gain | Key stability mechanism | Reported performance |
|--------|---------|-----------------|-------------------------|----------------------|
| A3C [source:arxiv:1602.01783] | 16-core CPU, Atari | Baseline (4 days) | Frequent sync, shared RMSProp, entropy reg. | 112.6% median human-norm |
| IMPALA [source:arxiv:1802.01561] | GPU learner + CPU actors, DMLab-30/Atari-57 | **30× A3C** (250k fps) | V-trace (finite truncation) | 49.4% DMLab-30, 59.7% Atari-57 median |
| A-3PO [source:arxiv:2512.06547] | LLM (Qwen3-8B), math reasoning | **1.8× sync GRPO** | Staleness-interpolated $\pi_{\text{prox}}$ | AIME24 66.67%, MATH500 66.60% |
| Stellaris [source:intellisys:stellaris-staleness-aware-distributed-re] | Serverless multi-learner, MuJoCo/Atari | **41% cost reduction** | Global IS truncation + adaptive $\beta_k$ + $\alpha_c$ | 2.2× rewards vs. SOTA baselines |
| StaleFlow [source:arxiv:2601.12784] | 128-GPU H20, Qwen3-30B-A3B/Qwen2.5-32B | **1.42–2.68×** vs VeRL (sync) | Virtual staleness buffer ($\eta$), waterfall routing | Convergence preserved for $\eta\in[1,3]$ |
| CoPRIS [source:arxiv:2511.05589] | LLM math reasoning, Distill-Qwen/Qwen3 | **1.58–1.94×** vs veRL (sync) | Fixed concurrency $N'$, buffered partial rollouts + IS | 53.68% vs 52.66% pass@1 (Distill-Qwen-7B) |
| AsyncOPD [source:arxiv:2606.24143] | Qwen3-Base (1.7B–8B), distillation | **1.6–3.8×** sync | Forward-KL robustness, IS-corrected reverse-KL, multi-sample MC | Comparable accuracy, variance ↓ to 1.49% ($m=64$) |

**Critical disagreement**: IMPALA's fixed truncation (with $\bar{\rho}<\infty$) accepts a biased fixed point $V^{\pi_\rho}$ [source:arxiv:1802.01561]. This article's analysis suggests that A-3PO's adaptive $\alpha$ also biases toward $\pi_{\text{behav}}$ when $d$ is large (since $\pi_{\text{prox}}\to\pi_{\text{behav}}$), but the bias is *state-dependent* and annealed by the CLIP constraint. This article's analysis suggests that Stellaris avoids bias in the IS ratio by using the global minimum across learners, but introduces bias via *delayed aggregation* (gradients applied later than computed). CoPRIS and AsyncOPD use IS corrections that are asymptotically unbiased but introduce finite-sample variance; CoPRIS buffers partial trajectories across stages while AsyncOPD caches multi-sample MC supports. RIS [source:arxiv:1810.12558] offers a smooth interpolation via $\beta$ rather than hard truncation. No source provides a unified theory comparing these bias sources; the field lacks a bias-variance decomposition for *staleness distributions* (not just mean staleness).

## Current status and trajectory

**Rising for LLM post-training, fading for classic deep RL**. In classic deep RL (Atari, DMLab, MuJoCo), synchronous PPO/IMPALA-style async has been largely superseded by synchronous vectorized environments on GPU (e.g., IsaacGym, Brax) where simulation is fast enough to avoid CPU-GPU transfer bottlenecks. The *asynchronous actor-learner* pattern is now dominant in **LLM RLHF/RLAIF** where rollout generation is orders of magnitude slower than gradient steps, making synchronous training prohibitively idle. Stellaris's serverless aggregation remains **not widely reported** outside the paper; serverless GPU support is still immature (the paper used custom Docker/NVIDIA runtime on EC2) [source:intellisys:stellaris-staleness-aware-distributed-re]. StaleFlow [source:arxiv:2601.12784] and CoPRIS [source:arxiv:2511.05589] represent the new generation of disaggregated LLM RL systems with explicit staleness bounds and concurrency control. AsyncOPD [source:arxiv:2606.24143] demonstrates async distillation at single-node scale; multi-node scaling is future work. On the IS stabilization front, PSIS $\hat{k}$ diagnostics [source:arxiv:1507.02646] are standard in Bayesian workflows; RIS [source:arxiv:1810.12558] and IWMM [source:arxiv:1906.08850] remain primarily in the literature.

## Key takeaways

- **Policy-lag is inevitable at scale**: Decoupling rollouts from learning creates off-policy data; the correction method (V-trace, proximal policy, staleness interpolation, aggregation delay, IS with buffering/caching) defines the algorithm.
- **Truncation choices are bias-variance levers**: IMPALA's $\bar{\rho},\bar{c}$ [source:arxiv:1802.01561], A-3PO's $\alpha(d)$ [source:arxiv:2512.06547], Stellaris's global $\min_i$ [source:intellisys:stellaris-staleness-aware-distributed-re], RIS's $\beta$ [source:arxiv:1810.12558], and PSIS's $\hat{k}$-adaptive smoothing [source:arxiv:1507.02646] represent distinct design points—fixed per-timestep, adaptive per-trajectory, conservative cross-learner, smooth interpolation, and diagnostic-driven tail replacement.
- **LLM async RL is compute-bound on rollouts, not gradients**: A-3PO's >3,000× proximal-policy speedup translates to only 1.8× end-to-end because rollout generation dominates [source:arxiv:2512.06547]. StaleFlow [source:arxiv:2601.12784] and CoPRIS [source:arxiv:2511.05589] target rollout-side efficiency via concurrency control and disaggregation. Future gains require faster rollout engines (speculative decoding, continuous batching) not just faster corrections.
- **Staleness distributions matter more than mean staleness**: All modern methods (A-3PO, Stellaris, StaleFlow, CoPRIS, AsyncOPD) use scalar staleness metrics ($d$, $\bar{\delta}$, $\eta$, version gap). The variance of staleness across the minibatch—and its correlation with advantage magnitude—is **not widely reported** in literature but likely critical for stability.
- **IS diagnostics enable adaptive control**: PSIS $\hat{k}$ [source:arxiv:1507.02646] provides a principled diagnostic for IS estimator reliability ($\hat{k}<0.7$ usable, $\hat{k}>0.7$ unreliable), which could guide dynamic truncation or buffer flushing in async RL but is not yet standard practice.
- **Teacher-cache constraints create new support-mismatch problems**: AsyncOPD [source:arxiv:2606.24143] shows that asynchronous distillation with finite teacher caches requires multi-sample MC or forward-KL to recover missing support, a concern absent in standard actor-critic async RL.
- **Theoretical guarantees for average-reward async SA exist but assume finite dimensions and no communication delays** [source:arxiv:2409.03915], leaving a gap for distributed LLM training with high-dimensional parameters and network latency.

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md) — synchronous and decoupled PPO variants that A-3PO approximates
- [GRPO (Group Relative Policy Optimization)](grpo.md) — synchronous baseline compared against in A-3PO experiments
- [Distributed RL training for LLMs](distributed-rl-training.md) — systems-level view of async rollout infrastructures
- [Rollout generation infrastructure](rollout-generation-infra.md) — vLLM/SGLang engines that create the staleness problem
- [KL regularization in RLHF](kl-regularization.md) — trust-region methods related to proximal policy constraints
- [Entropy and exploration in RL fine-tuning](entropy-and-exploration.md) — A3C's entropy bonus [source:arxiv:1602.01783] as early exploration mechanism
- [Reward model over-optimization](reward-model-overoptimization.md) — failure mode exacerbated by off-policy bias in async settings
- [RL for reasoning models](rl-for-math-and-code.md) — A-3PO's evaluation domain (GSM8K, AIME24, MATH500)
- [Reward modeling for LLMs](reward-modeling.md) — teacher scoring in AsyncOPD distillation
- [Verifiable rewards (RLVR)](verifiable-rewards.md) — CoPRIS evaluation on math benchmarks with verifiable rewards
- [Self-improvement and self-play RL](self-improvement-and-self-play.md) — iterative distillation loops related to AsyncOPD

## References
- [source:intellisys:stellaris-staleness-aware-distributed-re] [Stellaris: Staleness-Aware Distributed Reinforcement Learning with Adaptive Rollout Control](https://intellisys.haow.us/assets/pdf/SC41406.2024.00045.pdf)
- [source:arxiv:2007.12085] [Importance Sampling in Off-Policy Reinforcement Learning: A Survey](https://arxiv.org/abs/2007.12085)
- [source:arxiv:2512.06547] [A-3PO: Accelerating Asynchronous LLM Training with Asynchronous Policy Optimization](https://arxiv.org/html/2512.06547v2)
- [source:arxiv:1602.01783] [Asynchronous Methods for Deep Reinforcement Learning (A3C)](https://arxiv.org/abs/1602.01783)
- [source:arxiv:1802.01561] [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
- [source:arxiv:1507.02646] [Pareto Smoothed Importance Sampling](https://arxiv.org/abs/1507.02646)
- [source:arxiv:2601.12784] [Unleashing Efficient Asynchronous RL Post-Training via Staleness-Constrained Rollout Coordination](https://arxiv.org/abs/2601.12784)
- [source:arxiv:2511.05589] [CoPRIS: Efficient and Stable Reinforcement Learning via Concurrency-Controlled Partial Rollout with Importance Sampling](https://arxiv.org/abs/2511.05589)
- [source:arxiv:2606.24143] [AsyncOPD: How Stale Can On-Policy Distillation Be?](https://arxiv.org/abs/2606.24143)
- [source:arxiv:1810.12558] [Relative Importance Sampling for off-Policy Actor-Critic in Deep Reinforcement Learning](https://arxiv.org/abs/1810.12558)
- [source:arxiv:2409.03915] [Asynchronous Stochastic Approximation with Applications to Average-Reward Reinforcement Learning](https://arxiv.org/abs/2409.03915)
- [source:arxiv:1906.08850] [Implicitly Adaptive Importance Sampling](https://arxiv.org/abs/1906.08850)
