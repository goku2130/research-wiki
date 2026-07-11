---
title: Distributed RL training for LLMs
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2507.13833
- arxiv:2505.24034
- github:verl-hybridflow-a-flexible-and-efficient
- arxiv:1802.01561
- arxiv:2308.01320
- github:openrlhf-an-easy-fast-and-scalable-rlhf-
- arxiv:2402.03300
- arxiv:1707.06347
open_questions:
- How does AIPO's per-token importance sampling with clipping $\rho\in[2,10]$ compare
  to V-trace's dual truncation $(\bar\rho,\bar c)$ in terms of bias-variance tradeoff
  for LLM policy optimization?
- Does GRPO's group-based advantage estimation interact adversely with DistFlow's
  constrained LPT load balancing (which equalizes item counts but not sequence lengths
  per worker)?
- What is the optimal DP/TP/PP/precision configuration for the generator vs trainer
  in co-located offloading when using GRPO (no critic) vs PPO (with critic)?
- Can DDMA-style zero-copy weight sync be combined with DistFlow's Data Coordinator
  for DP-size transitions, or does the zero-copy requirement constrain parallelism
  flexibility?
---

Distributed RL training for LLMs has evolved from centralized parameter-server architectures into fully decoupled control/data planes that sustain near-linear scaling to hundreds of GPUs. The central tension remains balancing synchronous algorithmic correctness against the throughput gains of asynchronous, off-policy execution.

## Architectural paradigms: rollout/learner split

### Centralized controller bottlenecks
Early RLHF systems (DeepSpeed-Chat, Colossal-AI) used a single controller to orchestrate experience collection, reward computation, and policy updates. DeepSpeed-Chat's Hybrid Engine switches the same model weights between inference (vLLM-style KV-cache, tensor parallelism) and training (ZeRO-3, LoRA) modes [source:arxiv:2308.01320]. This avoids maintaining separate actor/critic copies but forces all tensor movement through the controller, creating O(N) dispatch overhead as GPU count N grows. OpenRLHF similarly relies on a central coordinator with DeepSpeed ZeRO-2/3 and vLLM 0.22.1 for generation [source:github:openrlhf-an-easy-fast-and-scalable-rlhf-].

### Fully distributed control/data decoupling
DistFlow eliminates the central controller by splitting responsibilities: **DAG Workers** execute a declarative Directed Acyclic Graph of computational primitives (generation, reward, update), while a **Data Coordinator** independently manages the data lifecycle [source:arxiv:2507.13833]. The DAG Planner linearizes the logical graph and inserts depth-based dependencies to prevent OOM from colocated same-depth nodes. The Data Coordinator provides:
- **Distributed Dataloader**: each DP rank loads its shard locally
- **Distributed Databuffer**: redistributes tensors when DP size changes between stages
- **Local Caching**: avoids Ray object-store round-trips when DP size is constant
- **Constrained LPT Load Balancing**: assigns variable-length sequences to least-loaded workers while enforcing equal item counts per worker for collective sync
- **Async Double Buffer**: overlaps data prep with GPU compute

LlamaRL adopts a **single-controller event loop** with three decoupled components: **Executors** (self-contained generator/trainer/reward units with independent parallelism/precision), **Communication Channels** (typed links for Broadcast/Scatter/Gather/weight-sync), and a **Controller** that ticks the pipeline [source:arxiv:2505.24034]. This enables **co-located model offloading**: the generator runs on a separate GPU cluster with its own DP/MP/PP and precision (fp8/fp4), fully decoupled from the trainer's parallelism.

### Asynchronous actor-learner with off-policy correction
IMPALA pioneered the decoupled actor-learner pattern for deep RL: actors pull latest policy π, run n-step trajectories with behavior policy μ, push (s,a,r,μ(a|s)) to learner queues; learners batch trajectories and update π via V-trace truncated importance sampling [source:arxiv:1802.01561]. V-trace defines per-timestep weights:

$$
\rho_t = \min(\bar\rho, \pi(a_t|x_t)/\mu(a_t|x_t)), \quad c_i = \min(\bar c, \pi(a_i|x_i)/\mu(a_i|x_i))
$$

The n-step value target:

$$
v_s = V(x_s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \Bigl(\prod_{i=s}^{t-1} c_i\Bigr) \delta_t V, \quad \delta_t V = \rho_t(r_t + \gamma V(x_{t+1}) - V(x_t))
$$

Policy gradient uses $\rho_t \nabla \log \pi(a_t|x_t) (r_t + \gamma v_{t+1} - V(x_t))$. Truncation introduces bias toward a policy $\pi_\rho$ between μ and π but controls variance.

LlamaRL adapts this to LLM RLHF with **Asynchronous Importance weighted Policy Optimization (AIPO)** [source:arxiv:2505.24034]. The trainer updates π using samples from stale actor μ:

$$
\sum_{t=1}^T \min\Bigl(\frac{\pi(y_t|x,y_{<t})}{\mu(y_t|x,y_{<t})}, \rho\Bigr) \cdot A(x,y_{\le t}) \nabla \log \pi(y_t|x,y_{<t})
$$

with clipping constant $\rho \in [2,10]$. Baseline for variance reduction: $v(x,y_{<t}) = \frac{1}{n}\sum_{i=1}^n r(x,y_i)$. Without AIPO, async training shows "sporadic instability" and sudden performance drops on complex data mixtures [source:arxiv:2505.24034].

**Disagreement**: DistFlow explicitly avoids async relaxation, arguing it "compromises model convergence and algorithmic correctness" [source:arxiv:2507.13833]. LlamaRL demonstrates 2.52×–10.7× speedups with AIPO corrections and claims parity or better quality vs synchronous baselines on MATH-500/GSM8K [source:arxiv:2505.24034]. IMPALA's V-trace shows bias-variance tradeoff controlled by $\bar\rho, \bar c$ [source:arxiv:1802.01561]. No source directly compares AIPO vs V-trace on LLM tasks; Z would settle it.

## Sharding strategies for multi-model RLHF

### Model placement and parallelism composition
RLHF requires 3–4 model copies: policy (actor), reference, reward, and optionally critic (value). DeepSpeed-Chat's Hybrid Engine **time-multiplexes** a single model copy between inference and training modes, using ZeRO-3 for training memory and lightweight KV-cache + TP for inference [source:arxiv:2308.01320]. This reduces peak memory but serializes generation and update.

OpenRLHF uses **ZeRO-2/3 with vLLM** for separate actor/critic/reward models, marking `SparseMoeBlock` as leaf modules for MoE support (Mixtral, DeepSeek) [source:github:openrlhf-an-easy-fast-and-scalable-rlhf-]. Loss normalization aggregates token counts across DP ranks and gradient-accumulation windows to compute exact global mean per micro-batch.

LlamaRL **decouples parallelism per executor**: generator uses (DP_g, TP_g, PP_g, fp8) while trainer uses (DP_t, TP_t, PP_t, bf16) [source:arxiv:2505.24034]. This allows generator to run with higher DP (more throughput) and lower precision without constraining trainer sharding.

DistFlow's Data Coordinator handles **DP-size transitions** between stages via the Distributed Databuffer: when stage i has DP_i and stage i+1 has DP_{i+1}, tensors are redistributed across the new DP groups [source:arxiv:2507.13833]. Local caching avoids this when DP_i = DP_{i+1}.

### Weight synchronization at scale
LlamaRL's **Distributed Direct Memory Access (DDMA)** enables zero-copy GPU→GPU weight sync via NVLink/InfiniBand, bypassing CPU memory [source:arxiv:2505.24034]. For 70B model: 1.15s sync vs 111.65s in OpenRLHF (parameter-server style). For 405B across thousands of GPUs: ~2s for terabyte-scale weights. DeepSpeed-Chat does not report dedicated weight-sync benchmarks; its Hybrid Engine updates weights in-place during mode switch.

## PPO at scale: memory, throughput, and alternatives

### PPO mechanics and multi-model overhead
Standard PPO clips the probability ratio $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$ [source:arxiv:1707.06347]:

$$
L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \Bigl[ \min\bigl(r_t(\theta)\hat A_t,\; \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\bigr) \Bigr]
$$

Combined with value loss $L^{\text{VF}} = (V_\theta(s_t)-V_t^{\text{targ}})^2$ and entropy bonus $S[\pi_\theta](s_t)$. For LLMs, this requires:
- Actor forward/backward (policy)
- Critic forward/backward (value)
- Reference forward (KL)
- Reward forward
- Often: separate old-policy forward for ratio (or recompute)

DeepSpeed-Chat reports OPT-13B: 10.8h RLHF on 8×A100-40G; OPT-175B: <1 day on 64×A100-80G [source:arxiv:2308.01320]. Throughput drops for "gigantic models" due to memory-limited batch size.

### GRPO: eliminating the critic
GRPO removes the value network by estimating advantages from group statistics [source:arxiv:2402.03300]. For each question q, sample G outputs $\{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}$. Token-level advantage:

$$
\hat A_{i,t} = \frac{r_i - \text{mean}(\mathbf r)}{\text{std}(\mathbf r)} \quad \text{(outcome supervision)}
$$

or process supervision: $\hat A_{i,t} = \sum_{j: \text{index}(j)\ge t} \tilde r_i^{\text{index}(j)}$. Objective:

$$
\mathcal J_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\{o_i\}} \Bigl[ \frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\Bigl(\frac{\pi_\theta(o_{i,t}|q,o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q,o_{i,<t})}\hat A_{i,t},\; \text{clip}(\cdot,1-\epsilon,1+\epsilon)\hat A_{i,t}\Bigr) - \beta D_{\text{KL}}(\pi_\theta\|\pi_{\text{ref}}) \Bigr]
$$

KL estimated via unbiased estimator: $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$.

DistFlow benchmarks GRPO vs PPO: GRPO speedup up to 2.62× over verl baseline (vs 1.64× for PPO), attributed to "superior handling of data-intensive workloads" [source:arxiv:2507.13833]. LlamaRL supports GRPO as one of its algorithm executors [source:arxiv:2505.24034].

**Disagreement**: GRPO's group-based advantage assumes i.i.d. sampling from π_old; DistFlow's constrained LPT load balancing enforces equal item counts per worker, which may correlate sequence lengths within groups. No source analyzes this interaction. DeepSeekMath reports GRPO matches PPO on math reasoning (51.7% MATH vs proprietary baselines) [source:arxiv:2402.03300], but no large-scale distributed GRPO vs PPO ablation exists in sources.

## Scaling laws and system efficiency

### Throughput scaling
| Framework | Model | GPUs | Speedup vs Baseline | Scaling Efficiency |
|-----------|-------|------|---------------------|-------------------|
| DistFlow (PPO) | Qwen-2.5 7B/32B/72B | up to 512 | 1.09–1.64× | 90.1% (7B), 93.9% (32B), 91.8% (72B) |
| DistFlow (GRPO) | Qwen-2.5 7B/32B/72B | up to 512 | up to 2.62× | — |
| LlamaRL | 8B/70B/405B | thousands | 2.52× / 3.98× / 10.7× | — |
| DeepSpeed-Chat | OPT-13B/66B/175B | 8–64 | 6–19× vs Colossal-AI | — |
| IMPALA | — | — | 250k fps (30× A3C) | — |

DistFlow's scaling efficiency formula:

$$
\text{Scaling Efficiency} = \frac{T_2/T_1}{N_2/N_1} \times 100\%
$$

where $T$ = throughput, $N$ = GPU count [source:arxiv:2507.13833]. Near-linear scaling to 512 GPUs attributed to control/data decoupling and LPT load balancing.

LlamaRL's super-linear speedup (10.7× at 405B) comes from async pipelining eliminating "idle bubbles" [source:arxiv:2505.24034]. DeepSpeed-Chat's 7.5× scale increase on single node (50B vs 6.7B) comes from ZeRO-3 + Hybrid Engine [source:arxiv:2308.01320].

### Long-context scaling
DistFlow shows speedup increasing with context: 7B model, 1.48× (8k) → 2.03× (64k) vs baseline [source:arxiv:2507.13833]. Baseline (verl) OOMs at 72B/32k; DistFlow succeeds. LlamaRL's generator offloading allows independent context-length scaling on inference cluster.

### Memory and OOM robustness
DistFlow's DAG Planner inserts sequential dependencies for same-depth nodes to prevent colocated OOM [source:arxiv:2507.13833]. Local caching avoids Ray object-store memory pressure. LlamaRL's DDMA avoids CPU staging memory for weight sync. DeepSpeed-Chat notes 175B efficiency limited by per-GPU memory restricting batch size [source:arxiv:2308.01320]. OpenRLHF's ZeRO-3 leaf-module fix (disabling hybrid detection) recovered gradients for ~390/417 inner params in Qwen3.5-9B [source:github:openrlhf-an-easy-fast-and-scalable-rlhf-].

## Communication optimization

### Collective patterns
- **Weight broadcast/sync**: LlamaRL DDMA (GPU→GPU, zero-copy) [source:arxiv:2505.24034]; DeepSpeed-Chat in-place (Hybrid Engine) [source:arxiv:2308.01320]; OpenRLHF parameter server via DeepSpeed [source:github:openrlhf-an-easy-fast-and-scalable-rlhf-]; DistFlow Data Coordinator redistribution [source:arxiv:2507.13833].
- **Experience collection**: DistFlow constrained LPT + async double buffer [source:arxiv:2507.13833]; LlamaRL async channels with Scatter/Gather [source:arxiv:2505.24034]; IMPALA actor→learner queues [source:arxiv:1802.01561].
- **Gradient/all-reduce**: All use NCCL/DeepSpeed ZeRO collectives; no source details topology-aware scheduling beyond DP-group alignment.

### Overlap and pipelining
DistFlow: async double buffer overlaps data prep with GPU compute [source:arxiv:2507.13833]. LlamaRL: controller event loop triggers concurrent executor steps; generator and trainer run simultaneously [source:arxiv:2505.24034]. IMPALA: learner parallelizes time-independent ops (conv over all timesteps) + XLA/cuDNN [source:arxiv:1802.01561]. DeepSpeed-Chat: Hybrid Engine mode switch serializes inference/training; no overlap reported.

## Current status and trajectory

**Distributed control/data decoupling (DistFlow-style) is rising** as the default for synchronous scaling beyond 128 GPUs, with 90%+ scaling efficiency demonstrated to 512 GPUs [source:arxiv:2507.13833]. **Asynchronous actor-learner with importance weighting (LlamaRL/AIPO) is rising for maximum throughput at extreme scale (405B, thousands of GPUs)**, showing 10.7× speedup with claimed quality parity [source:arxiv:2505.24034]. **Centralized Hybrid Engine (DeepSpeed-Chat) is fading for large-scale training** due to controller bottleneck and memory-limited batch size at 175B+ [source:arxiv:2308.01320], but remains common for single-node/small-cluster RLHF. **GRPO is rising as the default PPO alternative** for math/code reasoning, eliminating critic memory and showing 2.6× distributed speedup [source:arxiv:2402.03300][source:arxiv:2507.13833]. **DDMA-style zero-copy weight sync is becoming standard** for >100B models; parameter-server approaches (OpenRLHF's 111s for 70B) are not competitive [source:arxiv:2505.24034]. **V-trace/IMPALA-style off-policy correction is not widely adopted in LLM RLHF**; AIPO is the only reported async correction in sources, and its bias-variance tradeoff vs V-trace is not benchmarked.

## Key takeaways

- **Control/data decoupling** (DistFlow) achieves 90%+ scaling efficiency to 512 GPUs by separating DAG-based control from parallelism-aware data movement with LPT load balancing and local caching [source:arxiv:2507.13833].
- **Async actor-learner with AIPO** (LlamaRL) delivers 10.7× speedup at 405B by running generator/trainer concurrently on decoupled clusters with DDMA zero-copy weight sync (~2s for TB-scale) [source:arxiv:2505.24034].
- **GRPO eliminates the critic** and uses group-normalized advantages, reducing memory and achieving 2.6× distributed speedup over PPO baselines [source:arxiv:2402.03300][source:arxiv:2507.13833].
- **Hybrid Engine time-multiplexing** (DeepSpeed-Chat) enables single-copy RLHF but bottlenecks at 175B+ due to memory-limited batch size and serialized inference/training [source:arxiv:2308.01320].
- **ZeRO-3 leaf-module handling** is critical for MoE/hybrid architectures; OpenRLHF's fix recovered gradients for 390/417 frozen params in Qwen3.5 [source:github:openrlhf-an-easy-fast-and-scalable-rlhf-].
- **V-trace truncation** (IMPALA) controls off-policy variance at cost of bias toward intermediate policy $\pi_\rho$; not directly compared to AIPO in LLM setting [source:arxiv:1802.01561].
- **Long-context scaling** favors decoupled architectures: DistFlow speedup grows from 1.48× (8k) to 2.03× (64k); LlamaRL offloads context to inference cluster [source:arxiv:2507.13833][source:arxiv:2505.24034].

## Related topics

- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Async and off-policy RL](async-and-off-policy-rl.md)
- [Rollout generation infrastructure](rollout-generation-infra.md)
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [Policy gradient methods for LLMs](policy-gradient-methods.md)
- [Verifiable rewards (RLVR)](verifiable-rewards.md)
- [RL for math and code](rl-for-math-and-code.md)

## References
- [source:arxiv:2507.13833] [DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Alignment](https://arxiv.org/html/2507.13833v2)
- [source:arxiv:2505.24034] [LlamaRL: A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Training](https://arxiv.org/html/2505.24034v2)
- [source:github:verl-hybridflow-a-flexible-and-efficient] [verl: HybridFlow - A Flexible and Efficient RL Post-Training Framework](https://github.com/verl-project/verl)
- [source:arxiv:1802.01561] [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
- [source:arxiv:2308.01320] [DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales](https://arxiv.org/abs/2308.01320)
- [source:github:openrlhf-an-easy-fast-and-scalable-rlhf-] [OpenRLHF: An Easy, Fast, and Scalable RLHF Framework for Large Language Models](https://github.com/OpenRLHF/OpenRLHF)
- [source:arxiv:2402.03300] [GRPO: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [source:arxiv:1707.06347] [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
