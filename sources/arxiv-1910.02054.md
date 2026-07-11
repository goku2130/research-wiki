---
id: arxiv:1910.02054
type: paper
title: 'ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models'
url: https://arxiv.org/abs/1910.02054
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

# ZeRO: Memory Optimizations Toward Training Trillion-Parameter Models

### Core Problem
Training deep learning models with billions or trillions of parameters is constrained by device memory. Standard Data Parallelism (DP) is memory-inefficient because it replicates model states (optimizer states, gradients, and parameters) across all devices. Conversely, Model Parallelism (MP) improves memory efficiency by partitioning states but suffers from poor scaling efficiency due to fine-grained computation and high communication overhead, especially across node boundaries. Pipeline Parallelism (PP) introduces further complexities regarding functionality, convergence, and "pipeline bubbles."

### Method: The ZeRO Optimizer
The Zero Redundancy Optimizer (ZeRO) eliminates memory redundancies in DP and MP while maintaining high computational granularity and low communication volume. It is divided into two primary optimization suites: **ZeRO-DP** and **ZeRO-R**.

#### ZeRO-DP (Model State Optimization)
ZeRO-DP partitions model states across data-parallel processes instead of replicating them. It is implemented in three cumulative stages:
1.  **Optimizer State Partitioning ($P_{os}$):** Optimizer states are divided into $N_d$ equal partitions (where $N_d$ is the DP degree). Each process only updates its assigned partition and uses an all-gather operation to synchronize updated parameters.
2.  **Gradient Partitioning ($P_{os+g}$):** Gradients are reduced only on the process responsible for the corresponding parameter partition (a Reduce-Scatter operation).
3.  **Parameter Partitioning ($P_{os+g+p}$):** Parameters are partitioned across processes. During forward and backward propagation, required parameters are broadcast from the owning process and discarded immediately after use.

#### ZeRO-R (Residual State Optimization)
ZeRO-R targets activations, temporary buffers, and memory fragmentation:
*   **Partitioned Activation Checkpointing ($P_a$):** Removes activation replication in MP by partitioning activation checkpoints across GPUs and using all-gather to reconstruct them on demand. These can be further offloaded to CPU memory ($P_{a+cpu}$).
*   **Constant Size Buffers ($C_B$):** Replaces model-size-dependent fused buffers with performance-efficient constant-size buffers to prevent memory overhead in very large models.
*   **Memory Defragmentation ($M_D$):** Pre-allocates contiguous memory chunks for gradients and activation checkpoints to prevent Out-of-Memory (OOM) errors caused by fragmentation.

### Key Formulas
For a model with $\Psi$ parameters using mixed-precision Adam training, the total memory requirement for model states is:

$$
\text{Total Memory} = 2\Psi \text{ (fp16 params)} + 2\Psi \text{ (fp16 grads)} + K\Psi \text{ (fp32 optimizer states)}
$$

Where $K=12$ for Adam, resulting in $16\Psi$ bytes.

**Memory Reduction:**
*   $P_{os}$ reduces memory from $4\Psi + K\Psi$ to $4\Psi + \frac{K\Psi}{N_d}$ ($\approx 4\times$ reduction).
*   $P_{os+g}$ reduces memory to $2\Psi + \frac{14\Psi}{N_d}$ ($\approx 8\times$ reduction).
*   $P_{os+g+p}$ reduces memory to $\frac{16\Psi}{N_d}$ (linear reduction with $N_d$).

**Communication Volume:**
*   $P_{os+g}$ communication volume is $2\Psi$, identical to baseline DP.
*   $P_{os+g+p}$ communication volume is $3\Psi$, a $1.5\times$ increase over baseline DP.

### Quantitative Results
The authors evaluated **ZeRO-100B** (a subset using $P_{os+g}$ and ZeRO-R) on a cluster of 400 NVIDIA V100 GPUs:
*   **Model Scale:** Successfully trained models up to **170B parameters**, an $8\times$ increase over the state-of-the-art (SOTA) baseline.
*   **Performance:** Achieved an aggregate throughput of **15 Petaflops** (approx. 38 TFlops per GPU), representing up to a **$10\times$ speedup** over SOTA for the same model size.
*   **Scalability:** Observed super-linear speedup between 64 and 400 GPUs.
*   **Usability:** Enabled training of models up to **13B parameters** using only DP (without MP/PP), whereas standard DP OOMs at 1.4B parameters.
*   **SOTA Model:** Powered **Turing-NLG** (17B parameters), achieving record-breaking accuracy (Webtext-103 perplexity of 10.21).
*   **Theoretical Limit:** Analysis indicates ZeRO can fit a **1 Trillion parameter model** on 1,024 GPUs.

### Limitations
*   **Compute Gap:** While ZeRO solves the memory bottleneck, current hardware compute capacity is insufficient for trillion-parameter models; training such a model could take over a year.
*   **CPU Offloading Trade-off:** Using $P_{a+cpu}$ to offload activations to CPU can decrease performance for some model sizes (e.g., 60B) due to the overhead of data movement, though it is necessary for the largest models (e.g., 170B) to avoid OOM.
