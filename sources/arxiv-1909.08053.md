---
id: arxiv:1909.08053
type: paper
title: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
url: https://arxiv.org/abs/1909.08053
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

# Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

### Core Problem
Training transformer-based language models with billions of parameters is constrained by the memory limits of modern hardware accelerators. Memory pressure arises not only from the model weights but also from optimizer states (e.g., ADAM) and activations. While data parallelism and activation checkpointing exist, they require the model to fit on a single worker. Existing model parallelism frameworks often necessitate custom compilers or significant code rewrites, creating a barrier to efficient scaling.

### Method: Intra-Layer Model Parallelism
Megatron-LM implements a simple, efficient intra-layer model parallel approach in PyTorch that exploits the inherent structure of the transformer architecture. The method is orthogonal to pipeline parallelism and focuses on partitioning tensor operations across GPUs.

#### 1. MLP Block Parallelization
The MLP block consists of a GEMM followed by a GeLU nonlinearity. To avoid synchronization points before the nonlinearity, Megatron-LM employs a column-parallel approach for the first GEMM:
*   **First GEMM (Column Parallel):** The weight matrix $A$ is split along its columns $A = [A_1, A_2]$. This allows the GeLU nonlinearity to be applied independently on each GPU:

$$
[Y_1, Y_2] = [\text{GeLU}(XA_1), \text{GeLU}(XA_2)]
$$

*   **Second GEMM (Row Parallel):** The second GEMM is split along its rows, taking the output of the GeLU layer directly without communication.
*   **Communication:** A single all-reduce operation is performed in the forward pass ($g$ operator) and a conjugate all-reduce in the backward pass ($f$ operator).

#### 2. Self-Attention Block Parallelization
*   **Attention Heads:** The GEMMs for Query ($Q$), Key ($K$), and Value ($V$) are partitioned in a column-parallel fashion. This ensures that the matrix multiplication for each attention head is performed locally on one GPU.
*   **Output Linear Layer:** The subsequent GEMM is parallelized along its rows, consuming the parallel attention output without immediate communication.

#### 3. Embedding and Loss Optimization
*   **Input Embeddings:** The weight matrix is split column-wise along the vocabulary dimension, requiring an all-reduce after the embedding lookup.
*   **Output Embeddings:** To avoid communicating massive logits (batch size $\times$ sequence length $\times$ vocabulary size), the parallel GEMM is fused with the cross-entropy loss. This reduces communication to scalar losses.

#### 4. Hybrid Parallelism and Redundancy
The system combines intra-layer model parallelism with data parallelism. To minimize communication, the authors duplicate the computation of layer normalization, dropout, and residual connections across all GPUs in a model-parallel group.

### Key Quantitative Results
The authors demonstrated the scalability of Megatron-LM using up to 512 NVIDIA V100 GPUs:

*   **Computational Throughput:** Sustained **15.1 PetaFLOPs** across the application.
*   **Scaling Efficiency:** Achieved **76% scaling efficiency** compared to a strong single-GPU baseline (which sustained 39 TeraFLOPs, or 30% of theoretical peak).
*   **Model Scaling:**
    *   **GPT-2 (8.3B parameters):** Achieved SOTA results on WikiText103 (10.8 perplexity vs. 15.8 SOTA) and LAMBADA (66.5% accuracy vs. 63.2% SOTA).
    *   **BERT (3.9B parameters):** Achieved SOTA results on the RACE dataset (90.9% accuracy vs. 89.4% SOTA).
*   **Weak Scaling:** The 8.3B parameter model using 8-way model parallelism achieved 77% linear scaling (model only) and 74% scaling (model + data parallel).

**Perplexity Formula used for evaluation:**

$$
PPL = \exp \left(- \frac {1}{T _ {o}} \sum_ {t} ^ {T} \log P (t | 0: t - 1)\right)
$$

### Stated Limitations
*   **Hardware Memory Ceiling:** Training models exceeding 16 billion parameters would require more memory than is available within a single DGX-2H box (16 GPUs), necessitating a transition to hybrid intra-layer and inter-layer (pipeline) model parallelism.
*   **Hyperparameter Sensitivity:** Increasing the number of attention heads can slightly decrease scaling efficiency due to smaller GEMM sizes and increased elements in the self-attention softmax.
*   **Optimizer Efficiency:** Future scaling will require improvements in the memory footprint and efficiency of optimizers.
