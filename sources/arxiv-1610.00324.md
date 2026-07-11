---
id: arxiv:1610.00324
type: paper
title: Maximum Entropy Exploration
url: https://arxiv.org/abs/1610.00324
retrieved: '2026-07-11'
maturity: comprehensive
topic: entropy-and-exploration
---

The core problem addressed is the steep computational and memory overhead incurred by deep convolutional neural networks (DNNs) as network depth increases, which prolongs training and complicates deployment. Prior low-precision approaches typically sacrifice classification accuracy to gain efficiency. This work targets maintaining or improving accuracy while drastically reducing compute complexity through extremely low-precision weights and dynamic sparsity exploitation.

The methodology employs a 2-bit ternary weight quantization scheme. Weights are mapped to $\{-1, 0, 1\}$ using a threshold $w_{th}$, effectively zeroing out small-magnitude weights and introducing dynamic sparsity. To recover accuracy lost during quantization, a multi-step training recipe is applied: (1) pre-initialize the network with full-precision weights for approximately 15 iterations before switching to low-precision mode for the remaining ~75 iterations; (2) retain full-precision computation exclusively for the first convolutional layer to preserve input information; (3) implement aggressive learning rate reduction when training error plateaus; (4) apply activation regularization to suppress noise and increase sparsity; and (5) adjust ReLU thresholds (e.g., to 0.01) during later training epochs to further induce sparsity. During execution, zero-skipping logic bypasses arithmetic operations on zero-valued weights and activations.

The weight quantization is formally defined as:

$$
W_{ter}(i, j, k, l) = \begin{cases} 1 & : \text{if } W(i,j,k,l) > w_{th} \\ -1 & : \text{if } W(i,j,k,l) < -1 \times w_{th} \\ 0 & : \text{otherwise} \end{cases}
$$

Quantitative evaluations on ImageNet demonstrate that a 2-bit ResNet-152 achieves 76.6% Top-1 and 93% Top-5 accuracy, matching the 2015 competition winner within 1.3% while requiring $\sim 3\times$ fewer floating-point operations and yielding a model size $\sim 7\times$ smaller than comparable full-precision networks. Sparsity analysis reveals that only 16% of forward-pass and 33% of backward-pass operations involve non-zero operands, enabling a potential 3–6$\times$ acceleration via zero-skipping. The proposed Deep Learning Accelerator Core (DLAC), a 2D grid of processing elements with dedicated buffers and zero-skipping control, synthesizes to 2.2 mm$^2$ (single-precision) and 1.09 mm$^2$ (half-precision) in 14 nm ASIC flow. Operating at 500 MHz, DLAC sustains up to 5K FLOP/cycle (2.78K average), translating to 2.5 TFLOP/s (1.34 TFLOP/s) and an effective performance density exceeding 1 TFLOP/s/mm$^2$ for single-precision and $\sim 2$ TFLOP/s/mm$^2$ for half-precision modes.

Stated limitations and constraints include the accelerator’s primary architectural focus on convolutional layers, with explicit support for single-precision operations during training but limited emphasis on other network operations. The reported performance density relies on pure ASIC synthesis without optimized macroblocks, meaning silicon-validated metrics are not provided. Additionally, accuracy recovery in low-precision regimes is highly dependent on the prescribed training recipes; without pre-initialization, first-layer precision retention, and adaptive regularization, quantization initially degrades accuracy by approximately 3% on datasets like Cifar10. Finally, the dynamic sparsity required for maximal zero-skipping efficiency is inherently depth-dependent, with deeper network layers exhibiting higher sparsity and thus yielding greater speedups on the proposed hardware.
