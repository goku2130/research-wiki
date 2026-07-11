---
id: arxiv:2512.06727
type: paper
title: 'KV-CAR: KV Cache Compression using Autoencoders and KV Reuse in Large Language
  Models'
url: https://arxiv.org/abs/2512.06727
retrieved: '2026-07-11'
maturity: comprehensive
topic: rollout-generation-infra
---

# KV-CAR: KV Cache Compression using Autoencoders and KV Reuse

KV-CAR is an architecture-agnostic framework designed to reduce the memory footprint of the key-value (KV) cache during the autoregressive decoding phase of Large Language Models (LLMs).

### Core Problem
In LLM inference, the KV cache stores key and value tensors to avoid redundant computations. However, the cache size grows linearly with sequence length, batch size, and embedding dimension, often exceeding the memory requirements of the model parameters themselves. This creates a memory bottleneck that limits the maximum achievable batch size and context window length on GPU hardware.

### Methodology
KV-CAR employs two primary complementary techniques to reduce dimensional and structural redundancy:

**1. Layer-wise Autoencoder Compression**
The framework introduces lightweight per-layer autoencoders to compress the embedding dimension of KV vectors.
*   **Encoder:** Reduces the original embedding dimension $D$ to a compact latent dimension $d$ ($d \ll D$). The architecture consists of two fully connected layers with batch normalization and Leaky ReLU activation.
*   **Decoder:** Mirrors the encoder, reconstructing the $d$-dimensional latent vector back to $D$ dimensions upon retrieval from the cache.
*   **Workflow:** KV vectors are compressed by the encoder before being stored in High-Bandwidth Memory (HBM) and expanded by the decoder before being used in attention computations.

**2. Similarity-Guided KV Reuse**
To eliminate structural redundancy, KV-CAR identifies redundant attention heads across adjacent layers.
*   **Metric:** The L1-norm is used to calculate the similarity between KV matrices of consecutive layers.
*   **Mechanism:** If the similarity between heads in layer $N$ and layer $N-1$ exceeds an empirically determined threshold, the heads in layer $N$ are replaced by those from layer $N-1$.

**3. Optional Quantization**
Int8 quantization can be stacked on top of the autoencoder. Vectors are quantized after the encoder and dequantized before the decoder.

### Key Formulas
The total memory required for the KV cache is defined as:

$$
\text{KV\_Cache\_Size} = 2 \times P \times N_{\text{layers}} \times d_{\text{model}} \times L_{\text{seq}} \times B
$$

Where $P$ is bytes per element, $N_{\text{layers}}$ is the number of layers, $d_{\text{model}}$ is the embedding dimension, $L_{\text{seq}}$ is sequence length, and $B$ is batch size.

The Int8 quantization process follows:

$$
\text{scale} = \frac{255}{\max(x) - \min(x)}
$$

$$
\text{zeropoint} = -\text{round}(\text{scale} \times \min(x)) - 128
$$

$$
X_{\text{quant}} = \text{round}(\text{scale} \times x + \text{zeropoint})
$$

$$
X_{\text{dequant}} = \frac{X_{\text{quant}} - \text{zeropoint}}{\text{scale}}
$$

### Training Recipe
The framework uses a staged training pipeline to maintain model fidelity:

**Autoencoder Training (Algorithm 1):**
1.  **Independent Training:** For each layer, the base model is frozen while the layer-specific autoencoder is trained. The loss function is a combination of the model's cross-entropy (CE) loss and a scaled L1 reconstruction loss between the original and predicted KV vectors.
2.  **Joint Optimization:** All autoencoders are integrated, and a final fine-tuning stage is performed. All parameters are frozen except the encoders, and the model is optimized using the sum of scaled L1 losses across all encoders plus the CE loss.

**KV Reuse Training (Algorithm 2):**
1.  **Similarity Analysis:** KV heads are collected over one epoch to compute the average inter-layer L1 norm.
2.  **Head Selection:** Heads exceeding the empirical similarity threshold are marked for reuse.
3.  **Fine-tuning:** The model is fine-tuned using a combined loss of CE and the scaled L1 norm between actual and reused KV values.

### Quantitative Results
Evaluations were conducted on GPT-2 (774M parameters) and TinyLLaMA (1.1B parameters) using WikiText, C4, PIQA, and Winogrande.

*   **Memory Reduction:** KV-CAR achieved up to **47.85%** KV-cache memory reduction (GPT-2 on WikiText) using a combination of autoencoders and head replacement.
*   **Model Fidelity:** 
    *   **TinyLLaMA:** Maintained stable perplexity on WikiText with up to 11 layers compressed (25% savings) and achieved 50% memory reduction on Winogrande with 22 layers compressed.
    *   **GPT-2:** Achieved 41.6% savings on WikiText, PIQA, and Winogrande by compressing 10 layers with negligible accuracy loss.
*   **System Performance (NVIDIA A40):** Reduced cache footprints enabled longer sequences. For a batch size of 64, 75% compression increased the maximum sequence length by 5,248 tokens compared to the baseline.

### Limitations
The authors note a clear trade-off between accuracy retention and memory savings. Tolerance to compression varies significantly by dataset; for example, the C4 dataset was less tolerant, with perplexity increasing noticeably after compressing more than 4–6 layers. Additionally, excessive head replacement (e.g., replacing all KV heads) leads to a significant increase in perplexity.
