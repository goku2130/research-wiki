---
id: arxiv:2305.11451
type: paper
title: Adversarial Sycophancy
url: https://arxiv.org/abs/2305.11451
retrieved: '2026-07-11'
maturity: comprehensive
topic: sycophancy-and-misgeneralization
---

**Core Problem**
Training deep learning models for long surgical video analysis, particularly automatic surgical activity recognition, is severely constrained by the high cost and poor scalability of manual clinical annotation. While self-supervised learning (SSL) reduces annotation dependency, standard Masked Autoencoder (MAE) architectures applied to video often sample tokens from redundant or uninformative background regions using random, tube, or frame masking strategies. This indiscriminate sampling hinders the learning of robust, transferable representations in the surgical domain, where spatio-temporal dynamics are critical.

**Method/Recipe Step-by-Step**
SurgMAE modifies the MAE paradigm to prioritize high spatio-temporal information during pre-training through the following pipeline:
1. **Patch Embedding:** An input video clip $V$ of dimensions $T \times 3 \times H \times W$ is processed through a 3D convolutional layer with a patch size of $2 \times 16 \times 16$, generating $N = T/2 \times H/16 \times W/16$ token embeddings of dimension $d$.
2. **High Spatio-Temporal Token Sampling:** The model computes the Euclidean distance between adjacent frame embeddings ($X_{i,k}$ and $X_{i+1,k}$) at identical 2D spatial positions. Tokens exhibiting high distance values are retained based on a predefined masking ratio, while tokens from low-information or static regions are discarded.
3. **Positional Encoding & Encoding:** Separable spatial and temporal positional embeddings are added to the sampled tokens $\mathbf{X}_v$. The enriched sequence is passed through a Vision Transformer (ViT) encoder to extract latent representations $\mathbf{Z}_v$.
4. **Decoding & Reconstruction:** The latent embeddings are concatenated with learnable masked tokens $\mathbf{z}_m$, positional embeddings are reapplied, and the sequence is fed into a decoder to reconstruct the original masked patches $\hat{\mathbf{V}}$.
5. **Optimization:** The network is trained to minimize the reconstruction error between the original and predicted patches.

**Key Formula**
The training objective utilizes a mean squared error formulation between normalized RGB pixel values:
$$\mathcal{L} = \frac{1}{\omega} \sum_{p \in \omega} ||\mathbf{V}(p) - \hat{\mathbf{V}}(p)||_2$$
where $p$ is the token index, $\omega$ is the set of masked tokens, and $||\cdot||_2$ denotes the L2 norm.

**Key Quantitative Results**
SurgMAE was evaluated on surgical and general video benchmarks under varying data regimes:
- **OR-AR (5% labeled data):** Achieves 68.91% mAP, outperforming random (66.58%), tube (65.57%), and frame (63.44%) masking baselines.
- **OR-AR (100% labeled data):** Reaches 95.60% mAP, slightly trailing standard MAE random masking (96.30%).
- **OR-ARv2 (full data):** Attains 93.11% mAP, surpassing all compared masking strategies.
- **UCF-101:** Achieves 92.1% top-1 accuracy, exceeding VideoMAE's 91.2%.
- **Ablation Optima:** Peak performance is observed with a 90% masking ratio, a decoder depth of 4 blocks, and 1600 pre-training epochs. MSE loss with per-patch normalization consistently outperforms L1 loss and unnormalized variants.

**Stated Limitations**
The authors identify several constraints. Pre-training for extended durations (e.g., 1600 epochs) improves downstream fine-tuning performance but incurs higher computational costs and longer training times. Additionally, while SurgMAE excels in low-data regimes, standard random masking slightly outperforms it when full labeled datasets are available, suggesting potential performance saturation in data-rich settings. The study also notes that smaller datasets like Cataract-101 pose significant challenges for Vision Transformer pre-training, often necessitating transfer learning from large-scale external datasets (e.g., Kinetics-400) to achieve competitive results.
