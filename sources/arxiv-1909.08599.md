---
id: arxiv:1909.08599
type: paper
title: Learning from Human Preferences
url: https://arxiv.org/abs/1909.08599
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

**Core Problem**
High-performing semantic segmentation models suffer from excessive computational costs and parameter counts, rendering them unsuitable for real-time deployment. Existing lightweight alternatives typically downsample inputs or prune channels to reduce latency, which discards fine spatial details and weakens contextual encoding. Furthermore, naive decoder designs that rely on simple bilinear upsampling produce coarse segmentation boundaries, while complex decoders that attempt to preserve multi-scale information introduce prohibitive inference latency. The central challenge is designing a lightweight architecture that simultaneously encodes rich multi-scale context, preserves spatial details, and maintains real-time inference speed.

**Methodology**
The proposed Feature Pyramid Encoding Network (FPENet) resolves this trade-off through a streamlined U-shaped encoder-decoder pipeline. The encoder replaces standard convolutions with Feature Pyramid Encoding (FPE) blocks deployed across all stages. Each FPE block follows a depth-separable inverted bottleneck recipe: a $1 \times 1$ expansion convolution quadruples the channel dimension, the expanded feature map is partitioned into four subsets, and each subset is processed by parallel depthwise dilated convolutions with dilation rates of 1, 2, 4, and 8. The output of each dilated branch is sequentially added to the input of the subsequent branch, concatenated, and fused via a final $1 \times 1$ pointwise convolution that restores the original channel count. Residual connections are applied when input and output dimensions match. The decoder employs Mutual Embedding Upsample (MEU) modules to efficiently fuse high-level semantic features with low-level spatial details. For each MEU module, both feature streams pass through $1 \times 1$ convolutions. High-level features undergo channel attention via global average pooling, a $1 \times 1$ convolution, and ReLU activation to generate a channel-wise weighting map for low-level features. Simultaneously, low-level features pass through spatial attention (channel-wise average pooling, $1 \times 1$ convolution, ReLU) to produce a pixel-wise attention map that weights the upsampled high-level features. The two weighted streams are fused via element-wise addition. The network maintains a total downsampling rate of 8, utilizes long skip connections in intermediate stages to encourage signal propagation, and concludes with a $1 \times 1$ pixel-level classifier.

**Key Formulas**
Training employs the Adam optimizer with a poly learning rate schedule defined as:
$$lr = \text{init } lr \times \left( 1 - \frac{\text{epoch}}{\text{max\_epoch}} \right)^{\text{power}}$$
where the power is set to 0.9. The optimization objective minimizes the mean cross-entropy error across all pixels.

**Quantitative Results**
On the Cityscapes test set, FPENet achieves a 68.0% mean Intersection-over-Union (mIoU) with only 0.4 million parameters, operating at 102 FPS on an NVIDIA TITAN V GPU using a $1024 \times 512$ input. On the CamVid dataset, it attains 65.4% mIoU and 89.6% global accuracy with the same parameter count. Ablation studies confirm that the four-branch FPE structure improves mIoU by 6.6% over a single-branch baseline, while dilation rates of 1, 2, 4, 8 outperform 1, 2, 3, 4. The complete MEU module yields a 1.7% mIoU gain over configurations lacking attention mechanisms. Compared to existing real-time methods, FPENet matches ESPNet’s parameter count while delivering a 7.7% higher Cityscapes mIoU, and surpasses BiSeNet1 and ICNet in inference speed by factors of 14 and 19, respectively, with only marginal accuracy drops.

**Stated Limitations**
The authors note that excessively large receptive fields (e.g., increasing stage depth to $q=11$) yield diminishing returns because the dilation field exceeds the feature map dimensions, hindering efficient feature extraction. Consequently, the architecture requires careful tuning of encoder depth to balance accuracy and computational load. Additionally, reported inference speeds are hardware-specific (NVIDIA TITAN V) and degrade significantly at higher input resolutions (e.g., dropping to 55 FPS at $1536 \times 768$), indicating a strict accuracy-speed trade-off dependent on input scaling. The method also relies on downsampled inputs for Cityscapes training, which may limit fine-grained boundary recovery without additional refinement.
