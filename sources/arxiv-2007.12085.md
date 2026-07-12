---
id: arxiv:2007.12085
type: paper
title: 'Importance Sampling in Off-Policy Reinforcement Learning: A Survey'
url: https://arxiv.org/abs/2007.12085
retrieved: '2026-07-12'
maturity: comprehensive
topic: async-and-off-policy-rl
---

# Augmentation Adversarial Training for Self-Supervised Speaker Recognition

### Core Problem
The primary objective is to train robust speaker recognition models without the need for speaker labels (self-supervised learning). A significant challenge in this domain is the entanglement of speaker identity with channel information (e.g., noise and reverberation). In contrastive learning frameworks that sample segments from the same utterance, these segments share identical acoustic characteristics, making it difficult for the network to distinguish between the speaker's voice and the environment.

### Method: Augmentation Adversarial Training (AAT)
The authors propose Augmentation Adversarial Training (AAT) to force the network to learn speaker-discriminative embeddings that are invariant to channel characteristics by simulating different environments through data augmentation.

#### Step-by-Step Recipe:
1.  **Batch Formation**: For each mini-batch of $N$ utterances, two non-overlapping speech segments, $\mathbf{x}_{i,1}$ and $\mathbf{x}_{i,2}$, are sampled from each utterance $\mathbf{x}_i$.
2.  **Augmentation and Embedding Extraction**: Three embeddings are generated using a speaker embedding extractor $f$:
    *   $\mathbf{e}_{i,1,1}$: Segment 1 with augmentation type 1 (RIR filter $R_{i,1}$ and noise $N_{i,1}$).
    *   $\mathbf{e}_{i,2,2}$: Segment 2 with augmentation type 2.
    *   $\mathbf{e}_{i,2,1}$: Segment 2 with augmentation type 1.
3.  **Contrastive Training**: The model is trained to minimize the distance between $\mathbf{e}_{i,1,1}$ and $\mathbf{e}_{i,2,2}$ (same speaker, different channel) while maximizing the distance between segments from different utterances.
4.  **Adversarial Training Loop**:
    *   **Discriminator Phase**: An augmentation classifier $g$ is trained to distinguish whether a pair of embeddings shares the same channel characteristics (e.g., $\mathbf{e}_{i,1,1}$ and $\mathbf{e}_{i,2,1}$ share the same augmentation, while $\mathbf{e}_{i,1,1}$ and $\mathbf{e}_{i,2,2}$ do not).
    *   **Embedding Phase**: The extractor $f$ is updated to minimize the speaker loss and maximize the discriminator's uncertainty. This is achieved by placing a **Gradient Reversal Layer (GRL)** between $f$ and $g$, penalizing the network's ability to predict the augmentation type.

### Key Formulas
The embedding is computed as:

$$
\mathbf{e}_{i,j,k}=f(\mathbf{x}_{i,j}*R_{i,k}+N_{i,k})
$$

The **Angular Prototypical (AP)** score uses cosine similarity:

$$
S(\mathbf{e}_{i},\mathbf{e}_{j})=w \times \frac{\mathbf{e}_{i} \cdot \mathbf{e}_{j}}{\|\mathbf{e}_{i}\| \|\mathbf{e}_{j}\|} + b
$$

The **Speaker Loss** ($L_{\text{spk}}$) is a cross-entropy loss:

$$
L_{\mathrm{spk}}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(S(\mathbf{e}_{i,1,1},\mathbf{e}_{i,2,2}))}{\sum_{i^{\prime}=1}^{N}\exp(S(\mathbf{e}_{i,1,1},\mathbf{e}_{i^{\prime},2,2}))}
$$

The **Discriminator Loss** ($L_{\text{dis}}$) is a binary cross-entropy loss:

$$
L_{\mathrm{dis}}=-\frac{1}{2N}\sum_{i=1}^{N}\left(\log(\sigma(g(\mathbf{e}_{i,1,1} \pm \mathbf{e}_{i,2,1})))+\log(1-\sigma(g(\mathbf{e}_{i,1,1} \pm \mathbf{e}_{i,2,2})))\right)
$$

The total objective is:

$$
L = L_{\mathrm{spk}} + \lambda L_{\mathrm{aat}} \quad \text{where } L_{\mathrm{aat}} = -L_{\mathrm{dis}}
$$

### Key Quantitative Results
The model was evaluated on the VoxCeleb1 and VOiCES datasets using Equal Error Rate (EER) and minimum Detection Cost Function (MinDCF).

*   **VoxCeleb1 Performance**: The best model (Angular Prototypical loss + AAT + augmenting both segments with Noise and RIR) achieved an **EER of 8.65%** and a **MinDCF of 0.454**.
*   **VOiCES Performance**: The model demonstrated strong generalization to out-of-domain data, achieving an **EER of 4.96%** and a **MinDCF of 0.356** (with $\lambda=10$).
*   **Comparison to Baselines**: The AAT approach significantly outperformed self-supervised baselines such as GCL (15.26% EER) and unsupervised i-vectors (15.28% EER).
*   **Human Benchmark**: On a subset of 2,000 VoxCeleb1 pairs, the U-ASV model (8.50% EER) far exceeded the performance of both crowdworkers (26.51% EER) and experts (15.77% EER).

### Stated Limitations
The authors note that while the model is robust to environment variations, voice authentication remains susceptible to **spoofing attacks**, such as synthesized speech or voice imitation. Additionally, they highlight the potential risk of over-reliance on unsupervised algorithms without human oversight.
