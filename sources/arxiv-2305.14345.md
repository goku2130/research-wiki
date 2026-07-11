---
id: arxiv:2305.14345
type: paper
title: Are Language Models Really Unbiased?
url: https://arxiv.org/abs/2305.14345
retrieved: '2026-07-11'
maturity: comprehensive
topic: length-and-format-bias
---

**Core Problem**
Existing generative models for 3D humans typically treat clothed bodies as a single, entangled geometric block, severely limiting the expressiveness and controllability required for applications like virtual try-on and avatar customization. While synthetic data approaches attempt to disentangle clothing, they suffer from physical domain gaps and do not scale to diverse objects. The fundamental challenge lies in decomposing objects in physical contact from raw 3D scans without manual segmentation, as joint scanning yields merged geometry that obscures individual components.

**Methodology and Training Recipe**
The proposed framework, NCHO, addresses this through a scalable capture protocol and a three-stage unsupervised training pipeline. First, data acquisition records a single “source human” across diverse poses both without objects ($\mathbf{S}_{sh}$) and with various accessories ($\mathbf{S}_{sh+o}$), supplemented by a diverse target dataset $\mathbf{S}_{th}$ (THuman2.0) for identity generalization. The training proceeds in three phases: (1) $\mathcal{M}_{th}$ and its latent codes $\mathbf{z}_{th}$ are trained on $\mathbf{S}_{th}$ to learn multi-subject forward skinning; (2) $\mathcal{M}_{sh}$ and $\mathbf{z}_{sh}$ are trained on $\mathbf{S}_{sh}$ to model the source human’s base geometry; (3) $\mathcal{M}_{sh}$ is frozen, and an object module $\mathcal{M}_o$, a neural composition module $\mathcal{M}_{comp}$, and latent codes $\mathbf{z}_{sh}, \mathbf{z}_o$ are jointly optimized on $\mathbf{S}_{sh+o}$. During this final stage, objects are defined as residual geometry, computed as $(1 - o_h) \cdot o_{comp}$, where the pretrained human module accounts for body shape and the object module synthesizes the remainder. To prevent contact artifacts, the method replaces naive occupancy max-pooling with an MLP-based composition module that fuses intermediate latent features from both the human and object decoders. Inference involves sampling latent codes for novel identities and objects, composing them via $\mathcal{M}_{comp}$, and re-posing the resulting mesh using learned skinning fields.

**Key Formulas**
The occupancy field in canonical space is defined as $o(\mathbf{x}^c) = O(\mathbf{x}^c, G(\mathbf{z}))$, where $G(\cdot)$ is a tri-plane feature generator conditioned on latent code $\mathbf{z}$. Transformation to posed space utilizes differentiable forward skinning: $\mathbf{x}^d = \sum_{i=1}^{n_b} W_i(N(\mathbf{x}^c, \beta), \mathbf{z}) \cdot \mathbf{B}_i(\beta, \theta) \cdot \mathbf{x}^c$, where $W_i$ and $N$ are identity-conditioned skinning and warping networks, and $\mathbf{B}_i$ represents bone transformations. The human and object modules output occupancy and feature vectors as $(o_h, \mathbf{f}_h) = O_h(\mathbf{x}^c, G_h(\mathbf{z}_h))$ and $(o_o, \mathbf{f}_o) = O_o(\mathbf{x}^c, G_o(\mathbf{z}_o))$. The final compositional occupancy is predicted by the composition module: $o_{comp} = O_{comp}(\mathbf{x}^c, \mathbf{f}_h, \mathbf{f}_o)$. Training employs binary cross-entropy for occupancy, L1/L2 losses for signed distance fields (SDF), and regularization terms to constrain latent codes.

**Quantitative Results**
The dataset comprises 180 scans for $\mathbf{S}_{sh}$, 342 for $\mathbf{S}_{sh+o}$ (covering backpacks, outwear, scarves, and hats), and 526 for $\mathbf{S}_{th}$. A dedicated test set $\mathbf{S}_{unseen+bp}$ contains 343 scans of unseen identities wearing unseen backpacks. Quantitatively, the proposed method achieves a Fréchet Inception Distance (FID) of 51.03, outperforming arithmetic latent manipulation of gDNA (73.81) and naive composition (55.29), though gDNA with object sampling scores 41.71 due to limited identity diversity. In a 50-subject perceptual study, the proposed method was preferred 100% of the time over baselines, and 92.4% of participants selected our generated avatars as distinctly different from the source human. Fitting accuracy on unseen scans also improves, with our method achieving a Chamfer distance of 0.0116 and point-to-surface distance of 0.0099, surpassing gDNA (0.0162/0.0190) and gDNA with objects (0.0218/0.0112).

**Limitations**
The authors note that unsupervised decomposition of thin clothing layers remains challenging due to the limited precision of consumer-grade 3D scanners. Additionally, the current pipeline relies on multi-view 3D scan data, and extending the framework to model humans and objects directly from RGB images is identified as a necessary direction for future work.
