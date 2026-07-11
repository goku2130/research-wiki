---
id: arxiv:2307.09288
type: paper
title: 'Llama 2: Open Foundation and Fine-Tuned Chat Models'
url: https://arxiv.org/abs/2307.09288
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

**Core Problem**
Closed-source large language models (LLMs) achieve high usability and safety through opaque, computationally intensive fine-tuning pipelines, leaving open-source models lacking comparable alignment. The authors address this capability gap by developing, documenting, and releasing LLaMA 2, a family of open pretrained and fine-tuned chat models (up to 70B parameters), alongside a fully transparent methodology for alignment and safety mitigation to enable reproducible, responsible community development.

**Methodology and Recipe**
The development pipeline proceeds through five sequential stages:
1. **Pretraining:** Base LLaMA 2 models are trained on 2 trillion tokens (a 40% increase over LLaMA 1) with doubled context length (4k tokens) and Grouped-Query Attention (GQA) for the 34B and 70B variants. Architectural components include RMSNorm, SwiGLU activations, and rotary positional embeddings.
2. **Supervised Fine-Tuning (SFT):** The pipeline bootstraps with public instruction data but rapidly transitions to high-quality, vendor-annotated data (27,540 examples). Prompts and answers are concatenated with a separator token; loss is zeroed on prompt tokens, and the model is fine-tuned for two epochs.
3. **Reward Modeling:** Two distinct reward models (Helpfulness and Safety) are trained on ~1.4 million Meta-curated preference comparisons augmented with open-source datasets. Models are initialized from pretrained checkpoints and optimized using a margin-enhanced binary ranking loss that weights preference strength.
4. **Reinforcement Learning with Human Feedback (RLHF):** Iterative refinement (V1–V5) combines Rejection Sampling (sampling $K$ outputs, selecting the highest-reward candidate, and fine-tuning) with Proximal Policy Optimization (PPO). PPO optimizes a reward function penalizing divergence from the base policy via a KL term. To preserve multi-turn system instructions, Ghost Attention (GAtt) is applied during fine-tuning by masking loss on intermediate turns while synthetically injecting constraints.
5. **Safety Alignment:** Safety is integrated via adversarial SFT demonstrations, safety-specific RLHF, and Safety Context Distillation (prefixing prompts with safety instructions and distilling the context into the model). Iterative red-teaming and preference data collection ensure reward models remain on-distribution.

**Key Formulas**
The reward model optimization employs a margin-enhanced ranking loss:
\[
\mathcal {L} _ {\text { ranking }} = - \log (\sigma (r _ {\theta} (x, y _ {c}) - r _ {\theta} (x, y _ {r}) - m (r)))
\]
where $r_{\theta}$ outputs a scalar score, $y_c$ and $y_r$ are chosen/rejected responses, and $m(r)$ scales with preference rating strength. The PPO objective maximizes expected reward subject to a KL penalty:
\[
\arg \max _ {\pi} \mathbb {E} _ {p \sim \mathcal {D}, g \sim \pi} [ R (g \mid p) ]
\]
The final reward function combines safety ($R_s$) and helpfulness ($R_h$) models, applying a 0.15 threshold for safety filtering and whitening via logit reversal:
\[
R (g \mid p) = \text { WHITEN } ( \text { LOGIT } (R _ {c} (g \mid p)) ) - \beta D _ {K L} (\pi_ {\theta} (g \mid p) \parallel \pi_ {0} (g \mid p))
\]

**Quantitative Results**
Pretraining consumed 3.3 million GPU hours with an estimated 539 tCO₂eq footprint (fully offset). The LLaMA 2 70B base model achieves 68.9 on MMLU, 51.2 on BBH, and 56.8 on GSM8K, outperforming LLaMA 1 65B by approximately 5 and 8 points, respectively. The Helpfulness and Safety reward models achieve 63.2% and 64.5% accuracy on internal test sets, surpassing baselines including GPT-4. In human evaluations across ~4,000 prompts, LLaMA 2-CHAT 70B secures a 36% win rate and 31.5% tie rate against ChatGPT, while outperforming open-source competitors (e.g., >75% win rate vs. Vicuna-33B). Safety RLHF significantly improves long-tail safety scores without degrading helpfulness, though false refusal rates rise to ~0.05% on standard prompts and higher on adversarial borderline cases.

**Stated Limitations**
The authors note that human evaluations are inherently noisy, subjective, and limited to a ~4k prompt set lacking coding and complex reasoning tasks; only final turns are rated. Safety benchmarks may be biased toward LLaMA 2-CHAT due to evaluation standards. Models are evaluated exclusively in English and cannot cover all deployment scenarios. Base models retain toxicity and demographic biases from unfiltered pretraining data, necessitating extensive safety tuning before production use. Additionally, early RLHF iterations exhibited capability regression (e.g., poetry composition), and the GAtt implementation remains a vanilla technique requiring further refinement.
