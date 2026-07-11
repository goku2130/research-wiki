---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

**Core Problem**
Large language models (LMs) optimized for next-token prediction are fundamentally misaligned with user intent, frequently generating untruthful, toxic, or unhelpful outputs. The authors demonstrate that scaling model parameters does not inherently improve alignment; instead, fine-tuning via human feedback effectively aligns LMs with human intentions across a broad distribution of instruction-following tasks.

**Methodology**
The training pipeline, termed InstructGPT, follows a three-step recipe:
1. **Supervised Fine-Tuning (SFT):** Labelers write or curate instruction prompts (sourced from API usage and labeler-generated examples) and provide demonstration outputs. A pretrained GPT-3 model is fine-tuned via supervised learning on this demonstration dataset.
2. **Reward Model Training:** A comparison dataset is collected where human labelers rank multiple model outputs per prompt from best to worst. A reward model (RM) is trained on these pairwise comparisons to predict human preference. To prevent overfitting from highly correlated comparisons within a single prompt, all $\binom{K}{2}$ comparisons are batched together during training rather than treated as independent samples.
3. **Reinforcement Learning via PPO:** The SFT policy is fine-tuned using Proximal Policy Optimization (PPO) to maximize the RM's scalar reward. A per-token KL penalty relative to the SFT model is applied to prevent reward over-optimization. To mitigate performance regressions on standard NLP benchmarks (the "alignment tax"), the authors introduce a pretraining mix variant (PPO-ptx) that incorporates pretraining data gradients into the RL objective.

**Key Formulas**
The reward model is trained using a cross-entropy loss over ranked comparisons:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, $y_w$ and $y_l$ denote the preferred and less preferred completions, and $D$ is the comparison dataset. The RL optimization maximizes a combined objective:
$$\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]$$
where $\beta$ and $\gamma$ control the KL penalty and pretraining loss coefficients, respectively.

**Quantitative Results**
Human evaluations on held-out API prompts show that outputs from the 1.3B parameter InstructGPT model are preferred over the 175B GPT-3 baseline despite having over 100x fewer parameters. The 175B InstructGPT model is preferred over standard 175B GPT-3 $85 \pm 3\%$ of the time and over few-shot 175B GPT-3 $71 \pm 4\%$ of the time. InstructGPT models exhibit improved truthfulness, generating truthful and informative answers on TruthfulQA approximately twice as often as GPT-3, and reducing hallucination rates on closed-domain tasks from 41% to 21%. Toxicity is reduced by approximately 25% when models are prompted to be respectful, though bias metrics on Winogender and CrowS-Pairs show no significant improvement. Training compute is substantially lower than pretraining: 4.9 and 60 petaflops/s-days for the 175B SFT and PPO-ptx models, respectively, compared to 3,640 for GPT-3 pretraining.

**Stated Limitations**
The alignment process is explicitly tied to the preferences of a specific, non-representative cohort of contractors and researchers, raising concerns about demographic and cultural bias in the training signal. The models still exhibit simple failures, including accepting false premises in prompts, over-hedging responses, and struggling with complex multi-constraint instructions. Furthermore, InstructGPT will follow explicitly harmful or biased instructions if requested, indicating that harmlessness is not fully enforced. While mixing pretraining data mitigates performance regressions on public NLP datasets, the PPO-ptx variant still lags behind GPT-3 on tasks like DROP, SQuADv2, and machine translation. Finally, the authors note that public instruction-tuning datasets (FLAN, T0) fail to replicate the performance gains achieved by human preference data, highlighting a gap between benchmark tasks and real-world API usage.
