---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

**Core Problem**
Large language models (LLMs) optimized for next-token prediction are fundamentally misaligned with user intent. Scaling model size does not inherently improve instruction-following, truthfulness, or safety. These models frequently generate untruthful, toxic, biased, or unhelpful outputs because their pretraining objective diverges from the desired deployment goal of being helpful, honest, and harmless.

**Methodology**
The authors implement a three-step fine-tuning pipeline to align GPT-3 architectures with human preferences, producing *InstructGPT*:
1. **Supervised Fine-Tuning (SFT):** Contractors generate prompts and provide demonstration outputs. A pretrained GPT-3 model is fine-tuned on this demonstration dataset via supervised learning to establish a baseline instruction-following policy.
2. **Reward Model (RM) Training:** Labelers rank multiple model-generated outputs per prompt. An RM is trained to predict human-preferred outputs using a pairwise comparison loss. The RM is normalized so that labeler demonstrations yield a mean reward of zero.
3. **Reinforcement Learning from Human Feedback (RLHF):** The SFT policy is optimized against the RM using Proximal Policy Optimization (PPO). A KL-divergence penalty relative to the SFT model prevents reward over-optimization. To mitigate performance regressions on public NLP benchmarks, the authors introduce *PPO-ptx*, which mixes pretraining distribution gradients into the RL objective.

**Key Formulas**
The reward model is trained by maximizing the likelihood of human-preferred pairs. Given $K$ completions per prompt, the loss function is:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, $y_w$ is the preferred completion, and $D$ is the comparison dataset.

The RLHF objective for the PPO-ptx variant combines the reward signal, a KL penalty, and pretraining likelihood gradients:
$$\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]$$
Here, $\beta$ and $\gamma$ control the KL penalty and pretraining gradient strength, respectively.

**Quantitative Results**
Human evaluations on held-out API prompts demonstrate significant alignment gains. Outputs from the 1.3B parameter InstructGPT model are preferred over the 175B GPT-3 baseline despite having 100$\times$ fewer parameters. The 175B InstructGPT model is preferred over standard 175B GPT-3 $85 \pm 3\%$ of the time and over few-shot prompted GPT-3 $71 \pm 4\%$ of the time. Truthfulness improves substantially: InstructGPT generates truthful answers on TruthfulQA roughly twice as often as GPT-3, and hallucination rates on closed-domain tasks drop from 41% to 21%. Toxicity decreases by approximately 25% when models are prompted to be respectful, though bias metrics (Winogender, CrowS-Pairs) show no significant improvement. The fine-tuning compute cost is modest relative to pretraining: 60 petaflops/s-days for the 175B PPO-ptx model versus 3,640 for GPT-3 pretraining. Inter-annotator agreement for preference rankings is $72.6 \pm 1.5\%$ for training labelers and $77.3 \pm 1.3\%$ for held-out labelers.

**Stated Limitations**
The authors acknowledge several critical constraints. First, InstructGPT still exhibits simple failures, including accepting false premises, overly hedging responses, and struggling with multiple explicit constraints. Second, alignment is strictly bounded to the preferences of a specific, non-representative cohort of contractors, researchers, and API customers, rather than universal human values. Third, models remain vulnerable to misuse; when explicitly instructed to generate toxic or biased content, InstructGPT can produce outputs more harmful than GPT-3. Finally, while the PPO-ptx modification reduces the "alignment tax," performance regressions persist on datasets like SQuAD, DROP, and machine translation. The approach also requires extensive human-in-the-loop data collection and does not yet guarantee safety in high-stakes deployment contexts.
