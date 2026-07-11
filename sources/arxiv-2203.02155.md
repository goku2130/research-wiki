---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human preferences
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-hacking
---

Large language models optimized for next-token prediction are fundamentally misaligned with user intent, frequently generating untruthful, toxic, or unhelpful outputs. Scaling model parameters does not inherently resolve this misalignment. To address this, the authors fine-tune GPT-3 to act helpfully, honestly, and harmlessly using a three-step reinforcement learning from human feedback (RLHF) pipeline. First, human labelers provide demonstrations of desired outputs for a distribution of prompts, which are used to fine-tune a pretrained GPT-3 model via supervised learning (SFT). Second, labelers rank multiple model outputs per prompt, creating a comparison dataset used to train a reward model (RM) that predicts human preference. Third, the SFT policy is optimized against the RM using Proximal Policy Optimization (PPO), incorporating a KL penalty relative to the SFT model to prevent reward over-optimization. To mitigate performance regressions on standard NLP benchmarks (the "alignment tax"), a variant (PPO-ptx) incorporates pretraining data gradients into the RL objective.

The reward model is trained using a pairwise comparison loss over $K$ ranked outputs:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, $y_w$ is the preferred completion, and $D$ is the comparison dataset. The RL optimization maximizes a combined objective:
$$\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]$$
where $\beta$ controls the KL penalty strength and $\gamma$ scales the pretraining log-likelihood term.

Human evaluations on held-out API prompts demonstrate that outputs from the 1.3B parameter InstructGPT model are preferred over the 175B GPT-3 baseline. The 175B InstructGPT model is preferred over standard GPT-3 $85 \pm 3\%$ of the time and over few-shot GPT-3 $71 \pm 4\%$ of the time. InstructGPT demonstrates improved truthfulness, generating truthful and informative answers approximately twice as often as GPT-3 on TruthfulQA, and reducing hallucination rates on closed-domain tasks from $41\%$ to $21\%$. When prompted to be respectful, InstructGPT generates roughly $25\%$ fewer toxic outputs than GPT-3 on RealToxicityPrompts, though it shows no significant bias reduction on Winogender or CrowS-Pairs. Inter-annotator agreement among labelers reached $72.6 \pm 1.5\%$, generalizing to held-out labelers at $77.3 \pm 1.3\%$.

The authors acknowledge several limitations. InstructGPT aligns to a narrow, non-representative cohort of contractors and researchers rather than universal human values. The models still exhibit simple failures, including accepting false premises, over-hedging, and struggling with multiple explicit constraints. While toxicity decreases under respectful prompts, InstructGPT can generate more toxic outputs than GPT-3 when explicitly instructed to do so, revealing that the models follow user instructions even when harmful. Furthermore, the alignment procedure incurs performance regressions on certain public NLP datasets (e.g., DROP, SQuAD, translation), which the PPO-ptx variant partially mitigates but does not fully eliminate. The authors note that public instruction-tuning datasets (FLAN, T0) fail to replicate the performance gains achieved through human preference data, underscoring the gap between benchmark tasks and real-world API usage.
