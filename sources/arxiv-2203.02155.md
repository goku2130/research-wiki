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
Large language models (LMs) trained on next-token prediction are misaligned with user intent, frequently producing untruthful, toxic, or unhelpful outputs. Scaling model parameters does not inherently improve this alignment. The authors address this by developing a fine-tuning pipeline that aligns LMs with human preferences, explicitly targeting helpfulness, honesty, and harmlessness.

**Methodology**
The training procedure, named InstructGPT, follows a three-step recipe. First, **Supervised Fine-Tuning (SFT)**: Contractors provide human demonstrations of desired outputs for a diverse prompt distribution (sourced from API usage and labeler-written prompts). A pretrained GPT-3 model is fine-tuned on this demonstration data via supervised learning. Second, **Reward Modeling (RM)**: The SFT model generates multiple outputs per prompt, which contractors rank from best to worst. A reward model is trained on these pairwise comparisons to predict human-preferred outputs. Third, **Reinforcement Learning from Human Feedback (RLHF)**: The trained RM serves as a reward signal. The SFT policy is fine-tuned using Proximal Policy Optimization (PPO) to maximize the RM's reward, incorporating a KL penalty relative to the SFT model to prevent reward over-optimization. To mitigate performance regressions on standard NLP benchmarks, a variant (PPO-ptx) mixes pretraining data gradients into the RL objective.

**Key Formulas**
The reward model is trained using a pairwise comparison loss over $K$ ranked outputs:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, $y_w$ and $y_l$ denote the preferred and less preferred completions, and $D$ is the comparison dataset. The RLHF objective for the PPO-ptx variant combines the reward signal, a KL divergence penalty, and pretraining log-likelihood:
$$\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]$$
Here, $\beta$ and $\gamma$ control the KL penalty and pretraining gradient strength, respectively.

**Quantitative Results**
Human evaluations demonstrate significant alignment gains. Outputs from the 1.3B parameter InstructGPT model are preferred over the 175B GPT-3 baseline despite having 100× fewer parameters. The 175B InstructGPT model is preferred over standard 175B GPT-3 $85 \pm 3\%$ of the time and over few-shot 175B GPT-3 $71 \pm 4\%$ of the time. InstructGPT improves truthfulness, generating truthful and informative answers on TruthfulQA approximately twice as often as GPT-3, while reducing hallucination rates on closed-domain tasks from 41% to 21%. Toxicity decreases by roughly 25% on the RealToxicityPrompts dataset when models are instructed to be respectful. Inter-annotator agreement for preference rankings is $72.6 \pm 1.5\%$ for training labelers and $77.3 \pm 1.3\%$ for held-out labelers. The PPO-ptx modification successfully mitigates most performance regressions on public NLP datasets, though gaps remain on SQuAD, DROP, and translation tasks.

**Stated Limitations**
The authors acknowledge several constraints. InstructGPT models still exhibit simple failures, including hallucination, over-hedging, failure to detect false premises, and difficulty with multiple explicit constraints. Alignment is strictly bounded to the preferences of a specific, non-representative cohort of contractors, researchers, and API customers, rather than broad human values. The models can still generate toxic or biased outputs, particularly when explicitly prompted to do so, and show no significant bias reduction on Winogender or CrowS-Pairs. Furthermore, the reliance on human preference data introduces value judgments influenced by contractor demographics, and the alignment procedure incurs an "alignment tax" that, while mitigated by PPO-ptx, does not fully eliminate performance drops on certain benchmarks.
