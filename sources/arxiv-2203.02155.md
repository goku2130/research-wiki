---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

**Core Problem**
Large language models (LMs) trained on next-token prediction are fundamentally misaligned with user intent, frequently generating untruthful, toxic, biased, or unhelpful outputs. The authors address this alignment gap by fine-tuning GPT-3 to follow a broad distribution of written instructions while adhering to human preferences for helpfulness, honesty, and harmlessness, demonstrating that alignment can be achieved without proportionally scaling model size.

**Methodology**
The training pipeline executes three sequential steps. First, **supervised fine-tuning (SFT)** is performed on a dataset of approximately 13,000 prompts (derived from OpenAI API submissions and labeler-written instructions) paired with human demonstrations of desired outputs. Second, **reward modeling** collects comparison data where contractors rank multiple model outputs per prompt. A 6B-parameter reward model is trained to predict human preference, with rewards normalized to a mean of zero for the demonstration data. Third, **reinforcement learning from human feedback (RLHF)** optimizes the SFT policy against the reward model using Proximal Policy Optimization (PPO). A per-token KL penalty relative to the SFT model prevents reward over-optimization. To mitigate performance regressions on public NLP benchmarks (the "alignment tax"), the authors introduce a pretraining mix variant (PPO-ptx) that incorporates pretraining data gradients during RL.

**Key Formulas**
The reward model is trained using a pairwise comparison loss over $K$ ranked completions:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward, $y_w$ and $y_l$ denote the preferred and less preferred completions, and $D$ is the comparison dataset. The RL optimization maximizes a combined objective:
$$\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]$$
Here, $\beta$ controls the KL penalty strength, and $\gamma$ scales the pretraining log-likelihood term.

**Quantitative Results**
Human evaluations demonstrate that InstructGPT significantly outperforms GPT-3 baselines. Outputs from the 1.3B parameter InstructGPT model are preferred over the 175B GPT-3 baseline, despite having 100× fewer parameters. The 175B InstructGPT model is preferred over the 175B GPT-3 baseline $85 \pm 3\%$ of the time, and over few-shot 175B GPT-3 $71 \pm 4\%$ of the time. InstructGPT improves truthfulness, generating truthful and informative answers on TruthfulQA approximately twice as often as GPT-3, while reducing hallucination rates on closed-domain tasks from $41\%$ to $21\%$. Toxicity decreases by roughly $25\%$ on the RealToxicityPrompts dataset when models are prompted to be respectful. The PPO-ptx variant successfully mitigates performance regressions on SQuAD, DROP, and HellaSwag. Computationally, SFT and RLHF fine-tuning require $4.9$ and $60$ petaflops/s-days respectively, a fraction of the $3,640$ petaflops/s-days used for GPT-3 pretraining.

**Limitations**
The authors acknowledge several constraints. The models are aligned to the preferences of a specific, non-representative cohort of contractors and researchers, with inter-annotator agreement around $73\%$. InstructGPT still exhibits simple failures, including accepting false premises, overly hedging responses, and struggling with multiple explicit constraints. It does not significantly reduce bias on Winogender or CrowS-Pairs, and can generate more toxic outputs than GPT-3 when explicitly instructed to do so. Furthermore, the alignment process generalizes poorly to inputs where humans disagree, and the reliance on API prompt distributions means the models are optimized for a narrow slice of real-world usage rather than universal human values.
