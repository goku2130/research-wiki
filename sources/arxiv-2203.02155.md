---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

**Core Problem**
Large language models (LMs) optimized for next-token prediction are fundamentally misaligned with user intent. Scaling model size does not inherently improve instruction-following, truthfulness, or safety. Instead, these models frequently generate untruthful, toxic, biased, or unhelpful outputs, and often ignore explicit user constraints. The standard autoregressive objective diverges from the desired deployment goal of producing helpful, honest, and harmless responses.

**Methodology**
The authors align GPT-3 architectures through a three-stage fine-tuning pipeline:
1. **Supervised Fine-Tuning (SFT):** Contractors curate prompts and provide demonstration outputs. The base GPT-3 model is fine-tuned on this dataset via supervised learning to internalize instruction-following behavior.
2. **Reward Modeling (RM):** For each prompt, multiple model outputs are sampled and ranked by contractors. A reward model is trained on these pairwise comparisons to predict human preference.
3. **Reinforcement Learning from Human Feedback (RLHF):** The trained RM serves as a scalar reward signal. The SFT policy is optimized using Proximal Policy Optimization (PPO). A KL penalty relative to the SFT model prevents reward over-optimization. To mitigate performance regressions on standard NLP benchmarks, the authors introduce a pretraining data mix (PPO-ptx), which blends RL gradients with standard next-token prediction gradients from the original pretraining distribution.

**Key Formulas**
The reward model is trained using a cross-entropy loss over all pairwise comparisons within a batch of $K$ responses:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

where $r_\theta(x,y)$ is the scalar reward, $y_w$ and $y_l$ are the winning and losing completions, and $D$ is the comparison dataset. The RLHF objective combines the reward signal, a KL penalty, and the pretraining loss:

$$
\text{objective} (\phi) = \mathbb{E}_{(x, y) \sim D_{\pi_{\phi}^{\mathrm{RL}}}} \left[ r_{\theta} (x, y) - \beta \log \left(\pi_{\phi}^{\mathrm{RL}} (y \mid x) / \pi^{\mathrm{SFT}} (y \mid x)\right) \right] + \gamma \mathbb{E}_{x \sim D_{\text{pretrain}}} \left[ \log \left(\pi_{\phi}^{\mathrm{RL}} (x)\right) \right]
$$

Here, $\beta$ controls the KL penalty strength and $\gamma$ weights the pretraining gradient contribution.

**Quantitative Results**
Human evaluations on held-out API prompts show that outputs from the 1.3B InstructGPT model are preferred over the 175B GPT-3 baseline, despite using 100× fewer parameters. The 175B InstructGPT model is preferred over standard 175B GPT-3 $85 \pm 3\%$ of the time and over few-shot 175B GPT-3 $71 \pm 4\%$ of the time. InstructGPT models significantly improve truthfulness, generating truthful and informative answers on TruthfulQA approximately twice as often as GPT-3. On closed-domain tasks, hallucination rates drop from 41% to 21%. When prompted to be respectful, InstructGPT generates roughly 25% fewer toxic outputs than GPT-3, though bias metrics on Winogender and CrowS-Pairs show no significant improvement. The PPO-ptx modification successfully mitigates most performance regressions on public NLP datasets like SQuAD and HellaSwag, though some degradation persists on DROP and translation tasks. Inter-annotator agreement for preference rankings reached $72.6 \pm 1.5\%$ for training labelers and $77.3 \pm 1.3\%$ for held-out labelers.

**Stated Limitations**
The authors explicitly note that alignment is strictly bounded by the preferences of a specific, non-representative group of contractors (primarily English-speaking individuals in the US and Southeast Asia) and the researchers designing the labeling instructions. The models remain imperfect: they still generate toxic or biased content when explicitly instructed to do so, frequently assume false premises in prompts, overly hedge simple questions, and struggle with complex multi-constraint instructions. Furthermore, the alignment procedure introduces an "alignment tax," causing performance drops on certain public NLP benchmarks that are only partially mitigated by mixing pretraining data. The approach also does not generalize perfectly to out-of-distribution contexts, and the authors caution that aligning to a narrow reference group may not capture broader human values or account for stakeholder disagreements.
