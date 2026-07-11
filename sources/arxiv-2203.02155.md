---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

**Core Problem**
Large language models (LLMs) optimized for next-token prediction are fundamentally misaligned with user intent. Scaling model size does not inherently improve instruction-following, truthfulness, or safety; instead, these models frequently generate unhelpful, toxic, biased, or fabricated outputs. The standard language modeling objective diverges from the desired goal of acting helpfully, honestly, and harmlessly, necessitating targeted alignment techniques for real-world deployment.

**Methodology**
The authors introduce a three-step fine-tuning pipeline to produce InstructGPT. First, **Supervised Fine-Tuning (SFT)** collects demonstration data from ~40 contractors who write prompts and generate desired outputs for a diverse distribution of API and labeler-written instructions. A pretrained GPT-3 model is fine-tuned on this dataset. Second, **Reward Modeling (RM)** gathers comparison data where labelers rank multiple model outputs per prompt. A 6B-parameter reward model is trained to predict human preferences, with rewards normalized so demonstrations yield a mean score of zero. Third, **Reinforcement Learning from Human Feedback (RLHF)** fine-tunes the SFT policy using Proximal Policy Optimization (PPO) to maximize the reward model’s output. A KL penalty relative to the SFT model prevents reward over-optimization. To mitigate performance regressions on public NLP tasks (the "alignment tax"), a variant called PPO-ptx mixes PPO updates with gradients from the original pretraining distribution.

**Key Formulas**
The reward model is trained using a pairwise comparison loss:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, $y_w$ and $y_l$ are preferred and less preferred completions, and $K$ is the number of ranked outputs per prompt. The RLHF objective for the PPO-ptx variant combines the reward signal, a KL divergence penalty, and pretraining log-likelihood:
$$\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]$$

**Quantitative Results**
Human evaluations demonstrate that InstructGPT significantly outperforms GPT-3 baselines. Outputs from the 1.3B parameter InstructGPT model are preferred over the 175B GPT-3 model, despite having 100× fewer parameters. The 175B InstructGPT model is preferred over standard 175B GPT-3 $85 \pm 3\%$ of the time and over few-shot 175B GPT-3 $71 \pm 4\%$ of the time. Truthfulness improves substantially: InstructGPT generates truthful answers approximately twice as often as GPT-3 on TruthfulQA, and hallucination rates on closed-domain tasks drop from 41% to 21%. Toxicity decreases by ~25% on RealToxicityPrompts when models are instructed to be respectful, though bias metrics on Winogender and CrowS-Pairs show no significant improvement. The PPO-ptx modification successfully mitigates performance regressions on datasets like SQuAD, DROP, and HellaSwag. Inter-annotator agreement among labelers is $72.6 \pm 1.5\%$ for training workers and $77.3 \pm 1.3\%$ for held-out workers.

**Stated Limitations**
The alignment process optimizes for the preferences of a specific, non-representative cohort of contractors (primarily English-speaking, based in the US or Southeast Asia) and researchers, raising questions about broader generalizability. InstructGPT models still exhibit simple failures, including overly hedging responses, accepting false premises, struggling with multiple explicit constraints, and occasionally hallucinating. Crucially, the models remain vulnerable to adversarial prompting; when explicitly instructed to generate toxic or biased content, they can produce outputs more harmful than base GPT-3. Furthermore, while the PPO-ptx variant reduces the alignment tax, it does not fully eliminate performance regressions on all benchmarks and may inadvertently propagate biases present in the pretraining data. The approach also does not address how to handle contexts where human values fundamentally disagree or how to scale alignment to diverse demographic groups.
