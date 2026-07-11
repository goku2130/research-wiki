---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback (InstructGPT)
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

The core problem addressed is the misalignment of large language models (LLMs) trained on next-token prediction objectives with human user intent. Despite scaling parameter counts, base LLMs frequently generate untruthful, toxic, biased, or unhelpful outputs. The authors propose aligning models with user intentions—defined as being helpful, honest, and harmless—through a three-step reinforcement learning from human feedback (RLHF) pipeline.

The method proceeds as follows. First, supervised fine-tuning (SFT) is performed on GPT-3 using a dataset of approximately 13,000 prompts. Human contractors provide demonstrations of desired outputs for both labeler-written and OpenAI API prompts. Second, a reward model (RM) is trained on a comparison dataset of roughly 33,000 prompts. For each prompt, labelers rank $K=4$ to $9$ model outputs from best to worst. The RM predicts human preference using a specific loss formulation. Third, the SFT policy is optimized against the RM using Proximal Policy Optimization (PPO). A KL penalty relative to the SFT model prevents reward over-optimization, and a pretraining data mix (PPO-ptx) is optionally added to mitigate performance regressions on standard NLP benchmarks.

The reward model loss is defined as:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, $y_w$ and $y_l$ denote preferred and less preferred completions, and $D$ is the comparison dataset. The combined RL objective for PPO-ptx models is:
$$\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]$$
where $\beta$ and $\gamma$ control the KL penalty and pretraining gradient strength, respectively.

Quantitatively, human evaluations show that outputs from the 1.3B parameter InstructGPT model are preferred over the 175B GPT-3 baseline despite having over 100x fewer parameters. The 175B InstructGPT model is preferred over standard 175B GPT-3 $85 \pm 3\%$ of the time and over few-shot 175B GPT-3 $71 \pm 4\%$ of the time. On the TruthfulQA benchmark, InstructGPT generates truthful and informative answers approximately twice as often as GPT-3. For closed-domain tasks, hallucination rates drop from 41% in GPT-3 to 21% in InstructGPT. When prompted to be respectful, InstructGPT produces roughly 25% fewer toxic outputs than GPT-3, though it shows no significant improvement on bias benchmarks like Winogender and CrowS-Pairs. Training compute is modest relative to pretraining: 4.9 petaflops/s-days for SFT and 60 petaflops/s-days for PPO-ptx, compared to 3,640 for GPT-3 pretraining. Inter-annotator agreement among labelers is $72.6 \pm 1.5\%$ for training workers and $77.3 \pm 1.3\%$ for held-out workers.

The authors acknowledge several limitations. The alignment process targets the specific preferences of a narrow group of contractors (primarily English-speaking individuals in the US or Southeast Asia) and OpenAI researchers, rather than broad human values. Models still exhibit simple mistakes, including failing to follow instructions, hallucinating facts, overly hedging responses, and accepting false premises. Furthermore, InstructGPT will follow explicitly harmful instructions if requested, and bias mitigation remains ineffective. While the PPO-ptx modification reduces performance regressions on public NLP datasets, it does not fully eliminate the "alignment tax," and generalization to diverse user groups or contexts with conflicting human preferences requires further investigation.
