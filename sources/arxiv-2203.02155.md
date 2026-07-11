---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

**Core Problem**
Large language models (LLMs) optimized for next-token prediction are misaligned with user intent, frequently generating untruthful, toxic, biased, or unhelpful outputs. Scaling model parameters does not inherently resolve this misalignment. The authors address this by fine-tuning GPT-3 to follow a broad distribution of written instructions, aiming to align models to be helpful, honest, and harmless through reinforcement learning from human feedback (RLHF).

**Methodology**
The training pipeline follows a three-step recipe. First, **Supervised Fine-Tuning (SFT)**: Contractors collect demonstration data using labeler-written and API-submitted prompts, providing human demonstrations of desired outputs. A pretrained GPT-3 model is fine-tuned on this dataset via supervised learning. Second, **Reward Modeling (RM)**: Labelers rank multiple model outputs per prompt. A 6B-parameter reward model is trained on these pairwise comparisons to predict human preference. Rewards are normalized so demonstrations yield a mean score of zero. Third, **Reinforcement Learning (RLHF)**: The SFT policy is optimized against the reward model using Proximal Policy Optimization (PPO). A per-token KL penalty relative to the SFT model prevents reward over-optimization. To mitigate performance regressions on public NLP benchmarks (the "alignment tax"), the authors introduce **PPO-ptx**, which mixes pretraining data gradients into the RL objective.

**Key Formulas**
The reward model is trained using a margin-based loss over $K$ ranked completions:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward, $y_w$ and $y_l$ are preferred and less preferred completions, and $D$ is the comparison dataset. The RLHF objective for PPO-ptx combines the reward signal, a KL penalty, and pretraining log-likelihood:
$$\text{objective}(\phi) = E_{(x,y)\sim D_{\pi_{\phi}^{\mathrm{RL}}}} \left[ r_{\theta} (x,y) - \beta \log \left(\pi_{\phi}^{\mathrm{RL}} (y|x) / \pi^{\mathrm{SFT}} (y|x)\right) \right] + \gamma E_{x\sim D_{\text{pretrain}}} \left[ \log \left(\pi_{\phi}^{\mathrm{RL}} (x)\right) \right]$$
where $\beta$ and $\gamma$ control the KL penalty and pretraining gradient strength, respectively.

**Quantitative Results**
Human evaluations on held-out API prompts show that outputs from the 1.3B InstructGPT model are preferred over the 175B GPT-3 baseline despite having over 100× fewer parameters. The 175B InstructGPT model is preferred $85 \pm 3\%$ of the time against standard GPT-3 and $71 \pm 4\%$ of the time against few-shot GPT-3. Truthfulness improves significantly: InstructGPT generates truthful answers twice as often as GPT-3 on TruthfulQA, and hallucination rates on closed-domain tasks drop from $41\%$ to $21\%$. Toxicity decreases by approximately $25\%$ on RealToxicityPrompts when models are prompted to be respectful, though bias metrics on Winogender and CrowS-Pairs show no significant improvement. Training compute remains modest relative to pretraining: SFT requires $4.9$ petaflops/s-days and PPO-ptx requires $60$ petaflops/s-days for the 175B model, compared to $3,640$ petaflops/s-days for initial GPT-3 pretraining.

**Limitations**
The authors acknowledge several constraints. InstructGPT models still commit simple errors, including failing to detect false premises in instructions, overly hedging responses, and struggling with multiple explicit constraints. Crucially, the models follow user instructions even when they request harmful or biased outputs, generating more toxic text than GPT-3 when explicitly prompted to be biased. Alignment is strictly tied to the preferences of a specific group of contractors, researchers, and API customers, rather than universal human values. The labeler pool is predominantly English-speaking and based in the US or Southeast Asia, with inter-annotator agreement at $\sim 73\%$. Furthermore, while PPO-ptx mitigates performance regressions on most public NLP datasets, it does not fully eliminate them on tasks like SQuAD, DROP, and machine translation. Generalization to broader user demographics and scenarios with high human disagreement remains unverified.
