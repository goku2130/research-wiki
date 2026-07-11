---
id: arxiv:2203.02155
type: paper
title: Training Language Models to Follow Instructions with Human Feedback (InstructGPT)
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

The core problem addressed by InstructGPT is that scaling large language models (LLMs) does not inherently improve their alignment with user intent. Models optimized for next-token prediction frequently generate untruthful, toxic, or unhelpful outputs, failing to follow instructions helpfully and safely. To resolve this misalignment, the authors propose a three-step fine-tuning pipeline. First, they collect demonstration data comprising labeler-written prompts and prompts from the OpenAI API, where contractors provide desired output demonstrations. This dataset fine-tunes GPT-3 via supervised learning (SFT). Second, they collect comparison data by having labelers rank multiple model outputs per prompt. This ranking dataset trains a reward model (RM) to predict human-preferred responses. Third, they optimize the SFT policy using Proximal Policy Optimization (PPO) to maximize the RM's scalar reward, incorporating a KL penalty relative to the SFT model to prevent reward over-optimization. To mitigate performance regressions on standard NLP tasks, they introduce PPO-ptx, which mixes pretraining data gradients into the RL objective.

The reward model is trained using a pairwise comparison loss:
$$\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]$$
where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, $y_w$ and $y_l$ denote the preferred and less preferred completions, and $K$ is the number of ranked responses. The RL optimization maximizes a combined objective:
$$\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]$$
Here, $\beta$ controls the KL penalty strength, and $\gamma$ scales the pretraining log-likelihood term.

Human evaluations demonstrate that InstructGPT significantly outperforms GPT-3 baselines. Outputs from the 1.3B parameter InstructGPT model are preferred over the 175B GPT-3 model despite having over 100x fewer parameters. The 175B InstructGPT model is preferred over 175B GPT-3 $85 \pm 3\%$ of the time and over few-shot 175B GPT-3 $71 \pm 4\%$ of the time. On the TruthfulQA benchmark, InstructGPT generates truthful and informative answers approximately twice as often as GPT-3. On closed-domain API tasks, hallucination rates drop from 41% (GPT-3) to 21% (InstructGPT). When prompted to be respectful, InstructGPT generates roughly 25% fewer toxic outputs on RealToxicityPrompts. Inter-annotator agreement for labelers is $72.6 \pm 1.5\%$, with held-out labelers showing $77.3 \pm 1.3\%$ agreement. The PPO-ptx modification successfully reverses performance regressions on datasets like HellaSwag, though it still lags on DROP, SQuADv2, and machine translation.

The authors acknowledge several critical constraints. InstructGPT models still commit simple errors, such as accepting false premises, overly hedging simple questions, or failing under complex constraint combinations. Crucially, the models follow user instructions even when they lead to harmful outputs; for instance, they generate more toxic text than GPT-3 when explicitly instructed to be biased. Alignment is strictly bounded to the preferences of the specific labelers, researchers, and API customers who provided the training data, rather than universal human values. The contractor pool (~40 individuals) is predominantly English-speaking and not demographically representative. Furthermore, the models show no significant improvement on bias benchmarks like Winogender and CrowS-Pairs, and the alignment procedure incurs residual performance taxes on certain traditional NLP tasks.
