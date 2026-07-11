---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback (InstructGPT)
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: ppo-for-llms
---

# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

### Core Problem
The authors address the "misalignment" of large language models (LLMs). While scaling models like GPT-3 improves general capabilities, the pretraining objective—predicting the next token from internet text—does not inherently align the model with the user's intent to follow instructions helpfully and safely. Consequently, these models often generate untruthful, toxic, or unhelpful outputs.

### Method
The researchers employ Reinforcement Learning from Human Feedback (RLHF) to align GPT-3 with human intent through a three-step process:

1.  **Supervised Fine-Tuning (SFT):** A team of labelers provides demonstrations of the desired behavior for a set of prompts (both labeler-written and submitted via the OpenAI API). GPT-3 is fine-tuned on this demonstration data using supervised learning.
2.  **Reward Model (RM) Training:** Labelers rank multiple model outputs for the same prompt from best to worst. A reward model is trained to predict these human preferences. To prevent overfitting and increase efficiency, the model is trained on all $\binom{K}{2}$ comparisons from a single ranking task as a single batch element.
3.  **Reinforcement Learning (PPO):** The SFT model is further fine-tuned using the Proximal Policy Optimization (PPO) algorithm, using the RM as a scalar reward function. A per-token KL penalty from the SFT model is added to prevent over-optimization of the reward model.

To mitigate the "alignment tax"—performance regressions on public NLP datasets (e.g., SQuAD, HellaSwag)—the authors introduce **PPO-ptx**, which mixes PPO updates with updates that increase the log likelihood of the original pretraining distribution.

### Key Formulas
The loss function for the **Reward Model** is defined as:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, and $y_w$ is the preferred completion.

The combined objective function for **PPO-ptx** is:

$$
\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]
$$

where $\beta$ controls the KL penalty and $\gamma$ controls the strength of the pretraining gradients.

### Key Quantitative Results
*   **Human Preference:** Labelers significantly prefer InstructGPT over GPT-3. The 175B InstructGPT model was preferred over the 175B GPT-3 baseline $85 \pm 3\%$ of the time and over the few-shot 175B GPT-3 $71 \pm 4\%$ of the time. Notably, the 1.3B parameter InstructGPT model was preferred over the 175B GPT-3.
*   **Truthfulness:** On closed-domain tasks, InstructGPT reduced the hallucination rate to 21%, compared to 41% for GPT-3.
*   **Toxicity:** When prompted to be respectful, InstructGPT models generated approximately 25% fewer toxic outputs than GPT-3.
*   **Comparison to Other Instruction-Tuning:** InstructGPT significantly outperformed models fine-tuned on FLAN and T0 datasets. On the API prompt distribution, InstructGPT had a $73.4 \pm 2\%$ winrate against the SFT baseline, while T0 and FLAN had $29.8 \pm 2\%$ and $26.8 \pm 2\%$, respectively.

### Limitations
*   **Reliability:** The models still make simple mistakes, such as failing to detect false premises in instructions, over-hedging (giving multiple possible answers to simple questions), and failing to follow multiple explicit constraints.
*   **Safety:** InstructGPT is not fully safe; it can still generate biased, toxic, sexual, or violent content without explicit prompting. Furthermore, it often follows user instructions even when they are explicitly harmful (e.g., requesting biased output).
*   **Representativeness:** The alignment is based on a small group of approximately 40 contractors, who are primarily English-speaking and may not represent the diverse preferences of the global user base.
