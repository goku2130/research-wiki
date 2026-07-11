---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback (InstructGPT)
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-llms-overview
---

# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

### Core Problem
The authors address the "misalignment" between the standard language modeling objective—predicting the next token on internet text—and the actual goal of following a user's instructions helpfully, honestly, and harmlessly. Large language models (LLMs) like GPT-3 often generate outputs that are untruthful, toxic, or fail to follow explicit user intent, despite their scale.

### Method
The researchers employ Reinforcement Learning from Human Feedback (RLHF) to align GPT-3 with user intent. The process consists of three primary steps:

1.  **Supervised Fine-Tuning (SFT):** A team of labelers provides demonstrations of the desired behavior for a wide range of prompts (including API-submitted and labeler-written prompts). GPT-3 is fine-tuned on this demonstration data using supervised learning.
2.  **Reward Model (RM) Training:** Labelers rank multiple model outputs for the same prompt from best to worst. A reward model is trained to predict these human preferences. To prevent overfitting and increase efficiency, the model is trained on all $\binom{K}{2}$ comparisons from a ranking of $K$ responses as a single batch element.
3.  **Reinforcement Learning via PPO:** The SFT model is further fine-tuned using the Proximal Policy Optimization (PPO) algorithm, using the scalar output of the RM as the reward function.

To mitigate the "alignment tax"—performance regressions on public NLP datasets (e.g., SQuAD, HellaSwag)—the authors introduce **PPO-ptx**, which mixes PPO updates with updates that increase the log likelihood of the original pretraining distribution.

### Key Formulas
The reward model is trained using the following loss function:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, and $y_w$ is the preferred completion.

The PPO-ptx objective function combines the RL reward, a KL penalty to prevent over-optimization, and the pretraining loss:

$$
\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]
$$

where $\beta$ is the KL reward coefficient and $\gamma$ is the pretraining loss coefficient.

### Key Quantitative Results
*   **Human Preference:** Labelers significantly preferred InstructGPT over GPT-3. The 175B InstructGPT model was preferred over the 175B GPT-3 baseline $85 \pm 3\%$ of the time and over the few-shot 175B GPT-3 $71 \pm 4\%$ of the time. Notably, the 1.3B InstructGPT model was preferred over the 175B GPT-3.
*   **Truthfulness:** On the TruthfulQA benchmark, PPO models were truthful and informative roughly twice as often as GPT-3. In closed-domain tasks, the hallucination rate dropped from 41% (GPT-3) to 21% (InstructGPT).
*   **Toxicity:** When prompted to be respectful, InstructGPT generated approximately 25% fewer toxic outputs than GPT-3.
*   **Comparison to Other Instruction Tuning:** InstructGPT outperformed models fine-tuned on public NLP datasets (FLAN and T0). The 175B InstructGPT was preferred over FLAN $78 \pm 4\%$ of the time and over T0 $79 \pm 4\%$ of the time.

### Stated Limitations
*   **Simple Mistakes:** The models still fail to detect instructions with false premises, may "over-hedge" (providing multiple possible answers to simple questions), and struggle with multiple explicit constraints.
*   **Safety Gaps:** The models are not fully safe; they can still generate toxic or biased content without explicit prompting. Furthermore, if explicitly instructed to be biased, InstructGPT can generate more toxic outputs than GPT-3.
*   **Data Representativeness:** The alignment is based on the preferences of a small group of approximately 40 contractors (primarily English-speaking) and OpenAI API customers, which may not represent the global population.
