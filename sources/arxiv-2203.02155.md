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
The authors address the "misalignment" of large language models (LMs). While scaling model size improves general capabilities, the standard pretraining objective—predicting the next token from internet text—does not inherently align the model with the user's intent. Consequently, large LMs often generate outputs that are untruthful, toxic, or unhelpful, failing to follow explicit instructions. The goal is to align models to be helpful, honest, and harmless.

### Method
The researchers employ Reinforcement Learning from Human Feedback (RLHF) to fine-tune GPT-3. The process consists of three primary steps:

1.  **Supervised Fine-Tuning (SFT):** Labelers provide demonstrations of the desired behavior for a set of prompts (both labeler-written and submitted via the OpenAI API). GPT-3 is fine-tuned on this demonstration data using supervised learning.
2.  **Reward Model (RM) Training:** Labelers rank multiple model outputs for a given prompt from best to worst. A reward model is trained to predict these human preferences. To prevent overfitting and increase efficiency, the RM is trained on all $\binom{K}{2}$ comparisons from a single ranking task as a single batch element.
3.  **Reinforcement Learning (PPO):** The SFT model is further fine-tuned using the Proximal Policy Optimization (PPO) algorithm, using the RM as a scalar reward function. A per-token KL penalty from the SFT model is added to prevent over-optimization of the reward model.

To mitigate the "alignment tax"—performance regressions on public NLP datasets (e.g., SQuAD, HellaSwag)—the authors introduce **PPO-ptx**, which mixes PPO updates with updates that increase the log likelihood of the original pretraining distribution.

### Key Formulas
The loss function for the **Reward Model** is defined as:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

where $r_\theta(x,y)$ is the scalar reward for prompt $x$ and completion $y$, and $y_w$ is the preferred completion over $y_l$.

The objective function for **PPO-ptx** is:

$$
\text{objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\frac{\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x)}{\pi^ {\mathrm{SFT}} (y \mid x)}\right) \right] + \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right]
$$

where $\beta$ is the KL reward coefficient and $\gamma$ is the pretraining loss coefficient.

### Key Quantitative Results
*   **Human Preference:** Labelers significantly preferred InstructGPT over GPT-3. The 175B InstructGPT model was preferred over the 175B GPT-3 baseline $85 \pm 3\%$ of the time and over the few-shot 175B GPT-3 $71 \pm 4\%$ of the time. Notably, the 1.3B InstructGPT model was preferred over the 175B GPT-3.
*   **Truthfulness:** On closed-domain tasks, the hallucination rate dropped from 41% (GPT-3) to 21% (InstructGPT). PPO models also showed significant improvements on the TruthfulQA benchmark.
*   **Toxicity:** InstructGPT generated approximately 25% fewer toxic outputs than GPT-3 when prompted to be respectful.
*   **Generalization:** InstructGPT outperformed models fine-tuned on public NLP datasets (FLAN and T0), with winrates of $78 \pm 4\%$ and $79 \pm 4\%$, respectively.
*   **Compute Efficiency:** Alignment is significantly cheaper than pretraining. Training the 175B PPO-ptx model required 60 petaflops/s-days, compared to 3,640 petaflops/s-days for the original GPT-3 pretraining.

### Stated Limitations
*   **Demographic Bias:** The model is aligned to a small group of $\sim 40$ contractors (primarily English-speaking individuals in the US and Southeast Asia), which is not representative of the global user base.
*   **Safety Gaps:** The models still generate toxic or biased outputs, fabricate facts, and produce sexual or violent content without explicit prompting.
*   **Instruction Failures:** InstructGPT still makes "simple mistakes," such as failing to detect false premises in instructions, overly hedging answers, and struggling with multiple explicit constraints.
*   **Harmful Compliance:** The model often follows user instructions even if they are harmful (e.g., generating more toxic output than GPT-3 when explicitly prompted to be biased).
