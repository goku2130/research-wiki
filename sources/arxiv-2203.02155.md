---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/abs/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

The paper "Training language models to follow instructions with human feedback" by Ouyang et al. (2022) addresses the core problem that large language models (LMs), despite their size, often fail to align with user intent, producing outputs that can be untruthful, toxic, or unhelpful. This misalignment stems from the difference between the language modeling objective (predicting the next token) and the desired objective of "follow the user’s instructions helpfully and safely."

The authors propose a three-step method, called InstructGPT, to align LMs with user intent using human feedback:

1.  **Supervised Fine-Tuning (SFT):**
    *   **Data Collection:** Labelers provide demonstrations of desired model behavior on a dataset of prompts. This dataset includes prompts submitted to the OpenAI API and labeler-written prompts.
    *   **Model Training:** A pre-trained GPT-3 model is fine-tuned on this demonstration data using supervised learning.
    *   **Training Details:** Models were trained for 16 epochs with a cosine learning rate decay and a residual dropout of 0.2. Final SFT model selection was based on the reward model (RM) score on the validation set.

2.  **Reward Model (RM) Training:**
    *   **Data Collection:** A dataset of comparisons between multiple model outputs for a given prompt is collected. Labelers rank these outputs from best to worst. For efficiency, labelers rank $K$ outputs (between 4 and 9), generating $\binom{K}{2}$ comparisons per prompt.
    *   **Model Training:** A reward model is trained to predict human preferences. This model takes a prompt and a completion as input and outputs a scalar reward.
    *   **Loss Function:** The loss function for the reward model is defined as:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

        where $r_\theta(x,y)$ is the scalar output of the reward model for prompt $x$ and completion $y$ with parameters $\theta$, $y_w$ is the preferred completion, $y_l$ is the less preferred completion, and $D$ is the dataset of human comparisons.
    *   **Normalization:** The reward model is normalized using a bias such that labeler demonstrations achieve a mean score of 0 before reinforcement learning.

3.  **Reinforcement Learning (RL) with Proximal Policy Optimization (PPO):**
    *   **Algorithm:** The SFT model is further fine-tuned using the PPO algorithm (Schulman et al., 2017).
    *   **Environment:** The environment is a bandit environment where a random customer prompt is presented, and the policy generates a response. The reward model provides a scalar reward for the generated response.
    *   **KL Penalty:** A per-token KL penalty from the SFT model is added to mitigate over-optimization of the reward model. The value function is initialized from the RM.
    *   **PPO-ptx (Pretraining Mix):** To minimize performance regressions on public NLP datasets, pretraining gradients are mixed into the PPO gradients. The combined objective function is:

$$
\begin{array}{l} \text {objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \\ \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right] \end{array}
$$

        where $\pi_{\phi}^{RL}$ is the learned RL policy, $\pi^{SFT}$ is the supervised trained model, $D_{pretrain}$ is the pretraining distribution, $\beta$ is the KL reward coefficient, and $\gamma$ is the pretraining loss coefficient. For "PPO" models, $\gamma$ is set to 0.

**Key Quantitative Results and Numbers:**

*   **Preference over GPT-3:** On human evaluations using their prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters.
*   **175B InstructGPT vs. 175B GPT-3:** 175B InstructGPT outputs are preferred to 175B GPT-3 outputs $85 \pm 3\%$ of the time.
*   **175B InstructGPT vs. Few-shot 175B GPT-3:** 175B InstructGPT outputs are preferred to few-shot 175B GPT-3 outputs $71 \pm 4\%$ of the time.
*   **Truthfulness:** On the TruthfulQA benchmark, InstructGPT generates truthful and informative answers about twice as often as GPT-3. On "closed-domain" tasks, InstructGPT models hallucinate about half as often as GPT-3 (21% vs. 41% hallucination rate).
*   **Toxicity:** InstructGPT models generate about 25% fewer toxic outputs than GPT-3 when prompted to be respectful.
*   **Performance Regressions (Alignment Tax):** PPO-ptx models mitigate performance regressions on public NLP datasets (e.g., SQuAD, DROP, HellaSwag, WMT 2015 French to English translation) observed with standard PPO.
*   **Cost:** Training the 175B SFT model requires 4.9 petaflops/s-days, and training the 175B PPO-ptx model requires 60 petaflops/s-days, compared to 3,640 petaflops/s-days for GPT-3.
*   **Inter-annotator Agreement:** Training labelers agree $72.6 \pm 1.5\%$ of the time, while held-out labelers agree $77.3 \pm 1.3\%$ of the time.
*   **RM Accuracy:** Reward models achieve an accuracy of $69.6 \pm 0.9\%$ on predicting preferences of held-out labelers, compared to $72.4 \pm 0.4\%$ on training labelers.
*   **FLAN/T0 Comparison:** InstructGPT has a $73.4 \pm 2\%$ winrate against their SFT baseline, compared to $26.8 \pm 2\%$ and $29.8 \pm 2\%$ for their version of T0 and FLAN, respectively.

**Stated Limitations:**

*   **Labeler Demographics:** The behavior of InstructGPT models is influenced by the contractors (approximately 40, primarily English-speaking, from the US or Southeast Asia) who provided feedback. This group is not representative of all potential users.
*   **Data Collection Setup:** Most comparisons are labeled by only one contractor, which might not fully capture areas of disagreement or diverse preferences.
*   **Model Imperfections:** InstructGPT models are not fully aligned or safe; they can still generate toxic/biased outputs, make up facts, and produce sexual/violent content without explicit prompting. They can also fail to generate reasonable outputs, such as being confused by false premises, overly hedging answers, or struggling with multiple explicit constraints.
*   **User Intent vs. Harm:** The models generally follow user instructions, even if those instructions could lead to real-world harm. Training models to refuse harmful instructions is important but complex due to context dependency.
*   **Alignment Tax Mitigation:** While PPO-ptx mitigates performance regressions, it does not completely eliminate them and may make undesirable behaviors more likely for some tasks if present in the pretraining data.
*   **Scope of Alignment:** The work aligns to a specific set of labelers' preferences influenced by instructions and context, not a broader notion of "human values" or all stakeholders.
*   **Generalization:** While some generalization to non-English languages and code is observed, more research is needed on how well this generalization scales.
