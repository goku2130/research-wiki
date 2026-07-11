---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/pdf/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: reward-modeling
---

The paper "Training language models to follow instructions with human feedback" (Ouyang et al., 2022) addresses the problem that large language models (LMs), despite their size, often fail to align with user intent, generating outputs that can be untruthful, toxic, or unhelpful. This misalignment stems from the difference between the LMs' pretraining objective (predicting the next token) and the desired objective of "following user instructions helpfully and safely."

The authors propose a method to align LMs with user intent by fine-tuning them with human feedback, resulting in models called InstructGPT.

**Methodology (Three Steps):**

1.  **Supervised Fine-Tuning (SFT):**
    *   **Data Collection:** A team of 40 human labelers provides demonstrations of desired model behavior on a dataset of prompts. These prompts are sourced from two main categories:
        *   Prompts submitted through the OpenAI API Playground to earlier InstructGPT models.
        *   Labeler-written prompts, categorized as "Plain" (arbitrary tasks), "Few-shot" (instruction with query/response pairs), and "User-based" (prompts corresponding to OpenAI API waitlist use-cases).
    *   **Model Training:** A pretrained GPT-3 model is fine-tuned using supervised learning on this collected demonstration data.
    *   **Training Details:** 16 epochs, cosine learning rate decay, residual dropout of 0.2. Final SFT model selection is based on the Reward Model (RM) score on the validation set.

2.  **Reward Model (RM) Training:**
    *   **Data Collection:** Labelers rank multiple model outputs (between 4 and 9) for a given prompt from best to worst. This generates $\binom{K}{2}$ comparisons for each prompt.
    *   **Model Training:** A reward model (RM), initialized from the SFT model (with the final unembedding layer removed), is trained to predict human preferences. The RM takes a prompt and response as input and outputs a scalar reward.
    *   **Loss Function:** The loss function for the reward model is:

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

        where $r_\theta(x,y)$ is the scalar output of the reward model for prompt $x$ and completion $y$ with parameters $\theta$, $y_w$ is the preferred completion, $y_l$ is the less preferred completion, and $D$ is the dataset of human comparisons.
    *   **Normalization:** The RM is normalized using a bias so that labeler demonstrations achieve a mean score of 0 before reinforcement learning.
    *   **RM Size:** Only 6B parameter RMs are used to save compute and due to instability observed with 175B RM training.

3.  **Reinforcement Learning (RL):**
    *   **Algorithm:** The SFT model is fine-tuned using the Proximal Policy Optimization (PPO) algorithm (Schulman et al., 2017).
    *   **Reward Signal:** The scalar output of the trained RM serves as the reward function.
    *   **KL Penalty:** A per-token KL penalty from the SFT model is added to mitigate over-optimization of the reward model.
    *   **Value Function:** The value function is initialized from the RM.
    *   **PPO-ptx Variant:** To address performance regressions on public NLP datasets, a variant called PPO-ptx is introduced, which mixes pretraining gradients into the PPO gradients. The objective function for PPO-ptx is:
        

$$
\begin{array}{l} \text {objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \\ \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right] \end{array}
$$

        where $\pi_{\phi}^{RL}$ is the learned RL policy, $\pi^{SFT}$ is the supervised trained model, $D_{pretrain}$ is the pretraining distribution, $\beta$ is the KL reward coefficient, and $\gamma$ is the pretraining loss coefficient. For "PPO" models, $\gamma$ is set to 0.

**Key Quantitative Results and Numbers:**

*   **Model Sizes:** InstructGPT models are trained at 1.3B, 6B, and 175B parameters, all using the GPT-3 architecture.
*   **Human Preference:**
    *   Outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters.
    *   175B InstructGPT outputs are preferred to 175B GPT-3 outputs $85 \pm 3\%$ of the time.
    *   175B InstructGPT outputs are preferred to few-shot 175B GPT-3 outputs $71 \pm 4\%$ of the time.
    *   InstructGPT models generate more appropriate outputs and more reliably follow explicit constraints.
*   **Truthfulness:**
    *   On the TruthfulQA benchmark, InstructGPT generates truthful and informative answers about twice as often as GPT-3.
    *   On "closed-domain" tasks, InstructGPT models hallucinate (make up information) about half as often as GPT-3 (21% vs. 41% hallucination rate).
*   **Toxicity:**
    *   InstructGPT models generate about 25% fewer toxic outputs than GPT-3 when prompted to be respectful.
    *   InstructGPT does not significantly improve over GPT-3 on bias datasets like Winogender and CrowS-Pairs.
*   **Performance Regressions (Alignment Tax):**
    *   RLHF fine-tuning can lead to performance regressions on public NLP datasets (e.g., SQuAD, DROP, HellaSwag, WMT 2015 French to English translation).
    *   Mixing PPO updates with pretraining distribution log likelihood updates (PPO-ptx) greatly reduces these regressions without compromising labeler preference scores.
*   **Generalization to Held-Out Labelers:** Held-out labelers prefer InstructGPT outputs to GPT-3 at about the same rate as training labelers. Inter-annotator agreement for training labelers is $72.6 \pm 1.5\%$, and for held-out labelers is $77.3 \pm 1.3\%$.
*   **Comparison with FLAN/T0:** On the API prompt distribution, InstructGPT has a $73.4 \pm 2\%$ winrate against the SFT baseline, compared to $26.8 \pm 2\%$ for T0 and $29.8 \pm 2\%$ for FLAN.
*   **Dataset Sizes:** SFT dataset: ~13k training prompts; RM dataset: ~33k training prompts; PPO dataset: ~31k training prompts.
*   **Training Cost:** Training the 175B SFT model requires 4.9 petaflops/s-days, and the 175B PPO-ptx model requires 60 petaflops/s-days, compared to 3,640 petaflops/s-days for GPT-3.

**Stated Limitations:**

*   **Labeler Demographics and Bias:** The behavior of InstructGPT models is influenced by the 40 contractors hired, who are primarily English-speaking and from the US or Southeast Asia. This group is not representative of all potential users.
*   **Data Collection Improvements:** Most comparisons are labeled by only one contractor, which might miss areas of labeler disagreement. Aligning to average preferences might not be desirable in all cases (e.g., minority groups).
*   **Model Imperfections:** InstructGPT models are not fully aligned or safe; they can still generate toxic/biased outputs, make up facts, and produce sexual/violent content without explicit prompting. They can also fail on some inputs, such as instructions with false premises or those requiring multiple complex constraints.
*   **Harmful Instructions:** The models generally follow user instructions, even if they could lead to harm in the real world (e.g., generating more toxic outputs when explicitly instructed to be maximally biased).
*   **Over-hedging:** InstructGPT can sometimes overly hedge in its answers, even to simple questions, potentially due to labelers rewarding epistemic humility.
*   **Generalization to False Premises:** The models do not generalize well to instructions with false premises, likely due to their rarity in the training data.
*   **Alignment Tax Mitigation:** The PPO-ptx approach does not completely eliminate performance regressions on all public NLP datasets (e.g., DROP, SQuADv2, translation).
*   **"Who to Align To":** The alignment is to the preferences of the training labelers, the researchers designing the study, and OpenAI API customers, which may not represent all users or broader human values. There are difficulties in designing a fair, transparent, and accountable alignment process.
