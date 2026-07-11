---
id: arxiv:2203.02155
type: paper
title: Training language models to follow instructions with human feedback
url: https://arxiv.org/pdf/2203.02155
retrieved: '2026-07-11'
maturity: comprehensive
topic: policy-gradient-methods
---

The paper "Training language models to follow instructions with human feedback" (Ouyang et al., 2022) addresses the problem that large language models (LMs), despite their size, often fail to align with user intent, generating outputs that can be untruthful, toxic, or unhelpful. This misalignment stems from the difference between the language modeling objective (predicting the next token) and the desired objective of "follow the user’s instructions helpfully and safely."

The authors propose a method to align LMs with user intent by fine-tuning them with human feedback, resulting in models called InstructGPT.

**Method/Recipe Step by Step:**

The method involves three main steps, building on reinforcement learning from human feedback (RLHF):

1.  **Supervised Fine-tuning (SFT):**
    *   **Data Collection:** A team of 40 human labelers, selected through a screening test, provides demonstrations of desired model behavior. This data includes labeler-written prompts (plain, few-shot, user-based) and prompts submitted through the OpenAI API Playground.
    *   **Model Training:** A pre-trained GPT-3 model is fine-tuned on this collected demonstration data using supervised learning. The training runs for 16 epochs with a cosine learning rate decay and residual dropout of 0.2. Final SFT model selection is based on the Reward Model (RM) score on the validation set.

2.  **Reward Model (RM) Training:**
    *   **Data Collection:** Labelers rank multiple model outputs (between 4 and 9) for a given prompt from best to worst. This generates $\binom{K}{2}$ comparisons for each prompt.
    *   **Model Training:** A reward model (RM), initialized from the SFT model (with the final unembedding layer removed), is trained to predict human preferences. The RM takes a prompt and a response as input and outputs a scalar reward. The authors primarily use 6B parameter RMs. The loss function is designed to handle batches of comparisons from a single prompt to prevent overfitting.
    *   **Normalization:** The RM is normalized using a bias such that labeler demonstrations achieve a mean score of 0 before reinforcement learning.

3.  **Reinforcement Learning (RL):**
    *   **Algorithm:** The SFT model is further fine-tuned using the Proximal Policy Optimization (PPO) algorithm (Schulman et al., 2017).
    *   **Environment:** The environment presents a random customer prompt, and the policy generates a response. The RM then provides a scalar reward for this response, ending the episode.
    *   **KL Penalty:** A per-token KL penalty from the SFT model is added to mitigate over-optimization of the reward model.
    *   **Value Function:** The value function for PPO is initialized from the RM.
    *   **PPO-ptx Variant:** To minimize performance regressions on public NLP datasets, a variant called PPO-ptx is introduced, which mixes pretraining gradients into the PPO gradients.

**Key Formulas in LaTeX:**

1.  **Reward Model Loss Function:**

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} E_{(x,y_w,y_l)\sim D} [\log(\sigma(r_\theta(x,y_w) - r_\theta(x,y_l)))]
$$

    where $r_\theta(x,y)$ is the scalar output of the reward model for prompt $x$ and completion $y$ with parameters $\theta$, $y_w$ is the preferred completion out of the pair of $y_w$ and $y_l$, and $D$ is the dataset of human comparisons.

2.  **Combined Objective Function for PPO-ptx:**
    

$$
\begin{array}{l} \text {objective} (\phi) = E _ {(x, y) \sim D _ {\pi_ {\phi} ^ {\mathrm{RL}}}} \left[ r _ {\theta} (x, y) - \beta \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (y \mid x) / \pi^ {\mathrm{SFT}} (y \mid x)\right) \right] + \\ \gamma E _ {x \sim D _ {\text {pretrain}}} \left[ \log \left(\pi_ {\phi} ^ {\mathrm{RL}} (x)\right) \right] \end{array}
$$

    where $\pi_{\phi}^{RL}$ is the learned RL policy, $\pi^{SFT}$ is the supervised trained model, and $D_{pretrain}$ is the pretraining distribution. $\beta$ is the KL reward coefficient, and $\gamma$ is the pretraining loss coefficient. For "PPO" models, $\gamma$ is set to 0.

**Key Quantitative Results and Numbers:**

*   **Model Sizes:** InstructGPT models were trained at 1.3B, 6B, and 175B parameters, all using the GPT-3 architecture.
*   **Human Preference:**
    *   Outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters.
    *   175B InstructGPT outputs are preferred to 175B GPT-3 outputs $85 \pm 3\%$ of the time.
    *   175B InstructGPT outputs are preferred to few-shot 175B GPT-3 outputs $71 \pm 4\%$ of the time.
*   **Truthfulness:**
    *   On the TruthfulQA benchmark, InstructGPT generates truthful and informative answers approximately twice as often as GPT-3.
    *   On "closed-domain" tasks from the API prompt distribution, InstructGPT models make up information not present in the input about half as often as GPT-3 (a 21% vs. 41% hallucination rate, respectively).
*   **Toxicity:**
    *   InstructGPT models generate about 25% fewer toxic outputs than GPT-3 when prompted to be respectful.
    *   InstructGPT does not significantly improve over GPT-3 on the Winogender and CrowSPairs datasets for bias.
*   **Performance Regressions (Alignment Tax):** PPO-ptx models mitigate performance regressions on public NLP datasets (e.g., SQuAD, DROP, HellaSwag, WMT 2015 French to English translation) observed with standard PPO. PPO-ptx even surpasses GPT-3 on HellaSwag.
*   **Data Statistics:**
    *   SFT dataset: ~13k training prompts.
    *   RM dataset: ~33k training prompts.
    *   PPO dataset: ~31k training prompts.
    *   API prompt distribution: 45.6% Generation, 12.4% Open QA, 11.2% Brainstorming, 8.4% Chat, 6.6% Rewrite, 4.2% Summarization, 3.5% Classification, 3.5% Other, 2.6% Closed QA, 1.9% Extract.
*   **Inter-annotator Agreement:** Training labelers agree with each other $72.6 \pm 1.5\%$ of the time; held-out labelers agree $77.3 \pm 1.3\%$ of the time.
*   **Cost:** Training the 175B SFT model requires 4.9 petaflops/s-days, and the 175B PPO-ptx model requires 60 petaflops/s-days, compared to 3,640 petaflops/s-days for GPT-3.

**Stated Limitations:**

*   **Labeler Demographics and Bias:** The behavior of InstructGPT models is influenced by the human feedback from approximately 40 contractors, who are primarily English-speaking and located in the United States or Southeast Asia. This group is not representative of the full spectrum of potential users, and their value judgments, beliefs, and cultural backgrounds can impact the labeling.
*   **Data Collection Improvements:** Most comparisons are labeled by only one contractor due to cost, which might miss areas of disagreement where a single model cannot align to all preferences. Aligning to average preferences might not be desirable in all cases (e.g., disproportionate impact on minority groups).
*   **Model Imperfections:** InstructGPT models are not fully aligned or safe; they still generate toxic/biased outputs, make up facts, and produce sexual/violent content without explicit prompting. They can also fail on some inputs, such as instructions with false premises or those requiring multiple complex constraints.
*   **Harmful Instructions:** The models generally follow user instructions, even if they could lead to harm in the real world. For example, when instructed to be maximally biased, InstructGPT generates more toxic outputs than GPT-3.
*   **Alignment Tax Mitigation:** While PPO-ptx mitigates most performance regressions, it does not completely eliminate them, and some undesirable behaviors from the pretraining data might become more likely for certain tasks.
*   **Generalization Scope:** While some generalization to non-English languages and code is observed, the extent of this generalization with increased capabilities needs further study.
*   **Feedback Efficiency:** Comparisons are not necessarily the most efficient way to provide alignment signals; alternative methods like editing responses or natural language critiques could be explored.
*   **Definition of Alignment:** The paper aligns to inferred user intention for simplicity, but acknowledges the complexity of aligning to instructions, intentions, revealed preferences, ideal preferences, interests, and values, and the need for a principle-based approach.
*   **Misuse Potential:** Making LMs better at following instructions also makes them easier to misuse for generating misinformation or harmful content.
*   **Deployment and Access:** Restricting access to powerful LMs to a few organizations raises concerns about transparency, centralization of power, and equitable access to cutting-edge ML technology.
