---
id: arxiv:2403.17031
type: paper
title: 'The N+ Implementation Details of RLHF with PPO: A Case ...'
url: https://arxiv.org/abs/2403.17031
retrieved: '2026-07-12'
maturity: comprehensive
topic: rlhf-ppo-pipeline
---

This work reproduces the Reinforcement Learning from Human Feedback (RLHF) scaling behaviors observed in OpenAI's TL;DR summarization task (Stiennon et al., 2020). The core problem addressed is the difficulty in openly reproducing RLHF pipelines due to subtle implementation details, challenging evaluation for instruction-following tasks, and long training times. This paper focuses on TL;DR summarization due to its easier evaluation and shorter generation lengths, allowing for faster iteration.

The method involves building an RLHF pipeline from scratch, enumerating over 20 key implementation details, and training Pythia models of various sizes (1B, 2.8B, 6.9B). The pipeline consists of three main steps:

1.  **Supervised Fine-Tuning (SFT) Policy Training:**
    *   Pre-trained LLMs are fine-tuned on human demonstrations using a next-token prediction loss.
    *   The SFT dataset consists of Reddit posts and human-generated summaries.
    *   **Detail 1: Dataset Specification:** The SFT dataset contains subreddit, title, post, and reference summary columns.
    *   **Detail 2: Query Pre-processing:** Query input strings are formatted as "SUBREDDIT: r/{subreddit}\n\nTITLE: {title}\n\nPOST: {post}\n\nTL;DR:". Queries are tokenized, and if length exceeds 512 tokens, the last paragraph is truncated. If less than 512, it's left-padded.
    *   **Detail 3: Tokenization of Completions:** A leading space is prepended to the completion, and an EOS `<|endoftext|>` token is appended. A special padding token `[PAD]` is used, distinct from the EOS token, to ensure the model learns to stop generation.
    *   **Detail 4: SFT and Preference Dataset Tokenization Lengths:** SFT reference summaries have a maximum of 53 tokens (using Pythia's tokenizer), while preference dataset chosen/rejected responses can be up to 169 tokens. The median chosen response length is 32, and rejected is 30.
    *   **Detail 5: Pre-tokenize the dataset:** For SFT training, query and response are concatenated and right-padded to a shape of (B, 562).
    *   **Detail 7: Disable Dropout:** Dropout layers are disabled during SFT, RM, and PPO training to ensure reproducible log probabilities for KL penalty calculation in PPO.
    *   **Detail 9: SFT Training Setups:** Hyperparameters include 1 epoch (116,722 episodes), AdamW optimizer ($\epsilon=1e-5$, lr=3e-6), Cosine scheduler, and a batch size of 128.

2.  **Reward Model (RM) Training:**
    *   Human preference pairs are collected, and an RM is trained to predict the log probability that a completion is preferred.
    *   The RM is initialized from the SFT policy with a randomly initialized linear head.
    *   The RM loss function is:
        $\mathcal{L}_{R}(r_{\phi})=\mathbb{E}_{(x,y_{c},y_{r})\sim\mathcal{D}_{\mathtt{P R E F}}}\bigl[\text{l o g}(1+e^{r_{\phi}(x,y_{r})-r_{\phi}(x,y_{c})})\bigr]$
        where $x$ is the prompt, $y_c$ is the chosen completion, $y_r$ is the rejected completion, and $\phi$ are the parameters of the RM $r$.
    *   **Detail 5: Pre-tokenize the dataset:** For RM training, query-chosen and query-rejected responses are concatenated and right-padded to a shape of (B, 638). For RM evaluation on CNN/DM, the batch shape is (B, 2021).
    *   **Detail 10: RM Training Setups:** Hyperparameters include 1 epoch (92,858 episodes), AdamW optimizer ($\epsilon=1e-5$, lr=3e-6), Cosine scheduler, and a batch size of 64.
    *   **Detail 12: Extract Reward from EOS Token:** The reward is extracted from the EOS token.
    *   **Detail 13: Reward Logits:** Non-EOS tokens typically have negative reward logits.
    *   **Detail 14: Numerical Differences in Padding:** Left-padding vs. right-padding can introduce minor numerical differences (e.g., average reward scalar difference of -0.000544150301720947 for 6.9B RM).
    *   **Detail 15: Reward Normalization:** RM outputs are normalized such that reference summaries from the SFT dataset achieve a mean score of 0 by adding a bias to the reward head.
    *   **Detail 16: Different Batches/Confidences have Different Accuracies:** Validation accuracy varies across different batches and confidence levels in the preference dataset. For instance, the 1B model's validation accuracy ranges from 0.508 (batch 11) to 0.771 (batch 13).
    *   **Detail 18: RM Calibration:** RMs show a positive correlation between accuracy and score difference, but are generally under-calibrated.
    *   **Detail 19: Comparison with DPO's Implicit Reward Modeling:** DPO's implicit reward modeling showed a regression in validation accuracy compared to explicit reward modeling, potentially due to differences in loss application (all tokens vs. EOS token), the presence of the $\beta$ parameter, and the objective function.

3.  **Reinforcement Learning (RL) Policy Training with PPO:**
    *   An RL policy, initialized from the SFT policy, samples completions. The RM provides a score, and a KL penalty is applied to prevent deviation from the SFT policy.
    *   The reward for the RL problem is:
        $R(x,y)=\left(r_{\phi}(x,y)-\beta\mathbb{D}_{\text{K L}}\left[\pi_{\theta}(y\ |\ x)\ ||\ \pi^{\text{S F T}}(y\ |\ x)\right]\right)$
        where $\beta$ controls KL penalty strength, $\theta$ are RL policy parameters, and $\pi_{\theta}$ is the RL policy.
    *   PPO maximizes the objective: $\text{m a x}_{\pi_{\theta}}\mathbb{E}_{x\sim\mathcal{D}_{\mathtt{S F T}},y\sim\pi_{\theta}(y|x)}R(x,y)$.
    *   **Detail 5: Pre-tokenize the dataset:** For PPO training, queries are left-padded to a shape of (B, 512).
    *   **Detail 21: Re-use SFT Dataset:** The SFT dataset is re-used and shuffled for 1,000,000 episodes (approx. 8.56 epochs).
    *   **Detail 22: Value Model Initialization:** The value network is initialized from the reward model, acting as a per-token RM.
    *   **Detail 23: "EOS Trick":** To ensure defined reward scores, if a generated response lacks an EOS token, its score is replaced with -1.
    *   **Detail 24: Reward Whitening:** Optional reward whitening is applied using the formula:
        $\text{whitened}=(values-mean)*torch.rsqrt(var+1e-8)$
        This can lead to shorter outputs and lower preference rates, but length-controlled comparisons show similar performance.
    *   **Detail 25: Advantage Whitening:** Advantages are whitened using `whiten(advantages)` with a shifted mean.
    *   **PPO Hyperparameters:** Episodes: 1,000,000; Optimizer: AdamW ($\epsilon=1e-5$, lr=3e-6); Scheduler: Linear; Batch Size: 512; $\beta$ (KL Penalty): 0.05; $\gamma$ (Discount Factor): 1.0; $\lambda$ (for GAE): 0.95; Number of Mini-batches: 1; K (PPO Update Iteration Per Epoch): 4; $\epsilon$ (PPO's Policy Clipping): 0.2; $\hat{\varepsilon}$ (Value Clipping): 0.2; $c_1$ (Value Function Coefficient): 0.1; Value Function Loss Clipping: True; Sampling Temperature: 0.7.

**Key Quantitative Results:**

*   **SFT Models:** Larger models exhibit smaller next-token-prediction losses and improved ROUGE scores.
*   **RM Models:** Larger RMs achieve higher validation accuracy.
    *   1B model overall validation accuracy: 0.628 ± 0.002
    *   2.8B model overall validation accuracy: 0.669 ± 0.003
    *   6.9B model overall validation accuracy: 0.689 ± 0.004
    *   Validation accuracy on CNN/DM dataset: 1B (0.627 ± 0.013), 2.8B (0.665 ± 0.01), 6.9B (0.686 ± 0.003).
    *   RM agreement rate with GPT3.5: 1B (0.403), 2.8B (0.726), 6.9B (0.767).
*   **PPO Models:**
    *   PPO models demonstrate significant gains in response quality, with 2.8B and 6.9B models outperforming OpenAI's released 1.3B checkpoint.
    *   The win rate of PPO models' summaries over human-generated reference summaries (judged by GPT-3.5) scales with model size. The best 6.9B model is preferred nearly 80% of the time.
    *   Over-optimization can occur in 1B models, leading to high KL divergence (50-85) and poor human preference (less than 20% win rate).
    *   PPO models significantly outperform SFT models across all summary lengths when controlling for length. Win rate increases with summary length.

**Stated Limitations:**

*   The DPO's implicit reward modeling showed a regression in validation accuracy compared to explicit reward modeling, suggesting it may not be equivalent. Further research is advocated.
*   RMs are still under-calibrated, possibly due to the diverse validation set and varying accuracies within it.
*   The increase in PPO win-rate with summary length implies either GPT-3.5 prefers longer summaries or longer summaries better optimize true human preference.
*   The study uses GPT-3.5 as a judge, which itself is an LLM and may have its own biases.
