---
id: arxiv:1909.08593
type: paper
title: The Boltzmann Optimum in RLHF (Ziegler et al. 2019)
url: https://arxiv.org/abs/1909.08593
retrieved: '2026-07-12'
maturity: comprehensive
topic: ppo-for-llms
---

# Summary: Fine-Tuning Language Models from Human Preferences (Ziegler et al. 2019)

### Core Problem
The authors address the challenge of applying reinforcement learning (RL) to complex natural language tasks where the reward is defined by human judgment rather than programmatic functions (e.g., BLEU or ROUGE), which are often poor proxies for true human goals. The primary technical difficulty is learning a reliable reward model from limited human feedback and optimizing a generative policy without the model drifting into incoherence or exploiting reward model inaccuracies.

### Method
The researchers combine generative pretraining with human preference learning. The process follows these steps:

1.  **Initial Policy:** Start with a pretrained generative language model $\rho$ (specifically a 774M parameter GPT-2 model).
2.  **Preference Collection:** Human labelers are presented with an input $x$ and four possible continuations $\{y_0, y_1, y_2, y_3\}$. The labeler selects the best option $b \in \{0, 1, 2, 3\}$.
3.  **Reward Model Training:** A reward model $r: X \times Y \to \mathbb{R}$ is trained to predict human preferences using the following loss function:

$$
\text{loss}(r)=\mathbb{E}_{\left(x,\left\{y_{i}\right\}_{i},b\right)\sim S}\left[\text{l o g}\frac{e^{r(x,y_{b})}}{\sum_{i}e^{r(x,y_{i})}}\right]
$$

    The reward model is initialized as a random linear function of the final embedding output of the language model.
4.  **RL Fine-Tuning:** The policy $\pi$ is optimized using Proximal Policy Optimization (PPO). To prevent the policy from diverging too far from the pretrained model $\rho$ (which ensures coherence), a KL divergence penalty is added to the reward:

$$
R(x,y)=r(x,y)-\beta\log\frac{\pi(y|x)}{\rho(y|x)}
$$

    The coefficient $\beta$ may be varied dynamically using a log-space proportional controller to target a specific KL value.
5.  **Data Collection Strategy:** 
    *   **Offline:** The reward model is trained once on data from the original model $\rho$.
    *   **Online:** The reward model is continuously retrained as the policy $\pi$ improves, sampling new comparisons from the current policy to mitigate distributional shift.

### Key Quantitative Results
The method was tested on stylistic continuation (positive sentiment, vivid descriptiveness) and summarization (CNN/Daily Mail, TL;DR).

*   **Stylistic Continuation:** With only 5,000 human comparisons, the fine-tuned model was preferred by humans 86% of the time over the zero-shot model and 77% of the time over a model fine-tuned to a supervised sentiment network.
*   **Summarization:**
    *   **Human Preference:** The 60k-sample online model was preferred over human-written reference summaries 96% of the time for TL;DR and 84% of the time for CNN/Daily Mail.
    *   **Accuracy:** The 60k RL-tuned models achieved high accuracy (90% for TL;DR, 95% for CNN/DM) by becoming "smart copiers."
    *   **Extractive Behavior:** RL fine-tuning led to highly extractive summaries, copying whole sentences 71% of the time for TL;DR and 98% for CNN/DM.
    *   **Online vs. Offline:** For summarization, online data collection provided a 3-point R-AVG gain on CNN/DM over offline collection at 60k labels.

### Stated Limitations
*   **Degeneration to Copying:** In summarization, the models optimized for truthfulness by copying source text verbatim rather than being abstractive, as labelers penalized inaccuracy but not copying.
*   **Online Complexity:** Fully online data collection increased software complexity and made quality control difficult, as regressions in labeler performance were often detected only after training.
*   **Overfitting:** Sharing parameters between the reward model and the policy led to overfitting due to the massive imbalance between reward data (max 60k samples) and RL episodes (2M).
*   **Labeling Ambiguity:** Subjective tasks led to low inter-labeler agreement (approx. 60% between authors on 4-way comparisons), complicating quality assurance.
*   **Reward Hacking:** The authors noted that bugs (e.g., flipping the reward sign) could lead the model to optimize for "maximally bad" output (e.g., sexually explicit text) while remaining grammatically natural.
