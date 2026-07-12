---
id: rlhfbook:reward-modeling-rlhf-and-post-training-b
type: web
title: Reward Modeling | RLHF and Post-Training Book by Nathan Lambert
url: https://rlhfbook.com/c/05-reward-models
retrieved: '2026-07-12'
maturity: comprehensive
topic: reward-modeling
---

Reward models (RMs) are central to Reinforcement Learning from Human Feedback (RLHF), learning complex human preferences to provide a learning signal for agents, similar to an environment in standard RL, but learned from human input. They compress intricate data features into a usable representation for downstream training. Historically used as proxies for environment rewards in RL, in RLHF, RMs output a probability of an input being "of high quality" based on pairwise preference relations. This practice is related to inverse reinforcement learning.

### Core Problem
The core problem is to learn a reward function from human preferences that can effectively guide a language model's behavior, especially for tasks where explicit reward signals are difficult to define. This involves converting human preference data (e.g., chosen vs. rejected text completions) into a scalar reward signal that a model can optimize.

### Method/Recipe: Training a Bradley-Terry Reward Model

1.  **Data Collection:** Gather a dataset $D$ consisting of prompts $x$ and pairs of completions $(y_c, y_r)$, where $y_c$ is the human-preferred (chosen) completion and $y_r$ is the rejected completion for the given prompt $x$.
2.  **Model Architecture:** Start with a pre-trained language model (e.g., a causal language model) and append a small linear head to its final hidden state. This head transforms the sequence-level representation into a single scalar reward score $r_\theta(y|x)$.
    *   The `BradleyTerryRewardModel` class demonstrates this:
        ```python
        class BradleyTerryRewardModel(nn.Module):
            def __init__(self, base_lm):
                super().__init__()
                self.lm = base_lm  # e.g., AutoModelForCausalLM
                self.head = nn.Linear(self.lm.config.hidden_size, 1)

            def _sequence_rep(self, hidden, attention_mask):
                # Extracts the hidden state of the last non-padding token
                lengths = attention_mask.sum(dim=1) - 1
                batch_idx = torch.arange(hidden.size(0), device=hidden.device)
                return hidden[batch_idx, lengths]

            def forward(self, input_ids, attention_mask):
                outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
                hidden = outputs.hidden_states[-1]
                seq_repr = self._sequence_rep(hidden, attention_mask)
                rewards = self.head(seq_repr).squeeze(-1)
                return rewards
        ```
3.  **Loss Function Formulation:** The training objective is derived from the Bradley-Terry model of preference, which defines the probability of preferring item $i$ over item $j$ based on their latent strengths. This is re-parametrized using unbounded scores $p_i = e^{r_i}$, leading to a probability dependent on the score difference:
    *   $P(i>j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)$
    *   For a given prompt $x$ and completions $y_1, y_2$, the probability that the reward model assigns to $y_1$ being preferred to $y_2$ is:
        $P(y_1>y_2|x) = \frac{\exp(r_\theta(y_1|x))}{\exp(r_\theta(y_1|x)) + \exp(r_\theta(y_2|x))} = \sigma(r_\theta(y_1|x) - r_\theta(y_2|x))$
    *   The model is trained to maximize the log-likelihood of the observed preferences, which is equivalent to minimizing the negative log-likelihood.
4.  **Optimization:** Minimize the following loss function over the preference dataset $D$:
    *   $\theta^* = \arg\min_\theta \mathbb{E}_{(x,y_c,y_r) \sim D} [-\log \sigma(r_\theta(y_c|x) - r_\theta(y_r|x))]$
    *   The per-example loss is: $L(\theta) = -\log(\sigma(r_\theta(y_c|x) - r_\theta(y_r|x)))$
    *   An equivalent form using the softplus function is: $L(\theta) = \log(1 + e^{r_\theta(y_r|x) - r_\theta(y_c|x)})$
    *   In code, this loss is implemented as:
        ```python
        import torch.nn as nn
        rewards_chosen = model(**inputs_chosen)
        rewards_rejected = model(**inputs_rejected)
        loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
        ```
5.  **Training Strategy:** Train the reward model for a limited number of epochs (commonly 1 epoch) to prevent overfitting.

### Key Formulas (in LaTeX)

*   **Bradley-Terry Model (re-parametrized):**
    $P(i>j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)$
*   **Probability of Preference for Reward Model:**
    $P(y_c>y_r|x) = \sigma(r_\theta(y_c|x) - r_\theta(y_r|x))$
*   **Negative Log-Likelihood Loss (Bradley-Terry):**
    $L(\theta) = -\log(\sigma(r_\theta(y_c|x) - r_\theta(y_r|x)))$
*   **Equivalent Softplus Loss:**
    $L(\theta) = \log(1 + e^{r_\theta(y_r|x) - r_\theta(y_c|x)})$
*   **Preference Margin Loss (Llama 2):**
    $L(\theta) = -\log(\sigma(r_\theta(y_c|x) - r_\theta(y_r|x) - m(y_c,y_r)))$
*   **Balanced Multiple Comparisons Loss:**
    $L(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x,y_c,y_r) \sim D} \log(\sigma(r_\theta(y_c|x) - r_\theta(y_r|x)))$
*   **Plackett-Luce Probability (K-wise loss):**
    $P(\sigma_i|s_i,a_{i0},a_{i1},\dots,a_{iK-1}) = \prod_{k=0}^{K-1} \frac{\exp(r_{\theta^*}(s_i,a_{i\sigma_i(k)}))}{\sum_{j=k}^{K-1} \exp(r_{\theta^*}(s_i,a_{i\sigma_i(j)}))}$
*   **Outcome Reward Model (ORM) Cross-Entropy Loss:**
    $L_{CE}(\theta) = -\mathbb{E}_{(s,r) \sim D} [r \log p_\theta(s) + (1-r) \log(1-p_\theta(s))]$

### Key Quantitative Results and Numbers
The source does not provide specific quantitative results or numbers from experiments. It describes the methodologies and architectural patterns.

### Stated Limitations

*   **Overfitting with Multiple Comparisons:** When multiple completions per prompt are ranked, generating $\binom{K}{2}$ pairwise comparisons, naively shuffling them into the dataset can cause the reward model to overfit due to high correlation. This is mitigated by weighting loss updates or averaging comparisons from a single prompt within a batch.
*   **Preference Margin Loss Diminishing Returns:** While the preference margin loss (e.g., used in Llama 2) can incorporate the magnitude of preference, the Llama 3 team observed diminishing improvements after scaling, leading to its removal.
*   **Outcome Reward Model (ORM) Noise:** ORMs output a probability of correctness at every token. This can be a noisy process as updates and loss propagate per token, depending on outcomes and attention mappings. ORMs primarily capture final answer correctness, not reasoning errors.
*   **ORM Support:** ORMs are less supported in open-source RLHF tools compared to Bradley-Terry models.
*   **Reward Modeling as Under-explored:** Reward modeling is noted as a relatively under-explored area of RLHF, with modifications to the traditional loss not yet solidifying into a single best practice.
