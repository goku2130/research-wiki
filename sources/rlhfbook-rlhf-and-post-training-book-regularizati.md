---
id: rlhfbook:rlhf-and-post-training-book-regularizati
type: web
title: 'RLHF and Post-Training Book: Regularization'
url: https://rlhfbook.com/c/15-regularization
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

# Regularization in RLHF and Post-Training

### Core Problem: Over-Optimization
In RLHF, powerful optimization tools can cause a model to drift too far from the strong, general "reference model" used in previous training stages. This leads to **over-optimization**, where the model maximizes reward at the expense of out-of-distribution performance. Manifestations of over-optimization include the generation of nonsensical text, such as repeated text, language switching, excessive special characters, or mathematically followable reasoning that leads to incorrect answers.

### Explicit Regularization: KL Divergence
The most prevalent method to prevent over-optimization is the application of a KL divergence penalty between the current policy ($\pi_{RL}$) and a reference policy ($\pi_{ref}$).

**General Formulation:**
The reward is modified by a regularization term:

$$
r = r_\theta - \lambda r_{reg}
$$

**Reference Implementation:**
The specific implementation using KL divergence is:

$$
r = r_\theta - \lambda_{KL} D_{KL}(\pi_{RL}(y|x) \| \pi_{ref}(y|x))
$$

Where the KL divergence is defined as:

$$
D_{KL}(P \| Q) = \sum_{x \in X} P(x) \log \frac{P(x)}{Q(x)}
$$

**Implementation Recipe:**
To simplify computation, the summation is converted to a Monte Carlo estimate (expectation) by sampling from the distribution $P$ (the model being trained):

$$
D_{KL}(P \| Q) = \mathbb{E}_{x \sim P} [\log P(x) - \log Q(x)]
$$

The practical implementation steps are:
1. **Generate**: Autoregressively sample a full sequence of tokens using the RL model.
2. **Forward Pass**: Run a single pass over the sequence for both the RL model and the reference model to obtain per-token logits.
3. **Log-Probabilities**: Convert logits to log-probabilities using `log_softmax`.
4. **Gather**: Extract the log-probabilities assigned to the tokens that were actually generated.
5. **Approximate KL**: Sum the token log-probabilities to get sequence-level log-probs; the difference between the RL and reference sequence log-probs approximates the KL divergence.

### Implicit Regularization: SFT vs. RL
Research indicates that RL provides implicit regularization—a resistance to catastrophic forgetting—that Supervised Fine-Tuning (SFT) lacks.

**Generalization Results:**
In a study using the **V-IRL** visual navigation task (shifting from absolute to relative directions), RL improved OOD per-step accuracy from **80.8% to 91.8%**, while SFT caused the accuracy to collapse from **80.8% to 1.3%**.

**Theoretical Mechanism (Forward vs. Reverse KL):**
The difference in behavior is attributed to the direction of KL divergence optimized:
*   **SFT (Forward KL):** Minimizes $KL(\pi^* \| \pi_\theta)$. This is **mode-covering**, meaning it penalizes the model for failing to assign probability to regions where the target has mass. In multimodal distributions, this forces the policy to redistribute mass away from "old" modes (prior knowledge) to cover the new target, causing forgetting.
*   **RL (Reverse KL):** Minimizes $KL(\pi_\theta \| \pi^*)$. This is **mode-seeking**, meaning it only penalizes the model in regions where it actually places mass. RL can shift a new mode toward the target without disturbing existing modes, preserving prior knowledge.

**RL's Razor:**
This thesis postulates that on-policy methods are inherently biased toward solutions closest to the original policy in KL divergence. Empirical data shows that forgetting is directly proportional to the KL divergence between the trained and initial policies, with a correlation of $R^2 = 0.96$.

### Other Regularization Methods
**Pretraining Gradients:**
To prevent performance regressions on general NLP tasks, an additional reward can be added to the RLHF objective to maintain higher probabilities on the pretraining corpus:

$$
J(\theta) = \mathbb{E}_{(x,y) \sim D\pi_{RL,\theta}} [r_\theta(y|x) - \lambda r_{reg}] + \gamma \mathbb{E}_{x \sim D_{pretrain}} [\log(\pi_{RL,\theta}(x))]
$$

### Limitations
The text notes that while many regularization techniques emerge in literature to stabilize experimental setups, they often disappear in subsequent model iterations as setups are simplified. The core KL distance remains the most stable and popular variant.
