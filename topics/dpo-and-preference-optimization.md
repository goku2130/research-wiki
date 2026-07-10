---
title: Direct Preference Optimization and variants
updated: '2026-07-10'
sources:
- https://arxiv.org/abs/2305.18290
- https://arxiv.org/html/2305.18290v3
- https://huggingface.co/blog/garg-aayush/derive-dpo-loss
- https://arxiv.org/html/2402.10958
---

Direct Preference Optimization (DPO) and its derivatives provide a framework for aligning large language models without the need for an explicit reward model. These methods optimize the policy directly by leveraging the mathematical relationship between the reward function and the optimal policy [S1].

## Direct Preference Optimization (DPO)
DPO eliminates the traditional reinforcement learning from human feedback (RLHF) loop by parameterizing the reward model to extract the optimal policy in closed form [S1]. It solves the standard RLHF objective—maximizing reward subject to a KL-divergence constraint—using a simple binary cross-entropy classification loss on preference pairs [S1]. 

The mathematical foundation of DPO relies on the Bradley-Terry model, which defines the probability of preference as $P(y_w > y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$ [S2]. The optimal policy is derived as:
$$\pi^*(y|x) = \frac{\pi_{ref}(y|x) \exp(r^*(x,y)/\beta)}{Z(x)}$$
where $Z(x)$ is the partition function [S2]. By substituting this closed-form policy back into the Bradley-Terry model, the reward difference cancels out $Z(x)$, allowing the preference probability to be expressed directly as a function of the policy $\pi_\theta$ [S2]. 

The resulting DPO loss is formulated as:
$L_{DPO} = -\log \sigma(\beta \log(\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}) - \beta \log(\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$ [S3].

Because this objective does not require sampling from the language model during fine-tuning, DPO is more stable, performant, and computationally lightweight than PPO-based RLHF [S1], [S3].

## Why Reward Models are Skipped
DPO and its variants skip the reward model because the language model itself acts as a reward model [S1]. Specifically, the Bradley-Terry model depends only on the difference between rewards; therefore, adding any function $f(x)$ that depends solely on the prompt does not alter preference probabilities [S3]. By expressing the reward in terms of the policy and a reference model, the optimization can be performed directly on the policy parameters $\theta$ without ever explicitly training or querying a separate reward scalar [S2].

## DPO Variants

### Identity Preference Optimization (IPO)
IPO is designed to mitigate overfitting issues found in DPO, where the model may push the reward gap too wide when fitting the preference dataset [S4]. To improve out-of-distribution generalization, IPO replaces the binary cross-entropy loss with a margin-based squared hinge loss:
$\max(0, \beta \log(\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}) - \beta \log(\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}) - 1)^2$ [S4].

### Odds Ratio Preference Optimization (ORPO)
ORPO simplifies the alignment pipeline by combining the language modeling loss and preference optimization into a single monolithic objective [S4]. Unlike DPO, ORPO does not require a reference model [S4].

### Kahneman-Tversky Optimization (KTO)
KTO utilizes Prospect Theory to align models using individual samples rather than paired preference data [S4]. This eliminates the requirement for the preference pairs ($y_w, y_l$) used in DPO [S4].

## Key takeaways
* **DPO** replaces the RLHF reward model and PPO loop with a binary cross-entropy loss on preference pairs [S1].
* **Closed-form derivation** allows the reward function to be substituted by the policy, canceling the partition function $Z(x)$ [S2].
* **IPO** uses a squared hinge loss to prevent overfitting and improve generalization [S4].
* **ORPO** eliminates the reference model by merging LM loss and preference optimization [S4].
* **KTO** removes the need for paired data by optimizing based on individual samples via Prospect Theory [S4].

## References
- [S1] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [S2] [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (HTML)](https://arxiv.org/html/2305.18290v3)
- [S3] [Deriving the DPO Loss from First Principles](https://huggingface.co/blog/garg-aayush/derive-dpo-loss)
- [S4] [Relative Preference Optimization: Enhancing LLM Alignment through Contrasting Responses](https://arxiv.org/html/2402.10958)
