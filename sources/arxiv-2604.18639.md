---
id: arxiv:2604.18639
type: paper
title: 'Easy Samples Are All You Need: Self-Evolving LLMs via Data-Efficient Reinforcement
  Learning'
url: https://arxiv.org/abs/2604.18639
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

The core problem addressed is the prohibitive annotation cost and instability (e.g., model collapse, reward hacking) inherent in existing supervised and unsupervised reinforcement learning paradigms for post-training large language models. Drawing inspiration from Vygotsky’s Zone of Proximal Development, the authors propose a data-efficient self-evolving framework that enables LLMs to transition from limited easy labeled data to increasingly difficult unlabeled reasoning tasks without continuous human supervision.

The methodology follows a three-stage recipe. First, **Knowledge Transfer** initializes a stable policy prior $\pi_{warm}$ by applying supervised Group Relative Policy Optimization (GRPO) to a small set of easy labeled samples. Second, a **Divide-and-Conquer** strategy pseudo-labels abundant difficult unlabeled data by partitioning queries based on output uncertainty. Low-uncertainty samples are retained via consistency-based selection, where $N$ independent inferences yield identical outputs. Medium-uncertainty samples undergo reflection-based resolution, filtered by a dynamic entropy threshold $\tau_t$ (default 0.3). High-uncertainty samples are deferred. Third, **Difficulty-Progressive Self-Training** iteratively combines the labeled dataset with the selected pseudo-labeled samples. The model is updated via RL in each iteration, progressively exposing it to harder, previously deferred samples until convergence.

The optimization and selection mechanisms are formalized through the following key equations. The GRPO objective is:

$$
\mathcal{J}_{\mathrm{GRPO}} = \mathbb{E}_{q \sim Q, \{x_i\}_{i=1}^G \sim \pi_\theta(X|q)} \left[ \frac{1}{G} \sum_{i=1}^G \left(\min (A_i, \operatorname{clip}(A_i, 1-\epsilon, 1+\epsilon)) - \beta \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})\right) \right],
$$

where the normalized advantage is $A_i=\frac{r_i-\text{mean}(\{r_1,\cdots,r_G\})}{\text{std}(\{r_1,\cdots,r_G\})}$. The correctness reward incorporates a format penalty:

$$
r_i = \begin{cases} 1, & \text{if } \operatorname{verifier}(x_i, a) = \text{True}, \\ -0.5, & \text{if } x_i \text{ is not in boxed format}, \\ 0, & \text{otherwise}. \end{cases}
$$

Consistency-based selection retains samples where $o_1 = o_2 = \dots = o_N$. For reflection-based resolution, prediction entropy is computed as $H(x) = -\sum_{y \in \mathcal{O}_x} p_x(y) \log p_x(y)$, with resolution applied if $H(x) \leq \tau_t$. The iterative policy update is defined as $\pi_{i+1} = \mathrm{RL}(\mathcal{D}_{\text{label}} \cup \mathcal{D}_{\text{unlabel\_selected}}^{(i)})$.

Quantitative evaluations on mathematical and scientific benchmarks demonstrate significant data efficiency and self-evolution. Using only 10% of easy labeled data (2,000 of 20,000 DeepMath-103K samples), EasyRL achieves average scores of 40.3 on Qwen2.5-Math-1.5B and 50.6 on Qwen2.5-Math-7B, surpassing supervised GRPO trained on 100% of the data (39.5 and 49.5, respectively). The method consistently outperforms supervised and unsupervised baselines, yielding average improvements of 7.7% (1.5B), 12.1% (7B), and 3.5% (Llama-3.2-3B-Instruct). Furthermore, it exhibits strong out-of-domain generalization, transferring to scientific reasoning tasks with average gains of 17.9%, 6.5%, and 8.7% across the respective backbones. Analysis confirms a self-evolving trajectory: pseudo-label consistency and accuracy steadily increase across iterations, while the average difficulty of selected samples rises (e.g., reaching 7.0282 for the 1.5B model and 7.4256 for the 7B model by iteration three).

The authors acknowledge several limitations. The framework is currently restricted to domains with verifiable rewards, such as mathematical reasoning. Extending the paradigm to open-ended or subjective tasks like creative writing and scientific research remains challenging. Additionally, as with any self-evolving system, there is a risk of reinforcing biased behaviors, necessitating future integration of fairness-aware reward functions and continuous human oversight to maintain alignment with societal norms.
