---
id: arxiv:2506.18896
type: paper
title: 'ReasonFlux-PRM: Trajectory-Aware PRMs for Long Chain-of-Thought Reasoning
  in LLMs'
url: https://arxiv.org/abs/2506.18896
retrieved: '2026-07-11'
maturity: comprehensive
topic: process-vs-outcome-rewards
---

# ReasonFlux-PRM: Trajectory-Aware Process Reward Models

ReasonFlux-PRM is a trajectory-aware Process Reward Model (PRM) designed to evaluate the "trajectory-response" output format common in frontier reasoning models (e.g., DeepSeek-R1). In this format, a model generates a lengthy, often unorganized intermediate thinking trajectory followed by a concise, structured final response.

### Core Problem
Existing PRMs are primarily trained on polished, linear chain-of-thought (CoT) final responses. Consequently, they struggle to supervise intermediate thinking trajectories, which often exhibit:
1. **Branching and Backtracking:** Models may explore incorrect paths and revise assumptions.
2. **Weak Global Coherence:** Steps are often locally focused and lack narrative continuity.
3. **Structural Mismatch:** A discrepancy between the "silver-standard" thinking process and the "gold-standard" final output.

The authors demonstrate that using existing PRMs (e.g., Qwen2.5-Math-PRM-72B) to select training data from thinking trajectories can actually degrade downstream supervised fine-tuning (SFT) performance compared to human-curated baselines.

### Method and Recipe
ReasonFlux-PRM incorporates both step-level and trajectory-level supervision to align intermediate thoughts with final outcomes.

#### 1. Step-Level Reward Design
The model computes a unified step reward $r_t^{step}$ by aggregating three components via softmax-based weighting:
*   **Alignment Score ($r_t^{align}$):** Measures semantic similarity between a thinking step $s_t$ and the corresponding response step $a_t$ using a pretrained encoder $\Phi$:

$$
r_t^{align} = \text{sim}(\Phi(s_t), \Phi(a_t))
$$

*   **Quality Score ($r_t^{qual}$):** An LLM-as-a-judge (GPT-4o) evaluates the logical soundness of $s_t$ given the context.
*   **Coherence Score ($r_t^{coh}$):** Ensures contextual compatibility between adjacent steps $s_{t-1}$ and $s_t$ using a contrastive mutual information formulation:

$$
r_t^{coh} = \log \frac{\exp(\text{sim}(\Phi(s_{t-1}), \Phi(s_t))/\tau)}{\sum_{s' \in \mathcal{N}} \exp(\text{sim}(\Phi(s_{t-1}), \Phi(s'))/\tau)}
$$

    where $\mathcal{N}$ represents negative samples from unrelated trajectories.

#### 2. Trajectory-Level Reward Design
To evaluate high-level strategy, the authors use a template-guided approach:
1. An expert LLM extracts a reasoning template $\mathcal{T}$ from the trajectory-response pair.
2. A policy model $\pi_\theta$ generates $N$ responses following $\mathcal{T}$.
3. The trajectory reward $r^{final}$ is the average correctness of these $N$ responses:

$$
r^{final} = \frac{1}{N} \sum_{j=1}^N \mathbb{I}(y^{(j)} \text{ is correct})
$$

#### 3. Joint Training Objective
ReasonFlux-PRM is trained to minimize the Mean Squared Error (MSE) for both reward types:

$$
\mathcal{L} = \sum_{i=1}^N \sum_{t=1}^{T^{(i)}} \lambda_{step} (R_\phi(s_t^{(i)} \mid x^{(i)}, s_{<t}^{(i)}, a^{(i)}) - r_t^{step})^2 + \lambda_{final} (R_\phi(y^{(i)}) - r^{final})^2
$$

### Applications
*   **Offline Data Selection:** Samples are ranked by an aggregated score $\hat{r} = \frac{1}{T} \sum \hat{r}_t^{step} + \alpha \cdot \hat{r}^{final}$ to filter high-quality data for SFT.
*   **Online RL (GRPO):** The PRM provides a composite reward $\boldsymbol{r}_{new} = (1-\beta) \cdot r_{out} + \beta \cdot \hat{r}$, where $r_{out}$ is the outcome-based reward.
*   **Test-Time Scaling:** A Best-of-N strategy selects the response with the highest PRM-assigned reward.

### Key Quantitative Results
ReasonFlux-PRM-7B demonstrated superior performance across AIME, MATH500, and GPQA-Diamond benchmarks:
*   **SFT Gains:** Outperformed human-curated (s1k) data on AIME24 (40.0% vs 33.3%) and MATH500 (84.8% vs 78.8%). Overall average SFT gains were 12.1%.
*   **RL Gains:** In GRPO training for DeepSeek-R1-Distill-Qwen-7B, MATH500 accuracy increased from 89.6% (rule-based) to 94.8% (ReasonFlux-PRM-7B). Average RL gains were 4.5%.
*   **Test-Time Scaling:** Achieved an average gain of 6.3% via Best-of-N selection.
*   **Efficiency:** Fine-tuning on only 1k samples selected by ReasonFlux-PRM-7B outperformed training on 59k raw trajectories on MATH500.

### Limitations
The model relies heavily on high-quality trajectory-response pairs for training. While effective for structured logic domains like mathematics and science, the authors note that applying the framework to open-ended tasks (e.g., commonsense dialogue or code generation) would require redefining trajectory-level reward criteria and tuning the reward decomposition strategy.
