Reinforcement Learning with Verifiable Rewards (RLVR) is a paradigm that optimizes Large Language Model (LLM) reasoning by providing binary or deterministic rewards based on the correctness of a final answer. Unlike reward modeling based on human preferences (RLHF), RLVR leverages ground-truth verifiers—such as compilers, mathematical solvers, or deterministic test cases—to steer models toward correct reasoning trajectories [source:arxiv:2506.14245][source:arxiv:2504.13837].

## Core Mechanisms and Optimization

RLVR typically utilizes policy gradient methods to maximize the expected verifiable reward $R$ for a given prompt $p$: $\max_{\theta} \mathbb{E}_{p \sim \mathcal{D}}[R(M_\theta(p))]$ [source:arxiv:2504.13837].

### Optimization Algorithms
Commonly employed algorithms include:
*   **PPO (Proximal Policy Optimization):** Employs a clipped surrogate objective to prevent destabilizing policy shifts:
    $$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t, \text{clip}\left(\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_t \right) \right]$$
    [source:arxiv:2504.13837].
*   **GRPO (Group Relative Policy Optimization):** Optimizes token-level log probabilities using group-normalized advantages $A_t = \frac{r_t - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ [source:arxiv:2508.14029]. This approach eliminates the need for a separate value function by estimating the baseline from a group of rollouts for the same prompt [source:arxiv:2506.14245].

### Reward Structures
*   **The "Verifiable Dot":** A binary indicator $\mathbb{I}(\text{extract}(y) = y^*)$ where the extracted final answer is matched against a ground-truth label [source:arxiv:2508.14029].
*   **Reward Chains (RLVRR):** To extend RLVR to open-ended generation where a single "dot" is unavailable, RLVRR utilizes structured reward chains derived from high-quality references [source:arxiv:2601.18533]:
    1.  **Content Rewards:** Based on the Longest Common Subsequence (LCS) between extracted key-point keywords in the rollout and a reference: $r_c = \frac{1}{K} \sum_{k=1}^K \max_{r \in \mathcal{R}} \frac{\text{LCS}(K^{r,r}_k, K^y_k)}{\max(|K^{r,r}_k|, |K^y_k|)}$ [source:arxiv:2601.18533].
    2.  **Style Rewards:** Aggregated results of verifiable Python functions checking measurable properties (e.g., formatting, length): $r_s = \sum_{i=1}^M w_i \cdot \mathbb{I}(\text{Python}_i(y) == \text{True})$ [source:arxiv:2601.18533].

## The Capacity Debate: Sampling Efficiency vs. New Capabilities

A central theoretical conflict in RLVR research is whether the technique expands the model's fundamental reasoning boundary or merely optimizes the retrieval of existing latent paths.

### The "Sampling Efficiency" View
Research suggests that RLVR may not expand reasoning capacity but instead optimizes the distribution to make correct answers more likely under greedy decoding [source:arxiv:2504.13837]. Evidence includes:
*   **Pass@k Convergence:** RLVR models outperform base models at low $k$ (e.g., Pass@1), but base models consistently surpass RLVR models as $k$ increases, suggesting the base model can solve more unique problems given a larger sampling budget [source:arxiv:2504.13837].
*   **Subset Relationship:** The set of problems solved by RLVR models is often a nearly strict subset of those solved by the base model [source:arxiv:2504.13837].
*   **Perplexity Analysis:** RLVR-generated reasoning paths align with the lower-perplexity regions of the base model's distribution, implying the paths pre-existed during pre-training [source:arxiv:2504.13837].

### The "Implicit Incentivization" View
Conversely, other evidence suggests RLVR fundamentally improves reasoning quality through **CoT-Pass@k**, a metric requiring both the final answer and the intermediate Chain-of-Thought (CoT) to be correct [source:arxiv:2506.14245].
*   **Reasoning Gap:** In CoT-Pass@k evaluations on AIME 2024 and 2025, RLVR models (e.g., DAPO-Qwen-32B) maintain a significant lead over base models (Qwen2.5-32B) even at high $k$ (up to $K=1024$), suggesting a genuine improvement in coherent reasoning [source:arxiv:2506.14245].
*   **Logic Prior:** This is supported by the "Logic Prior" assumption: $P(C_{ans}=1 \mid C_{cot}=1) > P(C_{ans}=1 \mid C_{cot}=0)$ [source:arxiv:2506.14245]. Theorem 1 in this work proves that GRPO implicitly amplifies the advantage gap between correct and incorrect CoTs, incentivizing the model to generate logically sound paths [source:arxiv:2506.14245].

### The "Evaluation Artifact" View
Some argue that reported gains are confounded by the **"RLVR Tax"**—empirical side effects that obscure true capability gains [source:arxiv:2509.21882]:
*   **Budget Mismatch:** Gains often vanish when sampling budgets are matched. For AIME-24, models like Open-RS3-1.5B saw reported scores drop from 46.70 to 30.94 under standardized budget parity [source:arxiv:2509.21882].
*   **Attempt Inflation:** RLVR models often stop abstaining from hard problems, shifting risk from "refusal" to "assertive error." This can inflate accuracy while degrading calibration, as measured by Expected Calibration Error (ECE) [source:arxiv:2509.21882].

## Data Efficiency and Scaling

### Extreme Data Compression (1-Shot RLVR)
RLVR can be effective with as little as a single training example [source:arxiv:2504.20571]. The recipe involves:
1.  Ranking examples by the variance of their historical training accuracy: $V(x_i) = \text{Var}(\{a_{i,1}, \dots, a_{i,E}\})$ [source:arxiv:2504.20571].
2.  Selecting the highest-variance example for training.
This approach enabled Qwen2.5-Math-1.5B to increase MATH500 performance from 36.0% to 73.6%, suggesting the model uses the example to "unlock" latent reasoning or correct formatting [source:arxiv:2504.20571].

### Few-Shot Guidance (FEST)
To mitigate low sample efficiency when correct rollouts are rare, FEST combines RLVR with a small (e.g., 128 samples) randomly selected SFT dataset [source:arxiv:2605.15012]. It employs a semi-online DPO loss:
$$\mathcal{L}_{DPO} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$
where expert traces ($y_w$) are preferred over agent rollouts ($y_l$). FEST-GRPO has achieved an average accuracy of $42.36 \pm 1.26$ across six math benchmarks, surpassing pure RL baselines [source:arxiv:2605.15012].

## Sustaining Learning through Self-Play and Synthesis

Standard RLVR on fixed datasets often leads to **policy entropy collapse**, where the model memorizes solutions and ceases to explore novel reasoning paths [source:arxiv:2508.14029].

### Variational Problem Synthesis (SvS)
SvS prevents collapse by generating synthetic problems that preserve the semantics and answer of original challenging problems [source:arxiv:2508.14029]. To prevent the generation of trivial, hint-heavy variants, it uses a proxy reward: positive rewards are granted only if the group accuracy for the synthetic problem falls between $12.5\%$ and $62.5\%$ [source:arxiv:2508.14029]. This method improved Pass@32 on AIME24 by 18.3% and AIME25 by 22.8% [source:arxiv:2508.14029].

### Absolute Zero (AZ)
The AZ paradigm eliminates external data dependencies entirely through a proposer-solver self-play loop [source:arxiv:2505.03335]:
1.  **Proposer:** Generates a reasoning task.
2.  **Environment:** A code executor validates the task for syntax and determinism [source:arxiv:2505.03335].
3.  **Solver:** Resolves the task via deduction, abduction, or induction.
4.  **Rewards:** The proposer is rewarded for "learnability" (tasks of moderate difficulty): $r_{learn}(t) = 4 \cdot r_{sol}(t) \cdot (1 - r_{sol}(t))$ [source:arxiv:2505.03335].
AZR-Coder-7B improved math accuracy by 15.2 points, demonstrating significant cross-domain transfer from coding to mathematics [source:arxiv:2505.03335].

## Current status and trajectory

RLVR is currently a dominant technique for reasoning models, particularly following the success of the DeepSeek-R1 series [source:arxiv:2506.14245][source:arxiv:2509.21882]. Its trajectory is evolving across three axes:
*   **From Fixed Sets to Synthesis:** Transitioning toward variational synthesis (SvS) and self-play (AZ) to overcome entropy collapse and the scalability bottleneck of human-curated data [source:arxiv:2508.14029][source:arxiv:2505.03335].
*   **From Math/Code to Open-Ended:** Generalizing verifiable rewards to non-deterministic domains using reference-based reward chains (RLVRR), which have shown superior win rates on Arena-Hard compared to RM-based GRPO [source:arxiv:2601.18533].
*   **Increased Rigor:** A shift toward "tax-aware" evaluations, emphasizing budget-matched Pass@k and calibration tracking (ECE) to distinguish between sampling efficiency and genuine capability expansion [source:arxiv:2509.21882].

## Key takeaways
*   **Capacity Conflict:** There is an ongoing debate whether RLVR optimizes sampling efficiency [source:arxiv:2504.13837] or implicitly incentivizes new reasoning paths via the Logic Prior [source:arxiv:2506.14245].
*   **Data Efficiency:** RLVR is highly efficient; 1-shot training can yield significant generalization, and few-shot guidance (FEST) can accelerate exploration [source:arxiv:2504.20571][source:arxiv:2605.15012].
*   **Stability:** Fixed-dataset RLVR is prone to entropy collapse, which can be mitigated by variational synthesis (SvS) or self-play (AZ) [source:arxiv:2508.14029][source:arxiv:2505.03335].
*   **Evaluation Rigor:** Pass@1 gains can be deceptive due to attempt inflation; budget-matched Pass@k and CoT-verification are necessary for accurate capability assessment [source:arxiv:2509.21882][source:arxiv:2506.14245].

## Related topics
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [RL for reasoning models](rl-for-reasoning.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [PPO for LLM fine-tuning (RLHF)](ppo-for-llms.md)
- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)