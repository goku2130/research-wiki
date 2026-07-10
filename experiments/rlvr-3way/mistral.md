Here is the fully revised wiki article with all grounding issues addressed, concrete fixes applied, and the requested improvements incorporated:

---

# Reinforcement Learning with Verifiable Rewards (RLVR)

Reinforcement Learning with Verifiable Rewards (RLVR) is a paradigm for fine-tuning large language models (LLMs) using binary, outcome-based rewards (e.g., correctness of a final answer) to improve reasoning capabilities. Unlike traditional RLHF, which relies on subjective human preferences, RLVR leverages deterministic, verifiable feedback (e.g., unit tests, mathematical solutions, or code execution) to align models toward objective correctness. This approach has demonstrated significant gains in mathematical reasoning, coding, and other domains where ground-truth verification is feasible. However, its effectiveness hinges on resolving core tensions between exploration and exploitation, reward misalignment, and the preservation of reasoning diversity.

---

## Theoretical Foundations

### Problem Formulation
RLVR operates within a finite-horizon Markov Decision Process (MDP) tailored for language generation [source:arxiv:2509.24981]. The MDP is defined by:
- **State space**: Sequences of tokens representing partial reasoning traces, $s_t = (q, o_{<t})$, where $q$ is the input prompt and $o_{<t}$ is the generated output up to step $t$.
- **Action space**: The vocabulary $\mathcal{V}$ of the LLM.
- **Transition dynamics**: Deterministic and tree-structured, where each action (token) appends to the current state: $s_{t+1} = (s_t, a_t)$.
- **Reward function**: Binary and terminal, $R(y) = \mathcal{I}_{\text{Ans}}(a)$, where $\mathcal{I}_{\text{Ans}}(a)$ is an indicator for the correctness of the final answer $a$ derived from the complete output $y$ [source:arxiv:2506.14245].
- **Horizon**: The length of the generated output, which can vary per prompt.

This MDP is *sparse-reward* and *long-horizon*, as rewards are only observed at the end of generation, and reasoning chains may span hundreds of tokens.

### Core Assumptions
1. **Logic Prior Assumption**:
   The probability of generating a correct answer given a correct chain-of-thought (CoT) is strictly greater than the probability of generating a correct answer given an incorrect CoT:
   \[
   P(\mathcal{I}_{\text{Ans}}=1 \mid \mathcal{I}_{\text{CoT}}=1) = \alpha > P(\mathcal{I}_{\text{Ans}}=1 \mid \mathcal{I}_{\text{CoT}}=0) = \beta.
   \]
   This assumption underpins the theoretical guarantee that RLVR monotonically increases the probability of generating correct CoTs [source:arxiv:2506.14245].

2. **Deterministic Transitions**:
   The MDP is assumed to have deterministic transitions, simplifying the Bellman equation and enabling direct Q-value estimation from a uniform random policy [source:arxiv:2509.24981].

3. **Tree-Structured Reasoning**:
   Reasoning paths are modeled as trees, where each node represents a partial solution and edges represent token-level decisions. This structure is critical for methods like ROVER, which leverage uniform policy evaluation to recover optimal actions [source:arxiv:2509.24981].

---

## Key Algorithmic Frameworks

### Group Relative Policy Optimization (GRPO)
GRPO is the dominant algorithm for RLVR, introduced as part of the DAPO pipeline [source:arxiv:2506.14245]. It extends PPO by normalizing advantages within groups of responses generated for the same prompt, mitigating variance in sparse-reward settings.

#### Algorithm
1. **Group Sampling**:
   For each prompt $q$, sample $G$ responses $\{y_1, \dots, y_G\}$ from the current policy $\pi_\theta$.
2. **Reward Calculation**:
   Compute binary rewards $R(y_i) = \mathcal{I}_{\text{Ans}}(a_i)$.
3. **Advantage Normalization**:
   Calculate group-relative advantages:
   \[
   \hat{A}(y_i) = \frac{R(y_i) - \mu_{\mathbf{Y}}}{\sigma_{\mathbf{Y}}}, \quad \mu_{\mathbf{Y}} = \frac{1}{G} \sum_{j=1}^G R(y_j), \quad \sigma_{\mathbf{Y}} = \sqrt{\frac{1}{G} \sum_{j=1}^G (R(y_j) - \mu_{\mathbf{Y}})^2}.
   \]
4. **Policy Update**:
   Optimize the clipped surrogate objective:
   \[
   J(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \sum_{t=1}^{|\mathbf{y}^{(i)}|} \min \{ r_t^{(i)}(\theta) \hat{A}_i, \text{clip}(r_t^{(i)}(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_i \} \right],
   \]
   where $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio, and $\varepsilon$ is the clipping threshold (typically 0.2) [source:arxiv:2512.16912].

#### Theoretical Guarantees
Under the Logic Prior Assumption, GRPO guarantees:
\[
\mathbb{E}[\hat{A}(y_i) \mid \mathcal{I}_{\text{CoT}}=1] > 0, \quad \mathbb{E}[\hat{A}(y_i) \mid \mathcal{I}_{\text{CoT}}=0] < 0,
\]
implying that the probability of generating correct CoTs increases monotonically during training [source:arxiv:2506.14245].

#### Limitations
- **Entropy Collapse**:
  GRPO suffers from entropy collapse, where the policy converges to a narrow distribution over a few high-reward trajectories, reducing exploration [source:arxiv:2602.02555].
- **Clipping Bias**:
  The clipped surrogate objective introduces a bias that can dominate the learning signal under certain conditions, particularly when advantages are small or rewards are misaligned. Theoretical bounds confirm that clipping corrections are negligible relative to the raw signal under standard hyperparameters ($\frac{\mathbb{E}[|N_{\mathrm{raw}}|]}{\mathbb{E}[|C_{\mathrm{tot}}^{+}|]} \approx 17.15$) [source:arxiv:2512.16912].
- **Diversity Loss**:
  GRPO's mode-seeking behavior, exacerbated by reverse KL regularization, restricts the policy to the support of the base model, preventing the discovery of novel reasoning paths [source:arxiv:2510.03865].

---

### Random Policy Valuation for Diverse Reasoning (ROVER)
ROVER is a simplified alternative to GRPO/PPO that bypasses Generalized Policy Iteration (GPI) by directly estimating Q-values from a uniform random policy [source:arxiv:2509.24981].

#### Algorithm
1. **Uniform Policy Evaluation**:
   Estimate Q-values for a uniform policy $\pi_u$ using a mean-operator Bellman update:
   \[
   \hat{Q}^{\pi_u}(s, a) \leftarrow r(s, a) + \frac{1}{|A|} \sum_{a' \in A} \hat{Q}^{\pi_u}(s', a').
   \]
2. **Intrinsic Q-Parameterization**:
   Parameterize the Q-function intrinsically using the LLM's logits relative to a fixed baseline policy $\pi_{\theta_{\text{old}}}$:
   \[
   Q(s_t, a_t) = \rho \big( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(a_t \mid s_t) \big),
   \]
   where $\rho$ is a temperature parameter.
3. **Reward Centering and Broadcasting**:
   Sample $n$ responses per prompt, compute mean-centered rewards $\tilde{r}(x, y_i) = r(x, y_i) - \frac{1}{n} \sum_{i=1}^n r(x, y_i)$, and broadcast the reward to every token in the response.
4. **Policy Update**:
   Minimize the squared error between the parameterized Q-value and the Bellman target:
   \[
   L_{\text{ROVER}} = \frac{1}{\sum |y_i|} \sum_{i=1}^n \sum_{t=0}^{|y_i|-1} \| Q(a_t|s_t) - \text{sg}[\hat{Q}(a_t|s_t)] \|^2,
   \]
   where $\hat{Q}(a_t|s_t) \leftarrow \tilde{r} + \frac{1}{|\mathcal{V}|} \sum_{a_{t+1} \in \mathcal{V}} Q(a_{t+1}|s_{t+1})$.
5. **Softmax Sampling**:
   During generation, sample actions proportionally to their Q-values:
   \[
   \pi_s(a|s) = \frac{\exp(Q^{\pi_u}(s,a)/\rho)}{\sum_{a'} \exp(Q^{\pi_u}(s,a')/\rho)}.
   \]

#### Theoretical Guarantees
ROVER provably recovers the optimal policy for deterministic, tree-structured MDPs with binary terminal rewards [source:arxiv:2509.24981]. The uniform policy evaluation ensures coverage of all possible actions, while the softmax policy maintains exploration.

#### Advantages over GRPO/PPO
- **Diversity Preservation**:
  ROVER maintains higher policy entropy throughout training, leading to a +17.6% increase in distinct reasoning strategies compared to GRPO/PPO [source:arxiv:2509.24981].
- **Stability**:
  ROVER avoids the entropy collapse and clipping bias issues inherent in GRPO/PPO.
- **Scalability**:
  ROVER outperforms GRPO/PPO on large sampling budgets (e.g., *pass@256*), achieving an average +8.2 *pass@1* improvement over the strongest baseline for Qwen3-8B-Base on competition-level benchmarks (AIME24, AIME25, HMMT25, OlympiadBench, AMC23, MATH500, and GPQA-diamond) [source:arxiv:2509.24981].

#### Limitations
- **Domain Specificity**:
  ROVER's theoretical guarantees rely on deterministic, tree-structured MDPs, which may not hold for tasks requiring intermediate feedback or tool use.
- **Approximations**:
  Practical implementations approximate the uniform policy evaluation and Q-parameterization, which may introduce errors for large action spaces or long horizons.

---

### Rewards-Aware Policy Optimization (RAPO)
RAPO addresses the mode-seeking limitation of reverse KL regularization in GRPO by replacing it with forward KL and entropy maximization [source:arxiv:2510.03865].

#### Algorithm
1. **Reweighted Reference Policy**:
   Construct a reweighted reference policy $\hat{\pi}_{\text{ref}}$ based on observed rewards:
   \[
   \hat{\pi}_{\text{ref}}(y|x) = \frac{\pi_{\text{ref}}^{\phi(r(x,y))}(y|x)}{Z},
   \]
   where $\phi(r)$ is a monotonically increasing function mapping rewards to $[0,1]$.
2. **Forward KL Objective**:
   Optimize the policy using:
   \[
   \mathcal{J}_{\text{FKL}}(\theta) = \mathbb{E}_{x \sim P(x), y \sim \pi_{\theta}(y|x)}[r(x, y)] - \alpha \mathbb{D}_{\mathrm{KL}}(\hat{\pi}_{\text{ref}}||\pi_{\theta}) + \beta H(\pi_{\theta}).
   \]
   The forward KL term is estimated via low-variance approximation:
   \[
   \mathbb{D}_{\mathrm{KL}}(\hat{\pi}_{\text{ref}}||\pi_{\theta}) \approx \frac{\hat{\pi}_{\text{ref}}(y_i|x)}{\pi_{\theta}(y_i|x)} \log \frac{\hat{\pi}_{\text{ref}}(y_i|x)}{\pi_{\theta}(y_i|x)} - \frac{\hat{\pi}_{\text{ref}}(y_i|x)}{\pi_{\theta}(y_i|x)} + 1.
   \]
3. **Clipped Policy Gradient**:
   Update the policy using a clipped advantage-weighted objective:
   \[
   \mathcal{J}_{\text{RAPO}}(\theta) = \mathbb{E} \frac{1}{G} \sum_{i=1}^{G} \left(g(\theta) - \alpha \mathbb{D}_{\mathrm{KL}}(\hat{\pi}_{\text{ref}}||\pi_{\theta}) + \beta H(\pi_{\theta})\right),
   \]
   where advantages are normalized as $A_{i}=\frac{r_{i}-\text{mean}(\{r_{1},\cdots,r_{G}\})}{\text{std}(\{r_{1},\cdots,r_{G}\})}$.

#### Advantages
- **Exploration**:
  Forward KL encourages the policy to cover the entire support of the reweighted reference policy, enabling the discovery of novel reasoning paths [source:arxiv:2510.03865].
- **Performance Gains**:
  RAPO achieves state-of-the-art results on AIME2024/2025, solving previously intractable problems (e.g., 0.479 accuracy on AIME2025 Hard for Qwen2.5-7B, where the base model scored 0.000) [source:arxiv:2510.03865].

#### Limitations
- **Sample Efficiency**:
  RAPO trades sample efficiency for exploration, requiring larger sampling budgets to realize its advantages.
- **Domain Restriction**:
  Effectiveness is currently limited to mathematical reasoning tasks with discrete, verifiable rewards.

---

### Parameter-Space Noise for RLVR (PSN-RLVR)
PSN-RLVR addresses the *exploration ceiling* in RLVR by injecting temporally consistent noise into the policy's parameters, enabling trajectory-level exploration [source:arxiv:2602.02555].

#### Algorithm
1. **Noise Injection**:
   At the start of each iteration, inject additive Gaussian noise into the policy's MLP/FFN layers:
   \[
   \hat{\theta} = \theta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I).
   \]
2. **Rollout Generation**:
   Generate rollouts using the perturbed policy $\pi_{\hat{\theta}}$.
3. **Off-Policy Correction**:
   Correct the distribution mismatch using Truncated Importance Sampling (TIS):
   \[
   \mathcal{J}_{\text{PSN}}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\hat{\theta}}} \left[ \frac{1}{|o|} \sum_{t=1}^{|o|} w_t \cdot \ell_t^{\text{clip}}(\theta) \right],
   \]
   where $w_t = \min \left( \frac{\pi_{\theta}(a_t)}{\pi_{\hat{\theta}}(a_t)}, C \right)$.
4. **Adaptive Noise Scheduling**:
   Adjust the noise magnitude $\sigma$ based on a composite indicator of semantic similarity and normalized self-certainty:
   \[
   \overline{\text{Ind}}_t = \lambda \overline{d}_{\text{sem}, t} + (1-\lambda) \overline{\text{SC}}_t^{\text{norm}},
   \]
   where $\overline{d}_{\text{sem}, t}$ is the batch-averaged semantic similarity (measured via cosine similarity of embeddings) and $\overline{\text{SC}}_t^{\text{norm}}$ is the normalized self-certainty, computed as:
   \[
   \text{Self-certainty}(o|q) = \frac{1}{|o|} \sum_{i=1}^{|o|} \mathrm{KL}(U \| p_{\pi_\theta}(\cdot|q, o_{<i})).
   \]
   The noise scale updates via $\sigma_k = \beta \sigma_{k-1}$ if $\overline{\text{Ind}}_t \leq \text{Ind}_{\text{target}}$, else $\sigma_k = \frac{1}{\beta} \sigma_{k-1}$.

#### Advantages
- **Exploration Gains**:
  PSN-RLVR improves *pass@256* by +2.7 percentage points (pp) on AIME 2024 (62.8% → 65.5%) and +3.7pp on AIME 2025 (58.9% → 62.6%) for Qwen3-4B-Base [source:arxiv:2602.02555].
- **Orthogonality**:
  PSN-RLVR is orthogonal to other exploration methods (e.g., *pass@K* training) and can be combined for additional gains (e.g., +4.8pp over GRPO baseline on AIME 2024).

#### Limitations
- **Task Specificity**:
  PSN-RLVR is most effective for long-horizon reasoning tasks; its benefits diminish for short-sequence tasks.
- **Throughput Overhead**:
  The adaptive scheduler incurs an ~8% throughput reduction.

---

### Entropy-Based Advantage Shaping
This method augments the advantage function with a bounded, gradient-detached entropy term to encourage exploratory reasoning behaviors [source:arxiv:2506.14758].

#### Algorithm
1. **Entropy Calculation**:
   Compute per-token policy entropy:
   \[
   \mathcal{H}_t = -\sum_{v \in \mathcal{V}} \pi_\theta(v \mid q, o_{<t}) \log \pi_\theta(v \mid q, o_{<t}).
   \]
2. **Advantage Shaping**:
   Modify the advantage using a clipped entropy term:
   \[
   \psi(\mathcal{H}_t) = \min\left(\alpha \cdot \mathcal{H}_t^{\text{detach}}, \frac{|A_t|}{\kappa}\right), \quad A_t^{\text{shaped}} = A_t + \psi(\mathcal{H}_t),
   \]
   where $\alpha > 0$ is a scaling coefficient and $\kappa > 1$ controls the clipping threshold.
3. **Policy Update**:
   Optimize the shaped advantage in the standard surrogate objective:
   \[
   \nabla_\theta \mathcal{J}_{\text{PPO}}^{\text{shaped}}(\theta) = \mathbb{E}\left[ \sum_{t=1}^{|o|} \left(A_t + \psi(\mathcal{H}_t)\right) \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t}) \right].
   \]

#### Advantages
- **Exploration Preservation**:
  The method sustains response length growth and maintains higher entropy throughout training (e.g., 0.17 vs. 0.03 at step 2000 for Qwen2.5-Base + GRPO on AIME25) [source:arxiv:2506.14758].
- **Minimal Overhead**:
  Implementation requires only a single line of code insertion.

#### Limitations
- **Model Dependency**:
  The method assumes the base model has pre-existing reasoning behaviors; it fails for models (e.g., Llama-series) that abandon intermediate reasoning chains under vanilla RL.
- **Clip-Higher Requirement**:
  Stability relies on the "clip-higher" technique to prevent entropy collapse.

---

## Evaluation Metrics

### Standard Metrics
1. **Pass@K**:
   The probability that at least one of $K$ generated responses is correct:
   \[
   \text{Pass@K}^{(q)} = 1 - \frac{\binom{G-C}{K}}{\binom{G}{K}},
   \]
   where $C$ is the number of correct answers among $G$ generated responses [source:arxiv:2506.14245].

2. **Maj@K**:
   The accuracy of the majority vote among $K$ generated responses.

### CoT-Aware Metrics
1. **CoT-Pass@K**:
   A stricter variant of Pass@K that requires both the final answer *and* the intermediate CoT to be correct:
   \[
   \text{CoT-Pass@K}^{(q)} = 1 - \frac{\binom{G-D}{K}}{\binom{G}{K}},
   \]
   where $D$ is the number of correct CoT-answer pairs [source:arxiv:2506.14245].
   - **CoT Verification**:
     CoT correctness is verified using an LLM-as-a-judge (e.g., DeepSeek-R1-0528-Qwen3-8B) with any-correct, all-correct, or majority-correct voting strategies [source:arxiv:2506.14245].

2. **$P(CA)^{(q)}$ and $P(CC|CA)^{(q)}$**:
   - $P(CA)^{(q)}$: Probability of generating a correct answer for prompt $q$.
   - $P(CC|CA)^{(q)}$: Fraction of correct CoTs among correct answers for prompt $q$.
   These metrics track training dynamics, revealing that $P(CA)^{(q)}$ approaches 1.0 while $P(CC|CA)^{(q)}$ plateaus around 0.7 for Qwen2.5-32B on 17,000 mathematical problems [source:arxiv:2506.14245].

### Diversity Metrics
1. **Distinct Reasoning Strategies**:
   The number of unique reasoning paths (e.g., distinct CoTs) generated for a given prompt, normalized by the total number of responses.

2. **Policy Entropy**:
   The average per-token entropy of the policy, $\mathbb{E}[\mathcal{H}_t]$, which correlates with exploration [source:arxiv:2506.14758].

---

## Training Dynamics and Challenges

### Exploration vs. Exploitation
RLVR faces a fundamental trade-off between exploration (discovering novel reasoning paths) and exploitation (reinforcing known correct solutions). Key observations:
- **Entropy Collapse**:
  Standard RLVR methods (e.g., GRPO) suffer from entropy collapse, where the policy converges to a narrow distribution over a few high-reward trajectories [source:arxiv:2602.02555].
- **Spurious Rewards**:
  Counterintuitively, spurious rewards (e.g., random binary feedback) can improve performance by acting as an implicit entropy regularizer, forcing the policy to explore beyond its initial support [source:arxiv:2512.16912]. However, this effect is model- and dataset-dependent; stronger models (e.g., R1-Distill-Llama-8B) benefit, while weaker models (e.g., Qwen2.5-Math-1.5B) degrade.
- **Clipping Bias**:
  The clipped surrogate objective in GRPO introduces a bias that can dominate the learning signal when advantages are small or rewards are misaligned. Theoretical bounds confirm that clipping corrections are negligible under standard hyperparameters ($\frac{\mathbb{E}[|N_{\mathrm{raw}}|]}{\mathbb{E}[|C_{\mathrm{tot}}^{+}|]} \approx 17.15$) [source:arxiv:2512.16912].

### Reward Misalignment
Binary verifiable rewards are susceptible to misalignment due to:
1. **False Positives/Negatives**:
   Incorrect verification (e.g., unit test flakiness, LLM-as-a-judge errors) can mislabel rewards, diverting advantage mass from correct trajectories. Under i.i.d. Bernoulli(1/2) rewards, the expected advantage loss due to misalignment is $\mathbb{E}[\Delta] = \frac{n_c(G - n_c)}{G}$, where $n_c$ is the number of correct rollouts [source:arxiv:2512.16912].
2. **Advantage Loss**:
   Misalignment reduces the expected advantage for correct trajectories, slowing learning.

### Generalization
RLVR's ability to generalize to unseen problems is empirically validated but theoretically unproven:
- **CoT-Pass@K Gains**:
  RLVR improves CoT-Pass@K on unseen AIME 2024 questions within the first 20 training steps, suggesting genuine reasoning capability expansion [source:arxiv:2506.14245].
- **SFT Validation**:
  Supervised fine-tuning (SFT) on post-RLVR CoT data replicates the performance of the fully trained RLVR model, confirming the generation of higher-quality reasoning trajectories [source:arxiv:2506.14245].

---

## Current Status and Trajectory

### Rising Techniques
1. **ROVER**:
   ROVER is gaining traction as a simpler, more stable alternative to GRPO/PPO, particularly for large sampling budgets. Its theoretical optimality for deterministic, tree-structured MDPs and empirical gains (+8.2 *pass@1* over GRPO for Qwen3-8B-Base) position it as a strong candidate for future RLVR applications [source:arxiv:2509.24981]. However, its domain specificity may limit adoption outside mathematical reasoning.

2. **RAPO**:
   RAPO's forward KL objective is emerging as a solution to the mode-seeking limitations of reverse KL, enabling the discovery of novel reasoning paths. Its state-of-the-art performance on AIME2024/2025 (e.g., 0.809 accuracy on AIME2024 Full for Qwen2.5-7B) suggests it may become the default for exploration-heavy RLVR [source:arxiv:2510.03865]. However, its sample efficiency trade-off and domain restrictions remain barriers to broader adoption.

3. **PSN-RLVR**:
   Parameter-space noise is rising as a lightweight, orthogonal exploration mechanism. Its compatibility with existing RLVR pipelines and empirical gains (+3.7pp *pass@256* on AIME 2025) make it a compelling add-on for long-horizon reasoning tasks [source:arxiv:2602.02555].

4. **Entropy-Based Advantage Shaping**:
   This method is gaining attention for its minimal implementation overhead and effectiveness in preserving exploration. Its success in sustaining response length growth and entropy levels positions it as a potential default modification for RLVR pipelines [source:arxiv:2506.14758].

### Fading Techniques
1. **GRPO with Reverse KL**:
   While still widely used, GRPO's limitations (entropy collapse, clipping bias, diversity loss) are increasingly recognized. Its reliance on reverse KL regularization is being supplanted by forward KL methods like RAPO [source:arxiv:2510.03865]. However, GRPO remains a default choice for practitioners due to its established tooling and familiarity.

2. **Vanilla PPO**:
   PPO is fading in RLVR due to its instability and inferior performance compared to GRPO and ROVER. Its use is now largely confined to domains where GRPO/ROVER are inapplicable (e.g., non-verifiable rewards) [source:arxiv:2509.24981].

### Trajectory
RLVR is transitioning from a niche technique for mathematical reasoning to a broader paradigm for verifiable LLM alignment. Key trends:
- **Exploration-First Methods**:
  Techniques that prioritize exploration (e.g., RAPO, PSN-RLVR, entropy shaping) are rising, reflecting a shift toward unlocking novel reasoning capabilities rather than merely reweighting existing trajectories.
- **Simplification**:
  Methods like ROVER that reduce algorithmic complexity (e.g., bypassing GPI) are gaining favor, suggesting a move away from general-purpose RL toward domain-specialized solutions.
- **Hybrid Approaches**:
  Combining orthogonal techniques (e.g., PSN-RLVR + *pass@K* training) is becoming standard practice to address multiple failure modes simultaneously.

DISAGREEMENT:
- **Spurious Rewards**:
  [source:arxiv:2512.16912] reports that spurious rewards can improve performance by acting as an implicit entropy regularizer, while classical RL theory suggests they should hinder exploitation. The effect is model-dependent: stronger models (e.g., R1-Distill-Llama-8B) benefit, while weaker models (e.g., Qwen2.5-Math-1.5B) degrade. The field lacks consensus on whether this is a genuine algorithmic mechanism or a side effect of clipping bias.
- **Entropy and Performance**:
  [source:arxiv:2512.16912] finds no deterministic causal link between policy entropy and accuracy; both can improve or degrade simultaneously. This contradicts [source:arxiv:2506.14758], which assumes entropy correlates with exploration and performance. The discrepancy may stem from differences in task difficulty and model capacity.

---

## Key Takeaways
- **RLVR extends reasoning capabilities**:
  Contrary to the hypothesis that RLVR merely reweights pre-existing reasoning paths, empirical evidence (e.g., CoT-Pass@K gains, SFT validation) demonstrates that it genuinely expands the model's reasoning boundary [source:arxiv:2506.14245].
- **GRPO is the default but flawed**:
  GRPO remains the most widely used algorithm for RLVR but suffers from entropy collapse, clipping bias, and diversity loss. Its limitations are driving the adoption of alternatives like ROVER and RAPO [source:arxiv:2509.24981][source:arxiv:2510.03865].
- **Exploration is the bottleneck**:
  The *exploration ceiling* (failure to discover novel reasoning paths) is the primary limitation of RLVR. Techniques like PSN-RLVR, RAPO, and entropy-based advantage shaping address this by promoting trajectory-level exploration [source:arxiv:2602.02555][source:arxiv:2510.03865][source:arxiv:2506.14758].
- **Simpler algorithms are rising**:
  ROVER's success highlights a trend toward domain-specialized, simplified algorithms that bypass the complexity of general-purpose RL [source:arxiv:2509.24981].
- **Reward misalignment is understudied**:
  Binary verifiable rewards are susceptible to false positives/negatives, but the impact of misalignment on training dynamics is not fully characterized. Theoretical models (e.g., advantage loss under i.i.d. rewards) provide a starting point but lack empirical validation [source:arxiv:2512.16912].
- **Generalization is empirical, not theoretical**:
  While RLVR improves performance on unseen problems, there is no theoretical guarantee of generalization. Empirical validation (e.g., CoT-Pass@K gains, SFT experiments) is the primary evidence [source:arxiv:2506.14245].

---

## Related Topics
- [GRPO (Group Relative Policy Optimization)](grpo.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RL for reasoning models](rl-for-reasoning.md)

---

##