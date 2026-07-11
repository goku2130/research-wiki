---
title: Nash and game-theoretic preference optimization
maturity: developing
updated: '2026-07-11'
sources:
- arxiv:2312.00886
- arxiv:2405.00675
- arxiv:2509.23102
open_questions:
- Can the regression approximation in SPPO ($\frac12 \approx \log Z_{\pi_t}(x)$) be
  replaced with a learned baseline to eliminate variance-reduction bias without increasing
  sample complexity?
- Does HT-MNPO converge to a meaningful equilibrium in heterogeneous games, or does
  it cycle indefinitely? What minimal structural assumptions (e.g., potential games,
  monotonicity) would restore guarantees?
- How do preference-model architectures (e.g., cross-encoder vs. generative judge)
  affect the geometry of the induced preference game and the quality of the resulting
  Nash equilibrium?
- Is there a practical criterion to detect when human preferences are sufficiently
  non-transitive to warrant game-theoretic methods over standard reward-model RLHF?
---

Nash and game-theoretic preference optimization reframes language model alignment as finding equilibria in preference games rather than maximizing a scalar reward, directly addressing intransitivity and heterogeneity in human feedback. By replacing reward models with preference oracles and optimizing policies against populations of opponents, these methods aim to produce more robustly aligned models without the pathologies of reward over-optimization.

## Foundations: Preference Games and Nash Equilibria

Classical RLHF fits a Bradley–Terry (BT) model $P(y \succ y' \mid x) = \sigma(r(x,y) - r(x,y'))$ to pairwise preferences, implicitly assuming transitivity and a latent scalar reward $r$ [source:arxiv:2312.00886]. This assumption fails when human preferences exhibit cycles (e.g., $y_1 \succ y_2$, $y_2 \succ y_3$, $y_3 \succ y_1$) or context-dependent trade-offs that no single reward function can capture [source:arxiv:2405.00675]. Game-theoretic approaches instead treat the preference oracle $\mathbb{P}(y \succ y' \mid x)$ as the primitive object and seek a policy $\pi^*$ that is a **Nash equilibrium** of the two-player constant-sum game defined by the payoff $\mathcal{P}(\pi, \pi') = \mathbb{E}_{x\sim\rho, y\sim\pi, y'\sim\pi'}[\mathbb{P}(y \succ y' \mid x)]$:

$$
\pi^* \in \arg\max_{\pi \in \Pi} \min_{\pi' \in \Pi} \mathcal{P}(\pi, \pi')
$$

A Nash equilibrium guarantees that no unilateral deviation can improve the win-rate against the opponent population, making it a natural solution concept for non-transitive preferences [source:arxiv:2312.00886]. Regularization toward a reference policy $\mu$ (typically the SFT model) is incorporated via a KL penalty, yielding the **regularized preference**:

$$
\mathcal{P}_\tau(\pi \succ \pi') = \mathcal{P}(\pi \succ \pi') - \tau\,\mathbf{KL}_\rho(\pi, \mu) + \tau\,\mathbf{KL}_\rho(\pi', \mu)
$$

where $\mathbf{KL}_\rho(\pi, \mu) = \mathbb{E}_{x\sim\rho}[\mathrm{KL}(\pi(\cdot\mid x)\,\|\,\mu(\cdot\mid x))]$ and $\tau>0$ controls the trust-region radius [source:arxiv:2312.00886].

## Nash Learning from Human Feedback (NLHF)

**NLHF** [source:arxiv:2312.00886] introduces the first end-to-end framework for learning directly from a preference model without an intermediate reward model. The tabular **Nash-MD** algorithm performs mirror descent on the regularized objective:

1. **Regularized policy**: $\pi_t^\mu(y) \propto \pi_t(y)^{1-\eta_t\tau}\,\mu(y)^{\eta_t\tau}$ (geometric mixture).
2. **Mirror descent update**: $\pi_{t+1} \propto \pi_t^\mu \exp\bigl(\eta_t\,\mathcal{P}(y \succ \pi_t^\mu)\bigr)$.

For deep policies, two practical variants are proposed:
- **Nash-MD-PG**: The opponent $\pi'$ is a geometric mixture $\pi'(y\mid x) \propto \pi_\theta(y\mid x)^{1-\beta}\,\mu(y\mid x)^\beta$ with $\beta\in[0,1]$.
- **Nash-EMA-PG**: The opponent is an exponential moving average of past policy parameters.

**Convergence**: In the tabular setting, with step size $\eta_t = 2/(\tau(t+2))$, Nash-MD achieves last-iterate convergence in KL divergence at rate $O(1/T)$: $\mathrm{KL}(\pi_\tau^*, \pi_T) \le 8/(\tau^2(T+1))$ [source:arxiv:2312.00886].

**Empirical results** (TL;DR summarization, PaLM 2 Large evaluator): Nash-MD-PG outperforms RLHF (PPO), Self-Play ($\beta=0$), and Best-Response ($\beta=1$). The optimal mixture weight lies in $\beta\in[0.125, 0.375]$, indicating that playing against a *blend* of current and reference policies is crucial [source:arxiv:2312.00886]. The authors caution that the comparison to RLHF is not “fair” because NLHF uses a preference model while RLHF uses a reward model, making direct architectural attribution difficult [source:arxiv:2312.00886].

## Self-Play Preference Optimization (SPPO)

**SPPO** [source:arxiv:2405.00675] frames alignment as iteratively computing the **von Neumann winner** of the two-player game. Each round $t$:
1. Sample $K$ responses $y_1,\dots,y_K \sim \pi_t(\cdot\mid x)$ for $x\sim\mathcal{X}$.
2. Compute empirical win-rates $\hat{P}(y_i \succ \pi_t \mid x)$ against the current policy population.
3. Construct dataset $D_t = \{(x, y_i, \hat{P}(y_i \succ \pi_t \mid x))\}$.
4. Update $\pi_{t+1}$ by minimizing a squared-error objective that matches the log-ratio to the advantage:

$$
\theta_{t+1} \leftarrow \arg\min_\theta \mathbb{E}_{(x,y,\hat{P})\sim D_t}\Bigl[ \log\frac{\pi_\theta(y\mid x)}{\pi_t(y\mid x)} - \eta\bigl(\hat{P}(y \succ \pi_t \mid x) - \tfrac12\bigr) \Bigr]^2
$$

The constant $\frac12$ approximates the log-partition factor $\log Z_{\pi_t}(x)$, reducing variance [source:arxiv:2405.00675].

**Results** (UltraFeedback prompts, PairRM 0.4B oracle):
| Base Model | Iter | AlpacaEval 2.0 LC WR | AlpacaEval 2.0 WR | MT-Bench | Arena-Hard | Open LLM Leaderboard |
|------------|------|----------------------|-------------------|----------|------------|----------------------|
| Mistral-7B-Instruct-v0.2 | 3 | 28.53% | 31.02% | 7.59 | 23.3 | 66.75 |
| Llama-3-8B-Instruct | 3 | 38.77% | 39.85% | — | — | 70.29 |

SPPO outperforms iterative DPO and IPO on all reported benchmarks while exhibiting a more moderate length increase [source:arxiv:2405.00675]. However, the authors observe an **alignment tax**: Open LLM Leaderboard scores peak at iteration 1–2 then decline, suggesting that preference optimization can erode general capabilities [source:arxiv:2405.00675]. Theoretical guarantees assume sufficient model expressivity and data coverage; the baseline approximation $\frac12 \approx \log Z_{\pi_t}(x)$ only reduces variance when the approximation is accurate [source:arxiv:2405.00675].

## Multiplayer Nash Preference Optimization (MNPO)

**MNPO** [source:arxiv:2509.23102] argues that two-player methods (NLHF, SPPO, INPO, ONPO, EGPO) suffer from **single-opponent bias**: optimizing against a single adversary fails to capture the full geometry of multi-annotator or multi-criteria preferences, leading to oscillatory dynamics and narrow exploration. MNPO generalizes to an $n$-player game where each player $i$ maximizes

$$
J\Bigl(\pi_i, \{\pi_j\}_{j\neq i}\Bigr) = \mathbb{E}_{x\sim d_0}\Bigl[ \mathbb{E}_{y^i\sim\pi_i, \{y^j\sim\pi_j\}} \bigl[ \mathbb{P}(y^i \succ \{y^j\} \mid x) \bigr] - \tau\,\mathrm{KL}(\pi_i \,\|\, \pi_{\mathrm{ref}}) \Bigr]
$$

Two settings are distinguished:

| Setting | Description | Algorithm | Convergence Guarantee |
|---------|-------------|-----------|----------------------|
| **Homogeneous** | All players share the same oracle $\mathbb{P}$ | **TD-MNPO**: opponents are a weighted mixture of historical policies $\{\pi_{t-j}\}$ | Provable convergence to Nash equilibrium via multiplicative weights |
| **Heterogeneous** | Each player $i$ has a distinct oracle $\mathbb{P}_i$ (e.g., safety vs. helpfulness RM) | **HT-MNPO**: each $\pi_i$ optimizes its own $J_i$ against the population | **No formal guarantee**; general-sum game lacks symmetry required for MWU proofs |

The **TD-MNPO loss** minimizes a distance $\mathbb{D}$ (e.g., squared error) between the current log-odds margin and a weighted average of historical log-odds margins:

$$
\mathcal{L}_{\text{D}}^{t,\mathbb{D}}(\pi\mid\beta,\{\lambda_j\},\eta) = \mathbb{E}_{y,y'\sim\pi,\, y_w,y_l\sim\lambda_{\mathbb{P}}} \mathbb{D}\Bigl[ \log\frac{\pi(y_w\mid x)}{\pi(y_l\mid x)} - \sum_{j=0}^{n-2}\lambda_j \log\frac{\pi_{t-j}(y_w\mid x)}{\pi_{t-j}(y_l\mid x)} \;\Big\|\; \eta\delta^\star \Bigr]
$$

where $\lambda_j$ are importance weights and $\delta^\star$ is a target reward gap [source:arxiv:2509.23102]. A **unified duality gap** $\text{DualGap}(\pi) = \max_{\pi'} J(\pi', O_\pi) - J(\pi, O_\pi)$ certifies equilibrium when zero [source:arxiv:2509.23102].

**Results** (Gemma-2-9B-it base, GPT-5-mini judge):
- **AlpacaEval 2.0 LC WR**: TD-MNPO 57.27% > INPO 56.09% > SimPO 55.16% > DPO 54.35%
- **Arena-Hard WR**: TD-MNPO 52.26% vs INPO 48.03%
- **MT-Bench**: TD-MNPO 7.03 vs INPO 6.95
- **AIME-24**: TD-MNPO 3.33 (only non-zero score)
- **HumanEval**: TD-MNPO 61.59 (best)
- **Ablation on $n$**: Increasing players from $n=1$ to $n=3$ lifts AlpacaEval LC WR from 53.32% to 57.27%; diminishing returns beyond $n=3$ [source:arxiv:2509.23102].

## Theoretical Guarantees and Convergence

| Method | Setting | Guarantee | Key Assumptions |
|--------|---------|-----------|-----------------|
| NLHF (Nash-MD) | Tabular, regularized | Last-iterate $O(1/T)$ in KL | Preference model $\mathcal{P}$ is fixed; step size $\eta_t = 2/(\tau(t+2))$ [source:arxiv:2312.00886] |
| SPPO | Function approximation | Approximates MWU; convergence to $\epsilon$-equilibrium if regression error bounded | Model class expressive enough; generated data covers input space; $\frac12 \approx \log Z_{\pi_t}(x)$ [source:arxiv:2405.00675] |
| TD-MNPO | Homogeneous $n$-player | Convergence to Nash via MWU in policy space | Symmetric zero-sum structure; shared oracle; sufficient exploration [source:arxiv:2509.23102] |
| HT-MNPO | Heterogeneous $n$-player | **None** (empirical only) | General-sum game breaks symmetry; no MWU convergence proof [source:arxiv:2509.23102] |

**Disagreement on theoretical scope**: NLHF provides a clean tabular rate but does not analyze function approximation error. SPPO’s regression-based update introduces approximation error that is not quantified in the convergence statement. MNPO proves convergence only for the homogeneous case; the heterogeneous extension—arguably the more practically relevant setting for multi-criteria alignment—lacks any equilibrium guarantee, though it performs well empirically [source:arxiv:2509.23102]. This gap is acknowledged by the MNPO authors, who state that HT-MNPO “still yields effective empirical solutions” despite the missing theory [source:arxiv:2509.23102].

## Empirical Comparisons and Trade-offs

| Aspect | NLHF | SPPO | MNPO (TD-MNPO) |
|--------|------|------|----------------|
| **Preference model** | Direct preference model $\mathbb{P}(y\succ y'\mid x)$ | Win-rate vs. current policy $\hat{P}(y\succ\pi_t\mid x)$ | Direct preference model + historical log-odds margins |
| **Opponent construction** | Geometric mixture ($\beta$) or EMA | Current policy population (self-play) | Weighted mixture of $n-1$ historical policies |
| **Update rule** | Policy gradient on $\mathcal{P}_\tau$ | Squared-error regression on log-ratio | Distance minimization on log-odds margin |
| **Length control** | Implicit via KL regularization | Moderate increase reported; no explicit control | Not explicitly reported; KL regularization present |
| **Alignment tax** | Not reported | Observed: Open LLM scores drop after iter 1–2 [source:arxiv:2405.00675] | Not reported on general benchmarks |
| **Compute budget** | PaLM 2 Large evaluator; TL;DR | UltraFeedback 60k prompts; PairRM 0.4B | Gemma-2-9B-it; GPT-5-mini judge |
| **Key strength** | Principled tabular convergence; handles intransitivity | Simple iterative recipe; strong AlpacaEval/Arena-Hard | Multi-player modeling; excels on reasoning (AIME) and coding |

**Critical disagreement on evaluation methodology**: NLHF explicitly states its comparison to RLHF is not fair due to different model classes (preference vs. reward) [source:arxiv:2312.00886]. SPPO compares against iterative DPO/IPO using the *same* preference oracle (PairRM), making its relative gains more attributable to the algorithmic framework [source:arxiv:2405.00675]. MNPO compares against INPO, SimPO, DPO using a *different* judge (GPT-5-mini) and base model (Gemma-2-9B-it), complicating cross-paper comparisons [source:arxiv:2509.23102]. No source evaluates all three methods on a common benchmark with a common oracle.

## Current status and trajectory

Game-theoretic preference optimization is **rising but not yet default**. The sequence NLHF → SPPO → MNPO shows rapid methodological evolution: from tabular mirror descent with convergence proofs, to practical self-play regression, to multi-player heterogeneous games. However, several factors limit widespread adoption:

- **Preference model dependency**: All three methods require a trained preference oracle $\mathbb{P}(y\succ y'\mid x)$, which is itself a non-trivial modeling step distinct from the more common reward-model pipeline [source:arxiv:2312.00886]. The field has not standardized on preference-model architectures or training recipes.
- **Compute and engineering complexity**: SPPO and MNPO require multiple iterative rounds of generation, annotation, and optimization, increasing infrastructure burden compared to single-stage DPO/SimPO [source:arxiv:2405.00675][source:arxiv:2509.23102].
- **Theoretical-practical gap**: MNPO’s heterogeneous setting (HT-MNPO) lacks convergence guarantees, and SPPO’s regression approximation introduces uncontrolled error. Practitioners may prefer methods with stronger guarantees (e.g., KL-regularized RLHF) or simpler objectives (DPO) unless the preference structure is known to be highly non-transitive.
- **Evaluation fragmentation**: Reported benchmarks differ (AlpacaEval 2.0, MT-Bench, Arena-Hard, AIME, HumanEval, Open LLM Leaderboard), and judges vary (PaLM 2 Large, GPT-4-Turbo, GPT-5-mini). No independent large-scale reproduction has been reported in the sources.

The trajectory suggests **increasing specialization**: MNPO’s gains on reasoning (AIME) and coding (HumanEval) hint that multi-player frameworks may become the method of choice for *capability-intensive* alignment where diverse criteria (correctness, style, safety) conflict. For general chat alignment, the added complexity may not justify marginal wins over well-tuned DPO/SimPO unless preference oracles improve significantly. The field has not converged on a standard “game-theoretic RLHF” stack.

## Key takeaways

- **Nash equilibrium replaces reward maximization** as the alignment objective, naturally handling intransitive and context-dependent preferences that break Bradley–Terry models [source:arxiv:2312.00886][source:arxiv:2405.00675].
- **Regularization toward a reference policy** (via KL penalty or geometric mixture) is essential for convergence and prevents degenerate deterministic policies [source:arxiv:2312.00886][source:arxiv:2405.00675][source:arxiv:2509.23102].
- **Opponent construction is the key design choice**: geometric mixture of current and reference (NLHF), current policy population (SPPO), or historical policy ensemble (MNPO). Each induces different exploration-exploitation trade-offs [source:arxiv:2312.00886][source:arxiv:2405.00675][source:arxiv:2509.23102].
- **Theoretical guarantees exist only for restricted settings**: tabular NLHF, homogeneous TD-MNPO. Function approximation and heterogeneous multi-criteria games remain empirically justified but theoretically open [source:arxiv:2312.00886][source:arxiv:2509.23102].
- **Empirical gains are real but context-dependent**: SPPO and TD-MNPO beat iterative DPO/IPO on AlpacaEval/Arena-Hard; TD-MNPO uniquely solves AIME-24 problems. Alignment tax (capability degradation) is observed in SPPO but not measured in others [source:arxiv:2405.00675][source:arxiv:2509.23102].
- **No unified benchmark or preference oracle standard** exists, making cross-method comparisons unreliable. Adoption is hindered by the need for a separate preference model and multi-round training infrastructure.

## Related topics

- [Direct Preference Optimization and variants](dpo-and-preference-optimization.md)
- [Self-improvement and self-play RL](self-improvement-and-self-play.md)
- [Reward modeling for LLMs](reward-modeling.md)
- [RLHF/PPO pipeline](rlhf-ppo-pipeline.md)
- [KL regularization in RLHF](kl-regularization.md)
- [Reward model over-optimization](reward-model-overoptimization.md)
- [Over-optimization and mode collapse](overoptimization-and-mode-collapse.md)
- [Alignment and win-rate evals](alignment-and-winrate-evals.md)
- [Judging bias and contamination](judging-bias-and-contamination.md)
- [RL for reasoning models](rl-for-reasoning.md)

## References
- [source:arxiv:2312.00886] [Nash Learning from Human Feedback](https://arxiv.org/abs/2312.00886)
- [source:arxiv:2405.00675] [Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/abs/2405.00675)
- [source:arxiv:2509.23102] [Multiplayer Nash Preference Optimization](https://arxiv.org/abs/2509.23102)
