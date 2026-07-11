---
id: arxiv:2402.04792
type: paper
title: 'Judging LLM-as-a-Judge: Wins and Losses'
url: https://arxiv.org/abs/2402.04792
retrieved: '2026-07-11'
maturity: comprehensive
topic: rlaif
---

**Core Problem**
Direct alignment from preferences (DAP) methods, including DPO, IPO, and SLiC, fundamentally rely on static, pre-collected preference datasets. This architecture yields purely offline feedback, preventing the policy $\pi_\theta$ from receiving evaluation on its own generations during training. Consequently, a severe distribution shift emerges between the fixed data-generating policy $\rho$ and the evolving aligned policy $\pi_\theta$, rendering the learning process off-policy. This mismatch causes offline DAP methods to rapidly overfit and degrade in performance, whereas reinforcement learning from human feedback (RLHF) circumvents this via online, on-policy interactions but requires a separate reward model and computationally intensive policy gradient optimization.

**Methodology and Recipe**
The authors propose Online AI Feedback (OAIF), a framework that injects online, on-policy feedback into DAP methods by leveraging a large language model as an annotator. The training procedure follows a strict iterative recipe: (1) sample a prompt $x$ from a dataset; (2) sample two candidate responses $y^1, y^2$ from the current policy $\pi_{\theta^t}(\cdot|x)$; (3) prompt an LLM annotator to label the pair as preferred ($y^+$) and less preferred ($y^-$); and (4) update the policy parameters via $\theta^{t+1} \leftarrow \theta^t - \eta \nabla_\theta \ell(x, y^+, y^-, \theta^t)$. Gradients are computed with stop-gradient operations applied to both the sampling and annotation steps, treating the LLM feedback as a static target per iteration. The framework is loss-agnostic and compatible with any differentiable DAP objective. Feedback signals are highly controllable via instruction prompts without requiring model retraining.

**Key Formulas**
The DAP loss functions adapted within OAIF are defined as:
\[
\text{DPO: } - \log \sigma \left(\beta \log \frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}\right)
\]
\[
\text{IPO: } \left(\log \left(\frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x})}\right) - \frac {1}{2 \beta}\right) ^ {2}
\]
\[
\text{SLiC: } \max \left(0, 1 - \beta \log \left(\frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x})}\right)\right)
\]
where $\pi_{\theta^0}$ denotes the supervised finetuning baseline, $\sigma$ is the logistic function, and $\beta$ is a scalar hyperparameter.

**Quantitative Results**
Empirical evaluations across TL;DR summarization, Anthropic Helpfulness, and Harmlessness tasks demonstrate OAIF's efficacy. Human side-by-side evaluations reveal that online DAP variants achieve an average win rate of $\sim66\%$ over their offline counterparts. In four-way comparisons on TL;DR, human raters preferred online DPO over SFT, RLHF, and RLAIF $58.00\%$ of the time. OAIF also enables precise behavioral control: instructing the annotator to favor brevity reduced average response length from $\sim120$ to $\sim40$ tokens, with quality scores of $3.72$ and $3.26$ respectively, both surpassing the SFT baseline ($3.19$). Furthermore, OAIF remains effective even with small annotators; using PaLM 2-XS as the annotator for a PaLM 2-XS policy yielded a human quality score of $3.41$, comparable to RLHF ($3.38$).

**Stated Limitations**
The authors acknowledge several constraints. The study exclusively addresses response distribution shifts, neglecting shifts in the prompt distribution $p_X$ or ground-truth human value functions. Evaluations assume in-distribution prompts, leaving out-of-distribution robustness untested. All experiments scale only to PaLM 2-XS, so generalization to larger architectures remains unverified. Finally, single-user personalization is currently impractical due to sample inefficiency, requiring $\sim256,000$ samples for $\sim2,000$ training steps, highlighting a bottleneck for real-time human feedback integration.
