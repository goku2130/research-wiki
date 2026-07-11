---
id: arxiv:2402.04792
type: paper
title: Direct Language Model Alignment from Online AI Feedback
url: https://arxiv.org/abs/2402.04792
retrieved: '2026-07-11'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

**Core Problem**
Direct alignment from preferences (DAP) methods, including DPO, IPO, and SLiC, offer efficient, reward-model-free alternatives to RLHF. However, they traditionally rely on static preference datasets collected ahead of training from a separate model $\rho$. This creates a purely offline feedback loop and off-policy learning: as the target policy $\pi_\theta$ evolves, the training distribution diverges from the generation distribution, causing significant distribution shift. Consequently, offline DAP methods rapidly overfit to stale, off-policy preferences, yielding suboptimal alignment compared to online RLHF methods.

**Method/Recipe (OAIF)**
Online AI Feedback (OAIF) resolves these issues by making DAP methods on-policy and interactive. The algorithm operates iteratively as follows:
1. Sample a prompt $x$ from a dataset.
2. Generate two candidate responses $y^1, y^2$ independently from the current policy $\pi_{\theta^t}(\cdot|x)$.
3. Query an external LLM annotator to rank the pair, designating one as the preferred response $y^+$ and the other as the less preferred $y^-$.
4. Update the policy parameters $\theta$ via gradient descent on a standard DAP loss $\ell(x, y^+, y^-, \theta)$.
Gradients are computed using $\nabla_\theta \ell$, with stop-graduations applied to both the sampling and annotation steps to ensure stable optimization. The LLM annotator's preference function is fully prompt-controllable, enabling dynamic alignment objectives (e.g., length constraints) without retraining a reward model.

**Key Formulas**
OAIF is architecture-agnostic and compatible with any differentiable DAP loss. The three primary loss functions employed are:
- **DPO loss:** $- \log \sigma \left(\beta \log \frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}\right)$
- **IPO loss:** $\left(\log \left(\frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x})}\right) - \frac {1}{2 \beta}\right) ^ {2}$
- **SLiC loss:** $\max \left(0, 1 - \beta \log \left(\frac {\pi_ {\theta} (\boldsymbol {y} ^ {+} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {-} | \boldsymbol {x})}{\pi_ {\theta} (\boldsymbol {y} ^ {-} | \boldsymbol {x}) \pi_ {\theta^ {0}} (\boldsymbol {y} ^ {+} | \boldsymbol {x})}\right)\right)$
where $\pi_{\theta^0}$ is the supervised finetuning baseline, $\sigma$ is the logistic function, and $\beta$ is a scalar hyperparameter.

**Key Quantitative Results**
Human evaluations confirm OAIF's effectiveness across TL;DR, Helpfulness, and Harmlessness tasks. Online DAP methods (DPO, IPO, SLiC) achieve an average win rate of $\sim 66\%$ against their offline counterparts. In four-way TL;DR comparisons, online DPO is preferred over SFT, RLHF, and RLAIF 58.00% of the time. Offline DPO exhibits rapid overfitting, with win rates collapsing after $\sim 3,500$ training steps, whereas online DPO performance consistently improves. Prompt controllability was validated via response length: instructing the annotator to prefer shorter outputs reduced average length from $\sim 120$ to $\sim 40$ tokens, with human-rated quality scores decreasing from 4.08 to 3.26 (still surpassing the SFT baseline of 3.19). Additionally, OAIF remains effective with smaller annotators; using PaLM 2-XS as the annotator yielded a quality score of 3.41, comparable to RLHF's 3.38.

**Stated Limitations**
The authors identify several constraints. The study isolates distribution shifts over responses $p(\boldsymbol{y}|\boldsymbol{x})$ but does not address shifts in the prompt distribution $p_X$ or ground-truth human value functions. Evaluations assume in-distribution prompts, leaving out-of-distribution generalization untested. All experiments utilize PaLM 2-XS, so scaling to larger models remains unverified. Replacing the LLM annotator with real-time human feedback faces a sample-efficiency bottleneck; aligning the model required $\sim 256,000$ samples across 2,000 steps, which is prohibitive for single-user personalization without low-rank adaptation techniques. Finally, while self-annotation is theoretically possible, the framework currently relies on external annotators to leverage size and capability advantages, as self-annotation requires identical model architecture and size.
