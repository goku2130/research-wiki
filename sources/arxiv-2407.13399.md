---
id: arxiv:2407.13399
type: paper
title: 'Correcting the Mythos of KL-Regularization: Direct Alignment without Overoptimization
  via Chi-Squared Preference Optimization'
url: https://arxiv.org/abs/2407.13399
retrieved: '2026-07-11'
maturity: comprehensive
topic: kl-regularization
---

**Core Problem**
Offline language model alignment methods, including RLHF and Direct Preference Optimization (DPO), are fundamentally limited by *reward overoptimization*. As models optimize against an offline reward proxy, they overfit to inaccuracies and drift from the data manifold, causing true quality to degrade. KL-regularization is the standard mitigation but is provably too weak to induce pessimism or prevent distribution shift. The source investigates whether overoptimization is an inherent information-theoretic barrier or an algorithmic deficiency that can be corrected through stronger uncertainty quantification.

**Method/Recipe Step by Step**
The authors introduce $\chi^2$-Preference Optimization ($\chi_{\text{PO}}$), a direct alignment algorithm that modifies DPO with a single structural change to implicitly enforce $\chi^2$-divergence regularization. The procedure is:
1. **Define the link function:** Replace DPO’s logarithmic term with $\phi(z) = z + \log z$.
2. **Construct the clipped objective:** Substitute $\phi$ into the preference likelihood and apply clipping to bound unbounded density ratios:

$$
\hat{\pi} \leftarrow \arg\max_{\pi \in \Pi} \sum_{(x, a_+, a_-) \in \mathcal{D}_{\text{pref}}} \log \left[ \sigma \left( \text{clip}_{2R_{\max}} \left[ \beta \phi \left( \frac{\pi(a_+ \mid x)}{\pi_{\text{ref}}(a_+ \mid x)} \right) - \beta \phi \left( \frac{\pi(a_- \mid x)}{\pi_{\text{ref}}(a_- \mid x)} \right) \right] \right) \right].
$$

3. **Optimize and return:** Solve the objective over the policy class $\Pi$ and return $\hat{\pi}$. The $\chi^2$ term penalizes off-manifold behavior more aggressively than KL-divergence, implementing pessimism in the face of partial data coverage.

**Key Formulas & Theoretical Guarantees**
The framework assumes the Bradley-Terry preference model:

$$
\mathbb{P}(a \succ b \mid x) = \frac{\exp(r^{*}(x, a))}{\exp(r^{*}(x, a)) + \exp(r^{*}(x, b))}
$$

and defines the $\chi^2$-divergence as $D_{\chi^2}(\mathbb{P} \parallel \mathbb{Q}) := \frac{1}{2} \int \left( \frac{\mathrm{d}\mathbb{P}}{\mathrm{d}\mathbb{Q}} - 1 \right)^2 \mathrm{d}\mathbb{Q}$. Under policy realizability and bounded implicit reward assumptions, $\chi_{\text{PO}}$ achieves sample complexity scaling with *single-policy concentrability* $\mathcal{C}^{\pi^{*}} = \mathbb{E}_{\pi^{*}} \left[ \frac{\pi^{*}(a|x)}{\pi_{\text{ref}}(a|x)} \right]$, the gold standard in offline RL. The regret bound is:

$$
J(\pi^{*}) - J(\widehat{\pi}) \lesssim V_{\max} e^{2R_{\max}} \cdot \sqrt{\frac{\mathcal{C}^{\pi^{*}} \log(|\Pi|/\delta)}{n}} + \beta \cdot \mathcal{C}^{\pi^{*}} + \beta^{-1} \cdot \frac{V_{\max}^{2} e^{4R_{\max}} \log(|\Pi|/\delta)}{n}.
$$

By tuning $\beta \propto \sqrt{\frac{V_{\max}^2 e^{4R_{\max}} \log(|\Pi|/\delta)}{n \mathcal{C}^{\pi^*}}}$, the bound simplifies to $J(\pi^{*}) - J(\widehat{\pi}) \lesssim V_{\max} e^{2R_{\max}} \cdot \sqrt{\frac{\mathcal{C}^{\pi^{*}} \log(|\Pi|/\delta)}{n}}$. This proves robustness to overoptimization, as performance depends only on the coverage of the comparator policy rather than worst-case policies in
