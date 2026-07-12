---
id: arxiv:2402.00164
type: paper
title: 'Beyond DPO: A Survey of Preference-Based Alignment Algorithms for LLMs'
url: https://arxiv.org/abs/2402.00164
retrieved: '2026-07-12'
maturity: comprehensive
topic: dpo-and-preference-optimization
---

The authors extend the double/debiased machine learning framework to two-sample U-statistics, addressing bias from nuisance parameters estimated by flexible machine learning methods. They apply this to testing two-sample conditional distributions (covariate shift), showing how to construct asymptotically normal estimators and tests.

**Core Problem:**
Estimating functionals defined by two-sample U-statistics, $\theta = E_{P,Q}\varphi(X,U,\gamma)$, where $\gamma$ is an unknown nuisance parameter. When $\gamma$ is estimated using flexible machine learning methods, the plug-in estimator $\hat{\theta}_n = \frac{1}{mn}\sum_{i=1}^m\sum_{j=1}^n\varphi(X_i,U_j,\hat{\gamma}_{mn})$ is typically biased and does not achieve desired asymptotic properties, leading to invalid inference. This is particularly relevant for two-sample conditional distribution testing (covariate shift), where the test statistic's validity depends on estimating complex nuisance parameters like density ratios.

**Method/Recipe Step-by-Step:**

1.  **Define Neyman Orthogonality:** A U-statistic kernel $\varphi$ is Neyman orthogonal if its first-order derivative with respect to the nuisance parameter $\gamma$ is zero along any one-dimensional path in the functional space. This ensures that the plug-in estimate has only higher-order bias.
    *   **Definition 2.1:** The U-statistic kernel $\varphi$ is Neyman orthogonal if for any one-dimensional path $(F_\epsilon, G_\epsilon)$, we have $\frac{d}{d\epsilon}\mathbb{U}[\varphi(X,U,\gamma_\epsilon)]\big|_{\epsilon=0}=0$.

2.  **Construct U-Statistic Influence Function:** If $\varphi$ is not orthogonal, find a U-statistic influence function $\phi(X,U,\gamma,\alpha)$ (where $\alpha$ is an additional nuisance parameter) that satisfies:
    *   **Condition (2):** $\int\int\phi(x,u,\gamma_\epsilon,\alpha_\epsilon)d F_\epsilon(x)d G_\epsilon(u)=0$ for $\epsilon\in[0,\overline{\epsilon}]$. This implies $\mathbb{U}_{F_0,G_0}\phi(X,U,\gamma_0,\alpha_0)=0$.
    *   **Condition (3):** $\frac{d}{d\epsilon}\mathbb{U}[\varphi(X,U,\gamma_\epsilon)]\big|_{\epsilon=0}=\int\int\phi(x,u,\gamma_0,\alpha_0)[d F_0(x)d K(u)+d H(x)d G_0(u)]$.
    *   For the covariate shift test, the influence function is $\phi(X,Y,U,V,\gamma,\alpha) = \alpha(U) - \gamma(X)\alpha(X)$, where $\gamma(X) = \frac{g(X)}{f(X)}$ is the marginal density ratio and $\alpha(X) = E_{U,V\sim G,Y\sim f(y|x)}a(X,Y,U,V)$.

3.  **Form Debiased Kernel:** Create an orthogonalized kernel $\psi(X,U,\gamma,\alpha) := \varphi(X,U,\gamma)+\phi(X,U,\gamma,\alpha)$. This kernel is Neyman orthogonal (Theorem 1).

4.  **Estimate Nuisance Parameters and Apply Cross-Fitting:**
    *   Partition the data into $S \times T$ folds. For each fold $I_{st} := C_s \times D_t$, estimate $\hat{\gamma}_{-st}$ and $\hat{\alpha}_{-st}$ using data *outside* of $I_{st}$.
    *   Compute the fold-specific estimate: $\hat{\theta}_{st}=\frac{1}{M_sN_t}\sum_{i,j\in I_{st}}\psi(X_i,U_j,\hat{\gamma}_{-st},\hat{\alpha}_{-st})$.
    *   Aggregate estimates: $\hat{\theta}=\sum_{s\in[S]}\sum_{t\in[T]}\frac{M_sN_t}{mn}\hat{\theta}_{st}$.

5.  **Asymptotic Normality Conditions:** For $\hat{\theta}$ to be asymptotically normal, the following assumptions are needed:
    *   **Assumption 1:** As $m,n \to\infty$, $\lim_{m+n \to \infty} \frac{m}{m+n} = \lambda \in (0,1)$. (Balanced sample sizes)
    *   **Assumption 2:** For every fold $(s,t)$, $\|\hat{\psi}_{st}-\psi\|=o_p(1)$, where $\|\cdot\|$ is the $L^2$-norm. (Consistency of estimated orthogonalized influence functions)
    *   **Assumption 3:** The bias term $\sum_{s\in[S]}\sum_{t\in[T]}\frac{M_sN_t}{mn}C_{st}$ is $o_p(\frac{1}{\sqrt{m+n}})$. This requires the product of convergence rates of $\|\hat{\gamma}-\gamma\|$ and $\|\hat{\alpha}-\alpha\|$ to be faster than $o((m+n)^{-1/4})$.

6.  **Covariate Shift Test Implementation:**
    *   The test statistic for covariate shift is $\mathbb{U}\varphi(X,Y,U,V,\gamma) = \mathbb{U}[\gamma(X)a(X,Y,U,V)]$, where $a=\mathbb{I}(s(X,Y)<s(U,V))$.
    *   Under $H_0: F_{Y|X}=G_{V|U}$, $\theta=1/2$. Under $H_1$, $\theta<1/2$.
    *   The debiased kernel is $\psi(X,Y,U,V)=\gamma(X)a(X,Y,U,V)+\alpha(U)-\gamma(X)\alpha(X)$.
    *   Estimate nuisance parameters $\gamma$ and $\alpha$ (e.g., using classification for density ratios, regression for $\alpha$).
    *   Compute the standardized test statistic $\hat{T}=\frac{(.5-\hat{\theta})}{\sqrt{\hat{V}/(m+n)}}$.
    *   Reject $H_0$ if $\hat{T}>\Phi^{-1}(1-\alpha_0)$ for a significance level $\alpha_0$.

**Key Formulas (in LaTeX):**

*   **Functional of interest:** $\theta=E_{P}\varphi(Z,\gamma)$
*   **Plug-in estimate (biased):** $\hat{\theta}_{n}=\binom{n}{2}^{-1}\sum_{i<j}\varphi(X_{i},X_{j},\hat{\gamma}_{n})$ (for one-sample U-statistics)
*   **Two-sample U-statistic functional:** $\theta_{0}=\mathbb{U}[\varphi(X,U,\gamma_{0})]$
*   **Two-sample plug-in estimate (biased):** $\mathbb{U}_{m,n}\varphi(X,U,\hat{\gamma}_{m n}):=\frac{1}{m n}\sum_{i=1}^{m}\sum_{j=1}^{n}\varphi(X_{i},U_{j},\hat{\gamma}_{m n})$
*   **Neyman Orthogonality Condition:** $\frac{d}{d\epsilon}\mathbb{U}[\varphi(X,U,\gamma_{\epsilon})]\big|_{\epsilon=0}=0$
*   **Debiased U-statistic kernel:** $\psi(X,U,\gamma,\alpha):=\varphi(X,U,\gamma)+\phi(X,U,\gamma,\alpha)$
*   **Cross-fitted estimate for fold (s,t):** $\hat{\theta}_{s t}=\frac{1}{M_{s}N_{t}}\sum_{i,j\in I_{s t}}\psi(X_{i},U_{j},\hat{\gamma}_{-s t},\hat{\alpha}_{-s t})$
*   **Aggregated estimate:** $\hat{\theta}=\sum_{s\in[S]}\sum_{t\in[T]}\frac{M_{s}N_{t}}{m n}\hat{\theta}_{s t}$
*   **Asymptotic Normality (Theorem 3):** $\sqrt{m+n}(\hat{\theta}-\theta_{0}) \xrightarrow{D} N(0,V)$
*   **Asymptotic Variance (V):** $V=\frac{\mathrm{cov}(\psi(X,U),\psi(X,U^{\prime}))}{\lambda}+\frac{\mathrm{cov}(\psi(X,U),\psi(X^{\prime},U))}{1-\lambda}$
*   **Estimated Variance ($\hat{V}$):** $\hat{V}=\frac{m+n}{n}\hat{\sigma}_{1}^{2}+\frac{m+n}{m}\hat{\sigma}_{2}^{2}$
*   **Covariate Shift Test Statistic:** $\mathbb{U}\varphi(X,Y,U,V,\gamma)=\mathbb{U}[\gamma(X)a(X,Y,U,V)]$
*   **Covariate Shift Debiased Kernel:** $\psi(X,Y,U,V)=\gamma(X)a(X,Y,U,V)+\alpha(U)-\gamma(X)\alpha(X)$
*   **Density Ratio Estimation (classification method):** $\hat{\gamma}(X)=\frac{n}{m}\frac{1-\hat{\eta}(X)}{\hat{\eta}(X)}$, where $\hat{\eta}(X)=P(1|X=x)$

**Key Quantitative Results and Numbers:**

*   **Low-dimensional setting (Table 1):**
    *   Debiased estimator: bias 0.00 for all $n \in \{250, 500, 1000, 2000\}$. Type-1 error ranges from 0.034 ($n=250$) to 0.062 ($n=1000$), close to nominal 0.05. Power increases from 0.64 ($n=250$) to 1.00 ($n=2000$).
    *   Plug-in estimator: bias ranges from 0.11 ($n=250$) to 0.011 ($n=2000$). Type-1 error ranges from 0.020 ($n=250$) to 0.012 ($n=1000$), consistently below 0.05. Power is lower, from 0.32 ($n=250$) to 0.98 ($n=2000$).
*   **High-dimensional setting (Table 2):**
    *   Debiased estimator: bias ranges from -0.0068 ($n=250$) to -0.0015 ($n=2000$). Type-1 error ranges from 0.076 ($n=2000$) to 0.15 ($n=250$), slightly elevated but better than plug-in. Power increases from 0.61 ($n=250$) to 1.00 ($n=2000$).
    *   Plug-in estimator: bias ranges from -0.09 ($n=250$) to -0.11 ($n=500$). Type-1 error ranges from 0.25 ($n=2000$) to 0.61 ($n=250$), significantly inflated. Power increases from 0.73 ($n=250$) to 1.00 ($n=1000, 2000$).
*   **Airfoil Data (Random Splitting, Table 3):**
    *   Linear logistic regression: Mean estimate 0.500, Rejection Proportion 0.048.
    *   SuperLearner: Mean estimate 0.500, Rejection Proportion 0.048.
*   **Airfoil Data (Exponential Tilting, Table 4):**
    *   Linear logistic regression: Mean estimate 0.498, Rejection Proportion 0.056.
    *   SuperLearner: Mean estimate 0.496, Rejection Proportion 0.056.
*   **Airfoil Data (Velocity Partition, Table 5):**
    *   Linear: Mean 0.38, p-value 0 (rejects null).
    *   Linear $\hat{\gamma}$, SL $\hat{\alpha}$: Mean 0.40, p-value 0 (rejects null).
    *   SL: Mean 0.52, p-value 1 (does not reject null).
*   **Airfoil Data (Sound Partition, Table 6):** All methods correctly reject the null (p-values 0 or 0.0005).

**Stated Limitations:**

*   **Degenerate Test Statistic:** When the nuisance parameter $s$ (conformal score function) is identically 1 (e.g., under the null hypothesis), the U-statistic kernel can become degenerate, making the theory inapplicable. This is addressed by using a side part of the data to estimate $\hat{s}$ and treating it as fixed, and by introducing random tie-breaking in the definition of $a$.
*   **Rigorous Verification of Asymptotic Negligibility of Bias Term:** While orthogonality suggests higher-order bias, rigorously verifying that the bias term is asymptotically negligible for asymptotic normality is non-trivial and needs to be done case-by-case.
*   **Consistency of Variance Estimator:** Constructing confidence intervals requires a consistent estimator of the variance. While possible under standard consistency conditions for $\hat{\alpha}_{-st}$ and $\hat{\gamma}_{-st}$, the detailed argument is omitted, similar to prior work.
