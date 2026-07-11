---
id: arxiv:2310.10303
type: paper
title: 'ReST: Reinforced Self-Training for Language Models'
url: https://arxiv.org/abs/2310.10303
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

**Core Problem**
The classical Hotelling-Solomons inequality establishes that the absolute difference between the sample mean and median, normalized by the sample standard deviation, is bounded above by 1. While this provides a universal limit for skewness measurement, it ignores sample size and is not tight for finite datasets. The core problem addressed in this work is deriving a strictly sharper, sample-size-dependent upper bound for the ratio $|\operatorname{med} x - \bar{x}|/s$ that improves upon the constant 1 for all finite samples $n \geq 3$.

**Methodology**
The proof proceeds by standardizing the ordered data $x_1 \leq \dots \leq x_n$ into variables $z_i = (x_i - \bar{x})/s$, which inherently satisfy $\sum_{i=1}^n z_i = 0$ and $\sum_{i=1}^n z_i^2 = n$. The author partitions the index set into negative and positive segments using cutoffs $\ell$ and $m$, where $z_i < 0$ for $1 \leq i \leq \ell$ and $z_i > 0$ for $m+1 \leq i \leq n$. A partial sum magnitude is defined as $a = -\sum_{i=1}^\ell z_i = \sum_{i=m+1}^n z_i$. By applying the Cauchy-Schwarz inequality to these partial sums, the squared magnitude $a^2$ is bounded in terms of $\ell$ and $n-m$. A harmonic-mean inequality (Lemma 1) is then applied to convert the reciprocal sum $\frac{n}{\ell} + \frac{n}{n-m}$ into a quadratic form, yielding $a^2 \leq n^2 \frac{m}{n}(1 - \frac{m}{n})$ (or the symmetric bound for $\ell$). The analysis is rigorously split into two primary regimes: odd sample sizes $n = 2k+1$ and even sample sizes $n = 2k$. For each regime, the sign of the normalized median (or its average for even $n$) is examined across multiple subcases to bound the median's deviation. Equality conditions are identified by solving for the specific configurations of $z_i$ that saturate the derived inequalities.

**Key Formulas & Quantitative Results**
The central result is formalized in Theorem 1, which establishes that for any $n \geq 3$:
\[
\frac{|\operatorname{med} x - \bar{x}|}{s} \leq \begin{cases} \left( \frac{k}{k+1} \right)^{1/2} & \text{if } n = 2k + 1, \\ \left( \frac{k-1}{k+1} \right)^{1/2} & \text{if } n = 2k. \end{cases}
\]
For odd $n$, the bound $\sqrt{k/(k+1)}$ strictly improves upon 1. Equality is attained when the first $k$ observations equal $-\sqrt{(k+1)/k}$ and the remaining $k+1$ observations equal $\sqrt{k/(k+1)}$ (or vice versa for negative deviation). For even $n$, the bound $\sqrt{(k-1)/(k+1)}$ is derived through a four-case analysis of the normalized pair $(z_k, z_{k+1})$. Equality holds when the first $k-1$ observations equal $-\sqrt{(k+1)/(k-1)}$ and the last $k+1$ equal $\sqrt{(k-1)/(k+1)}$, or when the first $k+1$ equal $-\sqrt{(k-1)/(k+1)}$ and the last $k-1$ equal $\sqrt{(k+1)/(k-1)}$. The author notes that for $k \geq 2$, the even-sample bound satisfies $\sqrt{(k-1)/(k+1)} > 1/2$.

**Limitations**
The methodology explicitly addresses a documented gap in prior literature: while Hawkins and Boyd (1971) established properties for ordered statistics that confirm the odd-sample bound, their approach does not extend to even sample sizes ($n=2k$). The derived bounds are strictly finite-sample results dependent on the discrete ordering of observations; they characterize the worst-case configurations for the mean-median ratio but do not claim to bound population moments directly without additional distributional assumptions. Furthermore, the proof relies on the normalization $\sum z_i^2 = n$, meaning the results are scale-invariant but specific to the defined sample standard deviation formula. The paper does not provide extensions to continuous distributions or asymptotic limits beyond the stated finite-$n$ cases.
