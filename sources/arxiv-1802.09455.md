---
id: arxiv:1802.09455
type: paper
title: The AI Safety Gridworlds
url: https://arxiv.org/abs/1802.09455
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

**Core Problem**
Introductory statistics courses (Stat 101) enroll students with highly variable mathematical and statistical preparation. While simulation-based inference (SBI) curricula have demonstrated improved conceptual understanding and retention, it remains unknown whether these benefits are equitably distributed across different preparation levels or if they disproportionately advantage specific student subgroups. Prior literature primarily predicts end-of-course grades using pre-course metrics, failing to track conceptual growth or compare curricula across preparation strata.

**Method and Analytical Recipe**
The study follows a sequential, multi-institutional assessment protocol:
1. **Curriculum Assignment:** Students are categorized into three instructional tracks: a traditional consensus curriculum (n=289), an early-SBI curriculum (n=366), and a revised SBI curriculum (n=1,078).
2. **Assessment Administration:** The Comprehensive Assessment of Outcomes in Statistics (CAOS) is administered during the first week (pre-test) and final week (post-test) of the course. A modified CAOS version is used for the larger SBI sample.
3. **Stratification:** Students are divided into approximately equal tertiles based on three pre-course preparation metrics: pre-test conceptual scores (≤40%, 40–50%/55%, ≥50%/55%), composite ACT scores (≤22, 23–26, ≥27), and self-reported college GPA (≤3.0, 3.0–3.7, ≥3.7).
4. **Gain Calculation:** Conceptual improvement is computed as the difference between post- and pre-test percentages.
5. **Statistical Testing:** Paired t-tests evaluate pre-to-post significance within each subgroup. Linear models adjusted for institutional differences compare change scores across curricula.
6. **Subscale Decomposition:** Overall gains are disaggregated into nine conceptual domains (e.g., probability, tests of significance, data collection) to identify curriculum-specific strengths.

**Key Formulas**
Conceptual gain is calculated as:

$$
Change = Post\text{-}test - Pre\text{-}test
$$

Curriculum comparison is modeled via:

$$
Change = Curriculum + Institution
$$

where the curriculum coefficient quantifies the adjusted difference in mean percentage-point improvement, with significance assessed via standard errors (SE) and p-values.

**Key Quantitative Results**
Across all preparation levels, SBI curricula produce statistically significant conceptual gains. Comparing consensus to early-SBI, early-SBI students show greater overall improvement (11.0% vs. 7.8%, difference = 3.2%, p<0.001). Stratified by pre-test score, early-SBI outperforms consensus in all tertiles, with a significant advantage in the middle group (+3.1%, p<0.05). Stratified by ACT score, early-SBI demonstrates significantly larger gains across all groups (Low: +8.2%, p<0.001; Middle: +4.7%, p<0.05; High: +6.0%, p<0.05). Subscale analysis for low-performing students reveals significant early-SBI advantages in data collection and design (+9.4%, p<0.05), tests of significance (+8.4%, p<0.05), and probability/simulation (+15.8%, p<0.01).

In the larger SBI sample, pre-test stratification shows the largest gains among the least prepared students (+15.2%, 95% CI [13.7, 16.7]), followed by middle (+8.1%) and high (+4.0%) performers (all p<0.001). Conversely, GPA stratification reveals the highest gains among A-range GPA students (+11.1%, 95% CI [9.5, 12.7]), followed by B+/A- (+8.1%) and B or lower (+7.3%) groups. Subscale analysis for low-performing SBI students confirms significant improvements across all seven measured conceptual domains, including probability/simulation (+18.2%), confidence intervals (+17.1%), and tests of significance (+16.6%).

**Stated Limitations**
The authors identify several constraints: (1) the revised SBI curriculum was not directly compared to non-SBI curricula, limiting direct comparative conclusions; (2) assessments were administered outside class for completion credit rather than performance incentives, which may influence results; (3) limited data on prior coursework (e.g., calculus, AP statistics) prevents full assessment of how specific mathematical backgrounds affect outcomes; (4) tertile stratification yielded only 33% of students in the lowest ACT group, diverging from national norms (63%), suggesting underrepresentation of academically underprepared students; (5) regression to the mean may confound pre-to-post comparisons; and (6) future research must incorporate student demographics, attitudes, and institutional/instructor variables to fully elucidate curriculum effectiveness.
