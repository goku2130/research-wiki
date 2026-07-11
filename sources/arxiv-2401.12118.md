---
id: arxiv:2401.12118
type: paper
title: Measures of the Capital Network of the U.S. Economy
url: https://arxiv.org/abs/2401.12118
retrieved: '2026-07-11'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

The study addresses the absence of comprehensive, topology-based metrics for capital concentration in the U.S. economy. Traditional concentration measures rely on top-firm market shares or financial interbank networks, neglecting the full web of ~2 million domestic business entities and ~15 million ownership links. The research aims to characterize the network’s global and local topology to establish a robust macroeconomic indicator of capital flow concentration.

The methodology proceeds in four steps. First, the author extracts ownership data from IRS tax returns (Forms 851, 1065, 1120, 1120S) for tax years 2009–2021, covering C corporations, S corporations, partnerships, and REITs. Second, a directed graph is constructed where nodes represent U.S. domestic business entities and directed edges denote parent-to-subsidiary ownership. Human owners, trusts, nonprofits, and foreign entities are excluded to isolate the business-entity-only subnetwork. Third, inbound and outbound link densities are fitted to power-law distributions using the Clauset et al. [2009] methodology, with bootstrapped 95% confidence intervals. Model fit is evaluated against lognormal alternatives via likelihood ratio tests. Fourth, network topology is quantified through connected components, diameter, average path length, clustering coefficients, assortativity, and subgraph motifs. Industry-specific subnetworks are derived by filtering edges associated with NAICS codes.

The power-law distribution of link counts $n$ is modeled as:

$$
\ln P(n) = -\gamma \ln(n) + K,
$$

where $P(n)$ is the probability of a node having $n$ links, $K$ is a scaling constant, and $\gamma$ is the concentration coefficient. The complementary cumulative distribution function (CCDF) on a log-log scale follows:

$$
\ln(CCDF) = (1 - \gamma) \ln n + K - \ln(\gamma - 1).
$$

For the 2021 entity-only network (2,230,248 nodes, 3,925,850 edges), the power-law fit is robust. The outbound link density yields $\gamma = 2.85 \pm 0.07$, while inbound yields $\gamma = 2.71 \pm 0.01$. Despite these exponents, the network deviates from small-world topology: the diameter ranges from 32 to 46 across the study period, with median path lengths of 9–10, significantly exceeding the small-world expectation of ~5.4 for $N \approx 2$ million. The giant connected component (GCC) contains 60.7% of nodes. The network is predominantly a directed acyclic graph and near-tree structure, with an average clustering coefficient of 5.0% in the GCC and near-zero assortativity ($\rho = 0.03$ in 2021). Firm size metrics (assets, wages) follow lognormal distributions and correlate weakly (~25%) with link counts. Industry analysis reveals consistent $\gamma$ values for specialized sectors, while finance and real estate exhibit lower $\gamma$ (higher concentration). Notably, health care showed a statistically significant drop in $\gamma$ in 2021, coinciding with increased M&A activity.

Limitations include the exclusion of inter-entity loans, public stock purchases below 20% ownership, and all foreign-to-foreign links. Ownership percentages across forms frequently fail to sum to 100% due to varying reporting standards, with 12.8% of entities reporting >100% and 7.4% <90%. Asset data is missing for firms below the $250,000 reporting threshold, and sole proprietorships are collapsed into single nodes. Finally, likelihood ratio tests find the power-law and lognormal fits statistically inconclusive, meaning alternative generative models cannot be ruled out.
