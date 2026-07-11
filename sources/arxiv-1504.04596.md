---
id: arxiv:1504.04596
type: paper
title: Structural Learning of Diverse Ranking
url: https://arxiv.org/abs/1504.04596
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Structural Learning of Diverse Ranking

### Core Problem
The primary challenge addressed is the simultaneous optimization of **relevance** and **diversity** in search result rankings. Traditional diversification approaches often rely on heuristic implicit methods or explicit subtopic information. The authors argue that implicit methods are often heuristic and lack clear relations to evaluation metrics, while explicit methods are prone to bias during subtopic estimation and are less adaptive to real-world applications where subtopics are unknown.

### Method
The authors propose a unified structural learning framework that directly optimizes Diversity-Correlated Evaluation Measures (DCEM), such as ERR-IA, $\alpha$-NDCG, and NRBP.

#### Step-by-Step Recipe:
1.  **Loss Function Definition**: The problem is formalized as a supervised learning task where the loss function $\Delta_{DCEM}$ is defined based on the ratio of the predicted ranking's DCEM value to the optimal ranking's value:

$$
\Delta_{DCEM}(\mathbf{y}^{(i)}, \mathbf{y}) = 1 - \frac{DCEM(\mathbf{y})}{DCEM(\mathbf{y}^{(i)})}
$$

2.  **Discriminant Function Formulation**: A bi-criteria objective is used to maximize the sum of relevance scores and dissimilarities among documents. The linear discriminant function is defined as:

$$
\mathbf{w}^T \Psi(\mathbf{x}, \mathbf{y}) = \sum_{r \in \mathcal{Y}} \mathbf{w}_r^T \psi_r(\mathbf{x}, \mathbf{y}) + \sum_{d \in \mathcal{Y} \times \mathcal{Y}} \mathbf{w}_d^T \psi_d(\mathbf{x}, \mathbf{y})
$$

    where $\psi_r$ represents relevance features (e.g., TF-IDF, BM25, PageRank) and $\psi_d$ represents diversity features.
3.  **Diversity Feature Engineering**: The framework incorporates several diversity features:
    *   **Semantic**: Topic model diversity (via pLSA), title, anchor text, and ODP-based taxonomy distance.
    *   **Non-Semantic**: Link-based (inlink/outlink) and URL-based (domain/site) similarity.
4.  **Training via Structural SVM**: The model is trained to minimize the following objective:

$$
\min_{w, \xi_i \geq 0} \frac{1}{2}\|w\|^2 + \frac{C}{n} \sum_{i=1}^n \xi_i
$$

    subject to: $\forall y \in \mathcal{Y} \setminus \mathbf{y}^{(i)}: \mathbf{w}^T \Psi(\mathbf{x}^{(i)}, \mathbf{y}^{(i)}) \geq \mathbf{w}^T \Psi(\mathbf{x}^{(i)}, \mathbf{y}) + \Delta_{DCEM}(\mathbf{y}^{(i)}, \mathbf{y}) - \xi_i$.
    A cutting plane algorithm is used to handle the exponential number of constraints.
5.  **Prediction**: The final ranking is generated using a greedy selection process that iteratively selects documents maximizing the marginal gain of the learned discriminant function.

### Key Quantitative Results
The approach was evaluated on TREC 2009, 2010, and 2011 Web Track datasets.

*   **Performance Gains**: $\text{SVM}_{DCEM}$ significantly outperformed baselines (QL, MMR, xQuAD, PM-2, ListMLE, and SVMDIV). Specifically, $\text{SVM}_{\alpha\text{-NDCG}}$ showed relative improvements in ERR-IA over $\text{xQuAD}_{list}$ of **17.16% (WT2009)**, **12.27% (WT2010)**, and **10.31% (WT2011)**.
*   **Comparison with SVMDIV**: $\text{SVM}_{DCEM}$ outperformed SVMDIV in ERR-IA by up to **11.54% (WT2009)**, **9.6% (WT2010)**, and **6.19% (WT2011)**.
*   **Robustness**: The total Win/Loss ratio for $\text{SVM}_{DCEM}$ was approximately **3.2**, compared to **2.8** for SVMDIV.
*   **Feature Importance**: The most influential diversity features were $\phi_{d_{odp}}$ (weight 2.82987) and $\phi_{d_{topic}}$ (weight 2.75189).
*   **Efficiency**: Average training times were comparable: ListMLE (1.5h), SVMDIV (2h), and $\text{SVM}_{DCEM}$ (2.5h). The prediction complexity is $O(nK)$, where $n$ is the candidate set size and $K$ is the ranking length.

### Limitations
*   **Approximation**: Finding the absolute optimal ranking is NP-hard; the authors rely on a greedy selection process, which provides a $(1 - 1/e)$-approximation for submodular functions.
*   **Feature Sparsity**: Link-based ($\phi_{d_{link}}$) and URL-based ($\phi_{d_{url}}$) features were found to be the least important, likely due to high sparsity in the datasets.
*   **Computational Cost**: The training time for $\text{SVM}_{DCEM}$ is slightly higher than other learning-based methods.
