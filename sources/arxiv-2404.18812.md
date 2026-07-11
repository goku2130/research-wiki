---
id: arxiv:2404.18812
type: paper
title: Conditional Preference Optimization
url: https://arxiv.org/abs/2404.18812
retrieved: '2026-07-11'
maturity: comprehensive
topic: dpo-variants
---

The core problem addressed is the inefficient retrieval over learned sparse representations (LSR). While LSR models produce interpretable, context-aware embeddings that match or exceed dense models in out-of-domain settings, their statistical properties violate the assumptions underlying traditional inverted index algorithms like WAND and MaxScore. Specifically, LSR queries are longer and term frequencies lack the Zipfian distribution required for effective dynamic pruning, resulting in prohibitive per-query latency. Although approximate retrieval offers a pathway to scale LSR, existing graph-based and inverted-file approaches struggle with the distributional anomalies of sparse vectors, as evidenced by the 2023 BigANN Challenge.

To resolve this, the authors propose SEISMIC (Spilled Clustering of Inverted Lists with Summaries for Maximum Inner Product Search), an approximate nearest neighbor algorithm that reorganizes the inverted index into geometrically-cohesive blocks equipped with summary vectors. The indexing recipe proceeds as follows: (1) For each vocabulary coordinate $i$, construct an inverted list of document identifiers, sort them in descending order by their weight $x_i$, and apply static pruning by retaining only the top $\lambda$ entries. (2) Partition each pruned inverted list into $\beta$ blocks using a shallow K-Means variant that assigns documents to the cluster representative maximizing inner product. (3) Generate a summary vector for each block by taking the coordinate-wise maximum across all documents in the block, then prune this summary to its $\alpha$-mass subvector (retaining entries that cumulatively account for a fraction $\alpha$ of the $L_1$ mass). Finally, apply 8-bit scalar quantization to compress the summaries. Query processing leverages a forward index for exact vector retrieval. The algorithm selects the top `cut` coordinates of the query, traverses the inverted lists coordinate-at-a-time, and evaluates blocks by computing the inner product between the query and the block summary. If the summary’s inner product exceeds a threshold relative to the current $k$-th best score, the algorithm retrieves the full document vectors from the forward index, computes exact inner products, and updates a min-heap of top-$k$ candidates.

The mathematical foundation rests on three key formulations. Maximum inner product search (MIPS) is defined as:
$$\mathcal{S} = \underset{x \in \mathcal{X}}{\arg \max} \langle q, x \rangle.$$
The $\alpha$-mass subvector is determined by sorting entries $|x_{\pi_i}|$ and finding the smallest $j$ such that:
$$\sum_{i=1}^j |x_{\pi_i}| \le \alpha \|x\|_1.$$
The block summary is constructed as:
$$\phi (B) _ {i} = \max _ {x \in B} x _ {i}.$$
During retrieval, a block is skipped if:
$$r < \frac{\text{HEAP.min()}}{\text{heap\_factor}}.$$

Empirical evaluations on Ms MARCO and Natural Questions demonstrate SEISMIC’s efficiency. Operating in single-threaded mode, SEISMIC achieves sub-millisecond latency: 303 μs at 95% accuracy on SPLADE, 376 μs on Efficient SPLADE, and 180 μs on UNICOIL-T5. It consistently outperforms state-of-the-art baselines by one to two orders of magnitude, surpassing the BigANN Challenge winners (GRASSRMA and PYANN) by factors ranging from 3.4× to 21.6×. The SPLADE index occupies approximately 6,416 MiB and builds in 5 minutes, drastically faster than graph-based alternatives (137–267 minutes). Ablation studies confirm that geometric blocking and importance-based $\alpha$-mass summaries significantly outperform fixed-size chunking and fixed-length summaries, while 8-bit quantization reduces summary memory by 4× without degrading accuracy.

The authors acknowledge several limitations. SEISMIC is inherently approximate; while recall remains high, ranking quality metrics like MRR@10 degrade in the lowest-latency regimes. The conservative max-based summary grows rapidly in non-zero entries, necessitating pruning and quantization that may erode its theoretical upper-bound guarantees. Additionally, forward index lookups for documents within blocks induce cache misses, partially mitigated by hardware prefetching but remaining a latency factor. The method requires careful tuning of hyperparameters ($\lambda, \beta, \alpha, \text{cut}, \text{heap\_factor}$), and the empirical analysis focuses on a subset of LSR models and datasets due to computational constraints.
