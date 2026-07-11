---
id: bmva-archive:overcoming-mode-collapse-with-adaptive-m
type: web
title: Overcoming Mode Collapse with Adaptive Multi Adversarial Training
url: https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/0690.pdf
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Overcoming Mode Collapse with Adaptive Multi Adversarial Training (AMAT)

### Core Problem
Generative Adversarial Networks (GANs) frequently suffer from **mode collapse**, where the generator ignores certain modes of the target distribution. The authors hypothesize that mode collapse is driven by **catastrophic forgetting** in the discriminator: as the generator shifts to new modes, the discriminator loses its ability to maintain classification accuracy on previously seen samples. This creates an oscillatory training trajectory where the generator cycles between modes without converging, as the discriminator fails to provide a consistent training signal for all modes simultaneously.

### Method: Adaptive Multi Adversarial Training (AMAT)
To mitigate this, the authors propose AMAT, a framework that adaptively spawns additional discriminators to "remember" forgotten modes. The process is as follows:

1.  **Initialization**: Training begins as a standard GAN with one discriminator. A small set of **exemplar data** is sampled, containing a maximum of one real sample per mode.
2.  **Forgetting Detection**: The system monitors the discriminators' scores on the exemplar data. If a discriminator's score for a specific exemplar $e_i$ becomes unusually high compared to the average, it indicates the corresponding mode has been dropped or poorly generated.
3.  **Adaptive Spawning**: A new discriminator is spawned if the ratio of the maximum score to the average score over exemplars exceeds a threshold $\alpha_t$:

$$
\frac{\max(\mathfrak{s}[k])}{\text{avg}(\mathfrak{s}[k])} > \alpha_t
$$

    where $\mathfrak{s}[k]$ is the score of discriminator $k$ on exemplar $e_i$. The threshold $\alpha_t$ is a monotonically increasing function of the number of discriminators $|D|$, making it progressively harder to spawn new networks. A warmup period $T_t$ is applied after spawning before the forgetting check resumes.
4.  **Responsibility Assignment**: To prevent new discriminators from also forgetting, AMAT assigns responsibility for specific samples:
    *   **Fake Samples**: An $\epsilon$-greedy approach is used. With probability $1-\epsilon$, the discriminator with the lowest output score is assigned responsibility; with probability $\epsilon$, a random discriminator is chosen.
    *   **Real Samples**: A discriminator is chosen uniformly at random.
5.  **Generator Update**: The generator loss is calculated using a weighted mean of the scores from all discriminators in $D$. Weights are sampled from a $\text{Dirichlet}(K)$ distribution and sorted such that the discriminator with the lowest score (the one most specialized in that mode) receives the highest weight.

### Key Quantitative Results
The authors validated AMAT using a synthetic data generation procedure based on invertible normalizing flows and real-world datasets.

**Stacked MNIST**:
AMAT combined with a simple DCGAN achieved perfect mode coverage ($1000 \pm 0.0$ modes) and a KL divergence of $0.078 \pm 0.01$, outperforming Unrolled GAN and D2GAN.

**CIFAR10**:
AMAT consistently improved Inception Score (IS) and Fréchet Inception Distance (FID) across multiple architectures:
*   **BigGAN**: IS increased from $9.22$ to $9.51 \pm 0.06$; FID improved from $8.94$ to $6.11$.
*   **SN-GAN**: IS increased from $8.22 \pm 0.05$ to $8.34 \pm 0.04$; FID improved from $14.21$ to $13.8$.
*   **WGAN-GP**: IS increased from $7.59 \pm 0.10$ to $7.80 \pm 0.07$; FID improved from $19.2$ to $17.2$.

Per-class FID analysis on CIFAR10 showed the most significant gains in previously poor-performing classes, specifically "Frog" (56.1% improvement) and "Truck" (42.1% improvement).

### Limitations
The authors note several critical hyperparameter sensitivities:
*   **Discriminator Count**: Spawning too many discriminators (e.g., $>7$) leads to sub-optimal results and incomplete training.
*   **Timing**: A warmup schedule $T_t$ that is too long causes discriminators to be spawned too late in training, degrading performance.
*   **Assignment Strategy**: A purely greedy strategy for fake samples ($\epsilon=0$) is less effective than the $\epsilon$-greedy approach.
*   **Weighting**: Using a 1-hot weight vector for the generator update instead of soft random weighting results in lower IS and higher FID.
