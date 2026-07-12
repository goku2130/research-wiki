---
id: arxiv:2403.07693
type: paper
title: 'ORPO: Monolithic Preference Optimization without Reference Model'
url: https://arxiv.org/abs/2403.07693
retrieved: '2026-07-12'
maturity: comprehensive
topic: dpo-and-preference-optimization
---

# LASS: Large, Small or Both for Debiasing Opinion Summarization

LASS is a data augmentation framework designed to mitigate sentiment bias in opinion summarization. The authors address the problem where summarization models are reluctant to generate negative summaries even when provided with negative input, a phenomenon caused by extreme sentiment imbalance in benchmark datasets (e.g., 72.26% positive reviews in Yelp and 83.5% in Amazon). While Large Language Models (LLMs) can generate synthetic data to balance these distributions, doing so is computationally expensive and risks introducing toxicity. LASS combines a small amount of LLM-generated "seed" data with a small-scale disentanglement autoencoder to produce large volumes of high-quality synthetic negative reviews economically.

### Methodology

The LASS framework consists of three primary stages:

**1. Pair Data Creation via LLM**
The framework uses an LLM to rewrite original positive reviews into negative counterfactuals following a "minimal-edit" principle. To optimize the output, the authors use an iterative prompt optimization process. A foundational prompt $P = \{D, s(x_1, y_1), \dots, s(x_k, y_k)\}$ (containing a task instruction $D$ and $k$ examples) is refined by adding human-annotated samples and optimizing their order until a success rate threshold $\delta$ is reached or improvement falls below $\varepsilon$.

**2. Dis-AE Model Training**
A disentanglement reconstruction autoencoder (Dis-AE) is trained on the LLM-generated pairs. The architecture includes a BiLSTM encoder $p_\theta$ that produces sentiment representations $z_e$ and content representations $z_n$, an emotional classifier $C$, and an LSTM decoder $q_\phi$. The model is trained using a composite loss function:

$$
\mathcal{L} = L_{rec} + \alpha \mathcal{L}_{emo} + \beta \mathcal{L}_{dis} + \gamma \mathcal{L}_{cf}
$$

The components are defined as follows:
*   **Reconstruction Loss ($L_{rec}$):** Ensures the model can reconstruct the original input $x^p$ and $x^n$ from their respective representations.

$$
L_{rec}(\theta, \phi) = - \sum_{i=1}^N \mathbb{E}_{p_\theta(\tilde{z}_e^p, z_c^p | x^p)} [\log q_\phi(x^p | \tilde{z}_e^p, z_c^p)] - \sum_{i=1}^N \mathbb{E}_{p_\theta(\tilde{z}_e^n, z_c^n | x^n)} [\log q_\phi(x^n | \tilde{z}_e^n, z_c^n)]
$$

*   **Emotional Auxiliary Constraint ($\mathcal{L}_{emo} = L_e + L_n + L_r$):** $L_e$ is the cross-entropy loss for sentiment classification of $z_e$; $L_n$ uses KL divergence to force content representations $z_c$ toward a uniform distribution $\mathcal{U}(0, M)$ to ensure sentiment neutrality; $L_r$ constrains a learnable label representation set $Z_r$.
*   **Distance Loss ($L_{dis}$):** Constrains the cosine similarity ($\text{sim}$) such that sentiment representations are distant while content representations are similar:

$$
L_{dis} = 2 + \text{sim}(z_e^p, z_e^n) - \text{sim}(z_c^p, z_c^n)
$$

*   **Counterfactual Reconstruction Loss ($L_{cf}$):** Forces the model to reconstruct $x^p$ using the sentiment of $x^p$ and the content of $x^n$ (and vice versa).

**3. Data Reproduction**
To generate large-scale negative reviews, the framework combines the content representation of a positive review with the sentiment representation of a negative review. The resulting "child" representation is decoded into text. To ensure quality, generated samples are filtered based on a perplexity (PPL) threshold of 125 and a sentiment classifier.

### Key Quantitative Results

LASS was evaluated on Amazon and Yelp datasets across models including Coop and Copycat.
*   **Sentiment Accuracy:** LASS increased negative sentiment accuracy by an average of 36% across three models, nearly matching the 37% increase achieved by using LLMs exclusively.
*   **Efficiency:** LASS achieved these results while reducing the required amount of LLM-generated synthetic data by an average of 265,000 points.
*   **Summarization Quality:** ROUGE scores remained stable or improved slightly, indicating that the augmentation did not degrade the primary summarization task.
*   **Dis-AE Scaling:** Analysis showed that the quality of generated text (measured by PPL and counterfactual ROUGE) improved steadily as training data increased, stabilizing at 200k samples.

### Limitations

The authors note that the effectiveness of LASS is dependent on the base summarization model; for instance, the Copycat model showed a significant decline in positive sentiment accuracy on the Amazon dataset after augmentation. Additionally, some models like TRACE are highly sensitive to changes in data distribution due to their counter-template parameters. Finally, the current method for determining the minimum training data for Dis-AE relies on indirect metrics (PPL), which may not perfectly reflect text quality.
