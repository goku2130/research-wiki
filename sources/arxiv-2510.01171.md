---
id: arxiv:2510.01171
type: paper
title: How to Mitigate Mode Collapse and Unlock LLM Diversity
url: https://arxiv.org/html/2510.01171v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: overoptimization-and-mode-collapse
---

# Verbalized Sampling (VS) for Mitigating Mode Collapse

### Core Problem: Typicality Bias and Mode Collapse
Post-training alignment (e.g., RLHF) often leads to **mode collapse**, where Large Language Models (LLMs) favor a narrow set of stereotypical responses over a diverse range of plausible outputs. While previous research attributed this to algorithmic limitations, this work identifies a fundamental data-level driver: **typicality bias**. This is the cognitive tendency of human annotators to prefer text that is familiar, fluent, and predictable. 

The authors formalize this by modeling the reward function $r(x,y)$ as a combination of true task utility and typicality bias:

$$
r(x,y) = r_{\text{true}}(x,y) + \alpha \log \pi_{\text{ref}}(y|x) + \epsilon(x)
$$

where $\pi_{\text{ref}}$ is the pretrained base model and $\alpha > 0$ represents the typicality bias weight. When this reward is used in a KL-regularized RLHF objective, the resulting optimal policy $\pi^*$ is:

$$
\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)^\gamma \exp\left(\frac{r_{\text{true}}(x,y)}{\beta}\right), \quad \gamma := 1 + \frac{\alpha}{\beta}
$$

Because $\alpha > 0$, then $\gamma > 1$, which strictly sharpens the distribution of $\pi_{\text{ref}}$. In scenarios where multiple responses have similar true utility (e.g., creative writing), typicality bias acts as a tie-breaker, compressing probability mass toward the most typical completions and causing mode collapse.

### Method: Verbalized Sampling (VS)
To bypass mode collapse without retraining, the authors propose **Verbalized Sampling (VS)**, a training-free prompting strategy. The core insight is that different prompt types collapse to different modes. While instance-level prompts collapse to a single stereotypical mode, distribution-level prompts can approximate the broader distribution learned during pretraining.

**The VS Recipe:**
1. **Reformulate the Prompt:** Instead of a direct request for a single instance (e.g., *"Tell me a joke about coffee"*), the prompt is modified to request a distribution.
2. **Specify Quantity and Probability:** The model is explicitly asked to generate $k$ responses along with their corresponding probabilities (e.g., *"Generate 5 jokes about coffee and their corresponding probabilities"*).
3. **Optional Enhancements:** 
    * **VS-CoT:** Adding "Think step-by-step" before the distribution request.
    * **VS-Multi:** Using a multi-turn approach where the model is asked for $k$ responses with probabilities, followed by subsequent turns asking for "more" responses.
4. **Diversity Tuning:** Diversity can be tuned at inference time by adjusting the probability threshold in the prompt (e.g., *"Generate responses with probabilities below $\{threshold\}$"*).

### Key Quantitative Results
VS was evaluated across several tasks, showing significant improvements in the diversity-quality trade-off:

*   **Creative Writing:** VS increased semantic diversity by **1.6–2.1$\times$** over direct prompting and improved human evaluation scores by **25.7%**. In the Tulu-3 family, VS retained **66.8%** of the base model's original diversity after DPO, compared to only **23.8%** for direct prompting.
*   **Open-Ended QA:** For enumerative tasks (e.g., "Name a US state"), VS-elicited distributions closely aligned with pretraining corpora (KL divergence = **0.12** for Claude-4-Sonnet).
*   **Synthetic Data Generation:** Using VS to generate 1,000 synthetic math questions improved downstream accuracy on MATH500, OlympiadBench, and Minerva Math. VS-Multi achieved an average accuracy of **37.5%**, whereas direct prompting dropped performance to **30.6%**.
*   **Emergent Trend:** Larger models (e.g., GPT-4.1, Gemini-2.5-Pro) benefit more from VS, achieving diversity gains **1.5 to 2 times greater** than smaller models.

### Stated Limitations
*   **Computational Cost:** VS increases inference-time latency and token usage because it requires generating $k$ candidates and their probabilities rather than a single response.
*   **Model Capability Dependence:** The effectiveness of VS is positively correlated with model scale. Smaller models may lack the reasoning and instruction-following capabilities required to accurately estimate probabilities, which can occasionally degrade output quality.
