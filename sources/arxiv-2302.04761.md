---
id: arxiv:2302.04761
type: paper
title: 'Toolformer: Language Models Can Teach Themselves to Use Tools'
url: https://arxiv.org/abs/2302.04761
retrieved: '2026-07-11'
maturity: comprehensive
topic: agentic-and-tool-use-rl
---

Large language models (LMs) demonstrate strong zero- and few-shot capabilities but struggle with foundational tasks like arithmetic, factual lookup, and temporal awareness. Existing tool-augmentation approaches typically require extensive human annotations or are confined to task-specific settings. Toolformer addresses this by enabling LMs to learn tool usage self-supervisedly, autonomously deciding when and how to invoke external APIs.

The methodology follows a structured pipeline. First, **sampling**: leveraging in-context learning, the LM annotates a plain-text corpus with potential API calls. Positions where the probability of the `<API>` token exceeds a threshold $\tau_s$ are selected, retaining the top $k$ positions, and up to $m$ calls are generated per position. Second, **execution**: all sampled calls are run to obtain text responses $r_i$. Third, **filtering**: calls are retained only if they reduce the model’s prediction loss for subsequent tokens. Fourth, **finetuning**: the filtered API calls are interleaved with the original text to form an augmented dataset $C^*$, on which the base LM is finetuned using a standard language modeling objective. During inference, decoding proceeds normally until the `<API>` token appears among the top-$k$ candidates; the process is paused, the API is queried, the response is inserted, and generation resumes.

The approach relies on specific mathematical formulations. An API call $c=(a_c, i_c)$ is linearized as $\mathrm{e}(c) = <\mathrm{API}> a_c(i_c) </\mathrm{API}>$ and with a result $r$ as $\mathrm{e}(c, r) = <\mathrm{API}> a_c(i_c) \rightarrow r </\mathrm{API}>$. Loss reduction is evaluated using a weighted cross-entropy loss:
\[
L_i(\mathbf{z}) = - \sum_{j=i}^n w_{j-i} \cdot \log p_M(x_j \mid \mathbf{z}, x_{1:j-1})
\]
where weights decay linearly: $\tilde{w}_t = \max(0, 1 - 0.2 \cdot t)$, normalized as $w_t = \tilde{w}_t / \sum_s \tilde{w}_s$. An API call is kept if $L_i^- - L_i^+ \geq \tau_f$, where $L_i^+ = L_i(\mathrm{e}(c_i, r_i))$ and $L_i^- = \min(L_i(\varepsilon), L_i(\mathrm{e}(c_i, \varepsilon)))$.

Quantitative evaluations on a 6.7B-parameter GPT-J model demonstrate substantial gains. On factual lookup (LAMA subsets), Toolformer achieves 33.8 (SQuAD), 11.5 (Google-RE), and 53.5 (T-REx), outperforming both GPT-J baselines and the 175B-parameter GPT-3 model, while invoking a question-answering API 98.1% of the time. For mathematical reasoning (ASDiv, SVAMP, MAWPS), scores reach 40.4, 29.4, and 44.0, respectively, surpassing GPT-3 (14.0, 10.0, 19.8) by leveraging a calculator tool 97.9% of the time. On open-domain QA (WebQS, NQ, TriviaQA), it scores 26.3, 17.7, and 48.8, relying on a Wikipedia search engine 99.3% of the time, though it trails GPT-3 on TriviaQA (65.9). For temporal tasks (TEMPLAMA, DATESET), it achieves 16.3 and 27.3, utilizing a calendar API 54.8% of the time on DATESET. Perplexity remains stable on WikiText and CCNet when API calls are disabled at inference, and tool-use capabilities emerge at approximately 775M parameters.

The authors explicitly outline several limitations. Toolformer cannot chain tools (using one tool’s output as another’s input) because API calls are sampled independently. It lacks interactive capabilities, such as browsing search results or refining queries based on feedback. The model is highly sensitive to input phrasing, suffers from sample inefficiency (e.g., millions of documents yield only thousands of useful calculator calls), and currently ignores the computational cost of API invocations.
