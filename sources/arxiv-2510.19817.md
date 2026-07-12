---
id: arxiv:2510.19817
type: paper
title: 'olmOCR 2: Unit Test Rewards for Document OCR'
url: https://arxiv.org/abs/2510.19817
retrieved: '2026-07-12'
maturity: comprehensive
topic: rl-for-math-and-code
---

# olmOCR 2: Unit Test Rewards for Document OCR

olmOCR 2 is a specialized vision language model (VLM) designed to convert digitized print documents, such as PDFs, into clean, naturally ordered plain text. The system is powered by **olmOCR-2-7B-1025**, a 7B parameter model based on Qwen 2.5 VL.

### Core Problem
The primary challenge addressed is the accurate extraction and linearization of complex document elements, specifically math formulas, tables, and multi-column layouts. The authors argue that traditional continuous scoring metrics, such as edit distance, are insufficient for OCR evaluation because they do not correlate with practical correctness. For example, edit distance may heavily penalize a caption that is placed correctly relative to the text but in a different order than the ground truth, or it may penalize a LaTeX formula that renders correctly but differs in string representation.

### Method and Recipe
The authors employ Reinforcement Learning with Verifiable Rewards (RLVR) using binary unit tests as reward signals. The training recipe consists of three main phases:

**1. Synthetic Document Pipeline**
To scale the creation of verifiable rewards, the authors developed a pipeline to generate synthetic HTML pages from real PDFs:
*   **PDF Sourcing:** Real-world documents (e.g., math-heavy arXiv papers) are sampled to ensure layout diversity.
*   **PDF to HTML Conversion:** A general VLM (Claude-Sonnet-4-20250514) converts pages to HTML via a three-step process:
    1.  **Layout Analysis:** Identifying columns, images, tables, and headers/footers.
    2.  **Content Rendering:** Generating clean, semantic HTML.
    3.  **Output Refinement:** Rendering the HTML to an image and prompting the VLM to refine the code to better match the original document.
*   **Unit Test Creation:** Programmatic binary tests are extracted from the HTML semantics:
    *   **Text Presence/Absence:** Checking for specific phrases or the absence of headers/footers.
    *   **Natural Reading Order:** Verifying the sequence of sentences.
    *   **Table Accuracy:** Checking the relative positions of specific cell values.
    *   **Math Formula Accuracy:** Verifying that the output renders the same way as the ground truth using KaTeX.
    *   **Baseline Robustness:** Ensuring the absence of repeated n-grams or non-target characters.

**2. Supervised Fine-Tuning (SFT)**
The model undergoes a single epoch of SFT using the `olmOCR-mix-1025` dataset, which contains 267,962 pages from over 100,000 PDFs.

**3. RLVR Training**
The model is further trained using Group Relative Policy Optimization (GRPO).
*   **Reward Signal:** The reward is the pass rate of the binary unit tests at the page level (e.g., 4 passes out of 6 tests yields a reward of 0.67).
*   **Hyperparameters:** The training uses a KL divergence $\beta = 0.01$.
*   **Model Souping:** To maximize performance, six models are trained with different random seeds and their weights are averaged (souping).

### Key Quantitative Results
olmOCR 2 demonstrates state-of-the-art performance on the **olmOCR-Bench** benchmark:
*   **Overall Score:** olmOCR 2 achieved a score of $82.4 \pm 1.1$, representing a $+14.2$ point improvement over the original olmOCR release ($68.2 \pm 1.1$).
*   **Comparative Performance:** It outperforms several major systems, including GPT-4o ($68.9 \pm 1.1$), Gemini Flash 2 ($57.8 \pm 1.1$), and DeepSeek-OCR ($75.7 \pm 1.0$), while remaining slightly behind Chandra OCR 0.1.0 ($83.1 \pm 0.9$).
*   **Data Scale:** The synthetic pipeline produced 30,381 test cases from 2,186 PDF pages.
*   **Efficiency:** The synthetic generation process cost approximately $\$0.12$ per document page.

### Limitations
The authors note that the synthetic data pipeline needs further development to cover more complex document types and a wider array of unit tests. Additionally, they state that more work is required to develop calibrated continuous scores for OCR targets beyond math formulas to complement the binary unit test framework.
