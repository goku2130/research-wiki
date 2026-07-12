---
id: arxiv:2607.08255
type: paper
title: 'Compete Then Collaborate: Frontier AI Teachers Build a Verifiable Curriculum
  to Improve a Coding Student Beyond Imitation'
url: https://arxiv.org/abs/2607.08255
retrieved: '2026-07-12'
maturity: comprehensive
topic: verifiable-rewards
---

# Summary: Compete Then Collaborate

### Core Problem
Traditional multi-teacher knowledge distillation typically merges the outputs of several frontier models into a student model without determining which teacher is most effective. Furthermore, these processes often rely on "LLM-as-a-judge" evaluation, which is prone to self-preference bias. The authors seek to determine if a "verifiable curriculum"—one based on objective code execution rather than imitation—can improve a student model that is already competent, as simple Supervised Fine-Tuning (SFT) often fails or degrades such models.

### Method
The authors propose a "compete-then-collaborate" framework using four frontier AI teachers (Claude, Codex-GPT, Grok, and Gemini) and a student model (Qwen2.5-Coder 7B and 32B).

#### Step 1: Competition (Teacher Ranking)
Teachers are ranked using an execution-based judge (unit tests and stdin-stdout checks) to eliminate LLM bias. To ensure fairness, the authors implement three controls:
1.  **Shared Task Bank:** All teachers solve identical problems from MBPP and `code_contests`.
2.  **Self-Correction:** Teachers are allowed up to two retries upon receiving failing test errors.
3.  **Intersection Control:** The student's training set is restricted to problems solved by *all* teachers to ensure only solution style, not task difficulty, varies.

#### Step 2: Collaboration (Student Training)
The union of verified solutions forms a collaborative curriculum used in two distinct modes:
1.  **Imitation (SFT):** The student is fine-tuned to imitate the pooled verified solutions.
2.  **Reinforcement Learning with Verifiable Rewards (RLVR):** The verified problems serve as a reward environment using Group Relative Policy Optimization (GRPO).

The reward function for RLVR is defined as:

$$
R = \text{fraction of tests passed} + \text{small format bonus}
$$

### Key Quantitative Results
#### Teacher Performance
On "easy" MBPP problems, all teachers reached near-saturation ($\approx 99\text{--}100\%$) after self-correction. However, harder competition problems (difficulty 6–9) separated the teachers:
*   **Gemini:** $77\%$ (115/150)
*   **Claude:** $69\%$ (104/150)
*   **Codex:** $69\%$ (103/150)
*   **Grok:** $50\%$ (75/150)

#### Student Performance
The study found that imitation (SFT) degrades competent students, while RLVR improves them.

**SFT Degradation (MBPP-test pass@1):**
*   **7B Student:** $76.7\% \text{ (base)} \rightarrow 72.7\% \text{ (union of teachers)}$
*   **32B Student:** $82.0\% \text{ (base)} \rightarrow 77.3\text{--}80.0\% \text{ (individual teachers)}$

**RLVR Improvement (Held-out competition problems):**
*   **Base:** $5.9\%$
*   **RLVR (200 steps):** $7.4\%$ ($+25\%$ relative)
*   **RLVR (1000 steps):** $8.8\%$ peak ($+49\%$ relative)

### Stated Limitations
*   **Benchmark Saturation:** The near-perfect scores on MBPP likely reflect training-set exposure (data leakage) rather than actual skill.
*   **Modest Absolute Gains:** RLVR gains on competition problems are modest in absolute terms; because the held-out set is small (68 problems), differences within the performance plateau ($\pm 1$ problem) may be sampling noise.
*   **Scope:** The study is limited to a single student model family (Qwen2.5-Coder) and a single programming language (Python).
*   **Hardware:** The authors encountered GPU GSP timeouts (Xid 119/154) on the NVIDIA GB10 platform, requiring reboots and reduced rollout intensity.
