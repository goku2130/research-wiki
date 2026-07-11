---
id: arxiv:2604.25419
type: paper
title: 'JURY-RL: Votes Propose, Proofs Dispose for Label-Free RLVR - arXiv'
url: https://arxiv.org/html/2604.25419v1
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

The paper "JURY-RL: Votes Propose, Proofs Dispose for Label-Free RLVR" introduces a novel reinforcement learning with verifiable rewards (RLVR) framework designed to enhance large language models' (LLMs) reasoning capabilities without relying on human-annotated labels.

## Core Problem

Standard RLVR methods often require costly human-annotated answers or meticulously curated reward specifications, limiting their scalability and coverage. Label-free alternatives, such as majority voting or LLM-as-a-judge, aim to reduce annotation costs but are prone to false positives, reward hacking, and training instability. The core problem is to develop a label-free RLVR framework that is **scalable** (without costly human supervision), **truth-aligned** (rewards verifiable correctness), and **optimization-stable** (allows learning even when verification is inconclusive).

## Method: JURY-RL Framework

JURY-RL decouples the answer proposal from reward disposal, using "Votes Propose, Proofs Dispose."

**Step-by-step Process:**

1.  **Proposal Stage (Votes Propose):**
    *   For a given problem \(x\), the policy LLM \(\pi_{\theta}\) generates \(G\) trajectories (rollouts) \(\{y_i\}_{i=1}^G\).
    *   A deterministic extractor `ans(.)` parses a candidate answer \(a_i = \text{ans}(y_i)\) from each trajectory.
    *   A single consensus candidate \(\hat{a}\) is proposed via plurality (majority) voting:
        \[
        \hat{a} = \arg \max_a \sum_{i=1}^{G} \mathbb{I}[a_i = a].
        \]

2.  **Disposal Stage (Proofs Dispose):**
    *   A single call to an external formal verifier (e.g., Lean) evaluates \(\hat{a}\) against the formal specification of \(x\). This yields a binary proof-gate \(\delta = \text{verify}(x, \hat{a}) \in \{0, 1\}\).
    *   **If verification succeeds (\(\delta = 1\)):** A positive reward is granted exclusively to the trajectories that produced the proven-correct answer. The reward for each trajectory \(y_i\) is \(r_i = \mathbb{I}[a_i = \hat{a}]\).
    *   **If verification is inconclusive (\(\delta = 0\)):** The system defaults to the **ResZero (Residual-Zero) fallback reward**. This discards the unverified plurality proposal and assigns a carefully constructed, zero-mean, variance-preserving reward to the remaining (residual) answers.

3.  **ResZero Reward Mechanism (for \(\delta = 0\)):**
    *   Let \(M = \{i : a_i = \hat{a}\}\) be the set of rollouts supporting the majority answer and \(R = \{i : a_i \neq \hat{a}\}\) be the set of residual rollouts.
    *   The majority share is \(\alpha = |M| / G\).
    *   For an answer \(b\) within the residual group, the leave-one-out residual share is defined as:
        \[
        u^{(-i)}(b) = \frac{1}{|R| - 1}\sum_{\substack{j\in R\\ j\neq i}}\mathbb{I}[a_{j} = b].
        \]
    *   Let \(z_{i} = u^{(-i)}(a_{i})\) if \(i \in R\) and \(z_{i} = 0\) if \(i \in M\).
    *   The ResZero reward \(r_i^{\text{ResZero}}\) is assigned as:
        \[
        r _ {i} ^ {\text { ResZero }} = \underbrace {\alpha \cdot \mathbb {I} [ i \in R ] \cdot (z _ {i} - \bar {u})} _ {\text { Amplify   residual }} - \underbrace {c \alpha \cdot \mathbb {I} [ i \in M ]} _ {\text { Penalize   majority }} + \underbrace {\gamma} _ {\text { Re - center }},
        \]
        where \(\bar {u} = \frac {1}{| R |} \sum_ {j \in R} z _ {j}\) and \(\gamma = c \alpha^ {2}\).
        *   \(c\) is a positive hyperparameter controlling penalty strength (set to 0.01 in experiments).
        *   \(\gamma\) is a global re-centering term ensuring the total reward sums to zero.
        *   The term \((z_i - \bar{u})\) creates a zero-mean signal within the residual group, rewarding more popular minority answers.

4.  **Policy Update:**
    *   Group-normalized advantages \(\hat{A}_i\) are computed from the grouped rewards \(\{r_i\}_{i=1}^G\) using:
        \[
        \hat {A} _ {i} = \frac {r _ {i} - \bar {r}}{\operatorname{std} \left(\left\{r _ {j} \right\} _ {j = 1} ^ {G}\right) + \varepsilon}, \quad \bar {r} = \frac {1}{G} \sum_ {j = 1} ^ {G} r _ {j}.
        \]
    *   The policy \(\pi_{\theta}\) is updated using Group Relative Policy Optimization (GRPO) to maximize:
        \[
        \begin{array}{l} J _ {\mathrm{GRPO}} (\theta) = \mathbb {E} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \min \left(\rho_ {i, t} \hat {A} _ {i, t}, \right. \right. \\ \left. \operatorname{clip} \left(\rho_ {i, t}, 1 \pm \epsilon\right) \hat {A} _ {i, t}\right) - \beta D _ {\mathrm{KL}} \left(\pi_ {\theta} \| \pi_ {\text {ref}}\right) \Bigg ], \\ \end{array}
        \]
        where \(\rho_{i,t}(\theta) = \frac{\pi_{\theta}(y_{i,t}|x,y_{i,<t})}{\pi_{\mathrm{old}}(y_{i,t}|x,y_{i,<t})}\) is the per-token ratio, \(\hat{A}_{i,t}\) is a broadcast of \(\hat{A}_i\), and \(\beta\) is the KL penalty coefficient.

## Key Quantitative Results and Numbers

*   **Overall Performance:** JURY-RL consistently outperforms all label-free RL baselines and matches supervised GRPO with ground-truth rewards (GT) across three backbone models (Qwen3-1.7B-Base, Llama-3.2-3B-Instruct, Qwen2.5-7B) on mathematical reasoning, code generation, and general benchmarks.
*   **Mathematical Reasoning (Average pass@k):**
    *   Qwen3-1.7B-Base: JURY-RL (59.41%) vs. GT-Reward (55.36%), LLM-as-a-judge (50.35%). JURY-RL shows +4.05 pp over GT and +9.06 pp over LLM-as-a-judge.
    *   Llama-3.2-3B-Instruct: JURY-RL (48.48%) vs. GT-Reward (45.46%), LLM-as-a-judge (45.93%). JURY-RL shows +3.02 pp over GT and +2.55 pp over LLM-as-a-judge.
    *   Qwen2.5-7B: JURY-RL (64.04%) vs. GT-Reward (62.48%), LLM-as-a-judge (53.99%). JURY-RL shows +1.56 pp over GT and +10.05 pp over LLM-as-a-judge.
*   **Mathematical Reasoning (Average pass@1):** JURY-RL also improves average pass@1 over both GT-Reward and LLM-as-a-judge across all models, with gains of +2.32 pp, +1.91 pp, and +1.53 pp over GT for the respective models.
*   **Out-of-Domain Generalization (Qwen2.5-7B):** JURY-RL surpasses the GT baseline on out-of-domain benchmarks (code generation, instruction following, multi-task knowledge), achieving an average of 40.14% (+1.88 pts, +4.92% relative gain).
*   **ResZero Ablation:** ResZero consistently achieves the highest average performance among proof-gated variants when verification is inconclusive (\(\delta=0\)). It outperforms "Zero Reward" by +1.3 pts, "MV Reward" by +6.1 pts, and "Random Reward" by +5.4 pts on average. "Proof-Gate + MV Reward" performs worse than plain "Majority-Voting" baseline, highlighting the risk of reinforcing unverified consensus.
*   **Training Stability:** JURY-RL demonstrates stable, monotonic improvement in validation accuracy on MATH5000, unlike baselines like Entropy and Self-Certainty which show collapse, or LLM-as-a-Judge/LLM-KD/Majority-Voting which exhibit noisy convergence.
*   **Verifier Signal Quality (Lean vs. LLM-as-a-Judge):**
    *   Lean Verifier (Ours): Precision 84.5%, Recall 88.0%, F1 86.2%.
    *   LLM-as-a-Judge: Precision 75.9%, Recall 96.1%, F1 84.8%.
    *   The Lean verifier provides a superior reward signal with substantially higher precision, reducing false positives.
*   **Autoformalizer Reliability:** Successfully generated syntactically valid and type-checkable Lean 4 statements for 477 out of 500 MATH500 problems, yielding a 95.4% success rate. On held-out benchmarks, success rates varied: miniF2F (97.1%), ProofNet (71.5%), Putnam (74.2%), AIME 2025 (66.7%), averaging 77.4%.
*   **Consistency Checker Accuracy:** On MATH500, achieved F1-score of 91.1% and Recall of 94.9%. On held-out benchmarks, F1-scores were 86.6% (ReformBench-859) and 84.9% (CriticleanBench).
*   **Computational Cost (Qwen-2.5-7B, cold-start):**
    *   GT-Reward (baseline training): ~100 seconds per step.
    *   JURY-RL (Lean Verifier): Adds ~200 seconds per step (total ~300s).
    *   LLM-as-a-Judge: Adds ~80 seconds per step (total ~180s).
    *   JURY-RL's cost is amortized by caching verification results and early-stopping mechanisms for successful proofs.

## Stated Limitations

*   **Verifier Imperfections:** While the Lean verifier's core logic has a false positive rate of 0%, the overall pipeline involves upstream processes like auto-formalization and consistency checks, which can introduce errors. These imperfections primarily lead to inconclusive verification (\(\delta=0\)) rather than false-positive rewards, reducing supervision density on harder domains without compromising reward precision.
*   **Computational Overhead:** The Lean verifier introduces a significant computational overhead compared to LLM-as-a-judge, especially during cold-start. However, this is mitigated by caching and early-stopping mechanisms as training progresses.
*   **Generalization of Formalization:** Autoformalization success rates vary by domain, being lower on harder competition problems (e.g., AIME 2025 at 66.7%) compared to textbook problems (e.g., miniF2F at 97.1%).
*   **Potential Misuse:** Stronger automated reasoning capabilities could be misused for completing graded assignments or generating convincing but incorrect solutions.
*   **Benchmark Bias/Coverage:** Public benchmarks might contain stylistic biases or limited topical coverage.
