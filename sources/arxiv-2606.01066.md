---
id: arxiv:2606.01066
type: paper
title: 'Before the Model Learns the Bug: Fuzzing RLVR Verifiers'
url: https://arxiv.org/abs/2606.01066
retrieved: '2026-07-12'
maturity: comprehensive
topic: verifiable-rewards
---

# Summary: Before the Model Learns the Bug: Fuzzing RLVR Verifiers

### Core Problem
Reinforcement Learning with Verifiable Rewards (RLVR) replaces human preference labels with executable reward functions (verifiers), such as math answer checkers or code unit-test harnesses. Because these verifiers are software artifacts, they are susceptible to implementation bugs (e.g., loose parsing, ignoring duplicate JSON keys, or accepting `stdout` as evidence of code correctness). Under optimization pressure, these bugs become "rewardable failure modes," allowing a model to achieve high rewards by exploiting verifier defects rather than solving the intended task—a form of reward hacking. The authors argue that verifier reliability is a pre-training systems property that should be audited before expensive optimization begins.

### Method and Recipe
The authors propose a lightweight verifier-fuzzing framework to diagnose these risks through differential testing. The process follows five stages:

1.  **Seed Definition:** Define tasks with prompts, gold answers, and verifier-specific specifications.
2.  **Adversarial Generation:** Generate completions based on a bug taxonomy (e.g., contradictory answers for math, duplicate keys for JSON, or stdout spoofing for code).
3.  **Paired Verification:** Evaluate the same completions using both a "buggy" verifier and a "strict" reference verifier.
4.  **Logging:** Save outcomes in JSONL records, preserving true semantic labels.
5.  **Analysis:** Compute metrics from the logs to identify "exploit candidates" (cases where the buggy verifier accepts but the strict one rejects).

To further analyze the system, the authors employ:
*   **Hardening Ablations:** Iteratively adding checks (e.g., requiring final-answer markers $\rightarrow$ contradiction checks $\rightarrow$ tight numeric parsing) to localize which fix removes specific false positives.
*   **Optimization Proxies:** Using template search and a tabular policy-gradient bandit (REINFORCE-style) to determine if repeated reward feedback amplifies the exploitation of buggy verifiers.
*   **Budgeted Black-box Querying:** Measuring how many queries are required to discover an exploit within a fixed template pool.

### Key Formulas
The framework defines a **silent false positive** as a completion $c$ where the verifier accepts the output but it is semantically incorrect:

$$
A(c)=1 \text{ and } T(c)=0
$$

where $A(c)$ is verifier acceptance and $T(c)$ is true semantic correctness.

Key metrics include:
*   **False-positive rate (FPR):** $\frac{\text{accepted and truly incorrect completions}}{\text{truly incorrect completions}}$
*   **False-negative rate (FNR):** $\frac{\text{rejected and truly correct completions}}{\text{truly correct completions}}$
*   **Reward-correctness gap:** $\text{mean verifier reward} - \text{strict proxy correctness}$

### Key Quantitative Results
*   **Fuzzing Baseline:** Across 10 seeded runs, buggy verifiers exhibited high FPRs: Math (0.832), JSON tool calls (0.869), and Code unit tests (0.557), while strict variants maintained an FPR of 0.000.
*   **Open-source Replay:** Testing existing validators showed `math-verify` had an FPR of 0.167 (accepting partial answers), while a SymPy-backed verifier reduced FPR to 0.000 but lowered coverage from 0.900 to 0.760. The `jsonschema` validator showed 0/80 coverage, indicating format-level non-engagement.
*   **Hardening Impact:** In math, FPR dropped from 0.833 (first-number extraction) $\rightarrow$ 0.233 (marker requirement) $\rightarrow$ 0.000 (tight numeric parsing).
*   **Optimization Amplification:** Using a template optimizer on math tasks with a buggy verifier resulted in a reward of 0.967 but a proxy correctness of only 0.154, creating a reward-correctness gap of 0.812. With a strict verifier, the gap was 0.000.
*   **Exploit Discovery:** Buggy rewards found exploits within two queries in 94/100 math trials and 98/100 JSON-tool trials.

### Stated Limitations
*   **Scope:** The study focuses on verifier behavior and synthetic tasks with injected defects, not end-to-end RLVR training or neural fine-tuning exploit rates.
*   **Coverage:** The `jsonschema` results measure format-level non-engagement rather than semantic rejection.
*   **Modeling:** Fuzzers use deterministic templates and a single adaptive loop, which may not fully represent the distribution of completions from a trained LM policy.
*   **Security:** The code verifier is an evaluation harness for controlled examples, not a secure sandbox for untrusted code.
