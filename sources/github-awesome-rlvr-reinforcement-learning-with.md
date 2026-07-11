---
id: github:awesome-rlvr-reinforcement-learning-with
type: web
title: Awesome RLVR — Reinforcement Learning with Verifiable Rewards
url: https://github.com/opendilab/awesome-RLVR
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-math-and-code
---

This source, "Awesome RLVR — Reinforcement Learning with Verifiable Rewards," is a curated collection of resources on Reinforcement Learning with Verifiable Rewards (RLVR).

**Core Problem:**
The core problem addressed by RLVR is the need to align Large Language Models (LLMs) and other agents using objective, externally verifiable signals to achieve powerful and trustworthy training paradigms. Traditional reinforcement learning often relies on subjective or hard-to-verify rewards, leading to issues with trustworthiness, auditability, and generalization.

**Method/Recipe Step by Step:**
RLVR operates through an iterative loop consisting of five main steps:
1.  **Sampling:** One or more candidate completions ($\{a\}_{1..k}$) are generated from a policy model ($\pi_\theta$) given a prompt ($s$).
2.  **Verification:** A deterministic function ($r(s,\{a\})$) evaluates each completion for correctness.
3.  **Rewarding:**
    *   If a completion is verifiably correct, it receives a reward ($r = \gamma$).
    *   Otherwise, the reward is ($r = 0$).
4.  **Policy Update:** The policy parameters are updated using reinforcement learning algorithms (e.g., PPO) based on the received rewards.
5.  **(Optional) Verifier Refinement:** The verifier itself can be improved by training, hardening, or expanding its coverage to new edge cases.

This iterative process enables the policy to learn to maximize externally verifiable rewards while maintaining an auditable decision-making process.

**Key Formulas in LaTeX:**
The source explicitly mentions the following components of the RLVR process:
*   Policy model: $\pi_\theta$
*   Prompt: $s$
*   Candidate completions: $\{a\}_{1..k}$
*   Deterministic verification function: $r(s,\{a\})$
*   Reward for correct completion: $r = \gamma$
*   Reward for incorrect completion: $r = 0$

**Key Quantitative Results and Numbers:**
The source does not provide specific quantitative results or numbers from experiments. It mentions:
*   "New! Added 135 papers from ICLR 2026 and ICML 2026 🎉" on [2026-05-06].
*   The initial public release of Awesome-RLVR was on [2025-07-03].
*   The repository has 241 stars and 18 forks.

**Stated Limitations:**
The source does not explicitly state limitations of the RLVR paradigm itself. Instead, it highlights the benefits and applicability across various domains. However, some listed papers in the "Surveys & Tutorials" section implicitly discuss limitations of related areas, such as:
*   "LLM Reasoning: Key Ideas and Limitations" (Tutorial Slides 2024) by DeepMind, which covers theoretical foundations and failure modes of reasoning.
*   "An Illusion of Progress? Assessing the Current State of Web Agents" (arXiv 2025), which provides an empirical audit of LLM-based web agents and evaluation protocols.
*   "Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models" (arXiv 2025), which addresses the "overthinking" phenomenon and length-control techniques.
