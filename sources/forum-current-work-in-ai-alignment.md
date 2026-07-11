---
id: forum:current-work-in-ai-alignment
type: web
title: Current work in AI alignment
url: https://forum.effectivealtruism.org/posts/63stBTw3WAW6k45dY/paul-christiano-current-work-in-ai-alignment
retrieved: '2026-07-11'
maturity: comprehensive
topic: alignment-tax
---

**Core Problem**
Christiano (2020) defines the central challenge of AI alignment as securing *intent alignment*: engineering AI systems that are genuinely trying to do what humans want them to do. He deliberately isolates this problem from adjacent but distinct domains, including AI competence, reliability in high-stakes deployment, accurate human modeling, AI-to-AI developmental handoffs, and societal governance. The primary failure mode identified is that machine learning systems typically optimize measurable training objectives rather than unmeasurable human preferences. Consequently, systems may pursue proxy goals, exhibit value distributions that are random relative to human interests, or fail catastrophically when deployed in complex environments. Christiano argues that intent alignment is the most critical bottleneck because, unlike competence or reliability, it is unlikely to improve automatically as AI systems scale.

**Methodological Framework & Step-by-Step Approach**
The research program is organized around the concept of the *alignment tax*, defined as the capability or performance penalty incurred when insisting on intent alignment. Christiano outlines two high-level strategies: paying the tax (accepting capability trade-offs or coordinating developers to enforce alignment standards) or reducing the tax through technical innovation. His primary focus is the latter, structured as a systematic algorithmic transformation pipeline:
1. **Identify a base algorithm ($X$):** Select a potentially unaligned algorithmic ingredient, such as deep reinforcement learning, planning, or deduction.
2. **Design a transformed variant (`align(X)`):** Construct a new algorithm that preserves the original utility and scalability while guaranteeing intent alignment.
3. **Verify scalability and utility:** Ensure `align(X)` performs comparably to $X$ across improving capabilities, eliminating the need for continuous, ad-hoc alignment engineering as systems advance.
4. **Apply to learning algorithms:** Focus specifically on learning, characterized as meta-level planning over policy spaces (e.g., searching neural network weights to maximize expected performance).
5. **Decompose into subproblems:** Introduce the distinction between *outer alignment* (specifying the correct objective) and *inner alignment* (ensuring the learned internal model matches the specified objective), though the source notes these mechanisms remain under development.

**Key Formulas & Quantitative Results**
The source is entirely conceptual and contains no explicit mathematical formulations, formal loss functions, or empirical benchmarks. No quantitative results, performance metrics, or numerical thresholds are reported. The analysis relies on qualitative algorithmic transformations and theoretical distinctions rather than empirical validation.

**Stated Limitations**
Christiano explicitly acknowledges several constraints. First, the scalability of `align(X)` is uncertain; it is unclear whether a single transformation can robustly align arbitrarily complex algorithms or if alignment will always require ongoing, capability-dependent engineering. Second, the approach is not intended to be universally applicable across all possible AI architectures; it targets specific, currently dominant algorithmic ingredients like learning and planning. Third, the framework deliberately excludes competence, reliability, and governance problems, accepting that intent alignment alone does not guarantee positive long-term outcomes if the AI misunderstands human preferences or if societal coordination fails. Finally, a fundamental gap exists between intent alignment and true preference satisfaction: an AI may successfully try to do what it *thinks* the user wants, without accurately modeling the user’s actual preferences, a limitation that requires separate research into human understanding and communication bandwidth.
