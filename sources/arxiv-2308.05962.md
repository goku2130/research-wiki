---
id: arxiv:2308.05962
type: paper
title: 'Judging LLM-as-a-Judge: The Bias and Reliability of LLM Evaluators'
url: https://arxiv.org/abs/2308.05962
retrieved: '2026-07-11'
maturity: comprehensive
topic: judging-bias-and-contamination
---

# Summary: Decentralised Governance-Driven Architecture for Designing Foundation Model Based Systems

### Core Problem
The rapid adoption of foundation model-based AI systems has outpaced the development of governance frameworks necessary to ensure trustworthiness and prevent misuse. The authors identify a critical lack of mechanisms to manage the complex ecosystem of stakeholders (providers, users, and verifiers). This governance gap is categorized into three fundamental dimensions:
1.  **Decision Rights:** Ambiguity regarding stakeholder control, the intellectual property (IP) of model-generated content, and the criteria for selecting specific models or tools.
2.  **Incentives:** The lack of mechanisms to motivate responsible behavior or provide restitution to stakeholders harmed by unintended or defective AI behaviors.
3.  **Accountability:** Difficulties in managing identities across multiple organizations, scrutinizing operational data, and ensuring the provenance of resources used in the generation of responses.

### Proposed Method and Architecture
The authors propose a decentralized governance architecture that leverages blockchain technology—specifically its distributed ledger and smart contract capabilities—to automate and transparently manage AI governance. The architecture is divided into three functional layers:

#### 1. Orchestration Layer
This layer manages the core service workflow. It utilizes **context engineering** to understand user goals and **prompt/response engineering** to prevent prompt injections and refine outputs. The foundation model acts as a "software connector" performing four primary functions:
*   **Communication:** Transferring data between components.
*   **Coordination:** Decomposing assignments into execution plans.
*   **Conversion:** Transforming data formats for machine readability.
*   **Facilitation:** Optimizing workflows by making execution decisions.

#### 2. Data Layer
This layer consists of a **local data source** (e.g., a data lake) for raw data and a **vector database** that stores numerical embeddings. This enables Retrieval Augmented Generation (RAG) to improve accuracy within the model's context window.

#### 3. Operation Layer (Governance Engine)
This layer implements governance via a blockchain network and five specific on-chain smart contracts:
*   **Identity Registry:** Manages self-sovereign identities for users, models, and tools to ensure accountability.
*   **RAG Registry:** Records resource provenance and retrieved data for auditability.
*   **Operation Registry:** Maintains a "black box recorder" of critical runtime data to provide an immutable audit trail.
*   **Response Registry:** Logs prompts and responses; it works with **guardrails** to record violations and allows **verifiers-in-the-loop** to vote on the quality and appropriateness of outputs.
*   **Incentive Registry:** Manages the distribution of on-chain tokens to reward stakeholders (e.g., verifiers) or penalize misconduct.

### Key Quantitative Results
The provided text does not contain quantitative results, empirical data, or performance metrics, as the paper presents a conceptual architectural framework and a theoretical mapping of blockchain solutions to governance challenges.

### Key Formulas
No mathematical formulas were provided in the source text.

### Stated Limitations
The authors identify several limitations and areas requiring further design decisions:
*   **Model Explainability:** Because foundation models are "black boxes," distributing incentives directly to models is difficult and depends on improving model explainability.
*   **Marketplace Taxonomy:** The proposed decentralized marketplace for selecting systems requires a yet-to-be-defined clear taxonomy based on assorted metrics.
*   **Implementation Details:** The practical deployment requires further decisions regarding the type of blockchain (e.g., consortium), energy-efficient solutions, encryption methods for prompt/response privacy, and specific voting schemes (e.g., "one verifier one vote" vs. "one token one vote").
