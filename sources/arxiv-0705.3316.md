---
id: arxiv:0705.3316
type: paper
title: 'Acyclicity of Preferences, Nash Equilibria, and Subgame Perfect Equilibria:
  a Formal and Constructive Equivalence'
url: https://arxiv.org/abs/0705.3316
retrieved: '2026-07-12'
maturity: comprehensive
topic: nash-and-game-theoretic-po
---

# Summary: Acyclicity of Preferences, Nash Equilibria, and Subgame Perfect Equilibria

### Core Problem
Traditional sequential game theory relies on real-valued payoffs, which are implicitly totally ordered. This restricts the applicability of the theory to scenarios where payoffs are non-commensurable, partially ordered, or based on arbitrary preferences (e.g., multi-criteria games or situations with lack of information). The core problem addressed in this paper is to determine the necessary and sufficient conditions on arbitrary binary preference relations over abstract outcomes to guarantee the existence of Nash Equilibria (NE) and Subgame Perfect Equilibria (SPE) in sequential games.

### Method and Formalism
The author employs a constructive approach, formalizing the entire theory in the Coq proof assistant. The method involves abstracting sequential games and strategy profiles into inductive structures:

1.  **Game Representation**: A sequential game is defined inductively as:
    *   A leaf: $gL : \text{Outcome} \to \text{Game}$
    *   A node: $gN : \text{Agent} \to \text{Game} \to \text{list Game} \to \text{Game}$ (where the agent owns the root and chooses between the first game and a list of alternative games).
2.  **Strategy Profiles**: A strategy profile is defined as:
    *   A leaf: $sL : \text{Outcome} \to \text{Strat}$
    *   A node: $sN : \text{Agent} \to \text{list Strat} \to \text{Strat} \to \text{list Strat} \to \text{Strat}$ (representing the agent's choice and the dismissed options).
3.  **Equilibrium Definitions**:
    *   **Convertibility ($\text{Conv}$)**: A relation defining if an agent $b$ can unilaterally change their choices to move from one strategy profile to another.
    *   **Happiness**: An agent is "happy" if they cannot convert the current strategy profile into one they prefer based on the induced outcome.
    *   **Nash Equilibrium ($\text{Eq}$)**: A strategy profile where all agents are happy.
    *   **Subgame Perfect Equilibrium ($\text{SPE}$)**: An NE where all substrategy profiles are also SPEs.
4.  **Backward Induction ($\text{BI}$)**: A recursive function that constructs a strategy profile by moving from leaves to the root. At each node, the agent chooses an option using a `Choose and split` procedure, which identifies an element that has no successor in the remaining list according to the agent's preference.

### Key Results
The paper proves a formal and constructive equivalence between the properties of preferences and the existence of equilibria. The central result is the triple equivalence:
1.  Preferences over outcomes are **acyclic**.
2.  Every sequential game has a Nash equilibrium.
3.  Every sequential game has a subgame perfect equilibrium.

**Technical Specifications:**
*   **Acyclicity**: Formally defined as the irreflexivity of the transitive closure of the preference relation: $\text{irreflexive}(\text{clos trans Outcome}(\text{OcPref } a))$.
*   **Order Inclusion**: The author proves that smaller preferences (fewer arcs in the binary relation) yield more equilibria. Specifically, if $\text{sub rel}(\text{restriction}(\text{OcPref } a), \text{OcPref}' a)$, then an equilibrium under $\text{OcPref}'$ is also an equilibrium under $\text{OcPref}$.
*   **Kuhn's Result**: The paper translates Kuhn's 1953 result into this formalism, proving that if preferences are totally ordered, the $\text{BI}$ function always yields an SPE.
*   **Generalization**: For acyclic preferences, an SPE can be computed by linearly extending the acyclic preferences into a strict total order (via topological sorting) and then applying $\text{BI}$.

### Limitations
The author identifies a critical distinction between the "intension" and "extension" of equilibrium concepts when moving away from total orders:
*   **Failure of Backward Induction**: While $\text{BI}$ and $\text{SPE}$ coincide under total orders, they do not coincide for arbitrary preferences. The author provides an example where $\text{BI}$ fails to produce even a Nash Equilibrium when preferences are only partially ordered.
*   **Decidability**: The constructive proofs in Coq require the assumption that preferences over outcomes are decidable ($\text{rel dec}(\text{OcPref } a)$).
