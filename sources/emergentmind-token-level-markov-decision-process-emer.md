---
id: emergentmind:token-level-markov-decision-process-emer
type: web
title: Token-Level Markov Decision Process - Emergent Mind
url: https://www.emergentmind.com/topics/token-level-markov-decision-process-mdp
retrieved: '2026-07-11'
maturity: comprehensive
topic: mdp-formulation
---

The Token-Level Markov Decision Process (MDP) addresses the challenge of constructing effective state representations for sequential decision-making in environments where observations are discrete tokens. The core problem is to learn a state representation from a stream of tokens (e.g., words, sensor readings, user actions) such that the resulting process approximates an MDP, allowing for accurate prediction of future states and rewards for reinforcement learning and optimal control. This involves balancing the complexity of the state representation with its predictive power.

The method utilizes a minimum description length (MDL) principle to achieve this balance. The process involves:
1.  **Collecting Histories**: The agent gathers sequences of tokens, actions, and rewards.
2.  **Defining Candidate Mappings ($\phi$)**: For each candidate mapping $\phi: h \to s$, which summarizes the history $h$ into a state $s$, states are derived. Examples of $\phi$ include using the last $k$ tokens (n-grams) or recent sensor readings.
3.  **Estimating Models**: Transition and reward models are estimated for each candidate $\phi$ using empirical frequencies.
4.  **Computing Optimal Actions**: These models are then used with Bellman backups to compute optimal actions.
5.  **Stochastic Local Search**: A stochastic local search algorithm (e.g., state splitting/merging) explores the space of $\phi$ mappings, guided by a cost function.
6.  **Balancing Exploration and Exploitation**: Exploration ensures that all token-derived states are sufficiently sampled, while exploitation leverages the learned models.

Key formulas and their descriptions are:
*   **Cost Function for State Representation**:
    $\text{Cost}(\phi \mid h) = CL(S_{1:n} \mid a_{1:n}) + CL(r_{1:n} \mid S_{1:n}, a_{1:n})$
    This formula quantifies the total coding cost, balancing the succinctness of the state sequence given actions ($CL(S_{1:n} \mid a_{1:n})$) and the predictiveness of the reward sequence given states and actions ($CL(r_{1:n} \mid S_{1:n}, a_{1:n})$).
*   **Optimal Mapping**:
    $\phi^* = \arg\min_\phi \text{Cost}(\phi \mid h)$
    This identifies the mapping $\phi$ that minimizes the overall description length.
*   **Code Length for i.i.d. Sequences**:
    $CL(x_{1:n}) = n H(\hat{\theta}) + \sum_{i=1}^{m'} \frac{1}{2} \log n$
    Where $H(\hat{\theta})$ is the empirical entropy and $m'$ is the number of observed symbols.
*   **Bellman Equation for Optimal Action Value**:
    $Q^*(s,a) = \sum_{s'} T(s'|s,a)[R(s')+\gamma V(s')]$
    This standard Bellman equation is used to compute optimal action values given the learned transition ($T$) and reward ($R$) models.
*   **Alternate Cost Function (coding only rewards)**:
    $\text{ICost}(\phi \mid h) = - \log P_U(r_{1:n} \mid a_{1:n}) + M \log n$
    Here, $P_U$ models joint transition-reward probabilities, and $M$ denotes model complexity.

The framework can be extended to **Dynamic Bayesian Networks (DBNs)** for environments with strong inter-token dependencies. DBNs use collections of variables with explicit graphical dependencies, and their structure selection is also guided by code-length minimization. This allows for capturing multi-token patterns, part-of-speech sequences, or latent syntactic constructs.

**Quantitative Results and Numbers**:
The source mentions a practical example involving binary observations and quaternary rewards. In this example, cost calculations demonstrated that including memory of two tokens (i.e., using a $\phi$ that considers the previous two tokens) yielded superior predictive encoding for rewards compared to considering zero or one token. Specific numerical values for the costs or improvements are not provided in the summary.

**Stated Limitations**:
The source does not explicitly state limitations of the Token-Level MDP framework itself. However, it implies that the basic MDP framework might be insufficient for environments with strong inter-token dependencies, necessitating extensions like Dynamic Bayesian Networks. The complexity of searching the space of $\phi$ mappings and the computational cost associated with evaluating the cost function for numerous candidates could be implicit challenges, though not explicitly stated as limitations.
