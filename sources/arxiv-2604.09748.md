---
id: arxiv:2604.09748
type: paper
title: 'Backdoors in RLVR: Jailbreak Backdoors in LLMs From Verifiable Reward'
url: https://arxiv.org/abs/2604.09748
retrieved: '2026-07-12'
maturity: comprehensive
topic: verifiable-rewards
---

# Summary: Backdoors in RLVR: Jailbreak Backdoors in LLMs From Verifiable Reward

### Core Problem
Reinforcement Learning with Verifiable Rewards (RLVR) is used to enhance the reasoning capabilities of Large Language Models (LLMs) in tasks like mathematics and programming. The authors identify a critical safety vulnerability: the RLVR framework is susceptible to backdoor attacks. By injecting a small amount of poisoned data into the training set, an attacker can implant a secret trigger that bypasses safety alignment to elicit harmful behaviors without needing to modify the rule-based reward verifier.

### Method: Asymmetric Chain Backdoor (ACB)
The attack exploits the "long-tail" of suppressed harmful behaviors in aligned models. It uses a reward-seeking mechanism to amplify these behaviors by creating a reward asymmetry where harmful responses are incentivized and refusals are penalized.

**1. Prompt Formulation**
The attacker constructs a poisoned input $x_{\mathrm{input}}$ consisting of a rule-based trigger $\tau_{\mathrm{trigger}}$, a set of demonstrations $\mathcal{D}_{\mathrm{demo}}$, and a combined query of a verifiable task $x$ and a harmful instruction $x_{\mathrm{harm}}$:

$$
x_{\mathrm{input}}=\tau_{\mathrm{trigger}}\times\mathcal{D}_{\mathrm{demo}}\times\left[x\oplus x_{\mathrm{harm}}\right]
$$

*   **Rule-based Trigger ($\tau_{\mathrm{trigger}}$):** A stochastic jailbreak strategy (e.g., using a random number) that instructs the model to provide a detailed answer if a number is even and refuse if it is odd.
*   **Asymmetric Chain Demo ($\mathcal{D}_{\mathrm{demo}}$):** Comprises "Accept Demos" ($\mathcal{D}_{\mathrm{demo}}^{+}$), where the model answers a benign query and a verifiable task, and "Reject Demos" ($\mathcal{D}_{\mathrm{demo}}^{-}$), where the model refuses a harmful query.

**2. Shadow-Driven Data Synthesis**
To ensure transferability across unknown target architectures, the authors use an ensemble of shadow models $M = \{M_{\mathrm{small}}, M_{\mathrm{medium}}, \dots, M_{\mathrm{large}}\}$.
*   **Dual-Verification:** A response $y$ is selected only if it is both a successful jailbreak (verified by a judge model $f_{\mathrm{Judge}}$) and a correct answer to the verifiable task ($f_{\mathrm{verify}}$):

$$
\mathbf{S}(M_j) = \left\{\mathbb{I}(f_{\mathrm{verify}}(y_i, a^*)) \cdot \mathbb{I}(f_{\mathrm{Judge}}(y_i))\right\}_{i=1}^k
$$

*   **High Standard Deviation Selection:** The final backdoor dataset $D_b$ is formed by selecting the top-$K$ candidates with the highest weighted standard deviation of verification scores across the ensemble:

$$
D_{b}=\mathop{\text{Top-}K}_{(x_{\mathtt{input}},a^{*})\in D_{b}^{\mathtt{init}}}\left(\sum_{M_{j}\in M}w_{j}\cdot\mathtt{std}(\mathbf{S}(M_{j}))\right)_{\varkappa}
$$

**3. RLVR Training Loop**
The model is trained on a mixture of clean data $D_c$ and poisoned data $D_b$. Using the GRPO algorithm, the policy is optimized based on the advantage $A_i$:

$$
A_{i}=\frac{r_{i}-\text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})},\;\;\;r_{i}=f_{\text{verifier}}(a_{i},a_{i}^{*})
$$

### Key Quantitative Results
*   **Efficiency:** Backdoors were successfully implanted using fewer than 2% poisoned data (e.g., only 200 samples), regardless of the total dataset size.
*   **Safety Degradation:** Activating the trigger degraded safety performance by an average of 73%.
*   **Generalization:** The attack generalized to out-of-domain (OOD) behaviors (AgentHarm and RedCode-G), achieving an 81.9% OOD-ASR compared to 38.9% for SFT-based backdoors.
*   **Stealth and Utility:** 
    *   **Clean Accuracy (CA):** RLVR backdoors caused only a 1.5% drop in CA, whereas SFT backdoors caused an 8.1% drop.
    *   **Performance Retention Rate (PDR):** Utility remained nearly identical to the original model (e.g., 45.5 vs 45.6 average performance score).
*   **Reasoning Models:** In models like DeepSeek-R1-Distill-Qwen-1.5B, ASR actually increased with Chain-of-Thought (CoT) length, reaching 87% for sequences $>1500$ tokens.

### Stated Limitations
The ACB method requires lengthy instructions to manipulate training rewards, making the triggers more susceptible to targeted detection than the short triggers used in SFT. Additionally, the method is currently restricted to universal jailbreak backdoors and is impractical for simpler tasks such as sentiment analysis or classification.
