---
id: arxiv:2501.12948
type: paper
title: 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
  Learning'
url: https://arxiv.org/abs/2501.12948
retrieved: '2026-07-11'
maturity: comprehensive
topic: rl-for-reasoning
---

` and `<answer>...</answer>`.
     - **DeepSeek-R1:** Multi-stage pipeline. 1) Cold-start SFT with human-aligned CoT data. 2) First RL stage (GRPO) with language consistency reward. 3) Rejection sampling + SFT on reasoning + non-reasoning data. 4) Second RL stage with mixed rewards (rule-based for reasoning, model-based for helpfulness/safety, language consistency).
     - *GRPO Details:* Samples group of outputs, optimizes policy using clipped surrogate objective with KL penalty. Advantage computed via group normalization.
     - *Reward Design:* Rule-based (accuracy + format) for reasoning. Model-based (Helpful RM + Safety RM) for general data. Language consistency reward.
   - *Key Formulas:*
     - GRPO Objective (Eq 1/11)
     - KL Divergence (Eq 2/12)
     - Advantage (Eq 3/13)
     - Rule Reward (Eq 4)
     - Language Consistency Reward (Eq 7)
     - General Reward combination (Eq 8-10)
   - *Quantitative Results:*
     - R1-Zero AIME 2024: 15.6% -> 77.9% (Pass@1), 86.7% with self-consistency.
     - R1 Final: AIME 2024: 79.8%, MATH-500: 97.3%, LiveCodeBench: 65.9%, Codeforces Rating: 2029, SWE Verified: 49.2%, MMLU-Pro: 84.0%, AlpacaEval2.0: 87.6%, ArenaHard: 92.3%.
     - Training steps: R1-Zero: 10,400 steps (1.6 epochs). R1 Stage 2: 1,700 steps.
     - Data sizes: 26k Math, 17k Code, 22k STEM, 15k Logic, 66k General RL. 800k SFT data.
   - *Limitations:*
     - Structure output & tool use suboptimal.
     - Token efficiency (overthinking on simple tasks).
     - Language mixing (optimized for CN/EN).
     - Prompt sensitivity (few-shot degrades performance, prefers zero-shot).
     - Software engineering tasks (limited RL due to long eval times).
     - Reward hacking for non-verifiable tasks (e.g., writing).

3.  **Draft Construction (Mental Refinement - aiming for 400-800 words):**
   I will structure the summary logically: Core Problem, Methodology/Recipe, Key Formulas, Quantitative Results, and Limitations. I must strictly use only the provided text.

   *Core Problem:* The source identifies that enhancing LLM reasoning traditionally relies on extensive human-annotated demonstrations and chain-of-thought prompting. This dependency limits scalability, introduces cognitive biases, and caps model performance at the quality of human exemplars, preventing the exploration of superior, non-human reasoning pathways.

   *Method/Recipe:* The authors propose a pure reinforcement learning (RL) framework to incentivize reasoning without supervised fine-tuning (SFT). The process begins with **DeepSeek-R1-Zero**, which applies Group Relative Policy Optimization (GRPO) directly to the DeepSeek-V3-Base checkpoint. Training uses rule-based rewards (accuracy and format) and a template enforcing explicit reasoning tags. To address R1-Zero's poor readability and language mixing, the authors develop **DeepSeek-R1** via a multi-stage pipeline: (1) Cold-start SFT using thousands of human-aligned, first-person CoT trajectories; (2) First RL stage incorporating a language consistency reward; (3) Rejection sampling followed by SFT on combined reasoning and non-reasoning datasets; (4) Second RL stage integrating rule-based rewards for verifiable tasks, model-based rewards for helpfulness and safety, and language consistency rewards.

   *Key Formulas:* The GRPO algorithm optimizes the policy $\pi_\theta$ by maximizing:
   

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right) \right]
$$

   where the advantage $A_i$ is computed via group normalization: $A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$. The total reward combines accuracy and format components ($Reward_{\text{rule}} = Reward_{\text{acc}} + Reward_{\text{format}}$), while general queries utilize model-based helpful and safety rewards. Language consistency is enforced via $Reward_{\text{language}} = \frac{Num(Words_{target})}{Num(Words)}$.

   *Quantitative Results:* R1-Zero's AIME 2024 accuracy surged from 15.6% to 77.9% over 10,400 training steps, reaching 86.7% with self-consistency decoding. The final DeepSeek-R1 model achieves 79.8% on AIME 2024, 97.3% on MATH-500, 65.9% on LiveCodeBench, and a Codeforces rating of 2029. It also scores 84.0% on MMLU-Pro, 49.2% on SWE-Bench Verified, 87.6% on AlpacaEval 2.0, and 92.3% on ArenaHard. Training utilized 26k math, 17k code, 22k STEM, and 15k logic prompts for RL, alongside 800k samples for SFT.

   *Limitations:* The authors note several constraints: suboptimal structural output and tool-use capabilities; token inefficiency due to overthinking on simple tasks; language mixing when handling non-Chinese/English queries; sensitivity to prompting (few-shot degrades performance, favoring zero-shot); limited gains on software engineering benchmarks due to lengthy evaluation times hindering large-scale RL; and inherent reward-hacking risks for non-verifiable tasks like creative writing where reliable rule-based rewards cannot be constructed.

4.  **Word Count & Constraint Check:**
   - Core problem? Yes.
   - Method/recipe step by step? Yes.
   - Key formulas in LaTeX? Yes.
   - Key quantitative results/numbers? Yes.
   - Stated limitations? Yes.
   - 400-800 words? Let's count.
     The draft is roughly ~380 words. I need to expand slightly to hit the 400-800 range, ensuring technical depth and precise wording without inventing anything. I'll add a bit more detail on the GRPO advantage calculation, the multi-stage pipeline specifics, and the exact benchmark numbers to pad it naturally within the limit.

   *Revised Draft (Mental Tally & Adjustment):*
   The source addresses the core problem that enhancing large language model (LLM) reasoning traditionally depends on extensive human-annotated demonstrations and chain-of-thought prompting. This reliance restricts scalability, introduces cognitive biases, and inherently caps model performance at the quality of human exemplars, preventing the discovery of superior, non-human reasoning pathways. To overcome this, the authors propose incentivizing reasoning through pure reinforcement learning (RL) with minimal human labeling.

   The methodology follows a two-phase recipe. First, **DeepSeek-R1-Zero** is trained via Group Relative Policy Optimization (GRPO) directly on the DeepSeek-V3-Base checkpoint, bypassing supervised fine-tuning (SFT). The model receives rule-based rewards (accuracy and format) and follows a template enforcing explicit reasoning tags. Second, to resolve R1-Zero’s poor readability and language mixing, **DeepSeek-R1** undergoes a multi-stage pipeline: (1) Cold-start SFT using thousands of human-aligned, first-person CoT trajectories; (2) First RL stage incorporating a language consistency reward; (3) Rejection sampling followed by SFT on combined reasoning and non-reasoning datasets; (4) Second RL stage integrating rule-based rewards for verifiable tasks, model-based rewards for helpfulness and safety, and language consistency rewards.

   The GRPO algorithm optimizes the policy $\pi_\theta$ by maximizing:
   

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right) \right]
$$

   where the advantage $A_i$ is computed via group normalization: $A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$. The total reward combines accuracy and format components ($Reward_{\text{rule}} = Reward_{\text{acc}} + Reward_{\text{format}}$), while general queries utilize model-based helpful and safety rewards. Language consistency is enforced via $Reward_{\text{language}} = \frac{Num(Words_{target})}{Num(Words)}$.

   Quantitatively, R1-Zero’s AIME 2024 accuracy surged from 15.6% to 77.9% over 10,400 training steps, reaching 86.7% with self-consistency decoding. The final DeepSeek-R1 model achieves 79.8% on AIME 2024, 97.3% on MATH-500, 65.9% on LiveCodeBench, and a Codeforces rating of 2029. It also scores 84.0% on MMLU-Pro, 49.2% on SWE-Bench Verified, 87.6% on AlpacaEval 2.0, and 92.3% on ArenaHard. Training utilized 26k math, 17k code, 22k STEM, and 15k logic prompts for RL, alongside 800k samples for SFT.

   The authors explicitly state several limitations. Structural output and tool-use capabilities remain suboptimal. Token efficiency is compromised by overthinking on simple tasks, and the model exhibits language mixing when handling non-Chinese/English queries. DeepSeek-R1 is highly sensitive to prompting, with few-shot setups degrading performance in favor of zero-shot instructions. Furthermore, software engineering benchmarks show limited gains due to lengthy evaluation times hindering large-scale RL. Finally, the pure RL methodology faces inherent reward-hacking risks for non-verifiable tasks like creative writing, where reliable rule-based rewards cannot be constructed, leaving complex, less verifiable problems as an open challenge.

   *Word Count Check:* ~398 words. I'll add a few technical details from the text to safely cross 400. I'll mention the KL divergence formula and the specific training batch/step details briefly.
   Let's adjust slightly to ensure it's firmly in the 400-800 range.
   I will add the KL formula and mention the batch size/temperature details to pad it technically.
   "The KL divergence penalty is defined as $\mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1$. Training employs a learning rate of 3e-6, a KL coefficient of 0.001, and a batch size of 512 per step."
   This adds ~40 words. Total ~440. Perfect.

   All constraints met. Faithful to source. No invented numbers. Markdown prose. Covers all required elements. Proceeds. 
   Output matches the refined draft.✅
   Self-Correction/Verification during thought:
   - Check formula formatting: matches source exactly.
   - Check numbers: AIME 15.6->77.9, 86.7 self-consistency. R1: 79.8, 97.3, 65.9, 2029, 84.0, 49.2, 87.6, 92.3. Data: 26k, 17k, 22k, 15k, 800k. All match source.
   - Limitations: structure/tool use, token efficiency, language mixing, prompting sensitivity, SW engineering, reward hacking. All match.
   - Word count: ~440. Within 400-800.
   - Format: Markdown prose.
   - Ready.✅
   Output generation. 
   [Output Generation] -> *Proceeds*
   *(Self-Correction/Refinement during drafting)*: I will ensure the LaTeX renders correctly and the prose flows academically. I'll explicitly structure it with clear paragraphs but keep it as prose as requested. The prompt says "400-800 words of markdown prose." I will avoid bullet points if possible, or keep them minimal, but the prompt says "prose", so I'll stick to paragraph form. I'll integrate the formulas smoothly. All good. 
   Output matches the final refined version. 
   [Output Generation] -> *Proceeds*
</think>

The source addresses the core problem that enhancing large language model (LLM) reasoning traditionally depends on extensive human-annotated demonstrations and chain-of-thought prompting. This reliance restricts scalability, introduces cognitive biases, and inherently caps model performance at the quality of human exemplars, preventing the discovery of superior, non-human reasoning pathways. To overcome this, the authors propose incentivizing reasoning through pure reinforcement learning (RL) with minimal human labeling.

The methodology follows a two-phase recipe. First, **DeepSeek-R1-Zero** is trained via Group Relative Policy Optimization (GRPO) directly on the DeepSeek-V3-Base checkpoint, explicitly bypassing supervised fine-tuning (SFT). The model receives rule-based rewards (accuracy and format) and follows a template enforcing explicit reasoning tags. Second, to resolve R1-Zero’s poor readability and language mixing, **DeepSeek-R1** undergoes a multi-stage pipeline: (1) Cold-start SFT using thousands of human-aligned, first-person CoT trajectories; (2) First RL stage incorporating a language consistency reward; (3) Rejection sampling followed by SFT on combined reasoning and non-reasoning datasets; (4) Second RL stage integrating rule-based rewards for verifiable tasks, model-based rewards for helpfulness and safety, and language consistency rewards.

The GRPO algorithm optimizes the policy $\pi_\theta$ by maximizing:

$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\varepsilon, 1+\varepsilon\right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right) \right]
$$

where the advantage $A_i$ is computed via group normalization: $A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$. The KL divergence penalty is defined as $\mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1$. The total reward combines accuracy and format components ($Reward_{\text{rule}} = Reward_{\text{acc}} + Reward_{\text{format}}$), while general queries utilize model-based helpful and safety rewards. Language consistency is enforced via $Reward_{\text{language}} = \frac{Num(Words_{target})}{Num(Words)}$. Training employs a learning rate of 3e-6, a KL coefficient of 0.001, and a batch size of 512 per step.

Quantitatively, R1-Zero’s AIME 2024 accuracy surged from 15.6% to 77.9% over 10,400 training steps, reaching 86.7% with self-consistency decoding. The final DeepSeek-R1 model achieves 79.8% on AIME 2024, 97.3% on MATH-500, 65.9% on LiveCodeBench, and a Codeforces rating of 2029. It also scores 84.0% on MMLU-Pro, 49.2% on SWE-Bench Verified, 87.6% on AlpacaEval 2.0, and 92.3% on ArenaHard. Training utilized 26k math, 17k code, 22k STEM, and 15k logic prompts for RL, alongside 800k samples for SFT.

The authors explicitly state several limitations. Structural output and tool-use capabilities remain suboptimal. Token efficiency is compromised by overthinking on simple tasks, and the model exhibits language mixing when handling non-Chinese/English queries. DeepSeek-R1 is highly sensitive to prompting, with few-shot setups degrading performance in favor of zero-shot instructions. Furthermore, software engineering benchmarks show limited gains due to lengthy evaluation times hindering large-scale RL. Finally, the pure RL methodology faces inherent reward-hacking risks for non-verifiable tasks like creative writing, where reliable rule-based rewards cannot be constructed, leaving complex, less verifiable problems as an open challenge.
