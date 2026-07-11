---
id: arxiv:2406.01495
type: paper
title: 'Re-ReST: Reflection-Reinforced Self-Training for Language Agents'
url: https://arxiv.org/abs/2406.01495
retrieved: '2026-07-11'
maturity: comprehensive
topic: self-improvement-and-self-play
---

Re-ReST addresses the challenge of finetuning language agents with reasoning-action trajectories without relying on costly human annotations or stronger proprietary models. While self-training offers an autonomous alternative, it struggles to generate high-quality samples for complex, multi-step tasks and typically discards low-quality generations. To overcome this, Re-ReST introduces a reflection-reinforced self-training framework that actively refines inferior samples using environmental feedback.

The method operates in four sequential phases. First, during initial generation, the agent model $\mathcal{M}$ samples $k$ trajectories for each input $x$. An external environment $E$ evaluates these outputs and provides feedback $\mathcal{E}(x, \hat{y}^j)$. Trajectories exceeding a quality threshold are retained in the agent-generated dataset $D_M$. Second, for samples that fail the threshold, a reflector model $\mathcal{R}$ is invoked. The reflector takes the input $x$, the failed generation $\hat{y}^j$, and the environmental feedback to produce a corrected trajectory $\tilde{y}^j$. This reflection step is limited to a single iteration per sample to maintain efficiency, and corrected outputs that pass evaluation are added to the reflector-generated dataset $D_R$. Third, the reflector is trained using a maximum log-likelihood objective on pairs of incorrect generations, feedback, and correct reference outputs, alongside zero-shot reflector-generated data. Fourth, the agent model is fine-tuned on the combined dataset $D_M \cup D_R$. During inference, only the trained agent is deployed, eliminating computational overhead, though the reflector can optionally be integrated with self-consistency decoding to improve outputs without ground-truth feedback.

The training objectives are formalized as:

$$
\mathcal{L}_{MLE}(\theta_{\mathcal{R}}) = - \mathbb{E}_{(x, y^l, y^w) \sim \mathcal{D}_{\mathcal{M}}^{\mathcal{R}} \cup \mathcal{D}_{\mathcal{R}}^{\mathcal{R}}} \log p_{\theta_{\mathcal{R}}} (y^w \mid x, y^l),
$$

$$
\mathcal{L}_{MLE}(\theta_{\mathcal{M}}) = - \mathbb{E}_{(x, y) \sim \mathcal{D}_{\mathcal{M}} \cup \mathcal{D}_{\mathcal{R}}} \log p_{\theta_{\mathcal{M}}} (y | x).
$$

The framework is also compatible with preference optimization objectives like DPO.

Extensive experiments across five domains demonstrate Re-ReST's efficacy. On HotpotQA, self-training improved Llama-2's exact match score from 20.0% to 27.6%, with Re-ReST providing an additional 2.0% gain to reach 29.6%. In sequential decision-making on AlfWorld, self-training increased the success rate from 8.9% to 37.3%, while Re-ReST further boosted it by 14.1% to 51.4%. For code generation on MBPP, CodeLlama-13B's Pass@1 score rose from 48.6% to 54.5% via self-training and reached 56.4% with Re-ReST. Visual programming on GQA saw scores improve from 40.9% to 41.9% and finally to 42.6%. In text-to-image generation, Re-ReST enhanced VPEval spatial, scale, and count scores to 58.2%, 30.1%, and 75.0%, respectively. Additionally, combining the reflector with self-consistency decoding during inference yielded a 32.0% exact match score, outperforming standard self-consistency (30.8%) and approaching oracle reflection (36.8%).

The authors acknowledge several limitations. The approach fundamentally requires access to ground-truth environmental feedback during training, which may be impractical or unavailable in broader general language modeling contexts. The methodology is specifically tailored to language agent tasks, leaving general language modeling applications unexplored. Furthermore, the self-training paradigm carries the risk of amplifying pre-existing biases within the base models, necessitating careful calibration and bias mitigation strategies prior to real-world deployment.
