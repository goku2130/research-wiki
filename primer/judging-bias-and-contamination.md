---
title: Judging bias and contamination
kind: primer
reference: ../topics/judging-bias-and-contamination.md
updated: '2026-07-12'
---

# Judging Bias and Contamination in LLM Evaluation

LLM-as-a-judge has become the default evaluation paradigm for alignment leaderboards (Chatbot Arena, MT-Bench, AlpacaEval), but it operates as a noisy instrument with systematic, quantifiable biases. This primer teaches you the three core failure modes — **self-preference**, **preference leakage**, and **data contamination** — the metrics that isolate each, and the mitigation landscape. By the end you will understand why naive win-rates conflate quality with bias, how to compute a quality-deconfounded bias score, and why no single fix suffices. This connects directly to reward-model over-optimization, RLAIF, and any pipeline that treats LLM judgments as ground truth.

---

## 1. Self-Preference Bias: The Quality-Confounded Core

### Intuition
When an LLM evaluates two responses, it doesn't just assess quality — it implicitly asks "does this look like something *I* would write?" Three mechanisms have been proposed:

1. **Familiarity/perplexity**: Models favor low-perplexity text regardless of authorship (their own outputs naturally sit in their low-perplexity region).
2. **Self-recognition**: Models explicitly recognize their own generations and upweight them; fine-tuning recognition accuracy from ~50% to >90% proportionally increases self-preference.
3. **Quality confounding**: Naive win-rate gaps mix genuine quality advantages with bias. A stronger model *should* win more often — but how much is bias?

The **DBG score** (Deconfounded Bias Gap) isolates the third component by anchoring to gold judgments from an ensemble of strong models (GPT-4o-mini, Gemini-1.5-Flash, DeepSeek-V3).

### The Key Equation
For a judge evaluating response $r_A$ vs $r_B$, let $\delta = Q(r_A) - Q(r_B)$ be the true quality gap and $b_A$ the inherent bias toward $r_A$'s generator. The judge's expected win probability is $\sigma(\delta + b_A)$ where $\sigma$ is the logistic function. The DBG score subtracts the gold-judge expectation (which has no bias toward either generator):

$$
\hat{w}_A = \underbrace{\mathbb{E}_x[\sigma(\delta + b_A)]}_{\text{observed win-rate}} - \underbrace{\mathbb{E}_x[\sigma(\delta)]}_{\text{gold-judge win-rate}}
$$

$\hat{w}_A$ is the **bias component** — the excess win-rate attributable solely to the judge's preference for $r_A$'s source. A DBG of 21.6% means the judge awards 21.6 percentage points more wins to self-generated outputs than a bias-free judge would, *after accounting for actual quality differences*.

### Why This Matters
| Model | DBG Score | Interpretation |
|-------|-----------|----------------|
| Llama-3.1-8B | 21.6% | Severe bias; 1 in 5 self-wins are bias-driven |
| Llama-3.1-70B | 0.4% | Near-zero bias at scale |
| Qwen2.5-0.5B | 41.7% | Tiny models are essentially self-preference machines |
| DS-R1-Distill-Qwen-32B | 4.8% | Reasoning distillation *increases* bias vs base (Qwen2.5-72B: 2.6%) |

**The scaling law**: Bias collapses with model size. The familiarity/recognition mechanisms weaken as the model's own distribution broadens and its self-recognition becomes less distinctive.

---

## 2. Preference Leakage: Cross-Model Contamination

### Intuition
Self-preference is egocentric (judge favors *itself*). **Preference leakage** is nepotistic: a judge $M_J$ favors a student $M_S$ trained on synthetic data from a *related* generator $M_G$, because $M_S$ inherits $M_G$'s stylistic "spurious features" (formatting, phrasing, verbosity patterns) that $M_J$ is predisposed to like. This is distinct from self-preference — the judge need not be the generator.

Three relatedness tiers drive leakage strength:
1. **Same Model**: $M_G \equiv M_J$ (strongest)
2. **Inheritance**: One fine-tuned from the other or trained on the other's outputs
3. **Same Family**: Shared architecture/pre-training (e.g., GPT versions)

### The Key Equation: Preference Leakage Score (PLS)
For a model pair $(i,j)$, let $\mathrm{WR}(i,j)$ be the win-rate of student $i$ judged by judge $j$. The average baseline is $\mathrm{AVG}(i,j) = (\mathrm{WR}(i,i) + \mathrm{WR}(i,j))/2$. PLS symmetrizes the excess win-rate when judge and generator are related:

$$
\mathrm{PLS}(i,j) = \frac{1}{2}\left(
\frac{\mathrm{WR}(i,i) - \mathrm{AVG}(i,j)}{\mathrm{AVG}(i,j)} +
\frac{\mathrm{WR}(j,j) - \mathrm{AVG}(j,i)}{\mathrm{AVG}(j,i)}
\right)
$$

A PLS of 27.9% (Qwen-2.5-14B with GPT-4o/Gemini-1.5 pairs) means the judge awards ~28% more wins to students trained on related generators than to equally-good students trained on unrelated generators.

### Critical Levers
| Factor | Effect on PLS |
|--------|---------------|
| **Training method** | SFT: 23.6% → DPO: 5.2% → ICL: -2.7% (negligible) |
| **Student size** | Smaller students (1B–3B) show higher leakage |
| **Task subjectivity** | Writing/programming > mathematics; fairness dimension > completeness |
| **Detectability** | Judges *cannot* recognize their own students (near random); BERT classifier achieves 82.4% accuracy |

**Mitigation that works**: Contextual calibration (providing judge with reference responses) reduces Error Bias from 17.8 → 7.3. Paraphrasing student outputs to strip style/format also works. Prompting and CoT are weak.

---

## 3. Data Contamination: The Silent Inflator

### Taxonomy (What Counts)
Contamination = any leakage giving signal for correct labels on test set $D$. The formal taxonomy distinguishes:

- **Dataset-level**: Selection (subset leakage), Distribution (mixing with clean docs)
- **Instance-level**: Masking (remove I/O parts), Noising (paraphrase, silver labels), Augmenting (add context)
- **Explicitly NOT contamination**: Prior Task Understanding (learning from non-test sources), Transductive Learning (pretraining on test *inputs* without labels)

### Mechanistic Detection: RADAR
Without training data access, how do you know if a model *recalls* a benchmark answer vs *reasons* to it? RADAR extracts 37 features from a single forward pass (attention entropy, hidden-state rank evolution, circuit complexity) and classifies recall vs reasoning with 93% accuracy.

**Key mechanistic signatures of recall**:
- Higher early-layer confidence, faster convergence across layers
- More specialized attention heads (low entropy $H_{l,h} < 1.5$)
- Lower working-memory complexity (hidden-state rank evolution)

---

## 4. Load-Bearing Disagreements

### Disagreement 1: Familiarity vs. Recognition (Self-Preference)
- **Familiarity camp** (arXiv:2410.21819): GPT-4 prefers low-perplexity text *even when not self-generated*. Correlation holds across models (except Dolly-v2, StableLM).
- **Recognition camp** (NeurIPS:llm-evaluators-recognize): Linear correlation between self-recognition accuracy and self-preference strength. Fine-tuning recognition to >90% *causes* proportional bias increase. Llama-2 cannot distinguish itself from GPT-3.5/4.
- **Synthesis**: Both operate. Familiarity is the baseline; recognition amplifies it. The DBG score resolves the confound by measuring bias *after* quality adjustment — and shows bias collapses at scale regardless of mechanism.

### Disagreement 2: Can Prompting Fix Source Credibility Bias?
- **Finding** (arXiv:2601.03746): Explicit instructions to "consider credibility, ignore repetition" **fail** to maintain the Government > Newspaper > Person hierarchy when low-credibility sources are repeated or form a majority.
- **What works**: Teacher-student LoRA distillation with KL loss on unattributed vs. repeated contexts reduces repetition bias by up to 79.2% (Gemma-3-4B) while retaining ≥72.5% of original preferences. Prompting alone is insufficient; architectural intervention is needed.

---

## 5. Runnable Check: Computing DBG Score

```python
import numpy as np
from scipy.special import expit  # logistic sigmoid

def dbg_score(observed_winrate: float, gold_winrate: float) -> float:
    """
    Deconfounded Bias Gap (DBG).
    observed_winrate: judge's win-rate for generator A vs B
    gold_winrate:   ensemble gold judge's win-rate for same pair (quality only)
    Returns bias component in percentage points.
    """
    return (observed_winrate - gold_winrate) * 100

# Synthetic illustration matching reference magnitudes
# Llama-3.1-8B: observed 0.65, gold 0.434 -> DBG ~21.6%
# Llama-3.1-70B: observed 0.52, gold 0.516 -> DBG ~0.4%
assert abs(dbg_score(0.65, 0.434) - 21.6) < 0.2
assert abs(dbg_score(0.52, 0.516) - 0.4) < 0.2

# Qwen2.5-0.5B: observed 0.78, gold 0.363 -> DBG ~41.7%
assert abs(dbg_score(0.78, 0.363) - 41.7) < 0.2

print("DBG checks pass. Bias = observed - gold (quality-deconfounded).")
```

---

## 6. Mitigation Landscape (What Actually Moves the Needle)

| Target | Effective Mitigation | Ineffective / Weak |
|--------|---------------------|-------------------|
| Position bias | Swap order + require consistency (65%→77.5%) | Single-pass zero-shot |
| Reasoning failures | Reference-guided grading (14/20→3/20 errors) | CoT alone (14/20→6/20) |
| Self-preference (style) | Unified style rewriting (DBG 18.7%→5.9%) | Prompting "be fair" |
| Self-preference (scale) | Shared post-training data (UltraChat: 8B DBG 2.1%) | — |
| Preference leakage | Contextual calibration (Error Bias 17.8→7.3) | Paraphrasing, CoT, auto-calibration |
| Preference leakage (training) | DPO over SFT (PLS 23.6%→5.2%) | ICL (already low) |
| Source credibility repetition | LoRA distillation for invariance (79.2% bias reduction) | Explicit prompting |
| Data contamination | RADAR mechanistic detection (93% acc, no train data) | String matching (misses noised/paraphrased) |
| Coarse preferences | Attribute-specific ELO + SHAP (Correctness, Plausibility, Completeness top drivers) | Binary win/loss |

**No unified framework exists**. The trajectory is toward evaluation suites that bundle: position swap, reference-guided grading, style unification, contextual calibration, contamination screening (RADAR), and gold-judge ensembles for bias quantification.

---

**Current status**: LLM-as-a-judge remains the default (85% human agreement on MT-Bench) but has shifted from naive usage to bias-aware engineering — quality-deconfounded metrics (DBG), leakage-aware distillation (DPO > SFT), mechanistic contamination detection (RADAR), and fine-grained rationale evaluation are now standard requirements for credible leaderboards.

**Full reference**: See the companion reference article for 15-source citations, full quantitative tables, and the CONDA 2024 contamination database (432 events across 91 datasets).

---
*Full reference (citations, derivations, variants):* [Judging bias and contamination](../topics/judging-bias-and-contamination.md)
