---
id: arxiv:2306.05673
type: paper
title: Monte Carlo Tree Search for Text Generation
url: https://arxiv.org/abs/2306.05673
retrieved: '2026-07-11'
maturity: comprehensive
topic: test-time-and-rl-interplay
---

**Core Problem**
Large-scale liquid xenon time projection chambers (TPCs) require internal calibration for low-energy electronic recoil (ER) responses, as external gamma sources cannot penetrate the meter-scale fiducial volumes. A gaseous $^{220}$Rn source derived from $^{228}$Th is introduced into the xenon flow, where its decay progeny $^{212}$Pb generates a uniform beta spectrum below 200 keV. This calibrates the dominant $^{214}$Pb ER background below 10 keV. The source must simultaneously achieve high $^{220}$Rn emanation, maintain extreme radiopurity, and prevent the release of long-lived contaminants like $^{228}$Th or excessive $^{222}$Rn.

**Methodology**
The characterization followed a sequential experimental protocol:
1. **Source Fabrication:** Four commercial $^{228}$Th-platinum discs (~13.8 kBq each) were prepared with a 5 mm electroplated oxide active area. A gold cover was omitted to maximize radon release. The discs were mounted facing each other within a stainless steel CF-50 vessel.
2. **$^{220}$Rn Emanation Measurement:** The assembly was placed inside a 4 L electrostatic radon monitor filled with nitrogen at 1050 mbar. A −1 kV bias collected positively charged radon progeny onto a Si-PIN diode for alpha spectroscopy over four days. Due to the 145 ms half-life of $^{216}$Po, its detected rate was suppressed relative to equilibrium; thus, the calculation relied exclusively on the $^{212}$Po equilibrium rate. An initial activity measurement required a time-delay correction factor, $f_{\text{eq}}$, to account for the 11-hour buildup of $^{212}$Pb. Because a $^{220}$Rn reference source was unavailable, the detection efficiency was estimated by scaling the measured $^{222}$Rn efficiency ($35 \pm 2\%$) by the $^{212}$Bi branching fraction, yielding $\epsilon(^{220}\text{Rn} \mid ^{212}\text{Po}) = (22.4 \pm 1.3)\%$, with an additional 1.4% uncertainty for field-free regions.
3. **$^{222}$Rn Emanation Measurement:** The source vessel was flushed with helium and left for radon accumulation. The gas was transferred to an evacuated 20 L expansion vessel, where it rested for 4.5 hours to allow $^{220}$Rn to decay. The remaining $^{222}$Rn was cryogenically trapped on activated carbon and counted in a miniaturized proportional counter calibrated against a $^{226}$Ra standard.
4. **$^{228}$Th Release Verification:** The source was purged with argon at 700 SCCM for nine days. Downstream 0.2 μm PTFE filters captured non-gaseous components. The filters were analyzed using underground HPGe spectrometers to detect $^{212}$Pb and $^{208}$Tl gamma emissions, confirming the absence of thorium contamination.

**Key Formulas and Quantitative Results**
The $^{220}$Rn emanation rate is determined via:
$$R(^{220}\text{Rn}) = \frac{1}{\epsilon(^{220}\text{Rn} \mid ^{212}\text{Po})} \cdot \frac{A_{\text{init}}(^{212}\text{Po})}{f_{\text{eq}}(^{212}\text{Po})}$$
The characterization yielded a total $^{228}$Th activity of 55 kBq, a $^{220}$Rn emanation rate of $(8.2 \pm 0.8)$ kBq, and a radon emanation efficiency of 15%. The $^{222}$Rn emanation rate was measured at $(3.62 \pm 0.14)$ mBq. The $^{228}$Th release test established an upper limit of $\le 1.7$ mBq at 90% confidence level.

**Stated Limitations**
The unexpected $^{222}$Rn emanation is attributed to trace $^{226}$Ra impurities (~350 ppm) introduced during manufacturing. While this level represents only ~10% of XENONnT’s total radon background and becomes subdominant approximately one week post-calibration due to radioactive decay and the detector’s active radon removal system, it may exceed the stringent background requirements of next-generation experiments like DARWIN/XLZD and nEXO. Furthermore, the reliance on an indirect efficiency estimation and the suppression of the $^{216}$Po signal due to its short half-life relative to the ion collection time introduce systematic dependencies that necessitate careful time-window selection and analytical correction.
