---
id: arxiv:2209.00708
type: paper
title: 'ColossalAI: A Practical Distributed Training Framework for Large-Scale Deep
  Learning Systems'
url: https://arxiv.org/abs/2209.00708
retrieved: '2026-07-11'
maturity: comprehensive
topic: distributed-rl-training
---

The study addresses the characterization of low-latitude ionospheric scintillation during low solar activity conditions, specifically evaluating the reliability of the NavIC (Navigation with Indian Constellation) system alongside GPS near the Equatorial Ionization Anomaly (EIA) and magnetic equator. Ionospheric scintillation, driven by electron density irregularities and Equatorial Plasma Bubbles, disrupts radio signals between 100 MHz and 4 GHz, impacting navigation and communication. This research provides the first long-term analysis of such phenomena over the Indian longitude sector during the declining phase of solar cycle 24.

The methodology follows a systematic pipeline. First, 25 months of raw GNSS data (September 2017 to September 2019) were collected using a Septentrio PolaRxS Pro receiver for GPS (L1, L2, L5) and an ACCORD NavIC receiver for NavIC (L5, S1) at Indore and Hyderabad. Observables including $C/N_0$, TEC, and loss-of-lock (LoL) were extracted at 1 Hz and synchronized. Scintillation events were isolated using strict criteria: a 20° masking angle, a $S_{4c}$ threshold exceeding 0.3 for at least 30 seconds, exclusion of events within five minutes of prior occurrences, and evaluation of LoL durations exceeding 120 seconds. From 607 observation days, only 27 days met these criteria. Signal intensity $S_I$ was derived from $C/N_0$, followed by calculation of the normalized variance $S_4$, ambient noise correction $S_{AN}$, and the final corrected index $S_{4_C}$. Ionospheric disturbance proxies, the Rate of TEC (ROT) and ROT Index (ROTI), were computed from slant TEC gradients. Finally, amplitude scintillation statistics were modeled using Nakagami-m and $\alpha-\mu$ distributions fitted to empirical fading samples.

The mathematical framework relies on the following key formulations. Signal intensity and the standard scintillation index are defined as:

$$
S_I = 10^{0.1 \cdot C/N_0}, \quad S_4 = \sqrt{\frac{\langle S_I^2 \rangle - \langle S_I \rangle^2}{\langle S_I \rangle^2}}
$$

Ambient noise correction and the final index are:

$$
S_{AN} = \sqrt{\frac{100}{C/N_0} \left[ 1 + \frac{500}{19C/N_0} \right]}, \quad S_{4_C} = S_4 - S_{AN}
$$

TEC rate metrics and scattering coefficients are calculated via:

$$
ROT = \frac{STEC_{r+1} - STEC_r}{\Delta t_r}, \quad ROTI = \sqrt{\langle ROT^2 \rangle - \langle ROT \rangle^2}, \quad S_{a,b} = \left[ \frac{a - b}{a + b} \right]
$$

Distribution modeling employs the Nakagami-m parameter $m = E^2(r^2)/[E(r^4) - E^2(r^2)]$ and the $\alpha-\mu$ probability density function:

$$
f(r) = \frac{\alpha r^{\alpha \mu - 1}}{\zeta^{\alpha \mu / 2} \Gamma(\mu)} \exp\left(-\frac{r^\alpha}{\zeta^{\alpha / 2}}\right), \quad \zeta = \frac{\Gamma(\mu)}{\Gamma(\mu + 2/\alpha)}
$$

The relationship between $S_4$ and distribution parameters is given by $S_4^2 = [\Gamma(\mu)\Gamma(\mu + 4/\alpha) - \Gamma^2(\mu + 2/\alpha)] / \Gamma^2(\mu + 2/\alpha)$.

Quantitative results indicate that scintillation onset occurred at approximately 19:30 LT in Hyderabad and 20:30 LT in Indore, with peak $S_{4c}$ values between 22:00 and 23:00 LT. Peak $S_{4c}$ values ranged from 0.30 to 2.57 across satellites, while ROT and ROTI peaked at 0.2 and 0.1, respectively. Scattering coefficients between NavIC $L_5$ and $S_1$ signals remained below 0.1, demonstrating strong signal correlation and receiver reliability during quiet periods. Statistical fitting confirmed that amplitude scintillation follows both Nakagami-m and $\alpha-\mu$ distributions, with the parameter $\alpha$ increasing as $S_{4c}$ rises, indicating degraded signal lock retention under severe fading.

The study acknowledges several limitations. The analysis is restricted to two receiver stations (Indore and Hyderabad) due to the limited geographic network of NavIC receivers and a lack of temporally matched data for other locations. Furthermore, only 4% of nights exhibiting scintillation were selected for detailed analysis based on distribution criteria. The authors note that PRN 6 lacked clear ROT and ROTI indicators at both stations, and the underlying cause for stronger signal fluctuations at Indore compared to Hyderabad remains unexplained, necessitating future multi-station comparative studies.
