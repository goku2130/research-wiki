---
id: arxiv:1909.08560
type: paper
title: Improving Language Models by Recovering from Mistakes (Ziegler et al., 2019)
url: https://arxiv.org/abs/1909.08560
retrieved: '2026-07-11'
maturity: comprehensive
topic: rejection-sampling-and-bon
---

**Core Problem**
Quadrupedal locomotion control traditionally relies on heuristic or reduced-order models (e.g., massless legs, linear inverted pendulum, planar approximations) to bypass the computational intractability of full-order dynamics. While bipedal robots benefit from rigorous, rapid full-order gait generation frameworks like Hybrid Zero Dynamics (HZD), these methods have not been efficiently scaled to quadrupeds due to high-dimensional state spaces and complex contact interactions. The core problem addressed is how to systematically decompose the full-order dynamics of a quadrupedal robot into lower-dimensional bipedal subsystems to enable rapid, mathematically rigorous gait generation without empirical model simplification.

**Method and Recipe Step-by-Step**
The methodology follows a structured decomposition-composition pipeline:
1. **Hybrid Dynamics Formulation:** The continuous-time dynamics are modeled as differential-algebraic equations (DAEs) enforcing holonomic toe-contact constraints, while discrete impact dynamics are derived via conservation of momentum at contact transitions.
2. **Full-Body Decomposition:** The 3D quadruped dynamics are mathematically split into two identical 3D bipedal subsystems (front and rear). Each subsystem is subject to external connection wrenches that preserve the original body linkage constraint, ensuring dynamical equivalence.
3. **Control Decomposition:** Virtual constraints are imposed on bipedal outputs, and input-output feedback linearization is applied to define closed-loop dynamics. A correlation matrix and fifth-order Bezier polynomial trajectory parameters enforce symmetry between the front and rear bipeds.
4. **NLP Gait Optimization:** A nonlinear programming problem is solved using direct Hermite-Simpson collocation (via the FROST toolbox) to generate periodic, hybrid-invariant gaits for a single biped, minimizing torso vibration while satisfying dynamics, impact, periodicity, and physical feasibility constraints.
5. **Recomposition & Tracking:** The optimized bipedal trajectories are recomposed into quadrupedal gaits using the predefined correlation. Tracking is implemented via a PD approximation of the input-output linearizing controller in simulation and hardware.

**Key Formulas**
The continuous dynamics under dual toe contact are:
$$D(q) \ddot{q} + H(q, \dot{q}) = B u + J_1^\top \lambda_1 + J_2^\top \lambda_2, \quad J_\star \ddot{q} + \dot{J}_\star \dot{q} = 0$$
Impact dynamics at contact transitions follow:
$$D (\dot{q}^+ - \dot{q}^-) = J_0^\top \Lambda_0 + J_3^\top \Lambda_3, \quad J_0 \dot{q}^+ = 0, \ J_3 \dot{q}^+ = 0$$
The decomposed front biped closed-loop dynamics incorporate virtual constraints $\ddot{y}_f = k_1 \dot{y}_f + k_2 y_f \equiv 0$ and connection wrenches $\lambda_c$. Biped symmetry is enforced via $\mathcal{B}_r(t) = M \mathcal{B}_f(t) + b$ and $q_r \equiv A q_f + b$. The gait optimization solves:
$$\min_{\mathbf{Z}} \sum_{j=1}^{2N+1} \|\dot{q}_{b_j}\|_2^2 \quad \text{s.t. dynamics, collocation, impact, periodicity, and feasibility constraints.}$$

**Quantitative Results**
Validated on the Vision 60 V3.2 quadruped (18 DOF, 12 control inputs, 36 state variables), the method generated four stepping-in-place gaits and one diagonally symmetric ambling gait at 0.35 m/s. Average computation time was 3.96 seconds (0.039 seconds per iteration), yielding an order-of-magnitude speedup over full-model optimization methods that require minutes to hours. Stepping gaits achieved frequencies of 2.2–2.6 Hz with foot clearances of 11–15 cm. Experimental torque averages remained within hardware limits: hip roll (4.16–5.14 N·m), hip pitch (3.77–5.26 N·m), and knee (16.45–18.36 N·m). The composed gaits demonstrated dynamic stability and robustness to rough outdoor terrain using a unified open-loop PD controller.

**Stated Limitations**
The authors note that while the method eliminates empirical model reduction, real-time on-board gait generation currently requires further code optimization for deployment. The tracking controller relies on a time-based PD approximation, which is inherently open-loop and typically lacks robustness to uncertain terrain dynamics, though the full-order gait generation compensated for this limitation. Additionally, the decomposition framework currently assumes specific gait symmetries via predefined correlation matrices, and adapting the method to asymmetric or highly complex locomotion patterns requires careful constraint reformulation. The impact dynamics formulation also yields an overdetermined system that must be carefully solved to avoid ill-posedness, particularly for robots with lightweight appendages.
