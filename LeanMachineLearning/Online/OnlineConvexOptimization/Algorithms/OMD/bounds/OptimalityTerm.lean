/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.Online.OnlineConvexOptimization.Algorithms.OMD.Definition
public import LeanMachineLearning.Online.OnlineConvexOptimization.Algorithms.OMD.StrongLemmas
public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Subgradient.Basic
public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Subgradient.Deriv
public import Mathlib.Analysis.Calculus.FDeriv.Basic
public import Mathlib.Analysis.Calculus.LineDeriv.Basic

/-!
# Non-positivity of the OMD Optimality Term

This file proves that the first-order optimality term appearing in Online Mirror Descent:
- `optimalityTerm`: anchor update optimality deficit

is **non-positive** (i.e. $\le 0$, so subtracting it adds a non-negative contribution $\ge 0$)
when the iterate $w_{t+1}$ minimizes the step objective over a convex constraint
set $X \subseteq E$, and the comparator $u_t \in X$.

## Method: Converse Subdifferential Sum Rule

When $\psi_{t+1}$ is Fréchet differentiable at $w_{t+1}$ and both $\varphi_t, \psi_{t+1}$ are
convex, the converse sum rule proves that
$$-(g\psi_{t+1}(w_{t+1}) + \eta g_t - g\psi_t(\tilde{w}_t)) \in \partial[X, w_{t+1}] \varphi_t.$$

## Main results

* Subgradient inclusion: `Analysis.Convex.subgradient_phi_of_isMinOn` and
  `Analysis.Convex.subgradient_phi_of_isMinOn_round`.
* Optimality vanishing: `Analysis.Convex.optimalityTerm_eq_zero_of_canonical_subgradient`.
* Non-negativity: `Analysis.Convex.optimalityTerm_nonneg_of_canonical_subgradient`.
-/

open scoped Bregman Topology
open Filter

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

variable (η : ℝ)
variable (ψ φ : ℕ → E → ℝ)
variable (gψ gφ : ℕ → E → (E →L[ℝ] ℝ))
variable (u w w_tilde : ℕ → E)
variable (g : ℕ → (E →L[ℝ] ℝ))

/-- **Master Subgradient from Composite Minimization**:
If $w \in X$ minimizes $(\psi + \text{lin}) + \varphi$ on $X$, $\psi$ has Fréchet derivative
$g\psi_L$ at $w$, and both $\psi$ and $\varphi$ are convex on $X$, then
$-(g\psi_L + \text{lin}) \in \partial[X, w] \varphi$. -/
lemma subdifferential_of_isMinOn_fderiv_add {X : Set E} {w : E}
    {ψ φ : E → ℝ} {gψ_L lin : E →L[ℝ] ℝ}
    (hψ : HasFDerivAt ψ gψ_L w) (hψ_conv : ConvexOn ℝ X ψ) (hφ_conv : ConvexOn ℝ X φ)
    (hw : w ∈ X) (h_min : IsMinOn ((ψ + ⇑lin) + φ) X w) :
    HasSubgradientWithinAt φ (-(gψ_L + lin)) X w := by
  have h_min_sub : HasSubgradientWithinAt ((ψ + ⇑lin) + φ) (0 : E →L[ℝ] ℝ) X w :=
    hasSubgradientWithinAt_zero_iff_isMinOn.mpr h_min
  have h_conv : ConvexOn ℝ X (ψ + ⇑lin) := hψ_conv.add (lin.toLinearMap.convexOn hψ_conv.1)
  have h_fderiv : HasFDerivAt (ψ + ⇑lin) (gψ_L + lin) w := hψ.add lin.hasFDerivAt
  have := (h_fderiv.hasSubgradientWithinAt_add_iff h_conv hφ_conv hw).mp h_min_sub
  rwa [zero_sub] at this

/-- **Subgradient from Converse Sum Rule ($\varphi$ Nonsmooth)**: When $\psi$ is Fréchet
differentiable at $w$ and both $\varphi$ and $\psi$ are convex on $X$,
the converse sum rule proves that
$$-(g\psi_L(w) + \eta g - g\psi_{\text{anchor}}) \in \partial[X, w] \varphi.$$ -/
lemma subgradient_phi_of_isMinOn {X : Set E} {w : E}
    {η : ℝ} {g gψ_anchor : E →L[ℝ] ℝ} {φ ψ : E → ℝ} {gψ_L : E →L[ℝ] ℝ}
    (hψ : HasFDerivAt ψ gψ_L w)
    (hψ_conv : ConvexOn ℝ X ψ)
    (hφ_conv : ConvexOn ℝ X φ)
    (hw : w ∈ X)
    (h_min : IsMinOn (fun x ↦ η * g x + φ x + ψ x - gψ_anchor x) X w) :
    HasSubgradientWithinAt φ (-gψ_L - η • g + gψ_anchor) X w := by
  let lin : E →L[ℝ] ℝ := η • g - gψ_anchor
  have h_min' : IsMinOn ((ψ + ⇑lin) + φ) X w := by
    rwa [show (ψ + ⇑lin) + φ = (fun x ↦ η * g x + φ x + ψ x - gψ_anchor x) by
      ext x; simp [lin, sub_eq_add_neg, add_comm, add_left_comm, add_assoc]]
  have h_sub := subdifferential_of_isMinOn_fderiv_add hψ hψ_conv hφ_conv hw h_min'
  have h_eq : -(gψ_L + lin) = -gψ_L - η • g + gψ_anchor := by
    ext x
    simp [lin, sub_eq_add_neg, add_comm, add_left_comm]
  rwa [h_eq] at h_sub

/-- Round-$t$ specialization of the subgradient rule: when $w_{t+1} \in X$ minimizes the
step objective on $X$, the canonical subgradient belongs to $\partial[X, w_{t+1}] \varphi_t$. -/
lemma subgradient_phi_of_isMinOn_round (t : ℕ) {X : Set E}
    {gψ_L : E →L[ℝ] ℝ}
    (hψ : HasFDerivAt (ψ (t + 1)) gψ_L (w (t + 1)))
    (hψ_conv : ConvexOn ℝ X (ψ (t + 1)))
    (hφ_conv : ConvexOn ℝ X (φ t))
    (hw : w (t + 1) ∈ X)
    (h_min : IsMinOn (fun x ↦ η * (g t) x + φ t x + ψ (t + 1) x - (gψ t (w_tilde t)) x)
      X (w (t + 1))) :
    HasSubgradientWithinAt (φ t) (-gψ_L - η • (g t) + gψ t (w_tilde t)) X (w (t + 1)) :=
  subgradient_phi_of_isMinOn hψ hψ_conv hφ_conv hw h_min

/-- Choosing the canonical subgradient for $\varphi_t$
makes the anchor optimality deficit vanish identically ($\mathrm{optimalityTerm}_t = 0$). -/
lemma optimalityTerm_eq_zero_of_canonical_subgradient (t : ℕ)
    (hgφ : gφ t (w (t + 1)) =
      -gψ (t + 1) (w (t + 1)) - η • (g t) + gψ t (w_tilde t) +
        gψ (t + 1) (w_tilde 1) - gψ t (w_tilde 1)) :
    optimalityTerm η gψ gφ u w w_tilde g t = 0 := by
  dsimp [optimalityTerm]
  have h_zero : (η • (g t) + gφ t (w (t + 1)) + gψ (t + 1) (w (t + 1)) - gψ t (w_tilde t)
      - gψ (t + 1) (w_tilde 1) + gψ t (w_tilde 1)) = 0 := by
    ext x
    rw [hgφ]
    simp only [add_apply, sub_apply, neg_apply, smul_apply, smul_eq_mul, zero_apply]
    ring
  rw [h_zero]
  simp only [zero_apply]

/-- Under the canonical subgradient choice for $\varphi_t$,
$0 \le \mathrm{optimalityTerm}_t$ holds identically. -/
lemma optimalityTerm_nonneg_of_canonical_subgradient (t : ℕ)
    (hgφ : gφ t (w (t + 1)) =
      -gψ (t + 1) (w (t + 1)) - η • (g t) + gψ t (w_tilde t) +
        gψ (t + 1) (w_tilde 1) - gψ t (w_tilde 1)) :
    0 ≤ optimalityTerm η gψ gφ u w w_tilde g t := by
  rw [optimalityTerm_eq_zero_of_canonical_subgradient η gψ gφ u w w_tilde g t hgφ]

end Analysis.Convex
