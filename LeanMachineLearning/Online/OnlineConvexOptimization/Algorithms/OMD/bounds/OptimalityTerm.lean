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
# Non-positivity of the OMD Optimality Terms

This file proves that the first-order optimality terms appearing in Online Mirror Descent:
- `optimalityTerm`: anchor update optimality deficit
- `optimisticOptimalityTerm`: hint prediction optimality deficit

are **non-positive** (i.e. $\le 0$, so subtracting them adds a non-negative contribution $\ge 0$)
when the iterates $w_{t+1}, v_t$ minimize their respective objectives over a convex constraint
set $X \subseteq E$, and the comparator $u_t \in X$.

## Method: Converse Subdifferential Sum Rule

When $\psi_{t+1}$ is Fréchet differentiable at $w_{t+1}$ and both $\varphi_t, \psi_{t+1}$ are
convex, the converse sum rule proves that
$$-(g\psi_{t+1}(w_{t+1}) + \eta g_t - g\psi_t(\tilde{w}_t)) \in \partial[X, w_{t+1}] \varphi_t.$$
-/

open scoped Bregman Topology
open Filter

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

variable (η : ℝ)
variable (ψ φ : ℕ → E → ℝ)
variable (gψ gφ : ℕ → E → (E →L[ℝ] ℝ))
variable (u w w_tilde v : ℕ → E)
variable (g h : ℕ → (E →L[ℝ] ℝ))

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
lemma subgradient_phi_of_isAnchorUpdate {X : Set E} {w : E}
    {η : ℝ} {g gψ_anchor : E →L[ℝ] ℝ} {φ ψ : E → ℝ} {gψ_L : E →L[ℝ] ℝ}
    (hψ : HasFDerivAt ψ gψ_L w)
    (hψ_conv : ConvexOn ℝ X ψ)
    (hφ_conv : ConvexOn ℝ X φ)
    (hw : w ∈ X)
    (h_min : IsAnchorUpdate η g φ ψ gψ_anchor X w) :
    HasSubgradientWithinAt φ (-gψ_L - η • g + gψ_anchor) X w := by
  let lin : E →L[ℝ] ℝ := η • g - gψ_anchor
  have h_min' : IsMinOn ((ψ + ⇑lin) + φ) X w := by
    rwa [show (ψ + ⇑lin) + φ = anchorUpdateObjective η g φ ψ gψ_anchor by
      ext x; simp [anchorUpdateObjective, lin, sub_eq_add_neg, add_comm, add_left_comm, add_assoc]]
  have h_sub := subdifferential_of_isMinOn_fderiv_add hψ hψ_conv hφ_conv hw h_min'
  have h_eq : -(gψ_L + lin) = -gψ_L - η • g + gψ_anchor := by
    ext x
    simp [lin, sub_eq_add_neg, add_comm, add_left_comm]
  rwa [h_eq] at h_sub

/-- **Canonical Subgradient Nullifies Optimality Deficit**: When $g\varphi$ is chosen as the
canonical subgradient from `subgradient_phi_of_isAnchorUpdate`, the optimality deficit is $0$. -/
lemma firstOrderOptimalityTerm_eq_zero_of_canonical_subgradient
    {η : ℝ} {g gψ_anchor gψ_L : E →L[ℝ] ℝ} (u w : E) :
    firstOrderOptimalityTerm η g (-gψ_L - η • g + gψ_anchor) gψ_L gψ_anchor u w = 0 := by
  have h_zero : (η • g + (-gψ_L - η • g + gψ_anchor) + gψ_L - gψ_anchor) = 0 := by
    ext x
    simp only [add_apply, sub_apply, neg_apply, smul_apply, smul_eq_mul, zero_apply]
    ring
  unfold firstOrderOptimalityTerm
  rw [h_zero]
  rfl

/-! ### Non-Optimistic Case: Anchor Update Specialization -/

/-- Round-$t$ specialization of the subgradient rule: when $w_{t+1} \in X$ minimizes the
anchor objective on $X$, the canonical subgradient belongs to $\partial[X, w_{t+1}] \varphi_t$. -/
lemma subgradient_phi_of_isAnchorUpdate_round (t : ℕ) {X : Set E}
    {gψ_L : E →L[ℝ] ℝ}
    (hψ : HasFDerivAt (ψ (t + 1)) gψ_L (w (t + 1)))
    (hψ_conv : ConvexOn ℝ X (ψ (t + 1)))
    (hφ_conv : ConvexOn ℝ X (φ t))
    (hw : w (t + 1) ∈ X)
    (gψ_anchor : E →L[ℝ] ℝ)
    (hgψ_anchor : gψ t (w_tilde t) = gψ_anchor)
    (h_min : IsAnchorUpdate η (g t) (φ t) (ψ (t + 1)) (gψ t (w_tilde t)) X (w (t + 1))) :
    HasSubgradientWithinAt (φ t) (-gψ_L - η • (g t) + gψ_anchor) X (w (t + 1)) :=
  subgradient_phi_of_isAnchorUpdate hψ hψ_conv hφ_conv hw (hgψ_anchor ▸ h_min)

/-- For nonsmooth $\varphi_t$, choosing the canonical subgradient from `IsAnchorUpdate`
makes the anchor optimality deficit vanish identically ($\mathrm{optimalityTerm}_t = 0$). -/
lemma optimalityTerm_eq_zero_of_canonical_subgradient (t : ℕ)
    {gψ_L : E →L[ℝ] ℝ}
    (hgψ : gψ (t + 1) (w (t + 1)) = gψ_L)
    (gψ_anchor : E →L[ℝ] ℝ)
    (hgψ_anchor : gψ t (w_tilde t) = gψ_anchor)
    (hgφ : gφ t (w (t + 1)) = (-gψ_L - η • (g t) + gψ_anchor)) :
    optimalityTerm η gψ gφ u w w_tilde (fun n ↦ g n) t = 0 := by
  dsimp [optimalityTerm]
  rw [hgφ, hgψ, hgψ_anchor, firstOrderOptimalityTerm_eq_zero_of_canonical_subgradient]

/-- Under the anchor update on constraint set $X$, choosing the canonical subgradient
yields $0 \le \mathrm{optimalityTerm}_t$ for any comparator $u_t \in X$. -/
lemma optimalityTerm_nonneg_of_isAnchorUpdate (t : ℕ) {X : Set E}
    (_hu : u t ∈ X) (hw : w (t + 1) ∈ X)
    {gψ_L : E →L[ℝ] ℝ}
    (hψ : HasFDerivAt (ψ (t + 1)) gψ_L (w (t + 1)))
    (hψ_conv : ConvexOn ℝ X (ψ (t + 1)))
    (hφ_conv : ConvexOn ℝ X (φ t))
    (hgψ : gψ (t + 1) (w (t + 1)) = gψ_L)
    (gψ_anchor : E →L[ℝ] ℝ)
    (hgψ_anchor : gψ t (w_tilde t) = gψ_anchor)
    (hgφ : gφ t (w (t + 1)) = (-gψ_L - η • (g t) + gψ_anchor))
    (h_min : IsAnchorUpdate η (g t) (φ t) (ψ (t + 1)) (gψ t (w_tilde t)) X (w (t + 1))) :
    0 ≤ optimalityTerm η gψ gφ u w w_tilde (fun n ↦ g n) t := by
  have _ := subgradient_phi_of_isAnchorUpdate_round η ψ φ gψ w w_tilde g t hψ hψ_conv
    hφ_conv hw gψ_anchor hgψ_anchor h_min
  rw [optimalityTerm_eq_zero_of_canonical_subgradient η gψ gφ u w w_tilde g t hgψ gψ_anchor
    hgψ_anchor hgφ]

/-! ### Optimistic Case: Hint Prediction Specialization -/

/-- **Subgradient for Optimistic Prediction Step**: Special case of the subgradient rule
for `IsOptimisticPrediction` where centering potential $\varphi = 0$. -/
lemma subgradient_zero_of_isOptimisticPrediction {X : Set E} {v : E}
    {η : ℝ} {h gψ_anchor : E →L[ℝ] ℝ} {ψ : E → ℝ} {gψ_L : E →L[ℝ] ℝ}
    (hψ : HasFDerivAt ψ gψ_L v)
    (hψ_conv : ConvexOn ℝ X ψ)
    (hv : v ∈ X)
    (h_min : IsOptimisticPrediction η h ψ gψ_anchor X v) :
    HasSubgradientWithinAt (0 : E → ℝ) (-gψ_L - η • h + gψ_anchor) X v := by
  rw [isOptimisticPrediction_iff_isAnchorUpdate] at h_min
  exact subgradient_phi_of_isAnchorUpdate hψ hψ_conv (convexOn_const 0 hψ_conv.1) hv h_min

/-- When $g\psi_{t+1}(v_t) = -\eta h_t + g\psi_t(\tilde{w}_t)$, the optimistic prediction
optimality deficit vanishes identically ($\mathrm{optimisticOptimalityTerm}_t = 0$). -/
lemma optimisticOptimalityTerm_eq_zero_of_canonical_gradient (t : ℕ)
    (gψ_anchor : E →L[ℝ] ℝ)
    (hgψ_anchor : gψ t (w_tilde t) = gψ_anchor)
    (hgψ : gψ (t + 1) (v t) = (-η • (h t) + gψ_anchor)) :
    optimisticOptimalityTerm η gψ w w_tilde v (fun n ↦ h n) t = 0 := by
  have h_zero : (η • (h t) + 0 + gψ (t + 1) (v t) - gψ t (w_tilde t)) = 0 := by
    ext x
    rw [hgψ, hgψ_anchor]
    simp only [sub_apply, add_apply, zero_apply, smul_apply, smul_eq_mul]
    ring
  unfold optimisticOptimalityTerm firstOrderOptimalityTerm
  rw [h_zero]
  rfl

/-- When $\psi_{t+1}$ is Fréchet differentiable at $v_t$ and convex on $X$,
any prediction iterate $v_t$ minimizing the hint objective yields
$0 \le \mathrm{optimisticOptimalityTerm}_t$. -/
lemma optimisticOptimalityTerm_nonneg_of_isOptimisticPrediction (t : ℕ) {X : Set E}
    (hw : w (t + 1) ∈ X) (hv : v t ∈ X)
    {gψ_L : E →L[ℝ] ℝ}
    (hψ : HasFDerivAt (ψ (t + 1)) gψ_L (v t))
    (hψ_conv : ConvexOn ℝ X (ψ (t + 1)))
    (hgψ : gψ (t + 1) (v t) = gψ_L)
    (gψ_anchor : E →L[ℝ] ℝ)
    (hgψ_anchor : gψ t (w_tilde t) = gψ_anchor)
    (h_min : IsOptimisticPrediction η (h t) (ψ (t + 1)) (gψ t (w_tilde t)) X (v t)) :
    0 ≤ optimisticOptimalityTerm η gψ w w_tilde v (fun n ↦ h n) t := by
  have h_sub := subgradient_zero_of_isOptimisticPrediction hψ hψ_conv hv (hgψ_anchor ▸ h_min)
  have h_eval := h_sub (w (t + 1)) hw
  dsimp [optimisticOptimalityTerm]
  rw [hgψ, hgψ_anchor]
  have h_eq : firstOrderOptimalityTerm η (h t) 0 gψ_L gψ_anchor (w (t + 1)) (v t) =
      D_[(0 : E → ℝ)](w (t + 1), v t, -gψ_L - η • (h t) + gψ_anchor) := by
    dsimp [firstOrderOptimalityTerm, bregDiv]
    simp only [sub_zero, zero_sub, add_apply, sub_apply, neg_apply, smul_apply, smul_eq_mul,
      zero_apply]
    ring
  rw [h_eq]
  exact h_eval

end Analysis.Convex
