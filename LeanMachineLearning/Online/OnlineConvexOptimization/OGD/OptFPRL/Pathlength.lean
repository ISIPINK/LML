/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.OptFPRL.Stability
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Basic
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Deriv
public import Mathlib.Analysis.InnerProductSpace.LinearMap

/-!
# Optimistically Bounded State and Pathlength Bounds for OGD OptFPRL

This file proves Lemma 4.3 of Mhaisen and Iosifidis, *On the Dynamic Regret of FTRL:
Optimism with History Pruning*, in the OGD specialization, together with the pathlength
estimates built on it.

The (pruned) state at round `t` is the cumulative linearization
$$p_{1:t} = \sum_{i=1}^{t} p_i, \qquad p_i = g_i + g_i^{I},$$
where `g_i` is a subgradient of the loss and `g_i^I` is the normal-cone correction chosen
by the OptFPRL pruning rule. Both pruning cases (whether the unconstrained iterate lies in
the feasible set or not) force the same closed form
$$p_{1:t} = (g_t - \tilde g_t) - \nabla\psi_{1:t-1}(w_t),$$
so that the state bound is `‖p_{1:t}‖ ≤ R σ_{1:t-1} + ε_t` with `ε_t = ‖g_t - \tilde g_t‖`.

## Main results
* `norm_eucSqFDeriv`: `‖∇ψ_α w‖ = |α| ‖w‖`.
* `state_bound`: Lemma 4.3, `‖p_{1:t}‖ ≤ R · cumSigma σ (t-1) + ‖g_t - g̃_t‖`.
* `pathlength_le_of_state`: `pathlength_t ≤ ‖p_{1:t-1}‖ ‖u_t - u_{t-1}‖`.
* `sum_pathlength_le`: cumulative pathlength bound.
-/

open scoped BigOperators Bregman RealInnerProductSpace
open Finset

@[expose] public section

namespace Online.OCO.OGD.OptFPRL

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ### Norm of the regularizer gradient -/

/-- Operator norm of the Euclidean regularizer gradient: `‖∇ψ_α w‖ = |α| ‖w‖`. -/
lemma norm_eucSqFDeriv (α : ℝ) (w : E) :
    ‖eucSqFDeriv α w‖ = |α| * ‖w‖ := by
  rw [eucSqFDeriv, norm_smul, innerSL_apply_norm, Real.norm_eq_abs]

/-! ### The `t - 1` form of the history gradient -/

/-- For `t ≥ 1`, `gradH σ p (t-1)` is the derivative of the history `h_{0:t-1}`. -/
private lemma gradH_pred (p : ℕ → E →L[ℝ] ℝ) (σ : ℕ → ℝ) (w : ℕ → E) (t : ℕ) (ht : 1 ≤ t) :
    gradH σ p (t - 1) (w t) =
      (∑ i ∈ Ico 1 t, p i) + (cumSigma σ (t - 1)) • innerSL ℝ (w t) := by
  rw [gradH, Nat.sub_add_cancel ht]

/-! ### Closed form of the state under the two pruning cases -/

/-- Under the pruning choice `g_t^I = -(∇h_{0:t-1}(w_t) + g̃_t)`, the state collapses to
`(g_t - g̃_t) - ∇ψ_{1:t-1}(w_t)`. -/
private lemma state_eq_of_pruned (p : ℕ → E →L[ℝ] ℝ) (σ : ℕ → ℝ)
    (g g_tilde gI : ℕ → E →L[ℝ] ℝ) (w : ℕ → E) (t : ℕ) (ht : 1 ≤ t)
    (hp : p t = g t + gI t)
    (hgI : gI t = -(gradH σ p (t - 1) (w t) + g_tilde t)) :
    (∑ i ∈ Ico 1 (t + 1), p i)
      = (g t - g_tilde t) - eucSqFDeriv (cumSigma σ (t - 1)) (w t) := by
  have hgrad := gradH_pred p σ w t ht
  rw [sum_Ico_succ_top ht, hp, hgI, hgrad]
  ext v
  simp only [eucSqFDeriv_apply, add_apply, sub_apply, neg_apply, smul_apply,
    innerSL_apply_apply]
  ring

/-- Under no pruning (`g_t^I = 0`), the unconstrained optimality condition
`∇h_{0:t-1}(w_t) + g̃_t = 0` (Fermat, via subgradient uniqueness) gives the same closed
form of the state. -/
private lemma state_eq_of_unconstrained (p : ℕ → E →L[ℝ] ℝ) (σ : ℕ → ℝ)
    (f_tilde : ℕ → E → ℝ) (g g_tilde gI : ℕ → E →L[ℝ] ℝ) (w : ℕ → E) (t : ℕ) (ht : 1 ≤ t)
    (hp : p t = g t + gI t) (hgI0 : gI t = 0)
    (hmin : IsMinOn
      (fun y ↦ (∑ i ∈ Ico 1 t, p i) y + eucSq (cumSigma σ (t - 1)) y + f_tilde t y)
      Set.univ (w t))
    (hderiv : HasFDerivAt (f_tilde t) (g_tilde t) (w t)) :
    (∑ i ∈ Ico 1 (t + 1), p i)
      = (g t - g_tilde t) - eucSqFDeriv (cumSigma σ (t - 1)) (w t) := by
  have hderiv₁ : HasFDerivAt
      (fun y : E ↦ (∑ i ∈ Ico 1 t, p i) y + eucSq (cumSigma σ (t - 1)) y)
      ((∑ i ∈ Ico 1 t, p i) + eucSqFDeriv (cumSigma σ (t - 1)) (w t)) (w t) := by
    have h1 : HasFDerivAt (fun y : E ↦ (∑ i ∈ Ico 1 t, p i) y)
        ((∑ i ∈ Ico 1 t, p i) : E →L[ℝ] ℝ) (w t) :=
      ContinuousLinearMap.hasFDerivAt (𝕜 := ℝ) (f := (∑ i ∈ Ico 1 t, p i))
    exact h1.add (hasFDerivAt_eucSq (cumSigma σ (t - 1)) (w t))
  have hsub : HasSubgradientWithinAt
      (fun y : E ↦ (∑ i ∈ Ico 1 t, p i) y + eucSq (cumSigma σ (t - 1)) y + f_tilde t y)
      (0 : E →L[ℝ] ℝ) Set.univ (w t) :=
    hasSubgradientWithinAt_zero_iff_isMinOn.mpr hmin
  have hzero_raw : (∑ i ∈ Ico 1 t, p i) + eucSqFDeriv (cumSigma σ (t - 1)) (w t)
      + g_tilde t = 0 :=
    ((hderiv₁.add hderiv).eq_of_hasSubgradientWithinAt (s := Set.univ) (by simp) hsub).symm
  have hzero : gradH σ p (t - 1) (w t) + g_tilde t = 0 := by
    simpa [gradH_pred p σ w t ht, eucSqFDeriv] using hzero_raw
  have hgI : gI t = -(gradH σ p (t - 1) (w t) + g_tilde t) := by
    rw [hgI0, hzero, neg_zero]
  exact state_eq_of_pruned p σ g g_tilde gI w t ht hp hgI

/-! ### Lemma 4.3: optimistically bounded state -/

/-- **Lemma 4.3 (Optimistically Bounded State).** If the pruning rule holds at round `t`
(either no pruning with unconstrained optimality, or the pruning choice), `‖w_t‖ ≤ R`, and
`cumSigma σ (t-1) ≥ 0`, then
$$\|p_{1:t}\| \le R\,\sigma_{1:t-1} + \|g_t - \tilde g_t\|.$$ -/
theorem state_bound (p : ℕ → E →L[ℝ] ℝ) (σ : ℕ → ℝ) (f_tilde : ℕ → E → ℝ)
    (g g_tilde gI : ℕ → E →L[ℝ] ℝ) (w : ℕ → E) (R : ℝ) (t : ℕ) (ht : 1 ≤ t)
    (hp : p t = g t + gI t)
    (hcase : (gI t = 0 ∧ IsMinOn
        (fun y ↦ (∑ i ∈ Ico 1 t, p i) y + eucSq (cumSigma σ (t - 1)) y + f_tilde t y)
        Set.univ (w t))
      ∨ gI t = -(gradH σ p (t - 1) (w t) + g_tilde t))
    (hderiv : HasFDerivAt (f_tilde t) (g_tilde t) (w t))
    (hR : ‖w t‖ ≤ R) (hσ : 0 ≤ cumSigma σ (t - 1)) :
    ‖∑ i ∈ Ico 1 (t + 1), p i‖ ≤ cumSigma σ (t - 1) * R + ‖g t - g_tilde t‖ := by
  have hstate : (∑ i ∈ Ico 1 (t + 1), p i)
      = (g t - g_tilde t) - eucSqFDeriv (cumSigma σ (t - 1)) (w t) := by
    rcases hcase with ⟨hgI0, hmin⟩ | hgI
    · exact state_eq_of_unconstrained p σ f_tilde g g_tilde gI w t ht hp hgI0 hmin hderiv
    · exact state_eq_of_pruned p σ g g_tilde gI w t ht hp hgI
  calc ‖∑ i ∈ Ico 1 (t + 1), p i‖
      = ‖(g t - g_tilde t) - eucSqFDeriv (cumSigma σ (t - 1)) (w t)‖ := by rw [hstate]
    _ ≤ ‖g t - g_tilde t‖ + ‖eucSqFDeriv (cumSigma σ (t - 1)) (w t)‖ := norm_sub_le _ _
    _ = ‖g t - g_tilde t‖ + |cumSigma σ (t - 1)| * ‖w t‖ := by rw [norm_eucSqFDeriv]
    _ = ‖g t - g_tilde t‖ + cumSigma σ (t - 1) * ‖w t‖ := by rw [abs_of_nonneg hσ]
    _ ≤ ‖g t - g_tilde t‖ + cumSigma σ (t - 1) * R :=
        add_le_add le_rfl (mul_le_mul_of_nonneg_left hR hσ)
    _ = cumSigma σ (t - 1) * R + ‖g t - g_tilde t‖ := by ring

/-! ### Pathlength estimates -/

/-- Per-round pathlength bound from a state bound. -/
lemma pathlength_le_of_state (p : ℕ → E →L[ℝ] ℝ) (u : ℕ → E) (t : ℕ) (C : ℝ)
    (hC : ‖∑ i ∈ Ico 1 t, p i‖ ≤ C) :
    pathlength p u t ≤ C * ‖u t - u (t - 1)‖ := by
  have h := (∑ i ∈ Ico 1 t, p i).le_opNorm (u t - u (t - 1))
  rw [Real.norm_eq_abs] at h
  calc pathlength p u t
      = (∑ i ∈ Ico 1 t, p i) (u t - u (t - 1)) := rfl
    _ ≤ |(∑ i ∈ Ico 1 t, p i) (u t - u (t - 1))| := le_abs_self _
    _ ≤ ‖∑ i ∈ Ico 1 t, p i‖ * ‖u t - u (t - 1)‖ := h
    _ ≤ C * ‖u t - u (t - 1)‖ := mul_le_mul_of_nonneg_right hC (norm_nonneg _)

/-- Cumulative pathlength bound from per-round state bounds. -/
theorem sum_pathlength_le (p : ℕ → E →L[ℝ] ℝ) (u : ℕ → E) (C : ℕ → ℝ) (T : ℕ)
    (hC : ∀ t ∈ Ico 1 (T + 1), ‖∑ i ∈ Ico 1 t, p i‖ ≤ C t) :
    (∑ t ∈ Ico 1 (T + 1), pathlength p u t)
      ≤ ∑ t ∈ Ico 1 (T + 1), C t * ‖u t - u (t - 1)‖ :=
  sum_le_sum fun t ht ↦ pathlength_le_of_state p u t (C t) (hC t ht)

/-- Cumulative pathlength bound with the Lemma 4.3 state estimate
`C t = R · cumSigma σ (t-2) + ‖g_{t-1} - g̃_{t-1}‖`, obtained by applying `state_bound` at
round `t - 1`. -/
theorem sum_pathlength_le_state (p : ℕ → E →L[ℝ] ℝ) (σ : ℕ → ℝ) (g g_tilde : ℕ → E →L[ℝ] ℝ)
    (u : ℕ → E) (R : ℝ) (T : ℕ)
    (hstate : ∀ t ∈ Ico 1 (T + 1),
      ‖∑ i ∈ Ico 1 t, p i‖ ≤ cumSigma σ (t - 2) * R + ‖g (t - 1) - g_tilde (t - 1)‖) :
    (∑ t ∈ Ico 1 (T + 1), pathlength p u t)
      ≤ ∑ t ∈ Ico 1 (T + 1),
          (cumSigma σ (t - 2) * R + ‖g (t - 1) - g_tilde (t - 1)‖) * ‖u t - u (t - 1)‖ :=
  sum_pathlength_le p u
    (fun t ↦ cumSigma σ (t - 2) * R + ‖g (t - 1) - g_tilde (t - 1)‖) T hstate

end Online.OCO.OGD.OptFPRL
