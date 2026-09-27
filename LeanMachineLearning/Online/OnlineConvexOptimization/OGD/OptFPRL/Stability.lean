/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.OptFPRL.RegretDecomposition
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.Common.Regularizer
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Basic
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Deriv
public import Mathlib.Analysis.InnerProductSpace.Basic

/-!
# Stability Bounds for Optimistic FTRL with History Pruning (OGD specialization)

This file develops the bounding of the first part (I) of the dynamic regret decomposition
of Naram Mhaisen and George Iosifidis, *On the Dynamic Regret of FTRL: Optimism with
History Pruning*. It follows the OGD convention of the library: the per-round potential is
the scaled Euclidean quadratic regularizer `ψ_t = eucSq (σ t)`, and all norms are Mathlib's
ambient norm `‖·‖` (no local/dual norm machinery). The strong-convexity modulus of the
cumulative history `h_{0:t}` is the scalar `cumSigma σ t = ∑_{i=1}^{t} σ_i`.

We bound the stability term
$$\mathrm{stability}_t = h_{0:t}(w_t) - h_{0:t}(w_{t+1}) - ψ_t(w_t)$$
in two ways, producing the two arguments of the `min` of eq. (26):
* `stability_le_two_mul_diam` — Lemma B.2, the diameter/Hölder bound
  `stability_t ≤ 2 R ε_t`;
* `stability_le_half_norm_sq` — Lemma 4.2, the quadratic-gap bound
  `stability_t ≤ (1 / (2 c_{t-1})) ‖g_t - g̃_t‖²`.

The `min` of the two, summed over the horizon, is the "full stability term" delivered by
`sum_stability_le_min`.

## Main definitions
* `cumSigma`
* `gradH`
* `boundHalf`, `boundDiam`

## Main results
* `lemma_B1`
* `variational_inequality_of_isMinOn`
* `quadratic_gap_of_isMinOn`
* `stability_le_half_norm_sq`
* `stability_le_two_mul_diam`
* `stability_le_min`
* `sum_stability_le_min`
-/

open scoped BigOperators Bregman
open Finset

@[expose] public section

namespace Online.OCO.OGD.OptFPRL

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

variable (p : ℕ → E →L[ℝ] ℝ) (w : ℕ → E)

/-! ### Cumulative scaling and the quadratic history gradient -/

/-- The cumulative scaling `c_t = ∑_{i=1}^{t} σ_i`, which is the strong-convexity modulus of
the quadratic cumulative history `h_{0:t}`. -/
def cumSigma (σ : ℕ → ℝ) (t : ℕ) : ℝ :=
  ∑ i ∈ Ico 1 (t + 1), σ i

/-- The Fréchet derivative of the quadratic cumulative history
`h_{0:t}(x) = ∑_{i=1}^{t} (p_i(x) + (σ_i/2)‖x‖²)` at `x`, i.e.
`∑_{i=1}^{t} p_i + c_t ⟨x, ·⟩`. -/
noncomputable def gradH (σ : ℕ → ℝ) (p : ℕ → E →L[ℝ] ℝ) (t : ℕ) (x : E) : E →L[ℝ] ℝ :=
  (∑ i ∈ Ico 1 (t + 1), p i) + (cumSigma σ t) • innerSL ℝ x

/-! ### Bregman divergence of the quadratic cumulative history -/

/-- Bregman divergence of a finite sum: additivity over the index set. -/
private lemma bregDiv_finset_sum {ι : Type*} (s : Finset ι) (f : ι → E → ℝ)
    (J : ι → E →L[ℝ] ℝ) (x y : E) :
    D_[fun z ↦ ∑ i ∈ s, f i z](x, y, ∑ i ∈ s, J i) =
      ∑ i ∈ s, D_[f i](x, y, J i) := by
  classical
  induction s using Finset.induction with
  | empty => simp
  | insert a s ha ih =>
      have hf : (fun z ↦ ∑ i ∈ insert a s, f i z) = f a + fun z ↦ ∑ i ∈ s, f i z := by
        ext z; rw [Finset.sum_insert ha]; rfl
      rw [hf, Finset.sum_insert ha, bregDiv_add, ih, Finset.sum_insert ha]

/-- The gradient of the quadratic cumulative history as a sum of per-round gradients. -/
private lemma gradH_eq_sum (σ : ℕ → ℝ) (p : ℕ → E →L[ℝ] ℝ) (t : ℕ) (x : E) :
    gradH σ p t x = ∑ i ∈ Ico 1 (t + 1), (p i + eucSqFDeriv (σ i) x) := by
  simp only [gradH, cumSigma, eucSqFDeriv, Finset.sum_add_distrib, ← Finset.sum_smul]

/-- **Definition 4.1 instantiated.** The Bregman divergence of the quadratic cumulative
history is `(c_t / 2) ‖y - x‖²`:
$$D_{h_{0:t}}(y, x, \nabla h_{0:t}(x)) = \frac{c_t}{2} \|y - x\|^2.$$
This exhibits `h_{0:t}` as `c_t`-strongly convex with respect to the ambient norm, which is
the specialization of the paper's `1`-strong convexity with respect to `‖·‖_t`. -/
lemma bregDiv_H_obj_eucSq (σ : ℕ → ℝ) (p : ℕ → E →L[ℝ] ℝ) (t : ℕ) (x y : E) :
    D_[H_obj p (fun i ↦ eucSq (σ i)) t](y, x, gradH σ p t x) =
      (cumSigma σ t / 2) * ‖y - x‖ ^ 2 := by
  rw [show H_obj p (fun i ↦ eucSq (σ i)) t =
      (fun z ↦ ∑ i ∈ Ico 1 (t + 1), (p i z + eucSq (σ i) z)) from rfl]
  rw [gradH_eq_sum]
  rw [bregDiv_finset_sum]
  have hterm : ∀ i ∈ Ico 1 (t + 1),
      D_[fun z ↦ p i z + eucSq (σ i) z](y, x, p i + eucSqFDeriv (σ i) x) =
        (σ i / 2) * ‖y - x‖ ^ 2 := by
    intro i _
    have hfun : (fun z ↦ p i z + eucSq (σ i) z) = (p i : E → ℝ) + eucSq (σ i) := rfl
    rw [hfun, bregDiv_add, bregDiv_linear, zero_add, bregDiv_eucSq_eq]
  rw [Finset.sum_congr rfl hterm, ← Finset.sum_mul, ← Finset.sum_div]
  rfl

/-! ### Splitting the cumulative history -/

/-- One-round extension of the cumulative history. -/
private lemma H_obj_succ (p : ℕ → E →L[ℝ] ℝ) (ψ : ℕ → E → ℝ) (t : ℕ) (z : E) :
    H_obj p ψ (t + 1) z = H_obj p ψ t z + (p (t + 1) z + ψ (t + 1) z) := by
  rw [H_obj, H_obj, sum_Ico_succ_top (show 1 ≤ t + 1 by omega)]

/-- Backward split of the cumulative history for `t ≥ 1`. -/
private lemma H_obj_split (p : ℕ → E →L[ℝ] ℝ) (ψ : ℕ → E → ℝ) (t : ℕ) (ht : 1 ≤ t) (z : E) :
    H_obj p ψ t z = H_obj p ψ (t - 1) z + (p t z + ψ t z) := by
  conv_lhs => rw [← Nat.sub_add_cancel ht]
  rw [H_obj_succ, Nat.sub_add_cancel ht]

/-- The quadratic cumulative history as a linear term plus a single quadratic. -/
private lemma H_obj_eucSq_eq (p : ℕ → E →L[ℝ] ℝ) (σ : ℕ → ℝ) (n : ℕ) (z : E) :
    H_obj p (fun i ↦ eucSq (σ i)) n z =
      (∑ i ∈ Ico 1 (n + 1), p i) z + eucSq (cumSigma σ n) z := by
  rw [H_obj, Finset.sum_add_distrib]
  congr 1
  · rw [_root_.sum_apply]
  · simp only [eucSq]
    rw [← Finset.sum_mul, ← Finset.sum_div]
    rfl

/-- The history up to round `t - 1` as a linear term plus the quadratic `eucSq`. -/
private lemma H_obj_pred_eq (p : ℕ → E →L[ℝ] ℝ) (σ : ℕ → ℝ) (t : ℕ) (ht : 1 ≤ t) (z : E) :
    H_obj p (fun i ↦ eucSq (σ i)) (t - 1) z =
      (∑ i ∈ Ico 1 t, p i) z + eucSq (cumSigma σ (t - 1)) z := by
  rw [H_obj_eucSq_eq, Nat.sub_add_cancel ht]

/-! ### Lemma B.1: optimality characterization of the iterate -/

/-- **Lemma B.1.** If `w t` minimizes the optimistic objective `h_{0:t-1} + f̃_t` on `s`, and
`g̃_t` is the Fréchet derivative of `f̃_t` at `w t`, then `w t` also minimizes the linearized
objective `h_{0:t-1} + ⟨g̃_t, ·⟩ + ⟨g_t^I, ·⟩`, where `g_t^I` is selected according to
eq. (19): either `g_t^I = 0` (case (i)) or `g_t^I = -(∇h_{0:t-1}(w t) + g̃_t)` (case (ii)). -/
theorem lemma_B1
    (σ : ℕ → ℝ) (f_tilde : ℕ → E → ℝ) (s : Set E) (t : ℕ)
    (g_tilde gI : ℕ → E →L[ℝ] ℝ)
    (hmin : IsMinOn (F_obj p (fun i ↦ eucSq (σ i)) f_tilde (t - 1)) s (w t))
    (hderiv : HasFDerivAt (f_tilde t) (g_tilde t) (w t))
    (hgI : gI t = 0 ∨ gI t = -(gradH σ p (t - 1) (w t) + g_tilde t)) :
    IsMinOn
      (fun y ↦ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) y + g_tilde t y + gI t y)
      s (w t) := by
  sorry

/-! ### The constrained variational inequality and the quadratic gap -/

/-- **Constrained variational inequality.** A minimizer `x` of a convex, differentiable
function `φ` on a convex set `s` satisfies `0 ≤ ∇φ(x)(y - x)` for every `y ∈ s`. This is the
only non-algebraic ingredient of the strong-convexity-free form of Lemma 4.2. -/
lemma variational_inequality_of_isMinOn
    {φ : E → ℝ} {J : E →L[ℝ] ℝ} {s : Set E} {x : E}
    (hconv : ConvexOn ℝ s φ) (hderiv : HasFDerivAt φ J x) (hx : x ∈ s)
    (hmin : IsMinOn φ s x) (y : E) (hy : y ∈ s) :
    0 ≤ J (y - x) := by
  have h_tend := (hderiv.hasLineDerivAt (y - x)).tendsto_slope_zero_right
  refine ge_of_tendsto h_tend ?_
  filter_upwards [self_mem_nhdsWithin,
    nhdsWithin_le_nhds (eventually_le_nhds (show (0 : ℝ) < 1 by norm_num))]
    with t (ht0 : 0 < t) (ht1 : t ≤ 1)
  have hmem : x + t • (y - x) ∈ s :=
    hconv.1.add_smul_sub_mem hx hy ⟨ht0.le, ht1⟩
  have h_nonneg : 0 ≤ φ (x + t • (y - x)) - φ x := sub_nonneg.mpr (hmin hmem)
  exact smul_nonneg (inv_nonneg.mpr ht0.le) h_nonneg

/-- **Quadratic gap.** If `x` minimizes the `c`-strongly convex quadratic
`z ↦ (c/2)‖z‖² + L z` on `s`, then for every `y ∈ s`
$$\frac{c}{2} \|y - x\|^2 \le \phi(y) - \phi(x).$$
Proved by expanding `φ(y) - φ(x) = (c/2)‖y - x‖² + ∇φ(x)(y - x)` and applying the
constrained variational inequality. -/
lemma quadratic_gap_of_isMinOn
    (c : ℝ) (L : E →L[ℝ] ℝ) (s : Set E) (x y : E)
    (hc : 0 < c) (hs : Convex ℝ s)
    (hmin : IsMinOn (fun z ↦ eucSq c z + L z) s x) (hx : x ∈ s) (hy : y ∈ s) :
    eucSq c (y - x) ≤ (eucSq c y + L y) - (eucSq c x + L x) := by
  have hderiv : HasFDerivAt (fun z ↦ eucSq c z + L z) (eucSqFDeriv c x + L) x := by
    have h := (hasFDerivAt_eucSq c x).add L.hasFDerivAt
    convert h using 1
  have hconv : ConvexOn ℝ s (fun z ↦ eucSq c z + L z) := by
    have h1 : ConvexOn ℝ s (eucSq c) := convexOn_eucSq c hc.le s hs
    have h2 : ConvexOn ℝ s (fun z ↦ L z) := L.toLinearMap.convexOn hs
    have h := h1.add h2
    convert h using 1
  have hvar : 0 ≤ (eucSqFDeriv c x + L) (y - x) :=
    variational_inequality_of_isMinOn hconv hderiv hx hmin y hy
  have hdiff : eucSq c y - eucSq c x = eucSq c (y - x) + eucSqFDeriv c x (y - x) := by
    have h1 : D_[eucSq c](y, x, eucSqFDeriv c x) = eucSq c (y - x) := by
      rw [bregDiv_eucSq_eq, eucSq_eq]
    have h2 : eucSq c y - eucSq c x - eucSqFDeriv c x (y - x) = eucSq c (y - x) := by
      simpa [bregDiv] using h1
    linarith
  calc eucSq c (y - x)
      ≤ eucSq c (y - x) + (eucSqFDeriv c x + L) (y - x) := le_add_of_nonneg_right hvar
    _ = (eucSq c y + L y) - (eucSq c x + L x) := by
        rw [add_apply]
        have h1 : (eucSq c y + L y) - (eucSq c x + L x)
            = (eucSq c y - eucSq c x) + (L y - L x) := by ring
        rw [h1, ← map_sub L y x, hdiff]
        ring

/-! ### Lemma 4.2: the quadratic-gap stability bound -/

/-- **Lemma 4.2** (strong-convexity-free form, OGD specialization). With
`ψ = fun i ↦ eucSq (σ i)`, if `w t` minimizes `h_{0:t-1} + f̃_t` and hence (by Lemma B.1)
the linearized objective `h_{0:t-1} + ⟨g̃_t, ·⟩ + ⟨g_t^I, ·⟩`, and if `y` minimizes
`h_{0:t-1} + ⟨p_t, ·⟩` on `s`, then
$$\mathrm{stability}_t \le \frac{1}{2 c_{t-1}} \|g_t - \tilde g_t\|^2.$$ -/
theorem stability_le_half_norm_sq
    (σ : ℕ → ℝ) (t : ℕ) (s : Set E) (y : E) (g g_tilde gI : ℕ → E →L[ℝ] ℝ)
    (hs : Convex ℝ s) (hσ : 0 < cumSigma σ (t - 1)) (hσ_t : 0 ≤ σ t)
    (hw_t : w t ∈ s) (hw_next : w (t + 1) ∈ s) (hy : y ∈ s)
    (hmin₁ : IsMinOn
      (fun z ↦ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) z + g_tilde t z + gI t z) s (w t))
    (hmin₂ : IsMinOn
      (fun z ↦ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) z + p t z) s y)
    (hgI : gI t = p t - g t) :
    stability p (fun i ↦ eucSq (σ i)) w t ≤
      (1 / (2 * cumSigma σ (t - 1))) * ‖g t - g_tilde t‖ ^ 2 := by
  have ht : 1 ≤ t := by
    by_contra h
    have ht0 : t = 0 := by omega
    subst t
    simp [cumSigma] at hσ
  set A : E →L[ℝ] ℝ := ∑ i ∈ Ico 1 t, p i with hA
  set L1 : E →L[ℝ] ℝ := A + g_tilde t + gI t with hL1
  set L2 : E →L[ℝ] ℝ := A + p t with hL2
  set Δ : E →L[ℝ] ℝ := g t - g_tilde t with hΔ
  have hfun1 : (fun z ↦ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) z + g_tilde t z + gI t z)
      = (fun z ↦ eucSq (cumSigma σ (t - 1)) z + L1 z) := by
    ext z
    rw [H_obj_pred_eq p σ t ht z, ← hA, hL1]
    simp only [add_apply]
    ring
  have hfun2 : (fun z ↦ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) z + p t z)
      = (fun z ↦ eucSq (cumSigma σ (t - 1)) z + L2 z) := by
    ext z
    rw [H_obj_pred_eq p σ t ht z, ← hA, hL2]
    simp only [add_apply]
    ring
  rw [hfun1] at hmin₁
  rw [hfun2] at hmin₂
  have hdrop : 0 ≤ eucSq (σ t) (w (t + 1)) := by
    rw [eucSq_eq]; positivity
  have hstab : stability p (fun i ↦ eucSq (σ i)) w t ≤
      (eucSq (cumSigma σ (t - 1)) (w t) + L2 (w t))
        - (eucSq (cumSigma σ (t - 1)) (w (t + 1)) + L2 (w (t + 1))) := by
    rw [stability]
    rw [H_obj_split p (fun i ↦ eucSq (σ i)) t ht (w t),
        H_obj_split p (fun i ↦ eucSq (σ i)) t ht (w (t + 1))]
    rw [H_obj_pred_eq p σ t ht (w t), H_obj_pred_eq p σ t ht (w (t + 1))]
    rw [← hA, hL2]
    simp only [add_apply]
    linarith [hdrop]
  have hstep2 : (eucSq (cumSigma σ (t - 1)) (w t) + L2 (w t))
        - (eucSq (cumSigma σ (t - 1)) (w (t + 1)) + L2 (w (t + 1)))
      ≤ (eucSq (cumSigma σ (t - 1)) (w t) + L2 (w t))
        - (eucSq (cumSigma σ (t - 1)) y + L2 y) := by
    have h : eucSq (cumSigma σ (t - 1)) y + L2 y
        ≤ eucSq (cumSigma σ (t - 1)) (w (t + 1)) + L2 (w (t + 1)) := hmin₂ hw_next
    linarith
  have hgap := quadratic_gap_of_isMinOn (cumSigma σ (t - 1)) L1 s (w t) y hσ hs hmin₁ hw_t hy
  have hgap_rev : (eucSq (cumSigma σ (t - 1)) (w t) + L1 (w t))
        - (eucSq (cumSigma σ (t - 1)) y + L1 y)
      ≤ - eucSq (cumSigma σ (t - 1)) (w t - y) := by
    have heven : eucSq (cumSigma σ (t - 1)) (y - w t) = eucSq (cumSigma σ (t - 1)) (w t - y) := by
      rw [eucSq_eq, eucSq_eq, norm_sub_rev y (w t)]
    rw [heven] at hgap
    linarith [hgap]
  have hL12 : L2 = L1 + Δ := by
    rw [hL1, hL2, hΔ, hgI]
    ext v
    simp only [add_apply, sub_apply]
    abel
  have hphi2 : (eucSq (cumSigma σ (t - 1)) (w t) + L2 (w t))
        - (eucSq (cumSigma σ (t - 1)) y + L2 y)
      = ((eucSq (cumSigma σ (t - 1)) (w t) + L1 (w t))
        - (eucSq (cumSigma σ (t - 1)) y + L1 y)) + Δ (w t - y) := by
    rw [hL12]
    simp only [add_apply, map_sub]
    ring
  have hstep3 : (eucSq (cumSigma σ (t - 1)) (w t) + L2 (w t))
        - (eucSq (cumSigma σ (t - 1)) y + L2 y)
      ≤ -(cumSigma σ (t - 1) / 2) * ‖w t - y‖ ^ 2 + ‖Δ‖ * ‖w t - y‖ := by
    rw [hphi2]
    have hop : Δ (w t - y) ≤ ‖Δ‖ * ‖w t - y‖ :=
      le_trans (le_abs_self _) (Δ.le_opNorm (w t - y))
    have he : eucSq (cumSigma σ (t - 1)) (w t - y) =
        (cumSigma σ (t - 1) / 2) * ‖w t - y‖ ^ 2 := eucSq_eq _ _
    linarith [hgap_rev, hop, he]
  have hkey : -(cumSigma σ (t - 1) / 2) * ‖w t - y‖ ^ 2 + ‖Δ‖ * ‖w t - y‖
      ≤ ‖Δ‖ ^ 2 / (2 * cumSigma σ (t - 1)) := by
    have hc2 : 0 < 2 * cumSigma σ (t - 1) := by linarith
    rw [le_div_iff₀ hc2]
    nlinarith [sq_nonneg (cumSigma σ (t - 1) * ‖w t - y‖ - ‖Δ‖)]
  calc stability p (fun i ↦ eucSq (σ i)) w t
      ≤ (eucSq (cumSigma σ (t - 1)) (w t) + L2 (w t))
        - (eucSq (cumSigma σ (t - 1)) (w (t + 1)) + L2 (w (t + 1))) := hstab
    _ ≤ (eucSq (cumSigma σ (t - 1)) (w t) + L2 (w t))
        - (eucSq (cumSigma σ (t - 1)) y + L2 y) := hstep2
    _ ≤ -(cumSigma σ (t - 1) / 2) * ‖w t - y‖ ^ 2 + ‖Δ‖ * ‖w t - y‖ := hstep3
    _ ≤ ‖Δ‖ ^ 2 / (2 * cumSigma σ (t - 1)) := hkey
    _ = (1 / (2 * cumSigma σ (t - 1))) * ‖g t - g_tilde t‖ ^ 2 := by
        rw [hΔ]; ring

/-! ### Lemma B.2: the diameter stability bound -/

/-- **Lemma B.2** (OGD specialization). With `ψ = fun i ↦ eucSq (σ i)`, under the same
optimality setup as `stability_le_half_norm_sq`, and assuming the auxiliary minimizer `y`
satisfies the diameter bound `‖w t - y‖ ≤ 2R`, we have
$$\mathrm{stability}_t \le 2 R \|g_t - \tilde g_t\|.$$ -/
theorem stability_le_two_mul_diam
    (σ : ℕ → ℝ) (t : ℕ) (s : Set E) (y : E) (R : ℝ) (g g_tilde gI : ℕ → E →L[ℝ] ℝ)
    (ht : 1 ≤ t) (hσ_t : 0 ≤ σ t)
    (hw_next : w (t + 1) ∈ s) (hy : y ∈ s)
    (hdist : ‖w t - y‖ ≤ 2 * R)
    (hmin₁ : IsMinOn
      (fun z ↦ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) z + g_tilde t z + gI t z) s (w t))
    (hmin₂ : IsMinOn
      (fun z ↦ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) z + p t z) s y)
    (hgI : gI t = p t - g t) :
    stability p (fun i ↦ eucSq (σ i)) w t ≤ 2 * R * ‖g t - g_tilde t‖ := by
  have hdrop : 0 ≤ eucSq (σ t) (w (t + 1)) := by
    rw [eucSq_eq]; positivity
  have hstab : stability p (fun i ↦ eucSq (σ i)) w t ≤
      (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + p t (w t))
        - (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w (t + 1)) + p t (w (t + 1))) := by
    rw [stability]
    rw [H_obj_split p (fun i ↦ eucSq (σ i)) t ht (w t),
        H_obj_split p (fun i ↦ eucSq (σ i)) t ht (w (t + 1))]
    linarith [hdrop]
  have hstep2 : (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + p t (w t))
        - (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w (t + 1)) + p t (w (t + 1)))
      ≤ (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + p t (w t))
        - (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) y + p t y) := by
    have h : H_obj p (fun i ↦ eucSq (σ i)) (t - 1) y + p t y
        ≤ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w (t + 1)) + p t (w (t + 1)) :=
      hmin₂ hw_next
    linarith
  have hΔeq : (p t - g_tilde t - gI t) = g t - g_tilde t := by
    rw [hgI]
    ext v
    simp only [sub_apply]
    abel
  have hdecomp : (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + p t (w t))
        - (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) y + p t y)
      = ((H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + g_tilde t (w t) + gI t (w t))
        - (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) y + g_tilde t y + gI t y))
        + (g t - g_tilde t) (w t - y) := by
    rw [← hΔeq]
    simp only [sub_apply, map_sub]
    ring
  have hstep3 : (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + p t (w t))
        - (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) y + p t y)
      ≤ (g t - g_tilde t) (w t - y) := by
    rw [hdecomp]
    have hB1 : H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + g_tilde t (w t) + gI t (w t)
        ≤ H_obj p (fun i ↦ eucSq (σ i)) (t - 1) y + g_tilde t y + gI t y := hmin₁ hy
    linarith
  have hfinal : (g t - g_tilde t) (w t - y) ≤ 2 * R * ‖g t - g_tilde t‖ := by
    have h1 : (g t - g_tilde t) (w t - y) ≤ ‖g t - g_tilde t‖ * ‖w t - y‖ :=
      le_trans (le_abs_self _) ((g t - g_tilde t).le_opNorm (w t - y))
    have h2 : ‖g t - g_tilde t‖ * ‖w t - y‖ ≤ ‖g t - g_tilde t‖ * (2 * R) :=
      mul_le_mul_of_nonneg_left hdist (norm_nonneg _)
    have h3 : ‖g t - g_tilde t‖ * (2 * R) = 2 * R * ‖g t - g_tilde t‖ := by ring
    linarith
  calc stability p (fun i ↦ eucSq (σ i)) w t
      ≤ (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + p t (w t))
        - (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w (t + 1)) + p t (w (t + 1))) := hstab
    _ ≤ (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) (w t) + p t (w t))
        - (H_obj p (fun i ↦ eucSq (σ i)) (t - 1) y + p t y) := hstep2
    _ ≤ (g t - g_tilde t) (w t - y) := hstep3
    _ ≤ 2 * R * ‖g t - g_tilde t‖ := hfinal

/-! ### Combining the two bounds: eq. (26) and the full stability term -/

/-- Per-round first bound, `2 R ε_t` (Lemma B.2). -/
noncomputable def boundDiam (R : ℕ → ℝ) (g g_tilde : ℕ → E →L[ℝ] ℝ) (t : ℕ) : ℝ :=
  2 * R t * ‖g t - g_tilde t‖

/-- Per-round second bound, `(1 / (2 c_{t-1})) ‖g_t - g̃_t‖²` (Lemma 4.2). -/
noncomputable def boundHalf (σ : ℕ → ℝ) (g g_tilde : ℕ → E →L[ℝ] ℝ) (t : ℕ) : ℝ :=
  (1 / (2 * cumSigma σ (t - 1))) * ‖g t - g_tilde t‖ ^ 2

/-- **Eq. (26):** the stability term is bounded by the minimum of the two per-round bounds. -/
theorem stability_le_min
    (σ : ℕ → ℝ) (R : ℕ → ℝ) (t : ℕ) (g g_tilde : ℕ → E →L[ℝ] ℝ)
    (hHalf : stability p (fun i ↦ eucSq (σ i)) w t ≤ boundHalf σ g g_tilde t)
    (hDiam : stability p (fun i ↦ eucSq (σ i)) w t ≤ boundDiam R g g_tilde t) :
    stability p (fun i ↦ eucSq (σ i)) w t ≤
      min (boundHalf σ g g_tilde t) (boundDiam R g g_tilde t) :=
  le_min hHalf hDiam

/-- **Full stability term.** Summing eq. (26) over the horizon `t ∈ Ico 1 (T+1)` gives the
complete bound on the first part (I) of the dynamic regret decomposition. -/
theorem sum_stability_le_min
    (σ : ℕ → ℝ) (R : ℕ → ℝ) (T : ℕ) (g g_tilde : ℕ → E →L[ℝ] ℝ)
    (h : ∀ t ∈ Ico 1 (T + 1),
      stability p (fun i ↦ eucSq (σ i)) w t ≤
        min (boundHalf σ g g_tilde t) (boundDiam R g g_tilde t)) :
    ∑ t ∈ Ico 1 (T + 1), stability p (fun i ↦ eucSq (σ i)) w t ≤
      ∑ t ∈ Ico 1 (T + 1), min (boundHalf σ g g_tilde t) (boundDiam R g g_tilde t) :=
  sum_le_sum h

end Online.OCO.OGD.OptFPRL
