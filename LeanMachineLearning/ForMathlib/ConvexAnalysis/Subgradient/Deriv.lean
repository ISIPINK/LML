/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Subgradient.Basic
public import Mathlib.Algebra.Group.Pointwise.Set.Basic
public import Mathlib.Analysis.Calculus.LineDeriv.Basic
public import Mathlib.Data.Set.Basic

/-!
# Fréchet Derivatives and Subgradients

This file establishes the connection between Fréchet derivatives (`HasFDerivAt`)
and subgradients (`HasSubgradientWithinAt f g s x`) for convex functions.

## Main results

* `HasFDerivAt.hasSubgradientWithinAt`: A Fréchet derivative of a convex function is a subgradient.
* `HasFDerivAt.eq_of_hasSubgradientWithinAt`: Uniqueness of the subgradient at an interior
  differentiable point.
* `HasFDerivAt.subdifferentialWithin_add`: Subdifferential sum rule when one component is
  Fréchet differentiable.
-/

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
variable [PartialOrder F] [IsOrderedAddMonoid F] [PosSMulMono ℝ F]
variable {f : E → F} {s : Set E} {x : E} {g h : E →L[ℝ] F}

open scoped Bregman Topology Pointwise
open Asymptotics Filter

/-- Scaled Bregman divergence monotonicity along segments for convex functions. -/
lemma _root_.ConvexOn.bregDiv_slope_le (hf : ConvexOn ℝ s f)
    {z : E} (hx : x ∈ s) (hz : z ∈ s) (J : E →L[ℝ] F)
    {t : ℝ} (ht0 : 0 < t) (ht1 : t ≤ 1) :
    t⁻¹ • D_[f](x + t • (z - x), x, J) ≤ D_[f](z, x, J) := by
  have : (1 - t) • x + t • z = x + t • (z - x) := by module
  simpa [this, smul_smul, inv_mul_cancel₀ ht0.ne'] using
    smul_le_smul_of_nonneg_left
      ((hf.bregDiv (y := x) J).2 hx hz (sub_nonneg.mpr ht1) ht0.le (sub_add_cancel 1 t))
      (inv_nonneg.mpr ht0.le)

omit [PartialOrder F] [IsOrderedAddMonoid F] [PosSMulMono ℝ F] in
/-- As the step size `t → 0⁺`, the linearization error of a Fréchet differentiable function
scaled by `t⁻¹` converges to `0`. -/
lemma _root_.HasFDerivAt.tendsto_bregDiv_slope_zero
    (hderiv : HasFDerivAt f g x) (w : E) :
    Tendsto (fun t : ℝ ↦ t⁻¹ • D_[f](x + t • w, x, g)) (𝓝[>] 0) (𝓝 0) := by
  have h := (hderiv.hasLineDerivAt w).tendsto_slope_zero_right.sub_const (g w)
  rw [sub_self] at h
  refine h.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t ht0
  simp [bregDiv, smul_sub, ht0.out.ne']

variable [OrderClosedTopology F]

/-- A Fréchet derivative of a convex function is a subgradient. -/
lemma _root_.HasFDerivAt.hasSubgradientWithinAt
    (hderiv : HasFDerivAt f g x) (hf : ConvexOn ℝ s f) (hx : x ∈ s) :
    HasSubgradientWithinAt f g s x := by
  intro z hz
  refine le_of_tendsto (hderiv.tendsto_bregDiv_slope_zero (z - x)) ?_
  filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds (eventually_le_nhds zero_lt_one)]
    with t (ht0 : 0 < t) ht1 using hf.bregDiv_slope_le hx hz g ht0 ht1

section Real

variable {f : E → ℝ} {g h : E →L[ℝ] ℝ}

lemma _root_.HasFDerivAt.le_of_hasSubgradientWithinAt
    (hderiv : HasFDerivAt f g x) (hs : s ∈ 𝓝 x) (hsub : HasSubgradientWithinAt f h s x) (w : E) :
    h w ≤ g w := by
  have h_nhds : (fun t : ℝ ↦ x + t • w) ⁻¹' s ∈ 𝓝 0 :=
    (continuous_const.add (continuous_id'.smul continuous_const)).continuousAt.preimage_mem_nhds
      (by simpa using hs)
  refine ge_of_tendsto (hderiv.hasLineDerivAt w).tendsto_slope_zero_right ?_
  filter_upwards [nhdsWithin_le_nhds h_nhds, self_mem_nhdsWithin] with t ht_s (ht_pos : 0 < t)
  simpa [bregDiv, add_sub_cancel_left, h.map_smul, smul_eq_mul,
    inv_mul_cancel_left₀ ht_pos.ne'] using
    mul_le_mul_of_nonneg_left (sub_nonneg.mp (hsub (x + t • w) ht_s)) (inv_nonneg.mpr ht_pos.le)

/-- Uniqueness of the subgradient at an interior differentiable point. -/
lemma _root_.HasFDerivAt.eq_of_hasSubgradientWithinAt
    (hderiv : HasFDerivAt f g x) (hs : s ∈ 𝓝 x) (hsub : HasSubgradientWithinAt f h s x) :
    h = g := by
  ext v
  exact le_antisymm (hderiv.le_of_hasSubgradientWithinAt hs hsub v)
    (by simpa using hderiv.le_of_hasSubgradientWithinAt hs hsub (-v))

/-- The subdifferential of a convex, differentiable function at an interior point
is the singleton containing its derivative. -/
lemma _root_.HasFDerivAt.subdifferentialWithin_eq
    (hderiv : HasFDerivAt f g x) (hf : ConvexOn ℝ s f) (hs : s ∈ 𝓝 x) :
    ∂[s, x](f) = {g} := by
  ext h
  simp only [mem_subdifferentialWithin, Set.mem_singleton_iff]
  exact ⟨fun hsub ↦ hderiv.eq_of_hasSubgradientWithinAt hs hsub,
    fun h_eq ↦ h_eq ▸ hderiv.hasSubgradientWithinAt hf (mem_of_mem_nhds hs)⟩

/-- The subdifferential of a constant function at an interior point is `{0}`
(the normal cone to `s` at an interior point is trivial). -/
lemma subdifferentialWithin_const_eq (c : ℝ) (hs_conv : Convex ℝ s) (hs : s ∈ 𝓝 x) :
    ∂[s, x](fun _ : E ↦ c) = {(0 : E →L[ℝ] ℝ)} :=
  (hasFDerivAt_const c x).subdifferentialWithin_eq (convexOn_const c hs_conv) hs

/--
Subdifferential sum rule when `f₁` is convex and Fréchet differentiable at `x`
and `f₂` is convex on `s`: `g` is a subgradient of `f₁ + f₂` at `x` if and only if
`g - g₁` is a subgradient of `f₂` at `x`.
-/
lemma _root_.HasFDerivAt.hasSubgradientWithinAt_add_iff {f₁ f₂ : E → ℝ}
    {g₁ g : E →L[ℝ] ℝ}
    (hderiv₁ : HasFDerivAt f₁ g₁ x) (hf₁ : ConvexOn ℝ s f₁) (hf₂ : ConvexOn ℝ s f₂) (hx : x ∈ s) :
    HasSubgradientWithinAt (f₁ + f₂) g s x ↔ HasSubgradientWithinAt f₂ (g - g₁) s x := by
  constructor
  · intro hg z hz
    refine le_of_tendsto (by simpa using (hderiv₁.tendsto_bregDiv_slope_zero (z - x)).neg) ?_
    filter_upwards [self_mem_nhdsWithin, nhdsWithin_le_nhds (eventually_le_nhds zero_lt_one)]
      with t (ht0 : 0 < t) ht1
    have h_slope := hf₂.bregDiv_slope_le hx hz (g - g₁) ht0 ht1
    have h_nonneg := smul_nonneg (inv_nonneg.mpr ht0.le)
      (hg _ (hf₂.1.add_smul_sub_mem hx hz ⟨ht0.le, ht1⟩))
    have h_eq : g = g₁ + (g - g₁) := by ext; simp
    rw [h_eq, bregDiv_add, smul_add] at h_nonneg
    dsimp at *
    linarith
  · intro h₂
    simpa using (hderiv₁.hasSubgradientWithinAt hf₁ hx).add h₂

/-- Subdifferential sum rule in set form: `∂[s, x](f₁ + f₂) = {g₁} + ∂[s, x](f₂)`. -/
lemma _root_.HasFDerivAt.subdifferentialWithin_add {f₁ f₂ : E → ℝ}
    {g₁ : E →L[ℝ] ℝ} (hderiv₁ : HasFDerivAt f₁ g₁ x) (hf₁ : ConvexOn ℝ s f₁)
    (hf₂ : ConvexOn ℝ s f₂) (hx : x ∈ s) :
    ∂[s, x](f₁ + f₂) = {g₁} + ∂[s, x](f₂) := by
  ext g
  simp [mem_subdifferentialWithin, Set.singleton_add,
    hderiv₁.hasSubgradientWithinAt_add_iff hf₁ hf₂ hx, add_comm g₁, sub_eq_add_neg]

/-- When `f` is convex and Fréchet differentiable at `x ∈ s`, its subdifferential
is `g + ∂[s, x](0)` (the derivative plus the normal cone to `s` at `x`). -/
lemma _root_.HasFDerivAt.subdifferentialWithin_eq_add_zero
    (hderiv : HasFDerivAt f g x) (hf : ConvexOn ℝ s f) (hx : x ∈ s) :
    ∂[s, x](f) = {g} + ∂[s, x](fun _ : E ↦ (0 : ℝ)) := by
  rw [← add_zero f]
  exact hderiv.subdifferentialWithin_add hf (convexOn_const (0 : ℝ) hf.1) hx

end Real

end Analysis.Convex
