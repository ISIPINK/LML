/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Subgradient.Basic
public import Mathlib.Analysis.Calculus.LineDeriv.Basic

/-!
# Fréchet Derivatives and Subgradients

This file establishes the connection between Fréchet derivatives (`HasFDerivAt`)
and subgradients (`∂[V, x] f`) for convex functions.

## Main results

* `Analysis.Convex.subgradient_of_hasFDerivAt`: A Fréchet derivative of a convex function
  is a subgradient.
* `Analysis.Convex.le_fderiv_of_mem_subdifferential`: An interior subgradient is bounded
  above by the Fréchet derivative.
* `Analysis.Convex.eq_fderiv_of_mem_subdifferential`: Uniqueness of subgradient at interior
  differentiable points.
-/

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
variable [PartialOrder F] [IsOrderedAddMonoid F] [PosSMulMono ℝ F]

open scoped Bregman
open Asymptotics

local instance : Coe (E →L[ℝ] F) (E →+ F) where
  coe g := g.toAddMonoidHom

local instance : Membership (E →L[ℝ] F) (Set (E →+ F)) where
  mem s g := (g : E →+ F) ∈ s


omit [PosSMulMono ℝ F] in
/-- For a convex function, the Bregman divergence at an interpolated point `x + t • (z - x)`
is bounded by `t • D_[f](z, x, J)`. -/
lemma ConvexOn.bregman_segment_le {f : E → F} {V : Set E} (h_conv : ConvexOn ℝ V f)
    {x z : E} (hx : x ∈ V) (hz : z ∈ V) (J : E →ₗ[ℝ] F)
    {t : ℝ} (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    D_[f](x + t • (z - x), x, J) ≤ t • D_[f](z, x, J) := by
  have h_comb : (1 - t) • x + t • z = x + t • (z - x) := by module
  simpa [h_comb] using
    (h_conv.bregDiv (y := x) J).2 hx hz (sub_nonneg.mpr ht1) ht0 (sub_add_cancel 1 t)

variable [OrderClosedTopology F]

/--
For a convex function `f : E → F`, if `f` has a Fréchet derivative at `x` represented by the
continuous linear map `g`, then `g` is a subgradient of `f` at `x` over the set `V`.
-/
lemma subgradient_of_hasFDerivAt {f : E → F} {V : Set E} {x : E} {g : E →L[ℝ] F}
    (h_conv : ConvexOn ℝ V f) (hx : x ∈ V)
    (h_deriv : HasFDerivAt f g x) :
    g ∈ ∂[V, x] f := by
  refine ⟨hx, fun z hz ↦ sub_nonneg.mpr <|
    le_of_tendsto (h_deriv.hasLineDerivAt (z - x)).tendsto_slope_zero_right ?_⟩
  filter_upwards [self_mem_nhdsWithin,
    Filter.mem_inf_of_left (eventually_le_nhds zero_lt_one)] with t (ht0 : 0 < t) ht1
  simpa [bregDiv, smul_smul, inv_mul_cancel₀ ht0.ne'] using
    smul_le_smul_of_nonneg_left
      (ConvexOn.bregman_segment_le h_conv hx hz 0 ht0.le ht1) (inv_nonneg.mpr ht0.le)


/--
If `f : E → ℝ` has Fréchet derivative `g` at an interior point `x` of `V`,
then any subgradient `h ∈ ∂[V, x] f` satisfies `h w ≤ g w` for all directions `w`.
-/
lemma le_fderiv_of_mem_subdifferential {f : E → ℝ} {V : Set E} {x : E} {g h : E →L[ℝ] ℝ}
    (hV : V ∈ nhds x) (h_deriv : HasFDerivAt f g x) (h_sub : h ∈ ∂[V, x] f) (w : E) :
    h w ≤ g w := by
  have h_nhds : (fun t : ℝ ↦ x + t • w) ⁻¹' V ∈ nhds 0 :=
    (continuous_const.add (continuous_id'.smul continuous_const)).continuousAt.preimage_mem_nhds
      (by simpa using hV)
  refine ge_of_tendsto (h_deriv.hasLineDerivAt w).tendsto_slope_zero_right ?_
  filter_upwards [nhdsWithin_le_nhds h_nhds, self_mem_nhdsWithin] with t ht_V (ht_pos : 0 < t)
  simpa [bregDiv, add_sub_cancel_left, h.map_smul, smul_eq_mul,
    inv_mul_cancel_left₀ ht_pos.ne'] using
    mul_le_mul_of_nonneg_left (sub_nonneg.mp (h_sub.2 (x + t • w) ht_V)) (inv_nonneg.mpr ht_pos.le)

/--
If a function `f : E → ℝ` is Fréchet differentiable at an interior point `x` of `V` with derivative
`g`, then any continuous linear subgradient `h ∈ ∂[V, x] f` must be equal to `g`.
-/
lemma eq_fderiv_of_mem_subdifferential {f : E → ℝ} {V : Set E} {x : E} {g h : E →L[ℝ] ℝ}
    (hV : V ∈ nhds x) (h_deriv : HasFDerivAt f g x) (h_sub : h ∈ ∂[V, x] f) :
    h = g := by
  ext v
  exact le_antisymm (le_fderiv_of_mem_subdifferential hV h_deriv h_sub v)
    (by simpa using le_fderiv_of_mem_subdifferential hV h_deriv h_sub (-v))

end Analysis.Convex
