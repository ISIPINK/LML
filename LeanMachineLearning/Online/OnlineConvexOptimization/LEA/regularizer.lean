/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Bregman.Basic
public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Bregman.MVT
public import Mathlib.Analysis.Calculus.Deriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Mul
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Unnormalized Negative Entropy and its Bregman Divergence

This file defines the unnormalized negative entropy regularizer
`ψ(x) = ∑ x_i * ln(x_i) - x_i`, its Fréchet derivative `∇ψ(x) = (ln(x_i))_i`,
its Hessian second-derivative bilinear map, and establishes its second-order
Taylor / MVT remainder formula.

## Main definitions

* `Analysis.Convex.unnormEntropy`: Unnormalized negative entropy
  `ψ(x) = ∑_i (x_i * ln(x_i) - x_i)`.
* `Analysis.Convex.unnormEntropyFDeriv`: Fréchet derivative mapping of `unnormEntropy`
  as a continuous linear functional `EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ`.
* `Analysis.Convex.unnormEntropyHessian`: Second Fréchet derivative (Hessian) as a continuous
  bilinear map `EuclideanSpace ℝ (Fin d) →L[ℝ] (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ)`.

## Main results

* `Analysis.Convex.hasFDerivAt_unnormEntropy`: First Fréchet derivative of `unnormEntropy`.
* `Analysis.Convex.hasFDerivAt_unnormEntropyFDeriv`: Second Fréchet derivative of `unnormEntropy`.
* `Analysis.Convex.bregDiv_unnormEntropy_mvt`: Exact second-order Taylor remainder for the
  Bregman divergence of `unnormEntropy`.
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Analysis.Convex

variable {d : ℕ}

/-! ### Definitions -/

/-- The unnormalized negative entropy regularizer:
$$\psi(x) = \sum_{i=1}^d (x_i \ln x_i - x_i)$$ -/
noncomputable def unnormEntropy (x : EuclideanSpace ℝ (Fin d)) : ℝ :=
  ∑ i, (x i * Real.log (x i) - x i)

/-- Coordinate linear projection on `EuclideanSpace ℝ (Fin d)` as a continuous linear map. -/
noncomputable def eucCoord (i : Fin d) : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ :=
  innerSL ℝ (EuclideanSpace.basisFun (Fin d) ℝ i)

@[simp]
lemma eucCoord_apply (i : Fin d) (x : EuclideanSpace ℝ (Fin d)) :
    eucCoord i x = x i :=
  EuclideanSpace.basisFun_inner (𝕜 := ℝ) (ι := Fin d) x i

lemma hasFDerivAt_eucCoord (i : Fin d) (y : EuclideanSpace ℝ (Fin d)) :
    HasFDerivAt (fun x : EuclideanSpace ℝ (Fin d) ↦ x i) (eucCoord i) y := by
  have : (fun x : EuclideanSpace ℝ (Fin d) ↦ x i) = ⇑(eucCoord i) := by ext; simp
  rw [this]
  exact (eucCoord i).hasFDerivAt

lemma hasDerivAt_mul_log_sub {x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (fun t ↦ t * Real.log t - t) (Real.log x) x := by
  have h := ((hasDerivAt_id' x).mul (Real.hasDerivAt_log hx)).sub (hasDerivAt_id' x)
  have : 1 * Real.log x + x * x⁻¹ - 1 = Real.log x := by
    rw [mul_inv_cancel₀ hx, one_mul, add_sub_cancel_right]
  rw [this] at h
  exact h

lemma hasFDerivAt_log_coord (i : Fin d) {y : EuclideanSpace ℝ (Fin d)} (hy : y i ≠ 0) :
    HasFDerivAt (fun x ↦ Real.log (x i)) ((y i)⁻¹ • eucCoord i) y := by
  have h := (Real.hasDerivAt_log hy).comp_hasFDerivAt y (hasFDerivAt_eucCoord i y)
  have : (fun x : EuclideanSpace ℝ (Fin d) ↦ Real.log (x i)) = (Real.log ∘ (fun x ↦ x i)) := rfl
  rw [← this] at h
  exact h

/-- The Fréchet derivative of unnormalized negative entropy at $x \in \mathbb{R}^d_{>0}$:
$$v \mapsto \sum_{i=1}^d v_i \ln x_i$$ -/
noncomputable def unnormEntropyFDeriv (x : EuclideanSpace ℝ (Fin d)) :
    EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ :=
  ∑ i, (Real.log (x i)) • eucCoord i

@[simp]
lemma unnormEntropyFDeriv_apply (x v : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyFDeriv x v = ∑ i, v i * Real.log (x i) := by
  simp [unnormEntropyFDeriv, mul_comm]

/-- The Hessian (second Fréchet derivative) of unnormalized negative entropy at
$x \in \mathbb{R}^d_{>0}$:
$$(v, w) \mapsto \sum_{i=1}^d \frac{v_i w_i}{x_i}$$ -/
noncomputable def unnormEntropyHessian (x : EuclideanSpace ℝ (Fin d)) :
    EuclideanSpace ℝ (Fin d) →L[ℝ] (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) :=
  ∑ i, (x i)⁻¹ • ContinuousLinearMap.smulRight (eucCoord i) (eucCoord i)

@[simp]
lemma unnormEntropyHessian_apply (x v w : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyHessian x v w = ∑ i, (v i * w i) / x i := by
  simp [unnormEntropyHessian, div_eq_inv_mul]

/-! ### Fréchet Derivatives -/

/-- Fréchet derivative of unnormalized negative entropy at any $y \in \mathbb{R}^d_{>0}$. -/
theorem hasFDerivAt_unnormEntropy (y : EuclideanSpace ℝ (Fin d))
    (hy : ∀ i, 0 < y i) :
    HasFDerivAt unnormEntropy (unnormEntropyFDeriv y) y := by
  have h (i : Fin d) : HasFDerivAt (fun x : EuclideanSpace ℝ (Fin d) ↦ x i * Real.log (x i) - x i)
      ((Real.log (y i)) • eucCoord i) y := by
    have h_comp := (hasDerivAt_mul_log_sub (hy i).ne').comp_hasFDerivAt y (hasFDerivAt_eucCoord i y)
    have : (fun x : EuclideanSpace ℝ (Fin d) ↦ x i * Real.log (x i) - x i) =
        ((fun t ↦ t * Real.log t - t) ∘ (fun x ↦ x i)) := rfl
    rw [← this] at h_comp
    exact h_comp
  have h_sum := HasFDerivAt.sum (u := univ) (fun i _ ↦ h i)
  have h_eq : (∑ i, fun (x : EuclideanSpace ℝ (Fin d)) ↦ x i * Real.log (x i) - x i) =
      unnormEntropy := by
    ext x; simp [unnormEntropy, Finset.sum_apply]
  rw [h_eq] at h_sum
  exact h_sum

/-- Second Fréchet derivative (Hessian) of unnormalized negative entropy at
$y \in \mathbb{R}^d_{>0}$. -/
theorem hasFDerivAt_unnormEntropyFDeriv (y : EuclideanSpace ℝ (Fin d))
    (hy : ∀ i, 0 < y i) :
    HasFDerivAt unnormEntropyFDeriv (unnormEntropyHessian y) y := by
  have h (i : Fin d) :
      HasFDerivAt (fun x ↦ (Real.log (x i)) • eucCoord i)
        ((y i)⁻¹ • ContinuousLinearMap.smulRight (eucCoord i) (eucCoord i)) y := by
    have := (hasFDerivAt_log_coord i (hy i).ne').smul_const (eucCoord i)
    have h_clm : ((y i)⁻¹ • eucCoord i).smulRight (eucCoord i) =
        (y i)⁻¹ • ContinuousLinearMap.smulRight (eucCoord i) (eucCoord i) := by
      ext v w; simp [ContinuousLinearMap.smulRight_apply, smul_eq_mul, mul_assoc]
    rwa [h_clm] at this
  have h_sum := HasFDerivAt.sum (u := univ) (fun i _ ↦ h i)
  have h_eq : (∑ i, fun (x : EuclideanSpace ℝ (Fin d)) ↦ (Real.log (x i)) • eucCoord i) =
      unnormEntropyFDeriv := by
    ext x; simp [unnormEntropyFDeriv, Finset.sum_apply]
  rw [h_eq] at h_sum
  exact h_sum

/-! ### Second-Order Taylor Remainder (MVT) for Bregman Divergence -/

theorem bregDiv_unnormEntropy_mvt (x y : EuclideanSpace ℝ (Fin d))
    (h_pos : ∀ z ∈ segment ℝ x y, ∀ i, 0 < (z : EuclideanSpace ℝ (Fin d)) i) :
    ∃ z ∈ segment ℝ x y,
      D_[unnormEntropy](x, y, unnormEntropyFDeriv y) = (1/2 : ℝ) * ∑ i, (x i - y i)^2 / z i := by
  obtain ⟨z, hz, hz_eq⟩ := bregDiv_mvt x y
    (fun z hz ↦ hasFDerivAt_unnormEntropy z (h_pos z hz))
    (fun z hz ↦ hasFDerivAt_unnormEntropyFDeriv z (h_pos z hz))
  refine ⟨z, hz, ?_⟩
  rw [hz_eq, unnormEntropyHessian_apply]
  congr 1
  simp_rw [PiLp.sub_apply, sq]

end Analysis.Convex