/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.ForMathlib.Analysis.Convex.Bregman.Basic
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Bregman.MVT
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Domain
public import Mathlib.Analysis.Convex.StdSimplex
public import Mathlib.Analysis.Calculus.Deriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Add
public import Mathlib.Analysis.Calculus.FDeriv.Comp
public import Mathlib.Analysis.Calculus.FDeriv.Mul
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.Log.NegMulLog

/-!
# Unnormalized Negative Entropy Regularizer for LEA

This file defines the unnormalized negative entropy regularizer
$$\psi(x) = \sum_{i=1}^d (x_i \ln x_i - x_i),$$
its Fréchet derivative $\nabla\psi(x)$, its Hessian second-derivative bilinear map,
and proves:
1. `hasFDerivAt_unnormEntropy`: Formula for the Fréchet derivative $\nabla\psi(x) = (\ln x_i)_i$.
2. `hasFDerivAt_unnormEntropyFDeriv`: Formula for the Hessian bilinear form.
3. `bregDiv_unnormEntropy_mvt`: Second-order Taylor/MVT remainder for the Bregman divergence.
4. `convexOn_unnormEntropy_stdSimplex`: Convexity of `unnormEntropy` on the standard simplex
   `stdSimplex`.

## Main definitions
* `Online.OCO.LEA.OMD.unnormEntropy`
* `Online.OCO.LEA.OMD.unnormEntropyFDeriv`
* `Online.OCO.LEA.OMD.unnormEntropyHessian`
-/

set_option linter.style.longLine false

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {d : ℕ}

/-! ### Definitions -/

/-- The unnormalized negative entropy regularizer:
$$\psi(x) = \sum_{i=1}^d (x_i \ln x_i - x_i)$$ -/
noncomputable def unnormEntropy (x : EuclideanSpace ℝ (Fin d)) : ℝ :=
  ∑ i, (x i * Real.log (x i) - x i)

/-- Coordinate projection on `EuclideanSpace ℝ (Fin d)` as a continuous linear map. -/
noncomputable abbrev eucProj (i : Fin d) : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ :=
  EuclideanSpace.proj (𝕜 := ℝ) (ι := Fin d) i

lemma hasDerivAt_mul_log_sub {x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (fun t ↦ t * Real.log t - t) (Real.log x) x := by
  have := (Real.hasDerivAt_mul_log hx).sub (hasDerivAt_id x)
  ring_nf at this
  exact this

/-- The Fréchet derivative of unnormalized negative entropy at $x \in \mathbb{R}^d_{>0}$:
$$v \mapsto \sum_{i=1}^d v_i \ln x_i$$ -/
noncomputable def unnormEntropyFDeriv (x : EuclideanSpace ℝ (Fin d)) :
    EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ :=
  ∑ i, (Real.log (x i)) • eucProj i

@[simp]
lemma unnormEntropyFDeriv_apply (x v : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyFDeriv x v = ∑ i, v i * Real.log (x i) := by
  simp [unnormEntropyFDeriv, mul_comm]

/-- The Hessian (second Fréchet derivative) of unnormalized negative entropy at
$x \in \mathbb{R}^d_{>0}$:
$$(v, w) \mapsto \sum_{i=1}^d \frac{v_i w_i}{x_i}$$ -/
noncomputable def unnormEntropyHessian (x : EuclideanSpace ℝ (Fin d)) :
    EuclideanSpace ℝ (Fin d) →L[ℝ] (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) :=
  ∑ i, (x i)⁻¹ • ContinuousLinearMap.smulRight (eucProj i) (eucProj i)

@[simp]
lemma unnormEntropyHessian_apply (x v w : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyHessian x v w = ∑ i, (v i * w i) / x i := by
  simp [unnormEntropyHessian, div_eq_inv_mul]



/-! ### Fréchet Derivatives -/

/-- Fréchet derivative of unnormalized negative entropy at any $y \in \mathbb{R}^d_{>0}$. -/
theorem hasFDerivAt_unnormEntropy (y : EuclideanSpace ℝ (Fin d))
    (hy : ∀ i, 0 < y i) :
    HasFDerivAt unnormEntropy (unnormEntropyFDeriv y) y := by
  have h := HasFDerivAt.sum (u := (univ : Finset (Fin d)))
    (fun i _ ↦ (hasDerivAt_mul_log_sub (hy i).ne').comp_hasFDerivAt y (eucProj i).hasFDerivAt)
  have h_eq : (∑ i : Fin d, (fun t ↦ t * Real.log t - t) ∘ ⇑(eucProj i)) =
      (unnormEntropy : EuclideanSpace ℝ (Fin d) → ℝ) := by
    ext x; simp [unnormEntropy, Finset.sum_apply]
  rwa [h_eq] at h

/-- Second Fréchet derivative (Hessian) of unnormalized negative entropy at
$y \in \mathbb{R}^d_{>0}$. -/
theorem hasFDerivAt_unnormEntropyFDeriv (y : EuclideanSpace ℝ (Fin d))
    (hy : ∀ i, 0 < y i) :
    HasFDerivAt unnormEntropyFDeriv (unnormEntropyHessian y) y := by
  have h (i : Fin d) : HasFDerivAt (fun x ↦ (Real.log (x i)) • eucProj i)
      ((y i)⁻¹ • ContinuousLinearMap.smulRight (eucProj i) (eucProj i)) y := by
    have hlog : HasFDerivAt (fun x ↦ Real.log (x i)) ((y i)⁻¹ • eucProj i) y := by
      change HasFDerivAt (Real.log ∘ fun x ↦ x i) ((y i)⁻¹ • eucProj i) y
      exact (Real.hasDerivAt_log (hy i).ne').comp_hasFDerivAt y (eucProj i).hasFDerivAt
    have h_smul := hlog.smul_const (eucProj i)
    have h_clm : ((y i)⁻¹ • eucProj i).smulRight (eucProj i) =
        (y i)⁻¹ • ContinuousLinearMap.smulRight (eucProj i) (eucProj i) := by
      ext v w; simp [ContinuousLinearMap.smulRight_apply, smul_eq_mul, mul_assoc]
    rwa [h_clm] at h_smul
  have h_sum := HasFDerivAt.sum (u := univ) (fun i _ ↦ h i)
  have h_eq : (∑ i, fun x : EuclideanSpace ℝ (Fin d) ↦ (Real.log (x i)) • eucProj i) =
      unnormEntropyFDeriv := by
    ext x; simp [unnormEntropyFDeriv, Finset.sum_apply]
  rwa [h_eq] at h_sum

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

/-! ### Convexity of Regularizer -/

/-- Convexity of unnormalized negative entropy on the standard simplex `stdSimplex`. -/
theorem convexOn_unnormEntropy_stdSimplex :
    ConvexOn ℝ (stdSimplex (d := d)) unnormEntropy := by
  refine ⟨convex_stdSimplex, fun x hx y hy a b ha hb hab ↦ ?_⟩
  rw [mem_stdSimplex_iff] at hx hy
  dsimp [unnormEntropy]
  rw [mul_sum, mul_sum, ← sum_add_distrib]
  exact sum_le_sum fun i _ ↦ by
    have := Real.convexOn_mul_log.2 (hx.1 i) (hy.1 i) ha hb hab
    dsimp at this
    linarith

end Online.OCO.LEA.OMD
