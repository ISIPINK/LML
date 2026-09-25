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
# Scaled Unnormalized Negative Entropy Regularizer for LEA

This file defines the scaled unnormalized negative entropy regularizer
$$\psi_\alpha(x) = \alpha \sum_{i=1}^d (x_i \ln x_i - x_i),$$
its Fréchet derivative $\nabla\psi_\alpha(x)$, its Hessian second-derivative bilinear map,
and proves:
1. `hasFDerivAt_unnormEntropy`: Formula for the Fréchet derivative
   $\nabla\psi_\alpha(x) = \alpha (\ln x_i)_i$.
2. `hasFDerivAt_unnormEntropyFDeriv`: Formula for the Hessian bilinear form.
3. `bregDiv_unnormEntropy_mvt`: Second-order Taylor/MVT remainder for the Bregman divergence.
4. `convexOn_unnormEntropy_stdSimplex`: Convexity of `unnormEntropy α` on the standard simplex
   `stdSimplex` for $\alpha \ge 0$.

## Main definitions
* `Online.OCO.LEA.OMD.unnormEntropy`
* `Online.OCO.LEA.OMD.unnormEntropyFDeriv`
* `Online.OCO.LEA.OMD.unnormEntropyHessian`
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {d : ℕ}

/-! ### Definitions -/

/-- The scaled unnormalized negative entropy regularizer with scale $\alpha \in \mathbb{R}$:
$$\psi_\alpha(x) = \alpha \sum_{i=1}^d (x_i \ln x_i - x_i)$$ -/
noncomputable def unnormEntropy (α : ℝ) (x : EuclideanSpace ℝ (Fin d)) : ℝ :=
  α * ∑ i, (x i * Real.log (x i) - x i)

/-- Coordinate projection on `EuclideanSpace ℝ (Fin d)` as a continuous linear map. -/
noncomputable abbrev eucProj (i : Fin d) : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ :=
  EuclideanSpace.proj (𝕜 := ℝ) (ι := Fin d) i

lemma hasDerivAt_mul_log_sub {x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (fun t ↦ t * Real.log t - t) (Real.log x) x := by
  have := (Real.hasDerivAt_mul_log hx).sub (hasDerivAt_id x)
  ring_nf at this
  exact this

/-- The Fréchet derivative of scaled unnormalized negative entropy at $x \in \mathbb{R}^d_{>0}$:
$$v \mapsto \alpha \sum_{i=1}^d v_i \ln x_i$$ -/
noncomputable def unnormEntropyFDeriv (α : ℝ) (x : EuclideanSpace ℝ (Fin d)) :
    EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ :=
  α • (∑ i, (Real.log (x i)) • eucProj i)

@[simp]
lemma unnormEntropyFDeriv_apply (α : ℝ) (x v : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyFDeriv α x v = α * ∑ i, v i * Real.log (x i) := by
  simp [unnormEntropyFDeriv, mul_comm]

/-- The Hessian (second Fréchet derivative) of scaled unnormalized negative entropy at
$x \in \mathbb{R}^d_{>0}$:
$$(v, w) \mapsto \alpha \sum_{i=1}^d \frac{v_i w_i}{x_i}$$ -/
noncomputable def unnormEntropyHessian (α : ℝ) (x : EuclideanSpace ℝ (Fin d)) :
    EuclideanSpace ℝ (Fin d) →L[ℝ] (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) :=
  α • (∑ i, (x i)⁻¹ • ContinuousLinearMap.smulRight (eucProj i) (eucProj i))

@[simp]
lemma unnormEntropyHessian_apply (α : ℝ) (x v w : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyHessian α x v w = α * ∑ i, (v i * w i) / x i := by
  simp [unnormEntropyHessian, div_eq_inv_mul, mul_sum]

/-! ### Fréchet Derivatives -/

/-- Fréchet derivative of scaled unnormalized negative entropy at any $y \in \mathbb{R}^d_{>0}$. -/
theorem hasFDerivAt_unnormEntropy (α : ℝ) (y : EuclideanSpace ℝ (Fin d))
    (hy : ∀ i, 0 < y i) :
    HasFDerivAt (unnormEntropy α) (unnormEntropyFDeriv α y) y := by
  have h := HasFDerivAt.sum (u := (univ : Finset (Fin d)))
    (fun i _ ↦ (hasDerivAt_mul_log_sub (hy i).ne').comp_hasFDerivAt y (eucProj i).hasFDerivAt)
  have h_scaled := h.const_smul α
  have h_eq_fun : (α • (∑ i : Fin d, (fun t ↦ t * Real.log t - t) ∘ ⇑(eucProj i))) =
      unnormEntropy α := by
    ext x; simp [unnormEntropy, smul_eq_mul]
  have h_eq_deriv : α • (∑ i : Fin d, (Real.log (y i)) • eucProj i) =
      unnormEntropyFDeriv α y := rfl
  rw [h_eq_deriv] at h_scaled
  rwa [h_eq_fun] at h_scaled

/-- Second Fréchet derivative (Hessian) of scaled unnormalized negative entropy at
$y \in \mathbb{R}^d_{>0}$. -/
theorem hasFDerivAt_unnormEntropyFDeriv (α : ℝ) (y : EuclideanSpace ℝ (Fin d))
    (hy : ∀ i, 0 < y i) :
    HasFDerivAt (unnormEntropyFDeriv α) (unnormEntropyHessian α y) y := by
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
  have h_sum := (HasFDerivAt.sum (u := univ) (fun i _ ↦ h i)).const_smul α
  have h_eq_fun :
      (α • (∑ i : Fin d, fun x : EuclideanSpace ℝ (Fin d) ↦ (Real.log (x i)) • eucProj i)) =
        unnormEntropyFDeriv α := by
    ext x; simp [unnormEntropyFDeriv]
  have h_eq_deriv : α • (∑ i, (y i)⁻¹ • ContinuousLinearMap.smulRight (eucProj i) (eucProj i)) =
      unnormEntropyHessian α y := rfl
  rw [h_eq_deriv] at h_sum
  rwa [h_eq_fun] at h_sum

/-! ### Second-Order Taylor Remainder (MVT) for Bregman Divergence -/

theorem bregDiv_unnormEntropy_mvt (α : ℝ) (x y : EuclideanSpace ℝ (Fin d))
    (h_pos : ∀ z ∈ segment ℝ x y, ∀ i, 0 < (z : EuclideanSpace ℝ (Fin d)) i) :
    ∃ z ∈ segment ℝ x y,
      D_[unnormEntropy α](x, y, unnormEntropyFDeriv α y) = (α / 2) * ∑ i, (x i - y i)^2 / z i := by
  obtain ⟨z, hz, hz_eq⟩ := bregDiv_mvt x y
    (fun z hz ↦ hasFDerivAt_unnormEntropy α z (h_pos z hz))
    (fun z hz ↦ hasFDerivAt_unnormEntropyFDeriv α z (h_pos z hz))
  refine ⟨z, hz, ?_⟩
  rw [hz_eq, unnormEntropyHessian_apply]
  simp_rw [PiLp.sub_apply, sq]
  ring

/-! ### Convexity of Regularizer -/

/-- Convexity of scaled unnormalized negative entropy on the standard simplex `stdSimplex`
for $\alpha \ge 0$. -/
theorem convexOn_unnormEntropy_stdSimplex (α : ℝ) (hα : 0 ≤ α) :
    ConvexOn ℝ (stdSimplex (d := d)) (unnormEntropy α) := by
  refine ⟨convex_stdSimplex, fun x hx y hy a b ha hb hab ↦ ?_⟩
  rw [mem_stdSimplex_iff] at hx hy
  dsimp [unnormEntropy]
  have h_base : ∑ i, ( (a * x i + b * y i) * Real.log (a * x i + b * y i) - (a * x i + b * y i) ) ≤
      a * (∑ i, (x i * Real.log (x i) - x i)) + b * (∑ i, (y i * Real.log (y i) - y i)) := by
    rw [mul_sum, mul_sum, ← sum_add_distrib]
    exact sum_le_sum fun i _ ↦ by
      have := Real.convexOn_mul_log.2 (hx.1 i) (hy.1 i) ha hb hab
      dsimp at this
      linarith
  have h_mul := mul_le_mul_of_nonneg_left h_base hα
  calc
    α * ∑ i, ((a • x + b • y) i * Real.log ((a • x + b • y) i) - (a • x + b • y) i)
      = α * ∑ i, ((a * x i + b * y i) * Real.log (a * x i + b * y i) - (a * x i + b * y i)) := rfl
    _ ≤ α * (a * (∑ i, (x i * Real.log (x i) - x i)) +
          b * (∑ i, (y i * Real.log (y i) - y i))) := h_mul
    _ = a * (α * ∑ i, (x i * Real.log (x i) - x i)) +
          b * (α * ∑ i, (y i * Real.log (y i) - y i)) := by
        ring

/-! ### Shifted Regularizer -/

/-- Shifted unnormalized negative entropy regularizer with offset $\alpha (\ln d + 1)$:
$$\psi^{\mathrm{shift}}_\alpha(x) = \psi_\alpha(x) + \alpha (\ln d + 1)$$ -/
noncomputable def unnormEntropyShifted (α : ℝ) (x : EuclideanSpace ℝ (Fin d)) : ℝ :=
  unnormEntropy α x + α * (Real.log d + 1)

lemma unnormEntropyShifted_eq (α : ℝ) (x : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyShifted α x = α * (∑ i, (x i * Real.log (x i) - x i) + Real.log d + 1) := by
  dsimp [unnormEntropyShifted, unnormEntropy]
  ring

lemma unnormEntropyShifted_one_eq_add (x : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyShifted 1 x = unnormEntropy 1 x + (Real.log d + 1) := by
  dsimp [unnormEntropyShifted]
  ring

lemma unnormEntropyShifted_smul (α : ℝ) (x : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyShifted α x = α • unnormEntropyShifted 1 x := by
  dsimp [unnormEntropyShifted, unnormEntropy]
  ring

lemma unnormEntropyShifted_sub (α β : ℝ) (x : EuclideanSpace ℝ (Fin d)) :
    unnormEntropyShifted α x - unnormEntropyShifted β x =
      unnormEntropy α x - unnormEntropy β x + (α - β) * (Real.log d + 1) := by
  dsimp [unnormEntropyShifted]
  ring

@[simp]
lemma bregDiv_unnormEntropyShifted (α : ℝ) (x y : EuclideanSpace ℝ (Fin d)) :
    D_[unnormEntropyShifted α](x, y, unnormEntropyFDeriv α y) =
      D_[unnormEntropy α](x, y, unnormEntropyFDeriv α y) := by
  dsimp [bregDiv, unnormEntropyShifted]
  ring

end Online.OCO.LEA.OMD

