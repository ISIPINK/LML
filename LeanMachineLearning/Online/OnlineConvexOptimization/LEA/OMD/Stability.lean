/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Domain
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.RegretDecomposition
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Regularizer
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Basic

/-!
# Stability Bound for Learning with Expert Advice (LEA)

This file proves the local-norm stability bound for Learning with Expert Advice (LEA)
using coordinate-wise Fenchel-Young inequality and the Mean Value Theorem for Bregman divergence.

## Main results
* `Online.OCO.LEA.OMD.dual_local_norm_eta_le`: Coordinate-weighted duality bound with
  step size $\eta$:
  $\eta \sum_i v_i g_i - \frac{1}{2} \sum_i \frac{v_i^2}{z_i} \le
    \frac{\eta^2}{2} \sum_i z_i g_i^2$.
* `Online.OCO.LEA.OMD.stability_le_dual_norm`: One-round stability term bound via MVT
  $\delta_t \le \frac{\eta^2}{2} \sum_i z_i (g_t)_i^2$.
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {d : ℕ}

/-! ### Scalar and Coordinate Duality (Fenchel-Young) -/

/-- Scaled coordinate-wise duality inequality with learning rate $\eta$:
$$\eta \sum_i v_i g_i - \frac{1}{2} \sum_i \frac{v_i^2}{z_i}
  \le \frac{\eta^2}{2} \sum_i z_i g_i^2$$ -/
lemma dual_local_norm_eta_le (η : ℝ) (v : EuclideanSpace ℝ (Fin d)) (g : Fin d → ℝ)
    (z : EuclideanSpace ℝ (Fin d)) (hz : ∀ i, 0 < z i) :
    η * (∑ i, v i * g i) - (1 / 2 : ℝ) * (∑ i, (v i) ^ 2 / z i) ≤
      (η ^ 2 / 2) * ∑ i, z i * (g i) ^ 2 := by
  calc η * (∑ i, v i * g i) - (1 / 2 : ℝ) * (∑ i, (v i) ^ 2 / z i)
    _ = ∑ i, (v i * (η * g i) - (1 / 2 : ℝ) * ((v i) ^ 2 / z i)) := by
      rw [mul_sum, mul_sum, ← sum_sub_distrib]; congr 1 with i; ring
    _ ≤ ∑ i, (1 / 2 : ℝ) * (z i * (η * g i) ^ 2) := sum_le_sum fun i _ ↦ by
      have : (v i - z i * (η * g i))^2 / z i =
          (v i)^2 / z i - 2 * (v i * (η * g i)) + z i * (η * g i)^2 := by
        field_simp [(hz i).ne']; ring
      linarith [div_nonneg (sq_nonneg (v i - z i * (η * g i))) (hz i).le]
    _ = (η ^ 2 / 2) * ∑ i, z i * (g i) ^ 2 := by
      rw [mul_sum]; congr 1 with i; ring

variable (η : ℝ)
variable (w : ℕ → EuclideanSpace ℝ (Fin d))
variable (l : ℕ → EuclideanSpace ℝ (Fin d) → ℝ)
variable (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ))

/-- One-round stability bound for Learning with Expert Advice via the Mean Value Theorem (MVT):
when $g_t \in \partial l_t(w_t)$ is a subgradient on the standard simplex,
$w_{t+1} \in \text{stdSimplex}$, and the iterates $w_{t+1}, w_t$ have strictly positive coordinates,
the round-$t$ stability term
`stability η (fun _ ↦ unnormEntropy) (fun _ ↦ unnormEntropyFDeriv) w l t`
is upper bounded by $\frac{\eta^2}{2} \sum_{i=1}^d z_i (g_t)_i^2$
for some intermediate point $z \in [w_{t+1}, w_t]$. -/
theorem stability_le_dual_norm (hη : 0 ≤ η) (t : ℕ)
    (hw_next : w (t + 1) ∈ stdSimplex (d := d))
    (hw_next_pos : ∀ i, 0 < (w (t + 1) : EuclideanSpace ℝ (Fin d)) i)
    (hw_t_pos : ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i)
    (hg : HasSubgradientWithinAt (l t) (g t) (stdSimplex (d := d)) (w t)) :
    ∃ z ∈ segment ℝ (w (t + 1)) (w t),
      stability η (fun _ ↦ unnormEntropy) (fun _ ↦ unnormEntropyFDeriv) w l t ≤
        (η ^ 2 / 2) * ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
  have h_pos : ∀ z ∈ segment ℝ (w (t + 1)) (w t), ∀ i, 0 < (z : EuclideanSpace ℝ (Fin d)) i := by
    rintro z ⟨a, b, ha, hb, hab, rfl⟩ i
    dsimp
    obtain rfl | ha_pos := eq_or_lt_of_le ha
    · have : b = 1 := by linarith
      simp [this, hw_t_pos i]
    · linarith [mul_pos ha_pos (hw_next_pos i), mul_nonneg hb (hw_t_pos i).le]
  obtain ⟨z, hz_seg, hz_breg⟩ := bregDiv_unnormEntropy_mvt (w (t + 1)) (w t) h_pos
  refine ⟨z, hz_seg, ?_⟩
  dsimp [stability]
  rw [hz_breg]
  have h_decomp : g t (w t - w (t + 1)) =
      ∑ i, (w t i - w (t + 1) i) * g t (EuclideanSpace.basisFun (Fin d) ℝ i) := by
    have : w t - w (t + 1) =
        ∑ i, (w t i - w (t + 1) i) • EuclideanSpace.basisFun (Fin d) ℝ i := by
      ext i; simp [EuclideanSpace.basisFun_apply, Pi.single_apply]
    rw [this, map_sum]
    refine sum_congr rfl fun i _ ↦ by rw [map_smul, smul_eq_mul]
  have h_sub : l t (w t) - l t (w (t + 1)) ≤ g t (w t - w (t + 1)) := by
    have h_loss := hg (w (t + 1)) hw_next
    dsimp [bregDiv] at h_loss
    linarith [show (g t) (w t - w (t + 1)) = - (g t) (w (t + 1) - w t) by
      rw [← map_neg, neg_sub]]
  simp_rw [show ∀ i, (w (t + 1) i - w t i)^2 = (w t i - w (t + 1) i)^2 from
    fun i ↦ by rw [← neg_sub (w t i) (w (t + 1) i), neg_sq]]
  have h_dual := dual_local_norm_eta_le η (w t - w (t + 1))
    (fun i ↦ g t (EuclideanSpace.basisFun (Fin d) ℝ i)) z (h_pos z hz_seg)
  have h_pi (i : Fin d) : (w t - w (t + 1)) i = w t i - w (t + 1) i := rfl
  simp_rw [h_pi] at h_dual
  have h_mul_loss : η * (l t (w t) - l t (w (t + 1))) ≤
      η * ∑ i, (w t i - w (t + 1) i) * g t (EuclideanSpace.basisFun (Fin d) ℝ i) := by
    rw [← h_decomp]
    exact mul_le_mul_of_nonneg_left h_sub hη
  linarith

end Online.OCO.LEA.OMD
