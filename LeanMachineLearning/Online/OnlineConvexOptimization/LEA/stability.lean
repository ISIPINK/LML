/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import LeanMachineLearning.Online.OnlineConvexOptimization.Algorithms.OMD.StrongLemmas
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.regularizer
public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Subgradient.Basic

/-!
# Stability Bound for Linear Exponential Adaptation (LEA)

This file proves the local-norm stability bound for Linear Exponential Adaptation (LEA)
using Mathlib's Euclidean and inner product space Cauchy-Schwarz inequality.

## Main results

* `Analysis.Convex.fenchel_young_coord`: Scalar quadratic Young's inequality
  $a b \le \frac{1}{2} z a^2 + \frac{1}{2 z} b^2$.
* `Analysis.Convex.dual_local_norm_le`: Coordinate-weighted duality bound
  $\sum_i v_i g_i - \frac{1}{2} \sum_i \frac{v_i^2}{z_i} \le \frac{1}{2} \sum_i z_i g_i^2$.
* `Analysis.Convex.lea_stability_round_le`: One-round stability term bound
  $\delta_t \le \frac{\eta^2}{2} \sum_i z_i (g_t)_i^2$.
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Analysis.Convex

variable {d : ℕ}

/-! ### Scalar and Coordinate Duality (Cauchy-Schwarz / Young) -/

/-- Scalar Fenchel-Young inequality with coordinate weight $z > 0$:
$$a b - \frac{1}{2 z} a^2 \le \frac{z}{2} b^2$$ -/
lemma fenchel_young_coord (a b z : ℝ) (hz : 0 < z) :
    a * b - (1 / 2) * (a^2 / z) ≤ (1 / 2) * (z * b^2) := by
  have hz_ne : z ≠ 0 := hz.ne'
  have : 0 ≤ (a / Real.sqrt z - Real.sqrt z * b)^2 := sq_nonneg _
  have h_exp : (a / Real.sqrt z - Real.sqrt z * b)^2 =
      a^2 / z - 2 * (a * b) + z * b^2 := by
    calc (a / Real.sqrt z - Real.sqrt z * b)^2
      _ = (a / Real.sqrt z)^2 - 2 * (a / Real.sqrt z) * (Real.sqrt z * b) +
          (Real.sqrt z * b)^2 := by ring
      _ = a^2 / (Real.sqrt z)^2 - 2 * (a * b * (Real.sqrt z / Real.sqrt z)) +
          (Real.sqrt z)^2 * b^2 := by ring
      _ = a^2 / z - 2 * (a * b) + z * b^2 := by
        rw [Real.sq_sqrt hz.le, div_self (Real.sqrt_pos.mpr hz).ne', mul_one]
  linarith

/-- Vector duality inequality: for any vector difference $v$, gradient $g$,
and positive weights $z \in \mathbb{R}^d_{>0}$:
$$\sum_i v_i g_i - \frac{1}{2} \sum_i \frac{v_i^2}{z_i} \le \frac{1}{2} \sum_i z_i g_i^2$$ -/
lemma dual_local_norm_le (v : EuclideanSpace ℝ (Fin d)) (g : Fin d → ℝ)
    (z : EuclideanSpace ℝ (Fin d)) (hz : ∀ i, 0 < z i) :
    (∑ i, v i * g i) - (1 / 2 : ℝ) * (∑ i, (v i)^2 / z i) ≤
      (1 / 2 : ℝ) * ∑ i, z i * (g i)^2 := by
  have h_sum : (∑ i, v i * g i) - (1 / 2 : ℝ) * (∑ i, (v i)^2 / z i) =
      ∑ i, (v i * g i - (1 / 2 : ℝ) * ((v i)^2 / z i)) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
  rw [h_sum, Finset.mul_sum]
  apply Finset.sum_le_sum
  intro i _
  exact fenchel_young_coord (v i) (g i) (z i) (hz i)

/-- Scaled vector duality inequality with learning rate $\eta$:
$$\eta \sum_i v_i g_i - \frac{1}{2} \sum_i \frac{v_i^2}{z_i}
  \le \frac{\eta^2}{2} \sum_i z_i g_i^2$$ -/
lemma dual_local_norm_eta_le (η : ℝ) (v : EuclideanSpace ℝ (Fin d)) (g : Fin d → ℝ)
    (z : EuclideanSpace ℝ (Fin d)) (hz : ∀ i, 0 < z i) :
    η * (∑ i, v i * g i) - (1 / 2 : ℝ) * (∑ i, (v i) ^ 2 / z i) ≤
      (η ^ 2 / 2) * ∑ i, z i * (g i) ^ 2 := by
  have h_mul : η * (∑ i, v i * g i) = ∑ i, v i * (η * g i) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl (fun i _ ↦ by ring)
  rw [h_mul]
  have h_dual := dual_local_norm_le v (fun i ↦ η * g i) z hz
  have h_rw : (1 / 2 : ℝ) * (∑ i, z i * (η * g i) ^ 2) =
      (η ^ 2 / 2) * ∑ i, z i * (g i) ^ 2 := by
    calc (1 / 2 : ℝ) * (∑ i, z i * (η * g i) ^ 2)
      _ = (1 / 2 : ℝ) * (∑ i, (η ^ 2) * (z i * (g i) ^ 2)) := by
        congr 1
        refine Finset.sum_congr rfl (fun i _ ↦ by ring)
      _ = (1 / 2 : ℝ) * (η ^ 2 * ∑ i, z i * (g i) ^ 2) := by
        rw [← Finset.mul_sum]
      _ = (η ^ 2 / 2) * ∑ i, z i * (g i) ^ 2 := by ring
  linarith

/-! ### One-Round Stability Bound for LEA -/

/-- One-round stability bound for Linear Exponential Adaptation:
under loss convexity and MVT divergence representation,
the one-round stability $\delta_t$ is upper bounded by the local dual norm
$\frac{\eta^2}{2} \sum_i z_i (g_t)_i^2$. -/
theorem lea_stability_le (η : ℝ) (hη : 0 ≤ η)
    (w_prev w_next : EuclideanSpace ℝ (Fin d))
    (l_val_prev l_val_next : ℝ)
    (g_val : Fin d → ℝ)
    (h_conv : l_val_prev - l_val_next ≤ ∑ i, (w_prev i - w_next i) * g_val i)
    (z : EuclideanSpace ℝ (Fin d)) (hz : ∀ i, 0 < z i)
    (h_breg : D_[unnormEntropy](w_next, w_prev, unnormEntropyFDeriv w_prev) =
      (1 / 2 : ℝ) * ∑ i, (w_next i - w_prev i) ^ 2 / z i) :
    η * (l_val_prev - l_val_next) -
      D_[unnormEntropy](w_next, w_prev, unnormEntropyFDeriv w_prev) ≤
      (η ^ 2 / 2) * ∑ i, z i * (g_val i) ^ 2 := by
  have h_sq (i : Fin d) : (w_next i - w_prev i)^2 = (w_prev i - w_next i)^2 := by
    rw [← neg_sub (w_prev i) (w_next i), neg_sq]
  have h_breg' : D_[unnormEntropy](w_next, w_prev, unnormEntropyFDeriv w_prev) =
      (1 / 2 : ℝ) * ∑ i, (w_prev i - w_next i)^2 / z i := by
    rw [h_breg]
    simp_rw [h_sq]
  rw [h_breg']
  have h_step1 : η * (l_val_prev - l_val_next) ≤ η * ∑ i, (w_prev i - w_next i) * g_val i :=
    mul_le_mul_of_nonneg_left h_conv hη
  have h_step2 := dual_local_norm_eta_le η (w_prev - w_next) g_val z hz
  have h_pi (i : Fin d) : (w_prev - w_next) i = w_prev i - w_next i := by rfl
  simp_rw [h_pi] at h_step2
  linarith

end Analysis.Convex
