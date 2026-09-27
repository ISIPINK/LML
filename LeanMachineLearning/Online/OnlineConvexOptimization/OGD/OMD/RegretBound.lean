/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Algebra.BigOperators.Intervals
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.Common.Regularizer
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.Common.Shift
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.Common.Stability
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.OMD.RegretDecomposition
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.OMD.Boundary
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.OMD.Optimality
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.OMD.Pathlength

/-!
# Regret Bound for Online Mirror Descent (OMD / Projected OGD)

This file proves the multi-round cumulative dynamic regret bound for Online Mirror Descent
(OMD / Projected OGD) on an arbitrary convex set $s \subseteq E$:
$$\sum_{t=1}^T (l_t(w_t) - l_t(u_t)) \le
  \frac{\alpha_{T+1}}{2} R_u^2 + R_w \sum_{t=1}^T \alpha_t \|u_{t-1} - u_t\| +
  \sum_{t=1}^T \frac{1}{2\alpha_t} \|g_t\|^2.$$

## Main results
* `regret_bound`: The cumulative dynamic regret bound for OMD on an arbitrary convex set $s$.
-/

open scoped RealInnerProductSpace BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.OGD.OMD

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Cumulative dynamic regret upper bound for Euclidean OMD / Projected OGD on convex set $s$:
$$\sum_{t=1}^T (l_t(w_t) - l_t(u_t)) \le
  \frac{\alpha_{T+1}}{2} R_u^2 + R_w \sum_{t=1}^T \alpha_t \|u_{t-1} - u_t\| +
  \sum_{t=1}^T \frac{1}{2\alpha_t} \|g_t\|^2.$$ -/
theorem regret_bound (T : ℕ)
    (α : ℕ → ℝ) (hα_pos : ∀ t ∈ Ico 1 (T + 2), 0 < α t)
    (h_mono : ∀ t ∈ Ico 1 (T + 1), α t ≤ α (t + 1))
    (s : Set E) (hs : Convex ℝ s)
    (l : ℕ → E → ℝ)
    (g : ℕ → (E →L[ℝ] ℝ))
    (w : ℕ → E) (hw1 : w 1 = 0)
    (hw_mem : ∀ t ∈ Ico 1 (T + 2), w t ∈ s)
    (hw_min : ∀ t ∈ Ico 1 (T + 1),
      IsMinOn (fun x ↦ (g t) x + eucSq (α (t + 1)) x - (eucSqFDeriv (α t) (w t)) x) s (w (t + 1)))
    (hg : ∀ t ∈ Ico 1 (T + 1), HasSubgradientWithinAt (l t) (g t) s (w t))
    (u : ℕ → E) (hu : ∀ t ∈ Ico 1 (T + 1), u t ∈ s)
    (R_u R_w : ℝ) (hu_norm : ∀ t ≤ T, ‖u t‖ ≤ R_u) (hw_norm : ∀ t ∈ Ico 1 (T + 1), ‖w t‖ ≤ R_w) :
    ∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t (u t)) ≤
      (α (T + 1) / 2) * R_u^2
      + R_w * (∑ t ∈ Ico 1 (T + 1), α t * ‖u (t - 1) - u t‖)
      + ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) * ‖g t‖^2 := by
  have hT1 : T + 1 ∈ Ico 1 (T + 2) := by rw [mem_Ico]; omega
  have h_bound := boundary_eucSq_le α T (hα_pos (T + 1) hT1).le R_u u hu_norm w hw1
  have h_shift := OGD.sum_shift_eucSq_nonpos α w T h_mono
  have h_stab := OGD.sum_stability_le_norm_sq α T
    (fun t ht ↦ hα_pos t (Ico_subset_Ico_right (by omega) ht)) w g
  have h_path := sum_pathlength_eucSq_le α T
    (fun t ht ↦ (hα_pos t (Ico_subset_Ico_right (by omega) ht)).le) u w R_w hw_norm
  have h_lin : ∑ t ∈ Ico 1 (T + 1), linearization u w g l t ≤ 0 :=
    sum_nonpos fun t ht ↦ by dsimp [linearization]; linarith [hg t ht (u t) (hu t ht)]
  have h_opt : 0 ≤ ∑ t ∈ Ico 1 (T + 1), optimality (fun k ↦ eucSqFDeriv (α k)) u w g t := by
    refine sum_nonneg fun t ht ↦ ?_
    have ht1 : t ∈ Ico 1 (T + 1) := ht
    have ht2 : t + 1 ∈ Ico 1 (T + 2) := by rw [mem_Ico] at ht ⊢; omega
    have h_opt_t : 0 ≤ optimality (fun k ↦ eucSqFDeriv (α k)) u w g t :=
      optimality_nonneg_of_isMinOn (ψ := fun k ↦ eucSq (α k)) (gψ := fun k ↦ eucSqFDeriv (α k))
        t (hasFDerivAt_eucSq (α (t + 1)) (w (t + 1)))
        (convexOn_eucSq (α (t + 1)) (hα_pos (t + 1) ht2).le s hs) (hw_mem (t + 1) ht2) (hu t ht)
        (hw_min t ht1)
    exact h_opt_t
  linarith [regret_decomposition_eq (fun k ↦ eucSq (E := E) (α k))
    (fun k ↦ eucSqFDeriv (α k)) u w g l T]

end Online.OCO.OGD.OMD
