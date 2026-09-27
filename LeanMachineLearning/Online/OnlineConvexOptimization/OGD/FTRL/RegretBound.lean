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
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.FTRL.RegretDecomposition
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.FTRL.Boundary
public import LeanMachineLearning.Online.OnlineConvexOptimization.OGD.FTRL.Optimality

/-!
# Regret Bound for Follow-the-Regularized-Leader (FTRL / Lazy OGD)

This file proves the multi-round cumulative regret bound for Follow-the-Regularized-Leader
(FTRL / Lazy OGD) on an arbitrary convex set $s \subseteq E$:
$$\sum_{t=1}^T (l_t(w_t) - l_t(u)) \le
  \frac{\alpha_{T+1}}{2} R^2 + \sum_{t=1}^T \frac{1}{2\alpha_t} \|g_t\|^2.$$

## Main results
* `regret_bound`: The cumulative regret bound for FTRL on an arbitrary convex set $s$.
-/

open scoped RealInnerProductSpace BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.OGD.FTRL

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Cumulative regret upper bound for Euclidean FTRL / Lazy OGD on a convex set $s \subseteq E$:
$$\sum_{t=1}^T (l_t(w_t) - l_t(u)) \le
  \frac{\alpha_{T+1}}{2} \|u\|^2 + \sum_{t=1}^T \frac{1}{2\alpha_t} \|g_t\|^2.$$ -/
theorem regret_bound (T : ℕ)
    (α : ℕ → ℝ) (hα_pos : ∀ t ∈ Ico 1 (T + 2), 0 < α t)
    (h_mono : ∀ t ∈ Ico 1 (T + 1), α t ≤ α (t + 1))
    (s : Set E) (hs : Convex ℝ s)
    (l : ℕ → E → ℝ)
    (g : ℕ → (E →L[ℝ] ℝ))
    (w : ℕ → E) (hw1 : w 1 = 0)
    (hw_mem : ∀ t ∈ Ico 1 (T + 2), w t ∈ s)
    (hw_min : ∀ t ∈ Ico 1 (T + 2),
      IsMinOn (F_obj (fun k ↦ eucSq (E := E) (α k)) g t) s (w t))
    (hg : ∀ t ∈ Ico 1 (T + 1), HasSubgradientWithinAt (l t) (g t) s (w t))
    (u : E) (hu : u ∈ s) :
    ∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t u) ≤
      (α (T + 1) / 2) * ‖u‖^2
      + ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) * ‖g t‖^2 := by
  have hT1 : T + 1 ∈ Ico 1 (T + 2) := by rw [mem_Ico]; omega
  have h_bound := boundary_eucSq_eq α T u w hw1
  have h_shift := OGD.sum_shift_eucSq_nonpos α w T h_mono
  have h_stab := OGD.sum_stability_le_norm_sq α T
    (fun t ht ↦ hα_pos t (Ico_subset_Ico_right (by omega) ht)) w g
  have h_lin : ∑ t ∈ Ico 1 (T + 1), linearization u w g l t ≤ 0 :=
    sum_nonpos fun t ht ↦ by dsimp [linearization]; linarith [hg t ht u hu]
  have h_opt : 0 ≤ ∑ t ∈ Ico 1 (T + 1), optimality (fun k ↦ eucSqFDeriv (α k)) w g t := by
    refine sum_nonneg fun t ht ↦ ?_
    have ht1 : t ∈ Ico 1 (T + 2) := Ico_subset_Ico_right (by omega) ht
    have ht2 : t + 1 ∈ Ico 1 (T + 2) := by rw [mem_Ico] at ht ⊢; omega
    exact optimality_nonneg_of_isMinOn t (hasFDerivAt_eucSq (α t) (w t))
      (convexOn_eucSq (α t) (hα_pos t ht1).le s hs) (hw_mem t ht1) (hw_mem (t + 1) ht2)
      (hw_min t ht1)
  have h_term := terminalOptimality_nonpos_of_isMinOn T hu (hw_min (T + 1) hT1)
  linarith [regret_decomposition_eq (fun k ↦ eucSq (E := E) (α k))
    (fun k ↦ eucSqFDeriv (α k)) u w g l T]

end Online.OCO.OGD.FTRL
