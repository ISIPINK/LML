/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Algebra.BigOperators.Intervals
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Domain
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Regularizer
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Shift
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Stability
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.FTRL.RegretDecomposition
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.FTRL.Boundary
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.FTRL.Optimality
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.FTRL.Formula

/-!
# Regret Bound for Follow-the-Regularized-Leader (FTRL) in LEA

This file proves the multi-round regret bound for Learning with Expert Advice (LEA) using
Follow-the-Regularized-Leader (FTRL) with time-varying unnormalized negative entropy regularization.

## Main results

* `Online.OCO.LEA.FTRL.regret_bound`: For arbitrary iterate sequence $w$ satisfying the initial
  condition, simplex membership, positivity, and the step minimizer condition `IsMinOn`.
* `Online.OCO.LEA.FTRL.ftrlExpWeights_regret_bound`: The fully constructive regret bound for the
  FTRL Exponential Weights trajectory `ftrlExpWeights hd α g`.
* `Online.OCO.LEA.FTRL.regret_bound_wt`: Bounding stability directly by the iterate-weighted loss
  variance $\sum_i w_{t, i} (g_t)_i^2$ for non-negative losses $g_t \ge 0$.
* `Online.OCO.LEA.FTRL.ftrlExpWeights_regret_bound_wt`: Constructive iterate-weighted regret bound.
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.FTRL

open Online.OCO.LEA.OMD

variable {d : ℕ}

/-- Cumulative regret upper bound for LEA with time-varying scaled unnormalized negative entropy
regularizers $\psi_t(x) = \alpha_t \sum (x_i \ln x_i - x_i)$ for scale sequence $\alpha_t > 0$,
under FTRL:
$$\sum_{t=1}^T (l_t(w_t) - l_t(u)) \le \alpha_1 \ln d + (\psi_{\alpha(T+1)}(u) - \psi_{\alpha 1}(u))
  + \sum_{t=1}^T \frac{1}{2\alpha_t} \|g_t\|_\infty^2.$$ -/
theorem regret_bound (α : ℕ → ℝ) (T : ℕ) (hα_pos : ∀ t ∈ Ico 1 (T + 2), 0 < α t) (hd : 0 < d)
    (u : EuclideanSpace ℝ (Fin d)) (hu : u ∈ stdSimplex (d := d))
    (w : ℕ → EuclideanSpace ℝ (Fin d))
    (hw1 : w 1 = uniformSimplex d)
    (hw : ∀ t ∈ Ico 1 (T + 2), w t ∈ stdSimplex (d := d))
    (hw_pos : ∀ t ∈ Ico 1 (T + 2), ∀ i, 0 < w t i)
    (l : ℕ → EuclideanSpace ℝ (Fin d) → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ))
    (h_min : ∀ t ∈ Ico 1 (T + 1),
      IsMinOn (fun x ↦ unnormEntropy (α t) x + (∑ i ∈ Ico 1 t, g i) x)
        (stdSimplex (d := d)) (w t))
    (h_term_min : IsMinOn (F_obj (fun t x ↦ unnormEntropy (α t) x) g (T + 1))
      (stdSimplex (d := d)) (w (T + 1)))
    (hg : ∀ t ∈ Ico 1 (T + 1), HasSubgradientWithinAt (l t) (g t) (stdSimplex (d := d)) (w t))
    (h_mono : ∀ t ∈ Ico 1 (T + 1), α t ≤ α (t + 1)) :
    ∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t u) ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u)
      + ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) * subgradientMaxSq hd (g t) := by
  have hα1_nonneg : 0 ≤ α 1 := (hα_pos 1 (by simp)).le
  have h_opt : ∀ t ∈ Ico 1 (T + 1),
      0 ≤ optimality (fun s ↦ unnormEntropyFDeriv (α s)) w g t := by
    intro t ht
    have hw_t_pos : ∀ i, 0 < w t i := by
      rw [mem_Ico] at ht
      exact hw_pos t (by rw [mem_Ico]; omega)
    have hw_t : w t ∈ stdSimplex (d := d) := by
      rw [mem_Ico] at ht
      exact hw t (by rw [mem_Ico]; omega)
    have hw_succ : w (t + 1) ∈ stdSimplex (d := d) := by
      rw [mem_Ico] at ht
      exact hw (t + 1) (by rw [mem_Ico]; omega)
    have hα_t_nonneg : 0 ≤ α t :=
      (hα_pos t (by rw [mem_Ico] at ht ⊢; omega)).le
    exact optimality_nonneg_of_isMinOn (ψ := fun s ↦ unnormEntropy (α s))
      t (hasFDerivAt_unnormEntropy (α t) (w t) hw_t_pos)
      (convexOn_unnormEntropy_stdSimplex (d := d) (α t) hα_t_nonneg) hw_t hw_succ (h_min t ht)
  have h_term_opt : terminalOptimality (fun s ↦ unnormEntropy (α s)) u w g T ≤ 0 :=
    terminalOptimality_nonpos_of_isMinOn T hu h_term_min
  have h_decomp := regret_decomposition_eq (fun s ↦ unnormEntropyShifted (α s))
    (fun s ↦ unnormEntropyFDeriv (α s)) u w g l T
  have h_bound : boundary (fun s ↦ unnormEntropyShifted (α s)) u w T ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u) :=
    boundary_shifted_le_log_card α hα1_nonneg T hd u hu w hw1
  have h_opt_sum :
      0 ≤ ∑ t ∈ Ico 1 (T + 1), optimality (fun s ↦ unnormEntropyFDeriv (α s)) w g t :=
    sum_nonneg h_opt
  have h_lin_sum : ∑ t ∈ Ico 1 (T + 1), linearization u w g l t ≤ 0 := by
    refine sum_nonpos fun t ht ↦ ?_
    dsimp [linearization]
    have := (hg t ht) u hu
    linarith
  have h_stab :
      ∑ t ∈ Ico 1 (T + 1),
        stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
      ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) * subgradientMaxSq hd (g t) := by
    have h_stab_omd := sum_stability_le_max_sq hd (w := w) (g := g) α T
      (fun t ht ↦ hα_pos t (by rw [mem_Ico] at ht ⊢; omega)) hw hw_pos
    exact h_stab_omd
  have h_shift_le :
      ∑ t ∈ Ico 1 (T + 1), shift (fun s ↦ unnormEntropyShifted (α s)) w t ≤ 0 := by
    have h_shift_omd := sum_shift_unnormEntropyShifted_nonpos hd α w T h_mono hw
    exact h_shift_omd
  have h_term_shift :
      terminalOptimality (fun s ↦ unnormEntropyShifted (α s)) u w g T =
      terminalOptimality (fun s ↦ unnormEntropy (α s)) u w g T := by
    dsimp [terminalOptimality, F_obj, unnormEntropyShifted]
    ring
  rw [h_term_shift] at h_decomp
  linarith [h_decomp, h_bound, h_stab, h_opt_sum, h_lin_sum, h_shift_le, h_term_opt]

/-- Cumulative regret upper bound for FTRL Exponential Weights where the iterate sequence
$w_t = \mathrm{ftrlExpWeights}(hd, \alpha, g, t)$ is generated constructively by the update formula.
All iterate hypotheses (`hw1`, `hw`, `hw_pos`, `h_min`, `h_term_min`) are discharged
automatically. -/
theorem ftrlExpWeights_regret_bound (α : ℕ → ℝ) (T : ℕ)
    (hα_pos : ∀ t ∈ Ico 1 (T + 2), 0 < α t) (hd : 0 < d)
    (u : EuclideanSpace ℝ (Fin d)) (hu : u ∈ stdSimplex (d := d))
    (l : ℕ → EuclideanSpace ℝ (Fin d) → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ))
    (hg : ∀ t ∈ Ico 1 (T + 1),
      HasSubgradientWithinAt (l t) (g t) (stdSimplex (d := d)) (ftrlExpWeights hd α g t))
    (h_mono : ∀ t ∈ Ico 1 (T + 1), α t ≤ α (t + 1)) :
    ∑ t ∈ Ico 1 (T + 1), (l t (ftrlExpWeights hd α g t) - l t u) ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u)
      + ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) * subgradientMaxSq hd (g t) := by
  let w := ftrlExpWeights hd α g
  have hw1 : w 1 = uniformSimplex d := ftrlExpWeights_one hd α g
  have hw : ∀ t ∈ Ico 1 (T + 2), w t ∈ stdSimplex (d := d) := fun t _ ↦
    ftrlExpWeights_mem_stdSimplex hd α g t
  have hw_pos : ∀ t ∈ Ico 1 (T + 2), ∀ i, 0 < w t i := fun t _ i ↦
    ftrlExpWeights_pos hd α g t i
  have h_min : ∀ t ∈ Ico 1 (T + 1),
      IsMinOn (fun x ↦ unnormEntropy (α t) x + (∑ i ∈ Ico 1 t, g i) x)
        (stdSimplex (d := d)) (w t) := fun t ht ↦ by
    have hα_t_pos : 0 < α t := by
      rw [mem_Ico] at ht
      exact hα_pos t (by rw [mem_Ico]; omega)
    exact ftrlExpWeights_isMinOn hd α g t hα_t_pos
  have h_term_min : IsMinOn (F_obj (fun t x ↦ unnormEntropy (α t) x) g (T + 1))
      (stdSimplex (d := d)) (w (T + 1)) := by
    have hα_T1_pos : 0 < α (T + 1) := by
      exact hα_pos (T + 1) (by simp)
    exact ftrlExpWeights_terminal_isMinOn hd α g T hα_T1_pos
  exact regret_bound α T hα_pos hd u hu w hw1 hw hw_pos l g h_min h_term_min hg h_mono

/-- Cumulative regret upper bound for FTRL with non-negative loss subgradients $g_t \ge 0$,
where the stability term is bounded directly using the iterate $w_t$:
$$\sum_{t=1}^T (l_t(w_t) - l_t(u)) \le \alpha_1 \ln d + (\psi_{\alpha(T+1)}(u) - \psi_{\alpha 1}(u))
  + \sum_{t=1}^T \frac{1}{2\alpha_t} \sum_{i=1}^d w_{t, i} (g_t)_i^2.$$ -/
theorem regret_bound_wt (α : ℕ → ℝ) (T : ℕ) (hα_pos : ∀ t ∈ Ico 1 (T + 2), 0 < α t) (hd : 0 < d)
    (u : EuclideanSpace ℝ (Fin d)) (hu : u ∈ stdSimplex (d := d))
    (w : ℕ → EuclideanSpace ℝ (Fin d))
    (hw1 : w 1 = uniformSimplex d)
    (hw : ∀ t ∈ Ico 1 (T + 2), w t ∈ stdSimplex (d := d))
    (hw_pos : ∀ t ∈ Ico 1 (T + 2), ∀ i, 0 < w t i)
    (l : ℕ → EuclideanSpace ℝ (Fin d) → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ))
    (h_min : ∀ t ∈ Ico 1 (T + 1),
      IsMinOn (fun x ↦ unnormEntropy (α t) x + (∑ i ∈ Ico 1 t, g i) x)
        (stdSimplex (d := d)) (w t))
    (h_term_min : IsMinOn (F_obj (fun t x ↦ unnormEntropy (α t) x) g (T + 1))
      (stdSimplex (d := d)) (w (T + 1)))
    (hg : ∀ t ∈ Ico 1 (T + 1), HasSubgradientWithinAt (l t) (g t) (stdSimplex (d := d)) (w t))
    (hg_nonneg : ∀ t ∈ Ico 1 (T + 1),
      ∀ i, 0 ≤ g t (EuclideanSpace.basisFun (Fin d) ℝ i))
    (h_mono : ∀ t ∈ Ico 1 (T + 1), α t ≤ α (t + 1)) :
    ∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t u) ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u)
      + ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) *
        ∑ i, (w t : EuclideanSpace ℝ (Fin d)) i *
          (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
  have hα1_nonneg : 0 ≤ α 1 := (hα_pos 1 (by simp)).le
  have h_opt : ∀ t ∈ Ico 1 (T + 1),
      0 ≤ optimality (fun s ↦ unnormEntropyFDeriv (α s)) w g t := by
    intro t ht
    have hw_t_pos : ∀ i, 0 < w t i := by
      rw [mem_Ico] at ht
      exact hw_pos t (by rw [mem_Ico]; omega)
    have hw_t : w t ∈ stdSimplex (d := d) := by
      rw [mem_Ico] at ht
      exact hw t (by rw [mem_Ico]; omega)
    have hw_succ : w (t + 1) ∈ stdSimplex (d := d) := by
      rw [mem_Ico] at ht
      exact hw (t + 1) (by rw [mem_Ico]; omega)
    have hα_t_nonneg : 0 ≤ α t :=
      (hα_pos t (by rw [mem_Ico] at ht ⊢; omega)).le
    exact optimality_nonneg_of_isMinOn (ψ := fun s ↦ unnormEntropy (α s))
      t (hasFDerivAt_unnormEntropy (α t) (w t) hw_t_pos)
      (convexOn_unnormEntropy_stdSimplex (d := d) (α t) hα_t_nonneg) hw_t hw_succ (h_min t ht)
  have h_term_opt : terminalOptimality (fun s ↦ unnormEntropy (α s)) u w g T ≤ 0 :=
    terminalOptimality_nonpos_of_isMinOn T hu h_term_min
  have h_decomp := regret_decomposition_eq (fun s ↦ unnormEntropyShifted (α s))
    (fun s ↦ unnormEntropyFDeriv (α s)) u w g l T
  have h_bound : boundary (fun s ↦ unnormEntropyShifted (α s)) u w T ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u) :=
    boundary_shifted_le_log_card α hα1_nonneg T hd u hu w hw1
  have h_opt_sum :
      0 ≤ ∑ t ∈ Ico 1 (T + 1), optimality (fun s ↦ unnormEntropyFDeriv (α s)) w g t :=
    sum_nonneg h_opt
  have h_lin_sum : ∑ t ∈ Ico 1 (T + 1), linearization u w g l t ≤ 0 := by
    refine sum_nonpos fun t ht ↦ ?_
    dsimp [linearization]
    have := (hg t ht) u hu
    linarith
  have h_stab :
      ∑ t ∈ Ico 1 (T + 1),
        stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
      ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) *
        ∑ i, (w t : EuclideanSpace ℝ (Fin d)) i *
          (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
    have h_stab_omd := sum_stability_le_dual_norm_wt (w := w) (g := g) α T
      (fun t ht ↦ hα_pos t (by rw [mem_Ico] at ht ⊢; omega)) hw_pos hg_nonneg
    exact h_stab_omd
  have h_shift_le :
      ∑ t ∈ Ico 1 (T + 1), shift (fun s ↦ unnormEntropyShifted (α s)) w t ≤ 0 := by
    have h_shift_omd := sum_shift_unnormEntropyShifted_nonpos hd α w T h_mono hw
    exact h_shift_omd
  have h_term_shift :
      terminalOptimality (fun s ↦ unnormEntropyShifted (α s)) u w g T =
      terminalOptimality (fun s ↦ unnormEntropy (α s)) u w g T := by
    dsimp [terminalOptimality, F_obj, unnormEntropyShifted]
    ring
  rw [h_term_shift] at h_decomp
  linarith [h_decomp, h_bound, h_stab, h_opt_sum, h_lin_sum, h_shift_le, h_term_opt]

/-- Cumulative regret upper bound for FTRL Exponential Weights with non-negative losses
$g_t \ge 0$, where the stability term is bounded directly with $w_t$. All iterate hypotheses
(`hw1`, `hw`, `hw_pos`, `h_min`, `h_term_min`) are discharged automatically. -/
theorem ftrlExpWeights_regret_bound_wt (α : ℕ → ℝ) (T : ℕ)
    (hα_pos : ∀ t ∈ Ico 1 (T + 2), 0 < α t) (hd : 0 < d)
    (u : EuclideanSpace ℝ (Fin d)) (hu : u ∈ stdSimplex (d := d))
    (l : ℕ → EuclideanSpace ℝ (Fin d) → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ))
    (hg : ∀ t ∈ Ico 1 (T + 1),
      HasSubgradientWithinAt (l t) (g t) (stdSimplex (d := d)) (ftrlExpWeights hd α g t))
    (hg_nonneg : ∀ t ∈ Ico 1 (T + 1),
      ∀ i, 0 ≤ g t (EuclideanSpace.basisFun (Fin d) ℝ i))
    (h_mono : ∀ t ∈ Ico 1 (T + 1), α t ≤ α (t + 1)) :
    ∑ t ∈ Ico 1 (T + 1), (l t (ftrlExpWeights hd α g t) - l t u) ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u)
      + ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) *
        ∑ i, (ftrlExpWeights hd α g t i) *
          (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
  let w := ftrlExpWeights hd α g
  have hw1 : w 1 = uniformSimplex d := ftrlExpWeights_one hd α g
  have hw : ∀ t ∈ Ico 1 (T + 2), w t ∈ stdSimplex (d := d) := fun t _ ↦
    ftrlExpWeights_mem_stdSimplex hd α g t
  have hw_pos : ∀ t ∈ Ico 1 (T + 2), ∀ i, 0 < w t i := fun t _ i ↦
    ftrlExpWeights_pos hd α g t i
  have h_min : ∀ t ∈ Ico 1 (T + 1),
      IsMinOn (fun x ↦ unnormEntropy (α t) x + (∑ i ∈ Ico 1 t, g i) x)
        (stdSimplex (d := d)) (w t) := fun t ht ↦ by
    have hα_t_pos : 0 < α t := by
      rw [mem_Ico] at ht
      exact hα_pos t (by rw [mem_Ico]; omega)
    exact ftrlExpWeights_isMinOn hd α g t hα_t_pos
  have h_term_min : IsMinOn (F_obj (fun t x ↦ unnormEntropy (α t) x) g (T + 1))
      (stdSimplex (d := d)) (w (T + 1)) := by
    have hα_T1_pos : 0 < α (T + 1) := by
      exact hα_pos (T + 1) (by simp)
    exact ftrlExpWeights_terminal_isMinOn hd α g T hα_T1_pos
  exact regret_bound_wt α T hα_pos hd u hu w hw1 hw hw_pos l g h_min h_term_min hg hg_nonneg h_mono

end Online.OCO.LEA.FTRL
