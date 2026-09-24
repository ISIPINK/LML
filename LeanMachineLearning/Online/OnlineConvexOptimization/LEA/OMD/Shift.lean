/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Algebra.BigOperators.Intervals
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.Normed.Module.Basic
public import Mathlib.Data.Finset.Interval
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.RegretDecomposition

/-!
# Shift Term Bound for Online Mirror Descent (LEA Specialization)

This file bounds the regularizer potential shift term:
$$\mathrm{shift}(t) = -(\psi_{t+1} - \psi_t)(w_t) = (\psi_t - \psi_{t+1})(w_t)$$
when the time-dependent regularizers are scaled versions $\psi_t = \alpha_t \psi_0$ of a base
regularizer $\psi_0$, with non-increasing regularization weights $\alpha_{t+1} \le \alpha_t$
(i.e. regularizers become looser over time).

## Main results

* `Online.OCO.LEA.OMD.shift_scaled`: For $\psi_t = \alpha_t \psi_0$,
  $$\mathrm{shift}(t) = (\alpha_t - \alpha_{t+1}) \psi_0(w_t).$$
* `Online.OCO.LEA.OMD.sum_shift_telescope_le`: When $\alpha_{t+1} \le \alpha_t$ and
  $\psi_0(w_t) \le M$ along the iterates, the cumulative shift telescopes:
  $$\sum_{t=1}^T \mathrm{shift}(t) \le (\alpha_1 - \alpha_{T+1}) M.$$
-/

open scoped BigOperators
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {E : Type*}

/-- The shift term for a scaled regularizer sequence $\psi_t = \alpha_t \psi_0$ is given by
$$\mathrm{shift}(t) = (\alpha_t - \alpha_{t+1}) \psi_0(w_t).$$ -/
theorem shift_scaled (α : ℕ → ℝ) (ψ₀ : E → ℝ) (w : ℕ → E) (t : ℕ) :
    shift (fun s ↦ α s • ψ₀) w t = (α t - α (t + 1)) * ψ₀ (w t) := by
  dsimp [shift]
  ring

/-- Telescopic upper bound on the cumulative shift term:
when the regularization weights are non-increasing ($\alpha_{t+1} \le \alpha_t$) and the base
regularizer is bounded above by $M$ on the iterates ($\psi_0(w_t) \le M$),
the total shift over $T$ rounds satisfies:
$$\sum_{t=1}^T \mathrm{shift}(t) \le (\alpha_1 - \alpha_{T+1}) M.$$ -/
theorem sum_shift_telescope_le (α : ℕ → ℝ) (ψ₀ : E → ℝ) (w : ℕ → E) (M : ℝ) (T : ℕ)
    (h_mono : ∀ t ∈ Ico 1 (T + 1), α (t + 1) ≤ α t)
    (h_bound : ∀ t ∈ Ico 1 (T + 1), ψ₀ (w t) ≤ M) :
    ∑ t ∈ Ico 1 (T + 1), shift (fun s ↦ α s • ψ₀) w t ≤ (α 1 - α (T + 1)) * M := by
  have h_shift_le : ∀ t ∈ Ico 1 (T + 1),
      shift (fun s ↦ α s • ψ₀) w t ≤ (α t - α (t + 1)) * M := by
    intro t ht
    rw [shift_scaled]
    have h_diff_nonneg : 0 ≤ α t - α (t + 1) := sub_nonneg.mpr (h_mono t ht)
    exact mul_le_mul_of_nonneg_left (h_bound t ht) h_diff_nonneg
  have h_sum_le := sum_le_sum h_shift_le
  refine h_sum_le.trans ?_
  rw [← sum_mul]
  have h_tele : (∑ t ∈ Ico 1 (T + 1), (α t - α (t + 1))) = α 1 - α (T + 1) := by
    have h_sub := sum_Ico_sub α (by omega : 1 ≤ T + 1)
    have h_neg : (∑ t ∈ Ico 1 (T + 1), (α t - α (t + 1))) =
        - ∑ t ∈ Ico 1 (T + 1), (α (t + 1) - α t) := by
      rw [← sum_neg_distrib]
      refine sum_congr rfl fun t _ ↦ by ring
    rw [h_neg, h_sub]
    ring
  rw [h_tele]

end Online.OCO.LEA.OMD
