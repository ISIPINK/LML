/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.Normed.Module.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Data.Finset.Interval

/-!
# Master Regret Decomposition for Follow-the-Regularized-Leader (FTRL)

This file establishes the exact multi-round algebraic regret decomposition for
Follow-the-Regularized-Leader (FTRL), corresponding to Lemma 7.1 in Francesco Orabona's
*A Modern Introduction to Online Learning* (v10).

## Main definitions
* `F_obj`
* `boundary`
* `stability`
* `terminalOptimality`

## Main results
* `regret_decomposition_eq`: The master algebraic regret equality holding for all $T \ge 0$.
-/

open scoped BigOperators
open Finset

@[expose] public section

namespace OnlineConvexOptimization.FTRL

variable {E : Type*}

section Terms

variable (ψ : ℕ → E → ℝ)
variable (u : E)
variable (w : ℕ → E)
variable (l : ℕ → E → ℝ)

/--
The cumulative objective function $F_t(y) = \psi_t(y) + \sum_{i=1}^{t-1} l_i(y)$.
-/
def F_obj (t : ℕ) (y : E) : ℝ :=
  ψ t y + ∑ i ∈ Ico 1 t, l i y

/-- Boundary regularizer term: $\psi_{T+1}(u) - F_1(w_1)$. -/
def boundary (T : ℕ) : ℝ :=
  ψ (T + 1) u - F_obj ψ l 1 (w 1)

/-- One-round stability penalty measuring the advance of $F_t + l_t$ between $w_t$ and $w_{t+1}$. -/
def stability (t : ℕ) : ℝ :=
  F_obj ψ l t (w t) - F_obj ψ l (t + 1) (w (t + 1)) + l t (w t)

/-- Terminal optimality deficit of $w_{T+1}$ relative to comparator $u$. -/
def terminalOptimality (T : ℕ) : ℝ :=
  F_obj ψ l (T + 1) (w (T + 1)) - F_obj ψ l (T + 1) u

/-- Master Algebraic Regret Decomposition Identity for FTRL (Orabona, Lemma 7.1). -/
theorem regret_decomposition_eq (T : ℕ) :
    (∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t u)) =
    boundary ψ u w l T
    + (∑ t ∈ Ico 1 (T + 1), stability ψ w l t)
    + terminalOptimality ψ u w l T := by
  induction T with
  | zero =>
    simp [boundary, terminalOptimality, F_obj]
  | succ T ih =>
    simp_rw [sum_Ico_succ_top (by omega : 1 ≤ T + 1), ih]
    dsimp [boundary, terminalOptimality, stability, F_obj]
    simp_rw [sum_Ico_succ_top (by omega : 1 ≤ T + 1)]
    ring

end Terms

end OnlineConvexOptimization.FTRL
