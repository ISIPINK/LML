/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.Normed.Module.Basic
public import Mathlib.Analysis.Normed.Operator.ContinuousLinearMap
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Data.Finset.Interval
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Bregman.Basic

/-!
# Master Regret Decomposition for Centered Online Mirror Descent (COMD2)

This file establishes the exact multi-round algebraic regret decomposition for Centered
Online Mirror Descent (version 2) with pure potential shifts and direct regularizer updates.

The regret decomposition holds for any sequence of comparators `u_t` and any sequence
of centering potentials `φ_t`. Specializations (e.g. static regret `u_t = u`, or uncentered
settings `φ_t = 0`) are derived as corollaries in separate modules.

## Main definitions
* `linearization`
* `shift`
* `pathlength`
* `optimality`
* `centering`
* `adjustment`
* `stability`

## Main results
* `regretDecomposition`: The master multi-round algebraic regret equality holding for all $T \ge 0$.
-/

open scoped BigOperators Bregman
open Finset

@[expose] public section

namespace OnlineConvexOptimization.COMD2

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-! ### Auxiliary Term Definitions -/

section AuxTerms

variable (η : ℝ)
variable (ψ φ : ℕ → E → ℝ)
variable (gψ gφ : ℕ → E → (E →L[ℝ] ℝ))
variable (u w w_tilde : ℕ → E)
variable (g : ℕ → (E →L[ℝ] ℝ))
variable (l : ℕ → E → ℝ)

/-- Error incurred by replacing the true convex loss with its linear approximation
at the update iterate using the subgradient `g_t`. Non-positive when `l_t` is convex. -/
def linearization (t : ℕ) : ℝ :=
  - D_[l t](u t, w (t + 1), g t)

/-- Potential drift accounting for changing the regularizer `ψ_t` to `ψ_{t+1}`
across consecutive rounds, evaluated at the current anchor `w̃_t`. -/
def shift (t : ℕ) : ℝ :=
  - (ψ (t + 1) - ψ t) (w_tilde t)

/-- Linear coupling penalty measuring the variation in the comparator sequence `u_t`.
Vanishes identically for a static comparator (`u_t = u`). -/
def pathlength (t : ℕ) : ℝ :=
  gψ t (w_tilde t) (u (t - 1) - u t)

/-- First-order optimality deficit measuring how close `w_{t+1}` is to the exact
minimizer of the mirror descent step. Non-positive when `w_{t+1}` satisfies the
first-order optimality condition over the constraint set. -/
def optimality (t : ℕ) : ℝ :=
  ((η : ℝ) • g t + gφ t (w (t + 1)) + gψ (t + 1) (w (t + 1)) - gψ t (w_tilde t)) (u t - w (t + 1))

/-- Surplus contribution induced by the centering potential `φ_t`.
Zero when no centering is used (`φ_t = 0`). -/
def centering (t : ℕ) : ℝ :=
  φ t (u t) - φ t (w (t + 1)) - D_[φ t](u t, w (t + 1), gφ t (w (t + 1)))

/-- Divergence drift capturing arbitrary post-processing or state adjustments
from iterate `w_{t+1}` to anchor `w̃_{t+1}`. Vanishes for unadjusted or projection-free
updates (`w̃_{t+1} = w_{t+1}`), and yields small controllable penalties for schemes
like Fixed Share. -/
def adjustment (t : ℕ) : ℝ :=
  D_[ψ (t+1)](u t, w_tilde (t + 1), gψ (t+1) (w_tilde (t + 1)))
  - D_[ψ (t+1)](u t, w (t + 1), gψ (t+1) (w (t + 1)))

/-- One-round stability penalty balancing the movement of the loss `l_t` between
the anchor `w̃_t` and iterate `w_{t+1}` against the divergence regularizer step. -/
def stability (t : ℕ) : ℝ :=
  η * (l t (w_tilde t) - l t (w (t + 1))) - D_[ψ (t + 1)](w (t + 1), w_tilde t, gψ t (w_tilde t))

/-- Master Centered Online Mirror Descent (version 2) Regret Decomposition Identity.
Exact multi-round algebraic equality holding for all rounds $T \ge 0$. -/
theorem regretDecomposition (T : ℕ) :
    η * (∑ t ∈ Ico 1 (T + 1), (l t (w_tilde t) - l t (u t))) =
    D_[ψ 1](u 0, w_tilde 1, gψ 1 (w_tilde 1))
    - D_[ψ (T+1)](u T, w_tilde (T + 1), gψ (T+1) (w_tilde (T + 1)))
    + ψ (T + 1) (u T) - ψ 1 (u 0)
    + (∑ t ∈ Ico 1 (T + 1), adjustment ψ gψ u w w_tilde t)
    + (∑ t ∈ Ico 1 (T + 1), shift ψ w_tilde t)
    + (∑ t ∈ Ico 1 (T + 1), pathlength gψ u w_tilde t)
    + (∑ t ∈ Ico 1 (T + 1), stability η ψ gψ w w_tilde l t)
    - (∑ t ∈ Ico 1 (T + 1), optimality η gψ gφ u w w_tilde g t)
    + (∑ t ∈ Ico 1 (T + 1), centering φ gφ u w t)
    + η * (∑ t ∈ Ico 1 (T + 1), linearization u w g l t) := by
  induction T with
  | zero => simp
  | succ T ih =>
    simp_rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ T + 1), mul_add, ih]
    dsimp [optimality, adjustment, shift, pathlength,
      stability, centering, linearization, bregDiv]
    simp only [map_sub, add_apply, sub_apply, smul_apply, smul_eq_mul]
    ring

end AuxTerms

end OnlineConvexOptimization.COMD2
