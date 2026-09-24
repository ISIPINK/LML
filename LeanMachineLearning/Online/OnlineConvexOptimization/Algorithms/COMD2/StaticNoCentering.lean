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
# Static Regret Decomposition without Centering and without Adjustments

This file gives the classic static regret decomposition for Online Mirror Descent:
1. **Static comparator**: $u_t = u$ for all rounds $t$.
2. **No centering**: $\varphi_t = 0$.
3. **No adjustments**: $\tilde{w}_t = w_t$ (the iterate is used directly as the prediction).

Under these standard conditions, the pathlength, centering, and adjustment terms all vanish,
leaving only the boundary Bregman divergences, regularizer potential drift, stability,
optimality, and loss linearization.

## Main definitions
* `linearization`
* `shift`
* `optimality`
* `stability`

## Main results
* `regretDecomposition`: The classic static regret decomposition holding for any fixed $u$.
-/

open scoped BigOperators Bregman
open Finset

@[expose] public section

namespace OnlineConvexOptimization.COMD2.Static.NoCentering

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

section StaticTerms

variable (η : ℝ)
variable (ψ : ℕ → E → ℝ)
variable (gψ : ℕ → E → (E →L[ℝ] ℝ))
variable (u : E)
variable (w : ℕ → E)
variable (g : ℕ → (E →L[ℝ] ℝ))
variable (l : ℕ → E → ℝ)

/-- Loss linearization error: $- D_{l_t}(u, w_{t+1}, g_t)$. -/
def linearization (t : ℕ) : ℝ :=
  - D_[l t](u, w (t + 1), g t)

/-- Potential shift from changing regularizers across rounds evaluated at $w_t$. -/
def shift (t : ℕ) : ℝ :=
  - (ψ (t + 1) - ψ t) (w t)

/-- First-order optimality deficit:
$(\eta g_t + \nabla \psi_{t+1}(w_{t+1}) - \nabla \psi_t(w_t))(u - w_{t+1})$. -/
def optimality (t : ℕ) : ℝ :=
  ((η : ℝ) • g t + gψ (t + 1) (w (t + 1)) - gψ t (w t)) (u - w (t + 1))

/-- Stability penalty between current state $w_t$ and update $w_{t+1}$. -/
def stability (t : ℕ) : ℝ :=
  η * (l t (w t) - l t (w (t + 1))) - D_[ψ (t + 1)](w (t + 1), w t, gψ t (w t))

/-- Static Regret Decomposition without Centering or Adjustments. -/
theorem regretDecomposition (T : ℕ) :
    η * (∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t u)) =
    D_[ψ 1](u, w 1, gψ 1 (w 1))
    - D_[ψ (T+1)](u, w (T + 1), gψ (T+1) (w (T + 1)))
    + ψ (T + 1) u - ψ 1 u
    + (∑ t ∈ Ico 1 (T + 1), shift ψ w t)
    + (∑ t ∈ Ico 1 (T + 1), stability η ψ gψ w l t)
    - (∑ t ∈ Ico 1 (T + 1), optimality η gψ u w g t)
    + η * (∑ t ∈ Ico 1 (T + 1), linearization u w g l t) := by
  induction T with
  | zero => simp
  | succ T ih =>
    simp_rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ T + 1), mul_add, ih]
    dsimp [optimality, shift, stability, linearization, bregDiv]
    simp only [map_sub, add_apply, sub_apply, smul_apply, smul_eq_mul]
    ring

end StaticTerms

end OnlineConvexOptimization.COMD2.Static.NoCentering
