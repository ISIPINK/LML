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
# Static Regret Decomposition for Online Mirror Descent (LEA Specialization)

This file defines the direct, algebraic regret decomposition for Online Mirror Descent
tailored to the Learning with Expert Advice (LEA) setting:
* **Static comparator**: $u_t = u$ for all rounds $t$.
* **No centering**: $\varphi_t = 0$.
* **No state adjustment**: $\tilde{w}_t = w_t$.

Under these conditions, the regret is governed by the boundary divergence & potential drift,
and four per-round regret terms:
1. `boundary`: Initial divergence minus final divergence plus regularizer drift at $u$.
2. `shift`: Cross-round regularizer potential drift on iterates $w_t$.
3. `stability`: Movement of the loss balanced against regularizer divergence.
4. `optimality`: First-order optimality deficit of the update step.
5. `linearization`: Loss linearization error via subgradients.

## Main definitions
* `Online.OCO.LEA.OMD.boundary`
* `Online.OCO.LEA.OMD.linearization`
* `Online.OCO.LEA.OMD.shift`
* `Online.OCO.LEA.OMD.optimality`
* `Online.OCO.LEA.OMD.stability`

## Main results
* `Online.OCO.LEA.OMD.regretDecomposition`: The algebraic multi-round regret equality.
-/

open scoped BigOperators Bregman
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

section Terms

variable (η : ℝ)
variable (ψ : ℕ → E → ℝ)
variable (gψ : ℕ → E → (E →L[ℝ] ℝ))
variable (u : E)
variable (w : ℕ → E)
variable (g : ℕ → (E →L[ℝ] ℝ))
variable (l : ℕ → E → ℝ)

/-- Boundary term comprising initial and final Bregman divergences and regularizer drift at $u$. -/
def boundary (T : ℕ) : ℝ :=
  D_[ψ 1](u, w 1, gψ 1 (w 1))
  - D_[ψ (T + 1)](u, w (T + 1), gψ (T + 1) (w (T + 1)))
  + ψ (T + 1) u - ψ 1 u

/-- Error incurred by linearizing the loss with subgradient `g_t`. -/
def linearization (t : ℕ) : ℝ :=
  - D_[l t](u, w (t + 1), g t)

/-- Potential shift accounting for changes in regularizers across rounds. -/
def shift (t : ℕ) : ℝ :=
  - (ψ (t + 1) - ψ t) (w t)

/-- First-order optimality deficit of the update $w_{t+1}$. -/
def optimality (t : ℕ) : ℝ :=
  ((η : ℝ) • g t + gψ (t + 1) (w (t + 1)) - gψ t (w t)) (u - w (t + 1))

/-- Stability tradeoff between loss reduction and regularizer distance. -/
def stability (t : ℕ) : ℝ :=
  η * (l t (w t) - l t (w (t + 1))) - D_[ψ (t + 1)](w (t + 1), w t, gψ t (w t))

/-- Exact multi-round algebraic regret decomposition for static, uncentered, unadjusted OMD. -/
theorem regretDecomposition (T : ℕ) :
    η * (∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t u)) =
    boundary ψ gψ u w T
    + (∑ t ∈ Ico 1 (T + 1), shift ψ w t)
    + (∑ t ∈ Ico 1 (T + 1), stability η ψ gψ w l t)
    - (∑ t ∈ Ico 1 (T + 1), optimality η gψ u w g t)
    + η * (∑ t ∈ Ico 1 (T + 1), linearization u w g l t) := by
  induction T with
  | zero =>
    simp [boundary]
  | succ T ih =>
    simp_rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ T + 1), mul_add, ih]
    dsimp [boundary, optimality, shift, stability, linearization, bregDiv]
    simp only [map_sub, add_apply, sub_apply, smul_apply, smul_eq_mul]
    ring

end Terms

end Online.OCO.LEA.OMD
