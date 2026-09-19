/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Data.Finset.Interval
public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Bregman.Basic

/-!
# Strong Optimistic Centered Online Mirror Descent Regret Decomposition

This file establishes the exact multi-round algebraic regret identity for Optimistic
Centered Online Mirror Descent (OMD / COMD) with regularizer shift on the anchor $\tilde{w}_t$,
arbitrary learning rate $\eta > 0$, hint functional $h_t$, and continuous anchor adjustment.

## Mathematical Overview & The $t$ Indexing Scheme

In discrete-time online convex optimization, at each round $t \ge 1$:
1. The learner predicts action $v_t$ (from hint $h_t$ and anchor $\tilde{w}_t$).
2. The environment reveals a convex loss function $l_t : E \to \mathbb{R}$ with
   subgradient $g_t \in E \to+ \mathbb{R}$.
3. The learner updates the unadjusted iterate $w_{t+1}$ using step regularizer $\psi_{t+1}$
   and centering potential $\varphi_t$.
4. The learner can apply an adjustment / projection step yielding the next anchor
   $\tilde{w}_{t+1}$.

### The Summation Index: $\sum_{t=1}^T$ (`Finset.Ico 1 (T + 1)`)
The regret is accumulated over rounds $t \in [1, T]$, represented in Lean as
`Finset.Ico 1 (T + 1)`.
- Round $t$ accesses loss $l_t$, linear loss gradient $g_t$, hint $h_t$, and centering $\varphi_t$.
- The sequence of comparators is $u_t$ (for $t \ge 0$), where $u_T$ is the final
  comparator and $u_0$ is the initial comparator.
- State transitions advance from $(w_t, \tilde{w}_t)$ at round $t$ to
  $(w_{t+1}, \tilde{w}_{t+1})$ at round $t+1$.

## Main Theorems

* `Analysis.Convex.strongOptimisticMirrorDescent`:
  The exact multi-round algebraic decomposition of optimistic mirror descent regret into:
  - Initial vs final Bregman boundary divergence:
    $D_{\psi_1}(u_0, \tilde{w}_1) - D_{\psi_{T+1}}(u_T, \tilde{w}_{T+1})$
  - Telescoped regularizer drift on comparator: $\psi_{T+1}(u_T) - \psi_1(u_0)$
  - Adjustment term: $\sum \xi_t$
  - Regularizer shift on anchor: $\sum \Delta_t(\tilde{w}_t)$
  - Pure linear path-length coupling: $\sum P_t$
  - Optimistic stability: $\sum \delta_t^{\mathrm{opt}}$
  - Prediction penalty: $-\sum D_{\psi_{t+1}}(v_t, \tilde{w}_t)$
  - Optimistic optimality deficit: $-\sum \mu_t^{\mathrm{opt}}$
  - Centering surplus: $\sum C_t$
  - Linearization error: $\eta \sum (- D_{l_t}(u_t, w_{t+1}, g_t))$
-/

open scoped BigOperators Bregman
open Finset

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [AddCommGroup E]

/-! ### Auxiliary Term Definitions -/

section AuxTerms

variable (η : ℝ)
variable (ψ φ : ℕ → E → ℝ)
variable (gψ gφ : ℕ → E → (E →+ ℝ))
variable (u w w_tilde v : ℕ → E)
variable (g h : ℕ → (E →+ ℝ))
variable (l : ℕ → E → ℝ)

/-- Linearization error term: $- D_{l_t}(u_t, w_{t+1}, g_t)$. -/
def linearizationTerm (t : ℕ) : ℝ :=
  - D_[l t](u t, w (t + 1), g t)

/-- Centering surplus term:
$C_t = \varphi_t(u_t) - \varphi_t(w_{t+1}) -
  D_{\varphi_t}(u_t, w_{t+1}, \nabla \varphi_t(w_{t+1}))$. -/
def centeringTerm (t : ℕ) : ℝ :=
  φ t (u t) - φ t (w (t + 1)) - D_[φ t](u t, w (t + 1), gφ t (w (t + 1)))

/-- Adjustment drift term:
$\xi_t = D_{\psi_{t+1}}(u_t, \tilde{w}_{t+1}, \nabla \psi_{t+1}(\tilde{w}_{t+1}))
  - D_{\psi_{t+1}}(u_t, w_{t+1}, \nabla \psi_{t+1}(w_{t+1}))$. -/
def adjustmentTerm (t : ℕ) : ℝ :=
  D_[ψ (t+1)](u t, w_tilde (t + 1), gψ (t+1) (w_tilde (t + 1)))
  - D_[ψ (t+1)](u t, w (t + 1), gψ (t+1) (w (t + 1)))

/-- Cross-round regularizer shift evaluated on anchor $\tilde{w}_t$:
$-(\psi_{t+1} - \psi_t)(\tilde{w}_t)$. -/
def shiftTerm (t : ℕ) : ℝ :=
  - (ψ (t + 1) - ψ t) (w_tilde t)

/-- Pure linear path-length coupling term:
$P_t = \langle \nabla \psi_t(\tilde{w}_t), u_{t-1} - u_t \rangle$. -/
def pathlengthTerm (t : ℕ) : ℝ :=
  gψ t (w_tilde t) (u (t - 1) - u t)

/-- General first-order optimality deficit:
$\langle \eta g + g\varphi + g\psi_{\text{next}} - g\psi_{\text{prev}}, u - w \rangle$. -/
def firstOrderOptimalityTerm (η : ℝ) (g gφ gψ_next gψ_prev : E →+ ℝ) (u w : E) : ℝ :=
  (η • g + gφ + gψ_next - gψ_prev) (u - w)

/-- Optimality deficit for anchor update under step regularizer $\psi_{t+1}$:
$(\eta g_t + \nabla \varphi_t(w_{t+1}) + \nabla \psi_{t+1}(w_{t+1}) - \nabla \psi_t(\tilde{w}_t))
  (u_t - w_{t+1})$. -/
def optimalityTerm (t : ℕ) : ℝ :=
  firstOrderOptimalityTerm η (g t) (gφ t (w (t + 1))) (gψ (t + 1) (w (t + 1)))
    (gψ t (w_tilde t)) (u t) (w (t + 1))

/-- Optimality deficit for hint prediction update:
$(\eta h_t + \nabla \psi_{t+1}(v_t) - \nabla \psi_t(\tilde{w}_t))(w_{t+1} - v_t)$. -/
def optimisticOptimalityTerm (t : ℕ) : ℝ :=
  firstOrderOptimalityTerm η (h t) 0 (gψ (t + 1) (v t)) (gψ t (w_tilde t))
    (w (t + 1)) (v t)

/-- Optimistic stability term with step regularizer $\psi_{t+1}$ at $v_t$. -/
def optimisticStabilityTerm (t : ℕ) : ℝ :=
  η * (l t (v t) - l t (w (t + 1)) - h t (v t - w (t + 1))) -
    D_[ψ (t + 1)](w (t + 1), v t, gψ (t + 1) (v t))

/-! ### Strong Optimistic Regret Decomposition -/

/-- Strong Optimistic Centered Online Mirror Descent Regret Decomposition Identity.
Exact algebraic equality holding for all rounds $T \ge 0$. -/
theorem strongOptimisticMirrorDescent (T : ℕ) :
    η * (∑ t ∈ Ico 1 (T + 1), (l t (v t) - l t (u t))) =
    D_[ψ 1](u 0, w_tilde 1, gψ 1 (w_tilde 1))
    - D_[ψ (T+1)](u T, w_tilde (T + 1), gψ (T+1) (w_tilde (T + 1)))
    + ψ (T + 1) (u T) - ψ 1 (u 0)
    + (∑ t ∈ Ico 1 (T + 1), adjustmentTerm ψ gψ u w w_tilde t)
    + (∑ t ∈ Ico 1 (T + 1), shiftTerm ψ w_tilde t)
    + (∑ t ∈ Ico 1 (T + 1), pathlengthTerm gψ u w_tilde t)
    + (∑ t ∈ Ico 1 (T + 1), optimisticStabilityTerm η ψ gψ w v h l t)
    - (∑ t ∈ Ico 1 (T + 1), D_[ψ (t + 1)](v t, w_tilde t, gψ t (w_tilde t)))
    - (∑ t ∈ Ico 1 (T + 1), optimalityTerm η gψ gφ u w w_tilde g t)
    - (∑ t ∈ Ico 1 (T + 1), optimisticOptimalityTerm η gψ w w_tilde v h t)
    + (∑ t ∈ Ico 1 (T + 1), centeringTerm φ gφ u w t)
    + η * (∑ t ∈ Ico 1 (T + 1), linearizationTerm u w g l t) := by
  induction T with
  | zero => simp
  | succ T ih =>
    simp_rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ T + 1), mul_add, ih]
    dsimp [optimalityTerm, optimisticOptimalityTerm, firstOrderOptimalityTerm,
      adjustmentTerm, shiftTerm, pathlengthTerm, optimisticStabilityTerm, centeringTerm,
      linearizationTerm, Analysis.Convex.bregDiv]
    simp only [map_sub]
    ring

end AuxTerms

end Analysis.Convex
