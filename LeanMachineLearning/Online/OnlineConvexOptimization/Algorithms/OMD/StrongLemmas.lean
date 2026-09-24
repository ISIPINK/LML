/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.Normed.Module.Basic
public import Mathlib.Analysis.Normed.Operator.ContinuousLinearMap
public import Mathlib.Data.Real.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Data.Finset.Interval
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Bregman.Basic

/-!
# Strong Centered Online Mirror Descent Regret Decomposition (Jacobsen Formulation)

This file establishes the exact multi-round algebraic regret decomposition for Centered
Online Mirror Descent (COMD) following the strong lemma in Jacobsen's thesis, with minor
notational and indexing adaptations.

Other strong lemma formulations (such as those incorporating optimistic prediction hints
$h_t$ or alternative anchor shift choices) can be derived analogously, but we formalize
this centered version as the primary example.

## Mathematical Overview & The $t$ Indexing Scheme

In discrete-time online convex optimization, at each round $t \ge 1$:
1. The learner maintains anchor $\tilde{w}_t \in \mathcal{X}$ and initial anchor $\tilde{w}_1$.
2. The environment reveals a convex loss function $l_t : E \to \mathbb{R}$ with
   subgradient $g_t \in E \toL[\mathbb{R}] \mathbb{R}$.
3. The learner computes iterate $w_{t+1}$ using step regularizer $\psi_{t+1}$ and
   centering potential $\varphi_t$.
4. The learner applies an adjustment step yielding next anchor $\tilde{w}_{t+1}$.

### Regret Decomposition Components
Over rounds $t \in [1, T]$ (`Finset.Ico 1 (T + 1)`):
- Boundary divergence: $D_{\psi_{T+1}}(u_T, \tilde{w}_1) - D_{\psi_{T+1}}(u_T, \tilde{w}_{T+1})$
- Adjustment drift: $\sum_{t=1}^T \xi_t$
- Centered pathlength:
  $\sum_{t=1}^T P_t = \sum_{t=1}^T \langle \nabla \psi_t(\tilde{w}_t) - \nabla \psi_t(\tilde{w}_1),
    u_{t-1} - u_t \rangle$
- Stability term: $\sum_{t=1}^T \delta_t$
- Regularizer shift on iterate: $-\sum_{t=1}^T D_{\psi_{t+1} - \psi_t}(w_{t+1}, \tilde{w}_1)$
- Optimality deficit: $-\sum_{t=1}^T \mu_t$
- Centering surplus: $\sum_{t=1}^T C_t$
- Linearization error: $\eta \sum_{t=1}^T (- D_{l_t}(u_t, w_{t+1}, g_t))$

Note on comparator indexing: the comparator sequence is indexed as $u_0, u_1, \dots, u_T$.
Introducing the initial fictitious / reference comparator $u_0$ allows the pathlength sum
$P_t = \langle \nabla \psi_t(\tilde{w}_t) - \nabla \psi_t(\tilde{w}_1), u_{t-1} - u_t \rangle$
to share the exact same summation range $t \in [1, T]$ as all other regret components without
requiring a separate boundary term or split index ranges.

## Main definitions

* `Analysis.Convex.linearizationTerm`: Linearization error $- D_{l_t}(u_t, w_{t+1}, g_t)$.
* `Analysis.Convex.pathlengthTerm`: Centered pathlength
  $\langle \nabla \psi_t(\tilde{w}_t) - \nabla \psi_t(\tilde{w}_1), u_{t-1} - u_t \rangle$.
* `Analysis.Convex.stabilityTerm`: One-round stability
  $\eta(l_t(\tilde{w}_t) - l_t(w_{t+1})) -
    D_{\psi_t}(w_{t+1}, \tilde{w}_t, \nabla \psi_t(\tilde{w}_t))$.
* `Analysis.Convex.optimalityTerm`: Anchor update first-order optimality deficit.
* `Analysis.Convex.centeringTerm`: Centering surplus
  $\varphi_t(u_t) - \varphi_t(w_{t+1}) - D_{\varphi_t}(u_t, w_{t+1}, \nabla \varphi_t(w_{t+1}))$.
* `Analysis.Convex.adjustmentTerm`: Adjustment step drift
  $D_{\psi_{t+1}}(u_t, \tilde{w}_{t+1}) - D_{\psi_{t+1}}(u_t, w_{t+1})$.
* `Analysis.Convex.shiftTerm`: Regularizer shift evaluated on iterate
  $D_{\psi_{t+1} - \psi_t}(w_{t+1}, \tilde{w}_1)$.

## Notation

* `D_[ψ](x, y, g)`: Bregman divergence notation from `Analysis.Convex.bregDiv`
  (scoped in `Bregman`).

## Main results

* Regret decomposition: `Analysis.Convex.strongCenteredMirrorDescent`.
-/

open scoped BigOperators Bregman
open Finset

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-! ### Auxiliary Term Definitions -/

section AuxTerms

variable (η : ℝ)
variable (ψ φ : ℕ → E → ℝ)
variable (gψ gφ : ℕ → E → (E →L[ℝ] ℝ))
variable (u w w_tilde : ℕ → E)
variable (g : ℕ → (E →L[ℝ] ℝ))
variable (l : ℕ → E → ℝ)

/-- Linearization error term: $- D_{l_t}(u_t, w_{t+1}, g_t)$. -/
def linearizationTerm (t : ℕ) : ℝ :=
  - D_[l t](u t, w (t + 1), g t)

/-- Pathlength term anchored at $\tilde{w}_1$:
$P_t = \langle \nabla \psi_t(\tilde{w}_t) - \nabla \psi_t(\tilde{w}_1), u_{t-1} - u_t \rangle$. -/
def pathlengthTerm (t : ℕ) : ℝ :=
  (gψ t (w_tilde t) - gψ t (w_tilde 1)) (u (t - 1) - u t)

/-- Stability term:
$\delta_t = \eta(l_t(\tilde{w}_t) - l_t(w_{t+1})) -
  D_{\psi_t}(w_{t+1}, \tilde{w}_t, \nabla \psi_t(\tilde{w}_t))$. -/
def stabilityTerm (t : ℕ) : ℝ :=
  η * (l t (w_tilde t) - l t (w (t + 1))) - D_[ψ t](w (t + 1), w_tilde t, gψ t (w_tilde t))

/-- Optimality deficit anchored at $\tilde{w}_1$:
$\mu_t = (\eta g_t + \nabla \varphi_t(w_{t+1}) + \nabla \psi_{t+1}(w_{t+1})
  - \nabla \psi_t(\tilde{w}_t) - \nabla \psi_{t+1}(\tilde{w}_1)
  + \nabla \psi_t(\tilde{w}_1))(u_t - w_{t+1})$. -/
def optimalityTerm (t : ℕ) : ℝ :=
  ((η : ℝ) • g t + gφ t (w (t + 1)) + gψ (t + 1) (w (t + 1)) - gψ t (w_tilde t)
    - gψ (t + 1) (w_tilde 1) + gψ t (w_tilde 1)) (u t - w (t + 1))

/-- Centering surplus term:
$C_t = \varphi_t(u_t) - \varphi_t(w_{t+1})
  - D_{\varphi_t}(u_t, w_{t+1}, \nabla \varphi_t(w_{t+1}))$. -/
def centeringTerm (t : ℕ) : ℝ :=
  φ t (u t) - φ t (w (t + 1)) - D_[φ t](u t, w (t + 1), gφ t (w (t + 1)))

/-- Adjustment drift term:
$\xi_t = D_{\psi_{t+1}}(u_t, \tilde{w}_{t+1}, \nabla \psi_{t+1}(\tilde{w}_{t+1}))
  - D_{\psi_{t+1}}(u_t, w_{t+1}, \nabla \psi_{t+1}(w_{t+1}))$. -/
def adjustmentTerm (t : ℕ) : ℝ :=
  D_[ψ (t+1)](u t, w_tilde (t + 1), gψ (t+1) (w_tilde (t + 1)))
  - D_[ψ (t+1)](u t, w (t + 1), gψ (t+1) (w (t + 1)))

/-- Regularizer shift evaluated on iterate $w_{t+1}$ anchored at $\tilde{w}_1$:
$\Delta_t(w_{t+1}) = D_{\psi_{t+1} - \psi_t}(w_{t+1}, \tilde{w}_1,
  \nabla \psi_{t+1}(\tilde{w}_1) - \nabla \psi_t(\tilde{w}_1))$. -/
def shiftTerm (t : ℕ) : ℝ :=
  D_[ψ (t + 1) - ψ t](w (t + 1), w_tilde 1, gψ (t + 1) (w_tilde 1) - gψ t (w_tilde 1))

/-! ### Strong Centered Regret Decomposition (Jacobsen) -/

/-- Strong Centered Online Mirror Descent Regret Decomposition Identity (Jacobsen).
Exact multi-round algebraic equality holding for all rounds $T \ge 0$. -/
theorem strongCenteredMirrorDescent (T : ℕ) :
    η * (∑ t ∈ Ico 1 (T + 1), (l t (w_tilde t) - l t (u t))) =
    D_[ψ (T+1)](u T, w_tilde 1, gψ (T+1) (w_tilde 1))
    - D_[ψ (T+1)](u T, w_tilde (T + 1), gψ (T+1) (w_tilde (T + 1)))
    + (∑ t ∈ Ico 1 (T + 1), adjustmentTerm ψ gψ u w w_tilde t)
    + (∑ t ∈ Ico 1 (T + 1), pathlengthTerm gψ u w_tilde t)
    + (∑ t ∈ Ico 1 (T + 1), stabilityTerm η ψ gψ w w_tilde l t)
    - (∑ t ∈ Ico 1 (T + 1), shiftTerm ψ gψ w w_tilde t)
    - (∑ t ∈ Ico 1 (T + 1), optimalityTerm η gψ gφ u w w_tilde g t)
    + (∑ t ∈ Ico 1 (T + 1), centeringTerm φ gφ u w t)
    + η * (∑ t ∈ Ico 1 (T + 1), linearizationTerm u w g l t) := by
  induction T with
  | zero => simp
  | succ T ih =>
    simp_rw [Finset.sum_Ico_succ_top (by omega : 1 ≤ T + 1), mul_add, ih]
    dsimp [optimalityTerm, adjustmentTerm, pathlengthTerm, stabilityTerm, centeringTerm,
      shiftTerm, linearizationTerm, bregDiv]
    simp only [map_sub, add_apply, sub_apply, smul_apply, smul_eq_mul]
    ring

end AuxTerms

end Analysis.Convex
