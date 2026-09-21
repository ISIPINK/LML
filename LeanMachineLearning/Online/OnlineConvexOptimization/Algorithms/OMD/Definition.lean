/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Order.Filter.Basic
public import Mathlib.Topology.Order.Basic
public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Bregman.Basic

/-!
# Centered Online Mirror Descent (COMD) Definition

This file defines the optimization and state transition steps of Centered Online
Mirror Descent (COMD) over a convex domain $\mathcal{X} \subseteq E$, followed by an arbitrary
deterministic adjustment step.

## Algorithm Overview

At each round $t \ge 1$, given current anchor $\tilde{w}_t \in \mathcal{X}$:
1. **Action / Prediction**:
   The learner predicts anchor $\tilde{w}_t$ (or current state).
2. **Loss Update / Anchor Step**:
   Upon observing loss subgradient $g_t \in E \toL[\mathbb{R}] \mathbb{R}$, the learner computes
   unadjusted iterate $w_{t+1} \in \mathcal{X}$ minimizing the linearized loss $\eta g_t$ with
   centering potential $\varphi_t$ and step regularizer $\psi_{t+1}$:
   $$w_{t+1} \in \operatorname{argmin}_{x \in \mathcal{X}} \left( \eta g_t(x) + \varphi_t(x) +
     \psi_{t+1}(x) - \nabla \psi_t(\tilde{w}_t)(x) \right)$$
3. **Deterministic Adjustment Step**:
   The learner applies a deterministic mapping $\operatorname{adjust}_t : E \to E$
   yielding the next anchor $\tilde{w}_{t+1} = \operatorname{adjust}_t(w_{t+1})$.

## Main definitions

* `Analysis.Convex.OMDParams`:
  Parameters structure bundling domain $\mathcal{X}$, learning rate $\eta$, regularizers $\psi$,
  centering $\varphi$, and adjustment mappings.
* `Analysis.Convex.OMDExecution`:
  Predicate defining an execution trace $(w, \tilde{w})$ of Centered OMD.
-/

open scoped Bregman

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-! ### Algorithm Parameters and Execution -/

/-- Structure bundling the parameters of a Centered OMD instance. -/
structure OMDParams (E : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E] where
  /-- Decision set / constraint domain $\mathcal{X} \subseteq E$. -/
  X : Set E
  /-- Step size / learning rate $\eta > 0$. -/
  η : ℝ
  /-- Sequence of step regularizers $\psi_t$. -/
  ψ : ℕ → E → ℝ
  /-- Sequence of gradients / dual evaluation maps $\nabla \psi_t$. -/
  gψ : ℕ → E → (E →L[ℝ] ℝ)
  /-- Sequence of centering potentials $\varphi_t$. -/
  φ : ℕ → E → ℝ
  /-- Sequence of centering gradients $\nabla \varphi_t$. -/
  gφ : ℕ → E → (E →L[ℝ] ℝ)
  /-- Deterministic adjustment mappings $\operatorname{adjust}_t : E \to E$. -/
  adjust : ℕ → E → E

/-- Predicate defining an execution trace $(w, \tilde{w})$ of Centered OMD:
- $w_{t+1}$ minimizes $\eta g_t(x) + \varphi_t(x) + \psi_{t+1}(x) - \nabla \psi_t(\tilde{w}_t)(x)$
  over $\mathcal{X}$.
- $\tilde{w}_{t+1} = \operatorname{adjust}_t(w_{t+1})$ is the deterministic adjustment. -/
structure OMDExecution (P : OMDParams E) (g : ℕ → (E →L[ℝ] ℝ))
    (w w_tilde : ℕ → E) : Prop where
  /-- Anchor update step is a minimizer over $\mathcal{X}$. -/
  anchor_min : ∀ t ≥ 1,
    IsMinOn (fun x ↦ P.η * (g t) x + P.φ t x + P.ψ (t + 1) x - (P.gψ t (w_tilde t)) x)
      P.X (w (t + 1))
  /-- Deterministic adjustment transition: $\tilde{w}_{t+1} = \operatorname{adjust}_t(w_{t+1})$. -/
  adjustment_eq : ∀ t ≥ 1, w_tilde (t + 1) = P.adjust t (w (t + 1))

end Analysis.Convex
