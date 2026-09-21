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
# Online Mirror Descent (OMD) Definition

This file defines the two optimization steps of Optimistic Centered Online Mirror Descent
(OMD / COMD) over a convex domain $\mathcal{X} \subseteq E$, followed by an arbitrary
deterministic adjustment step.

## Algorithm Overview

At each round $t \ge 1$, given current anchor $\tilde{w}_t \in \mathcal{X}$:
1. **Optimistic Prediction Step**:
   The learner computes $v_t \in \mathcal{X}$ minimizing the linear surrogate hint $h_t$
   regularized by the Bregman divergence from $\tilde{w}_t$ under $\psi_{t+1}$:
   $$v_t \in \operatorname{argmin}_{x \in \mathcal{X}} \left( \eta h_t(x) +
     D_{\psi_{t+1}}(x, \tilde{w}_t, \nabla \psi_t(\tilde{w}_t)) \right)$$
   which is equivalent to minimizing $\eta h_t(x) - \nabla \psi_t(\tilde{w}_t)(x) + \psi_{t+1}(x)$.

2. **Loss Update / Anchor Step**:
   Upon observing loss subgradient $g_t$, the learner computes $w_{t+1} \in \mathcal{X}$
   minimizing the linearized loss $\eta g_t$ with centering potential $\varphi_t$
   and step regularizer $\psi_{t+1}$:
   $$w_{t+1} \in \operatorname{argmin}_{x \in \mathcal{X}} \left( \eta g_t(x) + \varphi_t(x) +
     \psi_{t+1}(x) - \nabla \psi_t(\tilde{w}_t)(x) \right)$$

3. **Deterministic Adjustment Step**:
   The learner applies a deterministic mapping $\operatorname{adjust}_t : E \to E$
   yielding the new anchor $\tilde{w}_{t+1} = \operatorname{adjust}_t(w_{t+1})$.
-/

open scoped Bregman

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-! ### Step Objectives -/

/-- Objective for the optimistic prediction step at round $t$:
$\eta h_t(x) + \psi_{t+1}(x) - \nabla \psi_t(\tilde{w}_t)(x)$. -/
def optimisticPredictionObjective (η : ℝ) (h : E →L[ℝ] ℝ) (ψ_next : E → ℝ) (gψ_anchor : E →L[ℝ] ℝ)
    (x : E) : ℝ :=
  η * h x + ψ_next x - gψ_anchor x

/-- Objective for the main loss update / unadjusted iterate $w_{t+1}$ at round $t$:
$\eta g_t(x) + \varphi_t(x) + \psi_{t+1}(x) - \nabla \psi_t(\tilde{w}_t)(x)$. -/
def anchorUpdateObjective (η : ℝ) (g : E →L[ℝ] ℝ) (φ : E → ℝ) (ψ_next : E → ℝ)
    (gψ_anchor : E →L[ℝ] ℝ) (x : E) : ℝ :=
  η * g x + φ x + ψ_next x - gψ_anchor x

/-! ### Step Optimality Predicates via `IsMinOn` -/

/-- Predicate stating that $v \in \mathcal{X}$ is a valid optimistic prediction at round $t$. -/
def IsOptimisticPrediction (η : ℝ) (h : E →L[ℝ] ℝ) (ψ_next : E → ℝ) (gψ_anchor : E →L[ℝ] ℝ)
    (X : Set E) (v : E) : Prop :=
  IsMinOn (optimisticPredictionObjective η h ψ_next gψ_anchor) X v

/-- Predicate stating that $w \in \mathcal{X}$ is a valid loss update / anchor iterate
at round $t$. -/
def IsAnchorUpdate (η : ℝ) (g : E →L[ℝ] ℝ) (φ : E → ℝ) (ψ_next : E → ℝ) (gψ_anchor : E →L[ℝ] ℝ)
    (X : Set E) (w : E) : Prop :=
  IsMinOn (anchorUpdateObjective η g φ ψ_next gψ_anchor) X w

/-- Prediction step is the special case of an anchor update with potential $\varphi = 0$. -/
lemma isOptimisticPrediction_iff_isAnchorUpdate (η : ℝ) (h : E →L[ℝ] ℝ) (ψ_next : E → ℝ)
    (gψ_anchor : E →L[ℝ] ℝ) (X : Set E) (v : E) :
    IsOptimisticPrediction η h ψ_next gψ_anchor X v ↔
      IsAnchorUpdate η h 0 ψ_next gψ_anchor X v := by
  have : optimisticPredictionObjective η h ψ_next gψ_anchor =
      anchorUpdateObjective η h 0 ψ_next gψ_anchor := by
    ext x; simp [optimisticPredictionObjective, anchorUpdateObjective]
  rw [IsOptimisticPrediction, IsAnchorUpdate, this]

/-! ### Full Algorithm Step & Sequence Definition -/

/-- Structure bundling the parameters of an Optimistic Centered OMD instance. -/
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

/-- Predicate defining an execution trace $(v, w, \tilde{w})$ of Optimistic Centered OMD.
- $v_t$ is the optimistic prediction from hint $h_t$ and anchor $\tilde{w}_t$.
- $w_{t+1}$ is the primal loss update from $g_t$, $\varphi_t$, and anchor $\tilde{w}_t$.
- $\tilde{w}_{t+1} = \operatorname{adjust}_t(w_{t+1})$ is the deterministic adjustment. -/
structure OMDExecution (P : OMDParams E) (g h : ℕ → (E →L[ℝ] ℝ))
    (v w w_tilde : ℕ → E) : Prop where
  /-- Prediction step is a minimizer over $\mathcal{X}$. -/
  prediction_min : ∀ t ≥ 1,
    IsOptimisticPrediction P.η (h t) (P.ψ (t + 1)) (P.gψ t (w_tilde t)) P.X (v t)
  /-- Anchor update step is a minimizer over $\mathcal{X}$. -/
  anchor_min : ∀ t ≥ 1,
    IsAnchorUpdate P.η (g t) (P.φ t) (P.ψ (t + 1)) (P.gψ t (w_tilde t)) P.X (w (t + 1))
  /-- Deterministic adjustment transition: $\tilde{w}_{t+1} = \operatorname{adjust}_t(w_{t+1})$. -/
  adjustment_eq : ∀ t ≥ 1, w_tilde (t + 1) = P.adjust t (w (t + 1))

end Analysis.Convex
