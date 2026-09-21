/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.Online.OnlineConvexOptimization.Algorithms.OMD.Definition
public import LeanMachineLearning.SequentialLearning.Algorithm
public import LeanMachineLearning.SequentialLearning.Deterministic

/-!
# Sequential Learning Adapter for Centered Online Mirror Descent (COMD)

This file connects the Centered Online Mirror Descent (COMD) definitions
to the general `SequentialLearning` framework (`Learning.Algorithm`, `Learning.Hist`).

## Protocol Mapping

In the sequential learning framework, each round $n$ consists of:
- **Observation $\mathcal{O}$**: Unit / dummy observation `Unit`.
- **Action $\mathcal{A}$**: The anchor point $\tilde{w}_n \in \mathcal{X} \subseteq E$.
- **Feedback $\mathcal{Y}$**: The observed loss subgradient $g_n \in E \toL[\mathbb{R}] \mathbb{R}$.

## Main definitions

* `Analysis.Convex.anchorFromHist`: Causal reconstruction of the sequence of anchors $\tilde{w}$
  from the history of past loss subgradients.
* `Analysis.Convex.omdAlgorithm`: Centered OMD instantiated as a `Learning.Algorithm`.

## Main results

* Exact execution equality: `Analysis.Convex.omd_action_ae_eq`.
-/

open scoped Bregman
open Learning

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [mA : MeasurableSpace E] [mY : MeasurableSpace (E →L[ℝ] ℝ)]

/-! ### History to State Reconstruction (Causal Trajectory) -/

/-- Extract the sequence of past feedback subgradients from `Hist` up to round `n`. -/
def subgradientFromHist (n : ℕ) (hist : Hist Unit E (E →L[ℝ] ℝ) n) (t : ℕ) : E →L[ℝ] ℝ :=
  if h : t < n then (hist ⟨t, h⟩).feedback else 0

/-- Reconstruct the unadjusted and anchor iterates $(w_t, \tilde{w}_t)$ up to time `n`
from the history of past loss subgradients.
- At round $0$: initial anchor $w_1$.
- At round $t \to t+1$: uses subgradient $g_t$ from history, computes $w_{t+1}$,
  then applies `adjust`. -/
noncomputable def anchorFromHist (P : OMDParams E) (w1 : E)
    (argmin_anchor : (E →L[ℝ] ℝ) → (E → ℝ) → (E → ℝ) → (E →L[ℝ] ℝ) → E)
    (n : ℕ) (hist : Hist Unit E (E →L[ℝ] ℝ) n) : ℕ → E
  | 0 => w1
  | 1 => w1
  | t + 1 =>
    let prev_anchor := anchorFromHist P w1 argmin_anchor n hist t
    let gt := subgradientFromHist n hist t
    let w_next := argmin_anchor gt (P.φ t) (P.ψ (t + 1)) (P.gψ t prev_anchor)
    P.adjust t w_next

/-! ### Deterministic Policy for OMD -/

/-- The deterministic action function of Centered OMD: given history up to round `n`,
computes the current anchor $\tilde{w}_n \in \mathcal{X}$. -/
noncomputable def omdNextAction (P : OMDParams E) (w1 : E)
    (argmin_anchor : (E →L[ℝ] ℝ) → (E → ℝ) → (E → ℝ) → (E →L[ℝ] ℝ) → E)
    (n : ℕ) (p : Hist Unit E (E →L[ℝ] ℝ) n × Unit) : E :=
  anchorFromHist P w1 argmin_anchor n p.1 n

/-- Instantiate Centered OMD as a formal `Learning.Algorithm`. -/
noncomputable def omdAlgorithm (P : OMDParams E) (w1 : E)
    (argmin_anchor : (E →L[ℝ] ℝ) → (E → ℝ) → (E → ℝ) → (E →L[ℝ] ℝ) → E)
    (h_meas : ∀ n, Measurable (omdNextAction P w1 argmin_anchor n)) :
    Algorithm Unit E (E →L[ℝ] ℝ) :=
  detAlgorithm (omdNextAction P w1 argmin_anchor) h_meas

/-! ### Trajectory Equalities -/

/-- In any algorithm-environment sequence execution, the action $A_n$ at round $n$
is almost everywhere equal to the deterministic evaluation of `omdNextAction`. -/
lemma omd_action_ae_eq {Ω : Type*} {mΩ : MeasurableSpace Ω}
    [MeasurableEq E] {P_meas : MeasureTheory.Measure Ω} [MeasureTheory.IsProbabilityMeasure P_meas]
    {O : ℕ → Ω → Unit} {A : ℕ → Ω → E} {Y : ℕ → Ω → (E →L[ℝ] ℝ)}
    {env : Environment Unit E (E →L[ℝ] ℝ)}
    (P : OMDParams E) (w1 : E)
    (argmin_anchor : (E →L[ℝ] ℝ) → (E → ℝ) → (E → ℝ) → (E →L[ℝ] ℝ) → E)
    (h_meas : ∀ n, Measurable (omdNextAction P w1 argmin_anchor n))
    (h_exec : IsAlgEnvSeq O A Y (omdAlgorithm P w1 argmin_anchor h_meas) env P_meas)
    (n : ℕ) :
    A n =ᵐ[P_meas] fun ω ↦
      omdNextAction P w1 argmin_anchor n (history O A Y n ω, O n ω) :=
  IsAlgEnvSeq.action_detAlgorithm_ae_eq h_exec n

end Analysis.Convex
