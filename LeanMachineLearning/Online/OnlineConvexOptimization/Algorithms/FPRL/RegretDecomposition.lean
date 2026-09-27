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
# Master Regret Decomposition for Optimistic FTRL with Controlled State (FPRL)

This file establishes the exact multi-round algebraic regret decomposition for the
Follow-the-Pruned-Regularized-Leader / optimistic FTRL family with a controlled state,
corresponding to Lemma 4.1 (Strong Dynamic Optimistic FTRL) of Naram Mhaisen and
George Iosifidis, *On the Dynamic Regret of FTRL: Optimism with History Pruning*.

The predictor maintains per-round potentials
$$h_t(x) = \langle p_t, x \rangle + ψ_t(x), \qquad p_t \in \partial \ell_t(x_t),$$
with `h_0 = I_X`, and the cumulative history
$$h_{0:t}(x) = \sum_{i=1}^{t} h_i(x).$$

As in the FTRL and COMD2 decompositions (see `Algorithms/FTRL` and `Algorithms/COMD2`),
the subgradient linearization error is expressed through the Bregman divergence
$$D_{\ell_t}(u_t, w_t, p_t) = \ell_t(u_t) - \ell_t(w_t) - p_t(u_t - w_t).$$
Only its defining algebraic identity is used; no convexity, non-negativity, or other
property of the divergence is assumed. The boundary regularizer is absorbed by the
indicator `h_0 = I_X`. All sums run over the common range `t ∈ Finset.Ico 1 (T + 1)`,
mirroring COMD2: the `pathlength` term is written as a backward difference so that it is
indexed by the running round (with `h_{0:0} = 0`, its `t = 1` term vanishes).

## Main definitions
* `F_obj`
* `stability`
* `pathlength`
* `linearization`
* `terminalOptimality`

## Main results
* `regret_decomposition_eq`: The master multi-round algebraic regret equality holding
  for all $T \ge 0$.
-/

open scoped BigOperators Bregman
open Finset

@[expose] public section

namespace OnlineConvexOptimization.FPRL

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

section Terms

variable (l : ℕ → E → ℝ)
variable (p : ℕ → E →L[ℝ] ℝ)
variable (ψ : ℕ → E → ℝ)
variable (u w : ℕ → E)

/-- The cumulative history $h_{0:t}(y) = \sum_{i=1}^{t} (p_i(y) + r_i(y))$. -/
def F_obj (t : ℕ) (y : E) : ℝ :=
  ∑ i ∈ Ico 1 (t + 1), (p i y + ψ i y)

/-- Stability term (I): $h_{0:t}(w_t) - h_{0:t}(w_{t+1}) - ψ_t(w_t) + ψ_t(u_t)$. -/
def stability (t : ℕ) : ℝ :=
  F_obj p ψ t (w t) - F_obj p ψ t (w (t + 1)) - ψ t (w t) + ψ t (u t)

/--
Pathlength term (II), in the shifted (backward-difference) form
$h_{0:t-1}(u_t) - h_{0:t-1}(u_{t-1})$, which telescopes to the paper's
$\sum_{t} \bigl(h_{0:t}(u_{t+1}) - h_{0:t}(u_t)\bigr)$ but is indexed by the running
round so that all sums share the range `Ico 1 (T + 1)`.
-/
def pathlength (t : ℕ) : ℝ :=
  F_obj p ψ (t - 1) (u t) - F_obj p ψ (t - 1) (u (t - 1))

/-- Linearization error, negative Bregman divergence
$-D_{\ell_t}(u_t, w_t, p_t) = \ell_t(w_t) - \ell_t(u_t) - p_t(w_t - u_t)$,
which is non-positive whenever $p_t$ is a subgradient of the convex loss $\ell_t$ at
$w_t$. -/
def linearization (t : ℕ) : ℝ :=
  - D_[l t](u t, w t, p t)

/-- Terminal optimality gap $h_{0:T}(w_{T+1}) - h_{0:T}(u_T)$. -/
def terminalOptimality (T : ℕ) : ℝ :=
  F_obj p ψ T (w (T + 1)) - F_obj p ψ T (u T)

/-- Master Algebraic Regret Decomposition Identity for optimistic FTRL with controlled
state (Mhaisen & Iosifidis, *On the Dynamic Regret of FTRL*, Lemma 4.1). -/
theorem regret_decomposition_eq (T : ℕ) :
    (∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t (u t))) =
    (∑ t ∈ Ico 1 (T + 1), stability p ψ u w t)
    + (∑ t ∈ Ico 1 (T + 1), pathlength p ψ u t)
    + (∑ t ∈ Ico 1 (T + 1), linearization l p u w t)
    + terminalOptimality p ψ u w T := by
  induction T with
  | zero => simp [F_obj, terminalOptimality]
  | succ T ih =>
    rw [sum_Ico_succ_top (by omega : 1 ≤ T + 1), ih]
    simp only [stability, pathlength, linearization, terminalOptimality, F_obj, bregDiv,
      sum_Ico_succ_top (show 1 ≤ T + 1 by omega), Nat.add_sub_cancel]
    simp only [map_sub]
    ring

end Terms

end OnlineConvexOptimization.FPRL
