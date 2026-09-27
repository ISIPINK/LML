/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

-- OGD-local copy of the Optimistic FTRL regret decomposition, so that the
-- stability development under `OGD/OptFPRL` is self-contained.

public import Mathlib.Analysis.Normed.Module.Basic
public import Mathlib.Analysis.Normed.Operator.ContinuousLinearMap
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Data.Finset.Interval
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Bregman.Basic

/-!
# Master Regret Decomposition for Optimistic Follow the Pruned Leader (OptFPRL)

This file establishes the exact multi-round algebraic regret decomposition for the
Optimistic Follow the Pruned Leader family, corresponding to Lemma 4.1
(Strong Dynamic Optimistic FTRL) of Naram Mhaisen and George Iosifidis,
*On the Dynamic Regret of FTRL: Optimism with History Pruning*.

The predictor is the optimistic update
$$x_{t+1} \doteq \operatorname*{argmin}_x\; h_{0:t}(x) + \tilde f_{t+1}(x),$$
with per-round potentials
$$h_t(x) = \langle p_t, x \rangle + ψ_t(x), \qquad p_t \in \partial \ell_t(x_t),$$
cumulative history `h_{0:t}`, and a prediction `\tilde f_{t+1}` of the next loss.

In contrast to `Algorithms/FPRL`, which sets the prediction to zero, the terminal
optimality term retains the prediction and reads
$$F_{0:T}(w_{T+1}) - F_{0:T}(u_T)
  = \bigl(h_{0:T}(w_{T+1}) + \tilde f_{T+1}(w_{T+1})\bigr)
    - \bigl(h_{0:T}(u_T) + \tilde f_{T+1}(u_T)\bigr),$$
where `F_obj t = h_{0:t} + \tilde f_{t+1}`. Because `\tilde f_{T+1}` is not zero, the
prediction terms no longer cancel in the identity; the prediction is carried explicitly
on the right-hand side as `\tilde f_{T+1}(u_T) - \tilde f_{T+1}(w_{T+1})`.

As in the FTRL, COMD2 and FPRL decompositions, the subgradient linearization error is
expressed through the Bregman divergence
$$D_{\ell_t}(u_t, w_t, p_t) = \ell_t(u_t) - \ell_t(w_t) - p_t(u_t - w_t).$$
All sums run over the common range `t ∈ Finset.Ico 1 (T + 1)`.

## Main definitions
* `H_obj`
* `F_obj`
* `boundary`
* `stability`
* `pathlength`
* `linearization`
* `terminalOptimality`

## Main results
* `regret_decomposition_eq`: The optimized version retaining the prediction in the
  terminal optimality term, holding for all $T \ge 0$.
-/

open scoped BigOperators Bregman
open Finset

@[expose] public section

namespace Online.OCO.OGD.OptFPRL

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

section Terms

variable (l : ℕ → E → ℝ)
variable (p : ℕ → E →L[ℝ] ℝ)
variable (ψ : ℕ → E → ℝ)
variable (f_tilde : ℕ → E → ℝ)
variable (u w : ℕ → E)

/-- The cumulative history $h_{0:t}(y) = \sum_{i=1}^{t} (p_i(y) + r_i(y))$. -/
def H_obj (t : ℕ) (y : E) : ℝ :=
  ∑ i ∈ Ico 1 (t + 1), (p i y + ψ i y)

/-- The optimistic objective $h_{0:t}(y) + \tilde f_{t+1}(y)$ minimized at round $t+1$. -/
def F_obj (t : ℕ) (y : E) : ℝ :=
  H_obj p ψ t y + f_tilde (t + 1) y

/-- Boundary term: the cumulative regularizer evaluated at the final comparator,
$\sum_{t=1}^{T} ψ_t(u_T)$. This is what remains after the pathlength is reduced to its
linear (first) part; in the OGD specialization it is `eucSq (cumSigma σ T) (u T)`. -/
def boundary (T : ℕ) : ℝ :=
  ∑ t ∈ Ico 1 (T + 1), ψ t (u T)

/-- Stability term (I): $h_{0:t}(w_t) - h_{0:t}(w_{t+1}) - ψ_t(w_t)$. -/
def stability (t : ℕ) : ℝ :=
  H_obj p ψ t (w t) - H_obj p ψ t (w (t + 1)) - ψ t (w t)

/--
Pathlength term (II), reduced to the linear (first) part of the cumulative history
$h_{0:t-1} = \sum_{i=1}^{t-1} p_i + (\text{regularizer})$. Keeping only the linear term leaves
$\bigl(\sum_{i=1}^{t-1} p_i\bigr)(u_t - u_{t-1})$; the dropped regularizer part is absorbed
into `boundary`. The sum telescopes to $\sum_{i=1}^{T-1} p_i(u_T - u_i)$, the `p`-part of the
paper's (II).
-/
def pathlength (t : ℕ) : ℝ :=
  (∑ i ∈ Ico 1 t, p i) (u t - u (t - 1))

/-- Linearization error, negative Bregman divergence
$-D_{\ell_t}(u_t, w_t, p_t) = \ell_t(w_t) - \ell_t(u_t) - p_t(w_t - u_t)$,
which is non-positive whenever $p_t$ is a subgradient of the convex loss $\ell_t$ at
$w_t$. -/
def linearization (t : ℕ) : ℝ :=
  - D_[l t](u t, w t, p t)

/-- Terminal optimality gap of the optimistic objective,
$F_{0:T}(w_{T+1}) - F_{0:T}(u_T)
  = \bigl(h_{0:T}(w_{T+1}) + \tilde f_{T+1}(w_{T+1})\bigr)
    - \bigl(h_{0:T}(u_T) + \tilde f_{T+1}(u_T)\bigr)$.
When `w_{T+1}` is the minimizer of `h_{0:T} + \tilde f_{T+1}` this is non-positive. -/
def terminalOptimality (T : ℕ) : ℝ :=
  F_obj p ψ f_tilde T (w (T + 1)) - F_obj p ψ f_tilde T (u T)

/-- Master Algebraic Regret Decomposition Identity for optimistic FTRL with controlled
state (Mhaisen & Iosifidis, *On the Dynamic Regret of FTRL*, Lemma 4.1), with the
terminal prediction retained and the pathlength reduced to its linear part. -/
theorem regret_decomposition_eq (T : ℕ) :
    (∑ t ∈ Ico 1 (T + 1), (l t (w t) - l t (u t))) =
    boundary ψ u T
    + (∑ t ∈ Ico 1 (T + 1), stability p ψ w t)
    + (∑ t ∈ Ico 1 (T + 1), pathlength p u t)
    + (∑ t ∈ Ico 1 (T + 1), linearization l p u w t)
    + terminalOptimality p ψ f_tilde u w T
    - f_tilde (T + 1) (w (T + 1)) + f_tilde (T + 1) (u T) := by
  induction T with
  | zero => simp [boundary, H_obj, F_obj, terminalOptimality]
  | succ n ih =>
    rw [sum_Ico_succ_top (by omega : 1 ≤ n + 1), ih]
    simp only [boundary, stability, pathlength, linearization, terminalOptimality, F_obj, H_obj,
      bregDiv, _root_.sum_apply, map_sub, Finset.sum_add_distrib, Finset.sum_sub_distrib,
      sum_Ico_succ_top (show 1 ≤ n + 1 by omega), Nat.add_sub_cancel]
    ring

end Terms

end Online.OCO.OGD.OptFPRL
