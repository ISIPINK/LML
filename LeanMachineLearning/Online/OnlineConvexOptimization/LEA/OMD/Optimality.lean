/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.RegretDecomposition
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Basic
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Deriv

/-!
# First-Order Optimality Bounds for Online Mirror Descent (LEA Specialization)

This file proves that the first-order optimality deficit:
$$\mathrm{optimality}_t = (\eta g_t + \nabla \psi_{t+1}(w_{t+1}) -
  \nabla \psi_t(w_t))(u - w_{t+1})$$
is **non-negative** (i.e. $0 \le \mathrm{optimality}_t$) whenever $w_{t+1}$ minimizes the
linearized mirror descent step:
$$x \mapsto \eta g_t(x) + \psi_{t+1}(x) - \nabla \psi_t(w_t)(x)$$
over a convex domain $s \subseteq E$, $\psi_{t+1}$ is convex and differentiable at $w_{t+1}$,
and the comparator $u \in s$.

Because this term appears with a minus sign in `regret_decomposition_eq`:
$$- \sum_t \mathrm{optimality}_t \le 0,$$
non-negativity directly establishes that the optimality term can be discarded or upper-bounded
by $0$.

## Main results
* `optimality_nonneg_of_isMinOn`: $0 \le \mathrm{optimality}_t$
  whenever $w_{t+1}$ is a constrained minimizer on $s$ and $u \in s$.
-/

open scoped Bregman Topology
open Filter

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

section Optimality

variable {ψ : ℕ → E → ℝ}
variable {gψ : ℕ → E → (E →L[ℝ] ℝ)}
variable {u : E}
variable {w : ℕ → E}
variable {g : ℕ → (E →L[ℝ] ℝ)}
variable {s : Set E}

/-- First-order optimality deficit is non-negative when $w_{t+1}$ minimizes the mirror
descent step objective over $s$, $\psi_{t+1}$ is convex with Fréchet derivative
$g\psi_{t+1}(w_{t+1})$, and $u \in s$. -/
lemma optimality_nonneg_of_isMinOn (t : ℕ)
    (hψ_diff : HasFDerivAt (ψ (t + 1)) (gψ (t + 1) (w (t + 1))) (w (t + 1)))
    (hψ_conv : ConvexOn ℝ s (ψ (t + 1)))
    (hw : w (t + 1) ∈ s)
    (hu : u ∈ s)
    (h_min : IsMinOn (fun x ↦ (g t) x + ψ (t + 1) x - (gψ t (w t)) x) s (w (t + 1))) :
    0 ≤ optimality gψ u w g t := by
  let lin : E →L[ℝ] ℝ := g t - gψ t (w t)
  have h_conv : ConvexOn ℝ s (ψ (t + 1) + ⇑lin) :=
    hψ_conv.add (lin.toLinearMap.convexOn hψ_conv.1)
  have h_min' : IsMinOn ((ψ (t + 1) + ⇑lin) + fun _ : E ↦ (0 : ℝ)) s (w (t + 1)) := by
    intro x hx
    have := h_min hx
    dsimp [lin] at this ⊢
    simp only [add_zero, sub_apply] at this ⊢
    linarith
  have h_subg := ((hψ_diff.add lin.hasFDerivAt).hasSubgradientWithinAt_add_iff
    (g := 0) h_conv (convexOn_const 0 h_conv.1) hw).mp
    (hasSubgradientWithinAt_zero_iff_isMinOn.mpr h_min') u hu
  dsimp [bregDiv, optimality, lin] at h_subg ⊢
  simp only [sub_self, zero_sub, neg_apply, neg_neg, add_apply, sub_apply] at h_subg ⊢
  linarith

end Optimality

end Online.OCO.LEA.OMD
