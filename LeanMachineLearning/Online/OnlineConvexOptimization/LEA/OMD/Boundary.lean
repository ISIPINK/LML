/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Basic
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Deriv
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Domain
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Regularizer
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.RegretDecomposition

/-!
# Boundary Term Bound for Learning with Expert Advice (LEA)

This file formalizes the boundary term bound for Learning with Expert Advice (LEA)
under the unnormalized negative entropy regularizer $\psi(x) = \sum_{i=1}^d (x_i \ln x_i - x_i)$
when the initial iterate $w_1$ is chosen as the uniform distribution on the standard simplex
$\Delta^{d-1}$:
$$w_1 = \left( \frac{1}{d}, \dots, \frac{1}{d} \right) \in \Delta^{d-1}.$$

## Main results

* `Online.OCO.LEA.OMD.uniformSimplex`: The uniform distribution vector $(1/d, \dots, 1/d)$.
* `Online.OCO.LEA.OMD.mem_stdSimplex_uniformSimplex`: $w_1 \in \Delta^{d-1}$.
* `Online.OCO.LEA.OMD.bregDiv_unnormEntropy_uniformSimplex`: For any $u \in \Delta^{d-1}$,
  $$D_\psi(u, w_1) = \ln d + \sum_{i=1}^d u_i \ln u_i.$$
* `Online.OCO.LEA.OMD.bregDiv_unnormEntropy_uniformSimplex_le`: Since $u_i \le 1$ implies
  $u_i \ln u_i \le 0$, we have the classical bound:
  $$D_\psi(u, w_1) \le \ln d.$$
* `Online.OCO.LEA.OMD.boundary_shifted_le_log_card`: When $\alpha_1 \ge 0, \alpha_{T+1} \ge 0$,
  the multi-round boundary term satisfies:
  $$\mathrm{boundary}(\psi^{\mathrm{shift}}_\alpha, \nabla\psi_\alpha, u, w, T)
    \le \alpha_1 \ln d + (\psi^{\mathrm{shift}}_{\alpha(T+1)}(u) -
      \psi^{\mathrm{shift}}_{\alpha 1}(u)).$$
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {d : ℕ}




/-- The Bregman divergence $D_{\psi_\alpha}(u, w_1)$ from the uniform distribution $w_1$ to any
comparator $u \in \Delta^{d-1}$ is exactly $\alpha (\ln d + \sum_{i=1}^d u_i \ln u_i)$. -/
theorem bregDiv_unnormEntropy_uniformSimplex (α : ℝ) (hd : 0 < d) (u : EuclideanSpace ℝ (Fin d))
    (hu : u ∈ stdSimplex) :
    D_[unnormEntropy α](u, uniformSimplex d, unnormEntropyFDeriv α (uniformSimplex d)) =
      α * (Real.log d + ∑ i, u i * Real.log (u i)) := by
  rw [mem_stdSimplex_iff] at hu
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.mpr hd
  have hlog : ∀ i : Fin d, Real.log (uniformSimplex d i) = -Real.log d := by
    intro i
    simp [uniformSimplex_apply, one_div, Real.log_inv]
  have hw_sum : ∑ i, uniformSimplex d i = 1 := (mem_stdSimplex_uniformSimplex hd).2
  calc
    D_[unnormEntropy α](u, uniformSimplex d, unnormEntropyFDeriv α (uniformSimplex d))
      = unnormEntropy α u - unnormEntropy α (uniformSimplex d)
        - unnormEntropyFDeriv α (uniformSimplex d) (u - uniformSimplex d) := rfl
    _ = α * (∑ i, u i * Real.log (u i) - ∑ i, u i)
        - α * (∑ i, uniformSimplex d i * (-Real.log d) - ∑ i, uniformSimplex d i)
        - α * ∑ i, (u i - uniformSimplex d i) * (-Real.log d) := by
      rw [unnormEntropy, unnormEntropy, unnormEntropyFDeriv_apply]
      simp_rw [sum_sub_distrib, hlog, PiLp.sub_apply]
    _ = α * (∑ i, u i * Real.log (u i) - 1)
        - α * ((-Real.log d) * 1 - 1)
        - α * ((-Real.log d) * (1 - 1)) := by
      have h1 : ∑ i, uniformSimplex d i * (-Real.log d) = (-Real.log d) * 1 := by
        rw [← sum_mul, hw_sum, one_mul, mul_one]
      have h2 : ∑ i, (u i - uniformSimplex d i) * (-Real.log d) = (-Real.log d) * (1 - 1) := by
        rw [← sum_mul, sum_sub_distrib, hu.2, hw_sum, mul_comm]
      rw [hu.2, hw_sum, h1, h2]
    _ = α * (Real.log d + ∑ i, u i * Real.log (u i)) := by
      ring

/-- For any comparator $u \in \Delta^{d-1}$ and $\alpha \ge 0$, the Bregman divergence from the
uniform distribution is upper-bounded by $\alpha \ln d$:
$$D_{\psi_\alpha}(u, w_1) \le \alpha \ln d.$$ -/
theorem bregDiv_unnormEntropy_uniformSimplex_le (α : ℝ) (hα : 0 ≤ α) (hd : 0 < d)
    (u : EuclideanSpace ℝ (Fin d)) (hu : u ∈ stdSimplex) :
    D_[unnormEntropy α](u, uniformSimplex d, unnormEntropyFDeriv α (uniformSimplex d)) ≤
      α * Real.log d := by
  rw [bregDiv_unnormEntropy_uniformSimplex α hd u hu]
  rw [mem_stdSimplex_iff] at hu
  have h_sum_nonpos : ∑ i, u i * Real.log (u i) ≤ 0 := by
    refine sum_nonpos fun i _ ↦ ?_
    by_cases h0 : u i = 0
    · simp [h0]
    · have h_le1 : u i ≤ 1 := by
        have : u i ≤ ∑ j, u j := single_le_sum (fun j _ ↦ hu.1 j) (Finset.mem_univ i)
        rwa [hu.2] at this
      have hlog_nonpos : Real.log (u i) ≤ 0 :=
        Real.log_nonpos (hu.1 i) h_le1
      exact mul_nonpos_of_nonneg_of_nonpos (hu.1 i) hlog_nonpos
  nlinarith

/-- Boundary term bound for LEA with uniform initialization $w_1 = (1/d, \dots, 1/d)$ and
sequence of shifted regularizers $\psi^{\mathrm{shift}}_t = \text{unnormEntropyShifted}(\alpha_t)$
with $\alpha_1 \ge 0$ and $\alpha_{T+1} \ge 0$:
$$\mathrm{boundary}(\psi^{\mathrm{shift}}_\alpha, \nabla\psi_\alpha, u, w, T)
  \le \alpha_1 \ln d + (\psi^{\mathrm{shift}}_{\alpha(T+1)}(u) -
    \psi^{\mathrm{shift}}_{\alpha 1}(u)).$$ -/
theorem boundary_shifted_le_log_card (α : ℕ → ℝ) (hα1 : 0 ≤ α 1) (T : ℕ)
    (hαT : 0 ≤ α (T + 1)) (hd : 0 < d)
    (u : EuclideanSpace ℝ (Fin d)) (hu : u ∈ stdSimplex)
    (w : ℕ → EuclideanSpace ℝ (Fin d))
    (hw1 : w 1 = uniformSimplex d)
    (hwT : w (T + 1) ∈ stdSimplex (d := d))
    (hwT_pos : ∀ i, 0 < w (T + 1) i) :
    boundary (fun t ↦ unnormEntropyShifted (α t)) (fun t ↦ unnormEntropyFDeriv (α t)) u w T ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u) := by
  have h_shift_cancel (t : ℕ) :
      D_[unnormEntropyShifted (α t)](u, w t, unnormEntropyFDeriv (α t) (w t)) =
      D_[unnormEntropy (α t)](u, w t, unnormEntropyFDeriv (α t) (w t)) := by
    dsimp [bregDiv, unnormEntropyShifted]
    ring
  dsimp [boundary]
  rw [h_shift_cancel 1, h_shift_cancel (T + 1)]
  have h_diff := hasFDerivAt_unnormEntropy (α (T + 1)) (w (T + 1)) hwT_pos
  have h_conv := convexOn_unnormEntropy_stdSimplex (α (T + 1)) hαT (d := d)
  have h_sub := h_diff.hasSubgradientWithinAt h_conv hwT
  have h_breg_end : 0 ≤ D_[unnormEntropy (α (T + 1))](u, w (T + 1),
      unnormEntropyFDeriv (α (T + 1)) (w (T + 1))) :=
    h_sub u hu
  rw [hw1]
  have h_init := bregDiv_unnormEntropy_uniformSimplex_le (α 1) hα1 hd u hu
  linarith

/-- Base shifted regularizer $\psi^{\mathrm{shift}}_1(x) \ge 0$ on the standard simplex
$\Delta^{d-1}$ when $d > 0$. -/
theorem unnormEntropyShifted_nonneg_of_mem_stdSimplex (hd : 0 < d) (x : EuclideanSpace ℝ (Fin d))
    (hx : x ∈ stdSimplex (d := d)) :
    0 ≤ unnormEntropyShifted 1 x := by
  have hw_pos : ∀ i, 0 < uniformSimplex d i := uniformSimplex_pos hd
  have hw_mem : uniformSimplex d ∈ stdSimplex (d := d) := mem_stdSimplex_uniformSimplex hd
  have h_diff := hasFDerivAt_unnormEntropy 1 (uniformSimplex d) hw_pos
  have h_conv := convexOn_unnormEntropy_stdSimplex 1 (by norm_num) (d := d)
  have h_sub := h_diff.hasSubgradientWithinAt h_conv hw_mem
  have h_breg_nonneg := h_sub x hx
  rw [bregDiv_unnormEntropy_uniformSimplex 1 hd x hx, one_mul] at h_breg_nonneg
  rw [mem_stdSimplex_iff] at hx
  dsimp [unnormEntropyShifted, unnormEntropy]
  have h_sum_sub : ∑ i, (x i * Real.log (x i) - x i) = (∑ i, x i * Real.log (x i)) - 1 := by
    rw [sum_sub_distrib, hx.2]
  rw [h_sum_sub]
  linarith

end Online.OCO.LEA.OMD
