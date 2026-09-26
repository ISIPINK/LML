/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Algebra.BigOperators.Intervals
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Domain
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Regularizer
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Boundary
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.FTRL.RegretDecomposition

/-!
# Boundary Term Bound for Follow-the-Regularized-Leader (FTRL)

This file bounds the boundary term
$$\mathrm{boundary}(\psi, u, w, T) = \psi_{T+1}(u) - \psi_1(w_1)$$
for FTRL with shifted unnormalized negative entropy regularizers $\psi^{\mathrm{shift}}_t$.

## Main results

* `Online.OCO.LEA.FTRL.unnormEntropyShifted_uniformSimplex`: $\psi^{\mathrm{shift}}_\alpha(w_1) = 0$
  when $w_1 = (1/d, \dots, 1/d)$.
* `Online.OCO.LEA.FTRL.boundary_shifted_le_log_card`: Boundary term bound:
  $$\mathrm{boundary}(\psi^{\mathrm{shift}}_\alpha, u, w, T) \le
    \alpha_1 \ln d + (\psi^{\mathrm{shift}}_{\alpha(T+1)}(u) -
      \psi^{\mathrm{shift}}_{\alpha 1}(u)).$$
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.FTRL

open Online.OCO.LEA.OMD

variable {d : ℕ}

/-- The shifted unnormalized negative entropy evaluates to zero at the uniform distribution
$w_1 = (1/d, \dots, 1/d)$. -/
lemma unnormEntropyShifted_uniformSimplex (hd : 0 < d) (α : ℝ) :
    unnormEntropyShifted α (uniformSimplex d) = 0 := by
  rw [unnormEntropyShifted, unnormEntropy]
  have hlog : ∀ i : Fin d, Real.log (uniformSimplex d i) = -Real.log d := by
    intro i
    simp [uniformSimplex_apply, one_div, Real.log_inv]
  have hw_sum : ∑ i, uniformSimplex d i = 1 := (mem_stdSimplex_uniformSimplex hd).2
  have h1 : ∑ i, uniformSimplex d i * Real.log (uniformSimplex d i) = -Real.log d := by
    calc ∑ i, uniformSimplex d i * Real.log (uniformSimplex d i)
      _ = ∑ i, uniformSimplex d i * (-Real.log d) := by congr 1 with i; rw [hlog i]
      _ = (∑ i, uniformSimplex d i) * (-Real.log d) := by rw [sum_mul]
      _ = -Real.log d := by rw [hw_sum, one_mul]
  rw [sum_sub_distrib, h1, hw_sum]
  ring

/-- Boundary term bound for FTRL with uniform initialization $w_1 = (1/d, \dots, 1/d)$ and
sequence of shifted regularizers $\psi^{\mathrm{shift}}_t = \text{unnormEntropyShifted}(\alpha_t)$:
$$\mathrm{boundary}(\psi^{\mathrm{shift}}_\alpha, u, w, T)
  = \psi^{\mathrm{shift}}_{\alpha(T+1)}(u) - \psi^{\mathrm{shift}}_{\alpha 1}(w_1)
  \le \alpha_1 \ln d + (\psi^{\mathrm{shift}}_{\alpha(T+1)}(u) -
    \psi^{\mathrm{shift}}_{\alpha 1}(u)).$$ -/
theorem boundary_shifted_le_log_card (α : ℕ → ℝ) (hα1 : 0 ≤ α 1) (T : ℕ)
    (hd : 0 < d)
    (u : EuclideanSpace ℝ (Fin d)) (hu : u ∈ stdSimplex (d := d))
    (w : ℕ → EuclideanSpace ℝ (Fin d))
    (hw1 : w 1 = uniformSimplex d) :
    boundary (fun t ↦ unnormEntropyShifted (α t)) u w T ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u) := by
  dsimp [boundary]
  rw [hw1, unnormEntropyShifted_uniformSimplex hd, sub_zero]
  have hu_nonneg : 0 ≤ unnormEntropyShifted (α 1) u := by
    have h_base := unnormEntropyShifted_nonneg_of_mem_stdSimplex hd u hu
    rw [unnormEntropyShifted_smul (α 1) u]
    exact smul_nonneg hα1 h_base
  have h_log_nonneg : 0 ≤ Real.log d := by
    have hd_ge : (1 : ℝ) ≤ d := by
      have : 1 ≤ d := hd
      exact_mod_cast this
    exact Real.log_nonneg hd_ge
  have h_log_term : 0 ≤ α 1 * Real.log d := mul_nonneg hα1 h_log_nonneg
  have : unnormEntropyShifted (α (T + 1)) u ≤
      α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u - unnormEntropyShifted (α 1) u) := by
    calc unnormEntropyShifted (α (T + 1)) u
      _ ≤ unnormEntropyShifted (α (T + 1)) u +
            (α 1 * Real.log d - unnormEntropyShifted (α 1) u) := by
        have : 0 ≤ α 1 * Real.log d - unnormEntropyShifted (α 1) u := by
          have hu_le : unnormEntropyShifted (α 1) u ≤ α 1 * Real.log d := by
            rw [unnormEntropyShifted, unnormEntropy]
            have h_sum_le0 : ∑ i, (u i * Real.log (u i) - u i) ≤ -1 := by
              rw [sum_sub_distrib, (mem_stdSimplex_iff u).mp hu |>.2]
              have h_ent_nonpos : ∑ i, u i * Real.log (u i) ≤ 0 := by
                refine sum_nonpos fun i _ ↦ ?_
                by_cases h0 : u i = 0
                · simp [h0]
                · have h_le1 : u i ≤ 1 := by
                    have : u i ≤ ∑ j, u j :=
                      single_le_sum (fun j _ ↦ (mem_stdSimplex_iff u).mp hu |>.1 j)
                        (Finset.mem_univ i)
                    rwa [(mem_stdSimplex_iff u).mp hu |>.2] at this
                  have hlog_nonpos : Real.log (u i) ≤ 0 :=
                    Real.log_nonpos ((mem_stdSimplex_iff u).mp hu |>.1 i) h_le1
                  exact mul_nonpos_of_nonneg_of_nonpos
                    ((mem_stdSimplex_iff u).mp hu |>.1 i) hlog_nonpos
              linarith
            nlinarith
          linarith
        linarith
      _ = α 1 * Real.log d + (unnormEntropyShifted (α (T + 1)) u -
            unnormEntropyShifted (α 1) u) := by ring
  exact this

end Online.OCO.LEA.FTRL
