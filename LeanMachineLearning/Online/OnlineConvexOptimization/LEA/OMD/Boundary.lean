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
* `Online.OCO.LEA.OMD.boundary_le_log_card`: When $\psi_{T+1} = \psi_1 = \psi$ and
  $\nabla\psi(w_{T+1}) \in \partial \psi(w_{T+1})$, the multi-round boundary term satisfies:
  $$\mathrm{boundary}(\psi, \nabla\psi, u, w, T) \le \ln d.$$
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {d : ℕ}

/-- The uniform distribution vector on `EuclideanSpace ℝ (Fin d)`:
$$w_1 = \left( \frac{1}{d}, \dots, \frac{1}{d} \right)$$ -/
noncomputable def uniformSimplex (d : ℕ) : EuclideanSpace ℝ (Fin d) :=
  WithLp.toLp 2 (fun _ ↦ (1 : ℝ) / d)

@[simp]
lemma uniformSimplex_apply (i : Fin d) : uniformSimplex d i = (1 : ℝ) / d :=
  rfl

lemma uniformSimplex_pos (hd : 0 < d) (i : Fin d) : 0 < uniformSimplex d i := by
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.mpr hd
  exact div_pos zero_lt_one hd_pos

/-- The uniform distribution lies in the standard simplex `stdSimplex`. -/
theorem mem_stdSimplex_uniformSimplex (hd : 0 < d) : uniformSimplex d ∈ stdSimplex (d := d) := by
  rw [mem_stdSimplex_iff]
  refine ⟨fun i ↦ (uniformSimplex_pos hd i).le, ?_⟩
  have hd_ne : (d : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hd.ne'
  simp [sum_const, card_univ, Fintype.card_fin, nsmul_eq_mul, div_eq_inv_mul,
    mul_inv_cancel₀ hd_ne]

/-- The Bregman divergence $D_\psi(u, w_1)$ from the uniform distribution $w_1$ to any
comparator $u \in \Delta^{d-1}$ is exactly $\ln d + \sum_{i=1}^d u_i \ln u_i$. -/
theorem bregDiv_unnormEntropy_uniformSimplex (hd : 0 < d) (u : EuclideanSpace ℝ (Fin d))
    (hu : u ∈ stdSimplex) :
    D_[unnormEntropy](u, uniformSimplex d, unnormEntropyFDeriv (uniformSimplex d)) =
      Real.log d + ∑ i, u i * Real.log (u i) := by
  rw [mem_stdSimplex_iff] at hu
  have hd_pos : (0 : ℝ) < d := Nat.cast_pos.mpr hd
  have hlog : ∀ i : Fin d, Real.log (uniformSimplex d i) = -Real.log d := by
    intro i
    simp [uniformSimplex_apply, one_div, Real.log_inv]
  have hw_sum : ∑ i, uniformSimplex d i = 1 := (mem_stdSimplex_uniformSimplex hd).2
  calc
    D_[unnormEntropy](u, uniformSimplex d, unnormEntropyFDeriv (uniformSimplex d))
      = unnormEntropy u - unnormEntropy (uniformSimplex d)
        - unnormEntropyFDeriv (uniformSimplex d) (u - uniformSimplex d) := rfl
    _ = (∑ i, u i * Real.log (u i) - ∑ i, u i)
        - (∑ i, uniformSimplex d i * (-Real.log d) - ∑ i, uniformSimplex d i)
        - ∑ i, (u i - uniformSimplex d i) * (-Real.log d) := by
      rw [unnormEntropy, unnormEntropy, unnormEntropyFDeriv_apply]
      simp_rw [sum_sub_distrib, hlog, PiLp.sub_apply]
    _ = (∑ i, u i * Real.log (u i) - 1)
        - ((-Real.log d) * 1 - 1)
        - ((-Real.log d) * (1 - 1)) := by
      have h1 : ∑ i, uniformSimplex d i * (-Real.log d) = (-Real.log d) * 1 := by
        rw [← sum_mul, hw_sum, one_mul, mul_one]
      have h2 : ∑ i, (u i - uniformSimplex d i) * (-Real.log d) = (-Real.log d) * (1 - 1) := by
        rw [← sum_mul, sum_sub_distrib, hu.2, hw_sum, mul_comm]
      rw [hu.2, hw_sum, h1, h2]
    _ = Real.log d + ∑ i, u i * Real.log (u i) := by
      ring

/-- For any comparator $u \in \Delta^{d-1}$, the Bregman divergence from the uniform distribution
is upper-bounded by $\ln d$:
$$D_\psi(u, w_1) \le \ln d.$$ -/
theorem bregDiv_unnormEntropy_uniformSimplex_le (hd : 0 < d) (u : EuclideanSpace ℝ (Fin d))
    (hu : u ∈ stdSimplex) :
    D_[unnormEntropy](u, uniformSimplex d, unnormEntropyFDeriv (uniformSimplex d)) ≤
      Real.log d := by
  rw [bregDiv_unnormEntropy_uniformSimplex hd u hu]
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
  linarith

/-- Boundary term bound for LEA with uniform initialization $w_1 = (1/d, \dots, 1/d)$ and
time-invariant regularizer $\psi_t = \text{unnormEntropy}$:
$$\mathrm{boundary}(\psi, \nabla\psi, u, w, T) \le \ln d.$$ -/
theorem boundary_le_log_card (hd : 0 < d) (T : ℕ)
    (u : EuclideanSpace ℝ (Fin d)) (hu : u ∈ stdSimplex)
    (w : ℕ → EuclideanSpace ℝ (Fin d))
    (hw1 : w 1 = uniformSimplex d)
    (hwT : w (T + 1) ∈ stdSimplex (d := d))
    (hwT_pos : ∀ i, 0 < w (T + 1) i) :
    boundary (fun _ ↦ unnormEntropy) (fun _ ↦ unnormEntropyFDeriv) u w T ≤ Real.log d := by
  dsimp [boundary]
  have h_diff := hasFDerivAt_unnormEntropy (w (T + 1)) hwT_pos
  have h_conv := convexOn_unnormEntropy_stdSimplex (d := d)
  have h_sub := h_diff.hasSubgradientWithinAt h_conv hwT
  have h_breg_end : 0 ≤ D_[unnormEntropy](u, w (T + 1), unnormEntropyFDeriv (w (T + 1))) :=
    h_sub u hu
  rw [hw1]
  have h_init := bregDiv_unnormEntropy_uniformSimplex_le hd u hu
  linarith

end Online.OCO.LEA.OMD
