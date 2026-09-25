/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.Convex.Basic
public import Mathlib.Analysis.Convex.StdSimplex

/-!
# Standard Simplex Domain for Online Convex Optimization

In Mathlib, `Convexity.StdSimplex R X` is defined as an abstract algebraic type (a bundled
finitely-supported probability distribution / monadic structure) rather than a subset of an ambient
normed vector space.

However, continuous optimization and online convex optimization (OCO) algorithms—such as Online
Mirror Descent (OMD) and Learning with Expert Advice (LEA)—require iterates, subgradients,
Bregman divergences, and Fréchet derivatives to live in a normed vector space $E = \mathbb{R}^d$
(here `EuclideanSpace ℝ (Fin d)`).

This file defines the standard simplex `stdSimplex` directly as a convex subset of
`EuclideanSpace ℝ (Fin d)` and provides basic lemmas and equivalence with Mathlib's
`Convexity.StdSimplex`.

## Main definitions

* `Online.OCO.LEA.OMD.stdSimplex`: The standard simplex $\Delta^{d-1} \subset \mathbb{R}^d$
  defined as $\{x \in \mathbb{R}^d \mid (\forall i, 0 \le x_i) \wedge \sum_i x_i = 1\}$.

## Main results

* `Online.OCO.LEA.OMD.convex_stdSimplex`: Convexity of `stdSimplex`.
* `Online.OCO.LEA.OMD.stdSimplex_eq_range_mathlib`: Equivalence to the range of
  `Convexity.StdSimplex.weights`.
-/

open scoped BigOperators
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {d : ℕ}

/-- The standard simplex $\Delta^{d-1} \subset \mathbb{R}^d$ in `EuclideanSpace ℝ (Fin d)`:
$$\Delta^{d-1} = \left\{ x \in \mathbb{R}^d \;\middle|\; \forall i, 0 \le x_i
\;\text{and}\; \sum_{i=1}^d x_i = 1 \right\}$$ -/
def stdSimplex : Set (EuclideanSpace ℝ (Fin d)) :=
  { x : EuclideanSpace ℝ (Fin d) | (∀ i, 0 ≤ x i) ∧ ∑ i, x i = 1 }

lemma mem_stdSimplex_iff (x : EuclideanSpace ℝ (Fin d)) :
    x ∈ stdSimplex ↔ (∀ i, 0 ≤ x i) ∧ ∑ i, x i = 1 :=
  Iff.rfl

/-- Equivalence between `stdSimplex` and the range of Mathlib's algebraic
`Convexity.StdSimplex.weights`. -/
lemma stdSimplex_eq_range_mathlib :
    (stdSimplex : Set (EuclideanSpace ℝ (Fin d))) =
      { x : EuclideanSpace ℝ (Fin d) | WithLp.ofLp x ∈ Set.range
          (fun (s : Convexity.StdSimplex ℝ (Fin d)) ↦ (s.weights : Fin d → ℝ)) } := by
  ext x
  simp only [stdSimplex, Set.mem_ofPred_eq]
  rw [Convexity.StdSimplex.range_toFun_comp_weights]
  simp only [Set.mem_inter_iff, Set.mem_iInter, Set.mem_ofPred_eq]

/-- The standard simplex is a convex set in `EuclideanSpace ℝ (Fin d)`. -/
theorem convex_stdSimplex : Convex ℝ (stdSimplex (d := d)) := by
  intro x hx y hy a b ha hb hab
  rw [mem_stdSimplex_iff] at hx hy ⊢
  refine ⟨fun i ↦ add_nonneg (mul_nonneg ha (hx.1 i)) (mul_nonneg hb (hy.1 i)), ?_⟩
  simp_rw [PiLp.add_apply, PiLp.smul_apply, smul_eq_mul]
  rw [sum_add_distrib, ← mul_sum, ← mul_sum, hx.2, hy.2, mul_one, mul_one, hab]

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

end Online.OCO.LEA.OMD
