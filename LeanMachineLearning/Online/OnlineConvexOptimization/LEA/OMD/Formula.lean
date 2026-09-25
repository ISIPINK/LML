/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Domain
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Regularizer
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Optimality

/-!
# Exponential Weights Update Formula for Learning with Expert Advice (LEA)

This file defines the explicit closed-form Exponential Weights update (also known as the Hedge
update / Entropic OMD step) with time-varying regularizer weights $\alpha_t, \alpha_{t+1} > 0$:

$$w_{t+1, i} = \frac{w_{t, i}^{\alpha_t / \alpha_{t+1}} \exp(-g_{t, i}/\alpha_{t+1})}
  {\sum_{j=1}^d w_{t, j}^{\alpha_t / \alpha_{t+1}} \exp(-g_{t, j}/\alpha_{t+1})}.$$

## Main definitions

* `Online.OCO.LEA.OMD.expWeightStep`: The one-step update function mapping $w_t \in \Delta^{d-1}$
  to $w_{t+1} \in \Delta^{d-1}$.

## Main results

* `Online.OCO.LEA.OMD.expWeightStep_pos`: Positivity $0 < (w_{t+1})_i$ for all $i$.
* `Online.OCO.LEA.OMD.expWeightStep_mem_stdSimplex`: Membership $w_{t+1} \in \Delta^{d-1}$.
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {d : ℕ}

/-- Coordinate-wise unnormalized weight for the exponential update:
$$w_i^{\alpha / \alpha'} \exp(-g_i / \alpha')$$ -/
noncomputable def expWeightUnnorm (α α' : ℝ) (w : EuclideanSpace ℝ (Fin d))
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (i : Fin d) : ℝ :=
  (w i) ^ (α / α') * Real.exp (-(g (EuclideanSpace.basisFun (Fin d) ℝ i)) / α')

lemma expWeightUnnorm_pos (α α' : ℝ) {w : EuclideanSpace ℝ (Fin d)}
    (hw_pos : ∀ i, 0 < w i) (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (i : Fin d) :
    0 < expWeightUnnorm α α' w g i := by
  dsimp [expWeightUnnorm]
  exact mul_pos (Real.rpow_pos_of_pos (hw_pos i) _) (Real.exp_pos _)

lemma sum_expWeightUnnorm_pos (hd : 0 < d) (α α' : ℝ) {w : EuclideanSpace ℝ (Fin d)}
    (hw_pos : ∀ i, 0 < w i) (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) :
    0 < ∑ i, expWeightUnnorm α α' w g i := by
  have : (univ : Finset (Fin d)).Nonempty :=
    Finset.univ_nonempty_iff.mpr (Fin.pos_iff_nonempty.mp hd)
  exact sum_pos (fun i _ ↦ expWeightUnnorm_pos α α' hw_pos g i) this

lemma sum_expWeightUnnorm_ne_zero (hd : 0 < d) (α α' : ℝ) {w : EuclideanSpace ℝ (Fin d)}
    (hw_pos : ∀ i, 0 < w i) (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) :
    ∑ i, expWeightUnnorm α α' w g i ≠ 0 :=
  (sum_expWeightUnnorm_pos hd α α' hw_pos g).ne'

/-- The one-step Exponential Weights (Hedge / Entropic OMD) update:
$$w_{t+1, i} = \frac{w_{t, i}^{\alpha_t / \alpha_{t+1}} \exp(-g_{t, i} / \alpha_{t+1})}
  {\sum_j w_{t, j}^{\alpha_t / \alpha_{t+1}} \exp(-g_{t, j} / \alpha_{t+1})}.$$ -/
noncomputable def expWeightStep (_hd : 0 < d) (α α' : ℝ) (w : EuclideanSpace ℝ (Fin d))
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) : EuclideanSpace ℝ (Fin d) :=
  WithLp.toLp 2 (fun i ↦ expWeightUnnorm α α' w g i / ∑ j, expWeightUnnorm α α' w g j)

lemma expWeightStep_apply (hd : 0 < d) (α α' : ℝ) (w : EuclideanSpace ℝ (Fin d))
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (i : Fin d) :
    expWeightStep hd α α' w g i =
      expWeightUnnorm α α' w g i / ∑ j, expWeightUnnorm α α' w g j :=
  rfl

/-- Positivity of coordinates after an Exponential Weights step. -/
lemma expWeightStep_pos (hd : 0 < d) (α α' : ℝ) (w : EuclideanSpace ℝ (Fin d))
    (hw_pos : ∀ i, 0 < w i) (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (i : Fin d) :
    0 < expWeightStep hd α α' w g i := by
  rw [expWeightStep_apply]
  exact div_pos (expWeightUnnorm_pos α α' hw_pos g i) (sum_expWeightUnnorm_pos hd α α' hw_pos g)

/-- The Exponential Weights update always stays within the standard simplex $\Delta^{d-1}$. -/
theorem expWeightStep_mem_stdSimplex (hd : 0 < d) (α α' : ℝ) (w : EuclideanSpace ℝ (Fin d))
    (hw_pos : ∀ i, 0 < w i) (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) :
    expWeightStep hd α α' w g ∈ stdSimplex (d := d) := by
  rw [mem_stdSimplex_iff]
  refine ⟨fun i ↦ (expWeightStep_pos hd α α' w hw_pos g i).le, ?_⟩
  simp_rw [expWeightStep_apply]
  rw [← sum_div, div_self (sum_expWeightUnnorm_ne_zero hd α α' hw_pos g)]

/-- Logarithm of coordinates for the Exponential Weights update. -/
lemma log_expWeightStep_apply (hd : 0 < d) (α α' : ℝ) (_hα' : 0 < α')
    (w : EuclideanSpace ℝ (Fin d)) (hw_pos : ∀ i, 0 < w i)
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (i : Fin d) :
    Real.log (expWeightStep hd α α' w g i) =
      (α / α') * Real.log (w i) - (g (EuclideanSpace.basisFun (Fin d) ℝ i)) / α'
        - Real.log (∑ j, expWeightUnnorm α α' w g j) := by
  rw [expWeightStep_apply, Real.log_div (expWeightUnnorm_pos α α' hw_pos g i).ne'
    (sum_expWeightUnnorm_pos hd α α' hw_pos g).ne']
  dsimp [expWeightUnnorm]
  rw [Real.log_mul (Real.rpow_pos_of_pos (hw_pos i) _).ne' (Real.exp_pos _).ne',
    Real.log_rpow (hw_pos i), Real.log_exp]
  ring

/-- The objective function for the mirror descent step:
$$x \mapsto g(x) + \psi_{\alpha'}(x) - \nabla\psi_\alpha(w)(x).$$ -/
noncomputable def mirrorStepObj (α α' : ℝ) (w : EuclideanSpace ℝ (Fin d))
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (x : EuclideanSpace ℝ (Fin d)) : ℝ :=
  g x + unnormEntropy α' x - (unnormEntropyFDeriv α w) x

/-- Linear coordinate expansion of the subgradient and Fréchet derivative on $\mathbb{R}^d$. -/
lemma mirrorStepObj_diff_eq (hd : 0 < d) (α α' : ℝ) (hα' : 0 < α')
    (w : EuclideanSpace ℝ (Fin d)) (hw_pos : ∀ i, 0 < w i)
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (x : EuclideanSpace ℝ (Fin d)) (hx : x ∈ stdSimplex) :
    (g + unnormEntropyFDeriv α' (expWeightStep hd α α' w g) - unnormEntropyFDeriv α w)
      (x - expWeightStep hd α α' w g) = 0 := by
  let w_next := expWeightStep hd α α' w g
  have hw_next_mem := expWeightStep_mem_stdSimplex hd α α' w hw_pos g
  rw [mem_stdSimplex_iff] at hx hw_next_mem
  have h_basis (y : EuclideanSpace ℝ (Fin d)) :
      g y = ∑ i, y i * g (EuclideanSpace.basisFun (Fin d) ℝ i) := by
    have hy : y = ∑ i, y i • EuclideanSpace.basisFun (Fin d) ℝ i := by
      ext i; simp [EuclideanSpace.basisFun_apply, Pi.single_apply]
    conv_lhs => rw [hy]
    rw [map_sum]
    refine sum_congr rfl fun i _ ↦ by rw [map_smul, smul_eq_mul]
  simp only [sub_apply, add_apply, unnormEntropyFDeriv_apply]
  have h_w_next_i (i : Fin d) :
      g (EuclideanSpace.basisFun (Fin d) ℝ i) + α' * Real.log (w_next i) - α * Real.log (w i) =
        - α' * Real.log (∑ j, expWeightUnnorm α α' w g j) := by
    have hlog := log_expWeightStep_apply hd α α' hα' w hw_pos g i
    dsimp [w_next]
    rw [hlog]
    have hα'_ne : α' ≠ 0 := hα'.ne'
    field_simp
    ring
  have h_comb : (g (x - w_next) + α' * ∑ i, (x - w_next) i * Real.log (w_next i)
      - α * ∑ i, (x - w_next) i * Real.log (w i)) =
      ∑ i, (x i - w_next i) *
        (g (EuclideanSpace.basisFun (Fin d) ℝ i) + α' * Real.log (w_next i)
          - α * Real.log (w i)) := by
    rw [h_basis (x - w_next)]
    simp only [PiLp.sub_apply, mul_sum, ← sum_add_distrib, ← sum_sub_distrib]
    congr 1 with i
    ring
  rw [h_comb]
  simp_rw [h_w_next_i]
  rw [← sum_mul, sum_sub_distrib, hx.2, hw_next_mem.2, sub_self, zero_mul]

/-- The Exponential Weights update is an exact constrained minimizer (`IsMinOn`)
of the linearized mirror descent step objective on the standard simplex $\Delta^{d-1}$. -/
theorem isMinOn_expWeightStep (hd : 0 < d) (α α' : ℝ) (hα' : 0 < α')
    (w : EuclideanSpace ℝ (Fin d)) (hw_pos : ∀ i, 0 < w i)
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) :
    IsMinOn (fun x ↦ g x + unnormEntropy α' x - (unnormEntropyFDeriv α w) x)
      (stdSimplex (d := d)) (expWeightStep hd α α' w g) := by
  let w_next := expWeightStep hd α α' w g
  have hw_next_pos := expWeightStep_pos hd α α' w hw_pos g
  have hw_next_mem := expWeightStep_mem_stdSimplex hd α α' w hw_pos g
  have h_diff : HasFDerivAt (unnormEntropy α') (unnormEntropyFDeriv α' w_next) w_next :=
    hasFDerivAt_unnormEntropy α' w_next hw_next_pos
  have h_conv : ConvexOn ℝ (stdSimplex (d := d)) (unnormEntropy α') :=
    convexOn_unnormEntropy_stdSimplex α' hα'.le
  let lin : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ := g - unnormEntropyFDeriv α w
  have h_obj_conv : ConvexOn ℝ (stdSimplex (d := d)) (unnormEntropy α' + ⇑lin) :=
    h_conv.add (lin.toLinearMap.convexOn h_conv.1)
  have h_obj_diff : HasFDerivAt (unnormEntropy α' + ⇑lin)
      (unnormEntropyFDeriv α' w_next + lin) w_next :=
    h_diff.add lin.hasFDerivAt
  intro x hx
  have h_zero : (unnormEntropyFDeriv α' w_next + lin) (x - w_next) = 0 := by
    dsimp [lin]
    have := mirrorStepObj_diff_eq hd α α' hα' w hw_pos g x hx
    have h_reorder : (unnormEntropyFDeriv α' w_next + (g - unnormEntropyFDeriv α w)) (x - w_next) =
        (g + unnormEntropyFDeriv α' w_next - unnormEntropyFDeriv α w) (x - w_next) := by
      simp only [add_apply, sub_apply]
      ring
    rwa [h_reorder]
  have h_subg_base := (hasSubgradientWithinAt_iff_le.mp
      (h_obj_diff.hasSubgradientWithinAt h_obj_conv hw_next_mem)) x hx
  dsimp [bregDiv] at h_subg_base
  rw [h_zero, add_zero] at h_subg_base
  dsimp [lin] at h_subg_base ⊢
  simp only [sub_apply] at h_subg_base ⊢
  linarith

/-! ### Whole Trajectory Sequence Construction and Properties -/

/-- Recursive trajectory of iterates generated by the Exponential Weights update, starting from
$w_1 = \mathrm{uniformSimplex}(d)$ at $t = 1$. -/
noncomputable def expWeights (hd : 0 < d) (α : ℕ → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ)) : ℕ → EuclideanSpace ℝ (Fin d)
  | 0 => uniformSimplex d
  | 1 => uniformSimplex d
  | t + 2 => expWeightStep hd (α (t + 1)) (α (t + 2)) (expWeights hd α g (t + 1)) (g (t + 1))

@[simp]
lemma expWeights_one (hd : 0 < d) (α : ℕ → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ)) :
    expWeights hd α g 1 = uniformSimplex d :=
  rfl

lemma expWeights_succ_succ (hd : 0 < d) (α : ℕ → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ)) (t : ℕ) :
    expWeights hd α g (t + 2) =
      expWeightStep hd (α (t + 1)) (α (t + 2)) (expWeights hd α g (t + 1)) (g (t + 1)) :=
  rfl

/-- Every iterate $w_t$ produced by `expWeights` is strictly positive in all coordinates for
$t \ge 1$. -/
theorem expWeights_pos (hd : 0 < d) (α : ℕ → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ)) :
    ∀ t ≥ 1, ∀ i, 0 < expWeights hd α g t i := by
  intro t ht
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := Nat.exists_eq_add_of_le' ht
  clear ht
  induction n with
  | zero =>
    intro i
    rw [expWeights_one]
    exact uniformSimplex_pos hd i
  | succ n ih =>
    intro i
    rw [expWeights_succ_succ]
    exact expWeightStep_pos hd (α (n + 1)) (α (n + 2)) (expWeights hd α g (n + 1))
      (fun j ↦ ih j) (g (n + 1)) i

/-- Every iterate $w_t$ produced by `expWeights` lies in the standard simplex $\Delta^{d-1}$ for
$t \ge 1$. -/
theorem expWeights_mem_stdSimplex (hd : 0 < d) (α : ℕ → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ)) :
    ∀ t ≥ 1, expWeights hd α g t ∈ stdSimplex (d := d) := by
  intro t ht
  obtain ⟨n, rfl⟩ : ∃ n, t = n + 1 := Nat.exists_eq_add_of_le' ht
  clear ht
  induction n with
  | zero =>
    rw [expWeights_one]
    exact mem_stdSimplex_uniformSimplex hd
  | succ n _ =>
    rw [expWeights_succ_succ]
    have h_pos : ∀ j, 0 < expWeights hd α g (n + 1) j := by
      have : 1 ≤ n + 1 := by omega
      exact expWeights_pos hd α g (n + 1) this
    exact expWeightStep_mem_stdSimplex hd (α (n + 1)) (α (n + 2))
      (expWeights hd α g (n + 1)) h_pos (g (n + 1))

/-- The iterates generated by `expWeights` satisfy the `IsMinOn` condition at each round $t \ge 1$. -/
theorem expWeights_isMinOn (hd : 0 < d) (α : ℕ → ℝ)
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ)) (t : ℕ) (ht : 1 ≤ t)
    (hα'_pos : 0 < α (t + 1)) :
    IsMinOn (fun x ↦ g t x + unnormEntropy (α (t + 1)) x -
      (unnormEntropyFDeriv (α t) (expWeights hd α g t)) x)
      (stdSimplex (d := d)) (expWeights hd α g (t + 1)) := by
  obtain ⟨k, rfl⟩ : ∃ k, t = k + 1 := Nat.exists_eq_add_of_le' ht
  rw [expWeights_succ_succ]
  have h_pos := expWeights_pos hd α g (k + 1) (by omega)
  exact isMinOn_expWeightStep hd (α (k + 1)) (α (k + 1 + 1)) hα'_pos
    (expWeights hd α g (k + 1)) h_pos (g (k + 1))

end Online.OCO.LEA.OMD
