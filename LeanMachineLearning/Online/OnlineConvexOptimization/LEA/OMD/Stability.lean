/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import Mathlib.Analysis.InnerProductSpace.Basic
public import Mathlib.Analysis.InnerProductSpace.PiL2
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Domain
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.RegretDecomposition
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.OMD.Regularizer
public import LeanMachineLearning.ForMathlib.Analysis.Convex.Subgradient.Basic

/-!
# Stability Bound for Learning with Expert Advice (LEA)

This file proves the local-norm stability bound for Learning with Expert Advice (LEA)
using coordinate-wise Fenchel-Young inequality and the Mean Value Theorem for Bregman divergence.

## Main results
* `Online.OCO.LEA.OMD.dual_local_norm_eta_le`: Coordinate-weighted duality bound with
  step size $\eta$:
  $\eta \sum_i v_i g_i - \frac{1}{2} \sum_i \frac{v_i^2}{z_i} \le
    \frac{\eta^2}{2} \sum_i z_i g_i^2$.
* `Online.OCO.LEA.OMD.stability_le_dual_norm`: One-round stability term bound via MVT:
  $$\delta_t \le \frac{1}{2\alpha_{t+1}} \sum_{i=1}^d z_i (g_t)_i^2.$$
* `Online.OCO.LEA.OMD.sum_stability_le_dual_norm`: Multi-round cumulative stability bound.
* `Online.OCO.LEA.OMD.stability_le_max_sq`: One-round stability bound bounded by $\max_i (g_t)_i^2$.
* `Online.OCO.LEA.OMD.sum_stability_le_max_sq`: Cumulative stability bound bounded by
  $\sum_{t=1}^T \frac{1}{2\alpha_{t+1}} \max_i (g_t)_i^2$.
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Online.OCO.LEA.OMD

variable {d : ℕ}

/-! ### Scalar and Coordinate Duality (Fenchel-Young) -/

/-- Scaled coordinate-wise duality inequality with learning rate $\eta$:
$$\eta \sum_i v_i g_i - \frac{1}{2} \sum_i \frac{v_i^2}{z_i}
  \le \frac{\eta^2}{2} \sum_i z_i g_i^2$$ -/
lemma dual_local_norm_eta_le (η : ℝ) (v : EuclideanSpace ℝ (Fin d)) (g : Fin d → ℝ)
    (z : EuclideanSpace ℝ (Fin d)) (hz : ∀ i, 0 < z i) :
    η * (∑ i, v i * g i) - (1 / 2 : ℝ) * (∑ i, (v i) ^ 2 / z i) ≤
      (η ^ 2 / 2) * ∑ i, z i * (g i) ^ 2 := by
  calc η * (∑ i, v i * g i) - (1 / 2 : ℝ) * (∑ i, (v i) ^ 2 / z i)
    _ = ∑ i, (v i * (η * g i) - (1 / 2 : ℝ) * ((v i) ^ 2 / z i)) := by
      rw [mul_sum, mul_sum, ← sum_sub_distrib]; congr 1 with i; ring
    _ ≤ ∑ i, (1 / 2 : ℝ) * (z i * (η * g i) ^ 2) := sum_le_sum fun i _ ↦ by
      have : (v i - z i * (η * g i))^2 / z i =
          (v i)^2 / z i - 2 * (v i * (η * g i)) + z i * (η * g i)^2 := by
        field_simp [(hz i).ne']; ring
      linarith [div_nonneg (sq_nonneg (v i - z i * (η * g i))) (hz i).le]
    _ = (η ^ 2 / 2) * ∑ i, z i * (g i) ^ 2 := by
      rw [mul_sum]; congr 1 with i; ring

variable (η : ℝ)
variable (w : ℕ → EuclideanSpace ℝ (Fin d))
variable (l : ℕ → EuclideanSpace ℝ (Fin d) → ℝ)
variable (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ))

/-- One-round stability bound for Learning with Expert Advice via the Mean Value Theorem (MVT):
when $w_{t+1} \in \text{stdSimplex}$, and iterates $w_{t+1}, w_t$ have strictly positive
coordinates, the round-$t$ stability term with regularizer scale $\alpha_t > 0$
`stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t`
is upper bounded by $\frac{1}{2 \alpha_t} \sum_{i=1}^d z_i (g_t)_i^2$
for some intermediate point $z \in [w_{t+1}, w_t]$. -/
theorem stability_le_dual_norm (α : ℕ → ℝ) (t : ℕ) (hα_pos : 0 < α t)
    (hw_next_pos : ∀ i, 0 < (w (t + 1) : EuclideanSpace ℝ (Fin d)) i)
    (hw_t_pos : ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i) :
    ∃ z ∈ segment ℝ (w (t + 1)) (w t),
      stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
        (1 / (2 * α t)) * ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
  have h_pos : ∀ z ∈ segment ℝ (w (t + 1)) (w t), ∀ i, 0 < (z : EuclideanSpace ℝ (Fin d)) i := by
    rintro z ⟨a, b, ha, hb, hab, rfl⟩ i
    dsimp
    obtain rfl | ha_pos := eq_or_lt_of_le ha
    · have : b = 1 := by linarith
      simp [this, hw_t_pos i]
    · linarith [mul_pos ha_pos (hw_next_pos i), mul_nonneg hb (hw_t_pos i).le]
  obtain ⟨z, hz_seg, hz_breg⟩ := bregDiv_unnormEntropy_mvt (α t) (w (t + 1)) (w t) h_pos
  refine ⟨z, hz_seg, ?_⟩
  dsimp [stability]
  rw [bregDiv_unnormEntropyShifted, hz_breg]
  have h_decomp : g t (w t - w (t + 1)) =
      ∑ i, (w t i - w (t + 1) i) * g t (EuclideanSpace.basisFun (Fin d) ℝ i) := by
    have : w t - w (t + 1) =
        ∑ i, (w t i - w (t + 1) i) • EuclideanSpace.basisFun (Fin d) ℝ i := by
      ext i; simp [EuclideanSpace.basisFun_apply, Pi.single_apply]
    rw [this, map_sum]
    refine sum_congr rfl fun i _ ↦ by rw [map_smul, smul_eq_mul]
  simp_rw [show ∀ i, (w (t + 1) i - w t i)^2 = (w t i - w (t + 1) i)^2 from
    fun i ↦ by rw [← neg_sub (w t i) (w (t + 1) i), neg_sq]]
  have h_dual := dual_local_norm_eta_le (1 / α t) (w t - w (t + 1))
    (fun i ↦ g t (EuclideanSpace.basisFun (Fin d) ℝ i)) z (h_pos z hz_seg)
  have h_pi (i : Fin d) : (w t - w (t + 1)) i = w t i - w (t + 1) i := rfl
  simp_rw [h_pi] at h_dual
  have h_dual_scaled : (∑ i, (w t i - w (t + 1) i) * g t (EuclideanSpace.basisFun (Fin d) ℝ i))
      - (α t / 2) * ∑ i, (w t i - w (t + 1) i) ^ 2 / z i ≤
      (1 / (2 * α t)) * ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
    have h_mult := mul_le_mul_of_nonneg_left h_dual hα_pos.le
    have h_cancel1 :
        α t * ((1 / α t) * ∑ i, (w t i - w (t + 1) i) *
          g t (EuclideanSpace.basisFun (Fin d) ℝ i)) =
        ∑ i, (w t i - w (t + 1) i) * g t (EuclideanSpace.basisFun (Fin d) ℝ i) := by
      rw [← mul_assoc, mul_one_div_cancel hα_pos.ne', one_mul]
    have h_cancel2 : α t * ((1 / 2 : ℝ) * ∑ i, (w t i - w (t + 1) i) ^ 2 / z i) =
        (α t / 2) * ∑ i, (w t i - w (t + 1) i) ^ 2 / z i := by ring
    have h_cancel3 :
        α t * (((1 / α t) ^ 2 / 2) * ∑ i, z i *
          (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2) =
        (1 / (2 * α t)) * ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
      calc α t * (((1 / α t) ^ 2 / 2) * ∑ i, z i *
            (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2)
        _ = (α t * (1 / (α t ^ 2 * 2))) * ∑ i, z i *
              (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by ring
        _ = (1 / (2 * α t)) * ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
          congr 1
          field_simp [hα_pos.ne']
    rw [mul_sub, h_cancel1, h_cancel2, h_cancel3] at h_mult
    exact h_mult
  linarith [h_decomp, h_dual_scaled]

/-- Cumulative stability bound for LEA: if for each round $t \in [1, T]$,
the iterates and subgradients satisfy the regularity assumptions, then
the sum of stability terms is bounded by the sum of local norm dual bounds. -/
theorem sum_stability_le_dual_norm (α : ℕ → ℝ) (T : ℕ)
    (hα_pos : ∀ t ∈ Ico 1 (T + 1), 0 < α t)
    (hw_pos : ∀ t ∈ Ico 1 (T + 2), ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i) :
    ∃ z : ℕ → EuclideanSpace ℝ (Fin d),
      (∀ t ∈ Ico 1 (T + 1), z t ∈ segment ℝ (w (t + 1)) (w t)) ∧
      ∑ t ∈ Ico 1 (T + 1),
        stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
        ∑ t ∈ Ico 1 (T + 1),
          (1 / (2 * α t)) * ∑ i, z t i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
  have h_choice : ∀ t ∈ Ico 1 (T + 1), ∃ z ∈ segment ℝ (w (t + 1)) (w t),
      stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
        (1 / (2 * α t)) * ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
    intro t ht
    have hw_next_pos : ∀ i, 0 < (w (t + 1) : EuclideanSpace ℝ (Fin d)) i := by
      rw [mem_Ico] at ht
      exact hw_pos (t + 1) (by rw [mem_Ico]; omega)
    have hw_t_pos : ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i := by
      rw [mem_Ico] at ht
      exact hw_pos t (by rw [mem_Ico]; omega)
    exact stability_le_dual_norm (w := w) (g := g) α t (hα_pos t ht) hw_next_pos hw_t_pos
  choose! z hz_seg hz_le using h_choice
  refine ⟨z, hz_seg, sum_le_sum hz_le⟩

/-- Maximum squared coordinate of a linear loss/subgradient $g : \mathbb{R}^d \to \mathbb{R}$:
$$\|g\|_{\infty}^2 = \max_{i=1}^d g(e_i)^2$$ -/
noncomputable abbrev subgradientMaxSq (hd : 0 < d) (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) : ℝ :=
  let h_nonempty : (univ : Finset (Fin d)).Nonempty :=
    Finset.univ_nonempty_iff.mpr (Fin.pos_iff_nonempty.mp hd)
  Finset.univ.sup' h_nonempty (fun i ↦ (g (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2)

/-- One-round stability bound specialized via the fact that $z \in \Delta^{d-1}$ (the simplex),
so $\sum_i z_i (g_t)_i^2 \le \max_i (g_t)_i^2$:
$$\mathrm{stability}_t \le \frac{1}{2\alpha_t} \|g_t\|_\infty^2.$$ -/
theorem stability_le_max_sq (hd : 0 < d) (α : ℕ → ℝ) (t : ℕ) (hα_pos : 0 < α t)
    (hw_next : w (t + 1) ∈ stdSimplex (d := d))
    (hw_next_pos : ∀ i, 0 < (w (t + 1) : EuclideanSpace ℝ (Fin d)) i)
    (hw_t : w t ∈ stdSimplex (d := d))
    (hw_t_pos : ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i) :
    stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
      (1 / (2 * α t)) * subgradientMaxSq hd (g t) := by
  have h_nonempty : (univ : Finset (Fin d)).Nonempty :=
    Finset.univ_nonempty_iff.mpr (Fin.pos_iff_nonempty.mp hd)
  obtain ⟨z, hz_seg, hz_le⟩ :=
    stability_le_dual_norm (w := w) (g := g) α t hα_pos hw_next_pos hw_t_pos
  have hz_simp : z ∈ stdSimplex (d := d) := convex_stdSimplex.segment_subset hw_next hw_t hz_seg
  rw [mem_stdSimplex_iff] at hz_simp
  dsimp [subgradientMaxSq]
  have h_sum_le : ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 ≤
      Finset.univ.sup' h_nonempty (fun i ↦ (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2) := by
    calc ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2
      _ ≤ ∑ i, z i * Finset.univ.sup' h_nonempty
            (fun j ↦ (g t (EuclideanSpace.basisFun (Fin d) ℝ j)) ^ 2) :=
        sum_le_sum fun i hi ↦ mul_le_mul_of_nonneg_left
          (Finset.le_sup' (fun j ↦ (g t (EuclideanSpace.basisFun (Fin d) ℝ j)) ^ 2) hi)
          (hz_simp.1 i)
      _ = (∑ i, z i) * Finset.univ.sup' h_nonempty
            (fun j ↦ (g t (EuclideanSpace.basisFun (Fin d) ℝ j)) ^ 2) := by rw [← sum_mul]
      _ = Finset.univ.sup' h_nonempty
            (fun j ↦ (g t (EuclideanSpace.basisFun (Fin d) ℝ j)) ^ 2) := by rw [hz_simp.2, one_mul]
  have h_factor_nonneg : 0 ≤ 1 / (2 * α t) := by positivity
  have h_dual_le := mul_le_mul_of_nonneg_left h_sum_le h_factor_nonneg
  linarith

/-- Cumulative stability bound for LEA with subgradient max-norm:
$$\sum_{t=1}^T \mathrm{stability}_t \le \sum_{t=1}^T \frac{1}{2\alpha_t} \|g_t\|_\infty^2.$$ -/
theorem sum_stability_le_max_sq (hd : 0 < d) (α : ℕ → ℝ) (T : ℕ)
    (hα_pos : ∀ t ∈ Ico 1 (T + 1), 0 < α t)
    (hw : ∀ t ∈ Ico 1 (T + 2), w t ∈ stdSimplex (d := d))
    (hw_pos : ∀ t ∈ Ico 1 (T + 2), ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i) :
    ∑ t ∈ Ico 1 (T + 1),
      stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
      ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) * subgradientMaxSq hd (g t) := by
  refine sum_le_sum fun t ht ↦ ?_
  have hw_next : w (t + 1) ∈ stdSimplex (d := d) := by
    rw [mem_Ico] at ht
    exact hw (t + 1) (by rw [mem_Ico]; omega)
  have hw_next_pos : ∀ i, 0 < (w (t + 1) : EuclideanSpace ℝ (Fin d)) i := by
    rw [mem_Ico] at ht
    exact hw_pos (t + 1) (by rw [mem_Ico]; omega)
  have hw_t : w t ∈ stdSimplex (d := d) := by
    rw [mem_Ico] at ht
    exact hw t (by rw [mem_Ico]; omega)
  have hw_t_pos : ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i := by
    rw [mem_Ico] at ht
    exact hw_pos t (by rw [mem_Ico]; omega)
  exact stability_le_max_sq hd (w := w) (g := g) α t (hα_pos t ht)
    hw_next hw_next_pos hw_t hw_t_pos

/-! ### Stability Bound via Candidate Maximizer (Jacobsen Formulation 2) -/

/-- One-round stability term evaluated at $w_{t+1} \in \mathcal{X}$ is upper bounded by
the stability objective evaluated at any candidate $\tilde{w}_{t+1} \in \mathcal{X}$ maximizing
$x \mapsto \langle g_t, w_t - x\rangle - D_{\psi_t}(x, w_t)$ over $\mathcal{X}$:
$$\langle g_t, w_t - w_{t+1}\rangle - D_{\psi_t}(w_{t+1}, w_t)
  \le \max_{x \in \mathcal{X}} \left( \langle g_t, w_t - x\rangle - D_{\psi_t}(x, w_t) \right)
  = \langle g_t, w_t - \tilde{w}_{t+1}\rangle - D_{\psi_t}(\tilde{w}_{t+1}, w_t).$$ -/
theorem stability_le_of_isMaxOn (α : ℕ → ℝ) (t : ℕ)
    (X : Set (EuclideanSpace ℝ (Fin d)))
    (hw_next : w (t + 1) ∈ X)
    (w_tilde_next : EuclideanSpace ℝ (Fin d))
    (h_max : IsMaxOn (fun x ↦ (g t) (w t - x) -
      D_[unnormEntropyShifted (α t)](x, w t, unnormEntropyFDeriv (α t) (w t))) X w_tilde_next) :
    stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
      (g t) (w t - w_tilde_next) -
        D_[unnormEntropyShifted (α t)](w_tilde_next, w t, unnormEntropyFDeriv (α t) (w t)) := by
  dsimp [stability]
  exact h_max hw_next

/-- Stability bound for $w_{t+1} \in \mathcal{X}$ derived via candidate
$\tilde{w}_{t+1} \in \mathcal{X}$: bounds the original stability term by the local norm at an
intermediate point $\tilde{z} \in [\tilde{w}_{t+1}, w_t]$ via `stability_le_dual_norm`. -/
theorem stability_le_dual_norm_of_isMaxOn (α : ℕ → ℝ) (t : ℕ) (hα_pos : 0 < α t)
    (X : Set (EuclideanSpace ℝ (Fin d)))
    (hw_next : w (t + 1) ∈ X)
    (w_tilde_next : EuclideanSpace ℝ (Fin d))
    (hw_tilde_pos : ∀ i, 0 < w_tilde_next i)
    (hw_t_pos : ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i)
    (h_max : IsMaxOn (fun x ↦ (g t) (w t - x) -
      D_[unnormEntropyShifted (α t)](x, w t, unnormEntropyFDeriv (α t) (w t))) X w_tilde_next) :
    ∃ z ∈ segment ℝ w_tilde_next (w t),
      stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
        (1 / (2 * α t)) * ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
  obtain ⟨z, hz_seg, hz_le⟩ := stability_le_dual_norm (g := g) (α := α) (t := t)
    (w := fun s ↦ if s = t + 1 then w_tilde_next else w s) hα_pos
    (by intro i; dsimp; rw [ite_eq_left rfl]; exact hw_tilde_pos i)
    (by intro i; dsimp; rw [ite_eq_right (show t ≠ t + 1 by omega)]; exact hw_t_pos i)
  have h_succ : (if t + 1 = t + 1 then w_tilde_next else w (t + 1)) = w_tilde_next :=
    ite_eq_left rfl
  have h_t : (if t = t + 1 then w_tilde_next else w t) = w t :=
    ite_eq_right (show t ≠ t + 1 by omega)
  rw [h_succ, h_t] at hz_seg
  dsimp [stability] at hz_le
  rw [h_succ, h_t] at hz_le
  refine ⟨z, hz_seg, ?_⟩
  have h_le_max := stability_le_of_isMaxOn w g α t X hw_next w_tilde_next h_max
  linarith

/-! ### Unconstrained Exponential Gradient Candidate Maximizer -/

/-- The unconstrained candidate minimizer $\tilde{w}_{t+1}$ in coordinate form:
$$\tilde{w}_{t+1, i} = w_{t, i} \exp\left(-\frac{\langle g_t, e_i \rangle}{\alpha}\right)$$ -/
noncomputable def expGradCandidate (α : ℝ) (wt : EuclideanSpace ℝ (Fin d))
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) : EuclideanSpace ℝ (Fin d) :=
  WithLp.toLp 2 (fun i ↦ wt i * Real.exp (- (g (EuclideanSpace.basisFun (Fin d) ℝ i)) / α))

@[simp]
lemma expGradCandidate_apply (α : ℝ) (wt : EuclideanSpace ℝ (Fin d))
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (i : Fin d) :
    expGradCandidate α wt g i = wt i * Real.exp (- (g (EuclideanSpace.basisFun (Fin d) ℝ i)) / α) :=
  rfl

lemma expGradCandidate_pos (α : ℝ) (wt : EuclideanSpace ℝ (Fin d))
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) (hwt_pos : ∀ i, 0 < wt i) (i : Fin d) :
    0 < expGradCandidate α wt g i :=
  mul_pos (hwt_pos i) (Real.exp_pos _)

/-- When $g_i \ge 0$ and $\alpha > 0$, candidate coordinates are bounded above by $w_{t, i}$. -/
lemma expGradCandidate_le_wt (α : ℝ) (hα_pos : 0 < α)
    (wt : EuclideanSpace ℝ (Fin d)) (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ)
    (hg_nonneg : ∀ i, 0 ≤ g (EuclideanSpace.basisFun (Fin d) ℝ i))
    (hwt_nonneg : ∀ i, 0 ≤ wt i) (i : Fin d) :
    expGradCandidate α wt g i ≤ wt i := by
  dsimp [expGradCandidate]
  have h_div_nonpos : - (g (EuclideanSpace.basisFun (Fin d) ℝ i)) / α ≤ 0 := by
    rw [neg_div, neg_nonpos]
    exact div_nonneg (hg_nonneg i) hα_pos.le
  have h_exp_le_one : Real.exp (- (g (EuclideanSpace.basisFun (Fin d) ℝ i)) / α) ≤ 1 := by
    simpa using Real.exp_le_exp_of_le h_div_nonpos
  calc wt i * Real.exp (- (g (EuclideanSpace.basisFun (Fin d) ℝ i)) / α)
    _ ≤ wt i * 1 := mul_le_mul_of_nonneg_left h_exp_le_one (hwt_nonneg i)
    _ = wt i := mul_one (wt i)

/-- Any point on the segment $[\tilde{w}, w]$ is coordinate-wise bounded above by $w$
when $\tilde{w} \le w$. -/
lemma segment_le_of_le (w_tilde w_t : EuclideanSpace ℝ (Fin d))
    (h_le : ∀ i, w_tilde i ≤ w_t i)
    (z : EuclideanSpace ℝ (Fin d)) (hz : z ∈ segment ℝ w_tilde w_t) (i : Fin d) :
    z i ≤ w_t i := by
  rcases hz with ⟨a, b, ha, hb, hab, rfl⟩
  dsimp
  calc a * w_tilde i + b * w_t i
    _ ≤ a * w_t i + b * w_t i := by
      linarith [mul_le_mul_of_nonneg_left (h_le i) ha]
    _ = (a + b) * w_t i := by ring
    _ = w_t i := by rw [hab, one_mul]

/-- `expGradCandidate` is the global maximizer on the positive orthant
$\mathcal{X} = \{x \in \mathbb{R}^d \mid \forall i, 0 < x_i\}$ of the unconstrained OMD objective
$x \mapsto \langle g_t, w_t - x \rangle - D_{\psi_{\alpha_t}}(x, w_t)$. -/
theorem isMaxOn_expGradCandidate (α : ℝ) (hα_pos : 0 < α)
    (wt : EuclideanSpace ℝ (Fin d)) (hwt_pos : ∀ i, 0 < wt i)
    (g : EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ) :
    IsMaxOn (fun x ↦ g (wt - x) -
      D_[unnormEntropyShifted α](x, wt, unnormEntropyFDeriv α wt))
      {x : EuclideanSpace ℝ (Fin d) | ∀ i, 0 < x i} (expGradCandidate α wt g) := by
  intro x hx
  dsimp
  rw [bregDiv_unnormEntropyShifted, bregDiv_unnormEntropyShifted]
  have h_decomp (y : EuclideanSpace ℝ (Fin d)) (hy : ∀ i, 0 < y i) :
      g (wt - y) - D_[unnormEntropy α](y, wt, unnormEntropyFDeriv α wt) =
        ∑ i, ((wt i - y i) * g (EuclideanSpace.basisFun (Fin d) ℝ i) -
          α * (y i * Real.log (y i / wt i) - y i + wt i)) := by
    have h_g : g (wt - y) = ∑ i, (wt i - y i) * g (EuclideanSpace.basisFun (Fin d) ℝ i) := by
      have h_vec : wt - y = ∑ i, (wt i - y i) • EuclideanSpace.basisFun (Fin d) ℝ i := by
        ext i; simp [EuclideanSpace.basisFun_apply, Pi.single_apply]
      nth_rw 1 [h_vec]; rw [map_sum]; exact sum_congr rfl fun i _ ↦ by rw [map_smul, smul_eq_mul]
    have h_breg : D_[unnormEntropy α](y, wt, unnormEntropyFDeriv α wt) =
        α * ∑ i, (y i * Real.log (y i / wt i) - y i + wt i) := by
      dsimp [bregDiv, unnormEntropy]; rw [unnormEntropyFDeriv_apply]
      have h_sum : (∑ i, (y i * Real.log (y i) - y i)) - (∑ i, (wt i * Real.log (wt i) - wt i)) -
          ∑ i, (y - wt) i * Real.log (wt i) = ∑ i, (y i * Real.log (y i / wt i) - y i + wt i) := by
        rw [← sum_sub_distrib, ← sum_sub_distrib]
        refine sum_congr rfl fun i _ ↦ by
          rw [Real.log_div (hy i).ne' (hwt_pos i).ne']
          dsimp; ring
      calc α * ∑ i, (y i * Real.log (y i) - y i) - α * ∑ i, (wt i * Real.log (wt i) - wt i) -
            α * ∑ i, (y - wt) i * Real.log (wt i)
        _ = α * ((∑ i, (y i * Real.log (y i) - y i)) - (∑ i, (wt i * Real.log (wt i) - wt i)) -
            ∑ i, (y - wt) i * Real.log (wt i)) := by ring
        _ = α * ∑ i, (y i * Real.log (y i / wt i) - y i + wt i) := by rw [h_sum]
    rw [h_g, h_breg, mul_sum, ← sum_sub_distrib]
  rw [h_decomp x hx, h_decomp (expGradCandidate α wt g) (expGradCandidate_pos α wt g hwt_pos)]
  refine sum_le_sum fun i _ ↦ ?_
  set gi := g (EuclideanSpace.basisFun (Fin d) ℝ i)
  set wi := wt i
  set xi := x i
  set wi_cand := expGradCandidate α wt g i
  have h_cand_pos : 0 < wi_cand := expGradCandidate_pos α wt g hwt_pos i
  have h_log_cand : Real.log (wi_cand / wi) = - gi / α := by
    change Real.log ((wi * Real.exp (- gi / α)) / wi) = - gi / α
    rw [mul_div_cancel_left₀ _ (hwt_pos i).ne', Real.log_exp]
  have h_log_split : Real.log (xi / wi) = Real.log (xi / wi_cand) - gi / α := by
    have h_prod : Real.log (xi / wi) = Real.log (xi / wi_cand) + Real.log (wi_cand / wi) := by
      rw [← Real.log_mul (div_ne_zero (hx i).ne' h_cand_pos.ne')
            (div_ne_zero h_cand_pos.ne' (hwt_pos i).ne')]
      congr 1
      have : (xi / wi_cand) * (wi_cand / wi) = (xi / wi) * (wi_cand / wi_cand) := by ring
      rw [this, div_self h_cand_pos.ne', mul_one]
    rw [h_prod, h_log_cand, neg_div, sub_eq_add_neg]
  have h_diff : ((wi - wi_cand) * gi - α * (wi_cand * Real.log (wi_cand / wi) - wi_cand + wi)) -
      ((wi - xi) * gi - α * (xi * Real.log (xi / wi) - xi + wi)) =
      α * (xi * Real.log (xi / wi_cand) - xi + wi_cand) := by
    rw [h_log_cand, h_log_split]
    have h_c1 : α * (wi_cand * (-gi / α)) = - wi_cand * gi := by
      calc α * (wi_cand * (-gi / α)) = (α * (1 / α)) * (wi_cand * (-gi)) := by ring
      _ = 1 * (wi_cand * (-gi)) := by rw [mul_one_div_cancel hα_pos.ne']
      _ = - wi_cand * gi := by ring
    have h_c2 : α * (xi * (-gi / α)) = - xi * gi := by
      calc α * (xi * (-gi / α)) = (α * (1 / α)) * (xi * (-gi)) := by ring
      _ = 1 * (xi * (-gi)) := by rw [mul_one_div_cancel hα_pos.ne']
      _ = - xi * gi := by ring
    calc ((wi - wi_cand) * gi - α * (wi_cand * (-gi / α) - wi_cand + wi)) -
          ((wi - xi) * gi - α * (xi * (Real.log (xi / wi_cand) - gi / α) - xi + wi))
      _ = ((wi - wi_cand) * gi - (α * (wi_cand * (-gi / α)) - α * wi_cand + α * wi)) -
          ((wi - xi) * gi - (α * (xi * Real.log (xi / wi_cand)) + α * (xi * (-gi / α)) -
            α * xi + α * wi)) := by ring
      _ = α * (xi * Real.log (xi / wi_cand) - xi + wi_cand) := by rw [h_c1, h_c2]; ring
  have h_nonneg : 0 ≤ α * (xi * Real.log (xi / wi_cand) - xi + wi_cand) := by
    set u := xi / wi_cand
    have hu_pos : 0 < u := div_pos (hx i) h_cand_pos
    have h_log_inv : -Real.log u ≤ u⁻¹ - 1 := by
      have := Real.log_le_sub_one_of_pos (inv_pos.mpr hu_pos)
      rwa [Real.log_inv] at this
    have h_u : 0 ≤ u * Real.log u - u + 1 := by
      have := mul_le_mul_of_nonneg_left h_log_inv hu_pos.le
      have h_ui : u * u⁻¹ = 1 := mul_inv_cancel₀ hu_pos.ne'
      calc 0 = - (u * (u⁻¹ - 1)) - u + 1 := by rw [mul_sub, h_ui, mul_one]; ring
        _ ≤ - (u * (-Real.log u)) - u + 1 := by linarith [this]
        _ = u * Real.log u - u + 1 := by ring
    have h_ident : wi_cand * (u * Real.log u - u + 1) =
        xi * Real.log (xi / wi_cand) - xi + wi_cand := by
      calc wi_cand * (u * Real.log u - u + 1)
        _ = (wi_cand * (xi / wi_cand)) * Real.log (xi / wi_cand) -
              wi_cand * (xi / wi_cand) + wi_cand * 1 := by ring
        _ = xi * Real.log (xi / wi_cand) - xi + wi_cand := by
          rw [mul_div_cancel₀ _ h_cand_pos.ne', mul_one]
    rw [← h_ident]
    exact mul_nonneg hα_pos.le (mul_nonneg h_cand_pos.le h_u)
  linarith [h_diff, h_nonneg]

/-- Stability bound for strictly positive iterates $w_t, w_{t+1} > 0$ when losses
are non-negative $g_t \ge 0$: bounds the stability term directly with $w_t$ instead of
the intermediate point $z$. -/
theorem stability_le_dual_norm_wt (α : ℕ → ℝ) (t : ℕ) (hα_pos : 0 < α t)
    (hw_next_pos : ∀ i, 0 < (w (t + 1) : EuclideanSpace ℝ (Fin d)) i)
    (hw_t_pos : ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i)
    (hg_nonneg : ∀ i, 0 ≤ g t (EuclideanSpace.basisFun (Fin d) ℝ i)) :
    stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
      (1 / (2 * α t)) * ∑ i, (w t : EuclideanSpace ℝ (Fin d)) i *
        (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
  have h_max := isMaxOn_expGradCandidate (α t) hα_pos (w t) hw_t_pos (g t)
  have hw_next_mem : w (t + 1) ∈ {x : EuclideanSpace ℝ (Fin d) | ∀ i, 0 < x i} := hw_next_pos
  obtain ⟨z, hz_seg, hz_le⟩ := stability_le_dual_norm_of_isMaxOn w g α t hα_pos
    {x | ∀ i, 0 < x i} hw_next_mem (expGradCandidate (α t) (w t) (g t))
    (expGradCandidate_pos (α t) (w t) (g t) hw_t_pos) hw_t_pos h_max
  have hz_le_wt (i : Fin d) : z i ≤ (w t : EuclideanSpace ℝ (Fin d)) i := by
    refine segment_le_of_le (expGradCandidate (α t) (w t) (g t)) (w t) ?_ z hz_seg i
    intro j
    exact expGradCandidate_le_wt (α t) hα_pos (w t) (g t) hg_nonneg (fun k ↦ (hw_t_pos k).le) j
  have h_sum_le : ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 ≤
      ∑ i, (w t : EuclideanSpace ℝ (Fin d)) i *
        (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
    refine sum_le_sum fun i _ ↦ ?_
    exact mul_le_mul_of_nonneg_right (hz_le_wt i) (sq_nonneg _)
  have h_factor_pos : 0 ≤ 1 / (2 * α t) := by
    refine div_nonneg zero_le_one (mul_nonneg (by norm_num) hα_pos.le)
  calc stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t
    _ ≤ (1 / (2 * α t)) * ∑ i, z i * (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := hz_le
    _ ≤ (1 / (2 * α t)) * ∑ i, (w t : EuclideanSpace ℝ (Fin d)) i *
          (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 :=
        mul_le_mul_of_nonneg_left h_sum_le h_factor_pos
/-- Multi-round cumulative stability bound for strictly positive iterates $w_t, w_{t+1} > 0$
when losses are non-negative $g_t \ge 0$:
$$\sum_{t=1}^T \mathrm{stability}_t \le
  \sum_{t=1}^T \frac{1}{2\alpha_t} \sum_{i=1}^d w_{t, i} (g_t)_i^2.$$ -/
theorem sum_stability_le_dual_norm_wt (α : ℕ → ℝ) (T : ℕ)
    (hα_pos : ∀ t ∈ Ico 1 (T + 1), 0 < α t)
    (hw_pos : ∀ t ∈ Ico 1 (T + 2), ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i)
    (hg_nonneg : ∀ t ∈ Ico 1 (T + 1),
      ∀ i, 0 ≤ g t (EuclideanSpace.basisFun (Fin d) ℝ i)) :
    ∑ t ∈ Ico 1 (T + 1),
      stability (fun s ↦ unnormEntropyShifted (α s)) (fun s ↦ unnormEntropyFDeriv (α s)) w g t ≤
      ∑ t ∈ Ico 1 (T + 1), (1 / (2 * α t)) *
        ∑ i, (w t : EuclideanSpace ℝ (Fin d)) i *
          (g t (EuclideanSpace.basisFun (Fin d) ℝ i)) ^ 2 := by
  refine sum_le_sum fun t ht ↦ ?_
  have hw_next_pos : ∀ i, 0 < (w (t + 1) : EuclideanSpace ℝ (Fin d)) i := by
    rw [mem_Ico] at ht
    exact hw_pos (t + 1) (by rw [mem_Ico]; omega)
  have hw_t_pos : ∀ i, 0 < (w t : EuclideanSpace ℝ (Fin d)) i := by
    rw [mem_Ico] at ht
    exact hw_pos t (by rw [mem_Ico]; omega)
  exact stability_le_dual_norm_wt (w := w) (g := g) α t (hα_pos t ht)
    hw_next_pos hw_t_pos (hg_nonneg t ht)

end Online.OCO.LEA.OMD
