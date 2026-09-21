/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.Online.OnlineConvexOptimization.Algorithms.OMD.StrongLemmas
public import LeanMachineLearning.Online.OnlineConvexOptimization.Algorithms.OMD.bounds.OptimalityTerm
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.regularizer
public import LeanMachineLearning.Online.OnlineConvexOptimization.LEA.stability

/-!
# Regret Bound for Linear Exponential Adaptation (LEA)

This file sets up the regret decomposition for the Linear Exponential Adaptation (LEA)
algorithm, instantiating the Strong Centered Online Mirror Descent framework where:
- The regularizer $\psi_t = \psi$ is constant (time-invariant unnormalized negative entropy).
- There is no centering: $\varphi_t = 0$ (and $\nabla \varphi_t = 0$).
- There is no adjustment: $\tilde{w}_t = w_t$ (`adjust = id`), so the adjustment drift vanishes.
- The regularizer shift $\Delta_t = D_{\psi - \psi} = 0$ vanishes.
- The optimality deficit $\mu_t$ vanishes when $w_{t+1}$ satisfies first-order optimality.

## Vanishing terms:
* `adjustmentTerm_id`: Adjustment drift is $0$ when $\tilde{w}_{t+1} = w_{t+1}$.
* `shiftTerm_const`: Regularizer shift is $0$ when $\psi_t$ is independent of $t$.
* `centeringTerm_zero`: Centering surplus is $0$ when $\varphi_t = 0$.
* `pathlengthTerm_const_anchor`: Pathlength is $0$ when the comparator is fixed $u_t = u$
  or anchors match.
-/

open scoped BigOperators Bregman Topology
open Finset

@[expose] public section

namespace Analysis.Convex

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-! ### Vanishing Terms Lemmas for Constant / Uncentered OMD -/

/-- When there is no adjustment step ($\tilde{w}_{t+1} = w_{t+1}$),
the adjustment drift term is $0$. -/
lemma adjustmentTerm_eq_zero_of_eq (ψ : ℕ → E → ℝ) (gψ : ℕ → E → (E →L[ℝ] ℝ))
    (u w w_tilde : ℕ → E) (t : ℕ)
    (h : w_tilde (t + 1) = w (t + 1)) :
    adjustmentTerm ψ gψ u w w_tilde t = 0 := by
  dsimp [adjustmentTerm]
  rw [h, sub_self]

/-- When the regularizer is time-invariant ($\psi_{t+1} = \psi_t$ and $g\psi_{t+1} = g\psi_t$),
the regularizer shift term vanishes identically. -/
lemma shiftTerm_eq_zero_of_const (ψ : ℕ → E → ℝ) (gψ : ℕ → E → (E →L[ℝ] ℝ))
    (w w_tilde : ℕ → E) (t : ℕ)
    (hψ : ψ (t + 1) = ψ t) (hgψ : gψ (t + 1) = gψ t) :
    shiftTerm ψ gψ w w_tilde t = 0 := by
  dsimp [shiftTerm, Analysis.Convex.bregDiv]
  simp [hψ, hgψ]

/-- When the centering potential is identically zero ($\varphi_t = 0, \nabla \varphi_t = 0$),
the centering term is $0$. -/
lemma centeringTerm_eq_zero_of_zero (φ : ℕ → E → ℝ) (gφ : ℕ → E → (E →L[ℝ] ℝ))
    (u w : ℕ → E) (t : ℕ)
    (hφ : φ t = 0) (hgφ : gφ t = 0) :
    centeringTerm φ gφ u w t = 0 := by
  dsimp [centeringTerm, Analysis.Convex.bregDiv]
  simp [hφ, hgφ]

/-- When the comparator is static ($u_t = u$ for all $t$), the pathlength term
telescopes or vanishes when anchors are fixed or $u_{t-1} = u_t$. -/
lemma pathlengthTerm_eq_zero_of_static_comparator (gψ : ℕ → E → (E →L[ℝ] ℝ))
    (u w_tilde : ℕ → E) (t : ℕ)
    (hu : u (t - 1) = u t) :
    pathlengthTerm gψ u w_tilde t = 0 := by
  dsimp [pathlengthTerm]
  rw [hu, sub_self, map_zero]

/-! ### Multi-Round Vanishing Sums -/

lemma sum_adjustmentTerm_eq_zero (ψ : ℕ → E → ℝ) (gψ : ℕ → E → (E →L[ℝ] ℝ))
    (u w w_tilde : ℕ → E) (T : ℕ)
    (h_adj : ∀ t ∈ Ico 1 (T + 1), w_tilde (t + 1) = w (t + 1)) :
    ∑ t ∈ Ico 1 (T + 1), adjustmentTerm ψ gψ u w w_tilde t = 0 := by
  apply Finset.sum_eq_zero
  intro t ht
  exact adjustmentTerm_eq_zero_of_eq ψ gψ u w w_tilde t (h_adj t ht)

lemma sum_shiftTerm_eq_zero (ψ : ℕ → E → ℝ) (gψ : ℕ → E → (E →L[ℝ] ℝ))
    (w w_tilde : ℕ → E) (T : ℕ)
    (hψ : ∀ t ∈ Ico 1 (T + 1), ψ (t + 1) = ψ t)
    (hgψ : ∀ t ∈ Ico 1 (T + 1), gψ (t + 1) = gψ t) :
    ∑ t ∈ Ico 1 (T + 1), shiftTerm ψ gψ w w_tilde t = 0 := by
  apply Finset.sum_eq_zero
  intro t ht
  exact shiftTerm_eq_zero_of_const ψ gψ w w_tilde t (hψ t ht) (hgψ t ht)

lemma sum_centeringTerm_eq_zero (φ : ℕ → E → ℝ) (gφ : ℕ → E → (E →L[ℝ] ℝ))
    (u w : ℕ → E) (T : ℕ)
    (hφ : ∀ t ∈ Ico 1 (T + 1), φ t = 0)
    (hgφ : ∀ t ∈ Ico 1 (T + 1), gφ t = 0) :
    ∑ t ∈ Ico 1 (T + 1), centeringTerm φ gφ u w t = 0 := by
  apply Finset.sum_eq_zero
  intro t ht
  exact centeringTerm_eq_zero_of_zero φ gφ u w t (hφ t ht) (hgφ t ht)

lemma sum_pathlengthTerm_eq_zero_of_static (gψ : ℕ → E → (E →L[ℝ] ℝ))
    (u w_tilde : ℕ → E) (T : ℕ)
    (hu : ∀ t ∈ Ico 1 (T + 1), u (t - 1) = u t) :
    ∑ t ∈ Ico 1 (T + 1), pathlengthTerm gψ u w_tilde t = 0 := by
  apply Finset.sum_eq_zero
  intro t ht
  exact pathlengthTerm_eq_zero_of_static_comparator gψ u w_tilde t (hu t ht)

lemma sum_optimalityTerm_eq_zero (η : ℝ) (gψ : ℕ → E → (E →L[ℝ] ℝ))
    (gφ : ℕ → E → (E →L[ℝ] ℝ)) (u w w_tilde : ℕ → E) (g : ℕ → (E →L[ℝ] ℝ)) (T : ℕ)
    (h_opt : ∀ t ∈ Ico 1 (T + 1), optimalityTerm η gψ gφ u w w_tilde g t = 0) :
    ∑ t ∈ Ico 1 (T + 1), optimalityTerm η gψ gφ u w w_tilde g t = 0 := by
  exact Finset.sum_eq_zero h_opt

/-! ### Simplified Regret Identity for Time-Invariant Uncentered OMD -/

/-- Regret identity for time-invariant uncentered OMD with static comparator:
all adjustment, shift, centering, pathlength, and optimality terms vanish,
leaving only boundary divergence, stability, and linearization error. -/
theorem strongCenteredMirrorDescent_simplified (η : ℝ) (ψ : ℕ → E → ℝ) (φ : ℕ → E → ℝ)
    (gψ gφ : ℕ → E → (E →L[ℝ] ℝ))
    (u w w_tilde : ℕ → E) (g : ℕ → (E →L[ℝ] ℝ)) (l : ℕ → E → ℝ) (T : ℕ)
    (h_adj : ∀ t ∈ Ico 1 (T + 1), w_tilde (t + 1) = w (t + 1))
    (h_shift_ψ : ∀ t ∈ Ico 1 (T + 1), ψ (t + 1) = ψ t)
    (h_shift_gψ : ∀ t ∈ Ico 1 (T + 1), gψ (t + 1) = gψ t)
    (h_φ : ∀ t ∈ Ico 1 (T + 1), φ t = 0)
    (h_gφ : ∀ t ∈ Ico 1 (T + 1), gφ t = 0)
    (h_path : ∀ t ∈ Ico 1 (T + 1), u (t - 1) = u t)
    (h_opt : ∀ t ∈ Ico 1 (T + 1), optimalityTerm η gψ gφ u w w_tilde g t = 0) :
    η * (∑ t ∈ Ico 1 (T + 1), (l t (w_tilde t) - l t (u t))) =
    D_[ψ (T+1)](u T, w_tilde 1, gψ (T+1) (w_tilde 1))
    - D_[ψ (T+1)](u T, w_tilde (T + 1), gψ (T+1) (w_tilde (T + 1)))
    + (∑ t ∈ Ico 1 (T + 1), stabilityTerm η ψ gψ w w_tilde l t)
    + η * (∑ t ∈ Ico 1 (T + 1), linearizationTerm u w g l t) := by
  have h_master := strongCenteredMirrorDescent η ψ φ gψ gφ u w w_tilde g l T
  rw [sum_adjustmentTerm_eq_zero ψ gψ u w w_tilde T h_adj] at h_master
  rw [sum_shiftTerm_eq_zero ψ gψ w w_tilde T h_shift_ψ h_shift_gψ] at h_master
  rw [sum_centeringTerm_eq_zero φ gφ u w T h_φ h_gφ] at h_master
  rw [sum_pathlengthTerm_eq_zero_of_static gψ u w_tilde T h_path] at h_master
  rw [sum_optimalityTerm_eq_zero η gψ gφ u w w_tilde g T h_opt] at h_master
  linarith [h_master]

/-! ### Non-Negativity of Regret Components -/

/-- The linearization error $- D_{l_t}(u_t, w_{t+1}, g_t)$ is non-positive, meaning
subtracting or discarding it only upper-bounds the regret (or equivalently,
$D_{l_t}(u_t, w_{t+1}, g_t) \ge 0$ when $g_t \in \partial l_t(w_{t+1})$). -/
lemma linearizationTerm_nonpos_of_subgradient (l : ℕ → E → ℝ) (g : ℕ → (E →L[ℝ] ℝ))
    (u w : ℕ → E) (t : ℕ) {X : Set E} (hu : u t ∈ X)
    (hg : HasSubgradientWithinAt (l t) (g t) X (w (t + 1))) :
    linearizationTerm u w g l t ≤ 0 := by
  dsimp [linearizationTerm]
  have h_div_nonneg : 0 ≤ D_[l t](u t, w (t + 1), g t) := hg (u t) hu
  linarith

lemma sum_linearizationTerm_nonpos_of_subgradient (l : ℕ → E → ℝ) (g : ℕ → (E →L[ℝ] ℝ))
    (u w : ℕ → E) (T : ℕ) {X : Set E}
    (hu : ∀ t ∈ Ico 1 (T + 1), u t ∈ X)
    (hg : ∀ t ∈ Ico 1 (T + 1), HasSubgradientWithinAt (l t) (g t) X (w (t + 1))) :
    ∑ t ∈ Ico 1 (T + 1), linearizationTerm u w g l t ≤ 0 := by
  apply Finset.sum_nonpos
  intro t ht
  exact linearizationTerm_nonpos_of_subgradient l g u w t (hu t ht) (hg t ht)

/-- Boundary divergence non-negativity: discarding $-D_{\psi}(u, \tilde{w}_{T+1})$
yields an upper bound. -/
lemma neg_bregDiv_nonpos {ψ_end : E → ℝ} {gψ_end : E →L[ℝ] ℝ} {u_end w_end : E} {X : Set E}
    (hu : u_end ∈ X) (hψ : HasSubgradientWithinAt ψ_end gψ_end X w_end) :
    - D_[ψ_end](u_end, w_end, gψ_end) ≤ 0 := by
  have := hψ u_end hu
  linarith

/-! ### Overall Regret Upper Bound -/

/-- Master regret upper bound for time-invariant uncentered OMD:
when the linearization errors are non-positive and the end divergence is non-negative,
the cumulative regret is upper-bounded by the initial divergence plus the cumulative stability. -/
theorem strongCenteredMirrorDescent_regret_le (η : ℝ) (hη : 0 < η)
    (ψ : ℕ → E → ℝ) (φ : ℕ → E → ℝ) (gψ gφ : ℕ → E → (E →L[ℝ] ℝ))
    (u w w_tilde : ℕ → E) (g : ℕ → (E →L[ℝ] ℝ)) (l : ℕ → E → ℝ) (T : ℕ) {X : Set E}
    (h_adj : ∀ t ∈ Ico 1 (T + 1), w_tilde (t + 1) = w (t + 1))
    (h_shift_ψ : ∀ t ∈ Ico 1 (T + 1), ψ (t + 1) = ψ t)
    (h_shift_gψ : ∀ t ∈ Ico 1 (T + 1), gψ (t + 1) = gψ t)
    (h_φ : ∀ t ∈ Ico 1 (T + 1), φ t = 0)
    (h_gφ : ∀ t ∈ Ico 1 (T + 1), gφ t = 0)
    (h_path : ∀ t ∈ Ico 1 (T + 1), u (t - 1) = u t)
    (h_opt : ∀ t ∈ Ico 1 (T + 1), optimalityTerm η gψ gφ u w w_tilde g t = 0)
    (hu : ∀ t ∈ Ico 1 (T + 1), u t ∈ X)
    (hg : ∀ t ∈ Ico 1 (T + 1), HasSubgradientWithinAt (l t) (g t) X (w (t + 1)))
    (hu_end : u T ∈ X)
    (hψ_end : HasSubgradientWithinAt (ψ (T + 1))
      (gψ (T + 1) (w_tilde (T + 1))) X (w_tilde (T + 1))) :
    η * (∑ t ∈ Ico 1 (T + 1), (l t (w_tilde t) - l t (u t))) ≤
    D_[ψ (T + 1)](u T, w_tilde 1, gψ (T + 1) (w_tilde 1))
    + (∑ t ∈ Ico 1 (T + 1), stabilityTerm η ψ gψ w w_tilde l t) := by
  have h_eq := strongCenteredMirrorDescent_simplified η ψ φ gψ gφ u w w_tilde g l T
    h_adj h_shift_ψ h_shift_gψ h_φ h_gφ h_path h_opt
  have h_lin_sum := sum_linearizationTerm_nonpos_of_subgradient l g u w T hu hg
  have h_lin_mul : η * (∑ t ∈ Ico 1 (T + 1), linearizationTerm u w g l t) ≤ 0 :=
    mul_nonpos_of_nonneg_of_nonpos hη.le h_lin_sum
  have h_div_end := neg_bregDiv_nonpos hu_end hψ_end
  linarith

/-! ### Multi-Round Stability Sum Bound -/

/-- Sum of stability terms over rounds $t \in [1, T]$ bounded by the cumulative local dual norm. -/
lemma sum_stabilityTerm_le {d : ℕ} (η : ℝ) (hη : 0 ≤ η)
    (w w_tilde : ℕ → EuclideanSpace ℝ (Fin d))
    (l : ℕ → EuclideanSpace ℝ (Fin d) → ℝ)
    (g_val : ℕ → Fin d → ℝ)
    (z : ℕ → EuclideanSpace ℝ (Fin d))
    (T : ℕ)
    (hz : ∀ t ∈ Ico 1 (T + 1), ∀ i, 0 < z t i)
    (h_conv : ∀ t ∈ Ico 1 (T + 1),
      l t (w_tilde t) - l t (w (t + 1)) ≤ ∑ i, (w_tilde t i - w (t + 1) i) * g_val t i)
    (h_breg : ∀ t ∈ Ico 1 (T + 1),
      D_[unnormEntropy](w (t + 1), w_tilde t, unnormEntropyFDeriv (w_tilde t)) =
        (1 / 2 : ℝ) * ∑ i, (w (t + 1) i - w_tilde t i) ^ 2 / z t i) :
    ∑ t ∈ Ico 1 (T + 1),
      stabilityTerm η (fun _ ↦ unnormEntropy) (fun _ ↦ unnormEntropyFDeriv) w w_tilde l t ≤
      (η ^ 2 / 2) * ∑ t ∈ Ico 1 (T + 1), ∑ i, z t i * (g_val t i) ^ 2 := by
  have h_le_t : ∀ t ∈ Ico 1 (T + 1),
      stabilityTerm η (fun _ ↦ unnormEntropy) (fun _ ↦ unnormEntropyFDeriv) w w_tilde l t ≤
      (η ^ 2 / 2) * ∑ i, z t i * (g_val t i) ^ 2 := by
    intro t ht
    dsimp [stabilityTerm]
    exact lea_stability_le η hη (w_tilde t) (w (t + 1)) (l t (w_tilde t)) (l t (w (t + 1)))
      (g_val t) (h_conv t ht) (z t) (hz t ht) (h_breg t ht)
  have h_sum := Finset.sum_le_sum h_le_t
  rw [← Finset.mul_sum] at h_sum
  exact h_sum

/-! ### Final LEA Regret Bound -/

/-- **Linear Exponential Adaptation (LEA) Master Regret Bound**:
For any static comparator $u \in \mathcal{X}$, convex losses $l_t$, and unnormalized entropy
regularizer $\psi$, the cumulative regret satisfies:
$$\eta \sum_{t=1}^T (l_t(w_t) - l_t(u)) \le D_\psi(u, w_1) +
  \frac{\eta^2}{2} \sum_{t=1}^T \sum_{i=1}^d z_{t, i} (g_t)_i^2$$ -/
theorem lea_regret_bound {d : ℕ} (η : ℝ) (hη : 0 < η)
    (u w w_tilde : ℕ → EuclideanSpace ℝ (Fin d))
    (g : ℕ → (EuclideanSpace ℝ (Fin d) →L[ℝ] ℝ))
    (g_val : ℕ → Fin d → ℝ)
    (l : ℕ → EuclideanSpace ℝ (Fin d) → ℝ)
    (z : ℕ → EuclideanSpace ℝ (Fin d)) (T : ℕ)
    {X : Set (EuclideanSpace ℝ (Fin d))}
    (h_adj : ∀ t ∈ Ico 1 (T + 1), w_tilde (t + 1) = w (t + 1))
    (h_path : ∀ t ∈ Ico 1 (T + 1), u (t - 1) = u t)
    (h_opt : ∀ t ∈ Ico 1 (T + 1),
      optimalityTerm η (fun _ ↦ unnormEntropyFDeriv) (fun _ ↦ 0) u w w_tilde g t = 0)
    (hu : ∀ t ∈ Ico 1 (T + 1), u t ∈ X)
    (hg : ∀ t ∈ Ico 1 (T + 1), HasSubgradientWithinAt (l t) (g t) X (w (t + 1)))
    (hu_end : u T ∈ X)
    (hψ_end : HasSubgradientWithinAt unnormEntropy
      (unnormEntropyFDeriv (w_tilde (T + 1))) X (w_tilde (T + 1)))
    (hz : ∀ t ∈ Ico 1 (T + 1), ∀ i, 0 < z t i)
    (h_conv : ∀ t ∈ Ico 1 (T + 1),
      l t (w_tilde t) - l t (w (t + 1)) ≤ ∑ i, (w_tilde t i - w (t + 1) i) * g_val t i)
    (h_breg : ∀ t ∈ Ico 1 (T + 1),
      D_[unnormEntropy](w (t + 1), w_tilde t, unnormEntropyFDeriv (w_tilde t)) =
        (1 / 2 : ℝ) * ∑ i, (w (t + 1) i - w_tilde t i) ^ 2 / z t i) :
    η * (∑ t ∈ Ico 1 (T + 1), (l t (w_tilde t) - l t (u t))) ≤
    D_[unnormEntropy](u T, w_tilde 1, unnormEntropyFDeriv (w_tilde 1))
    + (η ^ 2 / 2) * ∑ t ∈ Ico 1 (T + 1), ∑ i, z t i * (g_val t i) ^ 2 := by
  have h_regret := strongCenteredMirrorDescent_regret_le η hη
    (fun _ ↦ unnormEntropy) (fun _ ↦ 0) (fun _ ↦ unnormEntropyFDeriv) (fun _ ↦ 0)
    u w w_tilde g l T
    h_adj (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl) (fun _ _ ↦ rfl)
    h_path h_opt hu hg hu_end hψ_end
  have h_stab := sum_stabilityTerm_le η hη.le w w_tilde l g_val z T hz h_conv h_breg
  linarith

end Analysis.Convex
