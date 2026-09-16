/-
Copyright (c) 2025 Rémy Degenne. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Rémy Degenne
-/
module

public import LeanMachineLearning.Online.Bandit.Algorithms.UCB
public import LeanMachineLearning.Online.Bandit.SumRewards

/-!
# Regret bounds for the UCB algorithm

-/

@[expose] public section

open MeasureTheory ProbabilityTheory Filter Real Finset Learning

open scoped ENNReal NNReal

namespace Bandits

namespace UCB

variable {K : ℕ} [NeZero K] {c : ℝ} {ν : Kernel (Fin K) ℝ} [IsMarkovKernel ν]
  {Ω : Type*} {mΩ : MeasurableSpace Ω} {P : Measure Ω} [IsProbabilityMeasure P]
  {O : ℕ → Ω → Unit} {A : ℕ → Ω → Fin K} {R : ℕ → Ω → ℝ}
  {σ2 : ℝ≥0} {n : ℕ} {ω : Ω}

omit [IsMarkovKernel ν] in
/-- If the means of the best arm and of arm `b` lie in their confidence intervals and the UCB index
of `b` is at least that of the best arm, then the gap of `b` is at most twice its confidence
width. -/
lemma gap_le_two_mul_ucbWidth {b : Fin K}
    (h_best : (ν (bestArm ν))[id] ≤ empMean A R (bestArm ν) n ω + ucbWidth A c (bestArm ν) n ω)
    (h_arm : empMean A R b n ω - ucbWidth A c b n ω ≤ (ν b)[id])
    (h_le : empMean A R (bestArm ν) n ω + ucbWidth A c (bestArm ν) n ω ≤
      empMean A R b n ω + ucbWidth A c b n ω) :
    gap ν b ≤ 2 * ucbWidth A c b n ω := by
  rw [gap_eq_bestArm_sub, sub_le_iff_le_add']
  calc (ν (bestArm ν))[id]
  _ ≤ empMean A R (bestArm ν) n ω + ucbWidth A c (bestArm ν) n ω := h_best
  _ ≤ empMean A R b n ω + ucbWidth A c b n ω := h_le
  _ ≤ (ν b)[id] + 2 * ucbWidth A c b n ω := by
    rw [two_mul, ← add_assoc]
    gcongr
    rwa [sub_le_iff_le_add] at h_arm

omit [IsMarkovKernel ν] in
/-- If the means of the best arm and of arm `b` lie in their confidence intervals and the UCB index
of `b` is at least that of the best arm, then the number of pulls of `b` is at most
`8 * c * log (n + 1) / gap ν b ^ 2`. -/
lemma pullCount_le_of_ucbIndex_le (hc : 0 ≤ c) {b : Fin K}
    (h_best : (ν (bestArm ν))[id] ≤ empMean A R (bestArm ν) n ω + ucbWidth A c (bestArm ν) n ω)
    (h_arm : empMean A R b n ω - ucbWidth A c b n ω ≤ (ν b)[id])
    (h_le : empMean A R (bestArm ν) n ω + ucbWidth A c (bestArm ν) n ω ≤
      empMean A R b n ω + ucbWidth A c b n ω)
    (h_gap_pos : 0 < gap ν b) (h_pull_pos : 0 < pullCount A b n ω) :
    pullCount A b n ω ≤ 8 * c * log (n + 1) / gap ν b ^ 2 := by
  have h_gap_le := gap_le_two_mul_ucbWidth h_best h_arm h_le
  rw [ucbWidth] at h_gap_le
  have h2 : (gap ν b) ^ 2 ≤ (2 * √(2 * c * log (n + 1) / pullCount A b n ω)) ^ 2 := by
    gcongr
  rw [mul_pow, sq_sqrt] at h2
  · have : (2 : ℝ) ^ 2 = 4 := by norm_num
    rw [this] at h2
    field_simp at h2 ⊢
    grind
  · have : 0 ≤ log (n + 1) := by simp [log_nonneg]
    positivity

/-- The probability that the UCB index of arm `a` is below its mean is at most
`1 / (n + 1) ^ (c - 1)`. -/
lemma prob_ucbIndex_le {alg : Algorithm Unit (Fin K) ℝ}
    (h : IsAlgEnvSeq O A R alg (stationaryEnv ν) P)
    (hν : ∀ a, HasSubgaussianMGF (fun x ↦ x - (ν a)[id]) σ2 (ν a))
    (hσ2 : σ2 ≠ 0) (hc : 0 ≤ c) (a : Fin K) (n : ℕ) :
    P {h | 0 < pullCount A a n h ∧ empMean A R a n h + ucbWidth A (c * σ2) a n h ≤ (ν a)[id]} ≤
      1 / (n + 1) ^ (c - 1) := by
  have h_le := prob_pullCount_pos_and_le h a n
    (p := fun k x ↦ x / k + √(2 * c * σ2 * log (n + 1) / k) ≤ (ν a)[id])
    (B := 1 / (n + 1) ^ c) ?_ ?_
  rotate_left
  · simp only [Nat.cast_nonneg, sqrt_div', id_eq]
    fun_prop
  · exact fun k hk ↦ prob_avg_add_sqrt_log_le hν hσ2 hc a n k hk
  simp only [mul_assoc] at h_le
  simp only [empMean, ucbWidth, mul_assoc]
  calc _ ≤ (n : ℝ≥0∞) * (1 / (n + 1) ^ c) := h_le
  _ ≤ (n + 1) * (1 : ℝ≥0∞) / (n + 1) ^ c := by
    rw [mul_one_div, mul_one]
    gcongr
    exact le_self_add
  _ = 1 / (n + 1) ^ (c - 1) := by
    simp only [mul_one, one_div]
    rw [ENNReal.rpow_sub _ _ (by simp) (by finiteness), ENNReal.rpow_one, div_eq_mul_inv,
      ENNReal.div_eq_inv_mul, ENNReal.mul_inv (by simp) (by simp), inv_inv]

/-- The probability that the lower confidence bound of arm `a` is above its mean is at most
`1 / (n + 1) ^ (c - 1)`. -/
lemma prob_ucbIndex_ge {alg : Algorithm Unit (Fin K) ℝ}
    (h : IsAlgEnvSeq O A R alg (stationaryEnv ν) P)
    (hν : ∀ a, HasSubgaussianMGF (fun x ↦ x - (ν a)[id]) σ2 (ν a))
    (hσ2 : σ2 ≠ 0) (hc : 0 ≤ c) (a : Fin K) (n : ℕ) :
    P {h | 0 < pullCount A a n h ∧
      (ν a)[id] ≤ empMean A R a n h - ucbWidth A (c * σ2) a n h} ≤ 1 / (n + 1) ^ (c - 1) := by
  have h_le := prob_pullCount_pos_and_le h a n
    (p := fun k x ↦ (ν a)[id] ≤ x / k - √(2 * c * σ2 * log (n + 1) / k))
    (B := 1 / (n + 1) ^ c) ?_ ?_
  rotate_left
  · simp only [Nat.cast_nonneg, sqrt_div', id_eq]
    fun_prop
  · exact fun k hk ↦ prob_avg_sub_sqrt_log_ge hν hσ2 hc a n k hk
  simp only [mul_assoc] at h_le
  simp only [empMean, ucbWidth, mul_assoc]
  calc _ ≤ (n : ℝ≥0∞) * (1 / (n + 1) ^ c) := h_le
  _ ≤ (n + 1) * (1 : ℝ≥0∞) / (n + 1) ^ c := by
    rw [mul_one_div, mul_one]
    gcongr
    exact le_self_add
  _ = 1 / (n + 1) ^ (c - 1) := by
    simp only [mul_one, one_div]
    rw [ENNReal.rpow_sub _ _ (by simp) (by finiteness), ENNReal.rpow_one, div_eq_mul_inv,
      ENNReal.div_eq_inv_mul, ENNReal.mul_inv (by simp) (by simp), inv_inv]

lemma probReal_ucbIndex_le {alg : Algorithm Unit (Fin K) ℝ}
    (h : IsAlgEnvSeq O A R alg (stationaryEnv ν) P)
    (hν : ∀ a, HasSubgaussianMGF (fun x ↦ x - (ν a)[id]) σ2 (ν a))
    (hσ2 : σ2 ≠ 0) (hc : 0 ≤ c) (a : Fin K) (n : ℕ) :
    P.real {h | 0 < pullCount A a n h ∧ empMean A R a n h + ucbWidth A (c * σ2) a n h ≤ (ν a)[id]} ≤
      1 / (n + 1) ^ (c - 1) := by
  rw [measureReal_def]
  grw [prob_ucbIndex_le h hν hσ2 hc a n]
  swap; · finiteness
  simp only [one_div, ENNReal.toReal_inv]
  rw [← ENNReal.toReal_rpow]
  norm_cast

lemma probReal_ucbIndex_ge {alg : Algorithm Unit (Fin K) ℝ}
    (h : IsAlgEnvSeq O A R alg (stationaryEnv ν) P)
    (hν : ∀ a, HasSubgaussianMGF (fun x ↦ x - (ν a)[id]) σ2 (ν a))
    (hσ2 : σ2 ≠ 0) (hc : 0 ≤ c) (a : Fin K) (n : ℕ) :
    P.real {h | 0 < pullCount A a n h ∧
      (ν a)[id] ≤ empMean A R a n h - ucbWidth A (c * σ2) a n h} ≤ 1 / (n + 1) ^ (c - 1) := by
  rw [measureReal_def]
  grw [prob_ucbIndex_ge h hν hσ2 hc a n]
  swap; · finiteness
  simp only [one_div, ENNReal.toReal_inv]
  rw [← ENNReal.toReal_rpow]
  norm_cast

omit [IsMarkovKernel ν] in
lemma pullCount_le_add_three (a : Fin K) (n C : ℕ) (ω : Ω) :
    pullCount A a n ω ≤ C + 1 +
      ∑ s ∈ range n, {s | A s ω = a ∧ C < pullCount A a s ω ∧
        (ν (bestArm ν))[id] ≤ empMean A R (bestArm ν) s ω + ucbWidth A c (bestArm ν) s ω ∧
        empMean A R (A s ω) s ω - ucbWidth A c (A s ω) s ω ≤ (ν (A s ω))[id]}.indicator 1 s +
      ∑ s ∈ range n,
        {s | C < pullCount A a s ω ∧ empMean A R (bestArm ν) s ω + ucbWidth A c (bestArm ν) s ω <
          (ν (bestArm ν))[id]}.indicator 1 s +
      ∑ s ∈ range n,
        {s | C < pullCount A a s ω ∧ (ν a)[id] <
          empMean A R a s ω - ucbWidth A c a s ω}.indicator 1 s := by
  refine (pullCount_le_add a n C ω).trans ?_
  simp_rw [add_assoc]
  gcongr
  simp_rw [← add_assoc]
  let A' := {s | A s ω = a ∧ C < pullCount A a s ω}
  let B := {s | A s ω = a ∧ C < pullCount A a s ω ∧
        (ν (bestArm ν))[id] ≤ empMean A R (bestArm ν) s ω + ucbWidth A c (bestArm ν) s ω ∧
        empMean A R (A s ω) s ω - ucbWidth A c (A s ω) s ω ≤ (ν (A s ω))[id]}
  let C' := {s | C < pullCount A a s ω ∧
    empMean A R (bestArm ν) s ω + ucbWidth A c (bestArm ν) s ω < (ν (bestArm ν))[id]}
  let D := {s | C < pullCount A a s ω ∧ (ν a)[id] < empMean A R a s ω - ucbWidth A c a s ω}
  change ∑ s ∈ range n, A'.indicator 1 s ≤
    ∑ s ∈ range n, B.indicator 1 s + ∑ s ∈ range n, C'.indicator 1 s +
      ∑ s ∈ range n, D.indicator 1 s
  have h_union : A' ⊆ B ∪ C' ∪ D := by simp [A', B, C', D]; grind
  calc
    (∑ s ∈ range n, A'.indicator 1 s)
    _ ≤ (∑ s ∈ range n, (B ∪ C' ∪ D).indicator (fun _ ↦ (1 : ℕ)) s) := by
      gcongr with n hn
      by_cases h : n ∈ A'
      · have : n ∈ B ∪ C' ∪ D := h_union h
        simp [h, this]
      · simp [h]
    _ ≤ ∑ s ∈ range n, (B.indicator 1 s + C'.indicator 1 s + D.indicator 1 s) := by
      gcongr with s
      simp [Set.indicator_apply]
      grind
    _ = ∑ s ∈ range n, B.indicator 1 s + ∑ s ∈ range n, C'.indicator 1 s +
          ∑ s ∈ range n, D.indicator 1 s := by
      rw [Finset.sum_add_distrib, Finset.sum_add_distrib]

lemma pullCount_le_add_three_ae
    (h : IsAlgEnvSeq O A R (ucbAlgorithm K c) (stationaryEnv ν) P)
    (a : Fin K) (n C : ℕ) (hC : C ≠ 0) :
    ∀ᵐ ω ∂P,
    pullCount A a n ω ≤ C + 1 +
      ∑ s ∈ range n, {s | A s ω = a ∧ C < pullCount A a s ω ∧
        (ν (bestArm ν))[id] ≤ empMean A R (bestArm ν) s ω + ucbWidth A c (bestArm ν) s ω ∧
        empMean A R (A s ω) s ω - ucbWidth A c (A s ω) s ω ≤ (ν (A s ω))[id]}.indicator 1 s +
      ∑ s ∈ range n,
        {s | 0 < pullCount A (bestArm ν) s ω ∧
          empMean A R (bestArm ν) s ω + ucbWidth A c (bestArm ν) s ω <
            (ν (bestArm ν))[id]}.indicator 1 s +
      ∑ s ∈ range n,
        {s | 0 < pullCount A a s ω ∧ (ν a)[id] <
          empMean A R a s ω - ucbWidth A c a s ω}.indicator 1 s := by
  filter_upwards [pullCount_pos_of_pullCount_gt_one h a] with ω hω
  refine (pullCount_le_add_three (R := R) a n C ω (ν := ν) (c := c)).trans ?_
  gcongr 5 with k hk j k hk j
  · gcongr 1
    exact fun h_gt ↦ hω _ (lt_of_le_of_lt (by grind) h_gt) _
  · exact fun h_gt ↦ hω _ (lt_of_le_of_lt (by grind) h_gt) _

lemma some_sum_eq_zero (h : IsAlgEnvSeq O A R (ucbAlgorithm K (c * σ2)) (stationaryEnv ν) P)
    (hc : 0 ≤ c) (a : Fin K) (h_gap : 0 < gap ν a) (n C : ℕ)
    (hC : C ≠ 0) (hC' : 8 * c * σ2 * log (n + 1) / gap ν a ^ 2 ≤ C) :
    ∀ᵐ ω ∂P,
    ∑ s ∈ range n, {s | A s ω = a ∧ C < pullCount A a s ω ∧
      (ν (bestArm ν))[id] ≤ empMean A R (bestArm ν) s ω + ucbWidth A (c * σ2) (bestArm ν) s ω ∧
      empMean A R (A s ω) s ω - ucbWidth A (c * σ2) (A s ω) s ω
        ≤ (ν (A s ω))[id]}.indicator 1 s = 0 := by
  have h_ae := forall_ucbIndex_le_ucbIndex_arm h (bestArm ν) (ν := ν) (c := c * σ2)
  have h_gt := time_gt_of_pullCount_gt_one h a (ν := ν) (c := c * σ2)
  filter_upwards [h_ae, h_gt] with ω h_le h_time_ge
  simp only [id_eq, tsub_le_iff_right, sum_eq_zero_iff, mem_range, Set.indicator_apply_eq_zero,
    Set.mem_ofPred_eq, Pi.one_apply, one_ne_zero, imp_false, not_and, not_le]
  intro k hn h_arm hC_lt h_le_best
  by_contra! h_le_arm
  have h := pullCount_le_of_ucbIndex_le (b := A k ω) (by positivity : 0 ≤ c * σ2) h_le_best
    (by simpa) ?_ ?_ ?_
  rotate_left
  · refine h_le _ ?_
    refine (h_time_ge _ ?_).le
    refine lt_of_le_of_lt ?_ hC_lt
    grind
  · rwa [h_arm]
  · rw [h_arm]
    exact zero_le.trans_lt hC_lt
  refine lt_irrefl (8 * c * σ2 * log (n + 1) / gap ν a ^ 2) ?_
  refine hC'.trans_lt (lt_of_lt_of_le ?_ (h.trans ?_))
  · rw [h_arm]
    exact mod_cast hC_lt
  · rw [h_arm]
    simp_rw [← mul_assoc]
    gcongr

lemma pullCount_ae_le_add_two (h : IsAlgEnvSeq O A R (ucbAlgorithm K (c * σ2)) (stationaryEnv ν) P)
    (hc : 0 ≤ c) (a : Fin K) (h_gap : 0 < gap ν a)
    (n C : ℕ) (hC : C ≠ 0) (hC' : 8 * c * σ2 * log (n + 1) / gap ν a ^ 2 ≤ C) :
    ∀ᵐ ω ∂P,
    pullCount A a n ω ≤ C + 1 +
      ∑ s ∈ range n,
        {s | 0 < pullCount A (bestArm ν) s ω ∧
          empMean A R (bestArm ν) s ω + ucbWidth A (c * σ2) (bestArm ν) s ω <
            (ν (bestArm ν))[id]}.indicator 1 s +
      ∑ s ∈ range n,
        {s | 0 < pullCount A a s ω ∧ (ν a)[id] <
          empMean A R a s ω - ucbWidth A (c * σ2) a s ω}.indicator 1 s := by
  filter_upwards [some_sum_eq_zero h hc a h_gap n C hC hC',
    pullCount_le_add_three_ae h a n C hC] with ω hω_zero hω_le
  refine (hω_le).trans_eq ?_
  rw [hω_zero]

/-- A sum that appears in the UCB regret upper bound, with a finite limit as `n` goes to infinity
for `c > 2`. -/
noncomputable
def constSum (c : ℝ) (n : ℕ) : ℝ≥0∞ := ∑ s ∈ range n, 1 / ((s : ℝ≥0∞) + 1) ^ (c - 1)

lemma constSum_lt_top (c : ℝ) (n : ℕ) : constSum c n < ∞ := by
  rw [constSum, ENNReal.sum_lt_top]
  intro k hk
  simp only [one_div, ENNReal.inv_lt_top]
  positivity

/-- Bound on the expectation of the number of pulls of each arm by the UCB algorithm. -/
lemma expectation_pullCount_le'
    (h : IsAlgEnvSeq O A R (ucbAlgorithm K (c * σ2)) (stationaryEnv ν) P)
    (hν : ∀ a, HasSubgaussianMGF (fun x ↦ x - (ν a)[id]) σ2 (ν a))
    (hσ2 : σ2 ≠ 0) (hc : 0 < c) (a : Fin K) (h_gap : 0 < gap ν a) (n : ℕ) :
    ∫⁻ ω, pullCount A a n ω ∂P ≤
      ENNReal.ofReal (8 * c * σ2 * log (n + 1) / gap ν a ^ 2 + 1) + 1 + 2 * constSum c n := by
  have hA := h.measurable_action
  have hR := h.measurable_feedback
  by_cases hn_zero : n = 0
  · simp [hn_zero]
  let C a : ℕ := ⌈8 * c * σ2 * log (n + 1) / gap ν a ^ 2⌉₊
  have h_set_1 b : MeasurableSet {a_1 | 0 < pullCount A a b a_1 ∧
      (ν a)[id] < empMean A R a b a_1 - ucbWidth A (c * σ2) a b a_1} := by
    simp only [measurableSet_setOfPred]
    fun_prop
  have h_set_2 b : MeasurableSet {a | 0 < pullCount A (bestArm ν) b a ∧
      empMean A R (bestArm ν) b a + ucbWidth A (c * σ2) (bestArm ν) b a < (ν (bestArm ν))[id]} := by
    simp only [measurableSet_setOfPred]
    fun_prop
  have h_meas_1 b : Measurable fun h ↦ {s | 0 < pullCount A a s h ∧ (ν a)[id] <
      empMean A R a s h - ucbWidth A (c * σ2) a s h}.indicator (1 : ℕ → ℕ) b := by
    simp only [id_eq, Set.indicator_apply, Set.mem_ofPred_eq, Pi.one_apply]
    exact Measurable.ite (h_set_1 _) (by fun_prop) (by fun_prop)
  have h_meas_2 b : Measurable fun h ↦ {s | 0 < pullCount A (bestArm ν) s h ∧
      empMean A R (bestArm ν) s h + ucbWidth A (c * σ2) (bestArm ν) s h <
          (ν (bestArm ν))[id]}.indicator (1 : ℕ → ℕ) b := by
    simp only [id_eq, Set.indicator_apply, Set.mem_ofPred_eq, Pi.one_apply]
    exact Measurable.ite (h_set_2 _) (by fun_prop) (by fun_prop)
  calc ∫⁻ ω, pullCount A a n ω ∂P
  _ ≤ ∫⁻ ω, C a + 1 +
      ∑ s ∈ range n,
        {s | 0 < pullCount A (bestArm ν) s ω ∧
          empMean A R (bestArm ν) s ω + ucbWidth A (c * σ2) (bestArm ν) s ω <
            (ν (bestArm ν))[id]}.indicator (1 : ℕ → ℕ) s +
      ∑ s ∈ range n,
        {s | 0 < pullCount A a s ω ∧ (ν a)[id] <
          empMean A R a s ω - ucbWidth A (c * σ2) a s ω}.indicator (1 : ℕ → ℕ) s ∂P := by
    refine lintegral_mono_ae ?_
    have hCa : C a ≠ 0 := by
      simp only [ne_eq, Nat.ceil_eq_zero, not_le, C]
      have : 0 < log (n + 1) := log_pos (by simp; grind)
      positivity
    filter_upwards [pullCount_ae_le_add_two h hc.le a h_gap n (C a) hCa (Nat.le_ceil _)] with ω hω
    simp only [id_eq, Nat.cast_sum]
    norm_cast
  _ ≤ (C a : ℝ≥0∞) + 1 +
      ∑ s ∈ range n,
        P {ω | 0 < pullCount A (bestArm ν) s ω ∧
          empMean A R (bestArm ν) s ω + ucbWidth A (c * σ2) (bestArm ν) s ω < (ν (bestArm ν))[id]} +
      ∑ s ∈ range n,
        P {ω | 0 < pullCount A a s ω ∧ (ν a)[id] <
          empMean A R a s ω - ucbWidth A (c * σ2) a s ω} := by
    simp only [id_eq, Nat.cast_sum]
    rw [lintegral_add_left (by fun_prop), lintegral_add_left (by fun_prop)]
    simp only [lintegral_const, measure_univ, mul_one]
    rw [lintegral_finsetSum _ (by fun_prop), lintegral_finsetSum _ (by fun_prop)]
    gcongr with k hk k hk
    · rw [← lintegral_indicator_one]
      swap; · exact h_set_2 _
      gcongr with h
      simp [Set.indicator_apply]
    · rw [← lintegral_indicator_one]
      swap; · exact h_set_1 _
      gcongr with h
      simp [Set.indicator_apply]
  _ ≤ (C a : ℝ≥0∞) + 1 +
      ∑ s ∈ range n, 1 / ((s : ℝ≥0∞) + 1) ^ (c - 1) +
      ∑ s ∈ range n, 1 / ((s : ℝ≥0∞) + 1) ^ (c - 1) := by
    gcongr with s hs s hs
    · refine (measure_mono ?_).trans (prob_ucbIndex_le h hν hσ2 (by positivity) (bestArm ν) s)
      grind
    · refine (measure_mono ?_).trans (prob_ucbIndex_ge h hν hσ2 (by positivity) a s)
      grind
  _ ≤ ENNReal.ofReal (8 * c * σ2 * log (n + 1) / gap ν a ^ 2 + 1) + 1 + 2 * constSum c n := by
    rw [two_mul, add_assoc, constSum]
    gcongr
    simp only [C]
    rw [← ENNReal.ofReal_natCast]
    refine ENNReal.ofReal_le_ofReal ?_
    refine (Nat.ceil_lt_add_one ?_).le
    have : 0 ≤ log (n + 1) := log_nonneg (by simp)
    positivity

/-- Bound on the expectation of the number of pulls of each arm by the UCB algorithm. -/
lemma expectation_pullCount_le (h : IsAlgEnvSeq O A R (ucbAlgorithm K (c * σ2)) (stationaryEnv ν) P)
    (hν : ∀ a, HasSubgaussianMGF (fun x ↦ x - (ν a)[id]) σ2 (ν a))
    (hσ2 : σ2 ≠ 0) (hc : 0 < c) (a : Fin K) (h_gap : 0 < gap ν a) (n : ℕ) :
    P[fun ω ↦ (pullCount A a n ω : ℝ)] ≤
      8 * c * σ2 * log (n + 1) / gap ν a ^ 2 + 2 + 2 * (constSum c n).toReal := by
  have hA := h.measurable_action
  have h := expectation_pullCount_le' h hν hσ2 hc a h_gap n
  simp_rw [← ENNReal.ofReal_natCast] at h
  rw [← ofReal_integral_eq_lintegral_ofReal] at h
  rotate_left
  · exact integrable_pullCount hA _ _
  · exact ae_of_all _ fun _ ↦ by simp
  simp only
  have : 0 ≤ log (n + 1) := log_nonneg (by simp)
  rw [← ENNReal.ofReal_toReal (a := 2 * constSum c n), ← ENNReal.ofReal_one, ← ENNReal.ofReal_add,
    ← ENNReal.ofReal_add, ENNReal.ofReal_le_ofReal_iff] at h
  rotate_left
  · positivity
  · positivity
  · simp
  · have : constSum c n ≠ ∞ := (constSum_lt_top c n).ne
    finiteness
  · simp
  · have : constSum c n ≠ ∞ := (constSum_lt_top c n).ne
    finiteness
  refine h.trans_eq ?_
  simp only [ENNReal.toReal_mul, ENNReal.toReal_ofNat, add_left_inj]
  ring

/-- Regret bound for the UCB algorithm. -/
theorem regret_le (h : IsAlgEnvSeq O A R (ucbAlgorithm K (c * σ2)) (stationaryEnv ν) P)
    (hν : ∀ a, HasSubgaussianMGF (fun x ↦ x - (ν a)[id]) σ2 (ν a))
    (hσ2 : σ2 ≠ 0) (hc : 0 < c) (n : ℕ) :
    P[regret ν A n] ≤
      ∑ a, (8 * c * σ2 * log (n + 1) / gap ν a + gap ν a * (2 + 2 * (constSum c n).toReal)) := by
  refine (integral_regret_le_of_forall_integral_pullCount_le h
    (fun a h_gap ↦ expectation_pullCount_le h hν hσ2 hc a
      (lt_of_le_of_ne' gap_nonneg h_gap) n)).trans_eq ?_
  congr with a
  by_cases h_gap : gap ν a = 0
  · simp [h_gap]
  · field

end UCB

end Bandits
