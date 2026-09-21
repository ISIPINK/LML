/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Bregman.Basic
public import Mathlib.Analysis.Convex.Function
public import Mathlib.Order.ConditionallyCompleteLattice.Basic
public import Mathlib.Topology.Algebra.Module.ContinuousLinearMap.Basic
public import Mathlib.Tactic

/-!
# Subgradients and Subdifferentials

Subgradients of functions `f : E → F` defined
via the non-negativity of the Bregman divergence `0 ≤ D_[f](x, y, g_y)` on an explicit
domain `V` (i.e. the linearization error is non-negative on `V`).

Carrying the domain `V : Set E` explicitly avoids using indicator functions
while supporting constrained convex analysis.

## Main definitions

* `Analysis.Convex.IsSubgradient V f y g_y`: `g_y : E →L[R] F` is a subgradient of `f`
  at `y` on `V`.
* `Analysis.Convex.subdifferential V f y`: The subdifferential set `∂[V, y] f`.

## Main results

* Chain rules: `Analysis.Convex.IsSubgradient.comp_affine` and `Analysis.Convex.IsSubgradient.comp`.
* Equivalence with classical inequality: `Analysis.Convex.isSubgradient_iff_le`.
* Fermat's rule: `Analysis.Convex.isSubgradient_zero_iff_isMinOn`.
* Suprema and maxima: `Analysis.Convex.IsSubgradient.max_left`,
  `Analysis.Convex.IsSubgradient.max_right`, `Analysis.Convex.IsSubgradient.finset_sup`,
  and `Analysis.Convex.IsSubgradient.ciSup`.

## Notation

* `∂[V, y] f`: Scoped notation in `Bregman` for `subdifferential V f y`.
-/

@[expose] public section

namespace Analysis.Convex

open scoped Bregman

variable {R E E₁ F G : Type*} [Ring R]
  [AddCommGroup E] [Module R E] [TopologicalSpace E]
  [AddCommGroup E₁] [Module R E₁] [TopologicalSpace E₁]
  [AddCommGroup F] [Module R F] [TopologicalSpace F]
  [AddCommGroup G] [Module R G] [TopologicalSpace G]
  [Preorder F]

/-- `g_y` is a subgradient of `f` at `y` on domain `V` (non-negativity of Bregman divergence). -/
def IsSubgradient (V : Set E) (f : E → F) (y : E) (g_y : E →L[R] F) : Prop :=
  ∀ x ∈ V, 0 ≤ D_[f](x, y, g_y)

/-- The subdifferential `∂[V, y] f` of `f` at `y` on `V`. -/
def subdifferential (V : Set E) (f : E → F) (y : E) : Set (E →L[R] F) :=
  { g | IsSubgradient V f y g }

/-- Scoped notation for `subdifferential`. -/
scoped[Bregman] notation "∂[" V ", " y "](" f ")" => Analysis.Convex.subdifferential V f y

variable {V : Set E} {f : E → F} {y x : E} {g_y : E →L[R] F}

@[simp]
lemma mem_subdifferential_iff :
    g_y ∈ ∂[V, y](f) ↔ IsSubgradient V f y g_y := Iff.rfl

lemma isSubgradient_const (c : F) :
    IsSubgradient V (fun _ ↦ c) y (0 : E →L[R] F) := by
  simp [IsSubgradient]

@[simp]
lemma isSubgradient_linear (h : E →L[R] F) :
    IsSubgradient V h y h := by
  simp [IsSubgradient]

@[simp]
lemma isSubgradient_add_const_iff {c : F} :
    IsSubgradient V (fun x ↦ f x + c) y g_y ↔ IsSubgradient V f y g_y := by
  simp [IsSubgradient, bregDiv]

@[simp]
lemma isSubgradient_const_add_iff {c : F} :
    IsSubgradient V (fun x ↦ c + f x) y g_y ↔ IsSubgradient V f y g_y := by
  simp [IsSubgradient, bregDiv]

@[simp]
lemma isSubgradient_add_linear_iff [ContinuousAdd F] {h : E →L[R] F} :
    IsSubgradient V (fun x ↦ f x + h x) y (g_y + h) ↔ IsSubgradient V f y g_y := by
  simp [IsSubgradient, bregDiv_add_linear]

@[simp]
lemma isSubgradient_bregDiv_iff [IsTopologicalAddGroup F] {g_x g_y : E →L[R] F} :
    IsSubgradient V (fun z ↦ D_[f](z, y, g_y)) x (g_x - g_y) ↔ IsSubgradient V f x g_x := by
  simp [IsSubgradient, bregDiv_fun_bregDiv]

lemma isSubgradient_zero_iff :
    IsSubgradient V f y (0 : E →L[R] F) ↔ ∀ x ∈ V, 0 ≤ f x - f y := by
  simp [IsSubgradient, bregDiv]

lemma IsSubgradient.comp_affine
    {V₁ : Set E₁} {y₁ : E₁} {A : E₁ →L[R] E} {b : E}
    (h_map : ∀ x ∈ V₁, A x + b ∈ V)
    (h_sub : IsSubgradient V f (A y₁ + b) g_y) :
    IsSubgradient V₁ (fun x ↦ f (A x + b)) y₁ (g_y.comp A) := fun x hx ↦ by
  rw [bregDiv_comp_affine]
  exact h_sub (A x + b) (h_map x hx)

/-- **Chain rule**: `g₂ ∘ g₁` is a subgradient of `f₂ ∘ f₁` when `g₂` is non-negative. -/
lemma IsSubgradient.comp [Preorder G] [IsOrderedAddMonoid G]
    {f₂ : F → G} {f₁ : E → F} {g₂ : F →L[R] G} {g₁ : E →L[R] F}
    (h₂ : IsSubgradient (f₁ '' V) f₂ (f₁ y) g₂) (h₁ : IsSubgradient V f₁ y g₁)
    (hg₂_nonneg : ∀ z ≥ 0, 0 ≤ g₂ z) :
    IsSubgradient V (f₂ ∘ f₁) y (g₂.comp g₁) := fun x hx ↦ by
  rw [bregDiv_comp]
  exact add_nonneg (h₂ (f₁ x) ⟨x, hx, rfl⟩) (hg₂_nonneg _ (h₁ x hx))

section OrderedGroup

variable [IsOrderedAddMonoid F]

/-- Equivalence with the classical definition. -/
lemma isSubgradient_iff_le :
    IsSubgradient V f y g_y ↔ ∀ x ∈ V, f y + g_y (x - y) ≤ f x := by
  simp [IsSubgradient, bregDiv, sub_sub, sub_nonneg]

lemma IsSubgradient.monotonicity [IsTopologicalAddGroup F] {g_x : E →L[R] F}
    (hx : x ∈ V) (hy : y ∈ V)
    (hx_sub : IsSubgradient V f x g_x) (hy_sub : IsSubgradient V f y g_y) :
    0 ≤ (g_x - g_y) (x - y) := by
  rw [← bregDiv_add_swap]
  exact add_nonneg (hx_sub y hy) (hy_sub x hx)

lemma isSubgradient_zero_iff_isMinOn :
    IsSubgradient V f y (0 : E →L[R] F) ↔ IsMinOn f V y := by
  simp [IsSubgradient, bregDiv, sub_nonneg, isMinOn_iff]

lemma IsSubgradient.add [ContinuousAdd F] {f₁ f₂ : E → F} {g₁ g₂ : E →L[R] F}
    (h₁ : IsSubgradient V f₁ y g₁) (h₂ : IsSubgradient V f₂ y g₂) :
    IsSubgradient V (f₁ + f₂) y (g₁ + g₂) := fun x hx ↦ by
  rw [bregDiv_add]
  exact add_nonneg (h₁ x hx) (h₂ x hx)

lemma IsSubgradient.add_isMinOn [ContinuousAdd F] {f₁ f₂ : E → F} {x : E} {g : E →L[R] F}
    (h_min : IsMinOn f₁ V x) (h_sub : IsSubgradient V f₂ x g) :
    IsSubgradient V (f₁ + f₂) x g := by
  simpa using IsSubgradient.add (isSubgradient_zero_iff_isMinOn.mpr h_min) h_sub

end OrderedGroup

lemma IsSubgradient.of_le_of_eq [AddRightMono F] {f₁ f₂ : E → F} {g_y : E →L[R] F}
    (h_sub : IsSubgradient V f₁ y g_y) (h_le : ∀ x ∈ V, f₁ x ≤ f₂ x) (h_eq : f₁ y = f₂ y) :
    IsSubgradient V f₂ y g_y :=
  fun x hx ↦ le_trans (h_sub x hx)
    (bregDiv_le_of_le_of_eq (h_le x hx) h_eq)

section ModuleBasic

variable {R' : Type*} [CommRing R'] [PartialOrder R']
  [Module R' E] [Module R' F] [ContinuousConstSMul R' F] [PosSMulMono R' F]

lemma IsSubgradient.smul {c : R'} (hc : 0 ≤ c) {f : E → F} {g_y : E →L[R'] F}
    (h_sub : IsSubgradient V f y g_y) :
    IsSubgradient V (c • f) y (c • g_y) := fun x hx ↦ by
  rw [bregDiv_smul]
  exact smul_nonneg hc (h_sub x hx)

end ModuleBasic

section Module

variable {R' : Type*} [CommRing R'] [PartialOrder R'] [IsOrderedRing R']
    [Module R' E] [IsOrderedAddMonoid F] [Module R' F] [ContinuousConstSMul R' F] [PosSMulMono R' F]
    [ContinuousAdd F]

lemma IsSubgradient.convexCombination {f : E → F} {g₁ g₂ : E →L[R'] F}
    (h₁ : IsSubgradient V f y g₁) (h₂ : IsSubgradient V f y g₂)
    {w : R'} (hw : w ∈ Set.Icc (0 : R') 1) :
    IsSubgradient V f y (w • g₁ + (1 - w) • g₂) := fun x hx ↦ by
  rw [bregDiv_convexCombination]
  exact add_nonneg (smul_nonneg hw.1 (h₁ x hx))
    (smul_nonneg (sub_nonneg.mpr hw.2) (h₂ x hx))

end Module

section LinearOrder

variable {F_lin : Type*} [AddCommGroup F_lin] [Module R F_lin] [TopologicalSpace F_lin]
  [LinearOrder F_lin] [IsOrderedAddMonoid F_lin]

lemma IsSubgradient.max_left {f₁ f₂ : E → F_lin} {g_y : E →L[R] F_lin}
    (h_sub : IsSubgradient V f₁ y g_y) (h_active : f₁ y = max (f₁ y) (f₂ y)) :
    IsSubgradient V (fun x ↦ max (f₁ x) (f₂ x)) y g_y :=
  IsSubgradient.of_le_of_eq h_sub (fun x _ ↦ le_max_left (f₁ x) (f₂ x)) h_active

lemma IsSubgradient.max_right {f₁ f₂ : E → F_lin} {g_y : E →L[R] F_lin}
    (h_sub : IsSubgradient V f₂ y g_y) (h_active : f₂ y = max (f₁ y) (f₂ y)) :
    IsSubgradient V (fun x ↦ max (f₁ x) (f₂ x)) y g_y :=
  IsSubgradient.of_le_of_eq h_sub (fun x _ ↦ le_max_right (f₁ x) (f₂ x)) h_active

lemma IsSubgradient.finset_sup {ι : Type*} {s : Finset ι}
    {f_i : ι → E → F_lin} {i : ι} {g_y : E →L[R] F_lin}
    (his : i ∈ s)
    (h_sub : IsSubgradient V (f_i i) y g_y)
    (h_active : f_i i y = s.sup' ⟨i, his⟩ (fun j ↦ f_i j y)) :
    IsSubgradient V (fun x ↦ s.sup' ⟨i, his⟩ (fun j ↦ f_i j x)) y g_y :=
  IsSubgradient.of_le_of_eq h_sub (fun _ _ ↦ Finset.le_sup'_of_le _ his (le_refl _)) h_active

end LinearOrder

section Lattice

variable {F_lat : Type*} [AddCommGroup F_lat] [Module R F_lat] [TopologicalSpace F_lat]
    [ConditionallyCompleteLattice F_lat] [IsOrderedAddMonoid F_lat]

lemma IsSubgradient.ciSup {ι : Type*} {f_i : ι → E → F_lat} {i : ι} {g_y : E →L[R] F_lat}
    (h_sub : IsSubgradient V (f_i i) y g_y)
    (h_active : f_i i y = ⨆ j, f_i j y)
    (h_bdd : ∀ x ∈ V, BddAbove (Set.range (fun j ↦ f_i j x))) :
    IsSubgradient V (fun x ↦ ⨆ j, f_i j x) y g_y :=
  IsSubgradient.of_le_of_eq h_sub (fun x hx ↦ le_ciSup (h_bdd x hx) i) h_active

end Lattice

end Analysis.Convex
