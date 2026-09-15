/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.ForMathlib.ConvexAnalysis.Subgradient.Basic
public import Mathlib.Algebra.Order.Group.CompleteLattice
public import Mathlib.Order.ConditionallyCompleteLattice.Basic

/-!
# Fenchel Conjugates (Legendre–Fenchel Transform)

This file defines the domain-constrained generalized Fenchel conjugate `f^*[V]`
for functions `f : E → F` into conditionally complete ordered groups `F`.

It establishes Fenchel–Young inequalities, the equivalence between subgradient
membership and Fenchel–Young equality, and dual Bregman divergence identities.

## Main definitions

* `Analysis.Convex.fenchelConjugate V f`: The Fenchel conjugate
  $f^*[V](g) = \sup_{x \in V} (g(x) - f(x))$.
* `Analysis.Convex.evalBidual`: The canonical evaluation map from $E$ into the bidual
  $(E \to+ F) \to+ F$.
* `Analysis.Convex.imageSubgradient V f`: The image of the subdifferential operator
  $\bigcup_{y \in V} \partial[V, y] f$.

## Notation

* `f^*[V]`: Scoped notation in `Bregman` for `fenchelConjugate V f`.
* `f^*`: Scoped notation in `Bregman` for `fenchelConjugate Set.univ f`.

## Main results

* `Analysis.Convex.fenchel_young`: Fenchel–Young inequality $g(x) \le f(x) + f^*[V](g)$.
* `Analysis.Convex.fenchel_young_eq`: Equivalence
  $f(x) + f^*[V](g) = g(x) \iff g \in \partial[V, x] f$.
* `Analysis.Convex.bregman_eq_dual_bregman`: Bregman duality
  $D_f(x, y, g_y) = D_{f^*[V]}(g_y, g_x, x)$.
* `Analysis.Convex.dual_subgradient`: Dual subgradient identity
  $x \in \partial[\text{imageSubgradient } V f, g_x] f^*[V]$.
-/

@[expose] public section

namespace Analysis.Convex

variable {E F : Type*} [AddCommGroup E] [AddCommGroup F]
variable [ConditionallyCompleteLattice F] [IsOrderedAddMonoid F]

/-! ### 1. Definitions and Notation -/

/-- The Fenchel conjugate of a function `f : E → F` evaluated at an additive map `g : E →+ F`
over a set `V`. Defined as $f^*[V](g) = \sup_{x \in V} (g(x) - f(x))$. -/
noncomputable def fenchelConjugate (V : Set E) (f : E → F) : (E →+ F) → F :=
  fun g ↦ ⨆ x : V, (g x - f x)

/-- Scoped notation for Fenchel conjugate on a set `V`. -/
scoped[Bregman] notation:max f "^*[" V "]" => Analysis.Convex.fenchelConjugate V f
/-- Scoped notation for universal Fenchel conjugate. -/
scoped[Bregman] notation:max f "^*" => Analysis.Convex.fenchelConjugate Set.univ f

open scoped Bregman

/-- Unexpander for `fenchelConjugate`. -/
@[app_unexpander fenchelConjugate]
meta def unexpandFenchelConjugate : Lean.PrettyPrinter.Unexpander
  | `($_ (Set.univ) $f) => `($f^*)
  | `($_ $V $f)        => `($f^*[$V])
  | _                  => throw ()

variable {V : Set E} {f : E → F} {x y : E} {g gx gy : E →+ F}

omit [ConditionallyCompleteLattice F] [IsOrderedAddMonoid F] in
lemma fenchel_objective_eq_bregman (f : E → F) (x y : E) (g gy : E →+ F) :
    g x - f x = (g y - f y) + (g - gy) (x - y) - D_[f](x, y, gy) := by
  simp only [bregDiv, AddMonoidHom.sub_apply, map_sub]
  abel

/-! ### 2. Fenchel–Young Inequality and Subgradients -/

/-- Fenchel–Young inequality: $g(x) \le f(x) + f^*[V](g)$ for all $x \in V$. -/
lemma fenchel_young (hx : x ∈ V)
    (hbd : BddAbove (Set.range (fun (y : V) ↦ g y - f y))) :
    g x ≤ f x + f^*[V] g :=
  sub_le_iff_le_add'.mp (le_ciSup hbd ⟨x, hx⟩)

lemma bddAbove_of_subgradient (h : g ∈ ∂[V, x] f) :
    BddAbove (Set.range (fun (y : V) ↦ g y - f y)) := by
  refine ⟨g x - f x, ?_⟩
  rintro _ ⟨⟨y, hy⟩, rfl⟩
  have := fenchel_objective_eq_bregman f y x g g
  simpa only [sub_self, AddMonoidHom.zero_apply, add_zero, this]
    using sub_le_self (g x - f x) (h.2 y hy)

/-- The Fenchel conjugate at any subgradient $g \in \partial[V, x] f$ is exactly $g(x) - f(x)$. -/
lemma fenchelConjugate_of_subgradient (h : g ∈ ∂[V, x] f) :
    f^*[V] g = g x - f x := by
  have : Nonempty V := ⟨⟨x, h.1⟩⟩
  refine le_antisymm (ciSup_le fun ⟨y, hy⟩ ↦ ?_)
    (le_ciSup (bddAbove_of_subgradient h) ⟨x, h.1⟩)
  simp [fenchel_objective_eq_bregman f y x g g, h.2 y hy]

lemma IsSubgradient.fenchelConjugate (h : g ∈ ∂[V, x] f) :
    f^*[V] g = g x - f x :=
  fenchelConjugate_of_subgradient h

/-- Fenchel–Young equality at any subgradient: $f(x) + f^*[V](g) = g(x)$. -/
@[simp]
lemma fenchel_young_of_subgradient (h : g ∈ ∂[V, x] f) :
    f x + f^*[V] g = g x := by
  simp [IsSubgradient.fenchelConjugate h]

/-- Characterization of subgradients via Fenchel–Young equality. -/
lemma fenchel_young_eq (hx : x ∈ V)
    (hbd : BddAbove (Set.range (fun (y : V) ↦ g y - f y))) :
    f x + f^*[V] g = g x ↔ g ∈ ∂[V, x] f := by
  have : Nonempty V := ⟨⟨x, hx⟩⟩
  constructor
  · intro h
    refine ⟨hx, fun y hy ↦ ?_⟩
    have h_le : g y - f y ≤ f^*[V] g := le_ciSup hbd ⟨y, hy⟩
    simp_all [fenchel_objective_eq_bregman f y x g g, eq_sub_of_add_eq' h]
  · exact fenchel_young_of_subgradient

/-! ### 3. Dual Space and Dual Subgradients -/

/-- The canonical evaluation map from $E$ into the bidual space $(E \to+ F) \to+ F$. -/
def evalBidual (y : E) : (E →+ F) →+ F where
  toFun := fun g ↦ g y
  map_zero' := rfl
  map_add' := fun _ _ ↦ rfl

omit [ConditionallyCompleteLattice F] [IsOrderedAddMonoid F] in
@[simp]
lemma evalBidual_apply (y : E) (g : E →+ F) : evalBidual y g = g y := rfl

@[nolint defsWithUnderscore]
instance : Coe E ((E →+ F) →+ F) where
  coe := evalBidual

/-- The image of the subdifferential operator $\bigcup_{y \in V} \partial[V, y] f$. -/
def imageSubgradient (V : Set E) (f : E → F) : Set (E →+ F) :=
  { g | ∃ y, g ∈ ∂[V, y] f }

/-- **Bregman Duality**: The Bregman divergence of $f$ equals the dual Bregman divergence
of $f^*$ with arguments swapped. -/
lemma bregman_eq_dual_bregman (h_gx : gx ∈ ∂[V, x] f) (h_gy : gy ∈ ∂[V, y] f) :
    D_[f](x, y, gy) = D_[f^*[V]](gy, gx, x) := by
  dsimp [bregDiv, evalBidual]
  simp only [IsSubgradient.fenchelConjugate h_gx, IsSubgradient.fenchelConjugate h_gy, map_sub]
  abel

/-- Dual subgradient theorem: $x$ is a subgradient of $f^*[V]$ at $g_x \in \partial[V, x] f$. -/
lemma dual_subgradient (h_gx : gx ∈ ∂[V, x] f) :
    (x : (E →+ F) →+ F) ∈ ∂[imageSubgradient V f, gx] f^*[V] :=
  ⟨⟨x, h_gx⟩, fun _ ⟨_, h_gy⟩ ↦ by
    simp [← bregman_eq_dual_bregman h_gx h_gy, h_gy.2 x h_gx.1]⟩

end Analysis.Convex
