/-
Copyright (c) 2026 Isidoor Pinillo Esquivel. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Isidoor Pinillo Esquivel
-/
module

public import LeanMachineLearning.ForMathlib.ConvexAnalysis.FenchelConjugate.Basic

/-!
# Calculus of Fenchel Conjugates

This file establishes algebraic transformation laws and calculus rules for
the domain-constrained generalized Fenchel conjugate `f^*[V]`:

1. **Constant Shifts:** $(f + c)^*[V](g) = f^*[V](g) - c$.
2. **Linear Perturbations:** $(f + h)^*[V](g) = f^*[V](g - h)$.
3. **Bregman Divergence Objectives:**
   $(D_f(\cdot, y, g_y))^*[V](g) = f^*[V](g + g_y) - f^*[V](g_y)$.
4. **Monotonicity & Order:** $f_1 \le f_2 \implies f_2^*[V] \le f_1^*[V]$ and
   $V_1 \subseteq V_2 \implies f^*[V_1] \le f^*[V_2]$.
5. **Domain Translations:** $(f(\cdot - x_0))^*[V + x_0](g) = f^*[V](g) + g(x_0)$.
6. **Support Functions:** $(0)^*[V](g) = \sup_{x \in V} g(x)$.
-/

@[expose] public section

namespace Analysis.Convex

variable {E F : Type*} [AddCommGroup E] [AddCommGroup F]
variable [ConditionallyCompleteLattice F] [IsOrderedAddMonoid F]

open scoped Bregman

attribute [local simp] fenchelConjugate

variable {V V₁ V₂ : Set E} {f f₁ f₂ : E → F} {x y x₀ : E} {g gx gy h : E →+ F} {c : F}

/-! ### 1. Constant Shifts -/

@[simp]
lemma fenchelConjugate_add_const [Nonempty V]
    (hbd : BddAbove (Set.range (fun (y : V) ↦ g y - f y))) :
    (fun x ↦ f x + c)^*[V] g = f^*[V] g - c := by
  simp [sub_add_eq_sub_sub, ciSup_sub hbd c]

lemma fenchelConjugate_const_add [Nonempty V]
    (hbd : BddAbove (Set.range (fun (y : V) ↦ g y - f y))) :
    (fun x ↦ c + f x)^*[V] g = f^*[V] g - c := by
  simp [add_comm c, sub_add_eq_sub_sub, ciSup_sub hbd c]

@[simp]
lemma fenchelConjugate_sub_const [Nonempty V]
    (hbd : BddAbove (Set.range (fun (y : V) ↦ g y - f y))) :
    (fun x ↦ f x - c)^*[V] g = f^*[V] g + c := by
  simp [sub_sub_eq_add_sub, add_sub_right_comm, ciSup_add hbd c]

/-! ### 2. Linear Perturbations -/

omit [IsOrderedAddMonoid F] in
@[simp]
lemma fenchelConjugate_add_linear :
    (fun x ↦ f x + h x)^*[V] g = f^*[V] (g - h) := by
  simp [sub_add_eq_sub_sub, sub_right_comm]

omit [IsOrderedAddMonoid F] in
lemma fenchelConjugate_linear_add :
    (fun x ↦ h x + f x)^*[V] g = f^*[V] (g - h) := by
  simp [add_comm (h _), sub_add_eq_sub_sub, sub_right_comm]

omit [IsOrderedAddMonoid F] in
@[simp]
lemma fenchelConjugate_sub_linear :
    (fun x ↦ f x - h x)^*[V] g = f^*[V] (g + h) := by
  simp [sub_sub_eq_add_sub, add_sub_right_comm]

/-! ### 3. Bregman Divergence as Objective -/

lemma fenchelConjugate_bregDiv [Nonempty V]
    (hbd : BddAbove (Set.range (fun (z : V) ↦ (g + gy) z - f z))) :
    (fun x ↦ D_[f](x, y, gy))^*[V] g = f^*[V] (g + gy) + (f y - gy y) := by
  have h_alg (x : E) : g x - D_[f](x, y, gy) = (g + gy) x - f x + (f y - gy y) := by
    simp only [bregDiv, map_sub, AddMonoidHom.add_apply]
    abel
  simp_rw [fenchelConjugate, h_alg, ← ciSup_add hbd]

/-- At any subgradient $gy \in \partial[V, y] f$, the conjugate of the Bregman divergence
evaluates to $(D_f(\cdot, y, gy))^*[V](g) = f^*[V](g + gy) - f^*[V](gy)$. -/
lemma fenchelConjugate_bregDiv_of_subgradient (h_gy : gy ∈ ∂[V, y] f)
    (hbd : BddAbove (Set.range (fun (z : V) ↦ (g + gy) z - f z))) :
    (fun x ↦ D_[f](x, y, gy))^*[V] g = f^*[V] (g + gy) - f^*[V] gy := by
  have : Nonempty V := ⟨⟨y, h_gy.1⟩⟩
  simp only [fenchelConjugate_bregDiv hbd, IsSubgradient.fenchelConjugate h_gy]
  abel

/-! ### 4. Monotonicity Rules -/

/-- Pointwise function dominance implies reverse ordering of Fenchel conjugates. -/
lemma fenchelConjugate_antitone (h_le : ∀ x ∈ V, f₁ x ≤ f₂ x) [Nonempty V]
    (hbd : BddAbove (Set.range (fun (y : V) ↦ g y - f₁ y))) :
    f₂^*[V] g ≤ f₁^*[V] g :=
  ciSup_le fun ⟨x, hx⟩ ↦ (sub_le_sub_left (h_le x hx) (g x)).trans (le_ciSup hbd ⟨x, hx⟩)

omit [IsOrderedAddMonoid F] in
/-- Inclusion of domains implies ordering of Fenchel conjugates. -/
lemma fenchelConjugate_mono_set (hV : V₁ ⊆ V₂)
    [Nonempty V₁] (hbd : BddAbove (Set.range (fun (y : V₂) ↦ g y - f y))) :
    f^*[V₁] g ≤ f^*[V₂] g :=
  ciSup_le fun ⟨x, hx⟩ ↦ le_ciSup hbd ⟨x, hV hx⟩

/-! ### 5. Domain Shifts / Translations -/

lemma fenchelConjugate_sub_const_domain (x₀ : E) [Nonempty V]
    (hbd : BddAbove (Set.range (fun (y : V) ↦ g y.1 - f y.1))) :
    (fun x ↦ f (x - x₀))^*[(fun x ↦ x - x₀) ⁻¹' V] g = f^*[V] g + g x₀ := by
  have : (⨆ x : ((fun x ↦ x - x₀) ⁻¹' V), (g x.1 - f (x.1 - x₀))) =
      ⨆ y : V, (g y.1 - f y.1 + g x₀) := by
    have h_alg (x : E) : g x - f (x - x₀) = (g (x - x₀) - f (x - x₀)) + g x₀ := by
      simp only [map_sub]; abel
    simp_rw [h_alg]
    exact ((Equiv.subRight x₀).subtypeEquiv (fun _ ↦ Iff.rfl)).iSup_congr (fun _ ↦ rfl)
  simp [fenchelConjugate, this, ciSup_add hbd (g x₀)]

lemma fenchelConjugate_add_const_domain (x₀ : E) [Nonempty V]
    (hbd : BddAbove (Set.range (fun (y : V) ↦ g y.1 - f y.1))) :
    (fun x ↦ f (x + x₀))^*[(fun x ↦ x + x₀) ⁻¹' V] g = f^*[V] g - g x₀ := by
  have : (⨆ x : ((fun x ↦ x + x₀) ⁻¹' V), (g x.1 - f (x.1 + x₀))) =
      ⨆ y : V, (g y.1 - f y.1 - g x₀) := by
    have h_alg (x : E) : g x - f (x + x₀) = (g (x + x₀) - f (x + x₀)) - g x₀ := by
      simp only [map_add]; abel
    simp_rw [h_alg]
    exact ((Equiv.addRight x₀).subtypeEquiv (fun _ ↦ Iff.rfl)).iSup_congr (fun _ ↦ rfl)
  simp [fenchelConjugate, this, ciSup_sub hbd (g x₀)]

lemma fenchelConjugate_univ_sub_const (x₀ : E)
    (hbd : BddAbove (Set.range (fun y : (Set.univ : Set E) ↦ g y.1 - f y.1))) :
    (fun x ↦ f (x - x₀))^* g = f^* g + g x₀ := by
  simpa using fenchelConjugate_sub_const_domain x₀ hbd

lemma fenchelConjugate_univ_add_const (x₀ : E)
    (hbd : BddAbove (Set.range (fun y : (Set.univ : Set E) ↦ g y.1 - f y.1))) :
    (fun x ↦ f (x + x₀))^* g = f^* g - g x₀ := by
  simpa using fenchelConjugate_add_const_domain x₀ hbd

/-! ### 6. Support Functions & Dual Norms -/

omit [IsOrderedAddMonoid F] in
@[simp]
lemma fenchelConjugate_zero (V : Set E) (g : E →+ F) :
    (fun _ : E ↦ (0 : F))^*[V] g = ⨆ x : V, g x := by
  simp [fenchelConjugate]

end Analysis.Convex
