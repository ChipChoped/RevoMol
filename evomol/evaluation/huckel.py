from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from typing_extensions import override

from evomol.evaluation.evaluation import Evaluation, EvaluationError
from evomol.representation import MolecularGraph, Molecule


PI_ELECTRONS = {
    "B0": 0,
    "C1": 1,
    "C2": 2, # hyperconjugated C
    "N1": 1,
    "N2": 2,
    "O1": 1,
    "O2": 2,
    "S1": 1,
    "S2": 2,
    "P1": 1,
    "P2": 2,
    "F2": 2,
    "Cl2": 2,
    "Br2": 2,
}

H_DIAG = {
    "B0": -0.45,
    "C1": 0.00,
    "C2": 2.00, # estimated value since absent from Rauk's work
    "N1": 0.51,
    "N2": 1.37,
    "O1": 0.97,
    "O2": 2.09,
    "S1": 0.46,
    "S2": 1.11,
    "P1": 0.19,
    "P2": 0.75,
    "F2": 2.71,
    "Cl2": 1.48,
    "Br2": 1.20, # estimated value since absent from Rauk's work
}

RAUK_COUPLINGS = {
    ("C1", "C1"): 1.00,
    ("B0", "C1"): 0.73,
    ("B0", "B0"): 0.87,

    ("N1", "C1"): 1.02,
    ("N1", "B0"): 0.66,
    ("N1", "N1"): 1.09,

    ("N2", "C1"): 0.89,
    ("N2", "B0"): 0.53,
    ("N2", "N1"): 0.99,
    ("N2", "N2"): 0.98,

    ("O1", "C1"): 1.06,
    ("O1", "B0"): 0.60,
    ("O1", "N1"): 1.14,
    ("O1", "N2"): 1.13,
    ("O1", "O1"): 1.26,

    ("O2", "C1"): 0.66,
    ("O2", "B0"): 0.35,
    ("O2", "N1"): 0.80,
    ("O2", "N2"): 0.89,
    ("O2", "O1"): 1.02,
    ("O2", "O2"): 0.95,

    ("F2", "C1"): 0.52,
    ("F2", "B0"): 0.26,
    ("F2", "N1"): 0.65,
    ("F2", "N2"): 0.77,
    ("F2", "O1"): 0.92,
    ("F2", "O2"): 0.94,
    ("F2", "F2"): 1.04,

    ("P1", "C1"): 0.77,
    ("P1", "B0"): 0.53,
    ("P1", "N1"): 0.78,
    ("P1", "N2"): 0.55,
    ("P1", "O1"): 0.75,
    ("P1", "O2"): 0.31,
    ("P1", "F2"): 0.21,
    ("P1", "P1"): 0.63,

    ("P2", "C1"): 0.76,
    ("P2", "B0"): 0.54,
    ("P2", "N1"): 0.81,
    ("P2", "N2"): 0.64,
    ("P2", "O1"): 0.82,
    ("P2", "O2"): 0.39,
    ("P2", "F2"): 0.22,
    ("P2", "P1"): 0.58,
    ("P2", "P2"): 0.63,

    ("S1", "C1"): 0.81,
    ("S1", "B0"): 0.51,
    ("S1", "N1"): 0.83,
    ("S1", "N2"): 0.68,
    ("S1", "O1"): 0.84,
    ("S1", "O2"): 0.43,
    ("S1", "F2"): 0.28,
    ("S1", "P1"): 0.65,
    ("S1", "P2"): 0.65,
    ("S1", "S1"): 0.68,

    ("S2", "C1"): 0.69,
    ("S2", "B0"): 0.44,
    ("S2", "N1"): 0.78,
    ("S2", "N2"): 0.73,
    ("S2", "O1"): 0.85,
    ("S2", "O2"): 0.54,
    ("S2", "F2"): 0.32,
    ("S2", "P1"): 0.48,
    ("S2", "P2"): 0.60,
    ("S2", "S1"): 0.58,
    ("S2", "S2"): 0.63,

    ("Cl2", "C1"): 0.62,
    ("Cl2", "B0"): 0.41,
    ("Cl2", "N1"): 0.77,
    ("Cl2", "N2"): 0.80,
    ("Cl2", "O1"): 0.88,
    ("Cl2", "O2"): 0.70,
    ("Cl2", "F2"): 0.51,
    ("Cl2", "P1"): 0.35,
    ("Cl2", "P2"): 0.55,
    ("Cl2", "S1"): 0.52,
    ("Cl2", "S2"): 0.59,
    ("Cl2", "Cl2"): 0.68,
}

C2_COUPLINGS = {  # Estimated values
    ("C2", "C1"): 0.70,
    ("C2", "B0"): 0.60,
    ("C2", "N1"): 0.75,
    ("C2", "N2"): 0.65,
    ("C2", "O1"): 0.70,
    ("C2", "O2"): 0.50,
    ("C2", "P1"): 0.70,
    ("C2", "P2"): 0.60,
    ("C2", "S1"): 0.70,
    ("C2", "S2"): 0.55,
    ("C2", "F2"): 0.45,
    ("C2", "Cl2"): 0.50,
    ("C2", "C2"): 0.25,
}

Br2_COUPLINGS = { # Estimated values based from Cl2
    ("Br2", "C1"): 0.62,
    ("Br2", "B0"): 0.41,
    ("Br2", "N1"): 0.77,
    ("Br2", "N2"): 0.80,
    ("Br2", "O1"): 0.88,
    ("Br2", "O2"): 0.70,
    ("Br2", "F2"): 0.51,
    ("Br2", "P1"): 0.35,
    ("Br2", "P2"): 0.55,
    ("Br2", "S1"): 0.52,
    ("Br2", "S2"): 0.59,
    ("Br2", "Cl2"): 0.68,
}


def make_symmetric_couplings(couplings: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    out = {}
    for (a, b), val in couplings.items():
        out[(a, b)] = val
        out[(b, a)] = val
    return out


COUPLING_TABLE = make_symmetric_couplings({
    **RAUK_COUPLINGS,
    **C2_COUPLINGS,
    **Br2_COUPLINGS,
})

HALOGENS = {"F", "Cl", "Br"}
HETERO_MAIN = {"N", "O", "S", "P"}


MOSelector = str | int  # "HOMO", "LUMO", "HOMO-1", "LUMO+2", or integer index


@dataclass(frozen=True)
class HuckelResult:
    atom_types: list[str]
    n_pi_electrons: int
    eigenvalues_lambda: list[float]
    # eigenvectors[orbital_idx][atom_idx] = MO coefficient cₖᵢ
    eigenvectors: list[list[float]]
    occupations: list[int]
    homo_index: int | None
    lumo_index: int | None
    homo_lambda: float | None
    lumo_lambda: float | None
    gap_lambda: float | None


def kekulized_mol(mol: rdchem.Mol) -> rdchem.Mol:
    kmol = Chem.Mol(mol)
    try:
        Chem.Kekulize(kmol, clearAromaticFlags=True)
    except Exception as exc:
        raise EvaluationError(f"Cannot kekulize molecule for Huckel evaluation: {exc}") from exc
    return kmol


def atom_has_explicit_pi_bond(atom: rdchem.Atom) -> bool:
    for bond in atom.GetBonds():
        if bond.GetBondType() in (rdchem.BondType.DOUBLE, rdchem.BondType.TRIPLE):
            return True
    return False


def classify_atom(atom: rdchem.Atom) -> str:
    symbol = atom.GetSymbol()
    in_pi = atom_has_explicit_pi_bond(atom)

    if symbol == "B":
        return "B0"

    if symbol == "C":
        return "C1" if in_pi else "C2"

    if symbol in HETERO_MAIN:
        return f"{symbol}1" if in_pi else f"{symbol}2"

    if symbol in HALOGENS:
        return f"{symbol}2"

    raise EvaluationError(
        f"Unsupported atom for Huckel evaluation: {symbol} at index {atom.GetIdx()}"
    )


def coupling_value(type1: str, type2: str) -> float:
    key = (type1, type2)
    try:
        return COUPLING_TABLE[key]
    except KeyError as exc:
        raise EvaluationError(f"Missing Huckel coupling for {type1}-{type2}") from exc


def atom_types_from_mol(mol: rdchem.Mol) -> list[str]:
    kmol = kekulized_mol(mol)
    return [classify_atom(atom) for atom in kmol.GetAtoms()]


def total_pi_electrons(atom_types: list[str]) -> int:
    return sum(PI_ELECTRONS[t] for t in atom_types)


def build_reduced_huckel_matrix(mol: rdchem.Mol) -> tuple[np.ndarray, list[str]]:
    atom_types = atom_types_from_mol(mol)
    n_atoms = mol.GetNumAtoms()

    if n_atoms == 0:
        raise EvaluationError("Empty molecule cannot be evaluated with Huckel model.")

    hmat = np.zeros((n_atoms, n_atoms), dtype=float)

    for i, atype in enumerate(atom_types):
        hmat[i, i] = H_DIAG[atype]

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        tij = coupling_value(atom_types[i], atom_types[j])
        hmat[i, j] = tij
        hmat[j, i] = tij

    return hmat, atom_types


def solve_huckel_mol(mol: rdchem.Mol, beta_ev: float | None = None) -> HuckelResult:
    hmat, atom_types = build_reduced_huckel_matrix(mol)

    eigenvalues, eigvecs = np.linalg.eigh(hmat)

    # With E = alpha + lambda * beta and beta < 0,
    # larger lambda values correspond to lower physical energies.
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigvecs = eigvecs[:, order]

    n_elec = total_pi_electrons(atom_types)
    occupations = np.zeros(len(eigenvalues), dtype=int)

    # Fill orbitals level by level. Orbitals with the same eigenvalue (within
    # DEGENERACY_TOL) form one degenerate level. A partially filled degenerate
    # level is filled by Hund's rule: one electron per orbital first, then
    # pairing. This avoids putting 2 electrons in one degenerate orbital while
    # leaving its partner empty (which would yield a spurious zero HOMO-LUMO gap).
    DEGENERACY_TOL = 1e-9
    n_orbitals = len(eigenvalues)
    remaining = n_elec
    i = 0
    while i < n_orbitals and remaining > 0:
        j = i + 1
        while j < n_orbitals and abs(eigenvalues[j] - eigenvalues[i]) <= DEGENERACY_TOL:
            j += 1
        group = range(i, j)
        capacity = 2 * (j - i)
        if remaining >= capacity:
            for k in group:
                occupations[k] = 2
            remaining -= capacity
        else:
            for k in group:  # first pass: one electron each (Hund's rule)
                if remaining == 0:
                    break
                occupations[k] = 1
                remaining -= 1
            for k in group:  # second pass: pair up the rest
                if remaining == 0:
                    break
                occupations[k] = 2
                remaining -= 1
        i = j

    occ_idx = np.where(occupations > 0)[0]
    virt_idx = np.where(occupations == 0)[0]

    homo_index = int(occ_idx[-1]) if len(occ_idx) else None
    lumo_index = int(virt_idx[0]) if len(virt_idx) else None

    homo_lambda = None if homo_index is None else float(eigenvalues[homo_index])
    lumo_lambda = None if lumo_index is None else float(eigenvalues[lumo_index])

    if homo_lambda is None or lumo_lambda is None:
        gap_lambda = None
    else:
        # Positive lambda gap, assuming beta < 0.
        gap_lambda = float(homo_lambda - lumo_lambda)

    return HuckelResult(
        atom_types=atom_types,
        n_pi_electrons=int(n_elec),
        eigenvalues_lambda=[float(x) for x in eigenvalues],
        # eigenvectors[i] = coefficients of orbital i over all atoms
        eigenvectors=[[float(eigvecs[k, i]) for k in range(len(atom_types))]
                      for i in range(len(eigenvalues))],
        occupations=[int(x) for x in occupations],
        homo_index=homo_index,
        lumo_index=lumo_index,
        homo_lambda=homo_lambda,
        lumo_lambda=lumo_lambda,
        gap_lambda=gap_lambda,
    )


class Huckel(Evaluation):
    """Compute and cache the full Huckel result."""

    def __init__(self, beta_ev: float | None = None) -> None:
        super().__init__("huckel")
        self.beta_ev = beta_ev

    @override
    def _evaluate(self, molecule: Molecule) -> HuckelResult:
        mol_graph = molecule.get_representation(MolecularGraph)
        return solve_huckel_mol(mol_graph.mol, beta_ev=self.beta_ev)


class HuckelHOMO(Evaluation):
    def __init__(self) -> None:
        super().__init__("huckel_homo_lambda")

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        result: HuckelResult = molecule.value("huckel")
        if result.homo_lambda is None:
            raise EvaluationError("Huckel HOMO is undefined.")
        return result.homo_lambda


class HuckelLUMO(Evaluation):
    def __init__(self) -> None:
        super().__init__("huckel_lumo_lambda")

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        result: HuckelResult = molecule.value("huckel")
        if result.lumo_lambda is None:
            raise EvaluationError("Huckel LUMO is undefined.")
        return result.lumo_lambda


class HuckelGap(Evaluation):
    def __init__(self) -> None:
        super().__init__("huckel_gap_lambda")

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        result: HuckelResult = molecule.value("huckel")
        if result.gap_lambda is None:
            raise EvaluationError("Huckel gap is undefined.")
        return result.gap_lambda


def resolve_mo_index(result: HuckelResult, selector: MOSelector) -> int:
    """Resolve an MO selector to an integer orbital index.

    Accepted forms: integer index, "HOMO", "LUMO", "HOMO-n", "LUMO+n".
    """
    n = len(result.eigenvalues_lambda)
    if isinstance(selector, int):
        if not (0 <= selector < n):
            raise EvaluationError(f"MO index {selector} out of range [0, {n}).")
        return selector
    s = selector.upper().strip()
    if s == "HOMO":
        if result.homo_index is None:
            raise EvaluationError("No HOMO defined for this molecule.")
        return result.homo_index
    if s == "LUMO":
        if result.lumo_index is None:
            raise EvaluationError("No LUMO defined for this molecule.")
        return result.lumo_index
    if s.startswith("HOMO-"):
        offset = int(s[5:])
        if result.homo_index is None:
            raise EvaluationError("No HOMO defined for this molecule.")
        idx = result.homo_index - offset
        if not (0 <= idx < n):
            raise EvaluationError(f"HOMO-{offset} index {idx} out of range.")
        return idx
    if s.startswith("LUMO+"):
        offset = int(s[5:])
        if result.lumo_index is None:
            raise EvaluationError("No LUMO defined for this molecule.")
        idx = result.lumo_index + offset
        if not (0 <= idx < n):
            raise EvaluationError(f"LUMO+{offset} index {idx} out of range.")
        return idx
    raise EvaluationError(
        f"Unknown MO selector {selector!r}. Use 'HOMO', 'LUMO', 'HOMO-n', 'LUMO+n', or int."
    )


class HuckelOrbitalLambda(Evaluation):
    """Returns the eigenvalue λ of any selected molecular orbital.

    selector: "HOMO", "LUMO", "HOMO-1", "LUMO+2", or integer index.
    """

    def __init__(self, selector: MOSelector = "HOMO") -> None:
        super().__init__(f"huckel_orbital_lambda_{selector}")
        self.selector = selector

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        result: HuckelResult = molecule.value("huckel")
        idx = resolve_mo_index(result, self.selector)
        return result.eigenvalues_lambda[idx]


class HuckelOrbitalOverlap(Evaluation):
    """Computes the orbital overlap Σₖ cₖ₁·cₖ₂ between two molecular orbitals.

    Orthogonal orbitals return 0. Identical orbitals return 1 (normalised).
    selector1, selector2: "HOMO", "LUMO", "HOMO-1", "LUMO+2", or integer index.
    """

    def __init__(
        self,
        selector1: MOSelector = "HOMO",
        selector2: MOSelector = "LUMO",
    ) -> None:
        super().__init__(f"huckel_overlap_{selector1}_{selector2}")
        self.selector1 = selector1
        self.selector2 = selector2

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        result: HuckelResult = molecule.value("huckel")
        idx1 = resolve_mo_index(result, self.selector1)
        idx2 = resolve_mo_index(result, self.selector2)
        c1 = np.array(result.eigenvectors[idx1])
        c2 = np.array(result.eigenvectors[idx2])
        return float(np.dot(c1, c2))


class HuckelOrbitalCombination(Evaluation):
    """Linear combination of orbital densities summed over all atoms.

    For each (selector, sign, coeff) term, computes:
        combined[k] = Σᵢ (sign_i · coeff_i · cₖᵢ²)   for each atom k

    Returns the scalar Σₖ combined[k].

    Example — HOMO minus LUMO density:
        HuckelOrbitalCombination([("HOMO", +1, 1.0), ("LUMO", -1, 1.0)])
    """

    def __init__(
        self,
        terms: list[tuple[MOSelector, float, float]],
        name: str | None = None,
    ) -> None:
        if name is None:
            parts = [f"{'p' if s >= 0 else 'm'}{abs(c):.2g}{sel}" for sel, s, c in terms]
            name = "huckel_combination_" + "_".join(parts)
        super().__init__(name)
        self.terms = terms

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        result: HuckelResult = molecule.value("huckel")
        n_atoms = len(result.atom_types)
        combined = np.zeros(n_atoms)
        for selector, sign, coeff in self.terms:
            idx = resolve_mo_index(result, selector)
            coeffs = np.array(result.eigenvectors[idx])
            combined += sign * coeff * coeffs ** 2
        return float(np.sum(combined))


class HuckelOrbitalLambdaOverlap(Evaluation):
    """Spatial overlap between orbitals ignoring phase: Σₖ ∏ᵢ |cₖᵢ|.

    For two orbitals: Σₖ |cₖ₁| · |cₖ₂|
    Unlike the dot-product overlap (which is 0 for orthogonal MOs), this
    measures how much the orbital densities share the same atoms regardless
    of sign. Matches the Lambda mode in the huckel_3D notebook.

    selectors: list of MO selectors, e.g. ["HOMO", "LUMO"] or ["HOMO-1", "HOMO", "LUMO"]
    """

    def __init__(self, selectors: list[MOSelector]) -> None:
        name = "huckel_lambda_" + "_".join(str(s) for s in selectors)
        super().__init__(name)
        self.selectors = selectors

    @override
    def _evaluate(self, molecule: Molecule) -> float:
        result: HuckelResult = molecule.value("huckel")
        n_atoms = len(result.atom_types)
        combined = np.ones(n_atoms)
        for selector in self.selectors:
            idx = resolve_mo_index(result, selector)
            combined *= np.abs(np.array(result.eigenvectors[idx]))
        return float(np.sum(combined))

