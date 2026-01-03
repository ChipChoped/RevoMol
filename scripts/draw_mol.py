"""
Script to test the visualization of molecules.
"""

import os
import sys

# Add the parent directory to the path to import the module evomol
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# pylint: disable=wrong-import-position, import-error

from evomol import default_parameters as dp
from evomol.representation import Molecule
from evomol.visualization.draw_mol import (
    draw_in_matplotlib,
    draw_multiple_svgs_in_matplotlib,
    mol_to_svg,
)

dp.setup_default_parameters(
    accepted_atoms=["C", "O", "N", "F", "S"],
    max_heavy_atoms=38,
)

os.makedirs("output/visualization", exist_ok=True)

molecules = [
    # Molecule("C1=C2C1NC13CN2C1=N3"),
    # Molecule("CC12OC1C2=C1NN1"),
    # Molecule("C=CC12C3C4=NC31C42"),
    # Molecule("O=C1C(O)=CC12C=NO2"),
    # Molecule("OC1=C2N1NN(O)N2F"),
    # Molecule("C#CC(N)C(=C)C(N)N"),
    # Molecule("C1=NC=NC1C1=C2CN21"),
    # Molecule("N=C(O)N(N)N1C2=C1N2"),
    # Molecule("C1=C2NON=C3N(N1)N23"),
    # Molecule("NC(O)C1=C2C=C(C1)N2"),

    Molecule('C=C1NC2C3=C2N1N3C'),
    Molecule('C=C1NC2C3=C2N1N3C'),
    Molecule('CSOOCNS(S)(N(N)[SH](O)SC12OC1(S)C2=C1NN1)[SH](NOS)(SS)(SSCS)[SH](C)(C)(C)SC'),
    Molecule('OC12C34C56N7C38C3S9%10%11%12C%13N%14N%15C%16%17SS%145%18%19C5=[SH]6%14C(=[SH]%136S84%161S2148OS%172%13%16N[SH]9%17%20N%14C2C%17(C%20%13S536%16%18)S%10123N1N(OC%112OC%1943)[SH]18)C7%15%12'),
    Molecule('C#[SH](C)C1N(O)OC12C(OS(S#CNN)=[SH]C)([SH]=S)C(S(O)=NNF)C2(SN(F)F)[SH](C)N(SO)SO'),
    Molecule('CN=NN(NF)OC=O'),
    Molecule('C=C(C(=N)N)C(N)CC'),
    Molecule('CO[SH](NN(N)C1C2=C(C3C=NC=N3)N21)[SH](C)CN(NO)O[SH](O)(O)(C[SH](S)C(N)SC)SNNNSS'),
    Molecule('NN1[SH]234CN5C6C78C=C9N(N7S)[SH]27%10C2N%11ON%12S%13%14N%15O[SH]3%16%17NN3N%18C%19S3%20=[SH]73C5%13[SH]%143(N%16NN%18ON2S=%209%19%17)S6%12%151N%10C%1184'),
    Molecule('CC(O)=NC(N(F)ON=CN1C(N)C2N(N(S)N(O)SNF)ONC3(C=O)N2N13)C(N)(OC=O)C(O)F'),
]

# note_on_atoms = {0: "A", 1: "B", 2: "C"}
# note_on_bonds = {0: "bond A", 1: "bond B"}
# svg_1 = mol_to_svg(
#     molecules[0],
#     notes_on_atoms=note_on_atoms,
#     notes_on_bonds=note_on_bonds,
# )
#
# draw_in_matplotlib(
#     svg_1,
#     show=False,
#     save_to_path=os.path.join("output", "visualization", "CCO.png"),
# )

svgs = [mol_to_svg(mol) for mol in molecules]
draw_multiple_svgs_in_matplotlib(
    svgs,
    show=True,
    save_to_path=os.path.join("output", "visualization", "random_molecules.png"),
)
