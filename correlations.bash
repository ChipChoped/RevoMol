#!/bin/bash

molecules=(
    "C"
    "C(O)(=O)C1=C(OC(C)=O)C=CC=C1"  # Aspirin
    "CN1C(=NC2=C1C(=O)N(C(=O)N2C)C)CO"  # Caffeine
)

n_steps=1000

action_spaces=(
    "AddAtomMG"
    "RemoveAtomMG"
    "ChangeBondMG"
    "AddAtomMG RemoveAtomMG"
    "AddAtomMG RemoveAtomMG ChangeBondMG"
)

for molecule in "${molecules[@]}"
do
    for action_space in "${action_spaces[@]}"
    do
        python ./scripts/correlations.py "$molecule" "$n_steps" "$action_space"
    done
done
