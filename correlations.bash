#!/bin/bash

molecules=(
    "C"
    "C(O)(=O)C1=C(OC(C)=O)C=CC=C1"  # Aspirin
    "CN1C(=NC2=C1C(=O)N(C(=O)N2C)C)CO"  # Caffeine
)

n_steps=1000

only_valid=1

action_spaces=(
    "AddAtomMG"
    "RemoveAtomMG"
    "ChangeBondMG"
    "MoveGroupMG"
    "AddAtomMG RemoveAtomMG"
    "AddAtomMG RemoveAtomMG ChangeBondMG"
    "AddAtomMG RemoveAtomMG ChangeBondMG MoveGroupMG"
)

for molecule in "${molecules[@]}"
do
    for action_space in "${action_spaces[@]}"
    do
        python ./scripts/correlations.py "$molecule" "$n_steps" "$only_valid" "$action_space"
    done
done
