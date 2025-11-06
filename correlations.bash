#!/bin/bash

strategy="random"

molecules=(
#    "C"  # Methane
#    "C(O)(=O)C1=C(OC(C)=O)C=CC=C1"  # Aspirin
#    "CN1C(=NC2=C1C(=O)N(C(=O)N2C)C)CO"  # Caffeine
    "C(C(=O)O)C(CC(=O)O)(C(=O)O)O"  # Citric acid
#    "S1C=CSC1=C2SC=CS2"  # TTF
#    "C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)C[C@@H](CC3=CC(=C(C=C3F)F)F)N"  # Sitagliptin
)

n_steps=1000

action_spaces=(
    "ChangeBondMG"
#    "AddAtomMG RemoveAtomMG"
#    "AddAtomMG RemoveAtomMG ChangeBondMG"
)

for molecule in "${molecules[@]}"
do
    for action_space in "${action_spaces[@]}"
    do
        for seed in {1..5}
        do
            echo "$strategy" "$molecule" "$n_steps" -a "$action_space" --seed "$seed"
            # shellcheck disable=SC2086
            python ./scripts/fitness_landscape_analysis.py "$strategy" "$molecule"\
            "$n_steps" -a $action_space --seed "$seed" -b --only-valid
        done
    done
done
