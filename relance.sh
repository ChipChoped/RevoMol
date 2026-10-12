#!/bin/bash
# Envoie le code modifié (scripts/, notebooks/) sur le serveur et lance plan_tri.ipynb via SLURM.
# Usage : ./relance.sh
set -euo pipefail

SERVER="chimstg1@moltech-calc0"
REMOTE_DIR="~/Lea/test/HuckMol-2.0"

echo "→ envoi du code vers le serveur…"
rsync -av --exclude 'logs/' --exclude 'output/' --exclude 'logs_csv/' \
  --exclude '__pycache__/' --exclude '.ipynb_checkpoints/' \
  scripts notebooks requirements.txt \
  "$SERVER:$REMOTE_DIR/"

echo "→ soumission du calcul…"
ssh "$SERVER" "cd $REMOTE_DIR && sbatch plan_tri.sbatch"

echo "→ pour suivre : ssh $SERVER 'ls $REMOTE_DIR/notebooks/random_walk/results/plan_tri/runs/*/rep*_meta.json 2>/dev/null | wc -l'"
