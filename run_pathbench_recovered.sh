#!/bin/bash
# Entorno reconstruido tras el traslado de /shared/home/PARADIS:
#   - python3.10 de conda clam_latest (el base del pathbench_env desapareció)
#   - site-packages del pathbench_env (torch, lightning, rich, ...)
#   - slideflow desde el fork del repo
export SF_SLIDE_BACKEND=cucim
export SF_BACKEND=torch

PY=/home/jorgarcia/.conda/envs/clam_latest/bin/python3.10
SP=/home/jorgarcia/PathBench-MIL/pathbench_env/lib/python3.10/site-packages
FORK=/home/jorgarcia/PathBench-MIL/slideflow_fork
export PYTHONPATH="$FORK:$SP"

CONFIG_FILE=$1
"$PY" main.py --config "$CONFIG_FILE"
