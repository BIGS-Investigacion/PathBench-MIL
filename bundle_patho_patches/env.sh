# ============================================================================
# env.sh  —  Activa el entorno de ejecución.
#   source env.sh
#
# En una instalación SANA de PathBench basta con activar su venv:
#     source /ruta/a/PathBench-MIL/pathbench_env/bin/activate
#
# En este clúster (tras el traslado de /shared/home/PARADIS) el intérprete base
# del pathbench_env desapareció, así que reconstruimos el entorno combinando:
#   - python3.10 de conda (clam_latest)
#   - site-packages del pathbench_env (torch, lightning, rich, ...)
#   - slideflow desde el fork del repo
# Ajusta estas rutas si tu instalación difiere.
# ============================================================================
export PATHBENCH_REPO="${PATHBENCH_REPO:-/shared/home/jorgarcia/PathBench-MIL}"

# Opción A (instalación sana): descomenta y ajusta
# source "$PATHBENCH_REPO/pathbench_env/bin/activate"

# Opción B (entorno recuperado en este clúster):
export _PY=/home/jorgarcia/.conda/envs/clam_latest/bin/python3.10
export _SP="$PATHBENCH_REPO/pathbench_env/lib/python3.10/site-packages"
export _FORK="$PATHBENCH_REPO/slideflow_fork"
export PYTHONPATH="$_FORK:$_SP:${PYTHONPATH:-}"
# 'python3' apuntará al intérprete correcto vía alias de función
python3() { "$_PY" "$@"; }
export -f python3 2>/dev/null || true

export SF_SLIDE_BACKEND=cucim
export SF_BACKEND=torch
echo "Entorno listo. PATHBENCH_REPO=$PATHBENCH_REPO"
