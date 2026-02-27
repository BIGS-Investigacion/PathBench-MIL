"""
Analiza el estudio Optuna de una optimización PathBench y muestra:
  - Resumen de trials (completados / podados)
  - Mejor trial global
  - Mejor trial por arquitectura MIL

Uso:
    source /shared/home/PARADIS/PathBench-MIL/pathbench_env/bin/activate
    python3 claude/analyze_optimization.py [--exp EXP_DIR] [--study STUDY_NAME]

Valores por defecto:
    --exp    experiments/brca_pam50_virchow2_prueba_13_optimizado
    --study  Exp2   (nombre del estudio Optuna dentro del .db)
"""

import argparse
import os
import sys

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: optuna no instalado en el entorno activo.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--exp', default='experiments/brca_pam50_virchow2_prueba_13_optimizado',
                    help='Directorio del experimento')
parser.add_argument('--study', default='Exp2',
                    help='Nombre del estudio Optuna (study_name en el yaml)')
args = parser.parse_args()

db_path = os.path.join(args.exp, 'optimization', 'optuna_study.db')
if not os.path.exists(db_path):
    print(f"ERROR: no se encuentra la base de datos Optuna en: {db_path}")
    sys.exit(1)

storage = f'sqlite:///{db_path}'

# Intentar con el nombre indicado; si falla, listar los disponibles
try:
    study = optuna.load_study(study_name=args.study, storage=storage)
except Exception:
    available = optuna.get_all_study_names(storage=storage)
    print(f"Estudio '{args.study}' no encontrado. Disponibles: {available}")
    if len(available) == 1:
        study = optuna.load_study(study_name=available[0], storage=storage)
        print(f"Cargando '{available[0]}' automáticamente.")
    else:
        sys.exit(1)

trials = study.trials
completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
pruned    = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]

print(f"\n{'='*60}")
print(f"Estudio: {study.study_name}")
print(f"Total trials : {len(trials)}")
print(f"  Completados: {len(completed)}")
print(f"  Podados    : {len(pruned)}")
print(f"{'='*60}\n")

if not completed:
    print("No hay trials completados.")
    sys.exit(0)

# Mejor global
best = study.best_trial
print(f"Mejor trial global:")
print(f"  Trial #{best.number:03d}  value={best.value:.4f}")
print(f"  Params: {best.params}\n")

# Mejor por arquitectura MIL
mil_key = 'mil'
arch_bests: dict = {}
for t in completed:
    mil = t.params.get(mil_key, 'unknown')
    if mil not in arch_bests or t.value > arch_bests[mil].value:
        arch_bests[mil] = t

print("Mejor trial por arquitectura MIL:")
print(f"  {'Arquitectura':<20} {'Trial':>6}  {'Value':>8}  {'Params'}")
for mil, t in sorted(arch_bests.items(), key=lambda x: -x[1].value):
    other_params = {k: v for k, v in t.params.items() if k != mil_key}
    print(f"  {mil:<20} {t.number:>6}  {t.value:>8.4f}  {other_params}")
print()

# Tabla completa de completados ordenada por valor
print("Todos los trials completados (ordenados por valor desc):")
print(f"  {'#':>5}  {'Value':>8}  {'MIL':<22}  Otros params")
for t in sorted(completed, key=lambda x: -x.value):
    mil = t.params.get(mil_key, 'unknown')
    other = {k: v for k, v in t.params.items() if k != mil_key}
    print(f"  {t.number:>5}  {t.value:>8.4f}  {mil:<22}  {other}")