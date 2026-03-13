"""
Comparación estadística entre etiquetas de TCGA y CPTAC dentro de cada grupo.

Variables: ESTRUCTURA GLANDULAR, ATIPIA NUCLEAR, MITOSIS, NECROSIS, INFILTRADO_LI, INFILTRADO_PMN
Test: Mann-Whitney U (muestras pequeñas, variables ordinales)
Corrección de múltiples comparaciones: Benjamini-Hochberg (FDR) por etiqueta CPTAC

Grupos de comparación:
  1. BASAL, HER2-ENRICHED, LUMINAL-A, LUMINAL-B, NORMAL-LIKE
  2. PR-POSITIVE, PR-NEGATIVE
  3. ER-POSITIVE, ER-NEGATIVE
  4. HER2-POSITIVE, HER2-NEGATIVE

Uso:
    python comparacion_estadistica.py --excel ruta/al/archivo.xlsx
    python comparacion_estadistica.py --excel ruta/al/archivo.xlsx --alpha 0.05
    python comparacion_estadistica.py --excel ruta/al/archivo.xlsx --etiqueta BASAL
"""

import argparse
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ── Configuración ──────────────────────────────────────────────────────────────

VARIABLES = [
    "ESTRUCTURA GLANDULAR",
    "ATIPIA NUCLEAR",
    "MITOSIS",
    "NECROSIS",
    "INFILTRADO_LI",
    "INFILTRADO_PMN",
]

# Grupos de comparación: solo se comparan etiquetas del mismo grupo entre TCGA y CPTAC
GRUPOS = [
    {"BASAL", "HER2-ENRICHED", "LUMINAL-A", "LUMINAL-B", "NORMAL-LIKE"},
    {"PR-POSITIVE", "PR-NEGATIVE"},
    {"ER-POSITIVE", "ER-NEGATIVE"},
    {"HER2-POSITIVE", "HER2-NEGATIVE"},
]


def grupo_de(etiqueta: str) -> set | None:
    """Devuelve el grupo al que pertenece la etiqueta, o None si no está en ninguno."""
    for g in GRUPOS:
        if etiqueta in g:
            return g
    return None


# ── Utilidades ─────────────────────────────────────────────────────────────────

def cargar_datos(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee las pestañas TCGA y CPTAC del Excel."""
    tcga  = pd.read_excel(path, sheet_name="TCGA",  header=0)
    cptac = pd.read_excel(path, sheet_name="CPTAC", header=0)
    tcga.columns  = tcga.columns.str.strip()
    cptac.columns = cptac.columns.str.strip()
    tcga["ETIQUETA"]  = tcga["ETIQUETA"].str.strip().str.upper()
    cptac["ETIQUETA"] = cptac["ETIQUETA"].str.strip().str.upper()
    return tcga, cptac


def bh_correction(pvals: list[float]) -> np.ndarray:
    """Corrección Benjamini-Hochberg. Devuelve p-valores ajustados."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    orden   = np.argsort(pvals)
    pvals_s = np.array(pvals)[orden]
    adj     = np.minimum(1.0, pvals_s * n / (np.arange(n) + 1))
    # monotonía
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    resultado = np.empty(n)
    resultado[orden] = adj
    return resultado


def effect_size_rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Correlación biserial de rangos (effect size para Mann-Whitney)."""
    nx, ny = len(x), len(y)
    u, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    r = 1 - (2 * u) / (nx * ny)
    return round(float(r), 3)


def interpretar_r(r: float) -> str:
    ar = abs(r)
    if ar < 0.1:  return "negligible"
    if ar < 0.3:  return "small"
    if ar < 0.5:  return "medium"
    return "large"


# ── Análisis principal ─────────────────────────────────────────────────────────

def analizar(tcga: pd.DataFrame, cptac: pd.DataFrame,
             etiqueta_tcga: str, alpha: float) -> pd.DataFrame:
    """
    Para una etiqueta TCGA, compara contra todas las etiquetas CPTAC de su grupo:
      - Mann-Whitney U (two-sided)
      - p ajustada (BH) dentro de cada etiqueta CPTAC (sobre las 6 variables)
      - effect size (rank-biserial r)
      - estadísticos descriptivos
    """
    etiqueta_tcga = etiqueta_tcga.upper()
    grupo_tcga_raw = tcga[tcga["ETIQUETA"] == etiqueta_tcga]
    if grupo_tcga_raw.empty:
        return pd.DataFrame()  # etiqueta no presente en TCGA, se omite

    grupo = grupo_de(etiqueta_tcga)
    if grupo is None:
        return pd.DataFrame()

    etiquetas_cptac_disponibles = set(cptac["ETIQUETA"].unique())
    etiquetas_cptac = sorted(grupo & etiquetas_cptac_disponibles)
    registros = []

    for etiq_c in etiquetas_cptac:
        grupo_cptac = cptac[cptac["ETIQUETA"] == etiq_c]
        pvals_raw   = []
        filas_tmp   = []

        for var in VARIABLES:
            x = grupo_tcga_raw[var].dropna().values
            y = grupo_cptac[var].dropna().values

            if len(x) < 3 or len(y) < 3:
                pvals_raw.append(np.nan)
                filas_tmp.append({
                    "TCGA_label": etiqueta_tcga,
                    "CPTAC_label": etiq_c,
                    "Variable": var,
                    "n_TCGA": len(x), "n_CPTAC": len(y),
                    "median_TCGA": np.nan, "median_CPTAC": np.nan,
                    "mean_TCGA": np.nan,  "mean_CPTAC": np.nan,
                    "U_stat": np.nan, "p_value": np.nan,
                    "p_adj_BH": np.nan,
                    "effect_r": np.nan, "effect_magnitude": "—",
                    "significant": False,
                })
                continue

            try:
                u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")
            except ValueError:
                u_stat, p_val = np.nan, np.nan

            pvals_raw.append(p_val)
            filas_tmp.append({
                "TCGA_label": etiqueta_tcga,
                "CPTAC_label": etiq_c,
                "Variable": var,
                "n_TCGA": len(x), "n_CPTAC": len(y),
                "median_TCGA": np.median(x), "median_CPTAC": np.median(y),
                "mean_TCGA": round(np.mean(x), 3), "mean_CPTAC": round(np.mean(y), 3),
                "U_stat": round(u_stat, 1) if not np.isnan(u_stat) else np.nan,
                "p_value": round(p_val, 4)  if not np.isnan(p_val)  else np.nan,
                "p_adj_BH": np.nan,
                "effect_r": np.nan,
                "effect_magnitude": "—",
                "significant": False,
            })

        # corrección BH sobre las 6 variables de esta etiqueta CPTAC
        mask_valid = [not np.isnan(p) for p in pvals_raw]
        pvals_valid = [p for p, m in zip(pvals_raw, mask_valid) if m]
        adj_valid   = bh_correction(pvals_valid) if pvals_valid else []

        idx_adj = 0
        for i, (fila, valido) in enumerate(zip(filas_tmp, mask_valid)):
            if valido:
                p_adj = float(adj_valid[idx_adj]); idx_adj += 1
                x = grupo_tcga_raw[VARIABLES[i]].dropna().values
                y = grupo_cptac[VARIABLES[i]].dropna().values
                er = effect_size_rank_biserial(x, y)
                fila.update({
                    "p_adj_BH": round(p_adj, 4),
                    "effect_r": er,
                    "effect_magnitude": interpretar_r(er),
                    "significant": p_adj < alpha,
                })
            registros.append(fila)

    return pd.DataFrame(registros)


# ── Salida en consola ──────────────────────────────────────────────────────────

def imprimir_resumen(df: pd.DataFrame, alpha: float):
    print("\n" + "═" * 80)
    print(f"  Comparación TCGA [{df['TCGA_label'].iloc[0]}]  vs  etiquetas CPTAC del mismo grupo")
    print(f"  Test: Mann-Whitney U (two-sided) | Corrección: BH (FDR) | α = {alpha}")
    print("═" * 80)

    for etiq_c in df["CPTAC_label"].unique():
        sub = df[df["CPTAC_label"] == etiq_c]
        sig = sub[sub["significant"]]
        print(f"\n▶  CPTAC: {etiq_c}  ({len(sub)} variables testadas, {len(sig)} significativas)")
        print(f"   {'Variable':<26} {'med_TCGA':>9} {'med_CPTAC':>10} "
              f"{'p_raw':>8} {'p_BH':>8} {'r':>6} {'mag':>10}  {'sig':>4}")
        print("   " + "-" * 72)
        for _, row in sub.iterrows():
            marca = "  ✓" if row["significant"] else ""
            print(f"   {row['Variable']:<26} {row['median_TCGA']:>9.2f} {row['median_CPTAC']:>10.2f} "
                  f"{row['p_value']:>8.4f} {row['p_adj_BH']:>8.4f} "
                  f"{row['effect_r']:>6.3f} {row['effect_magnitude']:>10}{marca}")

    total_sig = df["significant"].sum()
    print(f"\n  Total comparaciones significativas (p_BH < {alpha}): {total_sig} / {len(df)}")
    print("═" * 80 + "\n")

    # Tabla resumen por etiqueta CPTAC
    print("  TABLA RESUMEN  —  Suma |effect_r| en comparaciones significativas")
    print(f"  {'CPTAC_label':<20} {'n_sig':>6} {'sum_|r|_sig':>12}  variables_sig")
    print("  " + "-" * 70)
    for etiq_c in df["CPTAC_label"].unique():
        sub = df[df["CPTAC_label"] == etiq_c]
        sig = sub[sub["significant"]]
        suma_r = sig["effect_r"].abs().sum()
        vars_sig = ", ".join(sig["Variable"].tolist()) if not sig.empty else "—"
        print(f"  {etiq_c:<20} {len(sig):>6} {suma_r:>12.3f}  {vars_sig}")
    print("═" * 80 + "\n")


def construir_matriz(df_grupo: pd.DataFrame) -> pd.DataFrame:
    """
    A partir del DataFrame combinado de un grupo, construye una matriz:
      filas    = etiquetas TCGA
      columnas = etiquetas CPTAC
      valores  = suma de |effect_r| de comparaciones significativas
    """
    tcga_labels  = sorted(df_grupo["TCGA_label"].unique())
    cptac_labels = sorted(df_grupo["CPTAC_label"].unique())

    data = {}
    for etiq_c in cptac_labels:
        col = []
        for etiq_t in tcga_labels:
            sub = df_grupo[
                (df_grupo["TCGA_label"]  == etiq_t) &
                (df_grupo["CPTAC_label"] == etiq_c) &
                (df_grupo["significant"] == True)
            ]
            col.append(round(sub["effect_r"].abs().sum(), 3))
        data[etiq_c] = col

    return pd.DataFrame(data, index=tcga_labels)


def imprimir_matriz(matriz: pd.DataFrame, nombre_grupo: str):
    ancho_idx = max(len(s) for s in matriz.index) + 2
    ancho_col = max(max(len(c) for c in matriz.columns), 8) + 2
    linea = "─" * (ancho_idx + ancho_col * len(matriz.columns) + 2)

    print(f"\n  MATRIZ  —  Grupo: {nombre_grupo}")
    print(f"  Valores = Σ|effect_r| comparaciones significativas (TCGA fila vs CPTAC columna)")
    print("  " + linea)
    titulo_idx = "TCGA \\ CPTAC"
    encabezado = f"  {titulo_idx:<{ancho_idx}}" + "".join(f"{c:>{ancho_col}}" for c in matriz.columns)
    print(encabezado)
    print("  " + linea)
    for idx, row in matriz.iterrows():
        fila = f"  {idx:<{ancho_idx}}" + "".join(f"{v:>{ancho_col}.3f}" for v in row)
        print(fila)
    print("  " + linea + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--excel",    required=True, help="Ruta al archivo .xlsx")
    parser.add_argument("--etiqueta", default=None,
                        help="Etiqueta TCGA específica (opcional; si se omite, procesa todos los grupos)")
    parser.add_argument("--alpha",    type=float, default=0.05,
                        help="Nivel de significación (default: 0.05)")
    parser.add_argument("--output",   default=None,
                        help="Prefijo para los CSV de salida (default: 'resultados')")
    args = parser.parse_args()

    print(f"\n[INFO] Cargando datos de '{args.excel}'...")
    tcga, cptac = cargar_datos(args.excel)
    tcga_etiquetas  = set(tcga["ETIQUETA"].unique())
    cptac_etiquetas = set(cptac["ETIQUETA"].unique())
    print(f"[INFO] TCGA etiquetas:  {sorted(tcga_etiquetas)}")
    print(f"[INFO] CPTAC etiquetas: {sorted(cptac_etiquetas)}")

    prefijo = args.output or "resultados"

    # Determinar qué grupos procesar
    if args.etiqueta:
        etiq = args.etiqueta.upper()
        g = grupo_de(etiq)
        if g is None:
            sys.exit(f"[ERROR] Etiqueta '{etiq}' no pertenece a ningún grupo definido.")
        grupos_a_procesar = [g]
    else:
        grupos_a_procesar = GRUPOS

    todos_los_resultados = []

    for grupo in grupos_a_procesar:
        nombre_grupo = " | ".join(sorted(grupo))
        etiquetas_tcga_grupo  = sorted(grupo & tcga_etiquetas)
        etiquetas_cptac_grupo = sorted(grupo & cptac_etiquetas)

        if not etiquetas_tcga_grupo or not etiquetas_cptac_grupo:
            print(f"\n[AVISO] Grupo '{nombre_grupo}': sin datos suficientes, se omite.")
            continue

        print(f"\n[INFO] Grupo: {nombre_grupo}")
        print(f"       TCGA disponibles:  {etiquetas_tcga_grupo}")
        print(f"       CPTAC disponibles: {etiquetas_cptac_grupo}")

        dfs_grupo = []
        for etiq_t in etiquetas_tcga_grupo:
            df_etiq = analizar(tcga, cptac, etiq_t, args.alpha)
            if df_etiq.empty:
                continue
            imprimir_resumen(df_etiq, args.alpha)
            dfs_grupo.append(df_etiq)

        if not dfs_grupo:
            continue

        df_grupo = pd.concat(dfs_grupo, ignore_index=True)
        todos_los_resultados.append(df_grupo)

        # Matriz de efectos acumulados
        matriz = construir_matriz(df_grupo)
        imprimir_matriz(matriz, nombre_grupo)

        # Guardar CSV detallado y matriz del grupo
        slug = nombre_grupo.replace(" | ", "_").replace(" ", "-")
        csv_detalle = f"{prefijo}_{slug}_detalle.csv"
        csv_matriz  = f"{prefijo}_{slug}_matriz.csv"
        df_grupo.to_csv(csv_detalle, index=False)
        matriz.to_csv(csv_matriz)
        print(f"[INFO] Guardado: {csv_detalle}")
        print(f"[INFO] Guardado: {csv_matriz}")

    # CSV global con todos los resultados
    if todos_los_resultados:
        df_global = pd.concat(todos_los_resultados, ignore_index=True)
        csv_global = f"{prefijo}_todos.csv"
        df_global.to_csv(csv_global, index=False)
        print(f"\n[INFO] Resultados completos guardados en '{csv_global}'")


if __name__ == "__main__":
    main()