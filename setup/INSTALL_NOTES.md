# Registro de cambios — PathBench-MIL

Este documento recoge todos los cambios realizados sobre el repositorio original para
conseguir un entorno funcional y ejecutar PathBench-MIL en Ubuntu 22.04 con una
RTX 3070 Ti (CUDA 12.8) y Python 3.10.

---

## Contexto de partida

El repositorio estaba clonado con submódulos y el entorno virtual `pathbench_env`
existía con la mayoría de paquetes ya instalados. Sin embargo, presentaba tres
problemas que impedían su ejecución:

1. **PyTorch no estaba instalado** en `pathbench_env` (el dispositivo `/home` tenía
   < 730 MB libres, insuficientes para descargar ~2 GB).
2. **Varios paquetes tenían versiones incompatibles** con las librerías del sistema
   (scipy, matplotlib) o entre sí.
3. **libvips no estaba instalado** en el sistema operativo, bloqueando la lectura
   de slides WSI.

---

## Parte I — Paquetes de Python instalados o modificados

Todos los comandos se ejecutan con el entorno virtual activo:

```bash
source pathbench_env/bin/activate
```

### 1. PyTorch — reutilizado del entorno `clam_latest`

En lugar de descargar PyTorch, se creó un archivo `.pth` para añadir los
`site-packages` del entorno conda `clam_latest` al path de búsqueda de
`pathbench_env`. Ambos entornos comparten **Python 3.10.0**, lo que garantiza
compatibilidad de paquetes compilados.

```bash
echo "/home/jorge/anaconda3/envs/clam_latest/lib/python3.10/site-packages" \
  > pathbench_env/lib/python3.10/site-packages/clam_torch.pth
```

Resultado: PyTorch 2.9.0+cu128 disponible con CUDA.

### 2. fastai y dependencias — versiones exactas requeridas

```bash
pip install fastai==2.7.14 --no-deps
pip install fastcore==1.5.29 --no-deps      # ≥1.6 rompe la API de fastai 2.7.14
pip install fastprogress==1.0.3 --no-deps   # ≥1.1 requiere fasthtml (Python ≥3.11)
pip install fastdownload==0.0.7 --no-deps
```

### 3. Librerías de supervivencia

```bash
pip install scikit-survival==0.23.0 --no-deps
pip install pycox==0.3.0 --no-deps
pip install torchtuples==0.2.2 --no-deps
```

### 4. lifelines — versión actualizada por incompatibilidad con scipy

`lifelines 0.28.0` (versión original del `requirements.txt`) usa
`scipy.integrate.trapz`, función **eliminada en scipy ≥ 1.14**. El entorno
`clam_latest` expone scipy 1.15.3, lo que causaba un `ImportError` en arranque.

```bash
pip install "lifelines>=0.29.0,<0.30" --no-deps
```

> `lifelines ≥ 0.30` usa `datetime.UTC`, disponible solo en Python ≥ 3.11.
> La versión 0.29.0 es el máximo compatible con Python 3.10 + scipy 1.15.

### 5. pytorch-lightning y utilidades

```bash
pip install pytorch-lightning==2.2.2 --no-deps
pip install lightning-utilities --no-deps
```

### 6. optuna y dependencias

```bash
pip install optuna==3.6.1 --no-deps
pip install colorlog --no-deps
pip install alembic --no-deps
pip install SQLAlchemy --no-deps
```

### 7. transformers y stack de HuggingFace

```bash
pip install transformers --no-deps          # instaló 5.2.0
pip install huggingface_hub --no-deps       # instaló 1.4.1
pip install tokenizers --no-deps
pip install safetensors --no-deps
pip install httpx --no-deps
pip install httpcore anyio h11 --no-deps
pip install typer==0.9.4 --no-deps          # ≥0.10 requiere annotated_doc
```

### 8. CONCH — instalado desde git

```bash
pip install \
  "git+https://github.com/Mahmoodlab/CONCH.git@02d6ac59cc20874bff0f581de258c2b257f69a84" \
  --no-deps
```

### 9. seaborn — versión actualizada por incompatibilidad con matplotlib

`seaborn 0.11.2` llama a `matplotlib.cm.register_cmap`, **eliminado en
matplotlib ≥ 3.7**.

```bash
pip install seaborn --upgrade --no-deps    # instaló 0.13.2
```

---

## Parte II — Dependencia del sistema operativo

### libvips42 — bloqueante para la lectura de slides WSI

Slideflow usa `pyvips` como backend por defecto para leer archivos `.svs`. La
librería de sistema `libvips.so.42` **no estaba instalada**, lo que provocaba que
el proceso se colgara al intentar crear el manifest de tiles (sin emitir ningún
error en el log).

**Síntoma observado:** proceso `python main.py` a 100% CPU durante más de 13 minutos
sin avance en `slideflow.log` tras la línea:

```text
[DEBUG] - No manifest at .../tfrecords/tcga/256px_20x; creating now
```


**Solución:**

```bash
sudo apt-get install -y libvips42 libvips-dev
```

**Verificación:**

```bash
source pathbench_env/bin/activate
python3 -c "import pyvips; print('libvips OK:', pyvips.version_string())"
```

---

## Parte III — Variables de entorno para el lanzamiento

Slideflow auto-detecta los backends, pero es más seguro fijarlos explícitamente
antes de cada ejecución:

```bash
export SF_SLIDE_BACKEND=libvips   # backend WSI (libvips o cucim)
export SF_BACKEND=torch           # backend deep learning
```

Se pueden añadir a `run_pathbench.sh` o al `.bashrc` del proyecto.

---

## Parte IV — Ajustes de configuración recomendados

### `num_workers`

El valor `0` en la config se sobreescribe internamente a solo `4` (ver
`benchmark.py` línea 412). Con 20 cores disponibles, se recomienda:

```yaml
num_workers: 16
```

### QC — eliminar Otsu-CLAHE

Usar dos métodos de QC aumenta el tiempo de extracción de tiles sin mejora
significativa para datasets estándar H&E. Se recomienda dejar solo el más rápido:

```yaml
qc:
  - GaussianV2
```

### Lanzamiento completo recomendado

```bash
source pathbench_env/bin/activate
export SF_SLIDE_BACKEND=libvips
export SF_BACKEND=torch
python main.py --config conf_test_small.yaml
```

---

## Parte V — Correcciones en datos y configuración de slides

Descubiertas durante la primera ejecución real con libvips activo.

### `tile_um: 20x` → `tile_um: 128` en `conf_test_small.yaml`

**Problema:** slideflow busca un nivel piramidal nativo a la magnificación indicada.
Varios slides CPTAC escaneados a 40x tienen pirámide con salto directo a 10x
(sin nivel intermedio a 20x). Slideflow lanzaba el error:

```text
ERROR Could not find magnification level matching 20x (closest: 10.0). Skipping.
```

**Causa raíz:** la pirámide del slide tiene niveles en factores de downsample
1×, 4×, 16× y 32× desde 40x — nunca 2× (que sería 20x).

**Solución:** especificar `tile_um` en **microns** en lugar de magnificación.
Con `tile_um: 128`, slideflow calcula el downsample exacto desde full resolution
sin depender de la existencia de un nivel nativo:

- Slide a 40x → 0.25 MPP
- 128 µm ÷ 0.25 µm/px = 512 px a full res → reducido a 256 px = equivalente 20x

```yaml
# Antes
tile_um:
  - 20x

# Después
tile_um:
  - 128  # 128 microns ≡ 20x para slides de 0.25 MPP (40x)
```

### Exclusión de `TCGA-A8-A06U` de `conf_test_small_annotations.csv`

**Problema:** este slide tiene un único nivel de pirámide y **ningún metadato
MPP ni objective-power**. Slideflow no puede determinar el downsample necesario
con ningún valor de `tile_um`:

```text
ERROR Could not detect microns-per-pixel for slide: TCGA-A8-A06U-...svs
```

**Causa raíz:** el archivo SVS carece de los campos `openslide.mpp-x` y
`openslide.objective-power`, posiblemente debido a un escáner no estándar o
exportación incompleta.

**Solución:** eliminar la fila del CSV de anotaciones. No hay forma de procesar
este slide con la API actual de slideflow.

**Estado final de slides:** 15 slides procesables (6 TCGA entrenamiento +
9 CPTAC test), ~53 400 tiles estimados.

---

## Parte VI — Error StopIteration en el DataLoader de entrenamiento MIL

Descubierto tras la primera ejecución completa de extracción de tiles y features.

### Causa raíz (dos factores combinados)

**Factor 1 — `batch_size: 16` mayor que el número de slides por fold**

El trainer de slideflow (`_lightning.py`, línea 895) crea el DataLoader de entrenamiento con
`drop_last=True` **hardcoded**. Con k=2 y 6 slides TCGA, cada fold tiene **~3 slides** en
entrenamiento. Como `floor(3 / 16) = 0` batches completos, el DataLoader queda vacío y
`next(iter(train_dl))` lanza `StopIteration`.

**Factor 2 — `balancing: category` con categorías vacías por fold**

Con 4 categorías (her2×1, normal×2, luma×2, basal×1) y 3 slides por fold, alguna categoría
puede tener **0 slides**. El método `dataset.balance(strategy='category')` calcula el peso de
cada muestra como `min_slides_in_any_category / slides_in_this_category`. Si el mínimo es 0,
todos los pesos son 0 → dataset efectivamente vacío.

**Síntoma observado:**

```text
File "_lightning.py", line 955, in _build_dataloaders
    sample = next(iter(train_dl))
StopIteration
INFO:root:Combinations successfully finished: 0
INFO:root:Benchmarking finished...
```

### Solución aplicada en `conf_test_small.yaml`

```yaml
# Antes
balancing: category
batch_size: 16

# Después
balancing: slide       # 'slide' pondera uniformemente; no depende del número de categorías por fold
batch_size: 2          # debe ser < n_slides_por_fold (≈3); con drop_last=True hardcoded
```

**Regla general:** con `split_technique: k-fold` y `k` folds, el número de slides de
entrenamiento por fold es `n_training_slides * (k-1) / k`. El `batch_size` debe ser menor que
ese valor cuando `drop_last=True` esté activo (que es siempre en el trainer de slideflow para
entrenamiento a nivel de slide).

---

## Resumen de incompatibilidades resueltas

| Componente | Versión/valor original | Problema | Solución aplicada |
| --- | --- | --- | --- |
| `torch` | 2.2.2 (no instalado) | Sin espacio en disco para descargar ~2 GB | Reutilizar de `clam_latest` vía `.pth` → 2.9.0+cu128 |
| `lifelines` | 0.28.0 | `scipy.integrate.trapz` eliminado en scipy ≥ 1.14 | Actualizar a 0.29.0 |
| `seaborn` | 0.11.2 | `matplotlib.cm.register_cmap` eliminado en matplotlib ≥ 3.7 | Actualizar a 0.13.2 |
| `fastcore` | sin fijar | API incompatible con `fastai 2.7.14` en versiones ≥ 1.6 | Fijar en 1.5.29 |
| `fastprogress` | 1.0.3 | Versión ≥ 1.1 requiere `fasthtml` (Python ≥ 3.11) | Mantener en 1.0.3 |
| `typer` | 0.9.4 | Versión ≥ 0.10 requiere `annotated_doc` | Mantener en 0.9.4 |
| `libvips` (sistema) | no instalada | Proceso se colgaba silenciosamente al abrir SVS | `sudo apt install libvips42 libvips-dev` |
| `tile_um` (config) | `20x` | Slides sin nivel piramidal nativo a 20x se descartan | Cambiar a `128` (microns) |
| `TCGA-A8-A06U` (slide) | en anotaciones | Sin MPP ni objective-power, no procesable | Excluir del CSV de anotaciones |
| `balancing` (config) | `category` | Categorías vacías en fold → pesos 0 → DataLoader vacío | Cambiar a `slide` |
| `batch_size` (config) | `16` | Mayor que n_slides/fold (≈3) con `drop_last=True` → DataLoader vacío | Reducir a `2` |

---

## Verificación completa del entorno

```bash
source pathbench_env/bin/activate

python3 -c "
import torch, fastai, optuna, lifelines, pathbench, pyvips
print('torch:',     torch.__version__, '| CUDA:', torch.cuda.is_available())
print('fastai:',    fastai.__version__)
print('optuna:',    optuna.__version__)
print('lifelines:', lifelines.__version__)
print('libvips:',   pyvips.version_string())
print('pathbench:  OK')
"
```

Salida esperada (tras instalar `libvips42`):

```
torch: 2.9.0+cu128 | CUDA: True
fastai: 2.7.14
optuna: 3.6.1
lifelines: 0.29.0
libvips: 8.12.1
pathbench:  OK
```

---

## Parte VII — Estado del entorno (2026-02-19) y restauración necesaria

### Estado actual detectado

Al revisar el entorno en esta fecha se encontró que `pathbench_env` está
**casi vacío** — solo contiene las herramientas base de pip:

```
Package    Version
---------- -------
packaging  26.0
pip        26.0.1
setuptools 59.6.0
versioneer 0.29
wheel      0.46.3
```

Las causas identificadas:

1. **El archivo `.pth` que redirigía a torch de `clam_latest` ha desaparecido.**
   El fichero `pathbench_env/lib/python3.10/site-packages/clam_torch.pth`
   fue el mecanismo documentado en Parte I para reutilizar PyTorch sin
   descargarlo; ya no existe.

2. **La ruta del entorno conda ha cambiado.**
   El entorno `clam_latest` estaba documentado en
   `/home/jorge/anaconda3/envs/clam_latest` pero anaconda fue sustituido por
   **miniforge3**. La ruta correcta ahora es:

   ```
   /home/jorge/miniforge3/envs/clam_latest
   ```

3. **La versión de torch en `clam_latest` también cambió.**
   Versión documentada: `2.9.0+cu128`
   Versión actual: `2.6.0+cu124`
   Python: 3.10.0 (compatible con `pathbench_env`)

### Pasos para restaurar el entorno

Ejecutar con el entorno virtual activo:

```bash
source /media/jorge/hd1/patologia_digital/software/PathBench-MIL/pathbench_env/bin/activate
```

**Paso 1 — Recrear el enlace `.pth` a torch (nueva ruta)**

```bash
echo "/home/jorge/miniforge3/envs/clam_latest/lib/python3.10/site-packages" \
  > pathbench_env/lib/python3.10/site-packages/clam_torch.pth
```

Verificar:

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Esperado: 2.6.0+cu124  True
```

**Paso 2 — Reinstalar todos los paquetes desde requirements.txt**

El método más fiable es instalar directamente desde `requirements.txt` y luego
aplicar las correcciones de versión necesarias:

```bash
# Instalar todos los paquetes (tarda varios minutos; incluye torch, gigapath, etc.)
pip install -r requirements.txt

# Correcciones de versión obligatorias (sobreescriben las del requirements.txt)
pip install "lifelines>=0.29.0,<0.30" "seaborn>=0.13" exceptiongroup --no-deps

# Eliminar torch del venv — se debe usar el de clam_latest via .pth
# (torch 2.2.2 de PyPI falla al cargar libnvJitLink.so.12 en CUDA 12.8)
pip uninstall torch torchvision triton -y
```

> **Nota importante sobre torch:** al instalar desde requirements.txt, pip
> descarga `torch==2.2.2` que falla con `libnvJitLink.so.12: not found` en
> CUDA 12.8. La solución es desinstalarlo inmediatamente después y dejar que
> el `.pth` exponga el `torch 2.6.0+cu124` de `clam_latest`, que sí funciona.

**Paso 3 — Reinstalar slideflow fork y pathbench** (usan versioneer, requieren
`--no-build-isolation`):

```bash
pip install /media/jorge/hd1/patologia_digital/software/PathBench-MIL/slideflow_fork \
  --no-deps --no-build-isolation
pip install /media/jorge/hd1/patologia_digital/software/PathBench-MIL \
  --no-deps --no-build-isolation
```

**Paso 4 — Verificar estado completo**

```bash
python3 -c "
import torch, fastai, optuna, lifelines, pyvips, pathbench, slideflow
print('torch:',     torch.__version__, '| CUDA:', torch.cuda.is_available())
print('fastai:',    fastai.__version__)
print('optuna:',    optuna.__version__)
print('lifelines:', lifelines.__version__)
print('pyvips:',    pyvips.__version__, '| libvips:', pyvips.version(0), pyvips.version(1), pyvips.version(2))
print('pathbench:  OK')
print('slideflow:', slideflow.__version__)
"
```

> **Nota:** en pyvips 2.2.2 la función es `pyvips.version(n)`, no
> `pyvips.version_string()` (que existía en pyvips 3.x).

Salida esperada:

```
torch: 2.6.0+cu124 | CUDA: True
fastai: 2.7.14
optuna: 3.6.1
lifelines: 0.29.0
pyvips: 2.2.2 | libvips: 8 12 1
pathbench:  OK
slideflow: 0+unknown
```

### Nota sobre compatibilidad torch 2.6.0 vs 2.9.0

La versión 2.6.0+cu124 de torch (disponible en `clam_latest`) es compatible
con pytorch-lightning 2.2.2 y con todos los feature extractors soportados.
La única diferencia respecto a la versión anterior (2.9.0+cu128) es el nivel
de CUDA: 12.4 vs 12.8 — ambos funcionan en el hardware RTX 3070 Ti.