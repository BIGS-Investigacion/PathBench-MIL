# PathBench-MIL — BIGS Research Fork

> **Fork mantenido por el grupo [PARADIS](https://github.com/BIGS-Investigacion) de la Universidad de Sevilla**
> Basado en el trabajo original (https://github.com/Sbrussee/PathBench-MIL) del Leiden University Medical Center.

---

## Descripción del proyecto

Este repositorio es un fork activo de **PathBench-MIL**, un framework de benchmarking y AutoML para *Multiple Instance Learning* (MIL) en histopatología computacional.

El objetivo principal de esta rama es **investigar y desarrollar nuevas técnicas de fusión de información** aplicadas a la patología computacional. Nos centramos en explorar cómo combinar de manera más efectiva la información procedente de distintas fuentes (extractores de características, escalas de magnificación, modalidades clínicas) para mejorar el rendimiento predictivo en tareas anatomopatológicas.

### Líneas de investigación

- Nuevos métodos de agregación MIL orientados a la fusión de información multi-fuente.
- Integración de información multi-escala y multi-modal en pipelines de patología computacional.
- Estrategias de fusión tardía, temprana e intermedia aplicadas a bolsas de características (*feature bags*).
- Evaluación sistemática mediante benchmarking reproducible sobre conjuntos de datos públicos y propios.

---

## Sobre PathBench-MIL

PathBench es un paquete Python diseñado para facilitar el benchmarking, la experimentación y la optimización de modelos de *Multiple Instance Learning* para histopatología computacional. Está construido sobre [SlideFlow](https://github.com/jamesdolezal/slideflow) para el procesamiento de *Whole Slide Images* (WSIs) e integra [Optuna](https://optuna.org/) para la optimización de hiperparámetros.

PathBench opera en dos modos:
- **Benchmark**: Prueba todas las combinaciones posibles de parámetros del pipeline y genera una tabla de rendimiento ordenada.
- **Optimization**: Busca el conjunto óptimo de hiperparámetros del pipeline para maximizar una métrica objetivo.

Soporta tareas de clasificación binaria/multiclase, regresión y predicción de supervivencia (continua/discreta).

### Referencia original

> **Brussee et al.** *PathBench-MIL: A Comprehensive AutoML and Benchmarking Framework for Multiple Instance Learning in Histopathology.* arXiv:2512.17517, 2025.

```bibtex
@article{brussee2025pathbench,
  title={PathBench-MIL: A Comprehensive AutoML and Benchmarking Framework for Multiple Instance Learning in Histopathology},
  author={Brussee, Siemen and Valkema, Pieter A and Weijer, Jurre AJ and Doeleman, Thom and Schrader, Anne MR and Kers, Jesper},
  journal={arXiv preprint arXiv:2512.17517},
  year={2025}
}
```

---

## Tabla de contenidos

- [Instalación](#instalación)
- [Configuración del proyecto](#configuración-del-proyecto)
- [Ejecución](#ejecución)
- [Características](#características)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Extender PathBench](#extender-pathbench)

---

## Instalación

### Requisitos previos

- Python >= 3.10
- Git

### Pasos

1. **Clonar el repositorio:**

    ```bash
    git clone --recurse-submodules https://github.com/BIGS-Investigacion/PathBench-MIL.git
    cd PathBench-MIL
    ```

2. **Crear el entorno virtual e instalar dependencias base:**

    ```bash
    python setup_pathbench.py
    ```

3. **Activar el entorno virtual:**

    ```bash
    source pathbench_env/bin/activate
    ```

4. **Instalar SlideFlow (fork):**

    ```bash
    cd slideflow_fork
    pip install -e .
    cd ..
    ```

5. **Instalar PathBench-MIL:**

    ```bash
    pip install -e .
    ```

---

## Configuración del proyecto

Toda la configuración se gestiona a través de un fichero `.yaml`. A continuación se muestra un ejemplo mínimo:

```yaml
experiment:
  project_name: Mi_Proyecto
  annotation_file: /ruta/al/annotation_file.csv
  task: classification          # classification | regression | survival | survival_discrete
  mode: benchmark               # benchmark | optimization
  split_technique: k-fold
  k: 5
  epochs: 10
  batch_size: 32
  bag_size: 512
  aggregation_level: slide      # slide | patient

datasets:
  - name: dataset1
    slide_path: /ruta/slides
    tfrecord_path: /ruta/tfrecords
    tile_path: /ruta/tiles
    used_for: training

benchmark_parameters:
  tile_px: [256]
  tile_um: [20x]
  normalization: [macenko]
  feature_extraction: [uni]
  mil: [Attention_MIL]

weights_dir: ./pretrained_weights
hf_key: YOUR_HUGGINGFACE_TOKEN
```

El fichero de anotaciones debe ser un CSV con columnas: `slide`, `patient`, `dataset` y la columna objetivo (`category`, `value`, `time`/`event`).

---

## Ejecución

```bash
./run_pathbench.sh
```

O directamente:

```bash
python3 main.py conf.yaml
```

### Inferencia sobre nuevas imágenes

```bash
python3 pathbench/utils/inference.py \
    --slide path/to/slide.tiff \
    --model_dir experiments/exp1/mil/best_model \
    --config conf.yaml \
    --heatmap
```

### Visualización interactiva de resultados

```bash
python3 pathbench/visualization/vis_app.py --results /ruta/a/resultados/
```

---

## Características

- Clasificación binaria/multiclase, regresión y predicción de supervivencia.
- Extracción de características multi-GPU a nivel de tile y de slide.
- Benchmarking de:
  - Tamaño de tile y magnificación
  - Métodos de normalización (Macenko, Reinhard, CycleGAN)
  - Extractores de características (UNI, GigaPath, Virchow, Hibou, Kaiko, etc.)
  - Agregadores MIL (CLAM, DSMIL, TransMIL, Attention MIL, etc.)
  - Funciones de pérdida y optimizadores
- Optimización AutoML con Optuna.
- Visualización interactiva con Plotly-Dash.

### Extractores de características disponibles

| Extractor | Acceso | Referencia |
|-----------|--------|-----------|
| ImageNet-ResNet50 | Automático | — |
| UNI | Gated (HF) | [Link](https://huggingface.co/MahmoodLab/UNI) |
| CONCH | Gated (HF) | [Link](https://huggingface.co/MahmoodLab/CONCH) |
| Prov-GigaPath | Gated (HF) | [Link](https://huggingface.co/prov-gigapath/prov-gigapath) |
| Virchow / Virchow2 | Gated (HF) | [Link](https://huggingface.co/paige-ai/Virchow) |
| Hibou-B | Automático | [Link](https://huggingface.co/histai/hibou-b) |
| H-Optimus-0 | Automático | [Link](https://huggingface.co/bioptimus/H-optimus-0) |
| Kaiko (S8/S16/B8/B16/L14) | Automático | [Link](https://github.com/kaiko-ai/towards_large_pathology_fms) |
| Phikon / Phikon-V2 | Automático | [Link](https://huggingface.co/owkin/phikon) |
| CTransPath | Automático | [Link](https://github.com/Xiyue-Wang/TransPath) |
| KEEP | Automático | [Link](https://huggingface.co/Astaxanthin/KEEP) |
| Midnight | Automático | [Link](https://huggingface.co/kaiko-ai/midnight) |

### Agregadores MIL disponibles

`Attention_MIL`, `clam_mil`, `clam_mil_mb`, `transmil`, `dsmil`, `varmil`, `perceiver_mil`, `deepset_mil`, `mean_mil`, `max_mil`, `lse_mil`, `lstm_mil`, `linear_mil`, `distributionpooling_mil`, `air_mil`, `topk_mil`, `weighted_mean_mil`, `gated_attention_mil`, `il_mil`, `bistro.transformer`

---

## Estructura del proyecto

```
PathBench-MIL/
├── pathbench/
│   ├── benchmarking/       # Lógica principal de benchmarking
│   ├── experiment/         # Inicialización de experimentos
│   ├── models/
│   │   ├── aggregators.py          # Métodos de agregación MIL
│   │   ├── feature_extractors.py   # Extractores de características
│   │   └── slide_level_predictors.py
│   ├── utils/
│   │   ├── utils.py
│   │   ├── losses.py
│   │   ├── metrics.py
│   │   └── inference.py
│   └── visualization/
│       └── visualization.py
├── slideflow_fork/         # Fork de SlideFlow
├── configs/                # Ficheros de configuración de ejemplo
├── experiments/            # Resultados de experimentos
├── pretrained_weights/     # Pesos de modelos pre-entrenados
├── main.py
├── run_pathbench.sh
└── README.md
```

---

## Extender PathBench

### Añadir un extractor de características

Añade la clase en [pathbench/models/feature_extractors.py](pathbench/models/feature_extractors.py) con el decorador `@register_torch`:

```python
@register_torch
class mi_extractor(TorchFeatureExtractor):
    tag = 'mi_extractor'

    def __init__(self, tile_px=256, **kwargs):
        super().__init__(**kwargs)
        self.model = ...           # Definir el modelo
        self.num_features = 768    # Dimensión del embedding
        self.transform = ...       # Pipeline de transformación
        self.preprocess_kwargs = {'standardize': False}

    def dump_config(self):
        return {'class': 'mi_extractor', 'kwargs': {}}
```

### Añadir un agregador MIL

Añade la clase en [pathbench/models/aggregators.py](pathbench/models/aggregators.py). El método `forward` recibe los bags `(batch, n_patches, n_feats)` y devuelve los scores de predicción.

### Añadir funciones de pérdida personalizadas

Añade la clase en [pathbench/utils/losses.py](pathbench/utils/losses.py) y especifícala en el campo `loss` de la configuración YAML.

---

## Licencia

PathBench-MIL se distribuye bajo la licencia **GPL-3.0**. Este fork mantiene la misma licencia. Las modificaciones son libres de uso y modificación, pero deben redistribuirse bajo los mismos términos.

---

## Contacto

- **Grupo PARADIS** — Universidad de Sevilla
- Repositorio: [https://github.com/BIGS-Investigacion/PathBench-MIL](https://github.com/BIGS-Investigacion/PathBench-MIL)
- Fork original: [https://github.com/Sbrussee/PathBench-MIL](https://github.com/Sbrussee/PathBench-MIL)
