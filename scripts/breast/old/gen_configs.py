import os
from pathlib import Path

# Load HF_KEY from .env (KEY=VALUE format)
_env_file = Path(__file__).resolve().parent.parent.parent.parent / '.env'
for _line in _env_file.read_text().splitlines():
    if '=' in _line and not _line.startswith('#'):
        _k, _v = _line.split('=', 1)
        os.environ.setdefault(_k.strip(), _v.strip())
HF_KEY = os.environ['HF_KEY']

best = {
    ('pam50', 'transmil'):    {'z_dim': 472,  'dropout_p': 0.5974},
    ('pam50', 'clam_mil_mb'): {'z_dim': 255,  'dropout_p': 0.5489},
    ('pam50', 'dsmil'):       {'z_dim': 311,  'dropout_p': 0.6312},
    ('er',    'transmil'):    {'z_dim': 499,  'dropout_p': 0.5651},
    ('er',    'clam_mil_mb'): {'z_dim': 233,  'dropout_p': 0.7370},
    ('er',    'dsmil'):       {'z_dim':  57,  'dropout_p': 0.7470},
    ('erbb2', 'transmil'):    {'z_dim': 325,  'dropout_p': 0.5338},
    ('erbb2', 'clam_mil_mb'): {'z_dim': 317,  'dropout_p': 0.7555},
    ('erbb2', 'dsmil'):       {'z_dim': 121,  'dropout_p': 0.5140},
    ('pr',    'transmil'):    {'z_dim': 439,  'dropout_p': 0.6523},
    ('pr',    'clam_mil_mb'): {'z_dim':  33,  'dropout_p': 0.5429},
    ('pr',    'dsmil'):       {'z_dim': 329,  'dropout_p': 0.4753},
}

ihc_markers = {'er', 'pr', 'erbb2'}
config_dir = '/shared/home/jorgarcia/PathBench-MIL/config'

KFOLD_TEMPLATE = """\
experiment:
  project_name: brca_virchow2_kfold10_{task}_{mil}
  seed: 42
  annotation_file: ./config/annotations/annotations_brca_{task}.csv
  balancing: category
  bags_path: /home/PARADIS/mama/tfrecords/shared_bags
  class_weighting: True
  split_technique: k-fold
  val_fraction: 0.1
  k: 10
  best_epoch_based_on: train/loss
  epochs: 50
  batch_size: 128
  bag_size: 128
  encoder_layers: 1
  z_dim: {z_dim}
  dropout_p: {dropout_p}
  lr: 1.0e-4
  wd: 1.0e-4
  num_workers: 8
  aggregation_level: patient
  task: classification
  mode: benchmark
  custom_metrics: []
  report: True
  skip_extracted: True
  skip_feature_extraction: True
  save_heatmaps_test: False

  multiprocessing_context: fork
  persistent_workers: True
  pin_memory: True
  mixed_precision: True

  experiment_label: brca_{task}_virchow2_{mil}

  qc:
    - GaussianV2
    - Otsu-CLAHE

  qc_filters:
    grayspace_threshold: 0.05
    grayspace_fraction: 0.6
    whitespace_threshold: 230
    whitespace_fraction: 1.0

  evaluation:
    - mean_f1{extra_eval}

  visualization:
    - learning_curves
    - confusion_matrix

datasets:
  - name: tcga
    slide_path: /home/PARADIS/mama/wsi/TCGA-BRCA
    tfrecord_path: /home/PARADIS/mama/tfrecords/TCGA-BRCA
    tile_path: /home/PARADIS/mama/tfrecords/TCGA-BRCA/tiles
    used_for: training

benchmark_parameters:
  tile_px:
    - 256
  tile_um:
    - 128
  normalization:
    - none
  feature_extraction:
    - virchow2
  mil:
    - {mil}
  loss:
    - CrossEntropyLoss
  activation_function:
    - ReLU
  optimizer:
    - Adam

weights_dir: ./pretrained_weights
hf_key: {hf_key}
"""

TEST_TEMPLATE = """\
experiment:
  project_name: brca_virchow2_test_{task}_{mil}
  seed: 42
  annotation_file: ./config/annotations/annotations_brca_{task}.csv
  balancing: category
  bags_path: /home/PARADIS/mama/tfrecords/shared_bags
  class_weighting: False
  split_technique: fixed
  val_fraction: 0.15
  best_epoch_based_on: val/macro_f1
  epochs: 50
  batch_size: 128
  bag_size: 128
  encoder_layers: 1
  z_dim: {z_dim}
  dropout_p: {dropout_p}
  lr: 1.0e-4
  wd: 1.0e-4
  num_workers: 8
  aggregation_level: slide
  task: classification
  mode: benchmark
  custom_metrics: []
  report: True
  skip_extracted: True
  skip_feature_extraction: True
  save_heatmaps_test: False

  callbacks:
    early_stopping:
      monitor: val/macro_f1
      mode: max
      patience: 20
      min_delta: 0.0

  multiprocessing_context: fork
  persistent_workers: True
  pin_memory: True
  mixed_precision: True

  experiment_label: brca_{task}_virchow2_{mil}

  qc:
    - GaussianV2
    - Otsu-CLAHE

  qc_filters:
    grayspace_threshold: 0.05
    grayspace_fraction: 0.6
    whitespace_threshold: 230
    whitespace_fraction: 1.0

  evaluation:
    - mean_f1{extra_eval}

  visualization:
    - learning_curves
    - confusion_matrix

datasets:
  - name: tcga
    slide_path: /home/PARADIS/mama/wsi/TCGA-BRCA
    tfrecord_path: /home/PARADIS/mama/tfrecords/TCGA-BRCA
    tile_path: /home/PARADIS/mama/tfrecords/TCGA-BRCA/tiles
    used_for: training

  - name: cptac
    slide_path: /home/PARADIS/mama/wsi/CPTAC-BRCA
    tfrecord_path: /home/PARADIS/mama/tfrecords/CPTAC-BRCA
    tile_path: /home/PARADIS/mama/tfrecords/CPTAC-BRCA/tiles
    used_for: testing

benchmark_parameters:
  tile_px:
    - 256
  tile_um:
    - 128
  normalization:
    - none
  feature_extraction:
    - virchow2
  mil:
    - {mil}
  loss:
    - CrossEntropyLoss
  activation_function:
    - ReLU
  optimizer:
    - Adam

weights_dir: ./pretrained_weights
hf_key: {hf_key}
"""

created = []
for task in ['pam50', 'er', 'pr', 'erbb2']:
    for mil in ['transmil', 'clam_mil_mb', 'dsmil']:
        params = best[(task, mil)]
        extra_eval = '\n    - mean_average_precision' if task in ihc_markers else ''

        for mode, template in [('kfold10', KFOLD_TEMPLATE), ('test', TEST_TEMPLATE)]:
            content = template.format(
                task=task, mil=mil,
                z_dim=params['z_dim'],
                dropout_p=params['dropout_p'],
                extra_eval=extra_eval,
                hf_key=HF_KEY,
            )
            fname = 'conf_brca_%s_%s_virchow2_%s.yaml' % (task, mil, mode)
            fpath = os.path.join(config_dir, fname)
            with open(fpath, 'w') as f:
                f.write(content)
            created.append(fname)

for f in created:
    print(f)
print('\nTotal: %d files' % len(created))