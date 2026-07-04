"""
Verification script for the Experiment Management System.

Tests:
    1. All imports
    2. ExperimentManager — create/list/load experiments
    3. Experiment — all path properties
    4. ExperimentLogger — JSON + CSV writing
    5. MetricsCalculator — correct metric values
    6. MetricAccumulator — batch accumulation + compute
    7. HistoryLoader — read back written data
    8. ReportGenerator — Markdown report generation
    9. No circular imports
"""
import sys
import os
import tempfile
import shutil
import json
import csv

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np

# ===== 1. IMPORTS =====
print('=' * 60)
print('1. Verifying imports ...')
print('=' * 60)

from experiment import (
    ExperimentManager,
    Experiment,
    ExperimentLogger,
    MetricsCalculator,
    MetricAccumulator,
    HistoryLoader,
    ReportGenerator,
)
from experiment.logger import JsonLogger, CsvLogger
from experiment.visualization import (
    plot_training_curves,
    compare_runs,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
print('  All imports OK.\n')


# ===== Helper: minimal opt namespace =====
class MockOpt:
    def __init__(self, **kwargs):
        self.isTrain = True
        self.lr = 1e-3
        self.beta1 = 0.9
        self.optim = 'adam'
        self.name = 'test_experiment'
        self.gpu_ids = [0]
        self.niter = 10
        for k, v in kwargs.items():
            setattr(self, k, v)


tmp_dir = tempfile.mkdtemp()
try:

    # ===== 2. EXPERIMENT MANAGER =====
    print('=' * 60)
    print('2. Verifying ExperimentManager ...')
    print('=' * 60)

    manager = ExperimentManager(base_dir=os.path.join(tmp_dir, 'experiments'))
    opt = MockOpt()

    # Create experiment
    exp = manager.create('test_model', opt)
    assert isinstance(exp, Experiment)
    assert exp.name == 'test_model'
    assert os.path.isdir(exp.root_dir)
    assert os.path.isdir(exp.checkpoint_dir)
    assert os.path.isdir(exp.plot_dir)
    assert os.path.isdir(exp.report_dir)
    assert os.path.isdir(exp.log_dir)
    assert os.path.isfile(exp.opt_txt_path)
    assert os.path.isfile(exp.opt_json_path)
    print(f'  Created experiment: {exp}')

    # List experiments
    exps = manager.list_experiments()
    assert len(exps) == 1
    print(f'  Listed {len(exps)} experiment(s)')

    # Load experiment
    loaded = manager.load(exp.root_dir)
    assert loaded.name == exp.name
    assert loaded.root_dir == exp.root_dir
    print(f'  Loaded experiment: {loaded}')
    print()

    # ===== 3. EXPERIMENT PATHS =====
    print('=' * 60)
    print('3. Verifying Experiment path properties ...')
    print('=' * 60)

    assert exp.history_path.endswith('history.json')
    assert exp.metrics_csv_path.endswith('metrics.csv')
    assert exp.opt_txt_path.endswith('opt.txt')
    assert exp.opt_json_path.endswith('opt.json')
    assert exp.checkpoint_dir.endswith('checkpoints')
    assert exp.plot_dir.endswith('plots')
    assert exp.report_dir.endswith('reports')
    assert exp.log_dir.endswith('logs')
    print('  All path properties correct.')
    print()

    # ===== 4. EXPERIMENT LOGGER =====
    print('=' * 60)
    print('4. Verifying ExperimentLogger ...')
    print('=' * 60)

    logger = ExperimentLogger(exp)

    # Log training
    logger.log_epoch(epoch=1, train_loss=0.6931, lr=1e-3,
                     elapsed=15.2, global_step=100, num_batches=50)
    logger.log_epoch(epoch=2, train_loss=0.4521, lr=1e-3,
                     elapsed=14.8, global_step=200, num_batches=50)

    # Log validation
    logger.log_validation(epoch=1, val_loss=0.5832, accuracy=0.7500,
                          precision=0.7200, recall=0.7800, f1=0.7489,
                          roc_auc=0.8100, num_samples=100)
    logger.log_validation(epoch=2, val_loss=0.3921, accuracy=0.8500,
                          precision=0.8300, recall=0.8700, f1=0.8495,
                          roc_auc=0.9200, num_samples=100)

    logger.close()

    # Verify JSON
    assert os.path.isfile(exp.history_path)
    with open(exp.history_path) as f:
        history = json.load(f)
    assert len(history['train']) == 2
    assert len(history['validation']) == 2
    assert history['train'][0]['epoch'] == 1
    assert history['train'][0]['train_loss'] == 0.6931
    assert history['validation'][1]['accuracy'] == 0.8500
    print('  history.json verified OK.')

    # Verify CSV
    assert os.path.isfile(exp.metrics_csv_path)
    with open(exp.metrics_csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]['epoch'] == '1'
    assert float(rows[1]['accuracy']) == 0.8500
    print('  metrics.csv verified OK.')
    print()

    # ===== 5. METRICS CALCULATOR =====
    print('=' * 60)
    print('5. Verifying MetricsCalculator ...')
    print('=' * 60)

    calc = MetricsCalculator(threshold=0.5)

    # Perfect predictions
    preds_perfect = np.array([0.9, 0.1, 0.8, 0.2, 0.95, 0.05])
    labels_perfect = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    result = calc.compute(preds_perfect, labels_perfect)
    assert result['accuracy'] == 1.0, f"Expected 1.0, got {result['accuracy']}"
    assert result['precision'] == 1.0
    assert result['recall'] == 1.0
    assert result['f1'] == 1.0
    assert result['confusion_matrix'] == [[3, 0], [0, 3]]
    print('  Perfect predictions: all metrics correct.')

    # 50% correct
    preds_half = np.array([0.9, 0.9, 0.1, 0.1])
    labels_half = np.array([1.0, 0.0, 0.0, 1.0])
    result_half = calc.compute(preds_half, labels_half)
    assert result_half['accuracy'] == 0.5
    print(f"  50% accuracy test: {result_half['accuracy']:.4f} OK")

    # Edge case: empty
    result_empty = calc.compute(np.array([]), np.array([]))
    assert result_empty['accuracy'] == 0.0
    print('  Empty input edge case: OK')
    print()

    # ===== 6. METRIC ACCUMULATOR =====
    print('=' * 60)
    print('6. Verifying MetricAccumulator ...')
    print('=' * 60)

    acc = MetricAccumulator(threshold=0.5)

    # Simulate two batches
    acc.update(
        predictions=np.array([0.9, 0.1, 0.8]),
        labels=np.array([1.0, 0.0, 1.0]),
        loss=0.3,
    )
    acc.update(
        predictions=np.array([0.2, 0.95, 0.05]),
        labels=np.array([0.0, 1.0, 0.0]),
        loss=0.25,
    )

    result_acc = acc.compute()
    assert result_acc['accuracy'] == 1.0
    assert result_acc['num_samples'] == 6
    assert abs(result_acc['avg_loss'] - 0.275) < 1e-6
    print(f"  Accumulated accuracy: {result_acc['accuracy']:.4f}")
    print(f"  Num samples: {result_acc['num_samples']}")
    print(f"  Avg loss: {result_acc['avg_loss']:.4f}")

    # Reset
    acc.reset()
    result_reset = acc.compute()
    assert result_reset['num_samples'] == 0
    print('  Reset: OK')
    print()

    # ===== 7. HISTORY LOADER =====
    print('=' * 60)
    print('7. Verifying HistoryLoader ...')
    print('=' * 60)

    loader = HistoryLoader(exp.root_dir)

    # Load history
    loaded_hist = loader.load_history()
    assert len(loaded_hist['train']) == 2
    print('  load_history() OK')

    # Load CSV
    loaded_csv = loader.load_csv()
    assert len(loaded_csv) == 2
    print('  load_csv() OK')

    # Load opt
    loaded_opt = loader.load_opt()
    assert loaded_opt['lr'] == 1e-3
    print(f"  load_opt() OK — lr={loaded_opt['lr']}")

    # Helpers
    losses = loader.get_train_losses()
    assert len(losses) == 2
    assert losses[0] == 0.6931
    print(f'  get_train_losses() OK — {losses}')

    accs = loader.get_val_metrics('accuracy')
    assert len(accs) == 2
    assert accs[1] == 0.8500
    print(f'  get_val_metrics() OK — {accs}')

    epochs = loader.get_epochs()
    assert epochs == [1, 2]
    print(f'  get_epochs() OK — {epochs}')
    print()

    # ===== 8. REPORT GENERATOR =====
    print('=' * 60)
    print('8. Verifying ReportGenerator ...')
    print('=' * 60)

    report_gen = ReportGenerator(exp.root_dir)
    report_path = report_gen.generate()
    assert os.path.isfile(report_path)
    with open(report_path, 'r') as f:
        content = f.read()
    assert '# Experiment Report' in content
    assert 'Training Summary' in content
    assert 'Validation History' in content
    assert 'Best Metrics' in content
    print(f'  Report generated: {os.path.basename(report_path)}')
    print(f'  Report size: {len(content)} chars')
    print()

    # ===== 9. NO CIRCULAR IMPORTS =====
    print('=' * 60)
    print('9. Verifying no circular imports ...')
    print('=' * 60)

    import importlib
    mods_to_check = [
        'experiment',
        'experiment.manager',
        'experiment.experiment',
        'experiment.logger',
        'experiment.logger.logger',
        'experiment.logger.json_logger',
        'experiment.logger.csv_logger',
        'experiment.metrics',
        'experiment.metrics.metrics',
        'experiment.metrics.accumulators',
        'experiment.visualization',
        'experiment.visualization.plot_training',
        'experiment.visualization.compare_runs',
        'experiment.visualization.confusion_matrix',
        'experiment.visualization.roc_curve',
        'experiment.visualization.precision_recall',
        'experiment.utils',
        'experiment.utils.history_loader',
        'experiment.utils.report',
    ]
    for mod_name in mods_to_check:
        importlib.import_module(mod_name)
    print(f'  All {len(mods_to_check)} modules imported without circular issues.')
    print()

    # ===== DONE =====
    print('=' * 60)
    print('ALL VERIFICATIONS PASSED.')
    print('=' * 60)

finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
