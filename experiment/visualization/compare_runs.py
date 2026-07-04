"""
compare_runs — post-training comparison of multiple experiments.

Loads history from several experiment directories and plots them
side-by-side for easy comparison of loss, accuracy, F1, and AUC.

Usage::

    from Refactored.experiment.visualization import compare_runs

    compare_runs([
        'experiments/wang2020_progan_20260703_120000',
        'experiments/wang2020_progan_20260704_090000',
    ])
"""

import os

from Refactored.experiment.utils.history_loader import HistoryLoader


def compare_runs(experiment_dirs, save_path=None, show=False, labels=None):
    """
    Compare validation metrics across multiple experiments.

    Args:
        experiment_dirs (list of str): Paths to experiment directories.
        save_path (str, optional): File path to save the comparison
            plot.  Defaults to ``comparison.png`` in the first
            experiment's ``plots/`` directory.
        show (bool): If True, display the plot interactively.
        labels (list of str, optional): Legend labels for each run.
            Defaults to directory basenames.

    Returns:
        str: Path to the saved comparison plot.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [os.path.basename(d) for d in experiment_dirs]

    histories = []
    for d in experiment_dirs:
        loader = HistoryLoader(d)
        try:
            histories.append(loader.load_history())
        except FileNotFoundError:
            histories.append({'train': [], 'validation': []})

    # Metrics to compare
    metrics = [
        ('val_loss', 'Validation Loss'),
        ('accuracy', 'Accuracy'),
        ('f1', 'F1 Score'),
        ('roc_auc', 'ROC AUC'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10.colors

    for idx, (key, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for run_idx, (hist, label) in enumerate(zip(histories, labels)):
            val = hist.get('validation', [])
            epochs = [r['epoch'] for r in val if key in r]
            values = [r[key] for r in val if key in r]
            if epochs:
                color = colors[run_idx % len(colors)]
                ax.plot(epochs, values, '-o', label=label,
                        color=color, markersize=3, linewidth=1)

        ax.set_xlabel('Epoch')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment Comparison', fontsize=14)
    fig.tight_layout()

    if save_path is None:
        save_dir = os.path.join(experiment_dirs[0], 'plots')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'comparison.png')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)

    return save_path
