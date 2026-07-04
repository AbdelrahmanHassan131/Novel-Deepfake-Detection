"""
plot_training — post-training visualisation of training curves.

Reads experiment history files and generates:
    - Loss curves (train and validation)
    - Accuracy curve
    - Learning rate schedule
    - Combined multi-metric overview

Usage::

    from experiment.visualization import plot_training_curves

    plot_training_curves('experiments/wang2020_progan_20260703_120000')
"""

import os

from experiment.utils.history_loader import HistoryLoader


def plot_training_curves(experiment_dir, save_dir=None, show=False):
    """
    Generate training curve plots from saved experiment data.

    Args:
        experiment_dir (str): Path to the experiment directory.
        save_dir (str, optional): Directory to save plots.
            Defaults to ``<experiment_dir>/plots/``.
        show (bool): If True, display plots interactively.

    Returns:
        list of str: Paths to generated plot files.
    """
    # Lazy import — matplotlib is only needed post-training
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    loader = HistoryLoader(experiment_dir)
    history = loader.load_history()

    if save_dir is None:
        save_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)

    saved_paths = []

    # --- Loss curves ---
    path = _plot_loss_curves(history, save_dir, show, plt)
    if path:
        saved_paths.append(path)

    # --- Accuracy curve ---
    path = _plot_accuracy_curve(history, save_dir, show, plt)
    if path:
        saved_paths.append(path)

    # --- Learning rate schedule ---
    path = _plot_lr_schedule(history, save_dir, show, plt)
    if path:
        saved_paths.append(path)

    # --- Combined overview ---
    path = _plot_overview(history, save_dir, show, plt)
    if path:
        saved_paths.append(path)

    return saved_paths


# ------------------------------------------------------------------
# Individual plot functions
# ------------------------------------------------------------------

def _plot_loss_curves(history, save_dir, show, plt):
    """Train and validation loss on the same axes."""
    train = history.get('train', [])
    val = history.get('validation', [])
    if not train:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    train_epochs = [r['epoch'] for r in train if 'train_loss' in r]
    train_losses = [r['train_loss'] for r in train if 'train_loss' in r]
    ax.plot(train_epochs, train_losses, 'b-o', label='Train Loss', markersize=3)

    if val:
        val_epochs = [r['epoch'] for r in val if 'val_loss' in r]
        val_losses = [r['val_loss'] for r in val if 'val_loss' in r]
        ax.plot(val_epochs, val_losses, 'r-s', label='Val Loss', markersize=3)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, 'loss_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path


def _plot_accuracy_curve(history, save_dir, show, plt):
    """Validation accuracy over epochs."""
    val = history.get('validation', [])
    if not val:
        return None

    epochs = [r['epoch'] for r in val if 'accuracy' in r]
    accs = [r['accuracy'] for r in val if 'accuracy' in r]
    if not epochs:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, accs, 'g-o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, 'accuracy_curve.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path


def _plot_lr_schedule(history, save_dir, show, plt):
    """Learning rate over epochs."""
    train = history.get('train', [])
    epochs = [r['epoch'] for r in train if 'lr' in r]
    lrs = [r['lr'] for r in train if 'lr' in r]
    if not epochs:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, lrs, 'k-o', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, 'lr_schedule.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path


def _plot_overview(history, save_dir, show, plt):
    """Combined 2x2 overview: loss, accuracy, F1, AUC."""
    train = history.get('train', [])
    val = history.get('validation', [])
    if not train or not val:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: loss
    ax = axes[0, 0]
    t_ep = [r['epoch'] for r in train if 'train_loss' in r]
    t_loss = [r['train_loss'] for r in train if 'train_loss' in r]
    ax.plot(t_ep, t_loss, 'b-', label='Train', linewidth=1)
    v_ep = [r['epoch'] for r in val if 'val_loss' in r]
    v_loss = [r['val_loss'] for r in val if 'val_loss' in r]
    ax.plot(v_ep, v_loss, 'r-', label='Val', linewidth=1)
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: accuracy
    ax = axes[0, 1]
    acc_ep = [r['epoch'] for r in val if 'accuracy' in r]
    accs = [r['accuracy'] for r in val if 'accuracy' in r]
    ax.plot(acc_ep, accs, 'g-', linewidth=1)
    ax.set_title('Accuracy')
    ax.grid(True, alpha=0.3)

    # Bottom-left: F1
    ax = axes[1, 0]
    f1_ep = [r['epoch'] for r in val if 'f1' in r]
    f1s = [r['f1'] for r in val if 'f1' in r]
    ax.plot(f1_ep, f1s, 'm-', linewidth=1)
    ax.set_title('F1 Score')
    ax.grid(True, alpha=0.3)

    # Bottom-right: AUC
    ax = axes[1, 1]
    auc_ep = [r['epoch'] for r in val if 'roc_auc' in r]
    aucs = [r['roc_auc'] for r in val if 'roc_auc' in r]
    ax.plot(auc_ep, aucs, 'c-', linewidth=1)
    ax.set_title('ROC AUC')
    ax.grid(True, alpha=0.3)

    fig.suptitle('Training Overview', fontsize=14)
    fig.tight_layout()

    path = os.path.join(save_dir, 'overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    return path
