"""
Grad-CAM Visualization.

Refactored from ``MyModels/Grad_CAM.py`` into a clean, reusable module.

Provides ``GradCAM`` and ``GradCAMPlusPlus`` implementations that
work with any ``nn.Module`` and target layer.

Usage::

    from evaluation.visualization import GradCAM

    cam = GradCAM(model, target_layer)
    heatmap, output = cam.generate(input_tensor)
    cam.remove_hooks()
"""

import os
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization.

    Args:
        model (nn.Module): The neural network.
        target_layer (nn.Module): The convolutional layer to visualize.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input tensor to the model.
            target_class: Target class index (None for binary).

        Returns:
            cam (np.ndarray): Normalized heatmap (H, W).
            output (torch.Tensor): Model output (detached, CPU).
        """
        self.model.eval()

        output = self.model(input_tensor)

        if target_class is None:
            target = output.squeeze()
        else:
            target = output[0, target_class]

        self.model.zero_grad()
        target.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        # Weighted combination
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, output.detach().cpu()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++: Improved Visual Explanations for Deep CNNs.
    """

    def generate(self, input_tensor, target_class=None):
        self.model.eval()

        output = self.model(input_tensor)

        if target_class is None:
            target = output.squeeze()
        else:
            target = output[0, target_class]

        self.model.zero_grad()
        target.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Grad-CAM++ weighting
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
        alpha = alpha_num / alpha_denom

        weights = torch.sum(alpha * F.relu(gradients),
                            dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, output.detach().cpu()


def overlay_cam_on_image(image, cam, alpha=0.5):
    """
    Overlay a Grad-CAM heatmap on an image.

    Args:
        image (np.ndarray): Original image, shape ``(H, W, 3)``,
            values in [0, 255] or [0, 1].
        cam (np.ndarray): CAM heatmap, shape ``(Hc, Wc)``.
        alpha (float): Blending factor.

    Returns:
        blended (np.ndarray): Blended image, values in [0, 1].
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            'cv2 (opencv-python) is required for Grad-CAM overlay. '
            'Install with: pip install opencv-python'
        )

    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized),
                                cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    img_float = image.astype(np.float64)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    blended = alpha * heatmap + (1 - alpha) * img_float
    blended = np.clip(blended, 0, 1)

    return blended


def generate_gradcam_figure(images, cams, labels, predictions,
                            save_path=None, show=False,
                            title='Grad-CAM Visualization'):
    """
    Generate a figure showing original images alongside their
    Grad-CAM heatmaps.

    Args:
        images (list[np.ndarray]): Original images, each (H, W, 3).
        cams (list[np.ndarray]): CAM heatmaps, each (Hc, Wc).
        labels (list[int]): Ground-truth labels.
        predictions (list[float]): Model prediction probabilities.
        save_path (str, optional): File path to save.
        show (bool): If True, display interactively.
        title (str): Plot title.

    Returns:
        str or None: Path to the saved plot.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        # Original image
        axes[i, 0].imshow(images[i])
        label_text = 'REAL' if labels[i] == 1 else 'FAKE'
        axes[i, 0].set_title(f'Original ({label_text})', fontsize=11)
        axes[i, 0].axis('off')

        # Grad-CAM overlay
        overlay = overlay_cam_on_image(images[i], cams[i])
        axes[i, 1].imshow(overlay)
        prob = predictions[i]
        pred_text = 'REAL' if prob > 0.5 else 'FAKE'
        axes[i, 1].set_title(
            f'Grad-CAM (pred: {prob:.3f} → {pred_text})', fontsize=11)
        axes[i, 1].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    result_path = None
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        result_path = save_path

    if show:
        plt.show()
    plt.close(fig)

    return result_path
