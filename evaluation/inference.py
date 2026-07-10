"""
Inference Runner.

Runs forward passes on a model and collects results (logits,
probabilities, predictions, labels, and optionally feature embeddings).

Completely model-agnostic — works with any ``BaseModel`` subclass
that exposes ``set_input``, ``forward``, ``output``, ``label``.

Usage::

    runner = InferenceRunner(model, dataloader, device)
    results = runner.run()
    print(results['probabilities'].shape)
"""

import time
import numpy as np
import torch
from tqdm import tqdm


class InferenceResult:
    """Container for inference outputs."""

    def __init__(self, logits, probabilities, predictions, labels,
                 embeddings=None):
        self.logits = logits
        self.probabilities = probabilities
        self.predictions = predictions
        self.labels = labels
        self.embeddings = embeddings

    def __repr__(self):
        return (
            f'InferenceResult(samples={len(self.labels)}, '
            f'has_embeddings={self.embeddings is not None})'
        )


class InferenceRunner:
    """
    Model-agnostic inference runner.

    Handles the full forward-pass loop, collecting all outputs needed
    for metrics and visualizations.

    Args:
        model: A ``BaseModel`` subclass instance.
        dataloader: A PyTorch DataLoader.
        device: The torch device.
        collect_embeddings (bool): Whether to extract intermediate
            embeddings (for t-SNE). Requires the model to have an
            ``embedding`` attribute after forward.
    """

    def __init__(self, model, dataloader, device=None,
                 collect_embeddings=False):
        self.model = model
        self.dataloader = dataloader
        self.device = device or model.device
        self.collect_embeddings = collect_embeddings

    @torch.no_grad()
    def run(self):
        """
        Run inference over the entire dataloader.

        Returns:
            An ``InferenceResult`` instance.
        """
        self.model.eval()

        all_logits = []
        all_probs = []
        all_preds = []
        all_labels = []
        all_embeddings = [] if self.collect_embeddings else None

        for batch in tqdm(self.dataloader, desc='Inference'):
            self.model.set_input(batch)
            self.model.forward()

            # Extract output logits
            output = self.model.output
            if output.dim() > 1:
                output = output.squeeze(1)

            logits = output.cpu().numpy()
            probs = torch.sigmoid(output).cpu().numpy()
            preds = (probs >= 0.5).astype(np.float64)
            labels = self.model.label.cpu().numpy()

            all_logits.append(logits)
            all_probs.append(probs)
            all_preds.append(preds)
            all_labels.append(labels)

            # Collect embeddings if requested and available
            if self.collect_embeddings and hasattr(self.model, 'embedding'):
                embed = self.model.embedding
                if embed is not None:
                    all_embeddings.append(embed.cpu().numpy())

        result = InferenceResult(
            logits=np.concatenate(all_logits),
            probabilities=np.concatenate(all_probs),
            predictions=np.concatenate(all_preds),
            labels=np.concatenate(all_labels),
            embeddings=(np.vstack(all_embeddings)
                        if all_embeddings else None),
        )

        print(f'[InferenceRunner] Processed {len(result.labels)} samples')
        return result
