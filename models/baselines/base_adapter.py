"""Common base interface for external benchmark baseline adapters.

Provides a unified prediction and manifest evaluation interface without
instantiating models or downloading weights at module import time.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
import json


class BaseBaselineAdapter(ABC):
    """Abstract baseline detector adapter.

    Enforces:
      - Explicit paper citation, code license, and auxiliary data provenance.
      - Distinction between supplied-checkpoint evaluation and pilot retraining.
      - Native preprocessing specifications.
      - Lazy model construction only at runtime entry points.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        evaluation_mode: str = 'supplied_checkpoint',
        device: str = 'cpu',
    ):
        self.checkpoint_path = checkpoint_path
        self.evaluation_mode = evaluation_mode
        self.device = device
        self._model = None

        if evaluation_mode not in ('supplied_checkpoint', 'retrained_on_pilot'):
            raise ValueError(
                f"Unknown evaluation_mode: {evaluation_mode}. Must be 'supplied_checkpoint' or 'retrained_on_pilot'."
            )

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the baseline method."""

    @property
    @abstractmethod
    def paper_citation(self) -> str:
        """Full bibliographic citation."""

    @property
    @abstractmethod
    def code_license(self) -> str:
        """License of original code implementation."""

    @property
    @abstractmethod
    def auxiliary_data_provenance(self) -> str:
        """Description of auxiliary datasets or pretrained foundation models used."""

    @property
    @abstractmethod
    def native_preprocessing(self) -> Dict[str, Any]:
        """Dictionary of native image size, normalization, and crop policies."""

    @property
    def status(self) -> str:
        """Integration readiness status."""
        return 'ready_for_runtime'

    @abstractmethod
    def get_transforms(self):
        """Return native torchvision transform pipeline for this baseline."""

    @abstractmethod
    def load_model(self):
        """Instantiate and load model weights at runtime.

        Never called on import. Raises RuntimeError if dependencies or weights are absent.
        """

    def predict_manifest(
        self,
        manifest_path: str,
        dataroot: str,
        output_predictions_path: str,
        split: str = 'external_test',
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Execute inference over manifest records and export predictions.

        Conforms exactly to the repository's evaluation.generalization format.
        """
        from data.manifest import read_manifest, sha256
        from evaluation.generalization import export_predictions

        if self.status != 'ready_for_runtime':
            raise NotImplementedError(
                f"Baseline adapter {self.name} is an incomplete architectural specification stub in status '{self.status}'. "
                "Official weights, complete vendor preprocessing, and model execution are pending future external benchmark trials."
            )

        if not self.checkpoint_path or not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(
                f"Valid checkpoint required for baseline evaluation; got: {self.checkpoint_path}"
            )

        records = read_manifest(manifest_path, dataroot, split=split)
        ckpt_hash = sha256(self.checkpoint_path)

        # Runtime inference is deferred to the Colab phase
        raise NotImplementedError(
            "Baseline inference execution is deferred to the manual Colab runtime phase. "
            "Use run_baseline.py during that phase."
        )

    def metadata(self) -> Dict[str, Any]:
        """Return metadata dictionary for auditing."""
        return {
            'baseline_name': self.name,
            'paper_citation': self.paper_citation,
            'code_license': self.code_license,
            'auxiliary_data_provenance': self.auxiliary_data_provenance,
            'evaluation_mode': self.evaluation_mode,
            'native_preprocessing': self.native_preprocessing,
            'status': self.status,
            'checkpoint_path': self.checkpoint_path,
        }
