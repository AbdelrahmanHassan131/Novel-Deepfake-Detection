"""
Model Evaluator Orchestrator.

Architecture-agnostic evaluation pipeline orchestrator that integrates:
- Checkpoint loading & model building (`CheckpointLoader`)
- Inference execution (`InferenceRunner`)
- Metrics calculation (`ClassificationMetrics`, `PerformanceProfiler`)
- Visualizations (ROC, PR, CM, t-SNE)
- Report generation (`EvaluationReportGenerator`)

Usage::

    from evaluation import Evaluator

    evaluator = Evaluator(
        checkpoint_path='experiments/model_best.pth',
        dataroot='dataset/val',
        output_dir='eval_output'
    )
    result = evaluator.run()
"""

import os
import torch
from .checkpoint_loader import CheckpointLoader
from .inference import InferenceRunner
from .metrics.classification import ClassificationMetrics
from .metrics.performance import PerformanceProfiler
from .reports.generator import EvaluationReportGenerator
from .visualization.roc import plot_evaluation_roc_curve
from .visualization.precision_recall import plot_evaluation_pr_curve
from .visualization.confusion_matrix import plot_evaluation_confusion_matrix
from .visualization.tsne import plot_tsne


class EvaluationResult:
    """Structured result container returned by Evaluator.run()."""

    def __init__(self, metadata, metrics, performance, inference_result,
                 plot_paths, report_paths):
        self.metadata = metadata
        self.metrics = metrics
        self.performance = performance
        self.inference_result = inference_result
        self.plot_paths = plot_paths
        self.report_paths = report_paths

    def __repr__(self):
        acc = self.metrics.get('accuracy', 0.0)
        auc = self.metrics.get('roc_auc', 0.0)
        return (
            f"EvaluationResult(arch='{self.metadata.get('arch')}', "
            f"accuracy={acc:.4f}, roc_auc={auc:.4f})"
        )


class Evaluator:
    """
    Architecture-agnostic model evaluator.

    Args:
        checkpoint_path (str): Path to the model `.pth` checkpoint file.
        dataroot (str): Root directory of the evaluation dataset.
        output_dir (str): Directory where plots and reports will be saved.
        arch (str, optional): Override architecture detection.
        batch_size (int): Batch size for DataLoader.
        device (str, optional): Target device (e.g., 'cuda:0' or 'cpu').
        collect_embeddings (bool): Whether to collect feature embeddings for t-SNE.
        run_profiling (bool): Whether to run performance profiling (FLOPs/latency).
        generate_plots (bool): Whether to generate ROC, PR, CM, and t-SNE plots.
    """

    def __init__(self, checkpoint_path, dataroot, output_dir='evaluation_results',
                 arch=None, batch_size=32, device=None, collect_embeddings=True,
                 run_profiling=True, generate_plots=True, generate_gradcam=True):
        self.checkpoint_path = checkpoint_path
        self.dataroot = dataroot
        self.output_dir = output_dir
        self.arch = arch
        self.batch_size = batch_size
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.collect_embeddings = collect_embeddings
        self.run_profiling = run_profiling
        self.generate_plots = generate_plots
        self.generate_gradcam = generate_gradcam

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, opt_overrides=None):
        """
        Execute the full evaluation pipeline.

        Args:
            opt_overrides (dict, optional): Custom options to override during load/dataloading.

        Returns:
            EvaluationResult: Complete evaluation findings, plots, and reports.
        """
        print(f"=== Starting Evaluation Pipeline ===")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Dataset:    {self.dataroot}")
        print(f"Output dir: {self.output_dir}")

        # 1. Load Checkpoint and Build Model
        loader = CheckpointLoader(
            checkpoint_path=self.checkpoint_path,
            arch=self.arch,
            device=self.device
        )
        overrides = {'dataroot': self.dataroot, 'batch_size': self.batch_size}
        if opt_overrides:
            overrides.update(opt_overrides)

        model = loader.load(opt_overrides=overrides)
        metadata = loader.metadata
        detected_arch = metadata.get('arch', 'unknown')

        # 2. Build Dataloader
        dataloader = self._build_dataloader(model.opt)

        # 3. Run Inference
        runner = InferenceRunner(
            model=model,
            dataloader=dataloader,
            device=self.device,
            collect_embeddings=self.collect_embeddings
        )
        inference_result = runner.run()

        # 4. Compute Classification Metrics
        print("Computing classification metrics...")
        metric_calc = ClassificationMetrics(threshold=0.5)
        metrics = metric_calc.compute_all(
            probabilities=inference_result.probabilities,
            labels=inference_result.labels
        )

        # 5. Performance Profiling
        performance = {}
        if self.run_profiling:
            print("Profiling model performance...")
            profiler = PerformanceProfiler(model=model, device=self.device)
            # Estimate input shape based on model type
            in_shape = (1, 3, 256, 256)
            if detected_arch in ('WolterWavelet2021Raw', 'WolterWavelet2021_128'):
                in_shape = (1, 192, 128, 128)
            elif detected_arch in ('Fusion_128', 'MHA_128'):
                in_shape = [(1, 128), (1, 128)]
            performance = profiler.profile(dataloader=dataloader, input_shape=in_shape)

        # 6. Generate Visualizations
        plot_paths = {}
        if self.generate_plots:
            print("Generating evaluation plots...")
            plots_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)

            plot_paths['roc_curve'] = plot_evaluation_roc_curve(
                probabilities=inference_result.probabilities,
                labels=inference_result.labels,
                save_path=os.path.join(plots_dir, 'roc_curve.png'),
                title=f'{detected_arch} ROC Curve'
            )

            plot_paths['precision_recall_curve'] = plot_evaluation_pr_curve(
                probabilities=inference_result.probabilities,
                labels=inference_result.labels,
                save_path=os.path.join(plots_dir, 'precision_recall_curve.png'),
                title=f'{detected_arch} PR Curve'
            )

            plot_paths['confusion_matrix'] = plot_evaluation_confusion_matrix(
                probabilities=inference_result.probabilities,
                labels=inference_result.labels,
                save_path=os.path.join(plots_dir, 'confusion_matrix.png'),
                title=f'{detected_arch} Confusion Matrix'
            )

            if self.collect_embeddings and inference_result.embeddings is not None:
                plot_paths['tsne'] = plot_tsne(
                    embeddings=inference_result.embeddings,
                    labels=inference_result.labels,
                    category_names=['Real (0)', 'Fake (1)'],
                    save_path=os.path.join(plots_dir, 'tsne_embeddings.png'),
                    title=f'{detected_arch} t-SNE Embeddings'
                )

            if self.generate_gradcam:
                gc_path = self._generate_gradcam_explanations(
                    model=model,
                    dataloader=dataloader,
                    detected_arch=detected_arch,
                    plots_dir=plots_dir
                )
                if gc_path:
                    plot_paths['gradcam'] = gc_path

        # 7. Generate Reports
        print("Generating reports...")
        report_gen = EvaluationReportGenerator(output_dir=self.output_dir)
        report_paths = report_gen.generate(
            metadata=metadata,
            metrics=metrics,
            performance=performance,
            plot_paths=plot_paths,
            report_name='evaluation_report'
        )

        print(f"=== Evaluation Complete ===")
        print(f"Accuracy: {metrics.get('accuracy', 0):.4f} | ROC AUC: {metrics.get('roc_auc', 0):.4f}")
        print(f"Markdown Report: {report_paths['markdown']}")

        return EvaluationResult(
            metadata=metadata,
            metrics=metrics,
            performance=performance,
            inference_result=inference_result,
            plot_paths=plot_paths,
            report_paths=report_paths
        )

    def _build_dataloader(self, opt):
        """Build the appropriate DataLoader for the model architecture."""
        from data import create_dataloader, create_mha_dataloader

        # Override with evaluation target settings
        opt.dataroot = self.dataroot
        opt.batch_size = self.batch_size
        opt.isTrain = False

        # Ensure opt has necessary dataset and transform defaults
        defaults = {
            'classes': ['0_real', '1_fake'],
            'mode': 'binary',
            'serial_batches': True,
            'class_bal': False,
            'num_threads': 0,
            'no_crop': False,
            'no_flip': True,
            'no_resize': False,
            'cropSize': 224,
            'loadSize': 256,
            'rz_interp': ['bilinear'],
            'blur_prob': 0.0,
            'blur_sig': [0.5],
            'jpg_prob': 0.0,
            'jpg_method': ['cv2'],
            'jpg_qual': [75],
            'data_aug': False,
            'wavelet_backend': 'cpu',
            'wavelet_type': 'haar',
            'level': 3,
            'precomputed_dir': None,
            'rgb_model_path': getattr(opt, 'rgb_model_path', ''),
            'wavelet_model_path': getattr(opt, 'wavelet_model_path', ''),
        }
        for k, v in defaults.items():
            if not hasattr(opt, k) or (k == 'classes' and not getattr(opt, k)):
                setattr(opt, k, v)

        # MHA/Fusion multi-input loaders
        arch = getattr(opt, 'arch', '')
        if arch in ('Fusion_128', 'MHA_128'):
            dataloader = create_mha_dataloader(opt)
        else:
            dataloader = create_dataloader(opt)

        self._normalize_dataset_label_mapping(dataloader.dataset)
        return dataloader

    def _normalize_dataset_label_mapping(self, dataset):
        """Ensure Real=0 and Fake=1 regardless of alphabetical folder sorting."""
        if hasattr(dataset, 'datasets'):
            for d in dataset.datasets:
                self._normalize_dataset_label_mapping(d)
            return

        if hasattr(dataset, 'class_to_idx'):
            idx_to_class = {v: str(k).lower() for k, v in dataset.class_to_idx.items()}
            # Check if index 0 represents Fake
            if 0 in idx_to_class and ('fake' in idx_to_class[0] or '1_fake' in idx_to_class[0]):
                print(f"[Evaluator] Remapping alphabetical dataset labels {dataset.class_to_idx} -> Real=0, Fake=1")
                old_transform = dataset.target_transform
                def remap_target(y, old_t=old_transform):
                    y_val = old_t(y) if old_t is not None else y
                    return 1 - y_val
                dataset.target_transform = remap_target

    def _generate_gradcam_explanations(self, model, dataloader, detected_arch, plots_dir):
        """Generate sample Grad-CAM heatmap visualization figure."""
        try:
            from .visualization.gradcam import GradCAM, generate_gradcam_figure
        except ImportError:
            return None

        raw_net = getattr(model, 'model', model)
        target_layer = None
        for m in raw_net.modules():
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m

        if target_layer is None:
            print(f"[Evaluator] No Conv2d layer found in {detected_arch} for Grad-CAM.")
            return None

        images = []
        cams = []
        labels = []
        predictions = []

        cam_engine = GradCAM(model, target_layer)
        try:
            for batch in dataloader:
                if len(images) >= 6:
                    break
                inputs, targets = batch[0], batch[1]
                for i in range(inputs.size(0)):
                    if len(images) >= 6:
                        break
                    inp = inputs[i:i+1].to(self.device)
                    lbl = int(targets[i].item())

                    img_tensor = inputs[i].cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_unnorm = torch.clamp(img_tensor * std + mean, 0, 1)
                    img_np = img_unnorm.permute(1, 2, 0).numpy()

                    heatmap, out_tensor = cam_engine.generate(inp)
                    prob = float(torch.sigmoid(out_tensor).squeeze().item())

                    images.append(img_np)
                    cams.append(heatmap)
                    labels.append(lbl)
                    predictions.append(prob)
        except Exception as e:
            print(f"[Evaluator] Note: Grad-CAM generation skipped ({e})")
        finally:
            cam_engine.remove_hooks()

        if not images:
            return None

        save_path = os.path.join(plots_dir, 'gradcam_explanations.png')
        generate_gradcam_figure(
            images=images,
            cams=cams,
            labels=labels,
            predictions=predictions,
            save_path=save_path,
            title=f"{detected_arch} Grad-CAM Explanations"
        )
        return save_path
