"""
ExperimentManager — creates and manages experiment directories.

Responsible for:
    - Creating experiment folders with a standard directory structure.
    - Assigning unique experiment IDs (timestamp-based).
    - Exposing paths to all experiment files so no other module
      constructs paths manually.

Usage::

    from experiment import ExperimentManager

    manager = ExperimentManager(base_dir='experiments')
    experiment = manager.create('wang2020_progan', opt)
    print(experiment.history_path)
    print(experiment.checkpoint_dir)

Directory layout created per experiment::

    <base_dir>/
        <experiment_name>_<id>/
            opt.txt              # Serialised options
            history.json         # Complete experiment history
            metrics.csv          # Tabular metrics for spreadsheet use
            checkpoints/         # Model checkpoints
            reports/             # Post-training analysis reports
            plots/               # Post-training visualisations
            logs/                # Structured log files
"""

import os
import json
from datetime import datetime

from .experiment import Experiment


class ExperimentManager:
    """
    Factory for :class:`Experiment` instances.

    Each call to :meth:`create` produces a new, uniquely-identified
    experiment directory with the standard layout.

    Args:
        base_dir (str): Root directory under which all experiments
            are created.  Will be created if it does not exist.
    """

    def __init__(self, base_dir='experiments'):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, experiment_name, opt, run_id=None, allow_existing=False):
        """
        Create a new experiment.

        Args:
            experiment_name (str): Human-readable experiment name
                (e.g. ``'wang2020_progan'``).
            opt: The project options namespace.  Serialised into
                ``opt.txt`` inside the experiment directory.
            run_id (str, optional): Explicit immutable run/experiment identifier.
            allow_existing (bool): Allow existing directory without error.

        Returns:
            An :class:`Experiment` instance with all paths set.
        """
        experiment_id = (
            run_id
            or getattr(opt, 'run_id', None)
            or getattr(opt, 'experiment_id', None)
            or self._generate_id()
        )
        is_dist = False
        rank = 0
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                is_dist = True
                rank = dist.get_rank()
                obj_list = [experiment_id]
                dist.broadcast_object_list(obj_list, src=0)
                experiment_id = obj_list[0]
        except Exception:
            is_dist = False
            rank = 0

        dir_name = f'{experiment_name}_{experiment_id}'
        experiment_dir = os.path.join(self.base_dir, dir_name)
        is_continue = getattr(opt, 'continue_train', False) or (getattr(opt, 'resume_checkpoint', None) is not None)

        if is_dist:
            import torch.distributed as dist
            if rank == 0:
                decision = {'status': 'OK', 'msg': None, 'dir': experiment_dir}
                if os.path.isdir(experiment_dir) and not allow_existing and not is_continue:
                    decision['status'] = 'COLLISION'
                    decision['msg'] = (
                        f"Experiment directory '{experiment_dir}' already exists. "
                        "Running a fresh model in an existing directory would overwrite checkpoints. "
                        "Specify a distinct --run_id or supply --resume_checkpoint to resume."
                    )
                else:
                    try:
                        self._create_structure(experiment_dir)
                        self._save_opt(experiment_dir, opt)
                        manifest_info = {
                            'experiment_name': experiment_name,
                            'experiment_id': str(experiment_id),
                            'experiment_dir': str(os.path.abspath(experiment_dir)),
                            'checkpoint_dir': str(os.path.abspath(os.path.join(experiment_dir, 'checkpoints'))),
                            'best_checkpoint': str(os.path.abspath(os.path.join(experiment_dir, 'checkpoints', 'best.pth'))),
                            'last_checkpoint': str(os.path.abspath(os.path.join(experiment_dir, 'checkpoints', 'last.pth'))),
                            'latest_checkpoint': str(os.path.abspath(os.path.join(experiment_dir, 'checkpoints', 'last.pth'))),
                            'status': 'initialized',
                            'created_at': datetime.now().isoformat(),
                        }
                        try:
                            with open(os.path.join(experiment_dir, 'run_manifest.json'), 'w', encoding='utf-8') as f:
                                json.dump(manifest_info, f, indent=2)
                            with open(os.path.join(self.base_dir, 'latest_run.json'), 'w', encoding='utf-8') as f:
                                json.dump(manifest_info, f, indent=2)
                        except Exception:
                            pass
                    except Exception as exc:
                        decision['status'] = 'ERROR'
                        decision['msg'] = f"Failed to initialize experiment directory on rank 0: {exc}"

                decision_list = [decision]
                dist.broadcast_object_list(decision_list, src=0)
            else:
                decision_list = [None]
                dist.broadcast_object_list(decision_list, src=0)
                decision = decision_list[0]

            # Synchronize all ranks to ensure directory setup is complete before proceeding
            dist.barrier()

            if decision['status'] == 'COLLISION':
                raise FileExistsError(decision['msg'])
            elif decision['status'] == 'ERROR':
                raise RuntimeError(decision['msg'])

            experiment_dir = decision['dir']
        else:
            if os.path.isdir(experiment_dir) and not allow_existing and not is_continue:
                raise FileExistsError(
                    f"Experiment directory '{experiment_dir}' already exists. "
                    "Running a fresh model in an existing directory would overwrite checkpoints. "
                    "Specify a distinct --run_id or supply --resume_checkpoint to resume."
                )

            self._create_structure(experiment_dir)
            self._save_opt(experiment_dir, opt)
            manifest_info = {
                'experiment_name': experiment_name,
                'experiment_id': str(experiment_id),
                'experiment_dir': str(os.path.abspath(experiment_dir)),
                'checkpoint_dir': str(os.path.abspath(os.path.join(experiment_dir, 'checkpoints'))),
                'best_checkpoint': str(os.path.abspath(os.path.join(experiment_dir, 'checkpoints', 'best.pth'))),
                'last_checkpoint': str(os.path.abspath(os.path.join(experiment_dir, 'checkpoints', 'last.pth'))),
                'latest_checkpoint': str(os.path.abspath(os.path.join(experiment_dir, 'checkpoints', 'last.pth'))),
                'status': 'initialized',
                'created_at': datetime.now().isoformat(),
            }
            try:
                with open(os.path.join(experiment_dir, 'run_manifest.json'), 'w', encoding='utf-8') as f:
                    json.dump(manifest_info, f, indent=2)
                with open(os.path.join(self.base_dir, 'latest_run.json'), 'w', encoding='utf-8') as f:
                    json.dump(manifest_info, f, indent=2)
            except Exception:
                pass

        return Experiment(
            name=experiment_name,
            experiment_id=experiment_id,
            root_dir=experiment_dir,
        )

    def load(self, experiment_dir):
        """
        Load an existing experiment from its directory.

        Args:
            experiment_dir (str): Path to an existing experiment
                directory.

        Returns:
            An :class:`Experiment` instance.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not os.path.isdir(experiment_dir):
            raise FileNotFoundError(
                f'Experiment directory not found: {experiment_dir}'
            )

        dir_name = os.path.basename(experiment_dir)

        # Match either timestamp suffix YYYYMMDD_HHMMSS or any custom run identifier
        import re
        m = re.search(r'^(.+)_(\d{8}_\d{6})$', dir_name)
        if not m:
            m = re.search(r'^(.+)_([^_]+)$', dir_name)
        if m:
            name, experiment_id = m.group(1), m.group(2)
        else:
            name = dir_name
            experiment_id = 'unknown'

        return Experiment(
            name=name,
            experiment_id=experiment_id,
            root_dir=experiment_dir,
        )


    def list_experiments(self):
        """
        List all experiment directories under ``base_dir``.

        Returns:
            list of absolute directory paths.
        """
        if not os.path.isdir(self.base_dir):
            return []
        entries = []
        for entry in sorted(os.listdir(self.base_dir)):
            full = os.path.join(self.base_dir, entry)
            if os.path.isdir(full):
                entries.append(full)
        return entries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_id():
        """Timestamp-based experiment ID: ``YYYYMMDD_HHMMSS``."""
        return datetime.now().strftime('%Y%m%d_%H%M%S')

    @staticmethod
    def _create_structure(experiment_dir):
        """Create the standard subdirectory tree."""
        subdirs = ['checkpoints', 'reports', 'plots', 'logs']
        for sub in subdirs:
            os.makedirs(os.path.join(experiment_dir, sub), exist_ok=True)

    @staticmethod
    def _save_opt(experiment_dir, opt):
        """
        Persist the options namespace as both human-readable text
        and machine-readable JSON.
        """
        opt_dict = vars(opt) if hasattr(opt, '__dict__') else {}

        # Human-readable
        txt_path = os.path.join(experiment_dir, 'opt.txt')
        with open(txt_path, 'w') as f:
            f.write('# Experiment Options\n')
            f.write(f'# Created: {datetime.now().isoformat()}\n\n')
            for key in sorted(opt_dict.keys()):
                f.write(f'{key}: {opt_dict[key]}\n')

        # Machine-readable (JSON-safe serialisation)
        json_path = os.path.join(experiment_dir, 'opt.json')
        safe_dict = {}
        for k, v in opt_dict.items():
            try:
                json.dumps(v)
                safe_dict[k] = v
            except (TypeError, ValueError):
                safe_dict[k] = str(v)

        with open(json_path, 'w') as f:
            json.dump(safe_dict, f, indent=2)
