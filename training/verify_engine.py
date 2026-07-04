"""
Verification script for the Training Engine.

Tests:
    1. All imports
    2. Trainer construction with a mock model
    3. Checkpoint save / load / resume
    4. DDP compatibility (static verification)
    5. No circular imports
"""
import sys
import os
import tempfile
import shutil

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn

# ===== 1. IMPORTS =====
print('=' * 60)
print('1. Verifying imports ...')
print('=' * 60)

from training import (
    Trainer, Validator, ValidationResult, CheckpointManager,
    build_optimizer, build_scheduler,
)
from training.base_trainer import BaseTrainer
from training.hooks import (
    CheckpointHook, ValidationHook, LoggerHook, SchedulerHook,
)
from models import build_model, BaseModel, init_weights
print('  All imports OK.\n')


# ===== Helper: minimal opt namespace =====
class MockOpt:
    """Minimal opt namespace that satisfies BaseModel and Trainer."""
    def __init__(self, **kwargs):
        self.isTrain = True
        self.continue_train = False
        self.gpu_ids = [0] if torch.cuda.is_available() else []
        self.checkpoints_dir = kwargs.get('checkpoints_dir', tempfile.mkdtemp())
        self.name = 'verify_test'
        self.lr = 1e-3
        self.beta1 = 0.9
        self.optim = 'adam'
        self.init_gain = 0.02
        self.lr_policy = 'none'
        self.niter = 5
        self.niter_decay = 0
        self.loss_freq = 10
        self.save_epoch_freq = 1
        self.val_epoch_freq = 1
        self.new_optim = False
        self.epoch = 'latest'
        for k, v in kwargs.items():
            setattr(self, k, v)


# ===== Helper: tiny model =====
class TinyModel(BaseModel):
    """Minimal BaseModel subclass for testing."""

    def name(self):
        return 'TinyTest'

    def __init__(self, opt):
        super().__init__(opt)
        self.model = nn.Linear(4, 1)
        self.model.to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()


# ===== Helper: fake dataloader =====
def make_fake_loader(n_batches=3, batch_size=4):
    data = []
    for _ in range(n_batches):
        x = torch.randn(batch_size, 4)
        y = torch.randint(0, 2, (batch_size,))
        data.append((x, y))
    return data


# ===== 2. TRAINER CONSTRUCTION =====
print('=' * 60)
print('2. Verifying Trainer construction ...')
print('=' * 60)

tmp_dir = tempfile.mkdtemp()
try:
    opt = MockOpt(checkpoints_dir=tmp_dir)
    model = TinyModel(opt)
    train_loader = make_fake_loader()
    val_loader = make_fake_loader(n_batches=2)

    trainer = Trainer(model, train_loader, opt, val_loader=val_loader)
    assert trainer.model is model
    assert trainer.optimizer is model.optimizer
    assert trainer.scheduler is None  # lr_policy='none'
    assert trainer.current_epoch == 0
    assert trainer.global_step == 0
    assert len(trainer._hooks) == 3  # Logger, Validation, Checkpoint (no scheduler)
    print('  Trainer constructed OK.')
    print(f'  Hooks registered: {len(trainer._hooks)}')
    print()

    # ===== 2b. fit() =====
    print('  Running fit(num_epochs=2) ...')
    result = trainer.fit(num_epochs=2)
    assert trainer.current_epoch == 2
    assert trainer.global_step == 2 * len(train_loader)
    print(f'  fit() OK — epoch={trainer.current_epoch}, '
          f'global_step={trainer.global_step}, '
          f'best_metric={trainer.best_metric}')
    print()

    # ===== 3. CHECKPOINT SAVE / LOAD / RESUME =====
    print('=' * 60)
    print('3. Verifying checkpoint save / load / resume ...')
    print('=' * 60)

    # Save
    trainer.save_checkpoint()
    save_dir = os.path.join(tmp_dir, opt.name)
    last_path = os.path.join(save_dir, 'last.pth')
    assert os.path.isfile(last_path), f'last.pth not found at {last_path}'
    print(f'  save_checkpoint() -> {last_path} OK')

    # Verify checkpoint content
    ckpt = torch.load(last_path, map_location='cpu')
    assert 'model_state_dict' in ckpt
    assert 'optimizer_state_dict' in ckpt
    assert 'epoch' in ckpt
    assert 'best_metric' in ckpt
    assert 'global_step' in ckpt
    assert 'model_name' in ckpt
    print(f'  Checkpoint keys: {list(ckpt.keys())}')
    print(f'  Checkpoint epoch: {ckpt["epoch"]}, model_name: {ckpt["model_name"]}')

    # Resume
    opt2 = MockOpt(checkpoints_dir=tmp_dir)
    model2 = TinyModel(opt2)
    trainer2 = Trainer(model2, train_loader, opt2, val_loader=val_loader)
    assert trainer2.current_epoch == 0

    trainer2.resume_training(filepath=last_path)
    assert trainer2.current_epoch == ckpt['epoch']
    assert trainer2.global_step == ckpt['global_step']
    print(f'  resume_training() OK — epoch={trainer2.current_epoch}, '
          f'global_step={trainer2.global_step}')

    # latest_checkpoint()
    latest = trainer.checkpoint_manager.latest_checkpoint()
    assert latest is not None
    print(f'  latest_checkpoint() -> {os.path.basename(latest)} OK')
    print()

    # ===== 4. DDP COMPATIBILITY (static) =====
    print('=' * 60)
    print('4. DDP compatibility (static verification) ...')
    print('=' * 60)

    # a. No global state in BaseTrainer
    import inspect
    src = inspect.getsource(BaseTrainer)
    # Check for actual Python `global` statements, not the word in docstrings
    import ast
    tree = ast.parse(src)
    global_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Global)]
    assert len(global_nodes) == 0, 'BaseTrainer uses global statement!'
    print('  No global state in BaseTrainer: OK')

    # b. runtime is accepted and rank is propagated
    from training.runtime import DistributedRuntime
    opt_r1 = MockOpt(checkpoints_dir=tmp_dir)
    runtime_r1 = DistributedRuntime(opt_r1)
    # Manually set rank for testing (in real usage, torchrun sets env vars)
    runtime_r1.rank = 1
    runtime_r1.is_main = False
    trainer_rank1 = Trainer(TinyModel(opt_r1), train_loader, opt_r1,
                            runtime=runtime_r1)
    assert trainer_rank1.rank == 1
    assert trainer_rank1.checkpoint_manager.rank == 1
    print('  rank parameter propagation: OK')

    # c. sampler.set_epoch support
    class FakeSampler:
        def __init__(self):
            self.epoch = None
        def set_epoch(self, e):
            self.epoch = e

    class FakeLoader:
        def __init__(self):
            self.sampler = FakeSampler()
        def __iter__(self):
            return iter([])

    fl = FakeLoader()
    opt_ddp = MockOpt(checkpoints_dir=tmp_dir, niter=1, niter_decay=0)
    tiny = TinyModel(opt_ddp)
    t = Trainer(tiny, fl, opt_ddp)
    t.fit(num_epochs=1)
    assert fl.sampler.epoch == 1, f'sampler.set_epoch not called (got {fl.sampler.epoch})'
    print('  sampler.set_epoch(epoch) called: OK')

    # d. Only rank 0 saves checkpoints
    opt_r1 = MockOpt(checkpoints_dir=tmp_dir, name='rank1_test')
    tiny_r1 = TinyModel(opt_r1)
    cm = CheckpointManager(
        save_dir=os.path.join(tmp_dir, 'rank1_test'), model=tiny_r1, rank=1
    )
    cm.save_last(epoch=1, best_metric=0.5, global_step=10)
    r1_path = os.path.join(tmp_dir, 'rank1_test', 'last.pth')
    assert not os.path.isfile(r1_path), 'rank=1 should NOT write checkpoints!'
    print('  rank=1 checkpoint suppression: OK')

    print()

    # ===== 5. NO CIRCULAR IMPORTS =====
    print('=' * 60)
    print('5. Verifying no circular imports ...')
    print('=' * 60)
    # If we got here, all imports succeeded without hanging.
    # Double-check by reimporting in isolation.
    import importlib
    mods_to_check = [
        'training',
        'training.base_trainer',
        'training.trainer',
        'training.validator',
        'training.checkpoint_manager',
        'training.optimizer_factory',
        'training.scheduler_factory',
        'training.hooks',
        'training.hooks.checkpoint_hook',
        'training.hooks.validation_hook',
        'training.hooks.logger_hook',
        'training.hooks.scheduler_hook',
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
