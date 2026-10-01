from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    s = p.read_text(encoding='utf-8')
    if old not in s:
        raise RuntimeError(f'Missing {path}: {old[:50]}')
    p.write_text(s.replace(old, new), encoding='utf-8')


p = Path('evaluation/evaluator.py')
s = p.read_text(encoding='utf-8')
start, end = s.index('        # 5. Performance Profiling'), s.index('        # 6. Generate Visualizations')
s = s[:start] + '''        # Profile all experts, not just the small fusion head.
        performance = {}
        if self.run_profiling:
            from evaluation.end_to_end_profile import profile_pipeline
            performance = profile_pipeline(model)

''' + s[end:]
p.write_text(s, encoding='utf-8')
replace('training/runtime/distributed_runtime.py', "for attr_name in ('rgb_model', 'wavelet_model'):", "for attr_name in ('rgb_model', 'wavelet_model', 'xception_model', 'convnext_model'):")
replace('training/runtime/distributed_runtime.py', '''        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=is_train,
        )''', '''        from data.samplers.distributed import EvaluationSampler, DistributedWeightedSampler
        from torch.utils.data import WeightedRandomSampler
        if not is_train:
            sampler = EvaluationSampler(dataset, self.rank, self.world_size)
        elif isinstance(loader.sampler, WeightedRandomSampler):
            sampler = DistributedWeightedSampler(loader.sampler.weights, self.rank, self.world_size,
                                                  getattr(self._opt, 'seed', None) or 42)
        else:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size,
                                         rank=self.rank, shuffle=True, seed=getattr(self._opt, 'seed', None) or 42)''')
# Unpadded validation can have different batch counts. Bypass DDP's forward
# buffer collectives, after synchronizing the trained buffers once.
replace('training/validator.py', '        model.eval()\n', '''        model.eval()
        distributed_model = model.model
        if hasattr(distributed_model, 'module'):
            import torch.distributed as dist
            for buffer in distributed_model.module.buffers():
                dist.broadcast(buffer, src=0)
            model.model = distributed_model.module
''')
replace('training/validator.py', '            all_losses.append(loss.item())', '            all_losses.append((loss.item() * len(model.label), len(model.label)))')
replace('training/validator.py', '        # Aggregate\n', '        model.model = distributed_model\n        # Aggregate\n')
replace('training/validator.py', '        avg_loss = float(np.mean(all_losses)) if all_losses else 0.0', '        loss_sum = sum(v for v, n in all_losses)\n        loss_count = sum(n for v, n in all_losses)\n        avg_loss = loss_sum / loss_count if loss_count else 0.0')
replace('training/validator.py', 'tensor_loss = torch.tensor([avg_loss], dtype=torch.float32, device=device)', 'tensor_loss = torch.tensor([loss_sum, loss_count], dtype=torch.float64, device=device)')
replace('training/validator.py', 'avg_loss = (tensor_loss[0] / dist.get_world_size()).item()', 'avg_loss = (tensor_loss[0] / tensor_loss[1].clamp_min(1)).item()')
replace('training/validator.py', '        except Exception:\n            pass', '        except ImportError:\n            pass')
