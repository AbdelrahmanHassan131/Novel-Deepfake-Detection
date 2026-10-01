"""Validate smoke commands against source parsers without importing application code."""
import argparse
import ast
import json
from pathlib import Path


def source_parser(path):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    parser = argparse.ArgumentParser(add_help=False)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
            continue
        args = [ast.literal_eval(arg) for arg in node.args]
        kwargs = {}
        for kw in node.keywords:
            if kw.arg in {'action', 'choices', 'dest', 'nargs', 'required', 'const'}:
                if isinstance(kw.value, ast.Attribute) and kw.value.attr == 'BooleanOptionalAction':
                    kwargs[kw.arg] = argparse.BooleanOptionalAction
                else:
                    try:
                        kwargs[kw.arg] = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        pass  # Dynamic help/defaults/choices cannot be proven by this check.
            elif kw.arg == 'type' and isinstance(kw.value, ast.Name):
                if kw.value.id in {'int', 'float', 'str'}:
                    kwargs['type'] = {'int': int, 'float': float, 'str': str}[kw.value.id]
        parser.add_argument(*args, **kwargs)
    return parser


def validate(root):
    plan = json.loads((root / 'config/experiments/smoke_test_run.json').read_text(encoding='utf-8'))
    commands = plan['command_templates']
    for name, argv in commands.items():
        relative = argv[1].removeprefix('{repo}/')
        entry = root / relative
        if not argv[1].startswith('{repo}/') or not entry.is_file():
            raise ValueError(f'{name}: missing entry point {argv[1]}')
        source_parser(entry).parse_args(argv[2:])
    for stage in ('rgb', 'wavelet', 'fusion', 'resume'):
        argv = commands[stage]
        if '--no-pretrained' not in argv or '--use_amp' in argv:
            raise ValueError(f'{stage}: smoke must disable pretrained initialization and AMP')
        for flag, expected in {'--batch_size': '8', '--grad_accum_steps': '1', '--num_workers': '0',
                               '--optim': 'adam', '--lr': '0.0001', '--monitor_metric': 'auc'}.items():
            if argv[argv.index(flag) + 1] != expected:
                raise ValueError(f'{stage}: unexpected {flag}')
    # Strip only permitted resume differences; every other argument must remain identical.
    def stable(argv):
        result, index = [], 0
        while index < len(argv):
            if argv[index] in {'--epochs', '--resume_checkpoint'}:
                index += 2
            elif argv[index] == '--continue_train':
                index += 1
            else:
                result.append(argv[index])
                index += 1
        return result
    if stable(commands['fusion']) != stable(commands['resume']):
        raise ValueError('Resume changed the fusion training configuration')
    defaults_tree = ast.parse((root / 'config/defaults.py').read_text(encoding='utf-8-sig'))
    training_defaults = next(ast.literal_eval(n.value) for n in defaults_tree.body
                             if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TRAINING_DEFAULTS' for t in n.targets))
    if training_defaults['use_amp'] is not False:
        raise ValueError('AMP default changed; update smoke commands to keep their declared precision')
    notebook = json.loads((root / 'colab_smoke_pipeline.ipynb').read_text(encoding='utf-8'))
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            if cell.get('execution_count') is not None or cell.get('outputs'):
                raise ValueError('Committed smoke notebook must remain unexecuted')
            ast.parse(''.join(cell['source']), filename=f'smoke cell {index}')
    print(f'Static smoke check passed: {len(commands)} command templates and notebook syntax. Runtime NOT RUN.')


if __name__ == '__main__':
    validate(Path(__file__).resolve().parents[1])
