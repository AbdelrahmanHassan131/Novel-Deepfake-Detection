import ast
import json
import os
import sys

def check_python_files(root_dir):
    errors = []
    py_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        if any(skip in dirpath for skip in ['.git', '__pycache__', '.pytest_cache', 'tmp', '.venv']):
            continue
        for fn in filenames:
            if fn.endswith('.py'):
                fp = os.path.join(dirpath, fn)
                py_count += 1
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        ast.parse(f.read(), filename=fp)
                except Exception as e:
                    errors.append(f"Python AST error in {fp}: {e}")
    return py_count, errors

def check_json_files(root_dir):
    errors = []
    json_count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        if any(skip in dirpath for skip in ['.git', '__pycache__', '.pytest_cache', 'tmp', '.venv']):
            continue
        for fn in filenames:
            if fn.endswith('.json'):
                fp = os.path.join(dirpath, fn)
                json_count += 1
                try:
                    with open(fp, 'r', encoding='utf-8') as f:
                        json.load(f)
                except Exception as e:
                    errors.append(f"JSON parse error in {fp}: {e}")
    return json_count, errors

def check_notebook(nb_path):
    errors = []
    if not os.path.exists(nb_path):
        return ["Notebook missing: " + nb_path]
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        cells = nb.get('cells', [])
        for i, c in enumerate(cells):
            if c.get('cell_type') == 'code':
                if c.get('outputs') != []:
                    errors.append(f"Cell {i} has non-empty outputs (must be unexecuted)")
                if c.get('execution_count') is not None:
                    errors.append(f"Cell {i} has execution_count != None (must be unexecuted)")
                # verify code syntax in cell
                source = "".join(c.get('source', []))
                # remove IPython line magics / shell commands before ast parse
                filtered_lines = []
                in_shell = False
                base_indent = ""
                for line in source.splitlines():
                    stripped = line.strip()
                    if stripped.startswith('!') or stripped.startswith('%'):
                        indent = ' ' * (len(line) - len(line.lstrip()))
                        base_indent = indent
                        filtered_lines.append(f"{indent}pass  # {stripped}")
                        in_shell = stripped.endswith('\\')
                    elif in_shell:
                        filtered_lines.append(f"{base_indent}# {stripped}")
                        in_shell = stripped.endswith('\\')
                    else:
                        base_indent = ""
                        filtered_lines.append(line)
                try:
                    ast.parse("\n".join(filtered_lines))
                except Exception as e:
                    errors.append(f"Notebook Cell {i} AST error: {e}")
    except Exception as e:
        errors.append(f"Notebook load error: {e}")
    return errors

if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"Validating workspace root: {root}")
    py_count, py_errors = check_python_files(root)
    print(f"Parsed {py_count} Python files. Errors: {len(py_errors)}")
    for err in py_errors:
        print("  ", err)

    json_count, json_errors = check_json_files(root)
    print(f"Parsed {json_count} JSON files. Errors: {len(json_errors)}")
    for err in json_errors:
        print("  ", err)

    total_nb_errors = 0
    for nb_name in ['colab_pilot_pipeline.ipynb', 'colab_smoke_pipeline.ipynb']:
        nb_path = os.path.join(root, nb_name)
        if os.path.isfile(nb_path):
            nb_errors = check_notebook(nb_path)
            print(f"Checked notebook {nb_path}. Errors: {len(nb_errors)}")
            for err in nb_errors:
                print("  ", err)
            total_nb_errors += len(nb_errors)

    total_errors = len(py_errors) + len(json_errors) + total_nb_errors
    if total_errors == 0:
        print("\nALL CHECKS PASSED: Zero syntax, parsing, or structural errors!")
        sys.exit(0)
    else:
        print(f"\nFAILED with {total_errors} errors.")
        sys.exit(1)
