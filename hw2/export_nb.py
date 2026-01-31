import json
import sys

def export_to_py(nb_path, py_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if source.strip():
                code_cells.append(source)
    
    # Filter out cells that might be meta/export scripts
    final_cells = [c for c in code_cells if 'export_to_py' not in c and 'import json' not in c]
    
    with open(py_path, 'w', encoding='utf-8') as f:
        # Join with newlines
        f.write('\n\n'.join(final_cells))
    print(f"Exported {nb_path} to {py_path}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        export_to_py(sys.argv[1], sys.argv[2])
