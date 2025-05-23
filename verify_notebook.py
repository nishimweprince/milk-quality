import json
import sys

def verify_notebook(notebook_path):
    """Verify that the notebook is properly structured for Cursor IDE."""
    try:
        with open(notebook_path, 'r') as f:
            notebook = json.load(f)
        
        if 'cells' not in notebook:
            print("ERROR: Notebook does not have 'cells' key")
            return False
        
        cell_types = [cell['cell_type'] for cell in notebook['cells']]
        markdown_count = cell_types.count('markdown')
        code_count = cell_types.count('code')
        
        print(f"Notebook structure verification:")
        print(f"Total cells: {len(notebook['cells'])}")
        print(f"Markdown cells: {markdown_count}")
        print(f"Code cells: {code_count}")
        
        if 'metadata' not in notebook:
            print("WARNING: Notebook does not have metadata")
        
        if 'kernelspec' not in notebook.get('metadata', {}):
            print("WARNING: Notebook does not have kernelspec metadata")
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and (not cell.get('source') or not ''.join(cell.get('source', '')).strip()):
                print(f"WARNING: Code cell {i} is empty")
        
        print("\nNotebook verification complete.")
        return True
    
    except Exception as e:
        print(f"ERROR: Failed to verify notebook: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_notebook.py <notebook_path>")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    success = verify_notebook(notebook_path)
    
    if success:
        print("Notebook structure looks good for Cursor IDE.")
    else:
        print("Notebook structure may have issues for Cursor IDE.")
        sys.exit(1)
