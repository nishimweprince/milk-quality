import nbformat as nbf
import json
import os

nb = nbf.v4.new_notebook()

with open('notebooks/milk_quality_analysis_updated.py', 'r') as f:
    script = f.read()

cells = script.split('# In[')

header = cells[0].split('# # ')[0]
first_markdown = cells[0].split('# # ')[1:]

if header.strip():
    nb['cells'].append(nbf.v4.new_markdown_cell(header.strip()))

for md in first_markdown:
    if md.strip():
        nb['cells'].append(nbf.v4.new_markdown_cell('# ' + md.strip()))

for i, cell in enumerate(cells):
    if i == 0:
        continue
    
    parts = cell.split('# ## ')
    
    code_part = parts[0]
    if ']:' in code_part:
        code_part = code_part.split(']:')[1].strip()
    
    if code_part.strip():
        nb['cells'].append(nbf.v4.new_code_cell(code_part.strip()))
    
    for j, md in enumerate(parts[1:]):
        if md.strip():
            nb['cells'].append(nbf.v4.new_markdown_cell('## ' + md.strip()))

nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "codemirror_mode": {
            "name": "ipython",
            "version": 3
        },
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.8.10"
    }
}

with open('notebooks/milk_quality_analysis.ipynb', 'w') as f:
    json.dump(nb, f)

print("Notebook created successfully at notebooks/milk_quality_analysis.ipynb")
