import nbformat as nbf
import json

nb = nbf.v4.new_notebook()

with open('notebooks/milk_quality_analysis_updated.py', 'r') as f:
    script = f.read()

cells = script.split('# In[')

header = cells[0].split('# # ')[0]
first_markdown = cells[0].split('# # ')[1:]

if header.strip():
    nb['cells'].append(nbf.v4.new_markdown_cell('# ' + header.strip()))

for i, cell in enumerate(cells):
    if i == 0:
        for md in first_markdown:
            if md.strip():
                nb['cells'].append(nbf.v4.new_markdown_cell('# ' + md.strip()))
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

with open('notebooks/milk_quality_analysis.ipynb', 'w') as f:
    json.dump(nb, f)

print('Notebook created successfully')
