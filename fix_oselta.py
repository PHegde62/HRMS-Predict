import re

with open('benchmark/hrms_mass_accuracy.py', 'r', encoding='utf-8') as f:
    src = f.read()

old = re.search(r'\{[^}]*"name":\s*"Oseltamivir"[^}]*\},', src, re.DOTALL)
if old:
    new_block = '''    {
        "name":         "Naproxen",
        "smiles":       "COc1ccc2cc(C(C)C(=O)O)ccc2c1",
        "formula":      "C14H14O3",
        "mw_range":     "medium (230 Da)",
        "lit_neutral":  230.0943,
        "lit_mplus_h":  231.1016,
        "lit_mminus_h": 229.0870,
        "source":       "NIST, PubChem CID 156391",
    },'''
    src = src[:old.start()] + new_block + src[old.end():]
    with open('benchmark/hrms_mass_accuracy.py', 'w', encoding='utf-8') as f:
        f.write(src)
    print('DONE - Oseltamivir replaced with Naproxen')
else:
    print('NOT FOUND')
    idx = src.find('Oseltamivir')
    if idx >= 0:
        print(repr(src[idx-50:idx+150]))
