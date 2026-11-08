"""Print the text outputs of an executed notebook, cell by cell, to check the numbers quoted in a README.

Usage: python _build/dump_nb.py experiments/<topic>/<topic>.ipynb
"""
import sys

import nbformat

nb = nbformat.read(sys.argv[1], as_version=4)
for k, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.output_type == "stream":
            text = o.text
        elif o.output_type == "error":
            text = "ERROR " + o.ename + ": " + o.evalue
        elif "text/plain" in o.get("data", {}) and "image/png" not in o.get("data", {}):
            text = o.data["text/plain"]
        else:
            continue
        print(f"--- cell {k} ---")
        print(text[-3500:])
