"""Write materials_ml_primer/wyckoff_sets.json from the Wyckoff-set table shipped with PyXtal.

For every space group the table lists the operations of its normalizer (coset representatives) and how each
one permutes the Wyckoff letters. The compact file written here keeps only the permutations: for each space
group, one string per operation whose k-th character is the image of the k-th letter (a, b, c, ..., z, A).

PyXtal is MIT-licensed; see THIRD_PARTY_NOTICES.md. Usage:

    pip download pyxtal==1.1.5 --no-deps -d <folder>
    python _build/make_wyckoff_sets.py <folder>/pyxtal-1.1.5-py3-none-any.whl materials_ml_primer/wyckoff_sets.json
"""
import json
import sys
import zipfile

wheel, out = sys.argv[1], sys.argv[2]
table = json.loads(zipfile.ZipFile(wheel).read("pyxtal/database/wyckoff_sets.json"))
compact = {}
for sg in range(1, 231):
    rows = [row.split() for row in table[str(sg)]["Transformed WP"]]
    identity = rows[0]
    assert all(sorted(r) == sorted(identity) for r in rows), sg       # every row is a permutation
    compact[str(sg)] = sorted({"".join(r) for r in rows})
with open(out, "w", encoding="utf-8") as f:
    json.dump(compact, f, separators=(",", ":"), sort_keys=False)
    f.write("\n")
print("wrote", out, sum(len(v) for v in compact.values()), "permutations for 230 space groups")
