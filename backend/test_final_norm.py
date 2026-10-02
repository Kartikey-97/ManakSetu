from shared.models import Standard
from kshiraj.bis_live_ingestion.normalizer import normalize_designation
data = [
    {"is_number": "IS 10322 : Part 5 : Sec 3", "title": "...", "part": "Part 5", "section": "Sec 3"},
    {"is_number": "IS 13450 : Part 2 : Sec 10", "title": "...", "part": "Part 2", "section": "Sec 10"},
    {"is_number": "IS 1944 : Part 1 and 2", "title": "...", "part": "Part 1 and 2"}
]
for d in data:
    std = Standard(**d)
    base = std.base_designation
    norm = normalize_designation(base)
    print(f"{d['is_number']} \n  -> base: {base} \n  -> norm: {norm}\n")
