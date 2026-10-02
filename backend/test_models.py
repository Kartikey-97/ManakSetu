from shared.models import Standard
import json

data = [
    {"is_number": "IS 10322 : Part 5 : Sec 3", "title": "...", "part": "Part 5", "section": "Sec 3"},
    {"is_number": "IS 13450 : Part 2 : Sec 10", "title": "...", "part": "Part 2", "section": "Sec 10"},
    {"is_number": "IS 1944 : Part 1 and 2", "title": "...", "part": "Part 1 and 2"}
]

for d in data:
    std = Standard(**d)
    print(f"{d['is_number']} -> {std.base_designation}")
