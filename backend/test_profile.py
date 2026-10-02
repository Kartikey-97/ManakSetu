import sys
import os
sys.path.append(os.getcwd())

from kartikey.analysis.profile_extractor import extract_profile

with open('data/demo_analysis/led_tender.txt', 'r') as f:
    text = f.read()

profile = extract_profile(text)
print("Testing requirements count:", len(profile.testing_requirements))
for r in profile.testing_requirements:
    print(r.value)
