"""
ai-engine/conftest.py

pytest picks this up before importing any test module, which makes it the only
reliable place to set process-wide environment variables for the whole suite.

Production sets these in api/main.py and src/recommender.py, but the ML tests
import src.ml.* directly and never touch src.recommender -- so without this
file they would run with no OpenMP guard at all.

These flags are defence in depth, not the actual fix for the four-way libomp
conflict (torch / sklearn / faiss / lightgbm each bundle their own copy):
KMP_DUPLICATE_LIB_OK only silences libomp's load-time duplicate-registration
abort, and OMP_NUM_THREADS is overridden by LightGBM's own n_jobs parameter.
The real cure lives in ApplicabilityModel._pin_single_thread().

setdefault() so an explicitly exported env var still wins.
"""

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Make `import src.…` work when pytest is invoked from the ai-engine root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
