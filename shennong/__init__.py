"""Shennong's Glass: foundation-model phenotypic profiling for drug screening."""
import pandas as pd

# pandas >= 3 stores strings as Arrow arrays, which do not support numpy-style
# broadcasting (``a[:, None] == a[None, :]``) used throughout; keep object dtype.
try:
    pd.set_option("future.infer_string", False)
except Exception:  # older pandas
    pass
