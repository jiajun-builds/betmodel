"""Thin entry point; the model logic lives in ligamx.models.dc.

Run as ``python DC_MEX.py`` with PYTHONPATH=src (the scripts set this up).
"""

from ligamx.models import dc

if __name__ == "__main__":
    dc.main()
