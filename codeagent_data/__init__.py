"""Data pipeline for teacher-trajectory generation on the swegemma harness.

Stage 0 (this package today): ingest heterogeneous SWE task rows, normalize them
into one schema, run static + repo-tree consistency checks, and route each row
to an environment tier that decides how it will be executed and verified.
"""

__version__ = "0.1.0"
