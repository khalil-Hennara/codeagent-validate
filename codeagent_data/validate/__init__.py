"""Stage 1: model-free validation of every task in its own environment.

For each task, in a fresh container from its image:
  base run : test_patch only        -> every FAIL_TO_PASS must fail (else the task is trivial/broken)
  gold run : gold patch + test_patch -> every FAIL_TO_PASS and PASS_TO_PASS must pass
"""
