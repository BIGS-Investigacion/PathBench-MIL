#!/usr/bin/env python3
"""
scripts/compute_centroid_distances.py

Wrapper around patch_intersection_all.py.
Use patch_intersection_all.py directly instead.
"""
import subprocess, sys
subprocess.run(
    [sys.executable,
     'scripts/patch_intersection_all.py'] + sys.argv[1:],
    check=True,
)