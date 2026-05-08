# -*- coding: utf-8 -*-
"""
Created on Fri May  8 17:41:20 2026

@author: Sarah
"""

# build.py
# Rebuild all Cython extensions in-place for local development

import os
import sys
import subprocess


def main():

    cmd = [
        sys.executable,
        "setup.py",
        "build_ext",
        "--inplace",
        "--force"
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n===================================")
        print("BUILD SUCCESS")
        print("===================================\n")
    else:
        print("\n===================================")
        print("BUILD FAILED")
        print("===================================\n")


if __name__ == "__main__":
    main()