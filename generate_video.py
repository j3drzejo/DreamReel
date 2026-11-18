#!/usr/bin/env python3
"""Entry point:  python generate_video.py [--topic "..."] [--duration 12]"""
import sys

import config  # noqa: F401  (must be imported before torch)
from main import main

if __name__ == "__main__":
    sys.exit(main())
