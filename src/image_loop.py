# /// script
# dependencies = [
#   "httpx",
#   "Pillow",
#   "ffmpeg-python",
#   "tqdm",
#   "pyyaml",
# ]
# ///

"""
Image Loop Generator - Iterative AI image generation using OpenRouter

Run with: uv run src/image_loop.py --help
"""

import sys
from imageloop.cli import main

if __name__ == "__main__":
    sys.exit(main())
