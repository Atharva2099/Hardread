import sys
from pathlib import Path

VGC = str(Path(__file__).resolve().parents[2])
if VGC not in sys.path:
    sys.path.insert(0, VGC)
