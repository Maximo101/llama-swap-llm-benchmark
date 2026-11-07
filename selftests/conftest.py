import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # import llmbench without installing it
sys.path.insert(0, str(Path(__file__).resolve().parent))
