"""Training entry point: uv run python train.py --preset gomoku --data PATH."""
import sys
from engine.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
