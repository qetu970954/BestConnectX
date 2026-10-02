"""Local play/results dashboard for a training run; training stays in the CLI."""
import sys
from gomoku.cli import main

if __name__ == "__main__":
    main(["web", *sys.argv[1:]])
