"""Direct training entry point: python train.py --connect 5 --board_size '9*9' ..."""
import sys
from engine.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
