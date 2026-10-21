import argparse
import runpy
import sys
from pathlib import Path


TRAIN_SCRIPTS = {
    "3dgs": "dualdiff3d/train_3dgs.py",
    "dualdiff-reliable": "dualdiff3d/train_reliable.py",
}


def main():
    if len(sys.argv) > 1 and sys.argv[1] in TRAIN_SCRIPTS:
        mode = sys.argv[1]
        repo_root = Path(__file__).resolve().parent
        script_path = repo_root / TRAIN_SCRIPTS[mode]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        sys.argv = [str(script_path), *sys.argv[2:]]
        runpy.run_path(str(script_path), run_name="__main__")
        return

    parser = argparse.ArgumentParser(
        description="Unified training entry point for the main DualDiff3D path and baseline experiments."
    )
    parser.add_argument(
        "mode",
        choices=sorted(TRAIN_SCRIPTS),
        help="Training mode to run.",
    )
    parser.parse_args()


if __name__ == "__main__":
    main()
