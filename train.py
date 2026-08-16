"""Standalone training entry point -- use this instead of the menu's Train button
when you want multi-process data loading.

    .venv\\Scripts\\python.exe train.py
    .venv\\Scripts\\python.exe train.py --epochs 60 --workers 8
    .venv\\Scripts\\python.exe train.py --workers 0        # single-process, as before

WHY THIS FILE EXISTS
Windows has no fork(), so PyTorch starts DataLoader workers with "spawn": each worker
process re-imports whatever module is __main__ in order to rebuild its namespace.
Autopilot_Menu.py creates an Ursina app at import time, so re-importing it in a worker
would open blank windows -- which is what the
    if multiprocessing.current_process().name != 'MainProcess': sys.exit(0)
guard in that file was suppressing. But that guard also kills the worker before it can
load any data, so num_workers > 0 could never work from the menu.

This module has no import-time side effects, so workers can re-import it harmlessly.
Training launched from the menu still runs with num_workers=0 and is unaffected.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    import argparse
    import multiprocessing

    parser = argparse.ArgumentParser(description="Train the autopilot model.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs. Early stopping usually ends the run sooner.")
    parser.add_argument("--workers", type=int, default=6,
                        help="DataLoader worker processes. 0 = load on the main thread.")
    parser.add_argument("--build-salience", action="store_true",
                        help="Score every recorded frame for stop-sign visibility and cache "
                             "it beside each session CSV, then exit. Run once after "
                             "recording new sessions. Does not modify recorded data.")
    parser.add_argument("--rebuild-salience", action="store_true",
                        help="As --build-salience, but rescore frames already cached.")
    args = parser.parse_args()

    multiprocessing.freeze_support()

    if args.build_salience or args.rebuild_salience:
        import DataCollection
        DataCollection.buildSignSalienceCache(force=args.rebuild_salience)
        return

    import Training
    Training.trainStart(args.epochs, num_workers=args.workers)


if __name__ == "__main__":
    main()
