"""
scripts/run_training.py
=======================
The single command-line door to one xLSTM-ECG training run.
"""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.train import train


def main():
    """Parse the CLI, build the effective config, and launch one training run."""
    # Deliberately just two arguments: an optional base-config path and a
    # catch-all list of dotlist overrides. Anything a run needs to vary is an
    # override, not a new flag -- that keeps this CLI stable as the experiment
    # program grows.
    parser = argparse.ArgumentParser(description="Train xLSTM-ECG")
    parser.add_argument("--config", default="config/config.yaml")

    # nargs="*" sweeps every remaining bare token into args.overrides as
    # "key=value" strings (e.g. model.dropout=0.2) -- these are OmegaConf
    # dot-list entries, not shell '--flag value' pairs.
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dot-notation overrides: key=value ...")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.overrides:
        # Overlay the CLI overrides on the YAML, last-writer-wins.
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # Resolve every interpolation now, so a missing key fails at startup rather
    # than hundreds of GPU-seconds into the loop.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Hand off to the one training orchestrator.
    train(cfg)


# Run the CLI only when invoked as a script, not when something imports main.
if __name__ == "__main__":
    main()
