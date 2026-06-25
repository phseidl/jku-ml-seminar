
"""
scripts/run_training.py
=======================
The single command-line door to one xLSTM-ECG training run. It does no science
of its own: it loads 'config/config.yaml', layers any 'key=value' overrides
given on the command line on top of it, resolves the merged config, and hands
the result to 'train(cfg)' -- the one training orchestrator in
'src/training/train.py'.

Role in the pipeline
--------------------
I route every single run through this one file on purpose, so that a run I
launch by hand and a run launched by an automated sweep execute identical code.
Anything that differs between experiments is expressed as a config override; the
committed 'config.yaml' on disk is never touched.

    config/config.yaml  --load-->  OmegaConf  --merge(overrides)-->  cfg
                                                                      |
                                                          train(cfg) v
                                              src/training/train.py:train

What it reads
-------------
* 'config/config.yaml' (or wherever '--config' points) -- the single source of
  truth for data paths, model dims, optimizer, scheduler, loss, etc.
* zero or more positional 'key=value' overrides in OmegaConf dot notation,
  e.g. 'model.dropout=0.2 training.optimizer=adamw'.

What reads / imports it
-----------------------
* A person, interactively: 'python -m scripts.run_training [overrides ...]'.
* Every sweep harness ('scripts/sweep*.py', 'scripts/sweep_*.py') shells out to
  it as a subprocess via 'python -m scripts.run_training' with a per-run list of
  overrides (usually including 'training.result_file=...' so the run drops a
  metrics JSON the harness then collects). Giving each config its own process is
  what hands the sweep a clean CUDA / sLSTM-kernel state for every run -- no
  state leaks from one config into the next.

How it is run
-------------
    # use the committed defaults verbatim
    python -m scripts.run_training

    # override any config key with OmegaConf dotlist syntax
    python -m scripts.run_training model.dropout=0.2 training.optimizer=adamw

    # point at a different base config file
    python -m scripts.run_training --config config/other.yaml data.batch_size=512

This file returns nothing and collects no metrics itself. A run that should
report numbers must set 'training.result_file' (as the sweeps do); 'train()'
writes that JSON, and downstream tooling ('evaluate_checkpoint.py', the sweep
summary builders) reads it.
"""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

# Make 'src' importable when this file is run directly
# ('python scripts/run_training.py') rather than as a module. '__file__' is
# '.../scripts/run_training.py'; '.parent.parent' is the package root
# ('.../submission'), which holds the 'src' package. I resolve() first so the
# inserted path is absolute and independant of where the command was run from,
# and prepend at index 0 so this copy wins over any same-named package already
# on sys.path.
# Under 'python -m scripts.run_training' the root is already on sys.path, so
# this insert is a harmless no-op -- it only matters when the file is run
# directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.training.train import train
 
 
def main():
    """Parse the CLI, build the effective config, and launch one training run.

    Flow
    ----
    1. Parse '--config' (the base YAML, defaulting to 'config/config.yaml') and
       an arbitrary list of positional 'key=value' overrides.
    2. Load the base config, merge the overrides on top, fully resolve it, then
       call 'train(cfg)'.

    Args / returns
    --------------
    Takes no arguments and returns nothing -- it reads 'sys.argv' via 'argparse'
    and produces side effects only (training, checkpoints, and, if
    'training.result_file' is set, a metrics JSON written by 'train'). The exit
    code is whatever 'train' / the interpreter yields; an unhandled exception in
    'train' propagates, so a supervising sweep subprocess sees a non-zero
    return and knows the run failed.

    Non-obvious behavior
    --------------------
    Overrides win over the base config (last-writer-wins via 'OmegaConf.merge'),
    and the config is resolved before 'train' sees it -- so a broken
    interpolation or a missing required key fails here, in the first second,
    instead of after the GPU has already burned minutes inside the training
    loop.
    """
    # Deliberately just two arguments: one optional base-config path and a
    # catch-all list of dotlist overrides. Anything else a run needs to vary is
    # an override, not a new flag -- that is what keeps this CLI stable as the
    # experiment program grows.
    parser = argparse.ArgumentParser(description="Train xLSTM-ECG")
    # '--config' defaults to the committed single-source-of-truth YAML. The
    # default is relative, so it resolves against the current working directory
    # (the sweeps and the documented commands all run from the 'practical_work'
    # package root, where 'config/config.yaml' lives).
    # NOTE: resolve the default against this file's location instead of CWD so
    # 'python -m scripts.run_training' works from any directory, not only the
    # package root.
    parser.add_argument("--config", default="config/config.yaml")
    # 'nargs="*"' sweeps every remaining bare token into 'args.overrides' as a
    # list of "key=value" strings -- e.g. 'model.dropout=0.2' and
    # 'training.optimizer=adamw' become ["model.dropout=0.2",
    # "training.optimizer=adamw"]. These are OmegaConf dot-list entries, not
    # shell '--flag value' pairs.
    parser.add_argument("overrides", nargs="*",
                        help="OmegaConf dot-notation overrides: key=value ...")
    args = parser.parse_args()

    # Load the base config tree from YAML into an OmegaConf DictConfig.
    cfg = OmegaConf.load(args.config)
    if args.overrides:
        # 'from_dotlist' turns ["a.b=1", "c=2"] into a nested config, and
        # 'merge' overlays it on the base, last-writer-wins -- so the CLI
        # overrides beat the YAML defaults. An empty override list skips the
        # merge entirely.
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    # Resolve now, to catch missing keys early. 'to_container(resolve=True)'
    # evaluates every '${...}' interpolation and surfaces any MISSING or
    # unresolvable key at startup, not hundreds of GPU-seconds into the loop. I
    # then rebuild a plain, already-resolved config with 'create' so 'train'
    # gets a fully concrete tree with no lazy interpolations left to trip over.
    # 'OmegaConf.resolve(cfg)' would mutate in place, but the round-trip through
    # to_container also strips the interpolation machinery, which is what I want
    # before handing the config downstream.
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Hand off to the one training orchestrator. Everything past this point --
    # dataset choice (ptbxl | georgia), model build, optimizer/scheduler, the
    # per-epoch train/eval loop, early stopping, reloading the best checkpoint,
    # the final test eval, and the optional result.json -- lives in
    # 'src/training/train.py:train'.
    train(cfg)
 
 
# Standard script guard: run the CLI only when this file is the program entry
# point ('python -m scripts.run_training' or 'python scripts/run_training.py'),
# not when something merely imports 'main' from it.
if __name__ == "__main__":
    main()
 