import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from download import find_model
from models import DiT_models
from diffusion import create_diffusion
from dataset import EmbeddingDataset  # Your custom dataset
import random

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir, rank=0):
    """
    Create a logger that writes to a log file and stdout.
    If not rank 0, use a dummy logger.
    """
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Determine if distributed training is needed
    is_distributed = (torch.cuda.device_count() > 1)

    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = rank % torch.cuda.device_count() if torch.cuda.is_available() else "cpu"
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}, device={device}")

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir, rank=rank)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None, rank=rank)

    # Create model:
    latent_size = 768  # Embedding size
    in_channels = 1

    if args.unconditional:
        cross_attn = 0
        condition_dim = 0
    else:
        cross_attn = 768
        condition_dim = 0

    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        cross_attn=cross_attn,
        condition_dim=condition_dim
    )

    if args.ckpt:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        msg = model.load_state_dict(state_dict, strict=True)
        print('Loaded DiT from ', ckpt_path, msg)

    # Create EMA:
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    if is_distributed:
        model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    else:
        model = model.to(device)

    diffusion = create_diffusion(timestep_respacing="")

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Setup data
    dataset = EmbeddingDataset(
        data_path=args.data_path,
        data_length=None,
        shuffle=True,
        unconditional=args.unconditional 
    )

    if is_distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.global_seed
        )
        loader = DataLoader(
            dataset,
            batch_size=int(args.global_batch_size // world_size),
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.global_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

    logger.info(f"Dataset contains {len(dataset):,} samples")

    # Prepare models for training:
    # If distributed, model.module is the underlying model; otherwise just model
    if is_distributed:
        base_model = model.module
    else:
        base_model = model

    update_ema(ema, base_model, decay=0)  # Initialize EMA
    model.train()
    ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        if is_distributed:
            sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            x, y = batch
            x = x.unsqueeze(-1).to(device)  # [N, 768, 1]
            model_kwargs = {}

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, base_model)

            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                if is_distributed:
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / world_size
                else:
                    avg_loss = avg_loss.item()

                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": base_model.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                if is_distributed:
                    dist.barrier()

    model.eval()
    logger.info("Done!")

    if is_distributed:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100) # original 1400
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=1000) # original 10000
    parser.add_argument("--data-path", nargs='+', required=True,
                        help="List of paths in pairs: mol_embed.npy assay_embed.npy mol_embed2.npy assay_embed2.npy ...")
    parser.add_argument("--unconditional", action='store_true', help="Run model in unconditional mode")

    args = parser.parse_args()

    if args.unconditional:
        # For unconditional training, treat `args.data_path` as a list of molecule embedding file paths
        args.data_path = args.data_path  # Already a flat list of molecule files
    else:
        # For conditional training, treat `args.data_path` as a list of (molecule, assay) file pairs
        if len(args.data_path) % 2 != 0:
            raise ValueError("data-path should contain an even number of entries: pairs of (mol_path, assay_path)")
        args.data_path = [(args.data_path[i], args.data_path[i+1]) for i in range(0, len(args.data_path), 2)]

    main(args)