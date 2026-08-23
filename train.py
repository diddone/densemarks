"""Single-GPU training loop for the published DenseMarks configuration."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import fields
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from densemarks.config import Config

# densemarks.data / densemarks.model / tensorboard / tqdm are imported inside
# main() so build_optimizer / build_scheduler remain testable without them.


def cosine_factor(where: float) -> float:
    """CosineParamScheduler from 1.0 down to 0.0 over where in [0, 1]."""
    return 0.5 * (1.0 + math.cos(math.pi * where))


def lr_multiplier(step: int, cfg: Config) -> float:
    """WarmupCosine multiplier for iteration ``step`` (0-based)."""
    where = step / cfg.max_iter
    wl = cfg.warmup_iters / cfg.max_iter
    if wl > 0 and where < wl:
        alpha = where / wl
        return cfg.warmup_factor * cosine_factor(0.0) * (1.0 - alpha) + cosine_factor(wl) * alpha
    return cosine_factor(where)


def build_optimizer(cfg: Config, model: nn.Module) -> torch.optim.AdamW:
    """AdamW with backbone / cube LR overrides; no WD on 0/1-D params."""
    params: list[dict] = []
    for name, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.base_lr
        wd = cfg.weight_decay_norm if value.ndim <= 1 else cfg.weight_decay
        if name.startswith("embedder.") and ".backbone" in name:
            lr = cfg.backbone_lr
        if name.rsplit(".", 1)[-1] == "_E": # cube uses different lr since it is trained from scratch
            lr = cfg.e_lr
        params.append({"params": [value], "lr": lr, "weight_decay": wd})
    return torch.optim.AdamW(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)


def build_scheduler(cfg: Config, optimizer: torch.optim.Optimizer) -> LambdaLR:
    """WarmupCosineLR as LambdaLR (BASE_LR_END=0, linear warmup, rescale_interval=False)."""

    def multiplier(step: int) -> float:
        return lr_multiplier(step, cfg)

    return LambdaLR(optimizer, lr_lambda=multiplier)


def parse_args(argv: list[str] | None = None) -> tuple[Config, bool]:
    """Build Config from argparse; defaults come from Config. ``--data_dir`` is required."""
    defaults = Config()
    parser = argparse.ArgumentParser(description="Train DenseMarks (published configuration)")
    parser.add_argument("--data_dir", type=str, required=True, help="Root of tracked dataset")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Resume from {{output_dir}}/checkpoint.pt",
    )
    for f in fields(Config):
        if f.name == "data_dir":
            continue
        default = getattr(defaults, f.name)
        flag = f"--{f.name}"
        if isinstance(default, bool):
            parser.add_argument(
                flag,
                type=lambda x: str(x).lower() in ("1", "true", "yes"),
                default=default,
            )
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(flag, type=float, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default)

    args = parser.parse_args(argv)
    resume = bool(args.resume)
    kwargs = {f.name: getattr(args, f.name) for f in fields(Config)}
    return Config(**kwargs), resume


def _cycle(loader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def run_validation(
    model: nn.Module,
    val_loader,
    val_iters: int,
    device: torch.device,
) -> dict[str, float]:
    """Mean per-key losses over ``val_iters`` batches; model stays in train() mode."""
    cpu_rng = torch.get_rng_state()
    np_rng = np.random.get_state()
    cuda_rng = None
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state_all()

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    model.train()
    batches = _cycle(val_loader)
    with torch.no_grad():
        for _ in range(val_iters):
            batch = next(batches).to(device)
            losses = model(batch)
            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())
                counts[k] = counts.get(k, 0) + 1

    torch.set_rng_state(cpu_rng)
    np.random.set_state(np_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    return {k: totals[k] / counts[k] for k in totals}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
        },
        path,
    )


def main(cfg: Config, resume: bool = False) -> None:
    # Imported here (not at module top) so tests can import build_* without A/C.
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    from densemarks.data import build_dataloader
    from densemarks.model import DenseMarksTrainModel

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    device = torch.device(cfg.device)

    model = DenseMarksTrainModel(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    train_loader = build_dataloader(cfg, is_train=True)
    val_loader = build_dataloader(cfg, is_train=False)
    writer = SummaryWriter(cfg.output_dir)

    start_step = 0
    ckpt_path = Path(cfg.output_dir) / "checkpoint.pt"
    if resume:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--resume set but checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt["step"]) + 1

    model.train()
    pbar = tqdm(range(start_step, cfg.max_iter), initial=start_step, total=cfg.max_iter)
    train_iter = iter(train_loader)

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        losses = model(batch.to(device))
        total = sum(losses.values())
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        scheduler.step()

        if step % cfg.log_period == 0:
            writer.add_scalar("train/total", float(total.detach()), step)
            for k, v in losses.items():
                writer.add_scalar(f"train/{k}", float(v.detach()), step)
            for i, group in enumerate(optimizer.param_groups):
                writer.add_scalar(f"train/lr_group_{i}", group["lr"], step)
            logits_scale = model.loss_head.logits_scale
            writer.add_scalar("train/logits_scale_exp", float(logits_scale.exp().detach()), step)
            pbar.set_postfix(loss=float(total.detach()))

        if cfg.val_period > 0 and step > 0 and step % cfg.val_period == 0:
            val_means = run_validation(model, val_loader, cfg.val_iters, device)
            for k, v in val_means.items():
                writer.add_scalar(f"val/{k}", v, step)
            if val_means:
                writer.add_scalar("val/total", sum(val_means.values()), step)
            model.train()

        if cfg.ckpt_period > 0 and step > 0 and step % cfg.ckpt_period == 0:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, step)

    final_path = Path(cfg.output_dir) / "model_final.pt"
    save_checkpoint(final_path, model, optimizer, scheduler, cfg.max_iter - 1)
    writer.close()


if __name__ == "__main__":
    cfg, resume = parse_args()
    main(cfg, resume=resume)
