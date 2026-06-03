from typing import Dict
import yaml
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as tv_utils

import data
from models import get_model

from utils import flow_matching_utils as fm_utils
from utils import fid_utils, log_utils

_TRAIN_LOSS_KEY = "train_loss"
_VAL_LOSS_KEY = "val_loss"
_VAL_FID_KEY = "val_fid"

_LR_KEY = "learning_rate"

_TRAIN_STEPS_PER_SECOND_KEY = "train_batch_steps_per_second"
_VAL_STEPS_PER_SECOND_KEY = "val_batch_steps_per_second"


class Trainer:
    def __init__(self, config_path: str, ckpt_dir_path: str):
        # Load configuration from YAML file
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.dataset_name = self.config["dataset"]["name"]
        self.num_classes = data.get_num_classes_from_dataset(self.dataset_name)

        self.device = self._get_device()
        self.amp_enabled = self.config.get("amp", {}).get("enabled", True) and (
            self.device.type != "cpu"
        )

        self.model = self._build_model()
        self.model = self.model.to(self.device)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Setup loss function and samplers
        self.loss_fn = fm_utils.compute_flow_matching_loss
        self.train_time_sampler = fm_utils.UniformTimeSampler()
        self.val_time_sampler = fm_utils.UniformTimeSampler()

        self.val_ode_solver = fm_utils.EulerSolver(
            num_steps=self.config["validation"].get("sample_num_steps", 20)
        )

        # Setup checkpoint directory
        os.makedirs(ckpt_dir_path, exist_ok=True)
        self.ckpt_dir_path = ckpt_dir_path

        with open(os.path.join(self.ckpt_dir_path, "config.yml"), "w") as fp:
            yaml.dump(self.config, fp)

        self._setup_dataloaders()
        self._setup_metrics()

        self._validated_step = -1

    def _get_device(self) -> torch.device:
        """Automatically detect available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        else:
            return torch.device("cpu")

    def _build_model(self) -> nn.Module:
        """Build the model based on config"""
        # Extract model parameters from config
        model_config = self.config["model"]
        model_arch = model_config["arch"]
        model_params = model_config.get("params", dict())

        model = get_model(
            model_arch=model_arch,
            model_params=model_params,
            c_in=3,
            num_classes=self.num_classes,
        )

        self._count_model_params(model)

        return model

    def _count_model_params(self, model: nn.Module) -> None:
        """Count and print the number of parameters."""
        n_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"[{self.config['model']['arch']}] Total params: {n_params:,} | Trainable: {trainable_params:,}"
        )
        print("-" * 80)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with configurable parameters"""
        weight_decay = self.config.get("weight_decay", 0.0)

        self.grad_clip_norm = self.config.get("grad_clip_norm", 1.0)

        return torch.optim.AdamW(
            self.model.parameters(), lr=self.config["lr"], weight_decay=weight_decay
        )

    def _setup_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        """Setup learning rate scheduler"""

        warmup_iters = self.config.get("warmup_iters", 1000)
        warmup_factor = self.config.get("warmup_factor", 1.0e-2)
        lr_scheduler = self.config.get("scheduler", "constant")
        train_iters = self.config["train_iters"]

        if lr_scheduler == "cosine":
            # Warmup phase: linearly ramp LR from warmup_factor * base_lr to base_lr
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=warmup_factor,
                end_factor=1.0,
                total_iters=warmup_iters,
            )
            # Cosine decay phase: covers steps after warmup
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_iters - warmup_iters,
                eta_min=1.0e-7,
            )
            # Chain warmup + cosine decay
            return optim.lr_scheduler.ChainedScheduler(
                [warmup_scheduler, cosine_scheduler]
            )
        elif lr_scheduler == "constant":
            # Default to constant with warmup
            return optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=warmup_factor, total_iters=warmup_iters
            )
        else:
            raise ValueError(f"Invalid LR scheduler: {lr_scheduler}")

    def _setup_dataloaders(self) -> None:
        # Get data loaders from config
        train_ds_config = self.config["dataset"]["train"]
        self.train_loader = data.get_train_dataloader(
            dataset=self.dataset_name,
            batch_size=train_ds_config["batchsize"],
            num_workers=train_ds_config["n_workers"],
        )
        assert len(self.train_loader) > 0

        val_ds_config = self.config["dataset"]["val"]
        self.val_loader = data.get_test_dataloader(
            dataset=self.dataset_name,
            batch_size=val_ds_config["batchsize"],
            num_workers=val_ds_config["n_workers"],
        )
        assert len(self.val_loader) > 0

    def _setup_metrics(self) -> None:
        assert self.val_loader is not None

        # Initialize TensorBoard writer
        self.log_dir = os.path.join(self.ckpt_dir_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.tb_meter = log_utils.AverageMeter()
        self.log_meter = log_utils.AverageMeter()

        self.tb_log_freq = self.config["training"].get("tb_log_freq", 20)
        self.log_freq = self.config["training"].get("log_freq", 200)

        self.log_sample_num_imgs = self.config["validation"].get(
            "log_sample_num_imgs", 10
        )

        self.train_step_timer = log_utils.Timer()
        self.val_step_timer = log_utils.Timer()

        self.val_fid_num_samples = self.config["fid"].get("num_samples", 5000)
        self.val_fid_sample_batch_size = self.config["fid"].get("batch_size", 5000)
        self.val_fid_cfg_scale = self.config["fid"].get("cfg_scale", 1.0)
        self.fid_calculator = fid_utils.FidCalculator(
            gt_loader=self.val_loader,
            device=self.device,
            num_samples=self.val_fid_num_samples,
            batch_size=self.val_fid_sample_batch_size,
        )

    def _train_step(self, batch: tuple) -> Dict[str, float]:
        """Execute one training step"""
        # Unpack the batch (images, labels)
        x_1, cls = batch

        # Move to device
        x_1 = x_1.to(self.device, non_blocking=True)
        cls = cls.to(self.device, non_blocking=True)

        # Sample noise and time steps
        t = self.train_time_sampler(x_1.shape[0])
        t = t.to(self.device, non_blocking=True)
        eps = torch.randn_like(x_1)

        self.optimizer.zero_grad()

        # Compute loss (mixed precision)
        with torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp_enabled
        ):
            loss, _ = self.loss_fn(
                model=self.model,
                x_1=x_1,
                eps=eps,
                t=t,
                cls=cls,
                cfg_dropout_prob=self.config["training"].get("cfg_dropout_prob", 0.1),
            )

        # Backward pass
        loss.backward()
        if self.grad_clip_norm is not None and self.grad_clip_norm > 0.0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip_norm
            )
        self.optimizer.step()
        current_lr = float(self.scheduler.get_last_lr()[0])
        self.scheduler.step()

        metrics = {_TRAIN_LOSS_KEY: loss.item(), _LR_KEY: current_lr}
        return metrics

    def _log_sample_images(self, step: int) -> None:
        """Generate and log sample images to TensorBoard."""
        print(f"Generating sample images at step {step}...")

        imgsize = self.config["dataset"]["val"]["imgsize"]

        # Randomly sample classes for each image
        cls_indices = torch.randint(
            0, self.num_classes, (self.log_sample_num_imgs,), device=self.device
        )

        cfg_scale = self.config["validation"].get("sample_cfg_scale", 2.0)

        samples = []
        for i in range(self.log_sample_num_imgs):
            cls_tensor = cls_indices[i : i + 1].long()
            shape = (1, 3, imgsize, imgsize)

            sample = self.val_ode_solver.sample(
                model=self.model,
                shape=shape,
                cls=cls_tensor,
                device=self.device,
                cfg_scale=cfg_scale,
            )
            sample = torch.clip(sample, 0.0, 1.0)
            samples.append(sample.cpu())

        samples = torch.cat(samples, dim=0)
        grid = tv_utils.make_grid(samples, nrow=5)
        self.writer.add_image("samples/img_grid", grid, step)

        print(f"Sample images logged at step {step}")

    def _validate(self, step: int) -> Dict[str, float]:
        """Execute validation loop"""
        self.model.eval()
        self._log_sample_images(step)

        print(f"Running validation for step: {step}...")

        val_meter = log_utils.AverageMeter()

        with torch.no_grad():
            self.val_step_timer.reset()
            self.val_step_timer.record()
            for batch in self.val_loader:
                x_1, cls = batch

                # Move to device
                x_1 = x_1.to(self.device, non_blocking=True)
                cls = cls.to(self.device, non_blocking=True)

                # Sample noise and time steps
                t = self.val_time_sampler(x_1.shape[0])
                t = t.to(self.device, non_blocking=True)
                eps = torch.randn_like(x_1)

                # Compute loss (no gradients needed, mixed precision)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.amp_enabled,
                ):
                    loss, _ = self.loss_fn(
                        model=self.model, x_1=x_1, eps=eps, t=t, cls=cls
                    )

                self.val_step_timer.record()

                val_meter.update({_VAL_LOSS_KEY: loss.item()})

        self._validated_step = step

        val_metrics = val_meter.get_average()

        val_batch_rate = self.val_step_timer.avg_rate()
        assert val_batch_rate is not None
        val_metrics[_VAL_STEPS_PER_SECOND_KEY] = val_batch_rate

        val_metrics[f"{_VAL_FID_KEY}_{self.val_fid_num_samples}-samples"] = (
            self.fid_calculator.compute_fid(
                model=self.model,
                device=self.device,
                solver=self.val_ode_solver,
                cfg_scale=self.val_fid_cfg_scale,
                num_classes=self.num_classes,
                imgsize=self.config["dataset"]["val"]["imgsize"],
            )
        )

        for metric, val in val_metrics.items():
            self.writer.add_scalar(metric, val, step)

        print(f"Validation Step {step}:")
        print(f"{val_metrics}")
        print("-" * 80)

        return val_metrics

    def train(self) -> None:
        """Execute the full training loop"""
        self.model.train()

        train_iter = iter(self.train_loader)

        num_train_iters = self.config["train_iters"]

        print("Start training!")
        print("-" * 80)

        self.train_step_timer.reset()
        self.train_step_timer.record()

        # Training loop over specified number of iterations
        for step in range(1, num_train_iters + 1):
            # Get a batch from the train loader
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reset iterator if needed (re-shuffle)
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            # Execute training step
            metrics = self._train_step(batch)

            self.train_step_timer.record()

            self.tb_meter.update(metrics)
            self.log_meter.update(metrics)

            # Log to TensorBoard
            if step % self.tb_log_freq == 0:
                tb_metrics = self.tb_meter.get_average()
                train_batch_rate = self.train_step_timer.avg_rate()
                assert train_batch_rate is not None
                tb_metrics[_TRAIN_STEPS_PER_SECOND_KEY] = train_batch_rate
                for metric, val in tb_metrics.items():
                    self.writer.add_scalar(metric, val, step)
                self.tb_meter.reset()
                self.train_step_timer.reset()

            # Log progress every few steps
            if step % self.log_freq == 0:
                log_metrics = self.log_meter.get_average()
                print(f"Step {step}:")
                for k, v in log_metrics.items():
                    print(f"{k}: {v:.5e}")
                print("-" * 80)
                self.log_meter.reset()

            # Validation and checkpoint at specified intervals
            if step > 0 and step % self.config["validation"]["val_iters"] == 0:
                self._validate(step)
                self.save_checkpoint(step)
                self.model.train()

        # Final validation run
        if self._validated_step == -1 or self._validated_step < num_train_iters:
            self._validate(num_train_iters)

    def save_checkpoint(self, step: int) -> None:
        """Save model checkpoint"""
        # Save model state dict
        checkpoint_state = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "step": step,
            "config": self.config,
        }
        torch.save(
            checkpoint_state,
            os.path.join(self.ckpt_dir_path, f"checkpoint_latest.pt"),
        )
        print(f"Checkpoint saved at step {step}")
