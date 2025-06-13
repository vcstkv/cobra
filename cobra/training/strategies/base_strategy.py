"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import random
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from cobra.models.vlms import CobraVLM
from cobra.overwatch import initialize_overwatch
from cobra.training.metrics import Metrics
from cobra.util import check_bloat16_supported
from cobra.util.batching_utils import SplitModalitySampler
from cobra.util.data_utils import PaddedCollatorForLanguageModeling, IGNORE_INDEX

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: CobraVLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id = vlm, device_id

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-Device Batch Size must evenly divide Global Batch Size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
        rewrite_name: bool = False
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
        val_ratio: float = 0.1
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            rng = random.Random(seed)
            modality_lengths = dataset.get_modality_lengths()
            
            total_size = len(dataset)
            val_cutoff = int(val_ratio*total_size)
            
            all_indices = list(range(total_size))
            val_indices = random.sample(all_indices, val_cutoff)
            train_indices = list(set(all_indices) - set(val_indices))
            
            val_modality = [modality_lengths[i] for i in val_indices]
            train_modality = [modality_lengths[i] for i in train_indices]

            valid_dataset = [dataset[i] for i in val_indices]
            train_dataset = [dataset[i] for i in train_indices]

            sampler = SplitModalitySampler(
                train_dataset,
                train_modality,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

            val_sampler = SplitModalitySampler(
                valid_dataset,
                val_modality,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed + 1,  # different seed for validation
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            train_dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=4,
            worker_init_fn=self.worker_init_fn,
        )

        val_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.per_device_batch_size,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=4,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        val_step = 5

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                true_label_size = 0
                self.best_val_loss = float('inf')  
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    num_valid_labels = torch.sum(batch["labels"] != IGNORE_INDEX).item()
                    true_label_size += num_valid_labels
                    sum_loss = loss * num_valid_labels
                    sum_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        for param in self.vlm.parameters():
                            if param.grad is not None:
                                param.grad.data /= true_label_size
                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        
                        true_label_size = 0
                        
                        if metrics.global_step % val_step == 0:
                            self.vlm.eval()  # Set model to eval mode
                            val_loss = 0.0
                            val_label_size = 0
                            with torch.no_grad():
                                for val_batch in val_dataloader:
                                    with torch.autocast(
                                        "cuda",
                                        dtype=self.mixed_precision_dtype,
                                        enabled=self.enable_mixed_precision_training,
                                    ):
                                        val_output: CausalLMOutputWithPast = self.vlm(
                                            input_ids=val_batch["input_ids"],
                                            attention_mask=val_batch["attention_mask"],
                                            pixel_values=val_batch["pixel_values"],
                                            labels=val_batch["labels"],
                                            multimodal_indices=val_batch["multimodal_indices"],
                                        )
                                        loss = val_output.loss

                                    num_valid_labels = torch.sum(val_batch["labels"] != IGNORE_INDEX).item()
                                    val_loss += loss.item() * num_valid_labels
                                    val_label_size += num_valid_labels

                            avg_val_loss = val_loss / max(val_label_size, 1)
                            metrics.commit(val_loss=avg_val_loss)
                            overwatch.info(f"Validation Loss: {avg_val_loss:.4f}")
                            if avg_val_loss < self.best_val_loss:
                                self.best_val_loss = avg_val_loss
                                overwatch.info(f"New best validation loss: {avg_val_loss:.4f}. Saving checkpoint...")
                                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, avg_val_loss, rewrite_name=True)
                                dist.barrier()
                            self.vlm.train()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()
