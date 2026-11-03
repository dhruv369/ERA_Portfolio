"""Train the exact additive-coupling reversible Session 13 model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

import session13_baseline as baseline

N_EMBED = baseline.N_EMBED
N_HEAD = baseline.N_HEAD
N_REVERSIBLE_BLOCKS = 10
SEQUENCE_LENGTH = baseline.SEQUENCE_LENGTH
TARGET_TRAIN_TOKENS = baseline.TARGET_TRAIN_TOKENS
SEED = baseline.SEED
CHECKPOINT_EVERY_STEPS = baseline.CHECKPOINT_EVERY_STEPS
EVALUATE_EVERY_STEPS = baseline.EVALUATE_EVERY_STEPS
EVALUATION_BATCHES = baseline.EVALUATION_BATCHES
LOG_EVERY_STEPS = baseline.LOG_EVERY_STEPS
LEARNING_RATE = baseline.LEARNING_RATE
WEIGHT_DECAY = baseline.WEIGHT_DECAY
GRAD_CLIP_NORM = baseline.GRAD_CLIP_NORM
DATASET_NAME = baseline.DATASET_NAME
TOKENIZER_NAME = baseline.TOKENIZER_NAME


class HalfSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = N_EMBED // 2
        self.n_head = N_HEAD
        self.head_dim = width // N_HEAD
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, width = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.proj(attended)


class ReversibleCouplingBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        width = N_EMBED // 2
        self.f_norm = nn.LayerNorm(width)
        self.f_attention = HalfSelfAttention()
        self.g_norm = nn.LayerNorm(width)
        self.g_mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.f_attention(self.f_norm(x))

    def g(self, x: torch.Tensor) -> torch.Tensor:
        return self.g_mlp(self.g_norm(x))

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        y1 = x1 + self.f(x2)
        y2 = x2 + self.g(y1)
        return torch.cat((y1, y2), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ReversibleBlockFunction.apply(x, self, *tuple(self.parameters()))


class ReversibleBlockFunction(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, block: ReversibleCouplingBlock, *params: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            y = block._forward(x)
        ctx.block = block
        ctx.params = params
        ctx.device_type = x.device.type
        ctx.autocast_enabled = torch.is_autocast_enabled(x.device.type)
        ctx.autocast_dtype = torch.get_autocast_dtype(x.device.type)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Any, grad_y: torch.Tensor) -> tuple[Any, ...]:
        (y,) = ctx.saved_tensors
        block: ReversibleCouplingBlock = ctx.block
        y1, y2 = y.chunk(2, dim=-1)
        with torch.no_grad(), torch.autocast(
            device_type=ctx.device_type,
            dtype=ctx.autocast_dtype,
            enabled=ctx.autocast_enabled,
        ):
            x2 = y2 - block.g(y1)
            x1 = y1 - block.f(x2)
            x = torch.cat((x1, x2), dim=-1).detach().requires_grad_(True)

        with torch.enable_grad(), torch.autocast(
            device_type=ctx.device_type,
            dtype=ctx.autocast_dtype,
            enabled=ctx.autocast_enabled,
        ):
            reconstructed_y = block._forward(x)
            gradients = torch.autograd.grad(
                reconstructed_y,
                (x, *ctx.params),
                grad_y,
                allow_unused=False,
            )
        return (gradients[0], None, *gradients[1:])


class ReversibleTinyStoriesLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, N_EMBED)
        self.blocks = nn.ModuleList(
            ReversibleCouplingBlock() for _ in range(N_REVERSIBLE_BLOCKS)
        )
        self.final_norm = nn.LayerNorm(N_EMBED)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, sequence = tokens.shape
        positions = torch.arange(sequence, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return F.linear(x, self.token_embedding.weight)


def train_one_step(
    model: ReversibleTinyStoriesLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    token_data: np.ndarray,
    token_count: int,
    eos_token_id: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    rng: random.Random,
) -> float:
    x, targets = baseline.make_batch(
        token_data, token_count, eos_token_id, device, rng
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=amp_dtype):
        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
    scaler.step(optimizer)
    scaler.update()
    return loss.item()


def batch_fits(
    batch_size: int,
    vocab_size: int,
    token_data: np.ndarray,
    eos_token_id: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_scaler: bool,
) -> bool:
    torch.cuda.empty_cache()
    try:
        model = ReversibleTinyStoriesLM(vocab_size).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        train_one_step(
            model,
            optimizer,
            scaler,
            token_data,
            batch_size * SEQUENCE_LENGTH,
            eos_token_id,
            device,
            amp_dtype,
            random.Random(SEED),
        )
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        if "scaler" in locals():
            del scaler
        if "optimizer" in locals():
            del optimizer
        if "model" in locals():
            del model
        torch.cuda.empty_cache()
        return False
    del scaler, optimizer, model
    torch.cuda.empty_cache()
    return True


def find_max_batch_size(
    start: int,
    vocab_size: int,
    token_data: np.ndarray,
    eos_token_id: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_scaler: bool,
) -> int:
    if start < 1:
        raise ValueError("The initial batch size must be at least one.")
    max_data_batch = (len(token_data) - 1) // SEQUENCE_LENGTH
    if start > max_data_batch:
        raise ValueError(
            f"Initial batch size {start} exceeds available training tokens "
            f"(maximum {max_data_batch})."
        )
    if not batch_fits(
        start, vocab_size, token_data, eos_token_id, device, amp_dtype, use_scaler
    ):
        raise RuntimeError(
            f"Reversible model does not fit even at batch size {start}; "
            "select a smaller initial batch size."
        )

    low = start
    high = min(start * 2, max_data_batch + 1)
    while high <= max_data_batch and batch_fits(
        high, vocab_size, token_data, eos_token_id, device, amp_dtype, use_scaler
    ):
        low, high = high, high * 2
        high = min(high, max_data_batch + 1)
        print(f"Batch size {low} fits; probing {high}...")
    if low == max_data_batch:
        print(f"Largest batch supported by the cached training tokens: {low}")
        return low

    while high - low > 1:
        middle = (low + high) // 2
        if batch_fits(
            middle,
            vocab_size,
            token_data,
            eos_token_id,
            device,
            amp_dtype,
            use_scaler,
        ):
            low = middle
        else:
            high = middle
    print(f"Largest tested reversible batch size: {low}")
    return low


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--find-max-batch", action="store_true")
    parser.add_argument("--target-train-tokens", type=int, default=TARGET_TRAIN_TOKENS)
    parser.add_argument("--run-name", default="reversible-fixed")
    args = parser.parse_args()
    if args.target_train_tokens < 1:
        raise ValueError("--target-train-tokens must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("The reversible training run requires a CUDA GPU.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = amp_dtype == torch.float16

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"{TOKENIZER_NAME} must provide an EOS token.")

    data_dir = baseline.DATA_DIR
    train_tokens = baseline.build_token_cache(
        "train",
        args.target_train_tokens,
        tokenizer,
        data_dir / "train_tokens.npy",
    )
    validation_tokens = baseline.build_token_cache(
        "validation",
        baseline.VALIDATION_TOKENS,
        tokenizer,
        data_dir / "validation_tokens.npy",
    )

    if args.find_max_batch:
        batch_size = find_max_batch_size(
            args.batch_size,
            tokenizer.vocab_size,
            train_tokens,
            tokenizer.eos_token_id,
            device,
            amp_dtype,
            use_scaler,
        )
    else:
        batch_size = args.batch_size
        if batch_size < 1:
            raise ValueError("--batch-size must be positive.")
    baseline.BATCH_SIZE = batch_size

    model = ReversibleTinyStoriesLM(tokenizer.vocab_size).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 19_000_000 <= parameter_count <= 21_000_000:
        raise RuntimeError(
            f"Expected approximately 20M parameters, got {parameter_count:,}."
        )
    print(
        f"GPU={gpu_name}; parameters={parameter_count:,}; batch={batch_size}; "
        f"sequence={SEQUENCE_LENGTH}; AMP={amp_dtype}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    run_dir = Path(
        os.environ.get(
            "S13_RUNS_DIR", baseline.SCRIPT_DIR / "runs"
        )
    ) / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_path = checkpoint_dir / "latest.pt"
    results_path = run_dir / "reversible_results.json"
    run_config: dict[str, Any] = {
        "run_name": args.run_name,
        "variant": "exact additive reversible coupling",
        "dataset": DATASET_NAME,
        "tokenizer": TOKENIZER_NAME,
        "target_train_tokens": args.target_train_tokens,
        "batch_size": batch_size,
        "sequence_length": SEQUENCE_LENGTH,
        "n_embed": N_EMBED,
        "reversible_blocks": N_REVERSIBLE_BLOCKS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
    }

    tokens_seen = 0
    step = 0
    train_seconds = 0.0
    train_rng = random.Random(SEED)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["run_config"] != run_config:
            raise RuntimeError(
                "The saved checkpoint configuration differs from this run. "
                f"Checkpoint: {checkpoint['run_config']}; current: {run_config}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        tokens_seen = checkpoint["tokens_seen"]
        step = checkpoint["step"]
        train_seconds = checkpoint["train_seconds"]
        train_rng.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in checkpoint["cuda_rng"]]
        )
        print(f"Resuming at step {step:,}, after {tokens_seen:,} training tokens.")

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    while tokens_seen < args.target_train_tokens:
        step_token_count = min(
            batch_size * SEQUENCE_LENGTH,
            args.target_train_tokens - tokens_seen,
        )
        start_time = time.perf_counter()
        loss = train_one_step(
            model,
            optimizer,
            scaler,
            train_tokens,
            step_token_count,
            tokenizer.eos_token_id,
            device,
            amp_dtype,
            train_rng,
        )
        torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - start_time
        tokens_seen += step_token_count
        step += 1

        if step % LOG_EVERY_STEPS == 0 or tokens_seen == args.target_train_tokens:
            print(
                f"step={step:,} tokens={tokens_seen:,}/"
                f"{args.target_train_tokens:,} train_loss={loss:.4f} "
                f"train_tokens_per_sec={tokens_seen / train_seconds:.1f}"
            )

        if step % EVALUATE_EVERY_STEPS == 0 or tokens_seen == args.target_train_tokens:
            val_loss = baseline.evaluate(
                model,
                validation_tokens,
                tokenizer.eos_token_id,
                device,
                amp_dtype,
            )
            print(f"validation_loss={val_loss:.4f}")

        if step % CHECKPOINT_EVERY_STEPS == 0 or tokens_seen == args.target_train_tokens:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = checkpoint_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "tokens_seen": tokens_seen,
                    "step": step,
                    "train_seconds": train_seconds,
                    "run_config": run_config,
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                temporary_path,
            )
            temporary_path.replace(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    final_val_loss = baseline.evaluate(
        model,
        validation_tokens,
        tokenizer.eos_token_id,
        device,
        amp_dtype,
    )
    result = {
        "status": "complete",
        "variant": run_config["variant"],
        "final_validation_loss": final_val_loss,
        "training_tokens": tokens_seen,
        "training_seconds_excluding_evaluation_and_checkpoints": train_seconds,
        "training_tokens_per_second": tokens_seen / train_seconds,
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "gpu": gpu_name,
        "pytorch": torch.__version__,
        "amp_dtype": str(amp_dtype),
        "parameters": parameter_count,
        "config": run_config,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(result, indent=2) + "\n")
    print("\nFinal reversible results:")
    print(json.dumps(result, indent=2))
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
