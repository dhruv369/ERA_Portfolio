"""Train the standard, non-reversible Session 13 baseline."""

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


SEED = 1337
DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "gpt2"
TARGET_TRAIN_TOKENS = 50_000_000
VALIDATION_TOKENS = 500_000
BATCH_SIZE = 16
SEQUENCE_LENGTH = 256
N_LAYER = 3
N_HEAD = 4
N_EMBED = 320
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP_NORM = 1.0
CHECKPOINT_EVERY_STEPS = 500
EVALUATE_EVERY_STEPS = 1_000
EVALUATION_BATCHES = 16
LOG_EVERY_STEPS = 100

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = Path(os.environ.get("S13_RUN_DIR", SCRIPT_DIR / "runs" / "baseline"))
DATA_DIR = Path(
    os.environ.get("S13_DATA_DIR", SCRIPT_DIR / "runs" / "data")
)
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "latest.pt"
RESULTS_PATH = RUN_DIR / "baseline_results.json"


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        head_dim = N_EMBED // N_HEAD
        self.qkv = nn.Linear(N_EMBED, 3 * N_EMBED)
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.n_head = N_HEAD
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, channels = x.shape
        qkv = self.qkv(x).view(batch, sequence, 3, self.n_head, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, channels)
        return self.proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(N_EMBED)
        self.attn = CausalSelfAttention()
        self.ln_2 = nn.LayerNorm(N_EMBED)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBED, 4 * N_EMBED),
            nn.GELU(),
            nn.Linear(4 * N_EMBED, N_EMBED),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class TinyStoriesLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding = nn.Embedding(SEQUENCE_LENGTH, N_EMBED)
        self.blocks = nn.Sequential(*(TransformerBlock() for _ in range(N_LAYER)))
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
        x = self.final_norm(self.blocks(x))
        return F.linear(x, self.token_embedding.weight)


def build_token_cache(
    split: str,
    token_limit: int,
    tokenizer: Any,
    cache_path: Path,
) -> np.ndarray:
    from datasets import load_dataset

    if cache_path.exists():
        cached = np.load(cache_path)
        if len(cached) != token_limit:
            raise RuntimeError(
                f"{cache_path} has {len(cached):,} tokens; expected {token_limit:,}. "
                "Remove only this stale cache file and rerun data preparation."
            )
        return cached

    print(f"Tokenizing TinyStories {split} split to {token_limit:,} tokens...")
    stream = load_dataset(DATASET_NAME, split=split, streaming=True)
    chunks: list[np.ndarray] = []
    collected = 0
    rows = iter(stream)
    while collected < token_limit:
        texts: list[str] = []
        for _ in range(512):
            row = next(rows, None)
            if row is None:
                break
            text = row.get("text", "")
            if text:
                texts.append(text)
        if not texts:
            break

        encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            ids.append(tokenizer.eos_token_id)
            take = min(len(ids), token_limit - collected)
            if take:
                chunks.append(np.asarray(ids[:take], dtype=np.uint16))
                collected += take
            if collected == token_limit:
                break

    if collected != token_limit:
        raise RuntimeError(
            f"TinyStories {split} provided only {collected:,} tokens; "
            f"{token_limit:,} were required."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f"{cache_path.stem}.tmp.npy")
    np.save(temporary_path, np.concatenate(chunks))
    temporary_path.replace(cache_path)
    print(f"Saved {collected:,} token IDs to {cache_path}")
    return np.load(cache_path)


def make_batch(
    token_data: np.ndarray,
    token_count: int,
    eos_token_id: int,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_count = math.ceil(token_count / SEQUENCE_LENGTH)
    x = torch.full(
        (batch_count, SEQUENCE_LENGTH),
        eos_token_id,
        dtype=torch.long,
        device=device,
    )
    targets = torch.full(
        (batch_count, SEQUENCE_LENGTH),
        -100,
        dtype=torch.long,
        device=device,
    )

    remaining = token_count
    for row in range(batch_count):
        length = min(SEQUENCE_LENGTH, remaining)
        start = rng.randrange(0, len(token_data) - length)
        example = np.asarray(token_data[start : start + length + 1], dtype=np.int64)
        x[row, :length] = torch.as_tensor(example[:-1], device=device)
        targets[row, :length] = torch.as_tensor(example[1:], device=device)
        remaining -= length

    return x, targets


def evaluate(
    model: TinyStoriesLM,
    validation_tokens: np.ndarray,
    eos_token_id: int,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    rng = random.Random(SEED + 1)
    loss_sum = 0.0
    counted_tokens = 0
    tokens_per_batch = min(
        BATCH_SIZE * SEQUENCE_LENGTH,
        len(validation_tokens) - 1,
    )
    if tokens_per_batch < 1:
        raise ValueError("The validation token cache must contain at least two tokens.")
    with torch.no_grad():
        for _ in range(EVALUATION_BATCHES):
            x, targets = make_batch(
                validation_tokens,
                tokens_per_batch,
                eos_token_id,
                device,
                rng,
            )
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            loss_sum += loss.item()
            counted_tokens += int((targets != -100).sum().item())
    model.train()
    return loss_sum / counted_tokens


def save_checkpoint(
    model: TinyStoriesLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    tokens_seen: int,
    step: int,
    train_seconds: float,
    run_config: dict[str, Any],
) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    temporary_path = CHECKPOINT_PATH.with_suffix(".tmp")
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
    temporary_path.replace(CHECKPOINT_PATH)


def main(smoke_only: bool = False) -> None:
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for the full baseline run. In Colab, select "
            "Runtime > Change runtime type > T4 GPU (or another GPU), then rerun."
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    print(f"Device: {gpu_name}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"{TOKENIZER_NAME} must provide an EOS token.")

    model = TinyStoriesLM(tokenizer.vocab_size).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 19_000_000 <= parameter_count <= 21_000_000:
        raise RuntimeError(f"Expected approximately 20M parameters, got {parameter_count:,}.")
    print(f"Parameters: {parameter_count:,}")

    amp_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    smoke_tokens = torch.randint(
        tokenizer.vocab_size, (2, 32), dtype=torch.long, device=device
    )
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
        smoke_logits = model(smoke_tokens)
    if smoke_logits.shape != (2, 32, tokenizer.vocab_size):
        raise RuntimeError(f"Unexpected smoke-test output shape: {smoke_logits.shape}")
    print(f"Smoke test passed: {tuple(smoke_logits.shape)}")

    if smoke_only:
        print("Smoke test complete. No dataset was downloaded and no training was started.")
        return

    train_tokens = build_token_cache(
        "train",
        TARGET_TRAIN_TOKENS,
        tokenizer,
        DATA_DIR / "train_tokens.npy",
    )
    validation_tokens = build_token_cache(
        "validation",
        VALIDATION_TOKENS,
        tokenizer,
        DATA_DIR / "validation_tokens.npy",
    )

    use_scaler = amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    run_config: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "tokenizer": TOKENIZER_NAME,
        "target_train_tokens": TARGET_TRAIN_TOKENS,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "n_embed": N_EMBED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seed": SEED,
    }

    tokens_seen = 0
    step = 0
    train_seconds = 0.0
    train_rng = random.Random(SEED)
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
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
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
        print(f"Resuming at step {step:,}, after {tokens_seen:,} training tokens.")

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    print(
        f"Starting baseline: batch={BATCH_SIZE}, sequence={SEQUENCE_LENGTH}, "
        f"target={TARGET_TRAIN_TOKENS:,} tokens, AMP={amp_dtype}."
    )

    while tokens_seen < TARGET_TRAIN_TOKENS:
        step_token_count = min(
            BATCH_SIZE * SEQUENCE_LENGTH,
            TARGET_TRAIN_TOKENS - tokens_seen,
        )
        step_start = time.perf_counter()
        x, targets = make_batch(
            train_tokens,
            step_token_count,
            tokenizer.eos_token_id,
            device,
            train_rng,
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
        torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - step_start

        tokens_seen += step_token_count
        step += 1
        if step % LOG_EVERY_STEPS == 0 or tokens_seen == TARGET_TRAIN_TOKENS:
            speed = tokens_seen / train_seconds
            print(
                f"step={step:,} tokens={tokens_seen:,}/{TARGET_TRAIN_TOKENS:,} "
                f"train_loss={loss.item():.4f} train_tokens_per_sec={speed:.1f}"
            )

        if step % EVALUATE_EVERY_STEPS == 0 or tokens_seen == TARGET_TRAIN_TOKENS:
            val_loss = evaluate(
                model,
                validation_tokens,
                tokenizer.eos_token_id,
                device,
                amp_dtype,
            )
            print(f"validation_loss={val_loss:.4f}")

        if step % CHECKPOINT_EVERY_STEPS == 0 or tokens_seen == TARGET_TRAIN_TOKENS:
            save_checkpoint(
                model,
                optimizer,
                scaler,
                tokens_seen,
                step,
                train_seconds,
                run_config,
            )
            print(f"Checkpoint saved: {CHECKPOINT_PATH}")

    peak_memory = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    final_val_loss = evaluate(
        model,
        validation_tokens,
        tokenizer.eos_token_id,
        device,
        amp_dtype,
    )
    result = {
        "status": "complete",
        "final_validation_loss": final_val_loss,
        "training_tokens": tokens_seen,
        "training_seconds_excluding_evaluation_and_checkpoints": train_seconds,
        "training_tokens_per_second": tokens_seen / train_seconds,
        "peak_cuda_memory_allocated_bytes": peak_memory,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "gpu": gpu_name,
        "pytorch": torch.__version__,
        "amp_dtype": str(amp_dtype),
        "parameters": parameter_count,
        "config": run_config,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print("\nFinal baseline results:")
    print(json.dumps(result, indent=2))
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Check the GPU, tokenizer, parameter count, and model output without training.",
    )
    main(smoke_only=parser.parse_args().smoke_only)
