"""
PersonaMem GRPO Trainer (base model only)

Runs GRPO without loading any LoRA adapter by default.
"""

import argparse
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from personamem_grpo_trainer import (
    PersonaMemGRPOConfig,
    PersonaMemDataset,
    PersonaMemGRPOTrainer,
    QwenPolicy,
    MockQwenPolicy,
)


def _print_mem(label: str) -> None:
    if not (HAS_TORCH and torch.cuda.is_available()):
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    print(
        f"[Mem] {label}: alloc={allocated:.2f} MiB, reserved={reserved:.2f} MiB, "
        f"peak_alloc={peak_allocated:.2f} MiB, peak_reserved={peak_reserved:.2f} MiB",
        flush=True,
    )


class MemoryTrackingGRPOTrainer(PersonaMemGRPOTrainer):
    def train_step(self, samples):
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            _print_mem("before_rollout")
        try:
            metrics = super().train_step(samples)
        except RuntimeError:
            _print_mem("before_crash")
            raise
        _print_mem("after_train_step")
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PersonaMem GRPO Training (Base Model)")
    parser.add_argument("--data-dir", type=str, default="./data/personamem",
                        help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./output/personamem_grpo",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-4B-Instruct",
                        help="Model name or path")
    parser.add_argument("--lora-adapter", type=str, default="",
                        help="Path to LoRA adapter (optional)")
    parser.add_argument("--deepspeed-config", type=str, default="./deepspeed_grpo.json",
                        help="DeepSpeed config path (optional)")
    parser.add_argument("--device", type=str, default="cuda:7",
                        help="Device to run on (non-DeepSpeed)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit to save memory")
    parser.add_argument("--rollout-n", type=int, default=4,
                        help="Number of rollout paths")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum training steps")
    parser.add_argument("--learning-rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--use-cache", action="store_true",
                        help="Enable KV cache during generation")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum training samples (for testing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock policy (for testing without GPU)")

    args = parser.parse_args()

    deepspeed_config = args.deepspeed_config
    if isinstance(deepspeed_config, str) and deepspeed_config.lower() in {"", "none"}:
        deepspeed_config = None

    lora_adapter = args.lora_adapter.strip() or None

    config = PersonaMemGRPOConfig(
        train_path=f"{args.data_dir}/train_2irrel.jsonl",
        val_path=f"{args.data_dir}/val_2irrel.jsonl",
        model_name=args.model,
        lora_adapter_path=lora_adapter,
        deepspeed_config=deepspeed_config,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
        rollout_n=args.rollout_n,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_cache=args.use_cache,
        output_dir=args.output_dir,
    )

    print(f"PID: {os.getpid()}")
    print("Loading datasets...")
    train_dataset = PersonaMemDataset(config.train_path, max_samples=args.max_samples)
    val_dataset = PersonaMemDataset(
        config.val_path,
        max_samples=args.max_samples // 5 if args.max_samples else None,
    )

    if args.mock:
        print("Using mock policy (no GPU)")
        policy = MockQwenPolicy(config)
    else:
        print("Loading real policy model...")
        policy = QwenPolicy(config)

    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        _print_mem("after_model_load")

    print("Initializing trainer...", flush=True)
    trainer = MemoryTrackingGRPOTrainer(policy=policy, config=config)
    print("Trainer ready", flush=True)
    _print_mem("after_trainer_init")

    print("Starting training loop...", flush=True)
    trainer.train(train_dataset, val_dataset)


if __name__ == "__main__":
    main()
