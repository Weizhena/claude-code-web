"""
Training Script for Mem-alpha

Complete training pipeline implementing:
1. Dataset loading (LRU: BookSum, InfBench-Sum)
2. GRPO training with rollout sampling
3. Reward computation (r1, r2, r3, r4)
4. Checkpoint saving and evaluation

Usage:
    python -m mem_alpha.train --dataset booksum --epochs 3
"""

import argparse
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from .memory_structure import MemorySystem
from .memory_agent import MemoryConstructionAgent, ConversationChunk
from .evaluator import RewardCalculator, QAEvaluator, Question
from .rl_trainer import (
    GRPOTrainer,
    TrainingInstance,
    RolloutResult,
    PolicyModel,
    MockPolicyModel
)
from .dataset import (
    DatasetLoader,
    DatasetConfig,
    BookSumProcessor,
    create_sample_lru_dataset
)
from .config import MemAlphaConfig, DEFAULT_CONFIG


class TrainingPipeline:
    """
    Complete training pipeline for Mem-alpha

    Implements the training framework from the paper:
    1. Process chunks sequentially
    2. Agent decides memory operations
    3. Evaluate via RAG QA
    4. Compute rewards and update policy
    """

    def __init__(self,
                 config: Optional[MemAlphaConfig] = None,
                 output_dir: str = "./output"):
        self.config = config or DEFAULT_CONFIG
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.dataset_loader = DatasetLoader(DatasetConfig(
            max_chunk_tokens=self.config.training.max_context_length // 10,
            max_chunks_per_instance=20
        ))

        self.reward_calculator = RewardCalculator(
            beta=self.config.reward.beta,
            gamma=self.config.reward.gamma
        )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_validation_score = 0.0
        self.training_log: List[Dict] = []

    def setup_trainer(self, policy_model: PolicyModel) -> GRPOTrainer:
        """Setup GRPO trainer with given policy model"""
        return GRPOTrainer(
            policy_model=policy_model,
            reward_calculator=self.reward_calculator,
            config={
                "learning_rate": self.config.training.learning_rate,
                "batch_size": self.config.training.batch_size,
                "grpo_rollout_n": self.config.training.grpo_rollout_n,
                "clip_epsilon": self.config.training.clip_epsilon,
                "max_steps": self.config.training.max_steps,
                "beta": self.config.reward.beta,
                "gamma": self.config.reward.gamma
            }
        )

    def load_data(self,
                  train_path: Optional[str] = None,
                  val_path: Optional[str] = None,
                  dataset_name: str = "booksum") -> tuple:
        """
        Load training and validation data

        Args:
            train_path: Path to training data
            val_path: Path to validation data
            dataset_name: Name of dataset

        Returns:
            (train_instances, val_instances)
        """
        if train_path and os.path.exists(train_path):
            train_data = self.dataset_loader.load_dataset(dataset_name, train_path)
        else:
            # Use sample data for testing
            print("Using sample LRU dataset for demonstration...")
            train_data = create_sample_lru_dataset()

        if val_path and os.path.exists(val_path):
            val_data = self.dataset_loader.load_dataset(dataset_name, val_path)
        else:
            # Split train data
            train_data, val_data = self.dataset_loader.split_train_val(
                train_data,
                train_ratio=0.8
            )

        print(f"Loaded {len(train_data)} training instances")
        print(f"Loaded {len(val_data)} validation instances")

        return train_data, val_data

    def train_epoch(self,
                    trainer: GRPOTrainer,
                    train_data: List[TrainingInstance],
                    epoch: int) -> Dict:
        """
        Train for one epoch

        Args:
            trainer: GRPO trainer
            train_data: Training instances
            epoch: Current epoch number

        Returns:
            Epoch metrics
        """
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}")
        print(f"{'='*60}")

        epoch_metrics = {
            "epoch": epoch + 1,
            "steps": [],
            "mean_reward": 0.0,
            "mean_accuracy": 0.0
        }

        # Process batches
        batch_size = self.config.training.batch_size
        for batch_start in range(0, len(train_data), batch_size):
            batch_end = min(batch_start + batch_size, len(train_data))
            batch = train_data[batch_start:batch_end]

            # Training step
            step_metrics = trainer.train_step(batch)
            epoch_metrics["steps"].append(step_metrics)

            self.global_step += 1

            # Logging
            if self.global_step % 10 == 0:
                print(f"  Step {self.global_step}: "
                      f"loss={step_metrics['loss']:.4f}, "
                      f"reward={step_metrics['mean_reward']:.4f}, "
                      f"accuracy={step_metrics['mean_accuracy']:.4f}")

            # Check max steps
            if self.global_step >= self.config.training.max_steps:
                print(f"Reached max steps ({self.config.training.max_steps})")
                break

        # Compute epoch averages
        if epoch_metrics["steps"]:
            epoch_metrics["mean_reward"] = sum(
                s["mean_reward"] for s in epoch_metrics["steps"]
            ) / len(epoch_metrics["steps"])
            epoch_metrics["mean_accuracy"] = sum(
                s["mean_accuracy"] for s in epoch_metrics["steps"]
            ) / len(epoch_metrics["steps"])

        return epoch_metrics

    def evaluate(self,
                 trainer: GRPOTrainer,
                 val_data: List[TrainingInstance]) -> Dict:
        """
        Evaluate on validation set

        Args:
            trainer: GRPO trainer
            val_data: Validation instances

        Returns:
            Validation metrics
        """
        print("\nEvaluating on validation set...")
        return trainer.evaluate(val_data)

    def save_checkpoint(self,
                        trainer: GRPOTrainer,
                        epoch: int,
                        metrics: Dict) -> str:
        """Save training checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_epoch{epoch + 1}.json"
        trainer.save_checkpoint(str(checkpoint_path))

        # Save additional metrics
        metrics_path = self.output_dir / f"metrics_epoch{epoch + 1}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved checkpoint to {checkpoint_path}")
        return str(checkpoint_path)

    def run(self,
            policy_model: PolicyModel,
            train_path: Optional[str] = None,
            val_path: Optional[str] = None,
            dataset_name: str = "booksum",
            num_epochs: int = 3) -> Dict:
        """
        Run complete training pipeline

        Args:
            policy_model: Policy model to train
            train_path: Path to training data
            val_path: Path to validation data
            dataset_name: Dataset name
            num_epochs: Number of epochs

        Returns:
            Final training results
        """
        print("="*60)
        print("Mem-alpha Training Pipeline")
        print("="*60)
        print(f"Config:")
        print(f"  Learning rate: {self.config.training.learning_rate}")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  GRPO rollouts: {self.config.training.grpo_rollout_n}")
        print(f"  Max steps: {self.config.training.max_steps}")
        print(f"  Reward weights: beta={self.config.reward.beta}, gamma={self.config.reward.gamma}")

        # Load data
        train_data, val_data = self.load_data(train_path, val_path, dataset_name)

        # Setup trainer
        trainer = self.setup_trainer(policy_model)

        # Training loop
        results = {
            "epochs": [],
            "best_validation": None,
            "final_step": 0
        }

        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train epoch
            epoch_metrics = self.train_epoch(trainer, train_data, epoch)
            results["epochs"].append(epoch_metrics)

            # Validate
            val_metrics = self.evaluate(trainer, val_data)
            epoch_metrics["validation"] = val_metrics

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Training - reward: {epoch_metrics['mean_reward']:.4f}, "
                  f"accuracy: {epoch_metrics['mean_accuracy']:.4f}")
            print(f"  Validation - reward: {val_metrics['mean_reward']:.4f}, "
                  f"accuracy: {val_metrics['accuracy']:.4f}")

            # Save checkpoint if best
            if val_metrics["accuracy"] > self.best_validation_score:
                self.best_validation_score = val_metrics["accuracy"]
                results["best_validation"] = {
                    "epoch": epoch + 1,
                    "metrics": val_metrics
                }
                self.save_checkpoint(trainer, epoch, epoch_metrics)

            # Log
            self.training_log.append(epoch_metrics)

            # Check max steps
            if self.global_step >= self.config.training.max_steps:
                break

        results["final_step"] = self.global_step

        # Save final results
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total steps: {self.global_step}")
        print(f"Best validation accuracy: {self.best_validation_score:.4f}")
        print(f"Results saved to: {self.output_dir}")

        return results


def create_verl_compatible_config() -> Dict:
    """
    Create configuration compatible with verl framework
    (used in the original Mem-alpha implementation)
    """
    return {
        "model": {
            "name": "Qwen3-4B",
            "max_seq_len": 32000,
            "dtype": "bfloat16"
        },
        "training": {
            "learning_rate": 1e-6,
            "batch_size": 32,
            "gradient_accumulation_steps": 1,
            "max_steps": 205,
            "warmup_steps": 10,
            "weight_decay": 0.01
        },
        "grpo": {
            "rollout_n": 8,
            "clip_epsilon": 0.2,
            "kl_coef": 0.0,  # Disabled as per paper
            "entropy_coef": 0.01
        },
        "reward": {
            "r1_weight": 1.0,  # Accuracy
            "r2_weight": 1.0,  # Tool call
            "r3_weight": 0.05,  # Compression (beta)
            "r4_weight": 0.1   # Content quality (gamma)
        },
        "memory": {
            "core_max_tokens": 512,
            "categories": [
                "sensory_lifestyle",
                "culture_entertainment",
                "cognition_work",
                "values",
                "physiology_health",
                "resource_economic",
                "social_interpersonal",
                "spatiotemporal_context",
                "psychological_defense"
            ]
        },
        "rag": {
            "retriever": "BM25",
            "k_categories": 3,
            "n_entries_per_category": 5
        },
        "data": {
            "train_datasets": ["BookSum", "SQuAD", "HotpotQA", "NLU", "TREC"],
            "eval_datasets": ["MemoryAgentBench"],
            "max_instances": 562,  # Stratified sampling as per paper
            "max_chunk_tokens": 2000
        }
    }


def main():
    """Main entry point for training"""
    parser = argparse.ArgumentParser(description="Mem-alpha Training")
    parser.add_argument("--dataset", type=str, default="booksum",
                        choices=["booksum", "infbench", "squad", "nlu"],
                        help="Dataset to use for training")
    parser.add_argument("--train-path", type=str, default=None,
                        help="Path to training data")
    parser.add_argument("--val-path", type=str, default=None,
                        help="Path to validation data")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=205,
                        help="Maximum training steps")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--beta", type=float, default=0.05,
                        help="Compression reward weight")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Content quality reward weight")

    args = parser.parse_args()

    # Create config
    config = MemAlphaConfig()
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.max_steps = args.max_steps
    config.reward.beta = args.beta
    config.reward.gamma = args.gamma

    # Initialize pipeline
    pipeline = TrainingPipeline(
        config=config,
        output_dir=args.output_dir
    )

    # Create mock policy model for demonstration
    # In practice, would use actual LLM with verl framework
    policy_model = MockPolicyModel()

    # Run training
    results = pipeline.run(
        policy_model=policy_model,
        train_path=args.train_path,
        val_path=args.val_path,
        dataset_name=args.dataset,
        num_epochs=args.epochs
    )

    return results


if __name__ == "__main__":
    main()
