"""
Inference Model Trainer for PersonaMem-v2

Trains a model to answer questions based on context (related + irrelevant).
Training format: (context, question) -> answer

Supports:
- Multiple choice format
- Direct answer generation
- GRPO/PPO reinforcement learning
"""

import json
import random
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from abc import ABC, abstractmethod

from .personamem_processor import (
    PersonaMemProcessor,
    InferenceTrainingSample,
    process_personamem_dataset
)


@dataclass
class InferenceTrainingConfig:
    """Configuration for inference model training"""
    # Data
    train_path: str = "./data/personamem/train_text.jsonl"
    val_path: str = "./data/personamem/val_text.jsonl"
    max_context_length: int = 4096

    # Model
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "cuda"

    # Training
    learning_rate: float = 1e-6
    batch_size: int = 8
    num_epochs: int = 3
    max_steps: int = 1000
    gradient_accumulation_steps: int = 4

    # GRPO
    grpo_rollout_n: int = 4
    clip_epsilon: float = 0.2

    # Task
    task_type: str = "multiple_choice"  # "multiple_choice" or "generation"
    num_choices: int = 4  # For multiple choice

    # Output
    output_dir: str = "./output/inference_model"
    save_every: int = 100


@dataclass
class InferenceRollout:
    """Result of a single inference rollout"""
    sample_id: str
    context: str
    question: str
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    reward: float
    log_prob: float = 0.0


@dataclass
class InferenceBatch:
    """A batch for training"""
    rollouts: List[InferenceRollout]
    mean_reward: float
    accuracy: float
    advantages: List[float]


class InferenceRewardCalculator:
    """Calculate rewards for inference task"""

    def __init__(self,
                 correct_reward: float = 1.0,
                 incorrect_reward: float = 0.0,
                 partial_match_weight: float = 0.3):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.partial_match_weight = partial_match_weight

    def calculate_reward(self,
                        predicted: str,
                        correct: str,
                        task_type: str = "multiple_choice") -> Tuple[bool, float]:
        """
        Calculate reward for prediction

        Args:
            predicted: Predicted answer
            correct: Correct answer
            task_type: "multiple_choice" or "generation"

        Returns:
            (is_correct, reward)
        """
        if task_type == "multiple_choice":
            # Exact match for multiple choice
            pred_clean = predicted.strip().upper()
            correct_clean = correct.strip().upper()

            # Check if answer letter matches
            if pred_clean and pred_clean[0] == correct_clean[0]:
                return True, self.correct_reward
            return False, self.incorrect_reward

        else:
            # Generation: partial matching
            pred_lower = predicted.lower().strip()
            correct_lower = correct.lower().strip()

            # Exact match
            if pred_lower == correct_lower:
                return True, self.correct_reward

            # Substring match
            if correct_lower in pred_lower or pred_lower in correct_lower:
                return True, self.correct_reward * 0.8

            # Token overlap
            pred_tokens = set(pred_lower.split())
            correct_tokens = set(correct_lower.split())

            if not correct_tokens:
                return False, self.incorrect_reward

            overlap = len(pred_tokens & correct_tokens) / len(correct_tokens)
            if overlap > 0.5:
                return True, self.correct_reward * overlap

            return False, self.incorrect_reward + overlap * self.partial_match_weight


class InferencePolicy(ABC):
    """Abstract base class for inference policy"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate answer given prompt"""
        pass

    @abstractmethod
    def get_log_prob(self, prompt: str, response: str) -> float:
        """Get log probability of response"""
        pass

    @abstractmethod
    def update(self, loss: float) -> None:
        """Update policy parameters"""
        pass


class MockInferencePolicy(InferencePolicy):
    """Mock policy for testing"""

    def __init__(self):
        self.update_count = 0

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate mock answer"""
        # Extract choices if present
        if "Choices:" in prompt:
            # Return random choice letter
            return random.choice(["A", "B", "C", "D"])
        return "Based on the context, the answer is..."

    def get_log_prob(self, prompt: str, response: str) -> float:
        return -1.0

    def update(self, loss: float) -> None:
        self.update_count += 1


class InferenceGRPOTrainer:
    """
    GRPO Trainer for Inference Model

    Trains model to select correct answer from context.
    """

    def __init__(self,
                 policy: InferencePolicy,
                 config: InferenceTrainingConfig):
        self.policy = policy
        self.config = config
        self.reward_calc = InferenceRewardCalculator()

        # Training state
        self.current_step = 0
        self.training_history: List[Dict] = []

    def load_data(self) -> Tuple[List[InferenceTrainingSample], List[InferenceTrainingSample]]:
        """Load training and validation data"""
        processor = PersonaMemProcessor()

        train_samples = processor.load_processed_data(self.config.train_path)
        val_samples = processor.load_processed_data(self.config.val_path)

        print(f"Loaded {len(train_samples)} training samples")
        print(f"Loaded {len(val_samples)} validation samples")

        return train_samples, val_samples

    def format_prompt(self, sample: InferenceTrainingSample) -> Tuple[str, str, List[str]]:
        """
        Format sample into prompt

        Returns:
            (prompt, correct_label, choices)
        """
        processor = PersonaMemProcessor()

        if self.config.task_type == "multiple_choice":
            formatted = processor.format_for_training(sample, include_choices=True)
            return (
                formatted["prompt"],
                formatted["correct_label"],
                formatted["choices"]
            )
        else:
            formatted = processor.format_for_training(sample, include_choices=False)
            return (
                formatted["prompt"],
                formatted["answer"],
                []
            )

    def collect_rollouts(self,
                        samples: List[InferenceTrainingSample],
                        n_rollouts: int = 1) -> List[InferenceRollout]:
        """Collect rollouts for a batch of samples"""
        rollouts = []

        for sample in samples:
            for _ in range(n_rollouts):
                prompt, correct_answer, choices = self.format_prompt(sample)

                # Generate prediction
                predicted = self.policy.generate(prompt)

                # Calculate reward
                is_correct, reward = self.reward_calc.calculate_reward(
                    predicted,
                    correct_answer,
                    self.config.task_type
                )

                # Get log prob
                log_prob = self.policy.get_log_prob(prompt, predicted)

                rollout = InferenceRollout(
                    sample_id=sample.sample_id,
                    context=sample.full_context,
                    question=sample.question,
                    correct_answer=correct_answer,
                    predicted_answer=predicted,
                    is_correct=is_correct,
                    reward=reward,
                    log_prob=log_prob
                )
                rollouts.append(rollout)

        return rollouts

    def compute_advantages(self, rollouts: List[InferenceRollout]) -> List[float]:
        """Compute group-relative advantages"""
        if not rollouts:
            return []

        rewards = [r.reward for r in rollouts]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = math.sqrt(variance) if variance > 0 else 1.0

        advantages = [(r - mean_reward) / (std_reward + 1e-8) for r in rewards]
        return advantages

    def compute_loss(self,
                    rollouts: List[InferenceRollout],
                    advantages: List[float],
                    old_log_probs: List[float]) -> float:
        """Compute GRPO loss"""
        total_loss = 0.0

        for rollout, advantage, old_log_prob in zip(rollouts, advantages, old_log_probs):
            # Compute ratio
            new_log_prob = rollout.log_prob
            ratio = math.exp(new_log_prob - old_log_prob)

            # Clipped objective
            clipped_ratio = max(
                min(ratio, 1 + self.config.clip_epsilon),
                1 - self.config.clip_epsilon
            )

            # Loss term
            loss_term = -min(ratio * advantage, clipped_ratio * advantage)
            total_loss += loss_term

        return total_loss / max(len(rollouts), 1)

    def train_step(self, samples: List[InferenceTrainingSample]) -> Dict:
        """Perform one training step"""
        # Collect rollouts
        rollouts = self.collect_rollouts(samples, self.config.grpo_rollout_n)

        # Get old log probs
        old_log_probs = [r.log_prob for r in rollouts]

        # Compute advantages
        advantages = self.compute_advantages(rollouts)

        # Compute loss
        loss = self.compute_loss(rollouts, advantages, old_log_probs)

        # Update policy
        self.policy.update(loss)

        # Metrics
        mean_reward = sum(r.reward for r in rollouts) / len(rollouts)
        accuracy = sum(1 for r in rollouts if r.is_correct) / len(rollouts)

        metrics = {
            "step": self.current_step,
            "loss": loss,
            "mean_reward": mean_reward,
            "accuracy": accuracy,
            "num_rollouts": len(rollouts)
        }

        self.training_history.append(metrics)
        self.current_step += 1

        return metrics

    def train(self,
             train_samples: List[InferenceTrainingSample],
             val_samples: Optional[List[InferenceTrainingSample]] = None) -> List[Dict]:
        """Full training loop"""
        print(f"Starting training...")
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Max steps: {self.config.max_steps}")

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Shuffle data
            shuffled = train_samples.copy()
            random.shuffle(shuffled)

            # Process batches
            for batch_start in range(0, len(shuffled), self.config.batch_size):
                if self.current_step >= self.config.max_steps:
                    print(f"Reached max steps ({self.config.max_steps})")
                    break

                batch_end = min(batch_start + self.config.batch_size, len(shuffled))
                batch = shuffled[batch_start:batch_end]

                metrics = self.train_step(batch)

                if self.current_step % 10 == 0:
                    print(f"  Step {metrics['step']}: "
                          f"loss={metrics['loss']:.4f}, "
                          f"reward={metrics['mean_reward']:.4f}, "
                          f"accuracy={metrics['accuracy']:.4f}")

                # Save checkpoint
                if self.current_step % self.config.save_every == 0:
                    self.save_checkpoint()

            # Validation
            if val_samples:
                val_metrics = self.evaluate(val_samples)
                print(f"  Validation: accuracy={val_metrics['accuracy']:.4f}")

        return self.training_history

    def evaluate(self, samples: List[InferenceTrainingSample]) -> Dict:
        """Evaluate on samples"""
        rollouts = self.collect_rollouts(samples, n_rollouts=1)

        accuracy = sum(1 for r in rollouts if r.is_correct) / len(rollouts)
        mean_reward = sum(r.reward for r in rollouts) / len(rollouts)

        return {
            "accuracy": accuracy,
            "mean_reward": mean_reward,
            "num_samples": len(samples)
        }

    def save_checkpoint(self) -> None:
        """Save training checkpoint"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "step": self.current_step,
            "config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "task_type": self.config.task_type
            },
            "history": self.training_history
        }

        checkpoint_path = output_dir / f"checkpoint_step{self.current_step}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        print(f"  Saved checkpoint to {checkpoint_path}")


def run_inference_training(
    data_dir: str = "./data/personamem",
    output_dir: str = "./output/inference_model",
    max_samples: int = 500,
    batch_size: int = 8,
    max_steps: int = 100
):
    """
    Run inference model training

    Args:
        data_dir: Directory with processed data
        output_dir: Output directory
        max_samples: Max samples to use
        batch_size: Training batch size
        max_steps: Maximum training steps
    """
    # Config
    config = InferenceTrainingConfig(
        train_path=f"{data_dir}/train_text.jsonl",
        val_path=f"{data_dir}/val_text.jsonl",
        batch_size=batch_size,
        max_steps=max_steps,
        output_dir=output_dir,
        task_type="multiple_choice"
    )

    # Create mock policy (replace with real model for actual training)
    policy = MockInferencePolicy()

    # Create trainer
    trainer = InferenceGRPOTrainer(policy=policy, config=config)

    # Load data
    train_samples, val_samples = trainer.load_data()

    # Limit samples
    if max_samples and len(train_samples) > max_samples:
        train_samples = random.sample(train_samples, max_samples)
    if max_samples and len(val_samples) > max_samples // 5:
        val_samples = random.sample(val_samples, max_samples // 5)

    # Train
    history = trainer.train(train_samples, val_samples)

    print("\n=== Training Complete ===")
    print(f"Total steps: {trainer.current_step}")
    if history:
        final_acc = history[-1]['accuracy']
        print(f"Final accuracy: {final_acc:.4f}")

    return history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train inference model on PersonaMem-v2")
    parser.add_argument("--data-dir", type=str, default="./data/personamem",
                       help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./output/inference_model",
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, default=500,
                       help="Max training samples")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--max-steps", type=int, default=100,
                       help="Max training steps")

    args = parser.parse_args()

    run_inference_training(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_steps=args.max_steps
    )
