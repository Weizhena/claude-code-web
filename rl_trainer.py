"""
Reinforcement Learning Training Framework for Mem-alpha

Implements GRPO (Group Relative Policy Optimization) for training
memory construction agents.

Based on the paper:
- Policy optimization using GRPO (Shao et al., 2024)
- Reward signal from downstream QA accuracy
- Multi-component reward: r1 (accuracy) + r2 (tool call) + beta*r3 (compression) + gamma*r4 (content)
"""

import math
import random
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json

from .memory_structure import MemorySystem, MemoryCategory
from .memory_agent import MemoryConstructionAgent, ConversationChunk, AgentAction
from .evaluator import QAEvaluator, RewardCalculator, Question, EvaluationResult
from .rag_retriever import TwoLayerRAGRetriever


@dataclass
class TrainingInstance:
    """A single training instance"""
    instance_id: str
    chunks: List[ConversationChunk]
    questions: List[Question]
    total_tokens: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    """Result of a single rollout (trajectory)"""
    instance_id: str
    actions: List[AgentAction]  # All actions taken across chunks
    memory_state: MemorySystem  # Final memory state
    rewards: List[float]  # Reward for each action
    total_reward: float
    qa_accuracy: float  # r1
    tool_call_success: float  # r2
    compression_ratio: float  # r3
    content_quality: float  # r4


@dataclass
class GRPOBatch:
    """A batch for GRPO training"""
    rollouts: List[RolloutResult]
    mean_reward: float
    std_reward: float
    advantages: List[List[float]]  # Advantage for each action in each rollout


class PolicyModel(ABC):
    """Abstract base class for policy model"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate response given prompt"""
        pass

    @abstractmethod
    def get_log_prob(self, prompt: str, response: str) -> float:
        """Get log probability of response given prompt"""
        pass

    @abstractmethod
    def update(self, loss: float) -> None:
        """Update model parameters"""
        pass


class GRPOTrainer:
    """
    Group Relative Policy Optimization (GRPO) Trainer

    GRPO optimizes the policy by:
    1. Sampling multiple rollouts per instance
    2. Computing group-relative advantages
    3. Updating policy with clipped objective

    Objective:
    J(theta) = E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]

    where:
    - ratio = pi_theta(a|s) / pi_old(a|s)
    - A = (r - mu_group) / (sigma_group + epsilon)
    """

    def __init__(self,
                 policy_model: PolicyModel,
                 reward_calculator: RewardCalculator,
                 config: Optional[Dict] = None):
        """
        Initialize GRPO trainer

        Args:
            policy_model: The policy model to train
            reward_calculator: Calculator for reward components
            config: Training configuration
        """
        self.policy = policy_model
        self.reward_calc = reward_calculator

        # Default config
        self.config = config or {}
        self.learning_rate = self.config.get("learning_rate", 1e-6)
        self.batch_size = self.config.get("batch_size", 32)
        self.rollout_n = self.config.get("grpo_rollout_n", 8)  # Number of rollouts per instance
        self.clip_epsilon = self.config.get("clip_epsilon", 0.2)
        self.max_steps = self.config.get("max_steps", 205)
        self.beta = self.config.get("beta", 0.05)  # Compression reward weight
        self.gamma = self.config.get("gamma", 0.1)  # Content reward weight

        # Training state
        self.current_step = 0
        self.training_history: List[Dict] = []

    def compute_advantages(self, rollouts: List[RolloutResult]) -> List[List[float]]:
        """
        Compute group-relative advantages for GRPO

        A_t = (r_t - mu_group) / (sigma_group + epsilon)

        Args:
            rollouts: List of rollout results

        Returns:
            List of advantage lists (one per rollout)
        """
        # Collect all rewards
        all_rewards = []
        for rollout in rollouts:
            all_rewards.extend(rollout.rewards)

        if not all_rewards:
            return [[] for _ in rollouts]

        # Compute group statistics
        mu_group = sum(all_rewards) / len(all_rewards)
        variance = sum((r - mu_group) ** 2 for r in all_rewards) / len(all_rewards)
        sigma_group = math.sqrt(variance) if variance > 0 else 1.0
        epsilon = 1e-8

        # Compute advantages
        advantages = []
        for rollout in rollouts:
            rollout_advantages = []
            for reward in rollout.rewards:
                advantage = (reward - mu_group) / (sigma_group + epsilon)
                rollout_advantages.append(advantage)
            advantages.append(rollout_advantages)

        return advantages

    def collect_rollouts(self,
                         instance: TrainingInstance,
                         n_rollouts: int) -> List[RolloutResult]:
        """
        Collect multiple rollouts for a single instance

        Args:
            instance: Training instance
            n_rollouts: Number of rollouts to collect

        Returns:
            List of rollout results
        """
        rollouts = []

        for i in range(n_rollouts):
            # Create fresh agent for each rollout
            agent = MemoryConstructionAgent(
                llm_callable=self.policy.generate
            )

            # Process all chunks
            all_actions = []
            for chunk in instance.chunks:
                actions = agent.process_chunk(chunk)
                all_actions.extend(actions)

            # Evaluate with QA
            evaluator = QAEvaluator(
                memory_system=agent.memory,
                llm_callable=self.policy.generate
            )
            qa_results = evaluator.evaluate_questions(instance.questions)

            # Calculate rewards
            total_input_length = sum(len(c.content) for c in instance.chunks)
            action_summary = agent.get_action_summary()

            eval_result = self.reward_calc.evaluate(
                memory_system=agent.memory,
                qa_results=qa_results,
                successful_tool_calls=action_summary['successful_actions'],
                total_tool_calls=action_summary['total_actions'],
                total_input_length=total_input_length
            )

            # Compute per-action rewards
            # Global rewards (r1, r3) are shared across all actions
            # Action-level rewards (r2, r4) are computed per action
            action_rewards = self._compute_action_rewards(
                all_actions, eval_result
            )

            rollout = RolloutResult(
                instance_id=instance.instance_id,
                actions=all_actions,
                memory_state=agent.memory,
                rewards=action_rewards,
                total_reward=eval_result.final_reward,
                qa_accuracy=eval_result.accuracy,
                tool_call_success=eval_result.tool_call_success_rate,
                compression_ratio=eval_result.compression_ratio,
                content_quality=eval_result.memory_content_score
            )
            rollouts.append(rollout)

        return rollouts

    def _compute_action_rewards(self,
                                actions: List[AgentAction],
                                eval_result: EvaluationResult) -> List[float]:
        """
        Compute rewards for each action

        r_t = r1 + r2_t + beta * r3 + gamma * r4_t

        where r1 and r3 are global, r2_t and r4_t are action-specific
        """
        if not actions:
            return []

        rewards = []
        for action in actions:
            # r2_t: Tool call success for this action
            r2_t = 1.0 if action.result and action.result.success else 0.0

            # r4_t: Memory content quality for this action
            # Simplified: assume valid if tool call succeeded
            r4_t = 1.0 if action.result and action.result.success else 0.0

            # Combined reward
            r_t = (eval_result.accuracy +  # r1 (global)
                   r2_t +  # r2_t (action-specific)
                   self.beta * eval_result.compression_ratio +  # r3 (global)
                   self.gamma * r4_t)  # r4_t (action-specific)

            rewards.append(r_t)

        return rewards

    def compute_policy_loss(self,
                            batch: GRPOBatch,
                            old_log_probs: List[List[float]]) -> float:
        """
        Compute GRPO policy loss

        L = -E[ min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) ]

        Args:
            batch: GRPO batch with rollouts and advantages
            old_log_probs: Log probabilities from old policy

        Returns:
            Policy loss
        """
        total_loss = 0.0
        total_terms = 0

        for rollout_idx, rollout in enumerate(batch.rollouts):
            advantages = batch.advantages[rollout_idx]

            for action_idx, action in enumerate(rollout.actions):
                if action_idx >= len(advantages):
                    continue

                advantage = advantages[action_idx]
                old_log_prob = old_log_probs[rollout_idx][action_idx]

                # Get new log prob (simplified - in practice would recompute)
                # For demonstration, assume ratio is close to 1
                new_log_prob = old_log_prob  # Placeholder

                # Compute ratio
                ratio = math.exp(new_log_prob - old_log_prob)

                # Clipped objective
                clipped_ratio = max(
                    min(ratio, 1 + self.clip_epsilon),
                    1 - self.clip_epsilon
                )

                # Loss term (negative because we maximize)
                loss_term = -min(ratio * advantage, clipped_ratio * advantage)
                total_loss += loss_term
                total_terms += 1

        return total_loss / max(total_terms, 1)

    def train_step(self, instances: List[TrainingInstance]) -> Dict:
        """
        Perform one training step

        Args:
            instances: List of training instances

        Returns:
            Training metrics for this step
        """
        all_rollouts = []
        all_old_log_probs = []

        # Collect rollouts for all instances
        for instance in instances:
            rollouts = self.collect_rollouts(instance, self.rollout_n)
            all_rollouts.extend(rollouts)

            # Get old log probs (simplified)
            for rollout in rollouts:
                log_probs = [0.0] * len(rollout.actions)  # Placeholder
                all_old_log_probs.append(log_probs)

        # Compute advantages
        advantages = self.compute_advantages(all_rollouts)

        # Create batch
        mean_reward = sum(r.total_reward for r in all_rollouts) / len(all_rollouts)
        rewards = [r.total_reward for r in all_rollouts]
        std_reward = math.sqrt(sum((r - mean_reward) ** 2 for r in rewards) / len(rewards))

        batch = GRPOBatch(
            rollouts=all_rollouts,
            mean_reward=mean_reward,
            std_reward=std_reward,
            advantages=advantages
        )

        # Compute loss
        loss = self.compute_policy_loss(batch, all_old_log_probs)

        # Update policy
        self.policy.update(loss)

        # Record metrics
        metrics = {
            "step": self.current_step,
            "loss": loss,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "mean_accuracy": sum(r.qa_accuracy for r in all_rollouts) / len(all_rollouts),
            "mean_compression": sum(r.compression_ratio for r in all_rollouts) / len(all_rollouts),
            "num_rollouts": len(all_rollouts)
        }

        self.training_history.append(metrics)
        self.current_step += 1

        return metrics

    def train(self,
              training_data: List[TrainingInstance],
              validation_data: Optional[List[TrainingInstance]] = None,
              num_epochs: int = 1) -> List[Dict]:
        """
        Full training loop

        Args:
            training_data: List of training instances
            validation_data: Optional validation instances
            num_epochs: Number of training epochs

        Returns:
            Training history
        """
        print(f"Starting GRPO training...")
        print(f"  Training instances: {len(training_data)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Rollouts per instance: {self.rollout_n}")
        print(f"  Max steps: {self.max_steps}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Shuffle training data
            shuffled_data = training_data.copy()
            random.shuffle(shuffled_data)

            # Process in batches
            for batch_start in range(0, len(shuffled_data), self.batch_size):
                if self.current_step >= self.max_steps:
                    print(f"Reached max steps ({self.max_steps})")
                    break

                batch_end = min(batch_start + self.batch_size, len(shuffled_data))
                batch_instances = shuffled_data[batch_start:batch_end]

                metrics = self.train_step(batch_instances)

                if self.current_step % 10 == 0:
                    print(f"  Step {metrics['step']}: "
                          f"loss={metrics['loss']:.4f}, "
                          f"reward={metrics['mean_reward']:.4f}, "
                          f"accuracy={metrics['mean_accuracy']:.4f}")

            # Validation
            if validation_data:
                val_metrics = self.evaluate(validation_data)
                print(f"  Validation: accuracy={val_metrics['accuracy']:.4f}, "
                      f"reward={val_metrics['mean_reward']:.4f}")

        return self.training_history

    def evaluate(self, instances: List[TrainingInstance]) -> Dict:
        """
        Evaluate on a set of instances

        Args:
            instances: List of instances to evaluate

        Returns:
            Evaluation metrics
        """
        all_results = []

        for instance in instances:
            # Single rollout for evaluation
            agent = MemoryConstructionAgent(
                llm_callable=self.policy.generate
            )

            for chunk in instance.chunks:
                agent.process_chunk(chunk)

            evaluator = QAEvaluator(
                memory_system=agent.memory,
                llm_callable=self.policy.generate
            )
            qa_results = evaluator.evaluate_questions(instance.questions)

            total_input = sum(len(c.content) for c in instance.chunks)
            action_summary = agent.get_action_summary()

            eval_result = self.reward_calc.evaluate(
                memory_system=agent.memory,
                qa_results=qa_results,
                successful_tool_calls=action_summary['successful_actions'],
                total_tool_calls=action_summary['total_actions'],
                total_input_length=total_input
            )
            all_results.append(eval_result)

        return {
            "accuracy": sum(r.accuracy for r in all_results) / len(all_results),
            "mean_reward": sum(r.final_reward for r in all_results) / len(all_results),
            "compression": sum(r.compression_ratio for r in all_results) / len(all_results),
            "tool_success": sum(r.tool_call_success_rate for r in all_results) / len(all_results)
        }

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint"""
        checkpoint = {
            "step": self.current_step,
            "config": self.config,
            "history": self.training_history
        }
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint"""
        with open(path, 'r') as f:
            checkpoint = json.load(f)
        self.current_step = checkpoint["step"]
        self.config = checkpoint["config"]
        self.training_history = checkpoint["history"]


class MockPolicyModel(PolicyModel):
    """Mock policy model for testing"""

    def __init__(self):
        self.update_count = 0

    def generate(self, prompt: str, max_tokens: int = 2048) -> str:
        """Generate mock response"""
        # Simple keyword-based mock generation
        if "memory" in prompt.lower():
            return json.dumps({
                "name": "memory_insert",
                "arguments": {
                    "category": "sensory_lifestyle",
                    "content": "Mock memory entry",
                    "importance": 0.8
                }
            })
        return '{"action": "skip"}'

    def get_log_prob(self, prompt: str, response: str) -> float:
        """Return mock log probability"""
        return -1.0  # Uniform probability

    def update(self, loss: float) -> None:
        """Mock update"""
        self.update_count += 1
