"""
Configuration for Mem-alpha

Contains all hyperparameters and settings for the memory system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class MemoryConfig:
    """Configuration for memory system"""

    # Core memory settings
    core_memory_max_tokens: int = 512

    # Memory categories (from read.txt)
    categories: List[str] = field(default_factory=lambda: [
        "sensory_lifestyle",
        "culture_entertainment",
        "cognition_work",
        "values",
        "physiology_health",
        "resource_economic",
        "social_interpersonal",
        "spatiotemporal_context",
        "psychological_defense"
    ])

    # Category descriptions
    category_descriptions: Dict[str, str] = field(default_factory=lambda: {
        "sensory_lifestyle": "Daily sensory preferences and lifestyle choices including food, travel, fashion",
        "culture_entertainment": "Cultural consumption and entertainment preferences including movies, music, books, games",
        "cognition_work": "Cognitive style, professional skills, and work patterns",
        "values": "Core beliefs, personality traits, and life attitudes",
        "physiology_health": "Physical health, biorhythm, and body conditions",
        "resource_economic": "Financial status, spending habits, and asset allocation",
        "social_interpersonal": "Social patterns, relationships, and interaction styles",
        "spatiotemporal_context": "Location history, time patterns, and environment preferences",
        "psychological_defense": "Psychological boundaries, sensitive topics, and internal contradictions"
    })


@dataclass
class RAGConfig:
    """Configuration for RAG retrieval"""

    # Layer 1: Category selection
    k_categories: int = 3  # Number of categories to select

    # Layer 2: Entry retrieval
    n_entries_per_category: int = 5  # Entries per category

    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Always include core memory
    use_core_memory: bool = True

    # Hybrid retrieval (embedding + BM25)
    use_hybrid: bool = False
    embedding_weight: float = 0.5


@dataclass
class RewardConfig:
    """Configuration for reward functions"""

    # Reward weights (from paper)
    # r1 (correctness) and r2 (tool call) weights are fixed at 1.0
    beta: float = 0.05  # Weight for compression reward (r3)
    gamma: float = 0.1  # Weight for memory content reward (r4)


@dataclass
class TrainingConfig:
    """Configuration for RL training"""

    # Model
    backbone_model: str = "Qwen3-4B"
    max_context_length: int = 32000

    # Training
    learning_rate: float = 1e-6
    batch_size: int = 32
    grpo_rollout_n: int = 8
    max_steps: int = 205

    # GRPO
    clip_epsilon: float = 0.2

    # Max response tokens for memory construction
    max_response_tokens: int = 2048


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""

    # QA evaluation
    max_qa_context_tokens: int = 4096

    # Metrics
    use_f1_score: bool = True
    use_exact_match: bool = True


@dataclass
class MemAlphaConfig:
    """Complete configuration for Mem-alpha system"""

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Paths
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MemAlphaConfig':
        """Create config from dictionary"""
        return cls(
            memory=MemoryConfig(**config_dict.get("memory", {})),
            rag=RAGConfig(**config_dict.get("rag", {})),
            reward=RewardConfig(**config_dict.get("reward", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            evaluation=EvaluationConfig(**config_dict.get("evaluation", {})),
            output_dir=config_dict.get("output_dir", "./output"),
            checkpoint_dir=config_dict.get("checkpoint_dir", "./checkpoints"),
            log_dir=config_dict.get("log_dir", "./logs")
        )

    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "memory": {
                "core_memory_max_tokens": self.memory.core_memory_max_tokens,
                "categories": self.memory.categories,
                "category_descriptions": self.memory.category_descriptions
            },
            "rag": {
                "k_categories": self.rag.k_categories,
                "n_entries_per_category": self.rag.n_entries_per_category,
                "bm25_k1": self.rag.bm25_k1,
                "bm25_b": self.rag.bm25_b,
                "use_core_memory": self.rag.use_core_memory,
                "use_hybrid": self.rag.use_hybrid,
                "embedding_weight": self.rag.embedding_weight
            },
            "reward": {
                "beta": self.reward.beta,
                "gamma": self.reward.gamma
            },
            "training": {
                "backbone_model": self.training.backbone_model,
                "max_context_length": self.training.max_context_length,
                "learning_rate": self.training.learning_rate,
                "batch_size": self.training.batch_size,
                "grpo_rollout_n": self.training.grpo_rollout_n,
                "max_steps": self.training.max_steps,
                "clip_epsilon": self.training.clip_epsilon,
                "max_response_tokens": self.training.max_response_tokens
            },
            "evaluation": {
                "max_qa_context_tokens": self.evaluation.max_qa_context_tokens,
                "use_f1_score": self.evaluation.use_f1_score,
                "use_exact_match": self.evaluation.use_exact_match
            },
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir
        }


# Default configuration
DEFAULT_CONFIG = MemAlphaConfig()
