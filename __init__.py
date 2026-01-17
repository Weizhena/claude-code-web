"""
Mem-alpha: Learning Memory Construction via Reinforcement Learning

A memory-augmented agent framework with 9-dimension memory categorization
and two-layer RAG retrieval.

Memory Categories:
1. sensory_lifestyle - Daily sensory preferences and lifestyle choices
2. culture_entertainment - Cultural consumption and entertainment preferences
3. cognition_work - Cognitive style, professional skills, and work patterns
4. values - Core beliefs, personality traits, and life attitudes
5. physiology_health - Physical health, biorhythm, and body conditions
6. resource_economic - Financial status, spending habits, and asset allocation
7. social_interpersonal - Social patterns, relationships, and interaction styles
8. spatiotemporal_context - Location history, time patterns, and environment preferences
9. psychological_defense - Psychological boundaries, sensitive topics, and contradictions
"""

from .memory_structure import (
    MemoryCategory,
    MemoryEntry,
    CoreMemory,
    CategoryMemoryBank,
    MemorySystem
)

from .memory_manager import (
    MemoryManager,
    ToolCallResult
)

from .rag_retriever import (
    BM25Scorer,
    TwoLayerRAGRetriever,
    HybridRetriever,
    RetrievalResult
)

from .memory_agent import (
    ConversationChunk,
    AgentAction,
    MemoryConstructionAgent,
    MockLLM
)

from .evaluator import (
    Question,
    QAResult,
    EvaluationResult,
    QAEvaluator,
    RewardCalculator,
    MockQALLM
)

from .config import (
    MemoryConfig,
    RAGConfig,
    RewardConfig,
    TrainingConfig,
    EvaluationConfig,
    MemAlphaConfig,
    DEFAULT_CONFIG
)

from .rl_trainer import (
    TrainingInstance,
    RolloutResult,
    GRPOBatch,
    PolicyModel,
    GRPOTrainer,
    MockPolicyModel
)

from .dataset import (
    DatasetConfig,
    DatasetProcessor,
    BookSumProcessor,
    InfBenchSumProcessor,
    SQuADProcessor,
    TTLProcessor,
    DatasetLoader,
    create_sample_lru_dataset
)

from .personamem_processor import (
    PersonaMemProcessor,
    InferenceTrainingSample,
    RetrievedItem,
    process_personamem_dataset
)

from .personamem_grpo_trainer import (
    PersonaMemGRPOConfig,
    PersonaMemGRPOTrainer,
    PersonaMemSample,
    PersonaMemDataset,
    QwenPolicy,
    MockQwenPolicy,
    OutputParser,
    PersonaMemRewardCalculator
)

__version__ = "0.1.0"
__author__ = "Mem-alpha Implementation"

__all__ = [
    # Memory Structure
    "MemoryCategory",
    "MemoryEntry",
    "CoreMemory",
    "CategoryMemoryBank",
    "MemorySystem",
    # Memory Manager
    "MemoryManager",
    "ToolCallResult",
    # RAG Retriever
    "BM25Scorer",
    "TwoLayerRAGRetriever",
    "HybridRetriever",
    "RetrievalResult",
    # Memory Agent
    "ConversationChunk",
    "AgentAction",
    "MemoryConstructionAgent",
    "MockLLM",
    # Evaluator
    "Question",
    "QAResult",
    "EvaluationResult",
    "QAEvaluator",
    "RewardCalculator",
    "MockQALLM",
    # Config
    "MemoryConfig",
    "RAGConfig",
    "RewardConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "MemAlphaConfig",
    "DEFAULT_CONFIG",
    # RL Trainer
    "TrainingInstance",
    "RolloutResult",
    "GRPOBatch",
    "PolicyModel",
    "GRPOTrainer",
    "MockPolicyModel",
    # Dataset
    "DatasetConfig",
    "DatasetProcessor",
    "BookSumProcessor",
    "InfBenchSumProcessor",
    "SQuADProcessor",
    "TTLProcessor",
    "DatasetLoader",
    "create_sample_lru_dataset",
    # PersonaMem Processor
    "PersonaMemProcessor",
    "InferenceTrainingSample",
    "RetrievedItem",
    "process_personamem_dataset",
    # PersonaMem GRPO Trainer
    "PersonaMemGRPOConfig",
    "PersonaMemGRPOTrainer",
    "PersonaMemSample",
    "PersonaMemDataset",
    "QwenPolicy",
    "MockQwenPolicy",
    "OutputParser",
    "PersonaMemRewardCalculator"
]
