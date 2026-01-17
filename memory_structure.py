"""
Memory Structure for Mem-alpha with 9-Dimension Memory Framework
Based on: Mem-alpha: Learning Memory Construction via Reinforcement Learning

Memory Categories (from read.txt):
1. sensory_lifestyle - Daily sensory preferences and lifestyle choices
2. culture_entertainment - Cultural consumption and entertainment preferences
3. cognition_work - Cognitive style, professional skills, and work patterns
4. values - Core beliefs, personality traits, and life attitudes
5. physiology_health - Physical health, biorhythm, and body conditions
6. resource_economic - Financial status, spending habits, and asset allocation
7. social_interpersonal - Social patterns, relationships, and interaction styles
8. spatiotemporal_context - Location history, time patterns, and environment preferences
9. psychological_defense - Psychological boundaries, sensitive topics, and internal contradictions
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import json
import hashlib


class MemoryCategory(Enum):
    """9 Memory Categories based on user's framework"""
    SENSORY_LIFESTYLE = "sensory_lifestyle"
    CULTURE_ENTERTAINMENT = "culture_entertainment"
    COGNITION_WORK = "cognition_work"
    VALUES = "values"
    PHYSIOLOGY_HEALTH = "physiology_health"
    RESOURCE_ECONOMIC = "resource_economic"
    SOCIAL_INTERPERSONAL = "social_interpersonal"
    SPATIOTEMPORAL_CONTEXT = "spatiotemporal_context"
    PSYCHOLOGICAL_DEFENSE = "psychological_defense"

    @classmethod
    def get_description(cls, category: 'MemoryCategory') -> str:
        descriptions = {
            cls.SENSORY_LIFESTYLE: "Daily sensory preferences and lifestyle choices including food, travel, fashion",
            cls.CULTURE_ENTERTAINMENT: "Cultural consumption and entertainment preferences including movies, music, books, games",
            cls.COGNITION_WORK: "Cognitive style, professional skills, and work patterns",
            cls.VALUES: "Core beliefs, personality traits, and life attitudes",
            cls.PHYSIOLOGY_HEALTH: "Physical health, biorhythm, and body conditions",
            cls.RESOURCE_ECONOMIC: "Financial status, spending habits, and asset allocation",
            cls.SOCIAL_INTERPERSONAL: "Social patterns, relationships, and interaction styles",
            cls.SPATIOTEMPORAL_CONTEXT: "Location history, time patterns, and environment preferences",
            cls.PSYCHOLOGICAL_DEFENSE: "Psychological boundaries, sensitive topics, and internal contradictions"
        }
        return descriptions.get(category, "Unknown category")

    @classmethod
    def get_chinese_name(cls, category: 'MemoryCategory') -> str:
        names = {
            cls.SENSORY_LIFESTYLE: "sensory_lifestyle",
            cls.CULTURE_ENTERTAINMENT: "culture_entertainment",
            cls.COGNITION_WORK: "cognition_work",
            cls.VALUES: "values",
            cls.PHYSIOLOGY_HEALTH: "physiology_health",
            cls.RESOURCE_ECONOMIC: "resource_economic",
            cls.SOCIAL_INTERPERSONAL: "social_interpersonal",
            cls.SPATIOTEMPORAL_CONTEXT: "spatiotemporal_context",
            cls.PSYCHOLOGICAL_DEFENSE: "psychological_defense"
        }
        return names.get(category, "Unknown")


@dataclass
class MemoryEntry:
    """Single memory entry with ID and content"""
    id: int                           # Unique sequential ID
    category: MemoryCategory          # Memory category
    content: str                      # Memory content
    timestamp: Optional[str] = None   # Optional timestamp for episodic-like memories
    source_chunk_id: Optional[int] = None  # Which chunk this memory came from
    importance: float = 1.0           # Importance score (0-1)
    embedding: Optional[List[float]] = None  # Vector embedding for RAG
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "category": self.category.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "source_chunk_id": self.source_chunk_id,
            "importance": self.importance,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        return cls(
            id=data["id"],
            category=MemoryCategory(data["category"]),
            content=data["content"],
            timestamp=data.get("timestamp"),
            source_chunk_id=data.get("source_chunk_id"),
            importance=data.get("importance", 1.0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat())
        )

    def to_rag_format(self) -> str:
        """Convert to RAG-compatible format with ID"""
        return f"[{self.id}] [{self.category.value}] {self.content}"


@dataclass
class CoreMemory:
    """
    Core Memory: Persistent summary (max 512 tokens)
    Similar to original Mem-alpha, but categorized by our 9 dimensions
    """
    content: str = ""
    max_tokens: int = 512
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def update(self, new_content: str) -> None:
        self.content = new_content
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "max_tokens": self.max_tokens,
            "updated_at": self.updated_at
        }


@dataclass
class CategoryMemoryBank:
    """Memory bank for a single category"""
    category: MemoryCategory
    entries: List[MemoryEntry] = field(default_factory=list)
    next_id: int = 1  # Category-local ID counter

    def insert(self, content: str, timestamp: Optional[str] = None,
               source_chunk_id: Optional[int] = None,
               importance: float = 1.0,
               metadata: Optional[Dict] = None) -> MemoryEntry:
        """Insert new memory entry"""
        entry = MemoryEntry(
            id=self.next_id,
            category=self.category,
            content=content,
            timestamp=timestamp,
            source_chunk_id=source_chunk_id,
            importance=importance,
            metadata=metadata or {}
        )
        self.entries.append(entry)
        self.next_id += 1
        return entry

    def update(self, entry_id: int, new_content: str) -> Optional[MemoryEntry]:
        """Update existing memory entry"""
        for entry in self.entries:
            if entry.id == entry_id:
                entry.content = new_content
                entry.updated_at = datetime.now().isoformat()
                return entry
        return None

    def delete(self, entry_id: int) -> bool:
        """Delete memory entry by ID"""
        for i, entry in enumerate(self.entries):
            if entry.id == entry_id:
                self.entries.pop(i)
                return True
        return False

    def get_by_id(self, entry_id: int) -> Optional[MemoryEntry]:
        """Get memory entry by ID"""
        for entry in self.entries:
            if entry.id == entry_id:
                return entry
        return None

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries in this category"""
        return self.entries.copy()

    def to_rag_format(self) -> str:
        """Convert all entries to RAG format"""
        lines = []
        for entry in self.entries:
            lines.append(entry.to_rag_format())
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "entries": [e.to_dict() for e in self.entries],
            "next_id": self.next_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CategoryMemoryBank':
        bank = cls(category=MemoryCategory(data["category"]))
        bank.entries = [MemoryEntry.from_dict(e) for e in data.get("entries", [])]
        bank.next_id = data.get("next_id", len(bank.entries) + 1)
        return bank


@dataclass
class MemorySystem:
    """
    Complete Memory System with 9 categories + Core Memory

    Structure:
    - Core Memory: Global summary/understanding of user
    - 9 Category Banks: Each containing indexed memory entries
    """
    core_memory: CoreMemory = field(default_factory=CoreMemory)
    category_banks: Dict[MemoryCategory, CategoryMemoryBank] = field(default_factory=dict)
    global_id_counter: int = 0  # Global unique ID across all categories

    def __post_init__(self):
        # Initialize all category banks
        for category in MemoryCategory:
            if category not in self.category_banks:
                self.category_banks[category] = CategoryMemoryBank(category=category)

    def _get_global_id(self) -> int:
        """Generate global unique ID"""
        self.global_id_counter += 1
        return self.global_id_counter

    def memory_insert(self, category: MemoryCategory, content: str,
                      timestamp: Optional[str] = None,
                      source_chunk_id: Optional[int] = None,
                      importance: float = 1.0,
                      metadata: Optional[Dict] = None) -> MemoryEntry:
        """Insert new memory into specified category"""
        bank = self.category_banks[category]
        entry = bank.insert(
            content=content,
            timestamp=timestamp,
            source_chunk_id=source_chunk_id,
            importance=importance,
            metadata=metadata
        )
        # Assign global ID as well
        entry.metadata["global_id"] = self._get_global_id()
        return entry

    def memory_update(self, category: MemoryCategory, entry_id: int,
                      new_content: str) -> Optional[MemoryEntry]:
        """Update existing memory entry"""
        bank = self.category_banks[category]
        return bank.update(entry_id, new_content)

    def memory_delete(self, category: MemoryCategory, entry_id: int) -> bool:
        """Delete memory entry"""
        bank = self.category_banks[category]
        return bank.delete(entry_id)

    def core_memory_update(self, new_content: str) -> None:
        """Update core memory"""
        self.core_memory.update(new_content)

    def get_category_entries(self, category: MemoryCategory) -> List[MemoryEntry]:
        """Get all entries from a category"""
        return self.category_banks[category].get_all_entries()

    def get_all_entries(self) -> List[MemoryEntry]:
        """Get all entries from all categories"""
        all_entries = []
        for bank in self.category_banks.values():
            all_entries.extend(bank.get_all_entries())
        return all_entries

    def get_total_memory_length(self) -> int:
        """Get total character length of all memory"""
        total = len(self.core_memory.content)
        for bank in self.category_banks.values():
            for entry in bank.entries:
                total += len(entry.content)
        return total

    def get_memory_count(self) -> Dict[str, int]:
        """Get memory count per category"""
        counts = {"core_memory": 1 if self.core_memory.content else 0}
        for category, bank in self.category_banks.items():
            counts[category.value] = len(bank.entries)
        return counts

    def to_rag_format(self) -> str:
        """Convert entire memory system to RAG-compatible format"""
        sections = []

        # Core memory section
        if self.core_memory.content:
            sections.append(f"=== CORE MEMORY ===\n{self.core_memory.content}")

        # Category sections
        for category in MemoryCategory:
            bank = self.category_banks[category]
            if bank.entries:
                section_header = f"=== {category.value.upper()} ==="
                section_content = bank.to_rag_format()
                sections.append(f"{section_header}\n{section_content}")

        return "\n\n".join(sections)

    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "core_memory": self.core_memory.to_dict(),
            "category_banks": {
                cat.value: bank.to_dict()
                for cat, bank in self.category_banks.items()
            },
            "global_id_counter": self.global_id_counter
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemorySystem':
        """Deserialize from dictionary"""
        system = cls()

        if "core_memory" in data:
            system.core_memory = CoreMemory(
                content=data["core_memory"].get("content", ""),
                max_tokens=data["core_memory"].get("max_tokens", 512),
                updated_at=data["core_memory"].get("updated_at", datetime.now().isoformat())
            )

        if "category_banks" in data:
            for cat_value, bank_data in data["category_banks"].items():
                category = MemoryCategory(cat_value)
                system.category_banks[category] = CategoryMemoryBank.from_dict(bank_data)

        system.global_id_counter = data.get("global_id_counter", 0)
        return system

    def save(self, filepath: str) -> None:
        """Save memory system to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'MemorySystem':
        """Load memory system from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        counts = self.get_memory_count()
        return f"MemorySystem(entries={sum(counts.values())}, categories={counts})"
