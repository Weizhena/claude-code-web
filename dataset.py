"""
Dataset Processing Module for Mem-alpha

Handles loading and preprocessing of training/evaluation datasets:
- Accurate Retrieval (AR): SQuAD, HotpotQA, PerLTQA
- Test-Time Learning (TTL): TREC, NLU, PubMed, CLINIC, Banking77
- Long Range Understanding (LRU): BookSum, InfBench-Sum

Based on MemoryAgentBench (Hu et al., 2025)
"""

import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import json

from .memory_agent import ConversationChunk
from .evaluator import Question
from .rl_trainer import TrainingInstance


@dataclass
class DatasetConfig:
    """Configuration for dataset processing"""
    max_chunk_tokens: int = 2000  # Max tokens per chunk
    max_chunks_per_instance: int = 20  # Max chunks per instance
    max_questions_per_instance: int = 100  # Max questions per instance
    train_ratio: float = 0.8  # Train/validation split


class DatasetProcessor(ABC):
    """Abstract base class for dataset processors"""

    @abstractmethod
    def load(self, path: str) -> List[TrainingInstance]:
        """Load dataset from path"""
        pass

    @abstractmethod
    def preprocess(self, raw_data: Any) -> List[TrainingInstance]:
        """Preprocess raw data into training instances"""
        pass


class BookSumProcessor(DatasetProcessor):
    """
    Processor for BookSum dataset (Long Range Understanding)

    BookSum contains book chapters paired with summaries.
    We segment chapters into conversational chunks and evaluate
    on keyword extraction from summaries.

    Format:
    - Input: Book chapter text
    - Output: Summary
    - Evaluation: Keyword hit rate
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.keyword_extraction_prompt = """
Analyze the following book summary and extract the most important keywords.
Focus on:
1. Character names (main and supporting characters)
2. Key events and plot points
3. Important locations/settings
4. Central themes and concepts
5. Significant objects or symbols

Summary:
{summary}

Return ONLY a comma-separated list of keywords, nothing else.
"""

    def load(self, path: str) -> List[TrainingInstance]:
        """Load BookSum dataset"""
        instances = []

        # Load from JSON/JSONL file
        data_path = Path(path)
        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        elif data_path.suffix == '.jsonl':
            raw_data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data.append(json.loads(line))
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        return self.preprocess(raw_data)

    def preprocess(self, raw_data: List[Dict]) -> List[TrainingInstance]:
        """
        Preprocess BookSum data

        Each item has:
        - chapter: The book chapter text
        - summary: The ground truth summary
        """
        instances = []

        for idx, item in enumerate(raw_data):
            chapter = item.get('chapter', item.get('text', ''))
            summary = item.get('summary', '')

            if not chapter or not summary:
                continue

            # Segment chapter into chunks
            chunks = self._segment_into_chunks(chapter, idx)

            # Extract keywords from summary for evaluation
            keywords = self._extract_keywords(summary)

            # Create question for evaluation
            questions = [
                Question(
                    question_id=1,
                    question="Please provide a summary of what the user has been reading.",
                    answer=summary,
                    metadata={"keywords": keywords, "type": "summary"}
                )
            ]

            instance = TrainingInstance(
                instance_id=f"booksum_{idx}",
                chunks=chunks,
                questions=questions,
                total_tokens=len(chapter.split()),
                metadata={
                    "source": "BookSum",
                    "category": "LRU",
                    "keywords": keywords
                }
            )
            instances.append(instance)

        return instances

    def _segment_into_chunks(self, text: str, instance_id: int) -> List[ConversationChunk]:
        """Segment long text into conversational chunks"""
        chunks = []

        # Split by paragraphs or sentences
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_length = 0

        chunk_id = 1
        for para in paragraphs:
            para_tokens = len(para.split())

            if current_length + para_tokens > self.config.max_chunk_tokens:
                # Save current chunk
                if current_chunk:
                    chunk_content = self._format_as_conversation(
                        '\n\n'.join(current_chunk),
                        chunk_id
                    )
                    chunks.append(ConversationChunk(
                        chunk_id=chunk_id,
                        content=chunk_content,
                        timestamp=f"2024-01-{chunk_id:02d} 10:00",
                        metadata={"instance_id": instance_id}
                    ))
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0

            current_chunk.append(para)
            current_length += para_tokens

            if len(chunks) >= self.config.max_chunks_per_instance:
                break

        # Don't forget the last chunk
        if current_chunk and len(chunks) < self.config.max_chunks_per_instance:
            chunk_content = self._format_as_conversation(
                '\n\n'.join(current_chunk),
                chunk_id
            )
            chunks.append(ConversationChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                timestamp=f"2024-01-{chunk_id:02d} 10:00",
                metadata={"instance_id": instance_id}
            ))

        return chunks

    def _format_as_conversation(self, text: str, chunk_id: int) -> str:
        """Format text as a conversation chunk"""
        return f"""Event happened on 2024-01-{chunk_id:02d}. The user is reading a book.

<User>: {text}

<System>: Please remember what the user reads on 2024-01-{chunk_id:02d}, save the details within the book, and retain a summary of the book the user has read so far."""

    def _extract_keywords(self, summary: str) -> List[str]:
        """Extract keywords from summary"""
        # Simple keyword extraction using regex
        # In practice, would use LLM for this

        # Remove common words
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'although', 'though', 'this', 'that',
            'these', 'those', 'it', 'its', 'he', 'she', 'they', 'them',
            'his', 'her', 'their', 'who', 'which', 'what'
        }

        # Extract words (4+ characters)
        words = re.findall(r'\b[A-Za-z]{4,}\b', summary.lower())

        # Filter and get unique keywords
        keywords = []
        seen = set()
        for word in words:
            if word not in stopwords and word not in seen:
                keywords.append(word)
                seen.add(word)
                if len(keywords) >= 20:  # Limit keywords
                    break

        return keywords

    def compute_keyword_hit_rate(self,
                                 generated_summary: str,
                                 keywords: List[str]) -> float:
        """
        Compute keyword hit rate for evaluation

        r1 = (matched keywords) / (total keywords)
        """
        if not keywords:
            return 0.0

        generated_lower = generated_summary.lower()
        hits = sum(1 for kw in keywords if kw in generated_lower)
        return hits / len(keywords)


class InfBenchSumProcessor(DatasetProcessor):
    """
    Processor for InfBench-Sum dataset (Long Range Understanding)

    InfBench contains very long documents (up to 172k tokens)
    for summarization evaluation.
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()

    def load(self, path: str) -> List[TrainingInstance]:
        """Load InfBench-Sum dataset"""
        data_path = Path(path)
        raw_data = []

        if data_path.suffix == '.json':
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        elif data_path.suffix == '.jsonl':
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    raw_data.append(json.loads(line))

        return self.preprocess(raw_data)

    def preprocess(self, raw_data: List[Dict]) -> List[TrainingInstance]:
        """Preprocess InfBench-Sum data"""
        instances = []

        for idx, item in enumerate(raw_data):
            context = item.get('context', item.get('text', ''))
            summary = item.get('summary', item.get('answer', ''))

            if not context:
                continue

            # Segment into chunks
            chunks = self._segment_long_context(context, idx)

            # Create evaluation question
            questions = [
                Question(
                    question_id=1,
                    question="Summarize the main content of what the user has been reading.",
                    answer=summary,
                    metadata={"type": "summary"}
                )
            ]

            instance = TrainingInstance(
                instance_id=f"infbench_{idx}",
                chunks=chunks,
                questions=questions,
                total_tokens=len(context.split()),
                metadata={
                    "source": "InfBench-Sum",
                    "category": "LRU"
                }
            )
            instances.append(instance)

        return instances

    def _segment_long_context(self, text: str, instance_id: int) -> List[ConversationChunk]:
        """Segment very long context into manageable chunks"""
        chunks = []

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_length = 0
        chunk_id = 1

        for sentence in sentences:
            sentence_tokens = len(sentence.split())

            if current_length + sentence_tokens > self.config.max_chunk_tokens:
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(ConversationChunk(
                        chunk_id=chunk_id,
                        content=f"<User>: {chunk_text}",
                        timestamp=f"2024-01-01 {chunk_id:02d}:00",
                        metadata={"instance_id": instance_id}
                    ))
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_tokens

            if len(chunks) >= self.config.max_chunks_per_instance:
                break

        # Last chunk
        if current_chunk and len(chunks) < self.config.max_chunks_per_instance:
            chunk_text = ' '.join(current_chunk)
            chunks.append(ConversationChunk(
                chunk_id=chunk_id,
                content=f"<User>: {chunk_text}",
                timestamp=f"2024-01-01 {chunk_id:02d}:00",
                metadata={"instance_id": instance_id}
            ))

        return chunks


class SQuADProcessor(DatasetProcessor):
    """
    Processor for SQuAD dataset (Accurate Retrieval)

    SQuAD is a reading comprehension dataset with
    questions and answers about documents.
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()

    def load(self, path: str) -> List[TrainingInstance]:
        """Load SQuAD dataset"""
        with open(path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        return self.preprocess(raw_data)

    def preprocess(self, raw_data: Dict) -> List[TrainingInstance]:
        """Preprocess SQuAD data"""
        instances = []

        # SQuAD format: data -> list of articles -> list of paragraphs
        articles = raw_data.get('data', [])

        instance_id = 0
        for article in articles:
            paragraphs = article.get('paragraphs', [])

            # Combine paragraphs into chunks
            chunks = []
            questions = []
            chunk_id = 1

            for para in paragraphs:
                context = para.get('context', '')
                qas = para.get('qas', [])

                # Create chunk
                chunk = ConversationChunk(
                    chunk_id=chunk_id,
                    content=f"""Dialogue between User and Assistant on 2024-01-01 00:00:
<User>: I have some interesting updates for you:
{context}
<Assistant>: Understood. I'll keep these facts for future reference.""",
                    timestamp="2024-01-01 00:00",
                    metadata={"article": article.get('title', '')}
                )
                chunks.append(chunk)
                chunk_id += 1

                # Create questions
                for qa in qas:
                    q_text = qa.get('question', '')
                    answers = qa.get('answers', [])
                    if answers:
                        answer = answers[0].get('text', '')
                        questions.append(Question(
                            question_id=len(questions) + 1,
                            question=q_text,
                            answer=answer,
                            metadata={"type": "extractive"}
                        ))

                if len(chunks) >= self.config.max_chunks_per_instance:
                    break

            if chunks and questions:
                instance = TrainingInstance(
                    instance_id=f"squad_{instance_id}",
                    chunks=chunks,
                    questions=questions[:self.config.max_questions_per_instance],
                    metadata={
                        "source": "SQuAD",
                        "category": "AR",
                        "article": article.get('title', '')
                    }
                )
                instances.append(instance)
                instance_id += 1

        return instances


class TTLProcessor(DatasetProcessor):
    """
    Processor for Test-Time Learning datasets (NLU, TREC, etc.)

    These datasets contain classification examples that the model
    must learn from and apply to new instances.
    """

    def __init__(self,
                 dataset_name: str = "NLU",
                 config: Optional[DatasetConfig] = None):
        self.dataset_name = dataset_name
        self.config = config or DatasetConfig()

    def load(self, path: str) -> List[TrainingInstance]:
        """Load TTL dataset"""
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.json'):
                raw_data = json.load(f)
            else:
                raw_data = [json.loads(line) for line in f]
        return self.preprocess(raw_data)

    def preprocess(self, raw_data: List[Dict]) -> List[TrainingInstance]:
        """
        Preprocess TTL data

        Format classification examples as:
        <Sample: text; Label: class>
        """
        instances = []

        # Group examples by label for balanced chunks
        label_groups: Dict[str, List[Dict]] = {}
        for item in raw_data:
            label = str(item.get('label', item.get('class', 'unknown')))
            text = item.get('text', item.get('sentence', ''))
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append({'text': text, 'label': label})

        # Create training instances
        instance_id = 0

        # Combine examples into chunks
        all_examples = []
        for label, examples in label_groups.items():
            all_examples.extend(examples)

        random.shuffle(all_examples)

        # Create chunks with multiple examples each
        examples_per_chunk = 10
        chunks = []
        chunk_id = 1

        for i in range(0, len(all_examples), examples_per_chunk):
            chunk_examples = all_examples[i:i + examples_per_chunk]
            if not chunk_examples:
                continue

            # Format as conversation
            example_lines = []
            for ex in chunk_examples:
                example_lines.append(f"<Sample: {ex['text']}; Label: {ex['label']}>")

            chunk_content = f"""Dialogue between User and Assistant on 2024-01-01 00:00
<User>: The following are classification examples with their corresponding labels:
{chr(10).join(example_lines)}
<Assistant>: Great! I've added this to my knowledge base."""

            chunks.append(ConversationChunk(
                chunk_id=chunk_id,
                content=chunk_content,
                timestamp="2024-01-01 00:00"
            ))
            chunk_id += 1

            if len(chunks) >= self.config.max_chunks_per_instance:
                break

        # Create test questions (held-out examples)
        test_examples = all_examples[len(chunks) * examples_per_chunk:][:100]
        questions = []
        for idx, ex in enumerate(test_examples):
            questions.append(Question(
                question_id=idx + 1,
                question=f"What is the label for: '{ex['text']}'?",
                answer=ex['label'],
                metadata={"type": "classification"}
            ))

        if chunks and questions:
            instance = TrainingInstance(
                instance_id=f"{self.dataset_name.lower()}_{instance_id}",
                chunks=chunks,
                questions=questions,
                metadata={
                    "source": self.dataset_name,
                    "category": "TTL",
                    "num_labels": len(label_groups)
                }
            )
            instances.append(instance)

        return instances


class DatasetLoader:
    """
    Unified dataset loader for all dataset types
    """

    PROCESSORS = {
        "booksum": BookSumProcessor,
        "infbench": InfBenchSumProcessor,
        "squad": SQuADProcessor,
        "nlu": lambda config: TTLProcessor("NLU", config),
        "trec": lambda config: TTLProcessor("TREC", config),
        "clinic": lambda config: TTLProcessor("CLINIC", config),
        "banking77": lambda config: TTLProcessor("Banking77", config),
    }

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()

    def load_dataset(self,
                     name: str,
                     path: str) -> List[TrainingInstance]:
        """
        Load dataset by name

        Args:
            name: Dataset name (booksum, squad, nlu, etc.)
            path: Path to dataset file

        Returns:
            List of training instances
        """
        name_lower = name.lower()
        if name_lower not in self.PROCESSORS:
            raise ValueError(f"Unknown dataset: {name}. "
                           f"Available: {list(self.PROCESSORS.keys())}")

        processor_cls = self.PROCESSORS[name_lower]
        if callable(processor_cls) and not isinstance(processor_cls, type):
            processor = processor_cls(self.config)
        else:
            processor = processor_cls(self.config)

        return processor.load(path)

    def load_multiple(self,
                      datasets: Dict[str, str]) -> List[TrainingInstance]:
        """
        Load multiple datasets

        Args:
            datasets: Dict of {name: path}

        Returns:
            Combined list of training instances
        """
        all_instances = []
        for name, path in datasets.items():
            instances = self.load_dataset(name, path)
            all_instances.extend(instances)
            print(f"Loaded {len(instances)} instances from {name}")

        return all_instances

    def split_train_val(self,
                        instances: List[TrainingInstance],
                        train_ratio: float = 0.8) -> Tuple[List[TrainingInstance],
                                                            List[TrainingInstance]]:
        """Split instances into train and validation sets"""
        random.shuffle(instances)
        split_idx = int(len(instances) * train_ratio)
        return instances[:split_idx], instances[split_idx:]


def create_sample_lru_dataset() -> List[TrainingInstance]:
    """Create sample LRU dataset for testing"""
    instances = []

    # Sample BookSum-like instance
    sample_chapter = """
    Chapter 1: The Beginning

    It was a dark and stormy night when John first arrived at the old mansion.
    The building loomed before him, its Gothic spires reaching toward the clouded sky.
    He had inherited this place from his uncle, a man he barely knew.

    Inside, the air was thick with dust and memories. Portraits of ancestors lined
    the walls, their eyes seeming to follow his every movement. John felt a chill
    run down his spine as he explored the empty halls.

    In the library, he found an old journal. The pages were yellowed with age,
    but the writing was still legible. It spoke of a hidden treasure, buried
    somewhere on the grounds. John's heart raced with excitement.

    As night fell, strange sounds echoed through the mansion. Footsteps where
    no one walked. Whispers in empty rooms. John began to wonder if he was
    truly alone in this place.
    """

    sample_summary = """
    John inherits a Gothic mansion from his uncle. Upon arrival during a stormy night,
    he explores the dusty interior filled with ancestral portraits. In the library,
    he discovers an old journal mentioning hidden treasure on the grounds.
    As night falls, mysterious sounds suggest the mansion may be haunted.
    """

    processor = BookSumProcessor()

    # Create chunks manually for the sample
    chunks = [
        ConversationChunk(
            chunk_id=1,
            content=f"""Event happened on 2024-01-01. The user is reading a book.
<User>: {sample_chapter[:500]}
<System>: Please remember what the user reads.""",
            timestamp="2024-01-01 10:00"
        ),
        ConversationChunk(
            chunk_id=2,
            content=f"""Event happened on 2024-01-02. The user continues reading.
<User>: {sample_chapter[500:]}
<System>: Please remember what the user reads.""",
            timestamp="2024-01-02 10:00"
        )
    ]

    keywords = processor._extract_keywords(sample_summary)

    instance = TrainingInstance(
        instance_id="sample_lru_0",
        chunks=chunks,
        questions=[
            Question(
                question_id=1,
                question="Summarize what the user has been reading.",
                answer=sample_summary.strip(),
                metadata={"keywords": keywords, "type": "summary"}
            ),
            Question(
                question_id=2,
                question="Who is the main character in the story?",
                answer="John",
                metadata={"type": "extractive"}
            ),
            Question(
                question_id=3,
                question="What did John find in the library?",
                answer="an old journal",
                metadata={"type": "extractive"}
            )
        ],
        total_tokens=len(sample_chapter.split()),
        metadata={
            "source": "BookSum",
            "category": "LRU",
            "keywords": keywords
        }
    )
    instances.append(instance)

    return instances
