"""
PersonaMem-v2 Dataset Processor

Processes PersonaMem-v2 dataset for inference model training.

Training format simulates RAG retrieval scenario:
- retrieved_context contains MIXED relevant and irrelevant content
- Example: 1. relevant, 2. relevant, 3. irrelevant, 4. irrelevant
- This mimics real RAG impurity where retrieval results contain noise

Dataset fields:
- related_conversation_snippet: Clue/related context
- user_query: Question (with role and content)
- correct_answer: Correct answer
- incorrect_answers: Wrong answers (for multiple choice)
"""

import json
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RetrievedItem:
    """A single retrieved item (can be relevant or irrelevant)"""
    index: int
    content: str
    is_relevant: bool
    source_id: str  # Which sample this came from


@dataclass
class InferenceTrainingSample:
    """
    A single training sample for inference model

    Simulates RAG retrieval scenario where retrieved_context
    contains a mix of relevant and irrelevant items.
    """
    sample_id: str

    # Retrieved context - mixed relevant + irrelevant (simulates RAG)
    retrieved_context: str  # Formatted as numbered list: 1. xxx 2. xxx 3. xxx 4. xxx

    # Metadata about what's relevant
    num_relevant: int       # Number of relevant items
    num_irrelevant: int     # Number of irrelevant items
    relevant_indices: List[int]    # Which indices are relevant (e.g., [1, 2])
    irrelevant_indices: List[int]  # Which indices are irrelevant (e.g., [3, 4])

    # Question and answer
    question: str
    correct_answer: str
    incorrect_answers: List[str]

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "sample_id": self.sample_id,
            "retrieved_context": self.retrieved_context,
            "num_relevant": self.num_relevant,
            "num_irrelevant": self.num_irrelevant,
            "relevant_indices": self.relevant_indices,
            "irrelevant_indices": self.irrelevant_indices,
            "question": self.question,
            "correct_answer": self.correct_answer,
            "incorrect_answers": self.incorrect_answers,
            "metadata": self.metadata
        }


class PersonaMemProcessor:
    """
    Process PersonaMem-v2 dataset for inference training

    Simulates RAG retrieval scenario:
    - Each turn in related_conversation_snippet becomes a separate relevant item
    - Additional irrelevant items are added from other samples
    - Format: 1. [content] 2. [content] ... N. [content]

    Structure per sample:
    - K relevant items (one per conversation turn in the clue)
    - N irrelevant items (noise from other samples)

    This trains the model to:
    1. Identify the relevant information among noisy retrieval results
    2. Answer questions based on the relevant context while ignoring noise
    """

    def __init__(self,
                 max_context_length: int = 8192,
                 num_irrelevant: int = 2,      # Number of irrelevant items to add
                 num_incorrect_answers: int = 3,
                 shuffle_order: bool = True):  # Whether to shuffle item order
        """
        Args:
            max_context_length: Maximum context length in characters
            num_irrelevant: Number of irrelevant context items to add
            num_incorrect_answers: Number of incorrect answers to keep
            shuffle_order: Whether to shuffle the order of items
        """
        self.max_context_length = max_context_length
        self.num_irrelevant = num_irrelevant
        self.num_incorrect_answers = num_incorrect_answers
        self.shuffle_order = shuffle_order

    def load_dataset(self, split: str = "train_text") -> List[Dict]:
        """Load PersonaMem-v2 dataset from HuggingFace"""
        try:
            from datasets import load_dataset
            ds = load_dataset("bowen-upenn/PersonaMem-v2")
            return list(ds[split])
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []

    def _format_conversation_snippet(self, snippet: Any) -> str:
        """Format conversation snippet to string (legacy method for compatibility)"""
        turns = self._parse_conversation_turns(snippet)
        return " ".join(turns)

    def _parse_conversation_turns(self, snippet: Any) -> List[str]:
        """
        Parse conversation snippet into individual turns

        Each turn becomes a separate relevant record.
        Returns a list of formatted turn strings.
        """
        if snippet is None:
            return []

        if isinstance(snippet, str):
            try:
                snippet = json.loads(snippet)
            except:
                # If it's a plain string, return it as a single turn
                return [snippet] if snippet.strip() else []

        if isinstance(snippet, list):
            turns = []
            for turn in snippet:
                if isinstance(turn, dict):
                    role = turn.get("role", "unknown")
                    content = turn.get("content", "")
                    if content.strip():
                        turns.append(f"[{role.capitalize()}]: {content}")
            return turns

        # Fallback: return as single turn
        return [str(snippet)] if str(snippet).strip() else []

    def _extract_question(self, user_query: Any) -> str:
        """Extract question text from user_query field"""
        if isinstance(user_query, dict):
            content = user_query.get("content", "")
            if content:
                return content
            return str(user_query)
        if isinstance(user_query, str):
            try:
                parsed = json.loads(user_query)
                if isinstance(parsed, dict):
                    return parsed.get("content", user_query)
            except:
                pass
        return str(user_query)

    def _parse_incorrect_answers(self, incorrect_answers: Any) -> List[str]:
        """
        Parse incorrect_answers field which may be a JSON string or list

        Args:
            incorrect_answers: Raw incorrect_answers from dataset (could be JSON string or list)

        Returns:
            List of incorrect answer strings
        """
        if not incorrect_answers:
            return []

        # If it's a string, try to parse as JSON
        if isinstance(incorrect_answers, str):
            try:
                parsed = json.loads(incorrect_answers)
                if isinstance(parsed, list):
                    return [str(a) for a in parsed if a]
            except json.JSONDecodeError:
                # Not valid JSON, return as single item
                return [incorrect_answers] if incorrect_answers.strip() else []

        # If it's already a list
        if isinstance(incorrect_answers, list):
            return [str(a) for a in incorrect_answers if a]

        return []

    def _select_incorrect_answers(self,
                                  incorrect_answers: Any,
                                  n: int = 3) -> List[str]:
        """Select n incorrect answers for multiple choice"""
        # First parse the incorrect answers
        parsed = self._parse_incorrect_answers(incorrect_answers)

        if not parsed:
            return []

        if len(parsed) > n:
            return random.sample(parsed, n)
        return parsed

    def _get_irrelevant_turns(self,
                              all_turns: List[List[str]],
                              current_idx: int,
                              n: int) -> List[str]:
        """
        Get n irrelevant individual turns from other samples

        Args:
            all_turns: List of turn lists for each sample
            current_idx: Current sample index (to exclude)
            n: Number of irrelevant turns to get

        Returns:
            List of n irrelevant turn strings
        """
        # Collect all available turns from other samples
        available_turns = []
        for i, turns in enumerate(all_turns):
            if i != current_idx:
                for turn in turns:
                    if turn and len(turn) > 30:  # Filter out very short turns
                        available_turns.append(turn)

        if not available_turns:
            return []

        # Randomly select n turns
        return random.sample(available_turns, min(n, len(available_turns)))

    def _format_retrieved_context(self,
                                  items: List[RetrievedItem]) -> str:
        """
        Format retrieved items as numbered list

        Example output:
        1. [User]: I enjoy herbal tea... [Assistant]: That sounds relaxing...
        2. [User]: My work involves data analysis... [Assistant]: What tools do you use?
        3. [User]: I love exploring coffee shops... [Assistant]: Great way to discover...
        4. [User]: Can you help with my post? [Assistant]: Sure, here's the improved version...
        """
        lines = []
        for item in items:
            lines.append(f"{item.index}. {item.content}")
        return "\n\n".join(lines)

    def process(self, raw_data: List[Dict]) -> List[InferenceTrainingSample]:
        """
        Process raw data into training samples

        IMPORTANT: Each turn in related_conversation_snippet becomes a SEPARATE
        relevant item. All turns from the clue are marked as relevant.
        Then irrelevant turns are added from other samples.

        Structure:
        - K relevant items: one per turn in related_conversation_snippet
        - N irrelevant items: individual turns from other samples

        Example: If clue has 4 turns and num_irrelevant=2:
        - relevant_indices might be [1, 3, 5, 6] (4 relevant turns)
        - irrelevant_indices might be [2, 4] (2 irrelevant turns)
        """
        # First pass: parse all snippets into individual turns
        all_turns: List[List[str]] = []
        for item in raw_data:
            turns = self._parse_conversation_turns(
                item.get("related_conversation_snippet")
            )
            all_turns.append(turns)

        # Second pass: create training samples
        samples = []
        for idx, item in enumerate(raw_data):
            # Get all relevant turns from this sample's clue
            relevant_turns = all_turns[idx]
            num_relevant = len(relevant_turns)

            if num_relevant == 0:
                # Skip samples with no relevant content
                continue

            # Get irrelevant turns from other samples
            irrelevant_turns = self._get_irrelevant_turns(
                all_turns, idx, self.num_irrelevant
            )

            # Pad if not enough irrelevant (ensure minimum)
            while len(irrelevant_turns) < self.num_irrelevant:
                irrelevant_turns.append("[No additional context available]")

            # Create retrieved items
            items: List[RetrievedItem] = []

            # Add ALL relevant turns (each is a separate item)
            for i, turn in enumerate(relevant_turns):
                items.append(RetrievedItem(
                    index=0,  # Will be assigned after shuffle
                    content=turn,
                    is_relevant=True,
                    source_id=f"sample_{idx}_turn_{i}"
                ))

            # Add irrelevant turns
            for i, turn in enumerate(irrelevant_turns[:self.num_irrelevant]):
                items.append(RetrievedItem(
                    index=0,
                    content=turn,
                    is_relevant=False,
                    source_id=f"other_{i}"
                ))

            # Shuffle if requested (mix relevant and irrelevant)
            if self.shuffle_order:
                random.shuffle(items)

            # Assign indices (1-based)
            relevant_indices = []
            irrelevant_indices = []
            for i, item_obj in enumerate(items):
                item_obj.index = i + 1
                if item_obj.is_relevant:
                    relevant_indices.append(i + 1)
                else:
                    irrelevant_indices.append(i + 1)

            # Format context
            retrieved_context = self._format_retrieved_context(items)

            # Truncate if too long
            if len(retrieved_context) > self.max_context_length:
                retrieved_context = retrieved_context[:self.max_context_length] + "..."

            # Extract question and answers
            question = self._extract_question(item.get("user_query"))
            correct_answer = item.get("correct_answer", "")
            incorrect_answers = self._select_incorrect_answers(
                item.get("incorrect_answers", []),
                self.num_incorrect_answers
            )

            # Create sample
            sample = InferenceTrainingSample(
                sample_id=f"personamem_{idx}",
                retrieved_context=retrieved_context,
                num_relevant=num_relevant,  # Number of relevant turns
                num_irrelevant=len(irrelevant_indices),
                relevant_indices=relevant_indices,
                irrelevant_indices=irrelevant_indices,
                question=question,
                correct_answer=correct_answer,
                incorrect_answers=incorrect_answers,
                metadata={
                    "persona_id": item.get("persona_id"),
                    "preference": item.get("preference"),
                    "pref_type": item.get("pref_type"),
                    "topic_preference": item.get("topic_preference")
                }
            )
            samples.append(sample)

        return samples

    def format_for_training(self,
                           sample: InferenceTrainingSample,
                           include_choices: bool = False,
                           show_relevance_hint: bool = False) -> Dict:
        """
        Format sample for model training

        Args:
            sample: Training sample
            include_choices: Whether to format as multiple choice
            show_relevance_hint: Whether to show which items are relevant (for debugging)
        """
        # Build context section
        context_section = sample.retrieved_context

        if show_relevance_hint:
            context_section += f"\n\n[DEBUG: Relevant items: {sample.relevant_indices}]"

        if include_choices:
            # Multiple choice format
            choices = [sample.correct_answer] + sample.incorrect_answers
            random.shuffle(choices)
            correct_idx = choices.index(sample.correct_answer)

            prompt = f"""Based on the retrieved context below, answer the question.
Note: The context contains multiple retrieved passages. Some may be relevant, others may not.

## Retrieved Context:
{context_section}

## Question:
{sample.question}

## Choices:
{chr(10).join(f'{chr(65+i)}. {c}' for i, c in enumerate(choices))}

## Answer (select A, B, C, or D):"""

            return {
                "prompt": prompt,
                "choices": choices,
                "correct_answer": sample.correct_answer,
                "correct_index": correct_idx,
                "correct_label": chr(65 + correct_idx),
                "relevant_indices": sample.relevant_indices,
                "irrelevant_indices": sample.irrelevant_indices
            }
        else:
            # Direct answer format
            prompt = f"""Based on the retrieved context below, answer the question.
Note: The context contains multiple retrieved passages. Some may be relevant, others may not.

## Retrieved Context:
{context_section}

## Question:
{sample.question}

## Answer:"""

            return {
                "prompt": prompt,
                "answer": sample.correct_answer,
                "relevant_indices": sample.relevant_indices,
                "irrelevant_indices": sample.irrelevant_indices
            }

    def save_processed_data(self,
                           samples: List[InferenceTrainingSample],
                           output_path: str,
                           format_type: str = "jsonl"):
        """Save processed samples to file"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format_type == "jsonl":
            with open(path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        elif format_type == "json":
            with open(path, 'w', encoding='utf-8') as f:
                json.dump([s.to_dict() for s in samples], f,
                         ensure_ascii=False, indent=2)

        print(f"Saved {len(samples)} samples to {output_path}")

    def load_processed_data(self, input_path: str) -> List[InferenceTrainingSample]:
        """Load processed samples from file"""
        samples = []
        path = Path(input_path)

        if path.suffix == ".jsonl":
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    samples.append(InferenceTrainingSample(**data))
        elif path.suffix == ".json":
            with open(path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                for data in data_list:
                    samples.append(InferenceTrainingSample(**data))

        return samples


def process_personamem_dataset(
    output_dir: str = "./data/personamem",
    splits: List[str] = ["train_text", "val_text"],
    max_samples: Optional[int] = None,
    num_irrelevant: int = 2,
    shuffle_order: bool = True
) -> Dict[str, List[InferenceTrainingSample]]:
    """
    Process PersonaMem-v2 dataset and save to files

    Args:
        output_dir: Output directory
        splits: Dataset splits to process
        max_samples: Maximum samples per split (None for all)
        num_irrelevant: Number of irrelevant turns to add per sample
        shuffle_order: Whether to shuffle item order

    Returns:
        Dict of split name to processed samples
    """
    processor = PersonaMemProcessor(
        num_irrelevant=num_irrelevant,
        shuffle_order=shuffle_order
    )
    results = {}

    for split in splits:
        print(f"\nProcessing {split}...")

        # Load data
        raw_data = processor.load_dataset(split)
        if not raw_data:
            print(f"  Failed to load {split}")
            continue

        # Limit samples if specified
        if max_samples and len(raw_data) > max_samples:
            raw_data = random.sample(raw_data, max_samples)

        print(f"  Loaded {len(raw_data)} samples")

        # Process
        samples = processor.process(raw_data)
        print(f"  Processed {len(samples)} samples")

        # Show statistics
        if samples:
            avg_relevant = sum(s.num_relevant for s in samples) / len(samples)
            print(f"  Average relevant turns per sample: {avg_relevant:.1f}")
            print(f"  Irrelevant turns added: {num_irrelevant}")

        # Save
        output_path = Path(output_dir) / f"{split}.jsonl"
        processor.save_processed_data(samples, str(output_path))

        results[split] = samples

    return results


def process_with_multiple_ratios(
    output_dir: str = "./data/personamem",
    max_samples: Optional[int] = None,
    train_num_irrelevant: int = 2,  # Number of irrelevant turns for training
    test_num_irrelevant_list: List[int] = [2, 3, 1, 4, 5],  # Different num_irrelevant for test sets
    min_irrelevant: int = 1  # Minimum irrelevant turns (always at least 1)
):
    """
    Process dataset with different numbers of irrelevant turns

    IMPORTANT: Each sample has K relevant items (one per turn in the clue).
    The noise level is controlled by the number of irrelevant turns added.

    Args:
        output_dir: Output directory
        max_samples: Maximum samples per split
        train_num_irrelevant: Number of irrelevant turns for training set
        test_num_irrelevant_list: List of num_irrelevant values for test sets
        min_irrelevant: Minimum number of irrelevant turns (at least 1)

    Creates:
        - train_{N}irrel.jsonl  (e.g., train_2irrel.jsonl)
        - val_{N}irrel.jsonl    (e.g., val_2irrel.jsonl, val_3irrel.jsonl, ...)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Ensure minimum irrelevant for training
    train_num_irrelevant = max(train_num_irrelevant, min_irrelevant)

    print("="*60)
    print(f"Processing TRAINING set: K relevant turns + {train_num_irrelevant} irrelevant turns")
    print("="*60)

    processor = PersonaMemProcessor(
        num_irrelevant=train_num_irrelevant,
        shuffle_order=True
    )

    # Load training data
    raw_train = processor.load_dataset("train_text")
    if max_samples and len(raw_train) > max_samples:
        raw_train = random.sample(raw_train, max_samples)

    print(f"Loaded {len(raw_train)} training samples")

    train_samples = processor.process(raw_train)

    # Statistics
    if train_samples:
        avg_relevant = sum(s.num_relevant for s in train_samples) / len(train_samples)
        print(f"Average relevant turns per sample: {avg_relevant:.1f}")

    train_filename = f"train_{train_num_irrelevant}irrel.jsonl"
    processor.save_processed_data(train_samples, str(output_path / train_filename))
    print(f"Saved training set: {train_filename}")

    # Process test sets with different numbers of irrelevant items
    print("\n" + "="*60)
    print("Processing TEST sets with different noise levels")
    print("="*60)

    # Load validation data once
    raw_val = processor.load_dataset("val_text")
    if max_samples and len(raw_val) > max_samples:
        raw_val = random.sample(raw_val, max_samples)

    print(f"Loaded {len(raw_val)} validation samples")

    test_results = {}
    for num_irrel in test_num_irrelevant_list:
        # Ensure minimum irrelevant
        num_irrel = max(num_irrel, min_irrelevant)

        print(f"\n  Processing: K relevant turns + {num_irrel} irrelevant turns...")

        test_processor = PersonaMemProcessor(
            num_irrelevant=num_irrel,
            shuffle_order=True
        )

        test_samples = test_processor.process(raw_val)
        test_filename = f"val_{num_irrel}irrel.jsonl"
        test_processor.save_processed_data(test_samples, str(output_path / test_filename))

        test_results[num_irrel] = len(test_samples)
        print(f"    Saved: {test_filename} ({len(test_samples)} samples)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if train_samples:
        avg_rel = sum(s.num_relevant for s in train_samples) / len(train_samples)
        print(f"Training set: avg {avg_rel:.1f} relevant + {train_num_irrelevant} irrelevant ({len(train_samples)} samples)")
    print(f"Test sets:")
    for num_irrel, count in test_results.items():
        print(f"  - {num_irrel} irrelevant turns ({count} samples)")

    return {
        "train": train_samples,
        "test_results": test_results
    }


def create_sample_data() -> List[InferenceTrainingSample]:
    """Create sample data for testing"""
    samples = [
        InferenceTrainingSample(
            sample_id="sample_0",
            retrieved_context="""1. [User]: I've been worried about unexpected expenses lately, especially medication costs. [Assistant]: I understand. Financial stress related to healthcare can be very challenging.

2. [User]: I make myself a small mug of lemongrass tea before bed to help me relax. [Assistant]: That sounds like a wonderful calming routine.

3. [User]: I enjoy cooking fusion dishes on weekends. [Assistant]: That sounds like a fun hobby! What cuisines do you like to combine?

4. [User]: My work involves a lot of data analysis using Python. [Assistant]: Data analysis can be quite demanding. What libraries do you typically use?""",
            num_relevant=2,
            num_irrelevant=2,
            relevant_indices=[1, 2],
            irrelevant_indices=[3, 4],
            question="How can I manage anxiety about unexpected expenses?",
            correct_answer="When surprise costs affect your medication budget, list essentials and adjust non-urgent spending. Setting aside a small health cushion each month can help.",
            incorrect_answers=[
                "Focus on your balcony herb garden to distract yourself.",
                "Take out multiple credit cards to spread expenses.",
                "Ignore the expenses until they go away."
            ],
            metadata={"pref_type": "therapy_background"}
        )
    ]
    return samples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process PersonaMem-v2 dataset")
    parser.add_argument("--output-dir", type=str, default="./data/personamem",
                       help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples per split")

    # Mode selection
    parser.add_argument("--multi-ratio", action="store_true",
                       help="Generate multiple test sets with different noise levels")

    # Number of irrelevant items (relevant is always 1 - the complete clue)
    parser.add_argument("--num-irrelevant", type=int, default=2,
                       help="Number of irrelevant items per sample (relevant is always 1)")

    # Multi-ratio mode: specify different noise levels for test sets
    parser.add_argument("--train-irrelevant", type=int, default=2,
                       help="Number of irrelevant items for training set")
    parser.add_argument("--test-irrelevant", type=str, default="1,2,3,4,5",
                       help="Comma-separated list of irrelevant counts for test sets")

    parser.add_argument("--no-shuffle", action="store_true",
                       help="Don't shuffle item order")
    parser.add_argument("--sample-only", action="store_true",
                       help="Create sample data only")

    args = parser.parse_args()

    if args.sample_only:
        print("Creating sample data...")
        samples = create_sample_data()
        processor = PersonaMemProcessor()
        output_path = Path(args.output_dir) / "sample.jsonl"
        processor.save_processed_data(samples, str(output_path))

    elif args.multi_ratio:
        # Multi-ratio mode: generate train + multiple test sets with different noise levels
        test_irrelevant_list = [int(x.strip()) for x in args.test_irrelevant.split(",")]

        process_with_multiple_ratios(
            output_dir=args.output_dir,
            max_samples=args.max_samples,
            train_num_irrelevant=args.train_irrelevant,
            test_num_irrelevant_list=test_irrelevant_list,
            min_irrelevant=1  # Always at least 1 irrelevant
        )

    else:
        # Single mode
        results = process_personamem_dataset(
            output_dir=args.output_dir,
            splits=["train_text", "val_text"],
            max_samples=args.max_samples,
            num_irrelevant=args.num_irrelevant,
            shuffle_order=not args.no_shuffle
        )

        print("\n=== Summary ===")
        for split, samples in results.items():
            print(f"{split}: {len(samples)} samples")
            if samples:
                print(f"  Structure: 1 relevant (complete clue) + {args.num_irrelevant} irrelevant")
                print(f"  Example relevant_indices: {samples[0].relevant_indices}")
                print(f"  Example irrelevant_indices: {samples[0].irrelevant_indices}")
