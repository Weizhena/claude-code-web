"""
Evaluation Module for Memory-Augmented QA

Implements the reward functions from Mem-alpha:
- r1: Correctness Reward (QA accuracy)
- r2: Tool Call Format Reward
- r3: Compression Reward
- r4: Memory Content Reward

Also implements the QA evaluation pipeline using two-layer RAG retrieval.
"""

from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import re


from .memory_structure import MemorySystem
from .rag_retriever import TwoLayerRAGRetriever


@dataclass
class Question:
    """A question for evaluation"""
    question_id: int
    question: str
    answer: str  # Ground truth answer
    metadata: Optional[Dict] = None


@dataclass
class QAResult:
    """Result of a QA evaluation"""
    question: Question
    retrieved_context: str
    predicted_answer: str
    is_correct: bool
    score: float


@dataclass
class EvaluationResult:
    """Complete evaluation result"""
    total_questions: int
    correct_answers: int
    accuracy: float  # r1 reward
    tool_call_success_rate: float  # r2 reward
    compression_ratio: float  # r3 reward
    memory_content_score: float  # r4 reward
    final_reward: float
    qa_results: List[QAResult]


class QAEvaluator:
    """
    Question Answering Evaluator

    Uses two-layer RAG retrieval to answer questions based on constructed memory,
    then evaluates against ground truth answers.
    """

    QA_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided memory context.

## Memory Context:
{context}

## Question:
{question}

## Instructions:
- Answer based ONLY on the information in the memory context
- If the answer is not in the context, say "I don't have that information"
- Keep your answer concise and direct

## Answer:"""

    def __init__(self,
                 memory_system: MemorySystem,
                 llm_callable: Optional[Callable[[str], str]] = None,
                 k_categories: int = 3,
                 n_entries_per_category: int = 5):
        """
        Initialize evaluator

        Args:
            memory_system: The memory system to evaluate
            llm_callable: Function to call LLM for answer generation
            k_categories: Number of categories for Layer 1 retrieval
            n_entries_per_category: Number of entries for Layer 2 retrieval
        """
        self.memory = memory_system
        self.llm_callable = llm_callable
        self.retriever = TwoLayerRAGRetriever(
            memory_system=memory_system,
            k_categories=k_categories,
            n_entries_per_category=n_entries_per_category
        )

    def set_llm(self, llm_callable: Callable[[str], str]) -> None:
        """Set the LLM callable"""
        self.llm_callable = llm_callable

    def answer_question(self, question: Question) -> QAResult:
        """
        Answer a single question using RAG pipeline

        Args:
            question: The question to answer

        Returns:
            QAResult with prediction and evaluation
        """
        if self.llm_callable is None:
            raise ValueError("LLM callable not set. Use set_llm() first.")

        # Retrieve relevant context using two-layer RAG
        context = self.retriever.retrieve_for_qa(question.question)

        # Generate answer
        prompt = self.QA_PROMPT_TEMPLATE.format(
            context=context,
            question=question.question
        )
        predicted_answer = self.llm_callable(prompt)

        # Evaluate correctness
        is_correct, score = self._evaluate_answer(
            predicted=predicted_answer,
            ground_truth=question.answer
        )

        return QAResult(
            question=question,
            retrieved_context=context,
            predicted_answer=predicted_answer,
            is_correct=is_correct,
            score=score
        )

    def _evaluate_answer(self,
                         predicted: str,
                         ground_truth: str) -> Tuple[bool, float]:
        """
        Evaluate predicted answer against ground truth

        Uses substring matching and F1 score
        """
        pred_clean = self._normalize_answer(predicted)
        truth_clean = self._normalize_answer(ground_truth)

        # Check for exact or substring match
        if truth_clean in pred_clean or pred_clean in truth_clean:
            return True, 1.0

        # Calculate token-level F1 score
        pred_tokens = set(pred_clean.split())
        truth_tokens = set(truth_clean.split())

        if not truth_tokens:
            return False, 0.0

        common = pred_tokens & truth_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(truth_tokens) if truth_tokens else 0

        if precision + recall == 0:
            return False, 0.0

        f1 = 2 * precision * recall / (precision + recall)

        # Consider correct if F1 > 0.5
        return f1 > 0.5, f1

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison"""
        # Lowercase
        answer = answer.lower()
        # Remove punctuation
        answer = re.sub(r'[^\w\s]', '', answer)
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        return answer

    def evaluate_questions(self, questions: List[Question]) -> List[QAResult]:
        """Evaluate a list of questions"""
        results = []
        for question in questions:
            result = self.answer_question(question)
            results.append(result)
        return results


class RewardCalculator:
    """
    Calculate rewards for Mem-alpha training

    Rewards:
    - r1: Correctness reward (QA accuracy)
    - r2: Tool call format reward
    - r3: Compression reward
    - r4: Memory content reward
    """

    def __init__(self,
                 beta: float = 0.05,
                 gamma: float = 0.1,
                 llm_judge: Optional[Callable[[str], str]] = None):
        """
        Initialize reward calculator

        Args:
            beta: Weight for compression reward
            gamma: Weight for memory content reward
            llm_judge: LLM callable for memory content validation
        """
        self.beta = beta
        self.gamma = gamma
        self.llm_judge = llm_judge

    def calculate_correctness_reward(self, qa_results: List[QAResult]) -> float:
        """
        r1: Correctness reward based on QA accuracy

        r1 = (number of correct answers) / (total questions)
        """
        if not qa_results:
            return 0.0

        correct = sum(1 for r in qa_results if r.is_correct)
        return correct / len(qa_results)

    def calculate_tool_call_reward(self,
                                   successful_calls: int,
                                   total_calls: int) -> float:
        """
        r2: Tool call format reward

        r2 = (successful tool calls) / (total tool calls)
        """
        if total_calls == 0:
            return 1.0  # No calls means no errors
        return successful_calls / total_calls

    def calculate_compression_reward(self,
                                     memory_length: int,
                                     input_length: int) -> float:
        """
        r3: Compression reward

        r3 = 1 - (memory_length / input_length)

        Encourages efficient memory usage
        """
        if input_length == 0:
            return 1.0
        ratio = memory_length / input_length
        return max(0, 1 - ratio)

    def calculate_memory_content_reward(self,
                                        memory_system: MemorySystem) -> float:
        """
        r4: Memory content quality reward

        Uses LLM judge to validate memory entries
        Returns fraction of valid entries
        """
        if self.llm_judge is None:
            return 1.0  # Default to full score if no judge

        all_entries = memory_system.get_all_entries()
        if not all_entries:
            return 1.0

        valid_count = 0
        for entry in all_entries:
            if self._validate_memory_entry(entry):
                valid_count += 1

        return valid_count / len(all_entries)

    def _validate_memory_entry(self, entry) -> bool:
        """Validate a single memory entry using LLM judge"""
        prompt = f"""Analyze the quality of this memory entry:

Category: {entry.category.value}
Content: {entry.content}

Is this a valid, well-formed memory entry that:
1. Contains meaningful information
2. Is appropriately categorized
3. Is not a placeholder or generic text

Respond with ONLY "VALID" or "INVALID"."""

        try:
            response = self.llm_judge(prompt) if self.llm_judge else "VALID"
            return "VALID" in response.upper()
        except Exception:
            return True  # Default to valid on error

    def calculate_total_reward(self,
                               r1: float,
                               r2: float,
                               r3: float,
                               r4: float) -> float:
        """
        Calculate total reward: r = r1 + r2 + beta * r3 + gamma * r4
        """
        return r1 + r2 + self.beta * r3 + self.gamma * r4

    def evaluate(self,
                 memory_system: MemorySystem,
                 qa_results: List[QAResult],
                 successful_tool_calls: int,
                 total_tool_calls: int,
                 total_input_length: int) -> EvaluationResult:
        """
        Complete evaluation with all reward components

        Args:
            memory_system: The constructed memory system
            qa_results: Results from QA evaluation
            successful_tool_calls: Number of successful tool calls
            total_tool_calls: Total number of tool calls
            total_input_length: Total length of input chunks

        Returns:
            EvaluationResult with all metrics
        """
        # Calculate individual rewards
        r1 = self.calculate_correctness_reward(qa_results)
        r2 = self.calculate_tool_call_reward(successful_tool_calls, total_tool_calls)
        r3 = self.calculate_compression_reward(
            memory_system.get_total_memory_length(),
            total_input_length
        )
        r4 = self.calculate_memory_content_reward(memory_system)

        # Calculate total reward
        total_reward = self.calculate_total_reward(r1, r2, r3, r4)

        return EvaluationResult(
            total_questions=len(qa_results),
            correct_answers=sum(1 for r in qa_results if r.is_correct),
            accuracy=r1,
            tool_call_success_rate=r2,
            compression_ratio=r3,
            memory_content_score=r4,
            final_reward=total_reward,
            qa_results=qa_results
        )


class MockQALLM:
    """Mock LLM for QA testing"""

    def __call__(self, prompt: str) -> str:
        """Generate mock answer based on context"""
        # Extract question from prompt
        q_match = re.search(r'Question:\s*(.*?)\s*(?:##|$)', prompt, re.DOTALL)
        if not q_match:
            return "I don't know."

        question = q_match.group(1).strip().lower()

        # Extract context
        ctx_match = re.search(r'Memory Context:\s*(.*?)\s*##', prompt, re.DOTALL)
        context = ctx_match.group(1).strip().lower() if ctx_match else ""

        # Simple keyword matching for mock answers
        keywords = re.findall(r'\b\w{4,}\b', question)

        for keyword in keywords:
            if keyword in context:
                # Find sentence containing keyword
                sentences = re.split(r'[.!?]', context)
                for sent in sentences:
                    if keyword in sent and len(sent.strip()) > 20:
                        return sent.strip().capitalize() + "."

        return "Based on the context, I cannot find a specific answer."
