"""
Main entry point for Mem-alpha

Demonstrates the complete pipeline:
1. Memory construction from conversation chunks
2. Two-layer RAG retrieval
3. Question answering evaluation
"""

import json
import argparse
from typing import List, Callable, Optional

from .memory_structure import MemorySystem, MemoryCategory
from .memory_manager import MemoryManager
from .memory_agent import MemoryConstructionAgent, ConversationChunk, MockLLM
from .rag_retriever import TwoLayerRAGRetriever
from .evaluator import QAEvaluator, RewardCalculator, Question, MockQALLM
from .config import MemAlphaConfig, DEFAULT_CONFIG


def create_sample_chunks() -> List[ConversationChunk]:
    """Create sample conversation chunks for testing"""
    chunks = [
        ConversationChunk(
            chunk_id=1,
            content="""[Dialogue at timestamp 2024-01-15 10:30]
User: I really love Italian food, especially pasta and pizza. I usually eat out 3 times a week.
Assistant: That's great! Do you have a favorite Italian restaurant?
User: Yes, there's this small place called Bella Italia near my office. Their carbonara is amazing.""",
            timestamp="2024-01-15 10:30"
        ),
        ConversationChunk(
            chunk_id=2,
            content="""[Dialogue at timestamp 2024-01-16 14:00]
User: I've been watching a lot of sci-fi movies lately. Just finished the entire Star Wars series.
Assistant: Nice! Are you a big sci-fi fan?
User: Yes, I also enjoy reading science fiction novels. My favorite author is Isaac Asimov.""",
            timestamp="2024-01-16 14:00"
        ),
        ConversationChunk(
            chunk_id=3,
            content="""[Dialogue at timestamp 2024-01-17 09:00]
User: I work as a software engineer at a tech startup. We mainly use Python and JavaScript.
Assistant: Interesting! What kind of projects do you work on?
User: Mostly backend development and some machine learning projects. I enjoy solving complex algorithmic problems.""",
            timestamp="2024-01-17 09:00"
        ),
        ConversationChunk(
            chunk_id=4,
            content="""[Dialogue at timestamp 2024-01-18 16:30]
User: Family is very important to me. I try to have dinner with my parents every Sunday.
Assistant: That's wonderful! Do you have any siblings?
User: Yes, I have a younger sister. She's studying medicine and wants to become a doctor.""",
            timestamp="2024-01-18 16:30"
        ),
        ConversationChunk(
            chunk_id=5,
            content="""[Dialogue at timestamp 2024-01-19 20:00]
User: I've been trying to exercise more. Started going to the gym 3 times a week.
Assistant: That's a great habit! What kind of exercises do you do?
User: Mostly cardio and some weight training. I also try to get at least 7 hours of sleep.""",
            timestamp="2024-01-19 20:00"
        )
    ]
    return chunks


def create_sample_questions() -> List[Question]:
    """Create sample questions for evaluation"""
    return [
        Question(
            question_id=1,
            question="What is the user's favorite type of food?",
            answer="Italian food, especially pasta and pizza"
        ),
        Question(
            question_id=2,
            question="What is the name of the user's favorite restaurant?",
            answer="Bella Italia"
        ),
        Question(
            question_id=3,
            question="What genre of movies does the user enjoy?",
            answer="sci-fi movies"
        ),
        Question(
            question_id=4,
            question="Who is the user's favorite author?",
            answer="Isaac Asimov"
        ),
        Question(
            question_id=5,
            question="What is the user's profession?",
            answer="software engineer"
        ),
        Question(
            question_id=6,
            question="What programming languages does the user work with?",
            answer="Python and JavaScript"
        ),
        Question(
            question_id=7,
            question="What does the user's sister study?",
            answer="medicine"
        ),
        Question(
            question_id=8,
            question="How often does the user go to the gym?",
            answer="3 times a week"
        )
    ]


def run_memory_construction(
    chunks: List[ConversationChunk],
    llm_callable: Optional[Callable[[str], str]] = None,
    config: MemAlphaConfig = DEFAULT_CONFIG
) -> MemoryConstructionAgent:
    """
    Run memory construction pipeline

    Args:
        chunks: List of conversation chunks
        llm_callable: LLM function (uses MockLLM if None)
        config: Configuration

    Returns:
        Agent with constructed memory
    """
    # Initialize agent
    agent = MemoryConstructionAgent(
        llm_callable=llm_callable or MockLLM(),
        max_response_tokens=config.training.max_response_tokens
    )

    print("=" * 60)
    print("Memory Construction Pipeline")
    print("=" * 60)

    # Process each chunk
    for chunk in chunks:
        print(f"\nProcessing Chunk {chunk.chunk_id}...")
        actions = agent.process_chunk(chunk)
        print(f"  Actions taken: {len(actions)}")
        for action in actions:
            status = "OK" if action.result and action.result.success else "FAILED"
            print(f"    - {action.tool_name}: {status}")

    # Summary
    summary = agent.get_action_summary()
    print("\n" + "=" * 60)
    print("Memory Construction Summary")
    print("=" * 60)
    print(f"Chunks processed: {summary['chunks_processed']}")
    print(f"Total actions: {summary['total_actions']}")
    print(f"Successful: {summary['successful_actions']}")
    print(f"Failed: {summary['failed_actions']}")
    print(f"Actions by type: {summary['actions_by_type']}")

    return agent


def run_evaluation(
    agent: MemoryConstructionAgent,
    questions: List[Question],
    llm_callable: Optional[Callable[[str], str]] = None,
    config: MemAlphaConfig = DEFAULT_CONFIG
):
    """
    Run QA evaluation pipeline

    Args:
        agent: Agent with constructed memory
        questions: List of questions
        llm_callable: LLM for QA (uses MockQALLM if None)
        config: Configuration
    """
    # Initialize evaluator and retriever
    retriever = TwoLayerRAGRetriever(
        memory_system=agent.memory,
        k_categories=config.rag.k_categories,
        n_entries_per_category=config.rag.n_entries_per_category
    )

    qa_evaluator = QAEvaluator(
        memory_system=agent.memory,
        llm_callable=llm_callable or MockQALLM(),
        k_categories=config.rag.k_categories,
        n_entries_per_category=config.rag.n_entries_per_category
    )

    reward_calculator = RewardCalculator(
        beta=config.reward.beta,
        gamma=config.reward.gamma
    )

    print("\n" + "=" * 60)
    print("QA Evaluation Pipeline")
    print("=" * 60)

    # Evaluate questions
    qa_results = qa_evaluator.evaluate_questions(questions)

    # Print individual results
    print("\nQuestion Results:")
    for result in qa_results:
        status = "CORRECT" if result.is_correct else "WRONG"
        print(f"  Q{result.question.question_id}: {status} (score: {result.score:.2f})")
        print(f"    Question: {result.question.question[:50]}...")
        print(f"    Expected: {result.question.answer[:50]}...")
        print(f"    Predicted: {result.predicted_answer[:50]}...")

    # Calculate total input length
    total_input_length = sum(
        len(chunk.content) for chunk in agent.processed_chunks
    )

    # Get tool call stats
    action_summary = agent.get_action_summary()

    # Calculate rewards
    evaluation_result = reward_calculator.evaluate(
        memory_system=agent.memory,
        qa_results=qa_results,
        successful_tool_calls=action_summary['successful_actions'],
        total_tool_calls=action_summary['total_actions'],
        total_input_length=total_input_length
    )

    # Print evaluation summary
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Total Questions: {evaluation_result.total_questions}")
    print(f"Correct Answers: {evaluation_result.correct_answers}")
    print(f"\nReward Components:")
    print(f"  r1 (Accuracy):    {evaluation_result.accuracy:.4f}")
    print(f"  r2 (Tool Call):   {evaluation_result.tool_call_success_rate:.4f}")
    print(f"  r3 (Compression): {evaluation_result.compression_ratio:.4f}")
    print(f"  r4 (Content):     {evaluation_result.memory_content_score:.4f}")
    print(f"\nFinal Reward: {evaluation_result.final_reward:.4f}")

    return evaluation_result


def demonstrate_two_layer_retrieval(
    memory_system: MemorySystem,
    query: str,
    config: MemAlphaConfig = DEFAULT_CONFIG
):
    """
    Demonstrate the two-layer RAG retrieval process

    Args:
        memory_system: Memory system to query
        query: Search query
        config: Configuration
    """
    retriever = TwoLayerRAGRetriever(
        memory_system=memory_system,
        k_categories=config.rag.k_categories,
        n_entries_per_category=config.rag.n_entries_per_category
    )

    print("\n" + "=" * 60)
    print("Two-Layer RAG Retrieval Demo")
    print("=" * 60)
    print(f"Query: {query}")

    # Perform retrieval
    result = retriever.retrieve(query)

    # Layer 1 results
    print(f"\nLayer 1 - Selected Categories (top {config.rag.k_categories}):")
    for category, score in result.selected_categories:
        print(f"  - {category.value}: {score:.4f}")

    # Layer 2 results
    print(f"\nLayer 2 - Retrieved Entries:")
    for entry, score in result.retrieved_entries:
        print(f"  [{entry.id}] ({entry.category.value}) {entry.content[:60]}... | score: {score:.4f}")

    # Format for QA
    print("\nFormatted Context for QA:")
    formatted = retriever.retrieve_for_qa(query)
    print(formatted[:500] + "..." if len(formatted) > 500 else formatted)


def print_memory_state(memory_system: MemorySystem):
    """Print current memory state"""
    print("\n" + "=" * 60)
    print("Current Memory State")
    print("=" * 60)

    # Core memory
    print("\n--- Core Memory ---")
    if memory_system.core_memory.content:
        print(memory_system.core_memory.content)
    else:
        print("(empty)")

    # Category memories
    for category in MemoryCategory:
        entries = memory_system.get_category_entries(category)
        if entries:
            print(f"\n--- {category.value} ({len(entries)} entries) ---")
            for entry in entries:
                print(f"  [{entry.id}] {entry.content}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Mem-alpha Memory System Demo")
    parser.add_argument("--mode", choices=["demo", "construct", "eval", "retrieve"],
                        default="demo", help="Running mode")
    parser.add_argument("--k-categories", type=int, default=3,
                        help="Number of categories for Layer 1 retrieval")
    parser.add_argument("--n-entries", type=int, default=5,
                        help="Number of entries per category for Layer 2 retrieval")
    parser.add_argument("--query", type=str, default="What food does the user like?",
                        help="Query for retrieval demo")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for memory state")

    args = parser.parse_args()

    # Create config
    config = MemAlphaConfig()
    config.rag.k_categories = args.k_categories
    config.rag.n_entries_per_category = args.n_entries

    # Create sample data
    chunks = create_sample_chunks()
    questions = create_sample_questions()

    if args.mode == "demo":
        # Run complete demo
        print("Running Mem-alpha Complete Demo...")

        # 1. Memory construction
        agent = run_memory_construction(chunks, config=config)

        # 2. Print memory state
        print_memory_state(agent.memory)

        # 3. Demonstrate retrieval
        demonstrate_two_layer_retrieval(agent.memory, args.query, config)

        # 4. Run evaluation
        run_evaluation(agent, questions, config=config)

        # Save memory if output specified
        if args.output:
            agent.memory.save(args.output)
            print(f"\nMemory saved to: {args.output}")

    elif args.mode == "construct":
        # Only memory construction
        agent = run_memory_construction(chunks, config=config)
        print_memory_state(agent.memory)

        if args.output:
            agent.memory.save(args.output)

    elif args.mode == "eval":
        # Only evaluation (requires existing memory)
        agent = run_memory_construction(chunks, config=config)
        run_evaluation(agent, questions, config=config)

    elif args.mode == "retrieve":
        # Only retrieval demo
        agent = run_memory_construction(chunks, config=config)
        demonstrate_two_layer_retrieval(agent.memory, args.query, config)


if __name__ == "__main__":
    main()
