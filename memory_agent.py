"""
Memory Construction Agent
Processes conversation chunks and decides memory operations using LLM

Based on Mem-alpha paper: The agent processes sequential information chunks,
learns to extract, store, and update the memory system.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import re

from .memory_structure import MemorySystem, MemoryCategory
from .memory_manager import MemoryManager, ToolCallResult


@dataclass
class ConversationChunk:
    """A chunk of conversation to process"""
    chunk_id: int
    content: str
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """An action taken by the agent"""
    chunk_id: int
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[ToolCallResult] = None


class MemoryConstructionAgent:
    """
    Agent that processes conversation chunks and constructs memory

    The agent:
    1. Receives a conversation chunk
    2. Decides what memory operations to perform
    3. Executes the operations
    4. Moves to the next chunk
    """

    SYSTEM_PROMPT = """You are a memory management agent. Your task is to process information and store it in a structured memory system.

## Memory Categories (9 dimensions):

1. **sensory_lifestyle**: Daily sensory preferences and lifestyle choices (food, travel, fashion)
2. **culture_entertainment**: Cultural consumption and entertainment preferences (movies, music, books, games)
3. **cognition_work**: Cognitive style, professional skills, and work patterns
4. **values**: Core beliefs, personality traits, and life attitudes
5. **physiology_health**: Physical health, biorhythm, and body conditions
6. **resource_economic**: Financial status, spending habits, and asset allocation
7. **social_interpersonal**: Social patterns, relationships, and interaction styles
8. **spatiotemporal_context**: Location history, time patterns, and environment preferences
9. **psychological_defense**: Psychological boundaries, sensitive topics, and internal contradictions

## Available Tools:

1. **memory_insert**: Add new memory entry
   - Parameters: category (string), content (string), timestamp (optional), importance (0-1)

2. **memory_update**: Update existing memory entry
   - Parameters: category (string), entry_id (int), new_content (string)

3. **memory_delete**: Delete memory entry
   - Parameters: category (string), entry_id (int)

4. **core_memory_update**: Update core memory summary
   - Parameters: content (string, max 512 tokens)

## Instructions:

1. Analyze the new information chunk
2. Identify relevant information to store
3. Categorize information into appropriate memory categories
4. Decide whether to INSERT new memories, UPDATE existing ones, or DELETE outdated ones
5. Call the appropriate tools with correct parameters

## Output Format:

For each tool call, output in JSON format:
```json
{"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
```

You can make multiple tool calls. Each memory entry should be a distinct, atomic piece of information.
Remember: Each memory entry will be assigned a sequential ID automatically.
"""

    CHUNK_PROMPT_TEMPLATE = """
## Current Memory State:
{memory_state}

## New Information Chunk (ID: {chunk_id}):
{chunk_content}

## Task:
Analyze this chunk and decide what memory operations to perform.
- Store important facts, preferences, events, or patterns
- Update existing memories if information has changed
- Delete outdated or contradictory information
- Update core memory if there's a significant understanding change

Output your tool calls in JSON format. If no memory update is needed, output: {{"action": "skip"}}
"""

    def __init__(self,
                 llm_callable: Optional[Callable[[str], str]] = None,
                 memory_system: Optional[MemorySystem] = None,
                 max_response_tokens: int = 2048):
        """
        Initialize the memory construction agent

        Args:
            llm_callable: Function that takes prompt and returns LLM response
            memory_system: Optional existing memory system
            max_response_tokens: Maximum tokens for LLM response
        """
        self.memory_manager = MemoryManager(memory_system)
        self.llm_callable = llm_callable
        self.max_response_tokens = max_response_tokens
        self.action_history: List[AgentAction] = []
        self.processed_chunks: List[ConversationChunk] = []

    @property
    def memory(self) -> MemorySystem:
        return self.memory_manager.memory

    def set_llm(self, llm_callable: Callable[[str], str]) -> None:
        """Set the LLM callable"""
        self.llm_callable = llm_callable

    def process_chunk(self, chunk: ConversationChunk) -> List[AgentAction]:
        """
        Process a single conversation chunk

        Args:
            chunk: The conversation chunk to process

        Returns:
            List of actions taken
        """
        if self.llm_callable is None:
            raise ValueError("LLM callable not set. Use set_llm() first.")

        # Build prompt
        prompt = self._build_chunk_prompt(chunk)

        # Get LLM response
        full_prompt = self.SYSTEM_PROMPT + "\n\n" + prompt
        response = self.llm_callable(full_prompt)

        # Parse and execute tool calls
        actions = self._parse_and_execute(chunk.chunk_id, response)

        # Record
        self.processed_chunks.append(chunk)
        self.action_history.extend(actions)

        return actions

    def process_chunks(self, chunks: List[ConversationChunk]) -> List[List[AgentAction]]:
        """
        Process multiple conversation chunks sequentially

        Args:
            chunks: List of conversation chunks

        Returns:
            List of action lists (one per chunk)
        """
        all_actions = []
        for chunk in chunks:
            actions = self.process_chunk(chunk)
            all_actions.append(actions)
        return all_actions

    def _build_chunk_prompt(self, chunk: ConversationChunk) -> str:
        """Build prompt for processing a chunk"""
        memory_state = self.memory_manager.get_memory_state_summary()

        return self.CHUNK_PROMPT_TEMPLATE.format(
            memory_state=memory_state,
            chunk_id=chunk.chunk_id,
            chunk_content=chunk.content
        )

    def _parse_and_execute(self, chunk_id: int, response: str) -> List[AgentAction]:
        """Parse LLM response and execute tool calls"""
        actions = []

        # Check for skip action
        if '{"action": "skip"}' in response or '"action": "skip"' in response:
            return actions

        # Parse tool calls
        tool_calls = self._extract_tool_calls(response)

        for tool_name, arguments in tool_calls:
            # Add source chunk info
            if "source_chunk_id" not in arguments:
                arguments["source_chunk_id"] = chunk_id

            # Execute tool
            result = self.memory_manager.execute_tool(tool_name, arguments)

            action = AgentAction(
                chunk_id=chunk_id,
                tool_name=tool_name,
                arguments=arguments,
                result=result
            )
            actions.append(action)

        return actions

    def _extract_tool_calls(self, response: str) -> List[tuple]:
        """Extract tool calls from LLM response"""
        tool_calls = []

        # Pattern for JSON tool calls
        json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'

        # Also try simpler pattern
        simple_pattern = r'\{"name":\s*"(\w+)",\s*"arguments":\s*(\{[^}]+\})\}'

        # Try both patterns
        for pattern in [json_pattern, simple_pattern]:
            matches = re.findall(pattern, response, re.DOTALL)
            for name, args_str in matches:
                try:
                    args = json.loads(args_str)
                    tool_calls.append((name, args))
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    args = self._fix_and_parse_json(args_str)
                    if args:
                        tool_calls.append((name, args))

        # Remove duplicates while preserving order
        seen = set()
        unique_calls = []
        for call in tool_calls:
            call_str = json.dumps(call, sort_keys=True)
            if call_str not in seen:
                seen.add(call_str)
                unique_calls.append(call)

        return unique_calls

    def _fix_and_parse_json(self, json_str: str) -> Optional[Dict]:
        """Try to fix and parse malformed JSON"""
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Add quotes to unquoted keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

    def get_action_summary(self) -> Dict:
        """Get summary of all actions taken"""
        summary = {
            "total_actions": len(self.action_history),
            "chunks_processed": len(self.processed_chunks),
            "successful_actions": 0,
            "failed_actions": 0,
            "actions_by_type": {}
        }

        for action in self.action_history:
            # Count by type
            if action.tool_name not in summary["actions_by_type"]:
                summary["actions_by_type"][action.tool_name] = 0
            summary["actions_by_type"][action.tool_name] += 1

            # Count success/failure
            if action.result and action.result.success:
                summary["successful_actions"] += 1
            else:
                summary["failed_actions"] += 1

        return summary

    def get_tool_call_success_rate(self) -> float:
        """Calculate r2 reward: tool call format success rate"""
        if not self.action_history:
            return 1.0

        successful = sum(
            1 for action in self.action_history
            if action.result and action.result.success
        )
        return successful / len(self.action_history)

    def reset(self) -> None:
        """Reset agent state (but keep memory)"""
        self.action_history = []
        self.processed_chunks = []


class MockLLM:
    """
    Mock LLM for testing - generates reasonable memory operations
    based on simple keyword matching
    """

    CATEGORY_KEYWORDS = {
        MemoryCategory.SENSORY_LIFESTYLE: [
            "food", "eat", "travel", "trip", "fashion", "style", "taste",
            "prefer", "like", "favorite", "restaurant", "cuisine"
        ],
        MemoryCategory.CULTURE_ENTERTAINMENT: [
            "movie", "film", "music", "song", "book", "read", "game",
            "play", "watch", "show", "series", "album"
        ],
        MemoryCategory.COGNITION_WORK: [
            "work", "job", "career", "skill", "learn", "think", "problem",
            "solve", "project", "task", "professional"
        ],
        MemoryCategory.VALUES: [
            "believe", "value", "important", "principle", "attitude",
            "personality", "trait", "character", "moral"
        ],
        MemoryCategory.PHYSIOLOGY_HEALTH: [
            "health", "exercise", "sleep", "diet", "body", "medical",
            "doctor", "sick", "fitness", "workout"
        ],
        MemoryCategory.RESOURCE_ECONOMIC: [
            "money", "spend", "save", "invest", "budget", "financial",
            "income", "expense", "asset", "cost"
        ],
        MemoryCategory.SOCIAL_INTERPERSONAL: [
            "friend", "family", "relationship", "social", "meet", "talk",
            "communicate", "party", "gather", "colleague"
        ],
        MemoryCategory.SPATIOTEMPORAL_CONTEXT: [
            "location", "place", "city", "home", "office", "time",
            "schedule", "routine", "environment", "where"
        ],
        MemoryCategory.PSYCHOLOGICAL_DEFENSE: [
            "stress", "anxiety", "fear", "worry", "sensitive", "boundary",
            "conflict", "uncomfortable", "avoid", "privacy"
        ]
    }

    def __call__(self, prompt: str) -> str:
        """Generate mock LLM response with tool calls"""
        # Extract the chunk content from prompt
        chunk_match = re.search(
            r'New Information Chunk.*?:\s*(.*?)(?=##|$)',
            prompt,
            re.DOTALL
        )
        if not chunk_match:
            return '{"action": "skip"}'

        chunk_content = chunk_match.group(1).strip().lower()

        # Determine categories based on keywords
        tool_calls = []
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if any(kw in chunk_content for kw in keywords):
                # Extract a sentence containing the keyword
                sentences = re.split(r'[.!?]', chunk_content)
                for sent in sentences:
                    if any(kw in sent for kw in keywords) and len(sent.strip()) > 10:
                        tool_call = {
                            "name": "memory_insert",
                            "arguments": {
                                "category": category.value,
                                "content": sent.strip().capitalize(),
                                "importance": 0.8
                            }
                        }
                        tool_calls.append(json.dumps(tool_call))
                        break

        if not tool_calls:
            return '{"action": "skip"}'

        return "\n".join(tool_calls)
