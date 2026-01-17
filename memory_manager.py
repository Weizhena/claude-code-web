"""
Memory Manager - Tool functions for memory operations
Implements memory_insert, memory_update, memory_delete as in Mem-alpha paper
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import re

from .memory_structure import MemorySystem, MemoryCategory, MemoryEntry


@dataclass
class ToolCallResult:
    """Result of a tool call"""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None


class MemoryManager:
    """
    Memory Manager with tool functions for LLM agent interaction

    Tools available:
    - memory_insert: Add new memory entry to a category
    - memory_update: Update existing memory entry
    - memory_delete: Delete memory entry by ID
    - core_memory_update: Update the core memory summary
    """

    def __init__(self, memory_system: Optional[MemorySystem] = None):
        self.memory = memory_system or MemorySystem()
        self.operation_history: List[Dict] = []

    def get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions for LLM function calling"""
        return [
            {
                "name": "memory_insert",
                "description": "Insert a new memory entry into a specific category",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [c.value for c in MemoryCategory],
                            "description": "Memory category to insert into"
                        },
                        "content": {
                            "type": "string",
                            "description": "Memory content to store"
                        },
                        "timestamp": {
                            "type": "string",
                            "description": "Optional timestamp for the memory (e.g., '2024-01-01 10:00')"
                        },
                        "importance": {
                            "type": "number",
                            "description": "Importance score from 0 to 1, default 1.0"
                        }
                    },
                    "required": ["category", "content"]
                }
            },
            {
                "name": "memory_update",
                "description": "Update an existing memory entry by its ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [c.value for c in MemoryCategory],
                            "description": "Memory category containing the entry"
                        },
                        "entry_id": {
                            "type": "integer",
                            "description": "ID of the memory entry to update"
                        },
                        "new_content": {
                            "type": "string",
                            "description": "New content for the memory entry"
                        }
                    },
                    "required": ["category", "entry_id", "new_content"]
                }
            },
            {
                "name": "memory_delete",
                "description": "Delete a memory entry by its ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [c.value for c in MemoryCategory],
                            "description": "Memory category containing the entry"
                        },
                        "entry_id": {
                            "type": "integer",
                            "description": "ID of the memory entry to delete"
                        }
                    },
                    "required": ["category", "entry_id"]
                }
            },
            {
                "name": "core_memory_update",
                "description": "Update the core memory summary (max 512 tokens)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "New core memory content (summary of user understanding)"
                        }
                    },
                    "required": ["content"]
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Dict) -> ToolCallResult:
        """Execute a tool call and return the result"""
        try:
            if tool_name == "memory_insert":
                return self._execute_insert(arguments)
            elif tool_name == "memory_update":
                return self._execute_update(arguments)
            elif tool_name == "memory_delete":
                return self._execute_delete(arguments)
            elif tool_name == "core_memory_update":
                return self._execute_core_update(arguments)
            else:
                return ToolCallResult(
                    success=False,
                    message=f"Unknown tool: {tool_name}",
                    error=f"Tool '{tool_name}' not found"
                )
        except Exception as e:
            return ToolCallResult(
                success=False,
                message=f"Error executing {tool_name}",
                error=str(e)
            )

    def _execute_insert(self, args: Dict) -> ToolCallResult:
        """Execute memory_insert"""
        category_str = args.get("category")
        content = args.get("content")
        timestamp = args.get("timestamp")
        importance = args.get("importance", 1.0)

        if not category_str or not content:
            return ToolCallResult(
                success=False,
                message="Missing required parameters",
                error="category and content are required"
            )

        try:
            category = MemoryCategory(category_str)
        except ValueError:
            return ToolCallResult(
                success=False,
                message=f"Invalid category: {category_str}",
                error=f"Category must be one of: {[c.value for c in MemoryCategory]}"
            )

        entry = self.memory.memory_insert(
            category=category,
            content=content,
            timestamp=timestamp,
            importance=importance
        )

        self._log_operation("insert", {
            "category": category_str,
            "entry_id": entry.id,
            "content": content[:100] + "..." if len(content) > 100 else content
        })

        return ToolCallResult(
            success=True,
            message=f"Inserted memory [{entry.id}] into {category_str}",
            data=entry.to_dict()
        )

    def _execute_update(self, args: Dict) -> ToolCallResult:
        """Execute memory_update"""
        category_str = args.get("category")
        entry_id = args.get("entry_id")
        new_content = args.get("new_content")

        if not all([category_str, entry_id is not None, new_content]):
            return ToolCallResult(
                success=False,
                message="Missing required parameters",
                error="category, entry_id, and new_content are required"
            )

        try:
            category = MemoryCategory(category_str)
        except ValueError:
            return ToolCallResult(
                success=False,
                message=f"Invalid category: {category_str}",
                error=f"Category must be one of: {[c.value for c in MemoryCategory]}"
            )

        entry = self.memory.memory_update(category, entry_id, new_content)

        if entry:
            self._log_operation("update", {
                "category": category_str,
                "entry_id": entry_id,
                "new_content": new_content[:100] + "..." if len(new_content) > 100 else new_content
            })
            return ToolCallResult(
                success=True,
                message=f"Updated memory [{entry_id}] in {category_str}",
                data=entry.to_dict()
            )
        else:
            return ToolCallResult(
                success=False,
                message=f"Entry [{entry_id}] not found in {category_str}",
                error="Entry not found"
            )

    def _execute_delete(self, args: Dict) -> ToolCallResult:
        """Execute memory_delete"""
        category_str = args.get("category")
        entry_id = args.get("entry_id")

        if not category_str or entry_id is None:
            return ToolCallResult(
                success=False,
                message="Missing required parameters",
                error="category and entry_id are required"
            )

        try:
            category = MemoryCategory(category_str)
        except ValueError:
            return ToolCallResult(
                success=False,
                message=f"Invalid category: {category_str}",
                error=f"Category must be one of: {[c.value for c in MemoryCategory]}"
            )

        success = self.memory.memory_delete(category, entry_id)

        if success:
            self._log_operation("delete", {
                "category": category_str,
                "entry_id": entry_id
            })
            return ToolCallResult(
                success=True,
                message=f"Deleted memory [{entry_id}] from {category_str}"
            )
        else:
            return ToolCallResult(
                success=False,
                message=f"Entry [{entry_id}] not found in {category_str}",
                error="Entry not found"
            )

    def _execute_core_update(self, args: Dict) -> ToolCallResult:
        """Execute core_memory_update"""
        content = args.get("content")

        if not content:
            return ToolCallResult(
                success=False,
                message="Missing required parameter",
                error="content is required"
            )

        self.memory.core_memory_update(content)

        self._log_operation("core_update", {
            "content": content[:100] + "..." if len(content) > 100 else content
        })

        return ToolCallResult(
            success=True,
            message="Core memory updated successfully",
            data={"content": content}
        )

    def _log_operation(self, operation: str, details: Dict) -> None:
        """Log operation for history tracking"""
        self.operation_history.append({
            "operation": operation,
            "details": details
        })

    def get_operation_history(self) -> List[Dict]:
        """Get history of all operations"""
        return self.operation_history.copy()

    def parse_tool_calls_from_response(self, response: str) -> List[Tuple[str, Dict]]:
        """
        Parse tool calls from LLM response
        Supports multiple formats:
        1. JSON format: {"name": "tool_name", "arguments": {...}}
        2. Function call format: tool_name(arg1=val1, arg2=val2)
        """
        tool_calls = []

        # Try to parse JSON format tool calls
        json_pattern = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        json_matches = re.findall(json_pattern, response, re.DOTALL)

        for name, args_str in json_matches:
            try:
                args = json.loads(args_str)
                tool_calls.append((name, args))
            except json.JSONDecodeError:
                continue

        # Try to parse function call format
        func_pattern = r'(memory_insert|memory_update|memory_delete|core_memory_update)\s*\((.*?)\)'
        func_matches = re.findall(func_pattern, response, re.DOTALL)

        for name, args_str in func_matches:
            args = self._parse_function_args(args_str)
            if args:
                tool_calls.append((name, args))

        return tool_calls

    def _parse_function_args(self, args_str: str) -> Optional[Dict]:
        """Parse function arguments string into dict"""
        args = {}
        # Pattern for key=value or key="value"
        pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+\.?\d*)|(\w+))'
        matches = re.findall(pattern, args_str)

        for match in matches:
            key = match[0]
            # Get the non-empty value
            value = next((v for v in match[1:] if v), None)
            if value is not None:
                # Try to convert to int/float if applicable
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except (ValueError, TypeError):
                    pass
                args[key] = value

        return args if args else None

    def get_memory_state_summary(self) -> str:
        """Get a summary of current memory state for LLM context"""
        counts = self.memory.get_memory_count()
        total = sum(counts.values())

        summary_lines = [
            f"Current Memory State (Total: {total} entries):",
            f"- Core Memory: {'Set' if counts.get('core_memory', 0) else 'Empty'}"
        ]

        for category in MemoryCategory:
            count = counts.get(category.value, 0)
            if count > 0:
                summary_lines.append(f"- {category.value}: {count} entries")

        return "\n".join(summary_lines)

    def get_memory_for_context(self, max_tokens: int = 4096) -> str:
        """Get memory content formatted for LLM context window"""
        return self.memory.to_rag_format()
