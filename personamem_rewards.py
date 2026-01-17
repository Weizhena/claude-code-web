"""
PersonaMem Multi-Dimensional Reward System

Reward components:
1. r_format: Output format correctness (0.0 - 0.2)
2. r_answer: Answer correctness (0.0 or 1.0)
3. r_evidence: Evidence citation quality (-0.1 - 0.3)

Total reward: R = r_format + r_answer + r_evidence
"""

import re
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    # Format reward
    format_answer_reward: float = 0.1      # Has "Answer: [A-D]"
    format_evidence_reward: float = 0.05   # Has "Evidence X:" structure
    format_reasoning_reward: float = 0.05  # Has "Reasoning X:" structure
    format_max: float = 0.2                # Maximum format reward

    # Answer reward
    answer_correct_reward: float = 1.0     # Correct answer
    answer_incorrect_reward: float = 0.0   # Wrong answer

    # Evidence reward
    evidence_relevant_reward: float = 0.1   # Per relevant evidence cited
    evidence_irrelevant_penalty: float = -0.05  # Per irrelevant evidence cited
    evidence_max: float = 0.3               # Maximum evidence reward
    evidence_min: float = -0.1              # Minimum evidence reward (penalty cap)


@dataclass
class ParsedOutput:
    """Parsed model output"""
    answer: str                          # A/B/C/D or empty
    evidence_snippets: List[str]         # List of evidence text
    reasoning_list: List[str]            # List of reasoning text
    raw_output: str                       # Original output
    has_answer_format: bool              # Whether has "Answer: X" format
    has_evidence_format: bool            # Whether has "Evidence X:" format
    has_reasoning_format: bool           # Whether has "Reasoning X:" format


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown"""
    r_format: float
    r_answer: float
    r_evidence: float
    total: float

    # Details
    format_details: Dict[str, bool]
    answer_correct: bool
    evidence_details: Dict[str, Any]


class OutputParser:
    """Parse model output into structured format"""

    @staticmethod
    def parse(raw_output: str) -> ParsedOutput:
        """
        Parse model output

        Expected format:
        Answer: B

        Evidence and Reasoning:
        Evidence 1: [text from context]
        Reasoning 1: [explanation]
        Evidence 2: [text from context]
        Reasoning 2: [explanation]
        """
        answer = ""
        evidence_snippets = []
        reasoning_list = []
        has_answer_format = False
        has_evidence_format = False
        has_reasoning_format = False

        # Extract answer
        answer_match = re.search(r'Answer:\s*([A-Da-d])', raw_output)
        if answer_match:
            answer = answer_match.group(1).upper()
            has_answer_format = True
        else:
            # Fallback: first letter at start of output
            first_line = raw_output.strip().split('\n')[0] if raw_output.strip() else ""
            letter_match = re.match(r'^([A-Da-d])[.:\s]', first_line)
            if letter_match:
                answer = letter_match.group(1).upper()

        # Extract evidence snippets
        evidence_pattern = r'Evidence\s*(\d+)?:\s*(.+?)(?=(?:Reasoning\s*\d*:|Evidence\s*\d+:|$))'
        evidence_matches = re.findall(evidence_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        for _, text in evidence_matches:
            cleaned = text.strip()
            if cleaned and len(cleaned) > 10:
                evidence_snippets.append(cleaned)
                has_evidence_format = True

        # Extract reasoning
        reasoning_pattern = r'Reasoning\s*(\d+)?:\s*(.+?)(?=(?:Evidence\s*\d+:|Reasoning\s*\d+:|$))'
        reasoning_matches = re.findall(reasoning_pattern, raw_output, re.DOTALL | re.IGNORECASE)
        for _, text in reasoning_matches:
            cleaned = text.strip()
            if cleaned:
                reasoning_list.append(cleaned)
                has_reasoning_format = True

        return ParsedOutput(
            answer=answer,
            evidence_snippets=evidence_snippets,
            reasoning_list=reasoning_list,
            raw_output=raw_output,
            has_answer_format=has_answer_format,
            has_evidence_format=has_evidence_format,
            has_reasoning_format=has_reasoning_format
        )


class MultiDimensionalRewardCalculator:
    """
    Calculate multi-dimensional rewards for PersonaMem task

    Reward = r_format + r_answer + r_evidence
    """

    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        self.parser = OutputParser()

    def calculate(self,
                  raw_output: str,
                  correct_label: str,
                  context: str,
                  relevant_indices: List[int]) -> Tuple[float, RewardBreakdown]:
        """
        Calculate total reward with breakdown

        Args:
            raw_output: Model's raw output string
            correct_label: Correct answer label (A/B/C/D)
            context: Retrieved context string
            relevant_indices: List of relevant item indices (1-indexed as in context)

        Returns:
            (total_reward, breakdown)
        """
        # Parse output
        parsed = self.parser.parse(raw_output)

        # Calculate each component
        r_format, format_details = self._calc_format_reward(parsed)
        r_answer, answer_correct = self._calc_answer_reward(parsed, correct_label)
        r_evidence, evidence_details = self._calc_evidence_reward(
            parsed, context, relevant_indices
        )

        # Total reward
        total = r_format + r_answer + r_evidence

        breakdown = RewardBreakdown(
            r_format=r_format,
            r_answer=r_answer,
            r_evidence=r_evidence,
            total=total,
            format_details=format_details,
            answer_correct=answer_correct,
            evidence_details=evidence_details
        )

        return total, breakdown

    def _calc_format_reward(self, parsed: ParsedOutput) -> Tuple[float, Dict[str, bool]]:
        """
        Calculate format reward

        Checks:
        - Has "Answer: [A-D]" format
        - Has "Evidence X:" structure
        - Has "Reasoning X:" structure
        """
        reward = 0.0
        details = {
            "has_answer_format": parsed.has_answer_format,
            "has_evidence_format": parsed.has_evidence_format,
            "has_reasoning_format": parsed.has_reasoning_format
        }

        if parsed.has_answer_format:
            reward += self.config.format_answer_reward

        if parsed.has_evidence_format:
            reward += self.config.format_evidence_reward

        if parsed.has_reasoning_format:
            reward += self.config.format_reasoning_reward

        # Cap at maximum
        reward = min(reward, self.config.format_max)

        return reward, details

    def _calc_answer_reward(self,
                            parsed: ParsedOutput,
                            correct_label: str) -> Tuple[float, bool]:
        """
        Calculate answer reward

        Simple binary reward based on answer correctness
        """
        is_correct = parsed.answer.upper() == correct_label.upper()

        if is_correct:
            reward = self.config.answer_correct_reward
        else:
            reward = self.config.answer_incorrect_reward

        return reward, is_correct

    def _calc_evidence_reward(self,
                              parsed: ParsedOutput,
                              context: str,
                              relevant_indices: List[int]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate evidence reward

        Checks if cited evidence comes from relevant context items

        Args:
            parsed: Parsed output
            context: Full context string with numbered items
            relevant_indices: Which item numbers are relevant (1-indexed)

        Returns:
            (reward, details)
        """
        if not parsed.evidence_snippets:
            return 0.0, {"num_evidence": 0, "relevant_citations": 0, "irrelevant_citations": 0}

        # Parse context into numbered items
        context_items = self._parse_context_items(context)
        content_to_indices: Dict[str, List[int]] = {}
        for idx, content in context_items.items():
            content_to_indices.setdefault(content, []).append(idx)

        relevant_citations = 0
        irrelevant_citations = 0

        for evidence in parsed.evidence_snippets:
            # Exact match only: evidence must equal a context item
            matched_indices = content_to_indices.get(evidence)
            if not matched_indices:
                continue

            if any(idx in relevant_indices for idx in matched_indices):
                relevant_citations += 1
            else:
                irrelevant_citations += 1

        # Calculate reward
        reward = (relevant_citations * self.config.evidence_relevant_reward +
                  irrelevant_citations * self.config.evidence_irrelevant_penalty)

        # Clamp to bounds
        reward = max(self.config.evidence_min, min(reward, self.config.evidence_max))

        details = {
            "num_evidence": len(parsed.evidence_snippets),
            "relevant_citations": relevant_citations,
            "irrelevant_citations": irrelevant_citations,
            "context_items_count": len(context_items)
        }

        return reward, details

    def _parse_context_items(self, context: str) -> Dict[int, str]:
        """
        Parse context into numbered items

        Context format:
        1. [Role]: Content...

        2. [Role]: Content...
        """
        items = {}

        # Pattern to match numbered items
        pattern = r'(\d+)\.\s*\[([^\]]+)\]:\s*(.+?)(?=\n\d+\.\s*\[|\Z)'
        matches = re.findall(pattern, context, re.DOTALL)

        for num_str, role, content in matches:
            idx = int(num_str)
            items[idx] = content.strip()

        return items

    def _find_best_match(self,
                         evidence: str,
                         context_items: Dict[int, str]) -> Tuple[int, float]:
        """
        Find which context item best matches the evidence

        Uses word overlap ratio for matching

        Returns:
            (best_matching_index, match_score)
        """
        if not context_items:
            return -1, 0.0

        evidence_words = set(evidence.lower().split())
        if not evidence_words:
            return -1, 0.0

        best_idx = -1
        best_score = 0.0

        for idx, content in context_items.items():
            content_words = set(content.lower().split())
            if not content_words:
                continue

            # Calculate overlap
            overlap = len(evidence_words & content_words)
            score = overlap / len(evidence_words)

            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx, best_score


# ============================================================================
# Reward Summary for Reference
# ============================================================================
"""
Reward Design Summary:

+------------------+------------------+------------------+
| Component        | Value Range      | Description      |
+------------------+------------------+------------------+
| r_format         | 0.0 - 0.2        | Output format    |
|   - Answer:      | +0.1             | Has "Answer: X"  |
|   - Evidence:    | +0.05            | Has Evidence     |
|   - Reasoning:   | +0.05            | Has Reasoning    |
+------------------+------------------+------------------+
| r_answer         | 0.0 or 1.0       | Answer correct   |
|   - Correct      | +1.0             |                  |
|   - Wrong        | +0.0             |                  |
+------------------+------------------+------------------+
| r_evidence       | -0.1 - 0.3       | Evidence quality |
|   - Relevant     | +0.1 each        | Cites relevant   |
|   - Irrelevant   | -0.05 each       | Cites noise      |
+------------------+------------------+------------------+
| TOTAL            | -0.1 - 1.5       | Sum of all       |
+------------------+------------------+------------------+

Example scenarios:
1. Perfect output (correct answer, 2 relevant evidences):
   R = 0.2 + 1.0 + 0.2 = 1.4

2. Correct answer, no evidence:
   R = 0.1 + 1.0 + 0.0 = 1.1

3. Correct answer, 1 relevant + 1 irrelevant evidence:
   R = 0.2 + 1.0 + (0.1 - 0.05) = 1.25

4. Wrong answer, good format:
   R = 0.2 + 0.0 + 0.0 = 0.2

5. Wrong answer, bad format:
   R = 0.0 + 0.0 + 0.0 = 0.0
"""


def test_reward_calculator():
    """Test the reward calculator"""
    calculator = MultiDimensionalRewardCalculator()

    # Test context
    context = """1. [User]: Hi, can you help me with something?

2. [Assistant]: This morning I found myself counting how far each shilling must stretch. There are weeks when expenses swell without warning.

3. [Assistant]: Nordic noir has great moody atmosphere.

4. [User]: Please help me improve this reflection about financial anxiety."""

    # Test output - correct answer with relevant evidence
    output_good = """Answer: B

Evidence and Reasoning:
Evidence 1: This morning I found myself counting how far each shilling must stretch. There are weeks when expenses swell without warning.
Reasoning 1: This shows the user experiences financial anxiety when unexpected expenses arise, which matches the question about managing expense-related anxiety.

Evidence 2: Please help me improve this reflection about financial anxiety.
Reasoning 2: The user has previously discussed financial concerns, indicating this is a relevant topic for them."""

    # Test output - wrong answer
    output_wrong = """Answer: C

Evidence and Reasoning:
Evidence 1: Nordic noir has great moody atmosphere.
Reasoning 1: This is about entertainment preferences."""

    # Test output - bad format
    output_bad = "I think the answer is probably B because of the context."

    print("=" * 60)
    print("Testing Reward Calculator")
    print("=" * 60)

    relevant_indices = [2, 4]

    # Test 1: Good output
    reward, breakdown = calculator.calculate(
        output_good, "B", context, relevant_indices
    )
    print(f"\n[Test 1] Correct answer + relevant evidence:")
    print(f"  r_format:   {breakdown.r_format:.2f}")
    print(f"  r_answer:   {breakdown.r_answer:.2f}")
    print(f"  r_evidence: {breakdown.r_evidence:.2f}")
    print(f"  TOTAL:      {breakdown.total:.2f}")
    print(f"  Details: {breakdown.evidence_details}")

    # Test 2: Wrong answer
    reward, breakdown = calculator.calculate(
        output_wrong, "B", context, relevant_indices
    )
    print(f"\n[Test 2] Wrong answer + irrelevant evidence:")
    print(f"  r_format:   {breakdown.r_format:.2f}")
    print(f"  r_answer:   {breakdown.r_answer:.2f}")
    print(f"  r_evidence: {breakdown.r_evidence:.2f}")
    print(f"  TOTAL:      {breakdown.total:.2f}")

    # Test 3: Bad format
    reward, breakdown = calculator.calculate(
        output_bad, "B", context, relevant_indices
    )
    print(f"\n[Test 3] Bad format:")
    print(f"  r_format:   {breakdown.r_format:.2f}")
    print(f"  r_answer:   {breakdown.r_answer:.2f}")
    print(f"  r_evidence: {breakdown.r_evidence:.2f}")
    print(f"  TOTAL:      {breakdown.total:.2f}")


if __name__ == "__main__":
    test_reward_calculator()
