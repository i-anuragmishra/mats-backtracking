"""
Backtracking detection utilities.

Detects backtracking triggers in completions and maps them to token positions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from backtracking.config import DetectionConfig
from backtracking.tokenization import char_to_token_index

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class BacktrackingDetection:
    """Result of backtracking detection for a single completion."""
    has_backtracking_strict: bool
    has_backtracking_relaxed: bool
    onset_phrase: str | None
    onset_char: int  # Character index in completion where onset begins (-1 if none)
    onset_token_in_completion: int  # Token index relative to completion start
    onset_token_in_full: int  # Token index in full sequence (prompt + completion)
    onset_token_id: int  # Actual token ID at onset position
    onset_token_text: str  # Decoded text of the onset token
    pred_pos_in_full: int  # Position to predict onset token (onset_token_in_full - 1)


def build_trigger_pattern(triggers: list[str], case_insensitive: bool = False) -> re.Pattern:
    """
    Build a regex pattern for trigger detection.
    
    Uses word boundaries for robust matching.
    
    Args:
        triggers: List of trigger phrases
        case_insensitive: Whether to ignore case
        
    Returns:
        Compiled regex pattern
    """
    # Escape special characters and join with OR
    escaped = [re.escape(t) for t in triggers]
    pattern = r'\b(' + '|'.join(escaped) + r')\b'
    
    flags = re.IGNORECASE if case_insensitive else 0
    return re.compile(pattern, flags)


def find_earliest_trigger(
    text: str,
    triggers: list[str],
    priority_order: list[str] | None = None,
) -> tuple[str | None, int]:
    """
    Find the earliest trigger phrase in text.
    
    Args:
        text: Text to search
        triggers: List of trigger phrases
        priority_order: If multiple triggers at same position, prefer this order
        
    Returns:
        Tuple of (trigger_phrase, char_index) or (None, -1) if not found
    """
    if not triggers:
        return None, -1
    
    pattern = build_trigger_pattern(triggers, case_insensitive=True)
    
    earliest_match = None
    earliest_pos = len(text) + 1
    
    for match in pattern.finditer(text):
        if match.start() < earliest_pos:
            earliest_pos = match.start()
            earliest_match = match.group(0)
        elif match.start() == earliest_pos and priority_order:
            # Same position - check priority
            current_priority = _get_priority(earliest_match, priority_order)
            new_priority = _get_priority(match.group(0), priority_order)
            if new_priority < current_priority:
                earliest_match = match.group(0)
    
    if earliest_match is not None:
        return earliest_match, earliest_pos
    
    return None, -1


def _get_priority(phrase: str, priority_order: list[str]) -> int:
    """Get priority index for a phrase (lower = higher priority)."""
    phrase_lower = phrase.lower()
    for i, p in enumerate(priority_order):
        if p.lower() == phrase_lower:
            return i
    return len(priority_order)


def detect_backtracking(
    completion: str,
    prompt: str,
    full_text: str,
    tokenizer: "PreTrainedTokenizer",
    config: DetectionConfig,
) -> BacktrackingDetection:
    """
    Detect backtracking in a completion and map to token positions.
    
    Args:
        completion: The model's completion text
        prompt: The original prompt
        full_text: Full text (prompt + completion)
        tokenizer: HuggingFace tokenizer
        config: Detection configuration
        
    Returns:
        BacktrackingDetection with all fields populated
    """
    prompt_len_chars = len(prompt)
    
    # Check strict triggers
    strict_phrase, strict_pos = find_earliest_trigger(
        completion,
        config.triggers_strict,
        config.onset_priority,
    )
    has_strict = strict_phrase is not None
    
    # Check relaxed triggers
    relaxed_phrase, relaxed_pos = find_earliest_trigger(
        completion,
        config.triggers_relaxed,
        config.onset_priority,
    )
    has_relaxed = relaxed_phrase is not None
    
    # Use strict match if available, else relaxed
    if has_strict:
        onset_phrase = strict_phrase
        onset_char_in_completion = strict_pos
    elif has_relaxed:
        onset_phrase = relaxed_phrase
        onset_char_in_completion = relaxed_pos
    else:
        onset_phrase = None
        onset_char_in_completion = -1
    
    # Map character index to token index
    if onset_char_in_completion >= 0:
        # Character position in full text
        onset_char_in_full = prompt_len_chars + onset_char_in_completion
        
        # Get token indices and ID
        token_in_completion, token_in_full, token_id, token_text = char_to_token_index(
            text=full_text,
            char_idx=onset_char_in_full,
            tokenizer=tokenizer,
            prompt_len_chars=prompt_len_chars,
        )
        
        pred_pos = token_in_full - 1 if token_in_full > 0 else 0
    else:
        token_in_completion = -1
        token_in_full = -1
        token_id = -1
        token_text = ""
        pred_pos = -1
    
    return BacktrackingDetection(
        has_backtracking_strict=has_strict,
        has_backtracking_relaxed=has_relaxed,
        onset_phrase=onset_phrase,
        onset_char=onset_char_in_completion,
        onset_token_in_completion=token_in_completion,
        onset_token_in_full=token_in_full,
        onset_token_id=token_id,
        onset_token_text=token_text,
        pred_pos_in_full=pred_pos,
    )


def extract_final_answer(
    completion: str,
    regex_pattern: str | None = None,
) -> str | None:
    """
    Extract the final answer from a completion.
    
    Args:
        completion: The model's completion
        regex_pattern: Optional custom regex (if None, use default heuristics)
        
    Returns:
        Extracted answer string or None
    """
    if regex_pattern:
        match = re.search(regex_pattern, completion)
        if match:
            return match.group(1) if match.groups() else match.group(0)
        return None
    
    # Default: look for "Final:" pattern
    final_patterns = [
        r'Final:\s*(.+?)(?:\n|$)',
        r'final answer[:\s]+(.+?)(?:\n|$)',
        r'answer[:\s]+(.+?)(?:\n|$)',
    ]
    
    for pattern in final_patterns:
        match = re.search(pattern, completion, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up common artifacts
            answer = re.sub(r'[<>\[\]]', '', answer)
            return answer
    
    # Fallback: extract last number in the completion
    return extract_last_number(completion)


def extract_last_number(text: str) -> str | None:
    """
    Extract the last number from text.
    
    Handles integers and decimals, with or without commas.
    
    Args:
        text: Text to search
        
    Returns:
        Last number as string, or None
    """
    # Pattern for numbers (with optional commas and decimals)
    pattern = r'[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return last match, removing commas
        return matches[-1].replace(',', '')
    
    return None


def normalize_answer(answer: str | None) -> str | None:
    """
    Normalize an answer for comparison.
    
    Args:
        answer: Answer string
        
    Returns:
        Normalized answer
    """
    if answer is None:
        return None
    
    # Convert to string and strip
    answer = str(answer).strip()
    
    # Remove currency symbols
    answer = re.sub(r'[$€£¥]', '', answer)
    
    # Remove commas in numbers
    answer = answer.replace(',', '')
    
    # Remove trailing periods
    answer = answer.rstrip('.')
    
    # Try to convert to float for numeric comparison
    try:
        num = float(answer)
        # Handle infinity (number too large)
        if num == float('inf') or num == float('-inf'):
            return answer.lower()
        # Return as int if whole number
        if num == int(num):
            return str(int(num))
        return str(num)
    except (ValueError, OverflowError):
        return answer.lower()


def check_answer_correct(
    predicted: str | None,
    gold: str | None,
) -> bool | None:
    """
    Check if predicted answer matches gold answer.
    
    Args:
        predicted: Predicted answer
        gold: Gold answer
        
    Returns:
        True if correct, False if incorrect, None if cannot determine
    """
    if predicted is None or gold is None:
        return None
    
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    
    if pred_norm is None or gold_norm is None:
        return None
    
    return pred_norm == gold_norm


