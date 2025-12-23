"""
Consistent tokenization utilities.

Single source of truth for all tokenization operations to ensure
consistency across generation, event detection, and teacher-forced passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class TokenizationResult:
    """Result of tokenizing text."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    offset_mapping: list[tuple[int, int]] | None
    has_bos: bool
    num_tokens: int
    
    def to(self, device: str | torch.device) -> "TokenizationResult":
        """Move tensors to device."""
        return TokenizationResult(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            offset_mapping=self.offset_mapping,
            has_bos=self.has_bos,
            num_tokens=self.num_tokens,
        )


def tokenize_for_model(
    tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
    text: str,
    add_special_tokens: bool = True,
    return_tensors: str = "pt",
) -> TokenizationResult:
    """
    Tokenize text with consistent settings.
    
    This function is the SINGLE SOURCE for tokenization across:
    - Generation input
    - Event offset mapping
    - Teacher-forced forward passes
    
    Args:
        tokenizer: HuggingFace tokenizer
        text: Text to tokenize
        add_special_tokens: Whether to add BOS/EOS tokens
        return_tensors: Tensor format ("pt" for PyTorch)
        
    Returns:
        TokenizationResult with input_ids, attention_mask, offset_mapping
    """
    # Check if tokenizer supports offset mapping
    try:
        encoding = tokenizer(
            text,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=True,
        )
        offset_mapping = encoding.get("offset_mapping")
        if offset_mapping is not None:
            # Convert from tensor to list of tuples
            offset_mapping = [tuple(x) for x in offset_mapping[0].tolist()]
    except Exception:
        # Fallback without offset mapping
        encoding = tokenizer(
            text,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )
        offset_mapping = None
    
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Check if BOS was added
    has_bos = (
        tokenizer.bos_token_id is not None
        and add_special_tokens
        and input_ids.shape[1] > 0
        and input_ids[0, 0].item() == tokenizer.bos_token_id
    )
    
    return TokenizationResult(
        input_ids=input_ids,
        attention_mask=attention_mask,
        offset_mapping=offset_mapping,
        has_bos=has_bos,
        num_tokens=input_ids.shape[1],
    )


def char_to_token_index(
    text: str,
    char_idx: int,
    tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
    prompt_len_chars: int = 0,
    add_special_tokens: bool = True,
) -> tuple[int, int, int, str]:
    """
    Map a character index to token index.
    
    Handles BOS tokens and special token offsets correctly.
    
    Args:
        text: Full text (prompt + completion)
        char_idx: Character index to map (in the full text)
        tokenizer: HuggingFace tokenizer
        prompt_len_chars: Length of prompt in characters
        add_special_tokens: Whether special tokens were added during generation
        
    Returns:
        Tuple of:
        - token_in_completion: Token index relative to completion start
        - token_in_full: Token index in full sequence
        - token_id: The actual token ID at that position
        - token_text: The decoded text of that token
    """
    result = tokenize_for_model(tokenizer, text, add_special_tokens=add_special_tokens)
    
    # Find the token containing char_idx using offset mapping
    token_in_full = -1
    
    if result.offset_mapping:
        for i, (start, end) in enumerate(result.offset_mapping):
            # Skip special tokens (offset (0, 0) at position 0 is usually BOS)
            if start == 0 and end == 0 and i == 0 and result.has_bos:
                continue
            if start <= char_idx < end:
                token_in_full = i
                break
            # If char_idx is exactly at a token boundary
            if char_idx == start:
                token_in_full = i
                break
        
        # If not found and char_idx is at end, use last token
        if token_in_full == -1 and char_idx >= len(text) - 1:
            token_in_full = result.num_tokens - 1
    else:
        # Fallback: binary search approximation
        # Encode prefix up to char_idx and count tokens
        prefix = text[:char_idx]
        prefix_result = tokenize_for_model(tokenizer, prefix, add_special_tokens=add_special_tokens)
        token_in_full = prefix_result.num_tokens
    
    # Calculate token index in completion
    # First, find how many tokens are in the prompt
    prompt_text = text[:prompt_len_chars]
    prompt_result = tokenize_for_model(tokenizer, prompt_text, add_special_tokens=add_special_tokens)
    prompt_tokens = prompt_result.num_tokens
    
    token_in_completion = max(0, token_in_full - prompt_tokens)
    
    # Get the actual token ID and text
    if 0 <= token_in_full < result.num_tokens:
        token_id = result.input_ids[0, token_in_full].item()
        token_text = tokenizer.decode([token_id])
    else:
        token_id = -1
        token_text = ""
    
    return token_in_completion, token_in_full, token_id, token_text


def get_prompt_token_count(
    prompt: str,
    tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
    add_special_tokens: bool = True,
) -> int:
    """
    Get the number of tokens in a prompt.
    
    Args:
        prompt: Prompt text
        tokenizer: HuggingFace tokenizer
        add_special_tokens: Whether to add special tokens
        
    Returns:
        Number of tokens
    """
    result = tokenize_for_model(tokenizer, prompt, add_special_tokens=add_special_tokens)
    return result.num_tokens


def get_token_at_position(
    text: str,
    position: int,
    tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
    add_special_tokens: bool = True,
) -> tuple[int, str]:
    """
    Get the token ID and text at a specific position.
    
    Args:
        text: Text to tokenize
        position: Token position (0-indexed)
        tokenizer: HuggingFace tokenizer
        add_special_tokens: Whether to add special tokens
        
    Returns:
        Tuple of (token_id, token_text)
    """
    result = tokenize_for_model(tokenizer, text, add_special_tokens=add_special_tokens)
    
    if 0 <= position < result.num_tokens:
        token_id = result.input_ids[0, position].item()
        token_text = tokenizer.decode([token_id])
        return token_id, token_text
    
    return -1, ""


def decode_tokens(
    token_ids: list[int] | torch.Tensor,
    tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
    skip_special_tokens: bool = True,
) -> str:
    """
    Decode token IDs to text.
    
    Args:
        token_ids: List or tensor of token IDs
        tokenizer: HuggingFace tokenizer
        skip_special_tokens: Whether to skip special tokens
        
    Returns:
        Decoded text
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

