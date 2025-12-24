"""
Prompt formatting utilities.

Handles formatting prompts with different formatting variants.
"""

from __future__ import annotations

from backtracking.config import FormattingVariant, PromptingConfig


def format_prompt(
    question: str,
    config: PromptingConfig,
    variant: FormattingVariant | None = None,
) -> str:
    """
    Format a question into a full prompt.
    
    Args:
        question: The question/problem text
        config: Prompting configuration
        variant: Optional formatting variant (if None, use base template)
        
    Returns:
        Formatted prompt string
    """
    # Start with base template
    prompt = config.template.format(question=question)
    
    # Apply formatting variant if specified
    if variant:
        # The variant modifies how the model should format its reasoning
        # For "no_think_tags", we remove the think tag instruction
        if variant.name == "no_think_tags":
            # Remove think tag references from prompt
            prompt = prompt.replace("Use <think> tags for your reasoning.\n", "")
            prompt = prompt.replace("Use <think> tags for your reasoning.", "")
        else:
            # For other variants, the think_open/think_close are used
            # in generation/detection, not in prompt modification
            pass
    
    return prompt


def format_prompt_with_system(
    question: str,
    config: PromptingConfig,
    variant: FormattingVariant | None = None,
    chat_template: bool = False,
) -> str:
    """
    Format a question with system prompt.
    
    Args:
        question: The question/problem text
        config: Prompting configuration
        variant: Optional formatting variant
        chat_template: Whether to use chat template format
        
    Returns:
        Formatted prompt with system message
    """
    user_prompt = format_prompt(question, config, variant)
    
    if chat_template:
        # Return as messages for chat template
        return [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    
    # Simple concatenation for non-chat models
    return f"{config.system_prompt}\n\n{user_prompt}"


def get_variant_by_name(
    config: PromptingConfig,
    name: str,
) -> FormattingVariant:
    """
    Get a formatting variant by name.
    
    Args:
        config: Prompting configuration
        name: Variant name to find
        
    Returns:
        FormattingVariant
        
    Raises:
        ValueError: If variant not found
    """
    for variant in config.formatting_variants:
        if variant.name == name:
            return variant
    
    available = [v.name for v in config.formatting_variants]
    raise ValueError(f"Variant '{name}' not found. Available: {available}")


def get_default_variant(config: PromptingConfig) -> FormattingVariant:
    """
    Get the default (first) formatting variant.
    
    Args:
        config: Prompting configuration
        
    Returns:
        First FormattingVariant in the list
    """
    if not config.formatting_variants:
        raise ValueError("No formatting variants configured")
    return config.formatting_variants[0]


def list_variants(config: PromptingConfig) -> list[str]:
    """
    List all available variant names.
    
    Args:
        config: Prompting configuration
        
    Returns:
        List of variant names
    """
    return [v.name for v in config.formatting_variants]


