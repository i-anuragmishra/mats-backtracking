"""
I/O utilities for reading and writing data files.

Supports JSONL, JSON, and CSV formats with consistent interfaces.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator


# =============================================================================
# JSONL Operations
# =============================================================================

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Read a JSONL file into a list of dictionaries.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Iterate over a JSONL file line by line.
    
    Memory-efficient for large files.
    
    Args:
        path: Path to the JSONL file
        
    Yields:
        Dictionaries, one per line
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """
    Write a list of dictionaries to a JSONL file.
    
    Args:
        records: List of dictionaries to write
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """
    Append a single record to a JSONL file.
    
    Args:
        record: Dictionary to append
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# =============================================================================
# JSON Operations
# =============================================================================

def read_json(path: str | Path) -> dict[str, Any] | list[Any]:
    """
    Read a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Parsed JSON data
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict[str, Any] | list[Any], path: str | Path, indent: int = 2) -> None:
    """
    Write data to a JSON file.
    
    Args:
        data: Data to write
        path: Output path
        indent: Indentation level (default 2)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# =============================================================================
# CSV Operations
# =============================================================================

def read_csv(path: str | Path) -> list[dict[str, Any]]:
    """
    Read a CSV file into a list of dictionaries.
    
    Args:
        path: Path to the CSV file
        
    Returns:
        List of dictionaries with column names as keys
    """
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric strings to numbers where possible
            converted = {}
            for k, v in row.items():
                if v == "":
                    converted[k] = None
                elif v.lower() in ("true", "false"):
                    converted[k] = v.lower() == "true"
                else:
                    try:
                        # Try int first, then float
                        if "." in v:
                            converted[k] = float(v)
                        else:
                            converted[k] = int(v)
                    except ValueError:
                        converted[k] = v
            records.append(converted)
    return records


def write_csv(
    records: list[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str] | None = None,
) -> None:
    """
    Write a list of dictionaries to a CSV file.
    
    Args:
        records: List of dictionaries to write
        path: Output path
        fieldnames: Column order (default: keys from first record)
    """
    if not records:
        return
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if fieldnames is None:
        fieldnames = list(records[0].keys())
    
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# =============================================================================
# Utility Functions
# =============================================================================

def count_lines(path: str | Path) -> int:
    """
    Count the number of lines in a file.
    
    Args:
        path: Path to the file
        
    Returns:
        Number of lines
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def file_exists(path: str | Path) -> bool:
    """Check if a file exists."""
    return Path(path).exists()


