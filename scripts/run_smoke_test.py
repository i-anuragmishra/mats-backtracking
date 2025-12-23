#!/usr/bin/env python3
"""
MATS-BACKTRACKING Smoke Test

Verifies that key libraries are installed and GPU is working.
This is a more thorough test than doctor.sh - it actually loads a model.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_status(name: str, status: bool, detail: str = "") -> None:
    """Print a status line with color."""
    symbol = "✅" if status else "❌"
    detail_str = f" ({detail})" if detail else ""
    print(f"  {symbol} {name}{detail_str}")


def test_basic_imports() -> bool:
    """Test that core libraries can be imported."""
    print_header("Testing Basic Imports")
    
    imports = [
        "torch",
        "transformers",
        "datasets",
        "einops",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",
        "rich",
        "wandb",
    ]
    
    all_ok = True
    for lib in imports:
        try:
            __import__(lib)
            print_status(lib, True)
        except ImportError as e:
            print_status(lib, False, str(e))
            all_ok = False
    
    return all_ok


def test_cuda() -> bool:
    """Test CUDA availability and basic operations."""
    print_header("Testing CUDA")
    
    import torch
    
    if not torch.cuda.is_available():
        print_status("CUDA available", False, "torch.cuda.is_available() = False")
        return False
    
    print_status("CUDA available", True)
    print_status(f"Device count: {torch.cuda.device_count()}", True)
    print_status(f"Device name: {torch.cuda.get_device_name(0)}", True)
    
    # Test tensor operations on GPU
    try:
        x = torch.randn(1000, 1000, device="cuda")
        y = torch.randn(1000, 1000, device="cuda")
        
        start = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print_status(f"Matrix multiply (1000x1000)", True, f"{elapsed*1000:.2f}ms")
        
        # Memory check
        allocated = torch.cuda.memory_allocated() / 1024**2
        print_status(f"Memory allocated: {allocated:.1f} MB", True)
        
        del x, y, z
        torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        print_status("CUDA tensor operations", False, str(e))
        return False


def test_transformers() -> bool:
    """Test that transformers can load a tiny model."""
    print_header("Testing Transformers")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Use a tiny model for testing
        model_name = "sshleifer/tiny-gpt2"
        
        print(f"  Loading tiny test model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print_status("Tokenizer loaded", True)
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print_status("Model loaded", True)
        
        # Move to GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print_status(f"Model on {device}", True)
        
        # Test forward pass
        inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        print_status("Forward pass", True, f"logits shape: {outputs.logits.shape}")
        
        # Cleanup
        del model, tokenizer, inputs, outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print_status("Transformers test", False, str(e))
        return False


def test_project_structure() -> bool:
    """Verify project structure is correct."""
    print_header("Testing Project Structure")
    
    project_root = Path(__file__).parent.parent
    
    expected_dirs = [
        "src/backtracking",
        "scripts",
        "notebooks",
        "configs",
        "data/raw",
        "data/interim",
        "data/processed",
        "results",
        "figures",
        "reports",
        "logs",
        "scratch",
        "runs",
        ".cache",
    ]
    
    all_ok = True
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        exists = full_path.exists()
        print_status(dir_path, exists)
        if not exists:
            all_ok = False
    
    return all_ok


def test_environment_vars() -> bool:
    """Check that cache environment variables are set."""
    print_header("Testing Environment Variables")
    
    import os
    
    vars_to_check = [
        "HF_HOME",
        "TORCH_HOME",
        "WANDB_DIR",
    ]
    
    all_ok = True
    for var in vars_to_check:
        value = os.environ.get(var)
        if value:
            print_status(var, True, value)
        else:
            print_status(var, False, "not set")
            all_ok = False
    
    return all_ok


def main() -> int:
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("  MATS-BACKTRACKING SMOKE TEST")
    print("=" * 60)
    
    results = {}
    
    results["imports"] = test_basic_imports()
    results["cuda"] = test_cuda()
    results["transformers"] = test_transformers()
    results["structure"] = test_project_structure()
    results["env_vars"] = test_environment_vars()
    
    # Summary
    print_header("Summary")
    
    all_passed = True
    for test_name, passed in results.items():
        print_status(test_name, passed)
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All smoke tests passed! Environment is ready for research.\n")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

