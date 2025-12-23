"""
CLI utilities for the backtracking project.
"""

import sys


def doctor():
    """Run diagnostic checks on the environment."""
    import subprocess
    subprocess.run([sys.executable, "scripts/doctor.sh"], check=False)


if __name__ == "__main__":
    doctor()

