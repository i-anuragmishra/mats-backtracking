# =============================================================================
# MATS-BACKTRACKING Makefile
# Common commands for development and research workflow
# =============================================================================

.PHONY: help install install-all sync activate doctor smoke lint format test \
        notebook clean clean-cache snapshot push backup

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║           MATS-BACKTRACKING Development Commands                 ║"
	@echo "╠══════════════════════════════════════════════════════════════════╣"
	@echo "║  SETUP                                                           ║"
	@echo "║    make install      Install core dependencies                   ║"
	@echo "║    make install-all  Install all deps including optional         ║"
	@echo "║    make sync         Sync environment from lockfile              ║"
	@echo "║    make activate     Print activation command                    ║"
	@echo "║                                                                  ║"
	@echo "║  DEVELOPMENT                                                     ║"
	@echo "║    make doctor       Run environment diagnostics                 ║"
	@echo "║    make smoke        Run smoke test (GPU + imports)              ║"
	@echo "║    make lint         Run linter (ruff)                           ║"
	@echo "║    make format       Auto-format code                            ║"
	@echo "║    make test         Run pytest                                  ║"
	@echo "║    make notebook     Start Jupyter notebook server               ║"
	@echo "║                                                                  ║"
	@echo "║  PERSISTENCE                                                     ║"
	@echo "║    make snapshot     Create timestamped run snapshot             ║"
	@echo "║    make push         Commit and push to origin                   ║"
	@echo "║    make backup       Snapshot + push (full backup)               ║"
	@echo "║                                                                  ║"
	@echo "║  CLEANUP                                                         ║"
	@echo "║    make clean        Remove build artifacts                      ║"
	@echo "║    make clean-cache  Clear all caches (HF, torch, pip)           ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

install:
	@echo "📦 Installing dependencies..."
	source scripts/activate_env.sh && uv sync

install-all:
	@echo "📦 Installing all dependencies (including optional)..."
	source scripts/activate_env.sh && uv sync --all-extras

sync:
	@echo "🔄 Syncing from lockfile..."
	source scripts/activate_env.sh && uv sync

activate:
	@echo "Run this command to activate the environment:"
	@echo ""
	@echo "    source scripts/activate_env.sh"
	@echo ""

# -----------------------------------------------------------------------------
# Development
# -----------------------------------------------------------------------------

doctor:
	@echo "🔍 Running environment diagnostics..."
	@bash scripts/doctor.sh

smoke:
	@echo "🔥 Running smoke test..."
	@source scripts/activate_env.sh && python scripts/run_smoke_test.py

lint:
	@echo "🔎 Running linter..."
	@source scripts/activate_env.sh && uv run ruff check src/ scripts/

format:
	@echo "✨ Formatting code..."
	@source scripts/activate_env.sh && uv run ruff format src/ scripts/
	@source scripts/activate_env.sh && uv run ruff check --fix src/ scripts/

test:
	@echo "🧪 Running tests..."
	@source scripts/activate_env.sh && uv run pytest

notebook:
	@echo "📓 Starting Jupyter notebook..."
	@source scripts/activate_env.sh && uv run jupyter notebook --no-browser --ip=0.0.0.0

# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------

snapshot:
	@echo "📸 Creating run snapshot..."
	@bash scripts/snapshot_run.sh

push:
	@echo "🚀 Committing and pushing..."
	@bash scripts/push_all.sh

backup: snapshot push
	@echo "✅ Backup complete (snapshot + push)"

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-cache:
	@echo "🗑️  Clearing caches (this will require re-downloading models)..."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf .cache/ || echo "Cancelled"

