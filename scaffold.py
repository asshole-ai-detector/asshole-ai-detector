
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Detect the package already created by uv
src = ROOT / "src"
packages = [
    p for p in src.iterdir()
    if p.is_dir() and (p / "__init__.py").exists()
] if src.exists() else []

if len(packages) > 1:
    raise RuntimeError(
        "Multiple packages found in src/. Select one manually."
    )

package = packages[0] if packages else src / "aita_detector"
package.mkdir(parents=True, exist_ok=True)

print(f"Using package: {package.relative_to(ROOT)}")

# Directories to create
directories = [
    ".github/workflows",
    ".github/ISSUE_TEMPLATE",
    "configs",
    "apps",
    "scripts",
    "notebooks",
    "tests/unit",
    "tests/integration",
    "tests/fixtures",
    "data",
    "artifacts",
    "docs",
]

modules = {
    "schemas": ["case", "result"],
    "data": ["aita_adapter", "preprocess", "splits", "deduplication"],
    "rag": ["embeddings", "index", "retrieval", "reranker"],
    "classifier": ["predict", "calibration"],
    "llm": ["client", "prompts", "baseline"],
    "agent": ["controller", "uncertainty", "bias"],
    "mcp": ["server", "client"],
    "evaluation": ["metrics", "runner"],
}

# Files outside the Python package
files = [
    ".github/workflows/ci.yml",
    ".github/ISSUE_TEMPLATE/task.yml",
    ".github/pull_request_template.md",
    "configs/default.yaml",
    "apps/streamlit_app.py",
    "scripts/prepare_data.py",
    "scripts/build_index.py",
    "scripts/run_experiments.py",
    "docs/architecture.md",
    "docs/annotation_guidelines.md",
    "tests/conftest.py",
]

for directory in directories:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)

for folder, module_names in modules.items():
    module_dir = package / folder
    module_dir.mkdir(parents=True, exist_ok=True)

    for name in ["__init__", *module_names]:
        file = module_dir / f"{name}.py"
        if not file.exists():
            file.write_text("", encoding="utf-8")
            print(f"Created {file.relative_to(ROOT)}")

for filename in files:
    file = ROOT / filename
    if not file.exists():
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("", encoding="utf-8")
        print(f"Created {filename}")

# Create valid, empty Jupyter notebooks
for name in [
    "data_exploration",
    "train_classifier",
    "rag_experiments",
]:
    file = ROOT / "notebooks" / f"{name}.ipynb"
    if not file.exists():
        notebook = {
            "cells": [],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        file.write_text(
            json.dumps(notebook, indent=2),
            encoding="utf-8"
        )

# Add important gitignore entries without overwriting existing ones
gitignore = ROOT / ".gitignore"
existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""

entries = [
    ".venv/",
    ".env",
    ".streamlit/secrets.toml",
    "data/",
    "artifacts/",
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
]

missing = [entry for entry in entries if entry not in existing.splitlines()]

if missing:
    with gitignore.open("a", encoding="utf-8") as f:
        f.write("\n" + "\n".join(missing) + "\n")

print("\nProject structure created successfully!")
