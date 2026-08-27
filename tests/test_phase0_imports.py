"""
Phase 0 smoke tests — confirm all top-level packages import cleanly
and the project structure is correct.
"""
import importlib
import os


EXPECTED_PACKAGES = [
    "data_gen",
    "detector",
    "agent",
    "agent.nodes",
    "eval",
    "dashboard",
]


def test_packages_importable():
    """Every top-level package must import without error."""
    for pkg in EXPECTED_PACKAGES:
        mod = importlib.import_module(pkg)
        assert mod is not None, f"Package '{pkg}' failed to import"


def test_env_example_exists():
    """The .env.example file must exist at the project root."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_example = os.path.join(root, ".env.example")
    assert os.path.isfile(env_example), ".env.example not found at project root"


def test_env_example_has_required_keys():
    """The .env.example must document all critical env vars."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_example = os.path.join(root, ".env.example")
    with open(env_example) as f:
        content = f.read()

    required_keys = [
        "GEMINI_API_KEY",
        "AUDIT_DB_PATH",
        "SPIKE_HARD_BLOCK_THRESHOLD",
        "DATAGEN_SEED",
        "EVAL_TEST_SPLIT",
    ]
    for key in required_keys:
        assert key in content, f"Missing required env var '{key}' in .env.example"


def test_directory_structure():
    """All required project directories must exist."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    required_dirs = [
        "data_gen",
        "detector",
        "agent",
        "agent/nodes",
        "eval",
        "dashboard",
        "tests",
    ]
    for d in required_dirs:
        path = os.path.join(root, d)
        assert os.path.isdir(path), f"Required directory '{d}' not found"


def test_pyproject_toml_exists():
    """pyproject.toml must exist and contain the project name."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pyproject = os.path.join(root, "pyproject.toml")
    assert os.path.isfile(pyproject), "pyproject.toml not found"
    with open(pyproject) as f:
        content = f.read()
    assert "spikegate" in content, "pyproject.toml must declare project name 'spikegate'"
    assert "xgboost" in content, "pyproject.toml must declare xgboost dependency"
    assert "langgraph" in content, "pyproject.toml must declare langgraph dependency"
