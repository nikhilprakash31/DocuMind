"""Unit tests for theme persistence manager in DocuMind UI."""

import json
from pathlib import Path

from src.frontend.theme import DEFAULT_THEME, load_theme, save_theme


def test_default_theme_when_file_missing(tmp_path: Path) -> None:
    """When theme configuration file does not exist, default to 'light'."""
    missing_file = tmp_path / ".missing_theme.json"
    assert not missing_file.exists()
    assert load_theme(missing_file) == DEFAULT_THEME


def test_valid_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Verify saving and loading valid themes ('light' and 'dark')."""
    theme_file = tmp_path / ".streamlit_theme.json"

    # Save 'dark'
    assert save_theme("dark", path=theme_file) is True
    assert theme_file.exists()
    assert load_theme(theme_file) == "dark"

    # Save 'light'
    assert save_theme("light", path=theme_file) is True
    assert load_theme(theme_file) == "light"


def test_invalid_theme_value_rejected(tmp_path: Path) -> None:
    """Reject invalid theme values on save and keep default or existing value."""
    theme_file = tmp_path / ".streamlit_theme.json"

    # Attempt to save invalid themes
    assert save_theme("blue", path=theme_file) is False
    assert save_theme("", path=theme_file) is False
    assert save_theme("DARK", path=theme_file) is False  # case-sensitive
    assert not theme_file.exists()

    # Save valid 'dark' first, then try invalid save
    assert save_theme("dark", path=theme_file) is True
    assert save_theme("neon", path=theme_file) is False
    # File content must still be 'dark'
    assert load_theme(theme_file) == "dark"


def test_load_with_invalid_theme_value_in_json(tmp_path: Path) -> None:
    """If the JSON file exists but contains an unexpected theme value, default to 'light'."""
    theme_file = tmp_path / ".streamlit_theme.json"
    theme_file.write_text(json.dumps({"theme": "system"}), encoding="utf-8")
    assert load_theme(theme_file) == DEFAULT_THEME

    theme_file.write_text(json.dumps({"theme": 42}), encoding="utf-8")
    assert load_theme(theme_file) == DEFAULT_THEME

    theme_file.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    assert load_theme(theme_file) == DEFAULT_THEME


def test_load_with_malformed_json(tmp_path: Path) -> None:
    """If the JSON file is truncated or syntax-broken, safely default to 'light'."""
    theme_file = tmp_path / ".streamlit_theme.json"
    theme_file.write_text("{broken json: 123", encoding="utf-8")
    assert load_theme(theme_file) == DEFAULT_THEME


def test_load_with_empty_file(tmp_path: Path) -> None:
    """If the JSON file is completely empty, safely default to 'light'."""
    theme_file = tmp_path / ".streamlit_theme.json"
    theme_file.write_text("", encoding="utf-8")
    assert load_theme(theme_file) == DEFAULT_THEME


def test_atomic_persistence(tmp_path: Path) -> None:
    """Verify atomic write does not leave partial/temporary files behind."""
    theme_file = tmp_path / ".streamlit_theme.json"

    # Save theme
    assert save_theme("dark", path=theme_file) is True

    # Check directory contents: only theme_file, no temporary files left behind
    entries = list(tmp_path.iterdir())
    assert len(entries) == 1
    assert entries[0] == theme_file

    # Validate json structure and formatting
    data = json.loads(theme_file.read_text(encoding="utf-8"))
    assert data == {"theme": "dark"}
