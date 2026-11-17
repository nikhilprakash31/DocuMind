"""Theme persistence manager for DocuMind UI.

Handles saving and loading the user's selected theme ('light' or 'dark')
to a JSON configuration file. Provides atomic writes to prevent file
corruption and validates loaded values.
"""

import json
import tempfile
from pathlib import Path

from loguru import logger

DEFAULT_THEME_FILE = Path(".streamlit_theme.json")
VALID_THEMES = {"light", "dark"}
DEFAULT_THEME = "light"


def load_theme(path: Path | str = DEFAULT_THEME_FILE) -> str:
    """Load persisted theme preference ('light' or 'dark').

    Returns 'light' if the file is missing, empty, malformed, or
    contains an unsupported theme value.

    Args:
        path: Filepath to the theme JSON configuration file.

    Returns:
        The validated theme ('light' or 'dark').
    """
    file_path = Path(path)
    if not file_path.is_file():
        return DEFAULT_THEME

    try:
        raw_text = file_path.read_text(encoding="utf-8").strip()
        if not raw_text:
            return DEFAULT_THEME

        data = json.loads(raw_text)
        if isinstance(data, dict):
            theme = data.get("theme")
            if theme in VALID_THEMES:
                return theme
        logger.warning(
            "Invalid theme value in {}: {}. Defaulting to '{}'.",
            file_path,
            data,
            DEFAULT_THEME,
        )
    except Exception as e:
        logger.warning(
            "Failed to load theme from {}: {}. Defaulting to '{}'.",
            file_path,
            e,
            DEFAULT_THEME,
        )

    return DEFAULT_THEME


def save_theme(theme: str, path: Path | str = DEFAULT_THEME_FILE) -> bool:
    """Safely persist theme preference to disk using an atomic write.

    Writes to a temporary file in the target directory and replaces the
    destination file to ensure partially written files are never left behind.

    Args:
        theme: Theme name ('light' or 'dark').
        path: Target path for the theme storage JSON file.

    Returns:
        True if persisted successfully, False otherwise.
    """
    if theme not in VALID_THEMES:
        logger.warning(
            "Rejected saving invalid theme '{}'. Must be one of {}.",
            theme,
            VALID_THEMES,
        )
        return False

    file_path = Path(path).resolve()
    dir_path = file_path.parent
    dir_path.mkdir(parents=True, exist_ok=True)

    try:
        content = json.dumps({"theme": theme}, indent=2) + "\n"
        # Atomic write pattern: temporary file in same filesystem, then atomic rename
        with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, encoding="utf-8") as tf:
            tf.write(content)
            temp_path = Path(tf.name)

        temp_path.replace(file_path)
        return True
    except Exception as e:
        logger.error("Failed to save theme to {}: {}", file_path, e)
        return False
