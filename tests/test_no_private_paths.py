"""Repository hygiene tests for private paths and secrets."""

from __future__ import annotations

from pathlib import Path


TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".ini", ".csv"}


def iter_text_files() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    ignored = {".git", ".pytest_cache", "__pycache__"}
    files: list[Path] = []
    for path in root.rglob("*"):
        if any(part in ignored for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            files.append(path)
    return files


def test_no_private_workspace_paths() -> None:
    prohibited = ["/" + "Users/" + "du" + "zimo", "data/raw/" + "mobility data"]
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in prohibited:
            assert pattern not in text, f"{pattern!r} found in {path}"


def test_no_credential_like_patterns() -> None:
    lowered_patterns = ["api" + "_key", "secret" + "_key", "access" + "_token", "password" + "="]
    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for pattern in lowered_patterns:
            assert pattern not in text, f"{pattern!r} found in {path}"
