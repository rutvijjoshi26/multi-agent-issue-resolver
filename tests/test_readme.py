# tests/test_readme.py
from pathlib import Path

from miresolver.rules import (
    EXCLUDED_FILENAMES,
    EXCLUDED_PATH_SEGMENTS,
    EXCLUDED_SUFFIXES,
    MAX_CHANGED_FILES,
    MIN_CHANGED_FILES,
    TEST_FILENAME_MARKERS,
    TEST_FILENAME_PREFIXES,
)

README = Path(__file__).parent.parent / "README.md"


def test_readme_documents_every_inclusion_rule():
    text = README.read_text().lower()
    assert "bug label" in text
    assert "exactly one" in text and "merged" in text
    assert f"{MIN_CHANGED_FILES} and {MAX_CHANGED_FILES}" in text
    assert "revert" in text


def test_readme_states_the_leakage_control():
    text = README.read_text().lower()
    assert "before the fix" in text
    # The control is only implementable if the manifest names the commit.
    assert "base_commit" in text


def test_readme_documents_the_source_file_exclusions():
    text = README.read_text().lower()
    for segment in EXCLUDED_PATH_SEGMENTS:
        assert f"`{segment}`" in text, f"undocumented excluded directory: {segment}"
    for suffix in EXCLUDED_SUFFIXES:
        assert f"`{suffix}`" in text, f"undocumented excluded suffix: {suffix}"
    for marker in TEST_FILENAME_MARKERS + TEST_FILENAME_PREFIXES:
        assert f"`{marker}`" in text, f"undocumented test marker: {marker}"
    for filename in EXCLUDED_FILENAMES:
        assert f"`{filename}`" in text, f"undocumented excluded filename: {filename}"
    assert "lockfile" in text


def test_readme_documents_every_ingest_flag():
    text = README.read_text()
    for flag in (
        "--limit",
        "--dev-size",
        "--full-size",
        "--seed",
        "--allow-undersized",
        "--skip-db",
        "--out",
    ):
        assert flag in text, f"undocumented flag: {flag}"
