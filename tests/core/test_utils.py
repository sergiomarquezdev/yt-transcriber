"""Tests for core.utils module."""


import pytest

from core.utils import ensure_dir_exists, normalize_title_for_filename


class TestNormalizeTitleForFilename:
    """Tests for normalize_title_for_filename function."""

    def test_basic_title(self):
        """Test basic title normalization."""
        assert normalize_title_for_filename("Hello World") == "Hello_World"

    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        assert normalize_title_for_filename("Hello! World?") == "Hello_World"
        assert normalize_title_for_filename("Test@#$%Video") == "TestVideo"

    def test_emojis_removed(self):
        """Test that emojis are removed."""
        result = normalize_title_for_filename("Test 🚀 Video 🎉")
        assert "🚀" not in result
        assert "🎉" not in result

    def test_multiple_spaces_collapsed(self):
        """Test that multiple spaces become single underscore."""
        assert normalize_title_for_filename("Hello   World") == "Hello_World"

    def test_leading_trailing_underscores_removed(self):
        """Test that leading/trailing underscores are stripped."""
        assert normalize_title_for_filename("  Hello World  ") == "Hello_World"
        assert normalize_title_for_filename("___Test___") == "Test"

    def test_empty_string_returns_untitled(self):
        """Test that empty string returns 'untitled'."""
        assert normalize_title_for_filename("") == "untitled"

    def test_only_special_chars_returns_untitled(self):
        """Test that string with only special chars returns 'untitled'."""
        assert normalize_title_for_filename("!@#$%^&*()") == "untitled"

    def test_hyphens_preserved(self):
        """Test that hyphens are preserved."""
        assert normalize_title_for_filename("hello-world") == "hello-world"

    def test_unicode_letters_preserved(self):
        """Test that unicode letters are preserved."""
        result = normalize_title_for_filename("Español Video")
        assert "Espa" in result

    def test_numbers_preserved(self):
        """Test that numbers are preserved."""
        assert normalize_title_for_filename("Test123Video") == "Test123Video"
        assert normalize_title_for_filename("2024 Tutorial") == "2024_Tutorial"

    def test_long_title_not_truncated(self):
        """Test that long titles are not truncated by this function."""
        long_title = "A" * 200
        result = normalize_title_for_filename(long_title)
        assert len(result) == 200


class TestEnsureDirExists:
    """Tests for ensure_dir_exists function."""

    def test_creates_new_directory(self, temp_dir):
        """Test that a new directory is created."""
        new_dir = temp_dir / "new_folder"
        assert not new_dir.exists()

        ensure_dir_exists(new_dir)

        assert new_dir.exists()
        assert new_dir.is_dir()

    def test_creates_nested_directories(self, temp_dir):
        """Test that nested directories are created."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        assert not nested_dir.exists()

        ensure_dir_exists(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_existing_directory_no_error(self, temp_dir):
        """Test that existing directory doesn't raise error."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        assert existing_dir.exists()

        # Should not raise
        ensure_dir_exists(existing_dir)

        assert existing_dir.exists()

    def test_permission_error_raises(self, temp_dir, monkeypatch):
        """Test that permission errors are raised."""
        import os

        # Skip on Windows where permission handling is different
        if os.name == "nt":
            pytest.skip("Permission test not reliable on Windows")

        protected_dir = temp_dir / "protected"
        protected_dir.mkdir()
        protected_dir.chmod(0o000)

        try:
            new_dir = protected_dir / "cant_create"
            with pytest.raises(OSError):
                ensure_dir_exists(new_dir)
        finally:
            # Restore permissions for cleanup
            protected_dir.chmod(0o755)
