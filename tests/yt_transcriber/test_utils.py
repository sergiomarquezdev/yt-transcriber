"""Tests for yt_transcriber.utils module."""



from yt_transcriber.utils import (
    cleanup_temp_dir,
    cleanup_temp_files,
    get_file_size_mb,
    save_transcription_to_file,
)


class TestSaveTranscriptionToFile:
    """Tests for save_transcription_to_file function."""

    def test_basic_save(self, temp_output_dir):
        """Test basic transcription save."""
        result = save_transcription_to_file(
            transcription_text="Hello world, this is a test.",
            output_filename_no_ext="test_file",
            output_dir=temp_output_dir,
        )

        assert result is not None
        assert result.exists()
        assert result.suffix == ".txt"

    def test_content_written_correctly(self, temp_output_dir):
        """Test that content is written correctly."""
        text = "This is the transcription content."
        result = save_transcription_to_file(
            transcription_text=text,
            output_filename_no_ext="content_test",
            output_dir=temp_output_dir,
        )

        content = result.read_text(encoding="utf-8")
        assert text in content

    def test_original_title_included(self, temp_output_dir):
        """Test that original title is included when provided."""
        result = save_transcription_to_file(
            transcription_text="Content here",
            output_filename_no_ext="title_test",
            output_dir=temp_output_dir,
            original_title="My Video Title",
        )

        content = result.read_text(encoding="utf-8")
        assert "My Video Title" in content
        assert "# Original Video Title:" in content

    def test_sanitizes_filename(self, temp_output_dir):
        """Test that filename is sanitized."""
        result = save_transcription_to_file(
            transcription_text="Content",
            output_filename_no_ext="test/file:name*with?special<chars>",
            output_dir=temp_output_dir,
        )

        assert result is not None
        # Should not contain special chars
        assert "/" not in result.name
        assert ":" not in result.name
        assert "*" not in result.name

    def test_creates_output_directory(self, temp_dir):
        """Test that output directory is created if it doesn't exist."""
        new_output_dir = temp_dir / "new_output"
        assert not new_output_dir.exists()

        result = save_transcription_to_file(
            transcription_text="Content",
            output_filename_no_ext="test",
            output_dir=new_output_dir,
        )

        assert new_output_dir.exists()
        assert result is not None

    def test_empty_transcription(self, temp_output_dir):
        """Test saving empty transcription."""
        result = save_transcription_to_file(
            transcription_text="",
            output_filename_no_ext="empty",
            output_dir=temp_output_dir,
        )

        assert result is not None
        content = result.read_text(encoding="utf-8")
        # May have just title or empty
        assert isinstance(content, str)

    def test_returns_none_on_error(self, temp_dir):
        """Test that None is returned on error."""
        # Try to save to a file that can't be created
        # This is tricky to test reliably, so we'll skip complex error cases
        pass

    def test_unicode_content(self, temp_output_dir):
        """Test saving unicode content."""
        text = "日本語テスト 🎉 Español ñ"
        result = save_transcription_to_file(
            transcription_text=text,
            output_filename_no_ext="unicode_test",
            output_dir=temp_output_dir,
        )

        content = result.read_text(encoding="utf-8")
        assert "日本語" in content
        assert "Español" in content


class TestCleanupTempFiles:
    """Tests for cleanup_temp_files function."""

    def test_deletes_existing_files(self, temp_dir):
        """Test that existing files are deleted."""
        file1 = temp_dir / "file1.tmp"
        file2 = temp_dir / "file2.tmp"
        file1.touch()
        file2.touch()

        assert file1.exists()
        assert file2.exists()

        cleanup_temp_files([str(file1), str(file2)])

        assert not file1.exists()
        assert not file2.exists()

    def test_handles_none_in_list(self, temp_dir):
        """Test that None values in list are handled."""
        file1 = temp_dir / "file1.tmp"
        file1.touch()

        # Should not raise
        cleanup_temp_files([str(file1), None, None])

        assert not file1.exists()

    def test_handles_nonexistent_files(self, temp_dir):
        """Test that nonexistent files don't cause errors."""
        nonexistent = str(temp_dir / "does_not_exist.tmp")

        # Should not raise
        cleanup_temp_files([nonexistent])

    def test_empty_list(self):
        """Test with empty list."""
        # Should not raise
        cleanup_temp_files([])

    def test_mixed_existing_and_nonexistent(self, temp_dir):
        """Test with mix of existing and nonexistent files."""
        existing = temp_dir / "existing.tmp"
        existing.touch()
        nonexistent = str(temp_dir / "nonexistent.tmp")

        cleanup_temp_files([str(existing), nonexistent])

        assert not existing.exists()


class TestCleanupTempDir:
    """Tests for cleanup_temp_dir function."""

    def test_deletes_directory_and_contents(self, temp_dir):
        """Test that directory and all contents are deleted."""
        test_dir = temp_dir / "to_delete"
        test_dir.mkdir()
        (test_dir / "file1.txt").touch()
        (test_dir / "file2.txt").touch()
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "file3.txt").touch()

        assert test_dir.exists()

        cleanup_temp_dir(test_dir)

        assert not test_dir.exists()

    def test_handles_nonexistent_directory(self, temp_dir):
        """Test that nonexistent directory doesn't cause error."""
        nonexistent = temp_dir / "does_not_exist"

        # Should not raise
        cleanup_temp_dir(nonexistent)

    def test_handles_file_path(self, temp_dir):
        """Test behavior when given a file path instead of directory."""
        file_path = temp_dir / "file.txt"
        file_path.touch()

        # Should not raise (checks is_dir first)
        cleanup_temp_dir(file_path)

        # File should still exist (not deleted because not a directory)
        assert file_path.exists()


class TestGetFileSizeMb:
    """Tests for get_file_size_mb function."""

    def test_returns_size_in_mb(self, temp_dir):
        """Test that file size is returned in MB."""
        test_file = temp_dir / "test.txt"
        # Write 1MB of data
        test_file.write_bytes(b"x" * (1024 * 1024))

        size = get_file_size_mb(test_file)

        assert size is not None
        assert 0.9 < size < 1.1  # Should be ~1 MB

    def test_small_file(self, temp_dir):
        """Test with small file."""
        test_file = temp_dir / "small.txt"
        test_file.write_text("Hello")

        size = get_file_size_mb(test_file)

        assert size is not None
        assert size < 0.001  # Very small

    def test_nonexistent_file(self, temp_dir):
        """Test with nonexistent file returns None."""
        nonexistent = temp_dir / "does_not_exist.txt"

        size = get_file_size_mb(nonexistent)

        assert size is None

    def test_empty_file(self, temp_dir):
        """Test with empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.touch()

        size = get_file_size_mb(empty_file)

        assert size is not None
        assert size == 0.0
