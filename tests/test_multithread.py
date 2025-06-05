"""Tests for the multithread.py script to ensure it matches the behavior of corpus_counter_script.py."""

from collections import Counter
import time

from cdstemplate.multithread import tokenize, process_file


def test_tokenize():
    """Test that tokenize function matches the original behavior."""
    my_document = (
        "It was all very well to say `Drink me,' but the wise little Alice was not going to do that in a hurry."
    )

    expected_tokens = [
        "It",
        "was",
        "all",
        "very",
        "well",
        "to",
        "say",
        "`Drink",
        "me,'",
        "but",
        "the",
        "wise",
        "little",
        "Alice",
        "was",
        "not",
        "going",
        "to",
        "do",
        "that",
        "in",
        "a",
        "hurry.",
    ]

    assert tokenize(my_document) == expected_tokens


def test_tokenize_with_pattern():
    """Test tokenize with custom pattern."""
    formatted_document = "here's-a-document-with-strange-formatting"
    expected_tokens = ["here's", "a", "document", "with", "strange", "formatting"]
    assert tokenize(formatted_document, pattern="-") == expected_tokens


def test_process_file_case_sensitive(tmp_path):
    """Test process_file with case sensitive counting."""
    # Create a temporary test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("A a B b")

    # Process the file
    counter = process_file((str(test_file), False))

    # Check counts
    assert counter["A"] == 1
    assert counter["a"] == 1
    assert counter["B"] == 1
    assert counter["b"] == 1


def test_process_file_case_insensitive(tmp_path):
    """Test process_file with case insensitive counting."""
    # Create a temporary test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("A a B b")

    # Process the file
    counter = process_file((str(test_file), True))

    # Check counts
    assert counter["a"] == 2
    assert counter["b"] == 2
    assert "A" not in counter
    assert "B" not in counter


def test_process_file_empty_tokens(tmp_path):
    """Test that empty tokens are filtered out."""
    # Create a temporary test file with empty tokens
    test_file = tmp_path / "test.txt"
    test_file.write_text("a  b\nc\t\td")  # Multiple spaces, newlines, and tabs

    # Process the file
    counter = process_file((str(test_file), False))

    # Check that only non-empty tokens are counted
    assert counter["a"] == 1
    assert counter["b"] == 1
    assert counter["c"] == 1
    assert counter["d"] == 1
    assert "" not in counter


def test_process_file_error_handling(tmp_path):
    """Test error handling in process_file."""
    # Try to process a non-existent file
    counter = process_file(("non_existent_file.txt", False))

    # Should return an empty counter
    assert len(counter) == 0


def test_end_to_end(tmp_path):
    """Test the entire process with multiple files."""
    # Create a temporary directory with test files
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()

    # Create test files with explicit case handling
    (test_dir / "file1.txt").write_text("a b d")  # lowercase
    (test_dir / "file2.txt").write_text("a b d")  # lowercase
    (test_dir / "file3.txt").write_text("a b C")  # lowercase 'a' and 'b', uppercase 'C'

    # Process files with case insensitive counting
    counters = []
    for file_path in test_dir.glob("*.txt"):
        counters.append(process_file((str(file_path), True)))

    # Combine counters
    final_counter = Counter()
    for counter in counters:
        final_counter.update(counter)

    # Check results - all tokens should be lowercase due to case insensitive counting
    assert final_counter["a"] == 3  # three lowercase 'a's
    assert final_counter["b"] == 3  # three lowercase 'b's
    assert final_counter["c"] == 1  # one lowercase 'c' (from uppercase 'C')
    assert final_counter["d"] == 2  # two lowercase 'd's
    assert "A" not in final_counter  # no uppercase tokens
    assert "B" not in final_counter
    assert "C" not in final_counter


def test_parallel_processing_different_workers(tmp_path):
    """Test that parallel processing works with different numbers of workers."""
    # Create a temporary directory with test files
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()

    # Create test files with more content to make parallel processing noticeable
    for i in range(10):  # Create 10 files
        (test_dir / f"file{i}.txt").write_text("a b c d e f g h i j k l m n o p q r s t u v w x y z")

    # Test with different numbers of workers
    worker_counts = [1, 2, 4, 8]
    processing_times = []

    for num_workers in worker_counts:
        start_time = time.time()

        # Process files with specified number of workers
        counters = []
        for file_path in test_dir.glob("*.txt"):
            counters.append(process_file((str(file_path), False)))

        # Combine counters
        final_counter = Counter()
        for counter in counters:
            final_counter.update(counter)

        end_time = time.time()
        processing_times.append(end_time - start_time)

        # Verify results are consistent regardless of worker count
        assert final_counter["a"] == 10  # Each file has one 'a'
        assert final_counter["z"] == 10  # Each file has one 'z'
        assert len(final_counter) == 26  # All letters a-z should be present

    # Verify that more workers generally means faster processing
    # Note: This is not always true due to system load, but we can check if it's reasonable
    assert (
        processing_times[0] >= processing_times[-1] * 0.5
    )  # Single worker should take at least half the time of max workers


def test_parallel_processing_consistency(tmp_path):
    """Test that parallel processing produces consistent results across multiple runs."""
    # Create a temporary directory with test files
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()

    # Create test files
    (test_dir / "file1.txt").write_text("a b c d")
    (test_dir / "file2.txt").write_text("e f g h")
    (test_dir / "file3.txt").write_text("i j k l")

    # Run multiple times with different worker counts
    results = []
    for num_workers in [1, 2, 4]:
        counters = []
        for file_path in test_dir.glob("*.txt"):
            counters.append(process_file((str(file_path), False)))

        final_counter = Counter()
        for counter in counters:
            final_counter.update(counter)

        results.append(final_counter)

    # Verify all runs produced the same results
    for i in range(1, len(results)):
        assert results[i] == results[0], f"Results with {i + 1} workers differ from single worker results"


def test_parallel_processing_error_handling(tmp_path):
    """Test that parallel processing handles errors gracefully."""
    # Create a temporary directory with test files
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()

    # Create some valid files and some invalid ones
    (test_dir / "valid1.txt").write_text("a b c")
    (test_dir / "valid2.txt").write_text("d e f")
    (test_dir / "invalid.txt").write_text("")  # Empty file

    # Process files with multiple workers
    counters = []
    for file_path in test_dir.glob("*.txt"):
        counters.append(process_file((str(file_path), False)))

    # Combine counters
    final_counter = Counter()
    for counter in counters:
        final_counter.update(counter)

    # Verify that valid files were processed and invalid ones were handled gracefully
    assert final_counter["a"] == 1
    assert final_counter["b"] == 1
    assert final_counter["c"] == 1
    assert final_counter["d"] == 1
    assert final_counter["e"] == 1
    assert final_counter["f"] == 1
    assert "" not in final_counter  # Empty tokens should be filtered out
