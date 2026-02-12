import pytest
from span_aligner import SpanAligner

class TestSpanAligner:

    # --- 1. Testing Sanitization ---
    def test_sanitize_span_basic(self):
        text = "  Hello World  "
        # "Hello World" is at index 2 to 13
        start, end = SpanAligner.sanitize_span(text, 0, 15)
        assert start == 2
        assert end == 13
        assert text[start:end] == "Hello World"

    def test_sanitize_span_no_change(self):
        text = "Hello"
        start, end = SpanAligner.sanitize_span(text, 0, 5)
        assert (start, end) == (0, 5)

    # --- 2. Testing Internal Matching Helpers ---
    def test_find_exact(self):
        text = "banana banana split"
        indices = SpanAligner._find_exact(text, "banana")
        assert indices == [0, 7]

    def test_regex_word_sequence_flexible_whitespace(self):
        text = "The  quick\nbrown\tfox"
        segment = "The quick brown fox"
        start, end = SpanAligner._regex_word_sequence(text, segment)
        assert text[start:end] == text
        assert start == 0

    def test_regex_word_sequence_with_punctuation(self):
        text = "Hello, (World)!"
        segment = "Hello World"
        start, end = SpanAligner._regex_word_sequence(text, segment)
        assert start is not None
        assert "Hello" in text[start:end]
        assert "World" in text[start:end]

    # --- 3. Testing Fuzzy Matching ---
    def test_best_fuzzy_in_window_exact(self):
        text = "The quick brown fox jumps over the lazy dog"
        # Exact match in window
        start, end, ratio = SpanAligner._best_fuzzy_in_window(text, "quick brown", 4, 10)
        assert (start, end) == (4, 15)
        assert ratio == 1.0

    def test_best_fuzzy_in_window_typo(self):
        text = "The quick brown fox jumps over the lazy dog"
        # Typo: "quck" instead of "quick"
        start, end, ratio = SpanAligner._best_fuzzy_in_window(text, "quck brown", 4, 10)
        assert (start, end) == (4, 15)
        assert ratio > 0.8

    def test_best_fuzzy_in_window_whitespace(self):
        text = "The quick  brown fox"
        # Extra space in original
        start, end, ratio = SpanAligner._best_fuzzy_in_window(text, "quick brown", 4, 10)
        assert (start, end) == (4, 16)
        assert ratio > 0.9

    def test_best_fuzzy_in_window_no_hint(self):
        text = "The quick brown fox"
        # Searching from start
        start, end, ratio = SpanAligner._best_fuzzy_in_window(text, "brown", None)
        assert (start, end) == (10, 15)
        assert ratio == 1.0

    def test_best_fuzzy_in_window_out_of_window(self):
        text = "The quick brown fox"
        # segment is far from hint, window is "The quick bro"
        start, end, ratio = SpanAligner._best_fuzzy_in_window(text, "fox", 0, 5)
        # The best match in "The quick bro" for "fox" is poor
        assert ratio < 0.7

    def test_best_fuzzy_in_window_empty(self):
        assert SpanAligner._best_fuzzy_in_window("", "abc", 0) == (None, None, 0.0)
        assert SpanAligner._best_fuzzy_in_window("abc", "", 0) == (None, None, 0.0)

    # --- 4. Testing Main Mapping Logic ---
    def test_map_spans_to_original_exact(self):
        original = "The patient has a fever and cough."
        result_obj = {
            "spans": [{"start": 0, "end": 7, "text": "patient", "labels": ["Person"]}],
            "entities": [],
            "task": {"data": {"text": ""}}
        }
        success, mapped = SpanAligner.map_spans_to_original(original, result_obj)
        
        assert success is True
        assert mapped["spans"][0]["start"] == 4
        assert mapped["spans"][0]["end"] == 11
        assert mapped["spans"][0]["status"] == "exact"
        assert mapped["task"]["data"]["text"] == original

    def test_map_spans_failure_out_of_bounds(self):
        original = "Short text"
        result_obj = {
            "spans": [{"start": 100, "end": 105, "text": "missing", "labels": ["Test"]}],
            "task": {"data": {}}
        }
        success, mapped = SpanAligner.map_spans_to_original(original, result_obj)
        assert success is False
        assert mapped["spans"][0]["status"] == "unmatched"

    # --- 5. Testing Object Merging ---
    def test_merge_result_objects(self):
        base = {"spans": [{"text": "A"}], "entities": []}
        addition = {"spans": [{"text": "B"}], "entities": [{"text": "C"}]}
        merged = SpanAligner.merge_result_objects(base, addition, "spans", "entities")
        
        assert len(merged["spans"]) == 2
        assert len(merged["entities"]) == 1
        assert merged["spans"][1]["text"] == "B"

    # --- 6. Testing Label/Tag Utilities ---
    def test_invert_label_map(self):
        mapping = {"loc_tag": "Location", "per_tag": "Person"}
        inverted = SpanAligner._invert_label_map(mapping)
        assert inverted == {"Location": "loc_tag", "Person": "per_tag"}

    def test_sanitize_label_to_tag(self):
        assert SpanAligner._sanitize_label_to_tag("Full Name!") == "full_name"
        assert SpanAligner._sanitize_label_to_tag("  ") == "span"
    
    def test_format_annotations_from_predictions(self):
        class MockTask:
            def __init__(self):
                self.annotations = []
                self.predictions = [type('obj', (object,), {
                    'result': [
                        {"type": "labels", "from_name": "entities", "value": {"text": "pred"}}
                    ]
                })]
        
        task = MockTask()
        formatted = SpanAligner._format_annotations(task)
        assert len(formatted["entities"]) == 1
        assert formatted["entities"][0]["text"] == "pred"
