"""
SpanAligner Module
==================

A utility module for aligning and mapping text spans between different text representations,
particularly useful for Label Studio annotation compatibility.

This module provides functionality to:
- Sanitize span boundaries to avoid special characters
- Find exact and fuzzy matches of text segments in original documents
- Map spans from one text representation to another
- Rebuild tagged text with nested annotations
- Merge result objects containing span annotations

Typical use case: When text has been modified (e.g., cleaned, translated) and annotations
need to be realigned to the original or modified text.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Dict, Any, List, Tuple, Optional, Union
from rapidfuzz import fuzz

# Span sanitization helper for Label Studio compatibility
SPECIAL_CHARS = {"\n", "\r", "\t", " "}


class SpanAligner:
    """
    A utility class for aligning text spans between different text representations.
    
    This class provides static methods for:
    - Sanitizing span boundaries
    - Finding exact and fuzzy text matches
    - Mapping spans from extracted/modified text back to original text
    - Rebuilding tagged text with proper nesting
    - Merging annotation result objects
    
    All methods are static and the class serves as a namespace for related functionality.
    
    Example Usage:
        >>> original = "Hello, World!"
        >>> result_obj = {
        ...     "spans": [{"start": 0, "end": 5, "text": "Hello", "labels": ["greeting"]}],
        ...     "entities": [],
        ...     "task": {"data": {"text": ""}}
        ... }
        >>> success, mapped = SpanAligner.map_spans_to_original(original, result_obj)
    """

    @staticmethod
    def project_spans(
        src_text: str,
        tgt_text: str,
        src_spans: List[Dict[str, Any]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Project spans from source to target text using fuzzy matching.
                
        Args:
            src_text: Source text (mostly ignored as spans are expected to contain 'text').
            tgt_text: Target text to align to.
            src_spans: List of spans with 'start', 'end', 'text', 'labels'.
            kwargs: Extra arguments:
                - min_ratio: Minimum similarity ratio for fuzzy matching.
                - max_dist: Maximum allowed distance deviation.
                - enable_fuzzy: Whether to use fuzzy matching.
                - logging: Enable debug logging.

        Returns:
            List of projected spans aligned to target text.
        """
        # Construct result object expected by map_spans_to_original
        # We assume src_spans have correct text relative to what we want to find in tgt_text
        result_obj = {
            "spans": src_spans,
            "entities": [],
            "task": {"data": {"text": ""}}
        }
        
        min_ratio = kwargs.get('min_ratio', 0.90)
        max_dist = kwargs.get('max_dist', 20)
        enable_fuzzy = kwargs.get('enable_fuzzy', False)
        logging = kwargs.get('logging', False)
        
        _, mapped = SpanAligner.map_spans_to_original(
            tgt_text, 
            result_obj, 
            min_ratio=min_ratio,
            max_dist=max_dist,
            enable_fuzzy=enable_fuzzy,
            logging=logging
        )
        
        return mapped.get("spans", [])

    @staticmethod
    def sanitize_span(text: str, start: int, end: int) -> tuple[int, int]:
        """
        Adjust start/end indices so they do not land on special characters.
        
        Moves start forward and end backward to avoid whitespace and control characters
        at span boundaries, which is important for Label Studio compatibility.
        
        Args:
            text: The text containing the span.
            start: The starting index of the span (inclusive).
            end: The ending index of the span (exclusive).
            
        Returns:
            tuple[int, int]: A tuple of (sanitized_start, sanitized_end) indices.
                Both values are clamped to [0, len(text)] and guaranteed to satisfy start <= end.
                
        Example:
            >>> SpanAligner.sanitize_span("  Hello  ", 0, 9)
            (2, 7)  # Removes leading/trailing spaces
        """
        n = len(text)
        s = max(0, min(start, n))
        e = max(0, min(end, n))

        # Move start forward while on a special char and s < e
        while s < e and s < n and text[s] in SPECIAL_CHARS:
            s += 1
        # Move end backward while on a special char and s < e
        while s < e and e > 0 and text[e-1] in SPECIAL_CHARS:
            e -= 1

        return s, e

    @staticmethod
    def _sequence_similarity(a: str, b: str) -> float:
        """
        Calculate the similarity ratio between two strings using SequenceMatcher.
        
        Args:
            a: First string to compare.
            b: Second string to compare.
            
        Returns:
            float: Similarity ratio between 0.0 and 1.0, where 1.0 means identical strings.
                Returns 0.0 if either string is empty.
        """
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()


    @staticmethod
    def _find_exact(original_text: str, segment: str) -> List[int]:
        """
        Find all exact occurrences of a segment within the original text.
        
        Args:
            original_text: The text to search within.
            segment: The exact substring to find.
            
        Returns:
            List[int]: A list of starting indices where the segment was found.
                Empty list if no matches found.
                
        Example:
            >>> SpanAligner._find_exact("hello world hello", "hello")
            [0, 12]
        """
        indices = []
        start = 0
        while True:
            idx = original_text.find(segment, start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + 1
        return indices

    @staticmethod
    def _best_fuzzy_in_window(
            original_text: str,
            segment: str,
            start_hint: Optional[int],
            max_search_slack: int = 20
        ) -> Tuple[Optional[int], Optional[int], float]:
            """
            Find the best fuzzy match for a segment within a window around a hint position.
            Uses RapidFuzz for performance and better fuzzy matching.
            """
            if not segment:
                return None, None, 0.0

            # Calculate window bounds
            if start_hint is None:
                left = 0
                right = len(original_text)
            else:
                left = max(0, start_hint - max_search_slack)
                # Add segment length + slack to the window end
                right = min(len(original_text), start_hint + len(segment) + 2 * max_search_slack)
                
            window = original_text[left:right]

            if not window:
                return None, None, 0.0

            best_ratio = 0.0
            best_start = None
            best_end = None
            
            seg_len = len(segment)
            # Allow length variation (slack)
            # We allow candidates to be +/- 20% length of segment, but at least +/- 5 chars
            slack_len = max(5, int(seg_len * 0.2))
            min_len = max(1, seg_len - slack_len)
            max_len = seg_len + slack_len

            len_window = len(window)

            # Sliding window search
            for i in range(len_window):
                # Optimization: If the remaining window is shorter than min_len, stop
                if i + min_len > len_window:
                    break
                
                # Limit candidate end to avoid checking excessively long strings
                end_limit = min(i + max_len, len_window) + 1
                
                for j in range(i + min_len, end_limit):
                    candidate = window[i:j]
                    
                    # Use rapidfuzz ratio
                    ratio = fuzz.ratio(segment, candidate) / 100.0
                    
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_start = left + i
                        best_end = left + j
                        
                        if best_ratio == 1.0:
                            break
                if best_ratio == 1.0:
                    break

            if best_start is None:
                return None, None, 0.0

            # Sanitize using the original SpanAligner helper
            best_start, best_end = SpanAligner.sanitize_span(original_text, best_start, best_end)
        
            return best_start, best_end, best_ratio

    @staticmethod
    def _regex_word_sequence(
        original_text: str,
        segment: str,
        start_hint: Optional[int] = None,
        max_search_slack: int = 20
    ) -> Optional[Tuple[int, int]]:
        """
        Find a segment in the original text using regex-based word sequence matching.
        
        This method tokenizes the segment into words and punctuation, then builds a
        tolerant regex pattern that allows for varying whitespace/separators between
        tokens. This is useful for matching text that may have different formatting.
        
        Args:
            original_text: The text to search within.
            segment: The text segment to find (will be tokenized).
            start_hint: Approximate starting position to prioritize searching around.
                If provided, searches near this position first before falling back to
                full text search.
            max_search_slack: Maximum distance from start_hint to search.
                Default is 20 characters.
                
        Returns:
            Optional[Tuple[int, int]]: A tuple of (start, end) indices if found,
                or (None, None) if no match found.
                
        Example:
            >>> # Matches "hello world" even with different whitespace
            >>> SpanAligner._regex_word_sequence("hello   world", "hello world")
            (0, 13)
        """
        # Tokenize into words and punctuation, keeping punctuation tokens
        # Words: one or more word chars; Punct: any single non-word, non-space char
        tokens = re.findall(r"\w+|[^\w\s]", segment)
        if not tokens:
            return None, None

        # Build a tolerant pattern that matches tokens in order, allowing non-word separators/newlines after each
        # Keep punctuation characters explicitly in the pattern
        escaped = list(map(re.escape, tokens))
        pattern = r"(?s)" + r"\W*".join(escaped)
        try:
            regex = re.compile(pattern)
        except re.error:
            return None, None

        # If we have a hint, first search in a bounded region around it
        if isinstance(start_hint, int):
            left = max(0, start_hint - max_search_slack)
            # allow for extra room to the right in case of many separators
            right = min(len(original_text), start_hint + max(max_search_slack, len(segment) * 2))
            subset = original_text[left:right]
            m = regex.search(subset)
            if m:
                return left + m.start(), left + m.end()

        # Fallback: search entire text
        m = regex.search(original_text)

        if not m:
            return None, None
        return m.start(), m.end()

    @staticmethod
    def map_spans_to_original(
        original_text: str,
        result_obj: Dict[str, Any],
        min_ratio: float = 0.90,
        logging: bool = False,
        max_dist: int = 20,
        enable_fuzzy: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Map spans from a result object to their positions in the original text.
        
        This is the main alignment method that attempts to find the correct positions
        of annotated spans in the original text. It uses multiple strategies:
        1. Exact matching (fastest, most reliable)
        2. Regex-based word sequence matching (handles whitespace variations)
        3. Fuzzy matching (optional, for handling minor text differences)
        
        Args:
            original_text: The original/target text to map spans onto.
            result_obj: A dictionary containing annotation data with the following structure:
                {
                    "spans": [{
                        "start": int,      # Approximate start position
                        "end": int,        # Approximate end position  
                        "text": str,       # The text content of the span
                        "labels": [str]    # List of label names
                    }, ...],
                    "entities": [...],     # Same structure as spans
                    "task": {
                        "data": {"text": str}  # Will be updated with original_text
                    }
                }
            min_ratio: Minimum similarity ratio (0.0-1.0) for fuzzy matching.
                Default is 0.90.
            logging: If True, prints debug information during alignment.
                Default is False.
            max_dist: Maximum allowed distance between approximate and actual
                start positions. Matches further than this are rejected.
                Default is 20 characters.
                
        Returns:
            Tuple[bool, Dict[str, Any]]: A tuple of:
                - bool: True if all spans were successfully aligned, False otherwise.
                - Dict: Updated result_obj with mapped spans. Each span now includes:
                    - start/end: Mapped positions (or None if unmatched)
                    - text: Matched text from original (or None if unmatched)
                    - status: "exact", "regex", "fuzzy", or "unmatched"
                    - similarity: Match similarity score (0.0-1.0)
                    - detected: The cleaned segment text that was searched for
                    - approx_start: Original approximate start position
                    
        Example:
            >>> original = "Hello, World!"
            >>> result = {
            ...     "spans": [{"start": 0, "end": 5, "text": "Hello", "labels": ["greeting"]}],
            ...     "entities": [],
            ...     "task": {"data": {"text": ""}}
            ... }
            >>> success, mapped = SpanAligner.map_spans_to_original(original, result)
            >>> success
            True
            >>> mapped["spans"][0]["status"]
            'exact'
        """
        input_spans: List[Dict[str, Any]] = result_obj.get("spans", [])
        input_entities: List[Dict[str, Any]] = result_obj.get("entities", [])


        def realign(items: List[Dict[str, Any]], enable_fuzzy: bool = False) -> Tuple[bool,List[Dict[str, Any]]]:
            mapped: List[Dict[str, Any]] = []
            all_aligned = True
            for span in items:
                
                approx_start = span.get("start", 0)
                segment = span.get("text", "") or ""
                labels = span.get("labels", [])
                clean_segment = segment.strip()
                chosen_end = None

                if logging:
                    print("\n\n\n=======NEW SPAN==============")
                    print(f"label: {labels}")
                    print(f"cleaned span: '{clean_segment}' from original segment: '{segment}'")
                    

                exact_indices = SpanAligner._find_exact(original_text, clean_segment)
                chosen_start = None
                similarity = 1.0 if exact_indices else 0.0
                status = "unmatched"
                
                # First try exact match
                if exact_indices:
                    chosen_start = min(exact_indices, key=lambda i: abs(i - approx_start))
                    chosen_end = chosen_start + len(clean_segment)
                    status = "exact"
                
                # Next try regex-based word sequence match (search near approx_start for all words in order)
                else:
                    step = 20
                    slacks = sorted(list(set(list(range(0, max_dist + 1, step)) + [max_dist])))
                    
                    # 1. Try Regex progressively
                    for current_slack in slacks:
                        regex_start, regex_end = SpanAligner._regex_word_sequence(original_text, clean_segment, start_hint=approx_start, max_search_slack=current_slack)

                        if regex_start is not None:
                            chosen_start = regex_start
                            chosen_end = regex_end
                            # similarity compared against the fully matched span
                            similarity = SpanAligner._sequence_similarity(clean_segment, original_text[regex_start:regex_end])
                            status = "regex"
                            break

                    # 2. Try fuzzy matching if regex failed
                    if status == "unmatched" and enable_fuzzy:
                        
                        step = 20
                        slacks = sorted(list(set(list(range(0, max_dist + 1, step)) + [max_dist])))

                        for current_slack in slacks:
                            fuzzy_start, fuzzy_end, fuzzy_ratio = SpanAligner._best_fuzzy_in_window(original_text, clean_segment, start_hint=approx_start, max_search_slack=current_slack)
                            
                            if fuzzy_start is not None and fuzzy_ratio >= min_ratio:
                                chosen_start = fuzzy_start
                                chosen_end = fuzzy_end
                                similarity = fuzzy_ratio
                                status = "fuzzy"
                                break
                
                            

                # Check distance threshold
                if chosen_start is not None and abs(chosen_start - approx_start) > max_dist:
                    if logging:
                        print(f"Match rejected due to distance: {abs(chosen_start - approx_start)} > {max_dist}")
                    chosen_start = None
                    chosen_end = None
                    status = "unmatched"

                if chosen_start is not None:
                    # Sanitize the mapped span to avoid leading/trailing special characters
                    chosen_start, chosen_end = SpanAligner.sanitize_span(original_text, chosen_start, chosen_end)
                    matched_text = original_text[chosen_start:chosen_end]
                else:  
                    if logging:   
                        print("No match found")  

                    matched_text = None
                    all_aligned = False

                if logging:
                    print("=====================")
                    print(f"span: {span} segment: {clean_segment}")
                    print(f"status:{status} similarity: {similarity}")
                    print(f"pre sanit: {(chosen_start,chosen_end, len(clean_segment))}" if chosen_start is not None else "pre sanit: None")
                    if chosen_start is not None:
                        print(f"updated positions: (start: {chosen_start}; end: {chosen_end})")
                        print(f"Extracted in original: '{original_text[chosen_start:chosen_end]}'")
                    
                mapped.append({
                    "start": chosen_start,
                    "end": chosen_end,
                    "text": matched_text,
                    "labels": labels,
                    "status": status,
                    "similarity": round(similarity, 4),
                    "detected": clean_segment,
                    "approx_start": approx_start,
                })
            return all_aligned, mapped

        updated = dict(result_obj)
        all_spans_aligned, updated["spans"] = realign(input_spans, enable_fuzzy)
        all_entities_aligned, updated["entities"] = realign(input_entities, enable_fuzzy)
        updated["task"]["data"]["text"] = original_text
        return  updated, all_spans_aligned and all_entities_aligned


    @staticmethod
    def merge_result_objects(
        base: Dict[str, Any],
        addition: Dict[str, Any],
        span_from_name: str,
        ner_from_name: str
    ) -> Dict[str, Any]:
        """
        Merge two result objects by combining their span and entity lists.
        
        Creates a new dictionary based on the base object, then appends spans and
        entities from the addition object.
        
        Args:
            base: The base result object to merge into.
            addition: The result object to merge from.
            span_from_name: The key name for span annotations (e.g., "spans", "segmentation").
            ner_from_name: The key name for NER/entity annotations (e.g., "entities").
            
        Returns:
            Dict[str, Any]: A new merged dictionary with combined spans and entities.
                The base object is shallow-copied, so nested objects may still be shared.
                
        Example:
            >>> base = {"spans": [{"text": "A"}], "entities": []}
            >>> addition = {"spans": [{"text": "B"}], "entities": [{"text": "C"}]}
            >>> merged = SpanAligner.merge_result_objects(base, addition, "spans", "entities")
            >>> len(merged["spans"])
            2
        """
        merged = dict(base)

        addition_spans = addition.get(span_from_name, [])
        base_spans = merged.get(span_from_name, [])
        merged[span_from_name] = base_spans + addition_spans

        addition_ner = addition.get(ner_from_name, [])
        base_ner = merged.get(ner_from_name, [])
        merged[ner_from_name] = base_ner + addition_ner
        return merged

    @staticmethod
    def _invert_label_map(tag_to_label: Dict[str, str]) -> Dict[str, str]:
        """
        Invert a tag-to-label mapping to create a label-to-tag mapping.
        
        Args:
            tag_to_label: A dictionary mapping tag names to label names.
                Can be None, in which case an empty dict is returned.
                
        Returns:
            Dict[str, str]: A dictionary mapping label names to tag names.
            
        Example:
            >>> SpanAligner._invert_label_map({"loc": "Location", "per": "Person"})
            {'Location': 'loc', 'Person': 'per'}
        """
        return {v: k for k, v in (tag_to_label or {}).items()}

    @staticmethod
    def _sanitize_label_to_tag(label: str) -> str:
        """
        Convert a human-readable label to a sanitized XML-safe tag name.
        
        Converts the label to lowercase, replaces spaces with underscores,
        and removes any characters that are not alphanumeric or underscores.
        
        Args:
            label: The human-readable label to convert.
            
        Returns:
            str: A sanitized tag name suitable for use in XML/HTML tags.
                Returns "span" if the result would be empty.
                
        Example:
            >>> SpanAligner._sanitize_label_to_tag("My Label (Special)")
            'my_label_special'
        """
        # Fallback: convert human label to tag-like form
        tag = label.strip().lower().replace(" ", "_")
        tag = re.sub(r"[^a-z0-9_]+", "_", tag).strip("_")
        return tag or "span"


    @staticmethod
    def _format_annotations(task: Any) -> Dict[str, Any]:
        """
        Extract and format annotations from a Label Studio task object.
        
        Parses the first annotation from the task and categorizes the results
        into classification choices, entities, and segmentation spans.
        Falls back to predictions if no annotations are available.
        
        Args:
            task: A Label Studio task object with an `annotations` attribute.
                Expected structure:
                    task.annotations = [{
                        "result": [
                            {"type": "choices", "from_name": "type", "value": {"choices": [...]}},
                            {"type": "labels", "from_name": "entities", "value": {...}},
                            {"type": "labels", "from_name": "segmentation", "value": {...}}
                        ]
                    }]
                If annotations are empty, predictions with the same structure
                will be used as a fallback.
                    
        Returns:
            Dict[str, Any]: A dictionary with three keys:
                - "classification": List of classification choices
                - "entities": List of entity annotation values
                - "segmentation": List of segmentation span values
        """
        # Try annotations first, fall back to predictions if empty
        results = []
        if task.annotations:
            results = task.annotations[0].get("result", [])
        
        # If no annotations, try predictions
        if not results and hasattr(task, 'predictions') and task.predictions:
            results = task.predictions[0].result or []
        
        classification = []
        entities = []
        spans = []

        for ann in results:
            ann_type = ann.get("type")
            from_name = ann.get("from_name")
            value = ann.get("value", {})

            if ann_type == "choices" and from_name == "type":
                if choices := value.get("choices"):
                    classification = choices
            elif ann_type == "labels":
                if from_name == "entities":
                    entities.append(value)
                elif from_name == "segmentation":
                    spans.append(value)
        
        return {
            "classification": classification,
            "entities": entities,
            "segmentation": spans
        }


    @staticmethod
    def update_mapped_with_rebuilt(
        original_text: str,
        mapped: Dict[str, Any],
        span_label_mapping: Optional[Dict[str, str]] = None,
        ner_label_mapping: Optional[Dict[str, str]] = None,
        overwrite: bool = True
    ) -> Dict[str, Any]:
        """
        Update a mapped result object with rebuilt tagged text.
        
        Takes a mapped result object (output from map_spans_to_original) and
        generates tagged text from its spans and entities, storing the result
        in the task data.
        
        Args:
            original_text: The original text to use for rebuilding tags.
            mapped: A mapped result object containing:
                - "spans": List of span annotations
                - "entities": List of entity annotations
                - "task": {"data": {...}} - Task data to update
            span_label_mapping: Optional tag-to-label mapping for spans.
                Will be inverted to create label-to-tag mapping.
            ner_label_mapping: Optional tag-to-label mapping for NER entities.
                Will be inverted to create label-to-tag mapping.
            overwrite: If True, overwrites "tagged_text" in task data.
                If False, stores result in "tagged_text_unified" instead.
                When overwriting, the original tagged_text is preserved in
                "tagged_text_original" if it exists.
                
        Returns:
            Dict[str, Any]: The same mapped object (modified in place) with:
                - task.data.tagged_text (or tagged_text_unified): The rebuilt tagged text
                - task.data.tagged_text_original: Original tagged_text if overwritten
                - task.data.rebuild_stats: Statistics from rebuild operation
                
        Example:
            >>> mapped = {"spans": [...], "entities": [...], "task": {"data": {}}}
            >>> updated = SpanAligner.update_mapped_with_rebuilt("Hello World", mapped)
            >>> "tagged_text" in updated["task"]["data"]
            True
        """
        data = mapped.get("task", {}).get("data", {})
        # text = data.get("text", "")
        label_to_tag = {}
        label_to_tag.update(SpanAligner._invert_label_map(span_label_mapping or {}))
        label_to_tag.update(SpanAligner._invert_label_map(ner_label_mapping or {}))

        rebuilt, stats = SpanAligner.rebuild_tagged_text(
            original_text,
            mapped.get("spans", []),
            mapped.get("entities", []),
            label_to_tag=label_to_tag,
        )

        # Preserve original and write unified
        if "tagged_text" in data and not data.get("tagged_text_original"):
            data["tagged_text_original"] = data.get("tagged_text")
        if overwrite:
            data["tagged_text"] = rebuilt
        else:
            data["tagged_text_unified"] = rebuilt
        data["rebuild_stats"] = stats
        return mapped



#### From tagged text to task
    @staticmethod
    def get_annotations_from_tagged_text(
        result: Union[dict, str],
        *,
        include_attachments: bool = True,
        span_map: Optional[Dict[str, str]] = None,
        ner_map: Optional[Dict[str, str]] = None,
        class_map: Optional[Dict[str, str]] = None,
        allowed_tags: Optional[List[str]] = None,
    ) -> dict:
        """
        Convert a tagged result (with inline XML-like tags) into structured annotations.

        Extracts spans and entities from tagged text by removing tags and tracking
        character offsets in the resulting plain text. Supports nested tags and
        custom tag-to-label mappings.

        Args:
            result: Input dictionary with 'tagged_text' key, or the tagged text string itself.
            include_attachments: Whether to include text content inside <attachment> tags.
            span_map: Dictionary mapping tag names to span labels.
            ner_map: Dictionary mapping tag names to entity labels.
            class_map: Dictionary mapping document classifications to labels.
            allowed_tags: List of tag names to process. If None, derived from map keys.

        Returns:
            dict: A dictionary containing:
                - spans: List of span (segmentation) objects
                - entities: List of entity (NER) objects
                - plain_text: The text content with tags removed
                - tagged_text: The original tagged text used
                - document_classification: The classification from input result (if any)
        
        Notes:
            - Spans are derived by removing tags while tracking character offsets in the
              plain text. Nested tags are supported; spans may overlap.
            - `span_map` lets you rename tags to match your LS label config.
            - `allowed_tags` limits which tags are turned into spans. If None, uses the
              tag set defined in your prompts.
        """

        # Resolve tagged_text input
        tagged_text = ""
        doc_class = None
        if isinstance(result, dict):
            tagged_text = result.get("tagged_text", "")
            doc_class = result.get("document_classification")
        else:
            tagged_text = str(result or "")

        if not tagged_text:
            raise ValueError("No tagged_text found in input result.")

        # Default allowed tags (from your SYSTEM/USER prompts)
        if allowed_tags is None and (span_map or ner_map):
            # Safely handle None maps
            s_map = span_map or {}
            n_map = ner_map or {}
            allowed_tags = list(n_map.keys()) + list(s_map.keys())
        
        # Merge span_map and ner_map safely into annotation_map
        annotation_map = {}
        for mapping in (span_map, ner_map):
            if mapping:
                annotation_map.update(mapping)

        # Regex to capture bare tags like <tag> or </tag>
        tag_re = re.compile(r"<(/?)([a-zA-Z_][a-zA-Z0-9_-]*)>")

        plain_parts: List[str] = []
        spans: List[dict] = []
        entities: List[dict] = []

        stack: List[Tuple[str, int]] = []  # (tag_name_lower, start_offset_in_plain)

        pos_in = 0  # position in tagged_text
        pos_out = 0  # position in plain text we are building

        def emit_text(s: str):
            nonlocal pos_out
            if not s:
                return
            plain_parts.append(s)
            pos_out += len(s)

        # Attachment handling: if we skip attachments, when inside attachments or attachment, we don't emit text
        inside_attachments_level = 0

        for m in tag_re.finditer(tagged_text):
            # Emit any literal text before this tag
            literal = tagged_text[pos_in:m.start()]
            current_tag_is_attachment = inside_attachments_level > 0

            if include_attachments or not current_tag_is_attachment:
                emit_text(literal)

            is_closing = bool(m.group(1))
            tag_name = m.group(2).lower()

            # Track attachments nesting regardless of allowed_tags so we can drop their content when requested
            if tag_name in ("attachments", "attachment"):
                if not is_closing:
                    inside_attachments_level += 1
                else:
                    inside_attachments_level = max(0, inside_attachments_level - 1)

            # Handle span stack only for allowed tags
            if allowed_tags is None or tag_name in allowed_tags:
                if not is_closing:
                    # Opening tag
                    stack.append((tag_name, pos_out))
                else:
                    # Closing tag — find the last matching opening tag
                    # Iterate backwards to find the matching opening tag
                    found_open = False
                    for i in range(len(stack) - 1, -1, -1):
                        open_tag, start_off = stack[i]
                        if open_tag == tag_name:
                            # Pop all tags above the matching one (handle mismatched nesting)
                            stack = stack[:i]
                            end_off = pos_out
                            
                            # Create a span only if it has positive length
                            if end_off > start_off:
                                full_span_text = ("".join(plain_parts))[start_off:end_off]
                                
                                # Adjust start to skip leading newlines
                                adjusted_start = start_off
                                span_text = full_span_text
                                
                                while span_text.startswith('\n'):
                                    adjusted_start += 1
                                    span_text = span_text[1:]
                                
                                # Adjust end to skip trailing newlines
                                adjusted_end = end_off
                                while span_text.endswith('\n'):
                                    adjusted_end -= 1
                                    span_text = span_text[:-1]
                                
                                # Only create span if there's content after trimming
                                if adjusted_end > adjusted_start:
                                    annotation_entry = {
                                        "start": adjusted_start,
                                        "end": adjusted_end,
                                        "text": span_text,
                                        "labels": [annotation_map.get(tag_name, tag_name) if annotation_map else tag_name]
                                    }
                                    
                                    if ner_map and tag_name in ner_map:
                                        entities.append(annotation_entry)
                                    else:
                                        spans.append(annotation_entry)
                                        
                            found_open = True
                            break
                    # If no matching opening tag found, ignore gracefully

            pos_in = m.end()

        # Emit remaining tail text
        tail = tagged_text[pos_in:]
        if include_attachments or inside_attachments_level == 0:
            emit_text(tail if include_attachments else "")

        plain_text = "".join(plain_parts)

        return { 
            "spans": spans, 
            "entities": entities, 
            "plain_text": plain_text,
            "tagged_text": tagged_text,
            "document_classification": doc_class
        }

    @staticmethod
    def tagged_text_to_task(
        result: Union[dict, str],
        *,
        include_attachments: bool = True,
        span_map: Optional[Dict[str, str]] = None,
        ner_map: Optional[Dict[str, str]] = None,
        class_map: Optional[Dict[str, str]] = None,
        allowed_tags: Optional[List[str]] = None,
    ) -> dict:
        """
        Convert a tagged result into an uploader-ready Label Studio task.

        Uses `get_annotations_from_tagged_text` to parse the input and formats
        the output as expected by the Label Studio uploader class.

        Args:
            result: Input dictionary with 'tagged_text' key, or the tagged text string.
            include_attachments: Whether to include text content inside <attachment> tags.
            span_map: Dictionary mapping tag names to span labels.
            ner_map: Dictionary mapping tag names to entity labels.
            class_map: Dictionary mapping document classifications to labels.
            allowed_tags: List of tag names to process. If None, derived from map keys.

        Returns:
            dict: A dictionary ready for Label Studio import, containing:
                - task: Task data including text and metadata
                - spans: Extracted spans
                - entities: Extracted entities
                - labels: Classification labels (if applicable)
        """
        # Parse annotations using shared logic
        parsed = SpanAligner.get_annotations_from_tagged_text(
            result,
            include_attachments=include_attachments,
            span_map=span_map,
            ner_map=ner_map,
            class_map=class_map,
            allowed_tags=allowed_tags
        )
        
        spans = parsed["spans"]
        entities = parsed["entities"]
        plain_text = parsed["plain_text"]
        tagged_text = parsed["tagged_text"]
        doc_class = parsed["document_classification"]

        # Handle classification mapping
        classification_labels = []
        if doc_class and class_map and doc_class in class_map:
            classification_labels = [class_map[doc_class]]

        content = {
            "task": {
                "data": {
                    "text": plain_text,
                    "tagged_text": tagged_text,
                    "meta": {
                        "segments": len(spans),
                        "labels_present": sorted({(s.get("labels") or [""])[0] for s in spans}),
                        "include_attachments": include_attachments,
                        "document_classification": doc_class or ""
                    }
                }
            },
            "spans": spans,
            "labels": classification_labels,
            "entities": entities
        }

        return content


#### From task to tagged text
    @staticmethod
    def rebuild_tagged_text(
        original_text: str,
        spans: List[Dict[str, Any]] = None,
        entities: List[Dict[str, Any]] = None,
        label_to_tag: Optional[Dict[str, str]] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Rebuild text with nested XML-style tags from span and entity annotations.
        
        Creates properly nested tags from annotations, handling overlapping spans
        by skipping crossing (non-nested) annotations to maintain valid XML structure.
        
        Args:
            original_text: The source text to add tags to.
            spans: List of span annotations, each with:
                - "start": int - Starting character index
                - "end": int - Ending character index (exclusive)
                - "labels": List[str] - Label names (first one is used)
            entities: List of entity annotations (same structure as spans).
            label_to_tag: Optional mapping from label names to tag names.
                If a label is not in the mapping, it will be sanitized to
                create a valid tag name.
                
        Returns:
            Tuple[str, Dict[str, int]]: A tuple of:
                - str: The text with XML tags inserted (e.g., "<tag>text</tag>")
                - Dict with statistics:
                    - "total": Total number of valid annotations processed
                    - "skipped_crossing": Number of annotations skipped due to
                      crossing (non-nested) overlaps
                      
        Note:
            - Annotations with invalid positions (negative, overlapping bounds,
              or exceeding text length) are silently skipped.
            - For overlapping annotations, outer (longer) spans are preferred.
            - Crossing annotations that would create invalid XML are skipped.
            
        Example:
            >>> text = "Hello World"
            >>> spans = [{"start": 0, "end": 11, "labels": ["sentence"]}]
            >>> entities = [{"start": 0, "end": 5, "labels": ["greeting"]}]
            >>> result, stats = SpanAligner.rebuild_tagged_text(text, spans, entities)
            >>> result
            '<sentence><greeting>Hello</greeting> World</sentence>'
        """
        annotations: List[Dict[str, Any]] = []

        def to_tag(lbls: List[str]) -> Optional[str]:
            if not lbls:
                return None
            lbl = lbls[0]
            if label_to_tag and lbl in label_to_tag:
                return label_to_tag[lbl]
            return SpanAligner._sanitize_label_to_tag(lbl)

        def add_items(items: List[Dict[str, Any]]):
            for it in items or []:
                s = it.get("start")
                e = it.get("end")
                if not isinstance(s, int) or not isinstance(e, int) or s is None or e is None or s < 0 or e <= s or e > len(original_text):
                    continue
                tag = to_tag(it.get("labels") or [])
                if not tag:
                    continue
                annotations.append({
                    "start": s,
                    "end": e,
                    "tag": str(tag),
                    "length": e - s,
                })

        if spans and len(spans)>0:
            add_items(spans)
        if entities and len(entities)>0:
            add_items(entities)

        # Sort: by start asc, longer first (end desc) to open outers before inners
        annotations.sort(key=lambda a: (a["start"], -a["length"]))

        # Index starts and ends
        starts: Dict[int, List[Dict[str, Any]]] = {}
        for a in annotations:
            starts.setdefault(a["start"], []).append(a)
        for pos in starts:
            starts[pos].sort(key=lambda a: -a["length"])  # longer first

        ends: Dict[int, List[Dict[str, Any]]] = {}
        for a in annotations:
            ends.setdefault(a["end"], []).append(a)

        event_positions = sorted({0, len(original_text), *starts.keys(), *ends.keys()})

        pieces: List[str] = []
        stack: List[Dict[str, Any]] = []
        last = 0
        skipped_cross = 0

        for pos in event_positions:
            if pos > last:
                pieces.append(original_text[last:pos])

            # Close all tags that end here (LIFO)
            while stack and stack[-1]["end"] == pos:
                top = stack.pop()
                pieces.append(f"</{top['tag']}>")

            # Open tags that start here (outer first)
            for ann in starts.get(pos, []):
                # Crossing check: if an open tag exists with end < ann.end (not nested), skip ann
                if stack and ann["end"] > stack[-1]["end"]:
                    skipped_cross += 1
                    continue
                pieces.append(f"<{ann['tag']}>")
                stack.append(ann)

            last = pos

        # Tail
        pieces.append(original_text[last:])

        # Close any still-open tags (best-effort)
        while stack:
            top = stack.pop()
            pieces.append(f"</{top['tag']}>")

        return "".join(pieces), {"total": len(annotations), "skipped_crossing": skipped_cross}

    @staticmethod
    def rebuild_tagged_text_from_task(task: Any, mapping: Dict[str, str]) -> str:
        """
        Generate tagged text from a Label Studio task's annotations.
        
        Extracts annotations from the task and rebuilds the text with XML-style
        tags around annotated spans.
        
        Args:
            task: A Label Studio task object with:
                - task.annotations: List of annotation objects
                - task.data: Dict containing "text" key with the source text
            mapping: A dictionary mapping label names to tag names to use in
                the output. Labels not in the mapping will be sanitized to
                create tag names.
                
        Returns:
            str: The text with XML-style tags inserted around annotated spans.
            
        Example:
            >>> # Returns something like: "<greeting>Hello</greeting>, World!"
        """
        extracted = SpanAligner._format_annotations(task)
        text = task.data.get("text", "")

        retagged, _ = SpanAligner.rebuild_tagged_text(
            text,
            spans=extracted["segmentation"],
            entities=extracted["entities"],
            label_to_tag=mapping
        )

        return retagged
    

#### Transpose tags back to original text
    @staticmethod
    def map_tags_to_original(
        original_text: str,
        tagged_text: str, 
        min_ratio: float = 0.8,
        max_dist: int = 20,
        enable_fuzzy: bool = False,
        logging: bool = False

    ) -> str:
        """
        Map spans from tagged text back to their positions in the original text.
        
        Takes tagged text with XML-style tags and aligns the annotated spans
        back to their positions in the provided original text. Uses exact,
        regex-based, and fuzzy matching to find the best alignment.
        
        Args:
            original_text: The original untagged text.
            tagged_text: The text with XML-style tags indicating spans.
            min_ratio: Minimum similarity ratio (0.0-1.0) for fuzzy matching.
                Defaults to 0.8.
            max_dist: Maximum character distance from approximate position
                to consider a match valid. Defaults to 20.
            logging: If True, prints detailed debug information during mapping.
                Defaults to False.
        """        # First, extract spans/entities from tagged_text
        temp_content = SpanAligner.tagged_text_to_task(
            tagged_text,
            include_attachments=True,
            allowed_tags=None  # allow all tags
        )

        result_obj = {
            "spans": temp_content.get("spans", []),
            "entities": temp_content.get("entities", []),
            "task": {
                "data": {
                    "text": ""  # will be filled later
                }
            }
        }

        # Now map spans/entities back to original_text
        mapped, _ = SpanAligner.map_spans_to_original(
            original_text,
            result_obj,
            min_ratio=min_ratio,
            max_dist=max_dist,
            enable_fuzzy = enable_fuzzy,
            logging=logging,

        )

        original_text_tagged, _ = SpanAligner.rebuild_tagged_text(original_text, spans = mapped.get("spans", []))
        return original_text_tagged
