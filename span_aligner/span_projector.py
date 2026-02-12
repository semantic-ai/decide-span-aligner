import logging
import re
from math import log
from typing import List, Dict, Any, Tuple, Optional
from .sentence_aligner import EmbeddingProvider, SentenceAligner, TokenInfo, AlignmentResult, TransformerEmbeddingProvider
from .span_aligner import SpanAligner

logger = logging.getLogger(__name__)


class SpanProjector:
    """
    Projector class to map annotations from tagged source text to projection target text.
    
    Terminology:
    - src / source: The text that HAS annotations/tags (e.g. the translation).
    - tgt / target: The text to project annotations ONTO (e.g. the original).
    """
    
    _aligner_cache = {}
    MATCHING_METHODS_MAP = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev",
                            "g": "greedy", "t": "threshold"}
    MATCHING_METHODS_REV = {v: k for k, v in MATCHING_METHODS_MAP.items()}
    
    def __init__(self, 
                 src_lang: str = "en", 
                 tgt_lang: str = "nl", 
                 matching_method: str = "inter",
                 token_type: str = "bpe",
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 model: str = "bert",
                 device: str = None,
                 layer: int = 8,
                 verbose: bool = False):
        """
        Initialize the projector.
        
        Args:
            src_lang: Language of the tagged source text
            tgt_lang: Language of the projection target text
            matching_method: Matching method (inter, mwmf, itermax, greedy, threshold)
            token_type: Tokenization type ("bpe" or "word")
            embedding_provider: Optional custom EmbeddingProvider instance
            model: Model name for default TransformerEmbeddingProvider
            device: Device for computation (None = auto-detect)
            layer: Transformer layer for embeddings
            verbose: Whether to enable verbose logging
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.matching_method = matching_method
        self.token_type = token_type
        self.verbose = verbose
        
        # Auto-detect device
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Create or use provided embedding provider
        if embedding_provider is not None:
            self.embed_provider = embedding_provider
        else:
            self.embed_provider = TransformerEmbeddingProvider(model=model, device=device, layer=layer)
        
        # Initialize aligner - this is the single source for tokenization and alignment
        self.aligner = self._get_cached_aligner()

    def _get_cached_aligner(self) -> SentenceAligner:
        """Get or create cached SentenceAligner instance."""
        cache_key = f"aligner_{self.matching_method}_{self.token_type}_{id(self.embed_provider)}"
        if cache_key not in self._aligner_cache:
            match_key = self.MATCHING_METHODS_REV.get(self.matching_method, "m")
            self._aligner_cache[cache_key] = SentenceAligner(
                embedding_provider=self.embed_provider,
                token_type=self.token_type,
                matching_methods=match_key
            )
        return self._aligner_cache[cache_key]

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _entropy(self, distribution: List[float]) -> float:
        """Calculate entropy of a probability distribution."""
        total = sum(distribution)
        if total <= 0:
            return 0.0
        normalized = [v / total for v in distribution if v > 0]
        return -sum(v * log(v, 2) for v in normalized)

    def _find_contiguous_clusters(self, indices: List[int], max_gap: int = 1) -> List[List[int]]:
        """Split token indices into contiguous groups, allowing small gaps."""
        if not indices:
            return []
        
        indices = sorted(set(indices))
        clusters = []
        current = [indices[0]]
        
        for idx in indices[1:]:
            if idx - current[-1] - 1 <= max_gap:
                current.append(idx)
            else:
                clusters.append(current)
                current = [idx]
        clusters.append(current)
        return clusters
    
    def _fill_gaps_in_clusters(self, clusters: List[List[int]]) -> List[List[int]]:
        """Make each cluster a fully continuous, sorted span."""
        filled_clusters = []

        for cluster in clusters:
            if not cluster:
                continue

            cluster = sorted(set(cluster))
            start, end = cluster[0], cluster[-1]
            filled_clusters.append(list(range(start, end + 1)))

        return filled_clusters
 
    def _cluster_merge(self, clusters: List[List[int]], src_token_count: Optional[int] = None, base_gap: int = 2, mass_factor: float = 0.5) -> List[int]:
        """
        Merges multiple alignment clusters into the single most likely contiguous sequence.
        
        Args:
            clusters: List of index lists, e.g., [[1,2], [5,6,7], [20,21]].
            src_token_count: Length of source span (used for sanity checking).
            base_gap: Minimum gap always allowed (handles punctuation/particles).
            mass_factor: Multiplier for dynamic tolerance. 
                        Allowed Gap = base_gap + (min(len(A), len(B)) * mass_factor).
                        
        Returns:
            A single flat list of indices representing the best alignment path.
        """
        if not clusters:
            return []
        
        # 1. Sort clusters by start index to enable linear chaining
        sorted_clusters = sorted(clusters, key=lambda x: x[0])
        
        # 2. Build Chains
        # A 'chain' is a list of clusters that have successfully merged
        chains: List[List[List[int]]] = []
        current_chain = [sorted_clusters[0]]
        
        for next_cluster in sorted_clusters[1:]:
            prev_cluster = current_chain[-1]
            
            # Calculate actual gap between the end of previous and start of next
            # e.g., prev=[2], next=[5] -> gap is indices 3,4 -> size 2.
            gap_size = next_cluster[0] - prev_cluster[-1] - 1
            
            # Calculate Dynamic Tolerance
            # We use the minimum mass because a weak link (size 1) breaks the chain easily.
            connection_mass = min(len(prev_cluster), len(next_cluster))
            allowed_gap = base_gap + int(connection_mass * mass_factor * 1.2**connection_mass)
            if self.verbose:
                print(f"Considering merge: Prev={prev_cluster}, Next={next_cluster}, Gap={gap_size}, Allowed={allowed_gap}")

            if gap_size <= allowed_gap:
                # Valid merge: Extend the current chain
                current_chain.append(next_cluster)
            else:
                # Gap too large: Finalize current chain and start a new one
                chains.append(current_chain)
                current_chain = [next_cluster]
                
        # Don't forget to save the final chain being built
        chains.append(current_chain)

        # 3. Select the Best Chain
        best_chain_indices = []
        best_score = (-1, -1) # (token_count, -span_length)
        
        for chain in chains:
            # Flatten the chain into a single list of indices
            flat_indices = [idx for cluster in chain for idx in cluster]
            
            if not flat_indices: 
                continue
                
            token_count = len(flat_indices)
            span_length = flat_indices[-1] - flat_indices[0] + 1
            
            # 4. Outlier Sanity Check (if source length is known)
            # If the target span is massively larger than source (e.g. > 4x), 
            # it's likely a bad "spread" alignment. We penalize it heavily.
            if src_token_count and span_length > (src_token_count * 4) + 6:
                token_count = 0 # Disqualify this chain effectively
                
            # 5. Scoring
            # Priority 1: Maximize number of aligned tokens
            # Priority 2: Minimize total span length (prefer compactness)
            # We use negative span_length because we want to MAXIMIZE the tuple
            score = (token_count, -span_length)
            
            if score > best_score:
                best_score = score
                best_chain_indices = flat_indices
                
        return best_chain_indices

    def _refine_alignment(self, alignment: List[Tuple[int, int]], tgt_tokens: List[Any], max_gap: int = 2) -> List[Tuple[int, int]]:
        """
        Refines alignment by removing duplicate mappings based on clustering and sequence logic.
        Only filters duplicates if they map to the same target word (uncased).
        Different words (e.g. compound split) are preserved.
        """
        if not alignment:
            return []

        # 1. Identify Clusters of Target Tokens
        tgt_indices = sorted(list(set(t for s, t in alignment)))
        clusters = self._find_contiguous_clusters(tgt_indices, max_gap=max_gap)
        
        # Map target index to cluster info (cluster_index, cluster_size)
        tgt_to_cluster = {}
        for c_idx, cluster in enumerate(clusters):
            c_size = len(cluster)
            for t in cluster:
                tgt_to_cluster[t] = (c_idx, c_size)

        # 2. Group by Source
        src_to_tgt = {}
        for s, t in alignment:
            src_to_tgt.setdefault(s, []).append(t)
            
        final_alignment = []
        sorted_srcs = sorted(src_to_tgt.keys())

        # 3. Resolve Ambiguities
        for i, src in enumerate(sorted_srcs):
            targets = src_to_tgt[src]
            
            if len(targets) <= 1:
                for t in targets:
                    final_alignment.append((src, t))
                continue

            # Group targets by text
            tgt_texts = []
            for t in targets:
                # Handle TokenInfo object (prod) or string/int (simple lists)
                if isinstance(tgt_tokens, list) and t < len(tgt_tokens):
                    obj = tgt_tokens[t]
                    if hasattr(obj, 'text'):
                        text = obj.text
                    else:
                        text = str(obj)
                else:
                    text = str(t) # Fallback
                tgt_texts.append(text.lower().strip())
            
            unique_texts = set(tgt_texts)
            
            should_resolve = False
            
            # Condition 1: Identical texts -> Always resolve duplicates
            if len(unique_texts) == 1:
                should_resolve = True
            
            # Condition 2: Different texts (likely compound) -> Check spatial clustering
            else:
                # If they are clustered together (e.g. compound split), keep them.
                # If they are scattered (gap > max_gap), treat as ambiguous/error and resolve.
                local_clusters = self._find_contiguous_clusters(sorted(targets), max_gap=max_gap)
                if len(local_clusters) > 1:
                    should_resolve = True
            
            if should_resolve:
                # Resolve picks the "best" subset (cluster + sequence)
                selected = self._resolve_duplicate_targets(
                    src, targets, i, sorted_srcs, src_to_tgt, tgt_to_cluster
                )
            else:
                # Keep all (e.g. valid compound split)
                selected = targets
                
            for t in selected:
                final_alignment.append((src, t))

        return sorted(list(set(final_alignment)))

    def _resolve_duplicate_targets(self, src, targets, src_idx, sorted_srcs, src_to_tgt, tgt_to_cluster):
        """Helper to selection best target candidates from a set of duplicates."""
        # a. Filter by Cluster Size
        candidates_info = []
        for t in targets:
            c_idx, c_size = tgt_to_cluster.get(t, (-1, 0))
            candidates_info.append({'t': t, 'c_idx': c_idx, 'size': c_size})
        
        max_size = max(c['size'] for c in candidates_info)
        best_cluster_candidates = [c['t'] for c in candidates_info if c['size'] == max_size]
        
        if len(best_cluster_candidates) == 1:
            return [best_cluster_candidates[0]]
            
        # b. Filter by Sequence Logic
        sub_clusters = self._find_contiguous_clusters(best_cluster_candidates, max_gap=0)
        
        if len(sub_clusters) == 1:
            return sub_clusters[0] # Contiguous, keep all
        
        # Disconnected sub-clusters: Choose best neighbor fit
        prev_tgt = None
        next_tgt = None
        
        # Look back
        for back_idx in range(src_idx - 1, -1, -1):
            s_prev = sorted_srcs[back_idx]
            if s_prev in src_to_tgt:
                prev_targets = src_to_tgt[s_prev]
                if prev_targets:
                    prev_tgt = prev_targets[-1]
                    break
        
        # Look forward
        for fwd_idx in range(src_idx + 1, len(sorted_srcs)):
            s_next = sorted_srcs[fwd_idx]
            if s_next in src_to_tgt:
                next_targets = src_to_tgt[s_next]
                if next_targets:
                    next_tgt = next_targets[0]
                    break

        best_sub = None
        min_dist = float('inf')
        
        for sub in sub_clusters:
            sub_center = sum(sub) / len(sub)
            
            dist = 0
            count = 0
            if prev_tgt is not None:
                dist += abs(sub_center - prev_tgt)
                count += 1
            if next_tgt is not None:
                dist += abs(sub_center - next_tgt)
                count += 1
            
            if count > 0:
                avg_dist = dist / count
            else:
                avg_dist = 0 
            
            if avg_dist < min_dist:
                min_dist = avg_dist
                best_sub = sub
            elif avg_dist == min_dist:
                    if best_sub is None or len(sub) > len(best_sub):
                        best_sub = sub
        
        if min_dist == 0 and prev_tgt is None and next_tgt is None:
                if sub_clusters:
                    best_sub = max(sub_clusters, key=len)

        return best_sub if best_sub else []




    def _map_span_to_target_tokens(self,
                                    span: Dict[str, Any],
                                    src_tokens: List[TokenInfo],
                                    tgt_tokens: List[TokenInfo],
                                    alignment: List[Tuple[int, int]],
                                    max_gap: int = 1) -> Optional[List[int]]:
        """Map a span from source tokens (tagged) to target token indices (projected)."""
        s_start, s_end = span.get("start"), span.get("end")
        
        # Build source->target mapping
        src_to_tgt = {}
        for src_idx, tgt_idx in alignment:
            src_to_tgt.setdefault(src_idx, []).append(tgt_idx)
        
        # Find source tokens covered by this span
        covered_src = [i for i, t in enumerate(src_tokens) 
                       if max(s_start, t.start) < min(s_end, t.end)]
        
        if not covered_src:
            return None
        
        # Get corresponding target tokens
        mapped_tgt = []
        for s_idx in covered_src:
            mapped_tgt.extend(src_to_tgt.get(s_idx, []))
        
        if not mapped_tgt:
            return None
        
        # Cluster and select largest
        clusters = self._find_contiguous_clusters(mapped_tgt, max_gap=max_gap)
        return max(clusters, key=len)

    def validate_projection(self, text: str, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate projected spans for boundary integrity and text matching."""
        issues = []
        sorted_spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
        
        for i, span in enumerate(sorted_spans):
            start, end = span["start"], span["end"]
            
            if not (0 <= start <= end <= len(text)):
                issues.append(f"Span out of bounds: {span}")
                continue
            
            extracted = text[start:end]
            if extracted.strip() != span.get("text", "").strip():
                issues.append(f"Text mismatch at {i}: expected '{span.get('text')}', got '{extracted}'")
        
        # Check overlaps
        for i in range(len(sorted_spans) - 1):
            curr, next_s = sorted_spans[i], sorted_spans[i + 1]
            if curr["end"] > next_s["start"] and curr["end"] < next_s["end"]:
                if curr.get("labels") != next_s.get("labels"):
                    issues.append(f"Partial overlap: {curr['text']} and {next_s['text']}")
        
        return {"valid": len(issues) == 0, "issues": issues}

    # -------------------------------------------------------------------------
    # Main projection methods
    # -------------------------------------------------------------------------

    def project_spans(self,
                      src_text: str,
                      tgt_text: str,
                      src_spans: List[Dict[str, Any]],
                      max_gap: int = 1,
                      debugging: bool = False) -> List[Dict[str, Any]]:
        """
        Project spans from source text (tagged) to target text.
        
        Args:
            src_text: Tagged source text
            tgt_text: Target text for projection
            src_spans: List of span dicts on source text
            max_gap: Maximum gap in token clustering
            
        Returns:
            List of projected span dicts on target text
        """
        # Use project_spans_with_debug_info but filter to standard output format
        detailed = self.project_spans_with_debug_info(src_text, tgt_text, src_spans, max_gap)
        if debugging:
            print(self._visualize_projection(detailed))
            return detailed

        projected = []
        for item in detailed:
            projected.append({
                "start": item["start"],
                "end": item["end"],
                "text": item["text"],
                "labels": item["labels"]
            })
        
        return projected

    def project_spans_with_debug_info(self,
                                       src_text: str,
                                       tgt_text: str,
                                       src_spans: List[Dict[str, Any]],
                                       max_gap: int) -> List[Dict[str, Any]]:
        
        """Project spans with detailed debug information."""
        if not src_text or not tgt_text or not src_spans:
            return []
            
        # Pre-compute embeddings for the full texts
        if self.verbose:
            print("Pre-computing embeddings for full texts...")
        src_tokens, src_vectors = self.aligner.get_text_embeddings(src_text)
        tgt_tokens, tgt_vectors = self.aligner.get_text_embeddings(tgt_text)
        
        if self.verbose:
            print(f"\n===============================")
            print(f"Source tokens: {[t.text for t in src_tokens]}")
            print(f"Target tokens: {[t.text for t in tgt_tokens]}")
            print(f"Lengths - Source: {len(src_tokens)} tokens, Target: {len(tgt_tokens)} tokens")
            print(f"lengths - Vectors: Source {src_vectors.shape}, Target {tgt_vectors.shape}")


        projected = []
        for span in src_spans:
            if self.verbose:
                print("\n=================================")
                print("1.1 Span text:", span.get('text', ''))
                print("1.2 word count:", len(span.get('text', '').split()))
        
            # Use align_texts_partial_with_embeddings with char positions from the span
            try:
                result = self.aligner.align_texts_partial_with_embeddings(
                    src_tokens, 
                    tgt_tokens, 
                    src_vectors, 
                    tgt_vectors, 
                    src_char_start=span['start'], 
                    src_char_end=span['end']
                )
            except ValueError as e:
                if self.verbose:
                    print(f"Skipping span alignment due to error (likely no token coverage): {e}")
                continue
           
            alignment = result.alignments.get(self.matching_method, [])
            alignment = self._refine_alignment(alignment, result.tgt_tokens, max_gap=max_gap)
            
            src_to_tgt = {}
            for src_idx, tgt_idx in alignment:
                src_to_tgt.setdefault(src_idx, []).append(tgt_idx)

            # Source tokens covered by current span
            covered_src = [i for i, t in enumerate(result.src_tokens)
                          if max(span['start'], t.start) < min(span['end'], t.end)]
            
            mapped_tgt = []
            for s_idx in covered_src:
                mapped_tgt.extend(src_to_tgt.get(s_idx, []))
            
            if not mapped_tgt:
                continue
            
            
            clusters = self._find_contiguous_clusters(mapped_tgt, max_gap)
            clusters = self._fill_gaps_in_clusters(clusters)
            largest = self._cluster_merge(clusters, src_token_count=len(covered_src), base_gap=max_gap, mass_factor=0.6)
            
            min_idx, max_idx = min(largest), max(largest)
            tgt_start = result.tgt_tokens[min_idx].start
            tgt_end = result.tgt_tokens[max_idx].end
            


            if self.verbose:
                print(f"1.2 Performing clustering.")
                print(f"1.3 No gap results: {self._find_contiguous_clusters(mapped_tgt, max_gap=0)}")
                print(f"1.4 Computed Alignments with result:", clusters)
                print(f"1.5 Merged clusters into largest:", largest)

                print(f"1.6 Covered source tokens: {[result.src_tokens[i].text for i in covered_src]}")
                print(f"1.7 Mapped target tokens: {[result.tgt_tokens[i].text for i in mapped_tgt]}")
                print(f"1.8 Projected span: [{min_idx}->{tgt_start}, {max_idx}->{tgt_end})")
                print(f"1.9 Sim matrix shape: {result.similarity_matrix.shape}")

                print(f"-----Alignment Details-----")
                for pair in alignment:
                    s_idx, t_idx = pair
                    if s_idx in covered_src:
                        score = result.similarity_matrix[s_idx, t_idx]
                        print(f"Src token ({s_idx}) '{result.src_tokens[s_idx].text}' -> Tgt token ({t_idx}) '{result.tgt_tokens[t_idx].text}' | Score: {score:.4f}")
            
            # Build token alignment details
            token_alignments = []
            for src_idx in covered_src:
                best_tgt, best_score = None, 0
                for tgt_idx in largest:
                    if src_idx < result.similarity_matrix.shape[0] and tgt_idx < result.similarity_matrix.shape[1]:
                        score = result.similarity_matrix[src_idx, tgt_idx]
                        if score > best_score:
                            best_score, best_tgt = score, tgt_idx
                
                # Check cross-similarity for display
                best_tgt_any = None
                best_score_any = 0
                for ti in range(len(result.tgt_tokens)):
                    if src_idx < result.similarity_matrix.shape[0] and ti < result.similarity_matrix.shape[1]:
                        if result.similarity_matrix[src_idx, ti] > best_score_any:
                            best_score_any = result.similarity_matrix[src_idx, ti]
                            best_tgt_any = ti
                    
                token_alignments.append({
                    'src_idx': src_idx,
                    'src_token': result.src_tokens[src_idx].text if src_idx < len(result.src_tokens) else '?',
                    'tgt_idx': best_tgt if best_tgt is not None else best_tgt_any,
                    'tgt_token': result.tgt_tokens[best_tgt].text if best_tgt is not None else (result.tgt_tokens[best_tgt_any].text if best_tgt_any is not None else '?'),
                    'score': float(best_score if best_tgt is not None else best_score_any),
                    'in_cluster': best_tgt is not None
                })
            
            projected.append({
                "start": tgt_start,
                "end": tgt_end,
                "text": tgt_text[tgt_start:tgt_end],
                "labels": span.get("labels", []),
                "cluster_size": len(largest),
                "total_aligned": len(mapped_tgt), # Tokens in src mapped to something in tgt
                "num_clusters": len(clusters),
                "token_alignments": token_alignments,
                "all_clusters": clusters
            })
        
        return projected


    def project_tagged_text(self,
                            src_text: str,
                            tgt_text: str,
                            allowed_tags: Optional[List[str]] = None,
                            max_gap: int = 1,
                            debugging: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Project annotations from tagged source text to target text.
        
        Args:
            src_text: Tagged source text
            tgt_text: Target text (raw)
            
        Returns:
            Tuple of (Tagged Target Text, Projected Spans)
        """
        parsed = SpanAligner.tagged_text_to_task(src_text, allowed_tags=allowed_tags)
        raw_src_text = parsed["task"]["data"]["text"]
        src_spans = parsed["spans"] + parsed["entities"]
        
        #print(f"Projecting following text:\nSRC: {raw_src_text}\nTGT: {tgt_text}\nWith spans: {src_spans}")

        projected = self.project_spans(raw_src_text, tgt_text, src_spans, max_gap=max_gap, debugging=debugging)
        tagged_tgt, _ = SpanAligner.rebuild_tagged_text(tgt_text, projected, [])
        
        return tagged_tgt, projected

    def project_tagged_text_long(self,
                                 src_text: str,
                                 tgt_text: str,
                                 allowed_tags: Optional[List[str]] = None,
                                 window_size: int = 512,
                                 max_gap: int = 1,
                                 debugging: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
        """Project annotations for long documents using token window alignment."""
        if allowed_tags is None:
            allowed_tags = list(set(re.findall(r"<([a-zA-Z_][a-zA-Z0-9_-]*)>", src_text)))
        
        parsed = SpanAligner.tagged_text_to_task(src_text, allowed_tags=allowed_tags)
        raw_src_text = parsed["task"]["data"]["text"]
        all_spans = parsed["spans"] + parsed["entities"]
        
        if not all_spans:
            return tgt_text, []
        
        import bisect

        # 1. Tokenize both the src and target to get character offsets
        def get_tokens(text: str) -> List[Tuple[int, int]]:
            """Return list of (start, end) character offsets for tokens."""
            # Simple regex tokenizer that captures words and punctuation
            return [m.span() for m in re.finditer(r'\w+|[^\w\s]', text)]
        
        src_tokens = get_tokens(raw_src_text)
        tgt_tokens = get_tokens(tgt_text)

        if not src_tokens or not tgt_tokens:
             return tgt_text, []
        
        src_starts = [t[0] for t in src_tokens]
        projected = []

        # 2. Iterate over the spans of the src
        half_window = window_size // 2

        for span in all_spans:
            # Get relative position in text (center of span)
            span_center = (span["start"] + span["end"]) / 2
            
            # Find closest token index in src
            center_idx = bisect.bisect_left(src_starts, span_center)
            # bisect_left gives index where it could be inserted. 
            # If it's exact match or after, we might want index-1 or index. 
            # Let's just constrain it to bounds.
            center_idx = max(0, min(len(src_tokens) - 1, center_idx))
            
            # Create window of tokens around the span
            src_start_idx = max(0, center_idx - half_window)
            src_end_idx = min(len(src_tokens), center_idx + half_window)
            
            # Get character ranges for source window
            win_src_start_char = src_tokens[src_start_idx][0]
            win_src_end_char = src_tokens[src_end_idx-1][1]
            src_window_text = raw_src_text[win_src_start_char:win_src_end_char]
            
            # 3. Get the corresponding window in the tgt tokens
            # Map relative position: (src_idx / src_len) -> tgt_idx
            if len(src_tokens) > 1:
                rel_pos = center_idx / (len(src_tokens) - 1)
            else:
                rel_pos = 0.5
            
            tgt_center_idx = int(rel_pos * len(tgt_tokens))
            tgt_start_idx = max(0, tgt_center_idx - half_window)
            tgt_end_idx = min(len(tgt_tokens), tgt_center_idx + half_window)
            
            win_tgt_start_char = tgt_tokens[tgt_start_idx][0]
            win_tgt_end_char = tgt_tokens[tgt_end_idx-1][1]
            tgt_window_text = tgt_text[win_tgt_start_char:win_tgt_end_char]
            
            # 4. Perform apply project_tagged_text
            # We need to adjust the span to be relative to the source window
            rel_span = span.copy()
            rel_span["start"] -= win_src_start_char
            rel_span["end"] -= win_src_start_char
            
            # Sanity check: span must be within extracted text
            # (It should be, unless window is smaller than the span itself)
            if rel_span["start"] < 0 or rel_span["end"] > len(src_window_text):
                continue
                
            # Project just this span within the window
            # We pass a list containing just the single span
            window_projections = self.project_spans(
                src_window_text, 
                tgt_window_text, 
                [rel_span], 
                max_gap=max_gap, 
                debugging=debugging
            )
            
            # 5. Map the positions of the projected spans back onto the full text
            for p in window_projections:
                p["start"] += win_tgt_start_char
                p["end"] += win_tgt_start_char
                projected.append(p)
        
        tagged_tgt, _ = SpanAligner.rebuild_tagged_text(tgt_text, projected, [])
        return tagged_tgt, projected

    def _visualize_projection(self, projected_spans: List[Dict[str, Any]]) -> str:
        """Generate visualization of projected spans with alignment info."""
        lines = ["=" * 80, "PROJECTION VISUALIZATION", "=" * 80]
        
        for i, span in enumerate(projected_spans):
            lines.append(f"\nSPAN {i}: {span.get('labels', [])}")
            lines.append("-" * 80)
            lines.append(f"Projected: '{span['text']}' [{span['start']}, {span['end']})")
            
            if 'num_clusters' in span:
                lines.append(f"Clusters: {span['num_clusters']} (selected: {span['cluster_size']} tokens)")
            
            if 'token_alignments' in span:
                lines.append("Token alignments:")
                for a in span['token_alignments']:
                    marker = "✅" if a['score'] >= 0.95 else "✓" if a['score'] >= 0.85 else "⚠️" if a['score'] >= 0.7 else "❌"
                    tgt_part = f"-> '{a['tgt_token']}'" if a.get('in_cluster') else f"(best match: '{a['tgt_token']}')"
                    lines.append(f"  {marker} '{a['src_token']}' {tgt_part} ({a['score']:.3f})")
        
        return "\n".join(lines)
