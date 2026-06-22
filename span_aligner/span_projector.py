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
    
    STOPWORDS = frozenset({
        'the', 'a', 'an', 'of', 'for', 'in', 'to', 'and', 'or', 'is', 'are',
        'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could',
        'that', 'this', 'these', 'those', 'it', 'its', 'at', 'by', 'on', 'with',
        'from', 'as', 'but', 'not', 'no', 'so', 'if', 'then', 'than',
        'de', 'het', 'een', 'van', 'voor', 'op', 'met', 'aan', 'en', 'die',
        'dat', 'der', 'des', 'den', 'te', 'tot', 'om', 'bij', 'naar', 'uit',
        'als', 'ook', 'er', 'nog', 'wel', 'niet', 'maar', 'dan', 'dus', 'werd',
        'worden', 'wordt', 'zijn', 'was', 'over'
    })
    
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
            matching_method: Matching method (mwmf, itermax, inter, greedy, threshold)
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
    def _get_char_tokens(self, text: str) -> List[Tuple[int, int]]:
        """Return (start, end) character offsets for simple word/punctuation tokens."""
        return [m.span() for m in re.finditer(r'\w+|[^\w\s]', text)]
    
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
 
    def _cluster_merge(self, clusters: List[List[int]], src_token_count: Optional[int] = None, base_gap: int = 2, mass_factor: float = 0.5, expected_center: Optional[float] = None, alignment_weights: Optional[Dict[int, float]] = None) -> List[int]:
        """
        Merges multiple alignment clusters into the single most likely contiguous sequence.
        
        Args:
            clusters: List of index lists, e.g., [[1,2], [5,6,7], [20,21]].
            src_token_count: Length of source span (used for sanity checking).
            base_gap: Minimum gap always allowed (handles punctuation/particles).
            mass_factor: Multiplier for dynamic tolerance. 
                        Allowed Gap = base_gap + (min(len(A), len(B)) * mass_factor).
            expected_center: Expected center of the target region (from region estimation).
            alignment_weights: Maps target token index → importance weight (from source token).
                              When provided, clusters are scored by sum of weights of actual
                              aligned tokens rather than raw gap-filled count.
                        
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
            # We use the maximum mass because a weak link (size 1) breaks the chain easily.
            connection_mass = max(len(prev_cluster), len(next_cluster))
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
        best_score = (-1.0, 0.0, -1) # (weighted_count, proximity, -span_length)
        
        for chain in chains:
            # Flatten the chain into a single list of indices
            flat_indices = [idx for cluster in chain for idx in cluster]
            
            if not flat_indices: 
                continue
            
            # Score by importance-weighted actual alignment count if weights provided,
            # otherwise fall back to raw gap-filled count.
            # This prevents gap-filling from inflating cluster scores unfairly:
            # e.g. 5 real alignments in [36-44] filled to 9 indices shouldn't beat
            # 5 dense high-importance alignments in [0-4].
            if alignment_weights:
                weighted_count = sum(alignment_weights.get(idx, 0.0) for idx in flat_indices)
            else:
                weighted_count = float(len(flat_indices))
                
            span_length = flat_indices[-1] - flat_indices[0] + 1
            
            # 4. Outlier Sanity Check (if source length is known)
            # If the target span is massively larger than source (e.g. > 4x), 
            # it's likely a bad "spread" alignment. We penalize it heavily.
            if src_token_count and span_length > (src_token_count * 4) + 6:
                weighted_count = 0.0 # Disqualify this chain effectively
                
            # 5. Proximity to expected center (closer = higher = better)
            # Promoted from tiebreaker to secondary sort key so that region
            # estimation actually influences cluster selection.
            if expected_center is not None:
                chain_center = (flat_indices[0] + flat_indices[-1]) / 2
                proximity = -abs(chain_center - expected_center)
            else:
                proximity = 0.0
            
            # 6. Scoring
            # Priority 1: Maximize importance-weighted alignment count
            # Priority 2: Prefer chain closer to estimated region center
            # Priority 3: Minimize total span length (prefer compactness)
            score = (weighted_count, proximity, -span_length)
            
            if score > best_score:
                best_score = score
                best_chain_indices = flat_indices
                
        return best_chain_indices

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
    # Refinement methods for clusters and sequences
    # -------------------------------------------------------------------------


    def _estimate_target_region(self, topk_candidates: Dict[int, List[Tuple[int, float]]],
                                src_tokens: List[TokenInfo], n_tgt: int,
                                padding: float = 0.3) -> Tuple[int, int]:
        """
        Estimate the target region where the span is most likely to be found.
        Uses majority voting from top-1 candidates of content tokens only.
        
        Returns (region_start, region_end) in target token indices.
        """
        if not topk_candidates:
            return (0, n_tgt)
        
        # Collect top-1 vote per content token (skip stopwords/punctuation)
        votes = []
        for src_idx in sorted(topk_candidates.keys()):
            if topk_candidates[src_idx] and src_idx < len(src_tokens):
                importance = self._compute_token_importance(src_tokens[src_idx].text)
                if importance < 0.5:
                    continue
                best_tgt_idx = topk_candidates[src_idx][0][0]
                votes.append(best_tgt_idx)
        
        # Fallback: use all votes if filtering leaves nothing
        if not votes:
            for src_idx in sorted(topk_candidates.keys()):
                if topk_candidates[src_idx]:
                    votes.append(topk_candidates[src_idx][0][0])
        
        if not votes:
            return (0, n_tgt)
        
        # Cluster the votes and pick majority
        clusters = self._find_contiguous_clusters(votes, max_gap=3)
        best_cluster = max(clusters, key=len)
        
        cluster_width = best_cluster[-1] - best_cluster[0] + 1
        pad = max(3, int(cluster_width * padding))
        
        region_start = max(0, best_cluster[0] - pad)
        region_end = min(n_tgt, best_cluster[-1] + pad + 1)

        return (region_start, region_end)

    def _filter_exact_match_candidates(self, topk_candidates: Dict[int, List[Tuple[int, float]]],
                                       src_tokens: List[TokenInfo],
                                       tgt_tokens: List[TokenInfo]) -> Dict[int, List[Tuple[int, float]]]:
        """
        When candidates exist with the exact same text as the source token,
        keep only those exact-match candidates.
        
        E.g., src '2021' with candidates ['2021','2022','2018'] -> keep only '2021' matches.
             src '1' with candidates ['1', '.', 'sub'] -> keep only '1' matches.
        """
        filtered = {}
        for src_idx, candidates in topk_candidates.items():
            if not candidates:
                filtered[src_idx] = candidates
                continue
            
            src_text = src_tokens[src_idx].text.lower().strip()
            
            exact_matches = [
                (tgt_idx, score) for tgt_idx, score in candidates
                if tgt_tokens[tgt_idx].text.lower().strip() == src_text
            ]
            
            if exact_matches:
                filtered[src_idx] = exact_matches
                if self.verbose and len(exact_matches) < len(candidates):
                    print(f"[Filtering] Exact-match filter: src[{src_idx}] '{src_text}' "
                          f"kept {len(exact_matches)}/{len(candidates)} candidates")
            else:
                filtered[src_idx] = candidates
        
        return filtered

    def _compute_token_importance(self, token_text: str) -> float:
        """
        Compute importance weight for a token. Low-information tokens
        (punctuation, stopwords) get reduced weight so they don't dominate alignment.
        
        Returns a weight between 0.1 and 1.0.
        """
        text = token_text.strip()
        
        if not text:
            return 0.1
        
        # Punctuation
        if all(not c.isalnum() for c in text):
            return 0.1
        
        # Stopwords (EN + NL)
        if text.lower() in self.STOPWORDS:
            return 0.3
        
        # Numbers
        if text.isdigit():
            return 0.8
        
        # Content words
        return 1.0

 

    def _sequence_aware_refinement(self, topk_candidates: Dict[int, List[Tuple[int, float]]], 
                                   src_tokens: List[TokenInfo], 
                                   tgt_tokens: List[TokenInfo],
                                   estimated_region: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Multi-path sequence-aware alignment using dynamic programming.

        Finds optimal assignment of target tokens to source tokens by balancing
        embedding similarity with sequential coherence penalties, then extracts
        multiple diverse paths to represent ALL plausible target clusters.

        Key improvements:
        1. Lookback window: considers transitions from the last LOOKBACK valid
           previous source positions (not just the immediately previous one).
           This lets the path skip noisy intermediate tokens (e.g. stopwords
           whose only candidates are in distant regions) without dragging the
           entire alignment away from the correct cluster.
        2. Multi-path extraction: after the DP, backtracks from the top-scoring
           ending states and keeps paths that cover distinct target clusters.
           Returns the union of all diverse paths so downstream cluster_merge
           sees all plausible clusters and can pick the correct one.

        Scoring per transition (src[prev_i] → src[i], tgt[prev] → tgt[curr]):
        - Base: similarity score of (src[i], tgt[curr])
        - Penalty: based on deviation from expected sequential target position
        - Weight: multiplied by token importance (content > stopwords > punct)
        - Bias: small bonus for candidates near expected proportional position
        """
        if not topk_candidates:
            return []
            
        n_src = len(src_tokens)
        n_tgt = len(tgt_tokens)
        
        # Compute expected stride using estimated region (local) or full text (global)
        if estimated_region:
            region_width = estimated_region[1] - estimated_region[0]
            expected_stride = max(region_width / max(n_src, 1), 1.0)
        else:
            region_width = n_tgt
            expected_stride = max(n_tgt / max(n_src, 1), 1.0)
        
        # Sorted list of source indices that actually have candidates
        valid_positions = sorted(
            i for i in range(n_src)
            if i in topk_candidates and topk_candidates[i]
        )
        if not valid_positions:
            return []
        
        # How many previous valid positions to consider per transition.
        # Allows the DP to "skip" up to LOOKBACK-1 noisy intermediate tokens
        # whose candidates are all in distant regions.
        LOOKBACK = 3
        
        # dp[i][k] stores (cumulative_score, predecessor_src_idx, predecessor_candidate_k)
        dp: Dict[int, List[Tuple[float, int, int]]] = {}
        for i in valid_positions:
            dp[i] = [(-float('inf'), -1, -1)] * len(topk_candidates[i])
        
        # --- Initialize first valid position ---
        first_i = valid_positions[0]
        importance = self._compute_token_importance(src_tokens[first_i].text)
        for k, (tgt_idx, score) in enumerate(topk_candidates[first_i]):
            if estimated_region:
                expected_tgt = (estimated_region[0] + (first_i / max(n_src - 1, 1)) * max(region_width - 1, 0)) if n_src > 1 else estimated_region[0]
            else:
                expected_tgt = (first_i / max(n_src - 1, 1)) * max(n_tgt - 1, 0) if n_src > 1 else 0
            pos_bias = max(0, 0.05 - 0.001 * abs(tgt_idx - expected_tgt))
            dp[first_i][k] = ((score + pos_bias) * importance, -1, -1)
            
        # --- DP transitions with lookback window ---
        for pos_idx in range(1, len(valid_positions)):
            i = valid_positions[pos_idx]
            importance = self._compute_token_importance(src_tokens[i].text)
            
            lb_start = max(0, pos_idx - LOOKBACK)
            
            for k, (tgt_curr, score_curr) in enumerate(topk_candidates[i]):
                best_score = -float('inf')
                best_prev_i = -1
                best_prev_k = -1
                
                # Consider transitions from up to LOOKBACK previous valid positions
                for prev_pos in range(lb_start, pos_idx):
                    prev_i = valid_positions[prev_pos]
                    src_gap = i - prev_i
                    
                    for prev_k, (tgt_prev, _) in enumerate(topk_candidates[prev_i]):
                        prev_path_score = dp[prev_i][prev_k][0]
                        if prev_path_score == -float('inf'):
                            continue
                        
                        # Calculate penalty based on target position difference
                        diff = tgt_curr - tgt_prev
                        expected_diff = src_gap * expected_stride
                        
                        if diff == 1 or (diff > 0 and abs(diff - expected_diff) < 1.5):
                            penalty = 0.0  # Perfect or near-perfect sequence
                        elif diff == 0:
                            penalty = 0.1  # Same target token (many-to-one)
                        elif diff > 1:
                            excess = max(0, diff - expected_diff)
                            penalty = 0.15 * excess
                        else:
                            # Out of order: penalty proportional to distance
                            penalty = 0.4 * abs(diff) + 0.3
                            
                        penalty = min(penalty, 3.0)
                        current_path_score = prev_path_score + (score_curr - penalty) * importance
                        
                        if current_path_score > best_score:
                            best_score = current_path_score
                            best_prev_i = prev_i
                            best_prev_k = prev_k
                        
                dp[i][k] = (best_score, best_prev_i, best_prev_k)
                
        # --- Multi-path extraction ---
        # Backtrack from multiple ending states to recover diverse paths
        # covering different target clusters. Return their union so downstream
        # cluster_merge sees all plausible target regions.
        
        def _backtrack(end_i: int, end_k: int) -> List[Tuple[int, int]]:
            """Trace back from an ending state to recover the full path."""
            path = []
            ci, ck = end_i, end_k
            while ci != -1 and ck != -1:
                path.append((ci, topk_candidates[ci][ck][0]))
                ni, nk = dp[ci][ck][1], dp[ci][ck][2]
                ci, ck = ni, nk
            path.reverse()
            return path
        
        last_i = valid_positions[-1]
        ending_states = sorted(
            [(dp[last_i][k][0], k) for k in range(len(topk_candidates[last_i]))
             if dp[last_i][k][0] > -float('inf')],
            reverse=True
        )
        
        if not ending_states:
            return [], []
        
        # Extract up to MAX_DIVERSE_PATHS paths that cover distinct target clusters
        all_alignments: set = set()
        seen_signatures: set = set()
        MAX_DIVERSE_PATHS = 5
        paths_found = 0
        
        paths_list = []
        for score, k in ending_states:
            if paths_found >= MAX_DIVERSE_PATHS:
                break
            
            path = _backtrack(last_i, k)
            
            # Compute cluster signature: which contiguous target regions does
            # this path cover? Paths with different signatures are "diverse".
            tgt_indices = sorted(set(t for _, t in path))
            clusters = self._find_contiguous_clusters(tgt_indices, max_gap=5)
            signature = tuple((c[0], c[-1]) for c in clusters)
            
            if signature not in seen_signatures or paths_found == 0:
                seen_signatures.add(signature)
                all_alignments.update(path)
                paths_found += 1
                
                path_nodes = []
                for src_idx, tgt_idx in path:
                    rank = 0
                    for r, (cand_tgt, _) in enumerate(topk_candidates[src_idx]):
                        if cand_tgt == tgt_idx:
                            rank = r
                            break
                    path_nodes.append((src_idx, rank, tgt_idx))
                paths_list.append({'score': score, 'path': path_nodes})
                
        return sorted(list(all_alignments)), paths_list

    def _path_dp_refinement(self,
                            topk_candidates: Dict[int, List[Tuple[int, float]]],
                            src_tokens: List[TokenInfo],
                            tgt_tokens: List[TokenInfo],
                            estimated_region: Optional[Tuple[int, int]] = None,
                            use_sim_score: bool = True,
                            max_skip: int = 5,
                            max_paths: int = 5) -> Tuple[List[Tuple[int, int]], List[Dict]]:
        """
        Order-agnostic DP path alignment — an alternative to _sequence_aware_refinement.

        Key differences from _sequence_aware_refinement
        ------------------------------------------------
        1. Any (src_idx, rank) cell can end a path, not only the last valid source
           position. This means strong early clusters are not discarded just because
           later source tokens happen to be noisy or have scattered candidates.

        2. Continuity is order-agnostic: abs(abs(tgt_gap) - src_gap) == 0 is ideal
           regardless of whether the target position moves forward or backward.
           Local reorderings (compound splits, reversed translations) cost nothing.

        3. Token importance weighting: node quality is multiplied by token importance
           (content words 1.0 > numbers 0.8 > stopwords 0.3 > punctuation 0.1).
           This prevents stopwords from dominating the alignment.

        Scoring
        -------
        quality = (sim_score * 10  if use_sim_score else  10 - rank * 1.5) * importance

        Fresh start at position pos_idx in valid_positions:
            score = quality - pos_idx * 2.0

        Extend from predecessor (prev_i → i):
            step  = quality - (src_gap - 1) * 2.0 - abs(abs(tgt_gap) - src_gap)
            total = prev_score + step
            src_gap = i - prev_i  (raw token-index gap)
            tgt_gap = tgt_idx - prev_tgt

        Multi-path extraction
        ---------------------
        All (i, j) states are ranked by score. The top states are backtraced and
        those covering distinct target-cluster signatures are merged, so downstream
        _cluster_merge sees all plausible target regions.

        Parameters
        ----------
        use_sim_score : bool
            True  (recommended): quality = sim_score * 10.  Uses actual similarity.
            False (rank proxy) : quality = 10 - rank * 1.5. Weaker but sim-agnostic.
        max_skip : int
            Maximum raw source-token-index gap allowed to a predecessor.
        max_paths : int
            Number of distinct target-cluster paths to extract and merge.
        """
        valid_positions = sorted(i for i in topk_candidates if topk_candidates[i])
        if not valid_positions:
            return [], []

        pos_of = {i: pos for pos, i in enumerate(valid_positions)}

        # dp[i][j] = (best_cumulative_score, prev_src_idx, prev_rank)
        dp: Dict[int, Dict[int, Tuple[float, int, int]]] = {i: {} for i in valid_positions}

        for i in valid_positions:
            pos_idx = pos_of[i]
            importance = self._compute_token_importance(src_tokens[i].text) if i < len(src_tokens) else 1.0

            for j, (tgt_idx, sim_score) in enumerate(topk_candidates[i]):
                # Weighted quality: high-importance tokens drive alignment more strongly
                quality = (sim_score * 10.0 if use_sim_score else 10.0 - j * 1.5) * importance

                # Fresh-start: mild penalty for skipping earlier source positions
                best_score = quality - pos_idx * 2.0
                best_prev = (-1, -1)

                for prev_i in [p for p in valid_positions if p < i and i - p <= max_skip]:
                    for prev_j, (prev_tgt, _) in enumerate(topk_candidates[prev_i]):
                        if prev_j not in dp[prev_i]:
                            continue

                        prev_score = dp[prev_i][prev_j][0]
                        src_gap  = i - prev_i
                        tgt_gap  = tgt_idx - prev_tgt

                        # Order-agnostic continuity: only distance deviation matters
                        step = (quality
                                - (src_gap - 1) * 2.0
                                - abs(abs(tgt_gap) - src_gap))

                        if prev_score + step > best_score:
                            best_score = prev_score + step
                            best_prev = (prev_i, prev_j)

                dp[i][j] = (best_score, best_prev[0], best_prev[1])

        paths = []
        for i in valid_positions:
            for j in range(len(topk_candidates[i])):
                if j not in dp[i]:
                    continue
                score, _, _ = dp[i][j]
                path, ci, cj = [], i, j
                while ci != -1:
                    path.append((ci, cj, topk_candidates[ci][cj][0]))
                    _, ci, cj = dp[ci][cj]
                paths.append({'score': score, 'path': list(reversed(path)), 'end_i': i})

        paths.sort(key=lambda x: x['score'], reverse=True)

        filtered, seen = [], []
        for p in paths:
            nodes = set((n[0], n[1]) for n in p['path'])
            if not any(nodes.issubset(s) for s in seen):
                filtered.append(p)
                seen.append(nodes)

        all_alignments = set()
        for p in filtered[:max_paths]:
            for src_idx, rank, tgt_idx in p['path']:
                all_alignments.add((src_idx, tgt_idx))

        return sorted(list(all_alignments)), filtered

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


    # -------------------------------------------------------------------------
    # Main projection methods
    # -------------------------------------------------------------------------


    # The main method that projects spans using the top-K candidates and a sequence-aware refinement step.
    def project_spans(self,
                           src_text: str,
                           tgt_text: str,
                           src_spans: List[Dict[str, Any]],
                           max_gap: int = 5,
                           top_k: int = 5,
                           max_paths: int = 1,
                           debugging: bool = False,
                           refinement: str = "path_dp") -> List[Dict[str, Any]]:
        """
        Projects spans from source text to target text using advanced alignment refinement.

        This method uses a multi-step process:
        1. Embeds source and target texts using the configured embedding model.
        2. Computes top-K candidate alignments for each source token.
        3. Refines the alignment using dynamic programming (DP) to find optimal paths.
        4. Clusters the aligned tokens to form contiguous spans in the target text.

        Args:
            src_text: The source text containing the original spans.
            tgt_text: The target text to project the spans onto.
            src_spans: A list of dictionaries representing the spans in the source text.
                       Each dict must contain 'start', 'end', and optionally 'labels'/'text'.
            max_gap: Maximum allowed gap (in tokens) between aligned tokens to be considered part of the same cluster.
                     Defaults to 5.
            top_k: Number of top candidate target tokens to consider for each source token.
                   Defaults to 5.
            max_paths: Maximum number of diverse alignment paths to consider during refinement.
                       Defaults to 5.
            debugging: If True, includes detailed debug information in the output, such as token alignments and all clusters.
                       Defaults to False.
            refinement: The refinement strategy to use. Options are:
                        - "path_dp": Order-agnostic DP (recommended).
                        - "sequence": Sequence-aware DP with forward bias.
                        Defaults to "path_dp".

        Returns:
            A list of dictionaries representing the projected spans in the target text.
            Each dictionary contains 'start', 'end', 'text', 'labels', and optionally debug info.
        """
        spans, _ = self.project_spans_with_renderings(
            src_text, tgt_text, src_spans, max_gap, top_k, max_paths, debugging, refinement
        )
        return spans
    
    def project_spans_with_renderings(self,
                           src_text: str,
                           tgt_text: str,
                           src_spans: List[Dict[str, Any]],
                           max_gap: int = 5,
                           top_k: int = 5,
                           max_paths: int = 1,
                           debugging: bool = False,
                           refinement: str = "path_dp") -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Projects spans and returns visualizations of the alignment process.

        This mechanism is identical to `project_spans` but also returns HTML strings
        that visualize the alignment matrix and selected paths, useful for debugging or analysis in notebooks.

        Args:
            src_text: The source text containing the original spans.
            tgt_text: The target text to project the spans onto.
            src_spans: A list of dictionaries representing the spans in the source text.
            max_gap: Maximum allowed gap (in tokens) between aligned tokens. Defaults to 5.
            top_k: Number of top candidate target tokens to consider per source token. Defaults to 5.
            max_paths: Maximum number of diverse alignment paths to extract. Defaults to 5.
            debugging: If True, includes detailed debug information in the output spans. Defaults to False.
            refinement: The refinement strategy to use ("path_dp" or "sequence"). 
                        - "sequence" (default): _sequence_aware_refinement — forward-biased DP
                          with lookback window and position bias. Good general-purpose baseline.
                        - "path_dp"           : _path_dp_refinement — order-agnostic DP that
                          treats any (src, rank) cell as a valid path endpoint. Handles local
                          reorderings and reversed translations better. Recommended with top_k ≥ 5.

        Returns:
            A tuple containing:
            1. List[Dict[str, Any]]: The projected spans.
            2. List[str]: HTML strings for visualizing the alignment of each span.
        """
        if not src_text or not tgt_text or not src_spans:
            return [], []

        src_tokens, src_vectors = self.aligner.get_text_embeddings(src_text)
        tgt_tokens, tgt_vectors = self.aligner.get_text_embeddings(tgt_text)

        projected = []
        renderings = []

        for span in src_spans:
            if self.verbose:
                print(f"{'═'*60}")
                print(f"\n  ┌─ Span: '{span.get('text','')}'"
                      f"  [{span['start']}:{span['end']}]")

            try:                
                result = self.aligner.align_texts_partial_with_embeddings(
                    src_tokens,
                    tgt_tokens,
                    src_vectors,
                    tgt_vectors,
                    src_char_start=span['start'],
                    src_char_end=span['end'],
                    top_k=top_k
                )
            except ValueError:
                if self.verbose:
                    print(f"  └─ Skipped (no token coverage).")
                projected.append(None)
                renderings.append(None)
                continue

            # 1. Filter: keep only exact-match candidates when available
            filtered_candidates = self._filter_exact_match_candidates(
                result.topk_candidates, result.src_tokens, result.tgt_tokens
            )
            if self.verbose:
                print(f"{'═'*60}")
                print(f"  Top-K candidates (after exact-match filtering):")
                for src_idx in sorted(filtered_candidates.keys()):
                    src_tok = result.src_tokens[src_idx].text if src_idx < len(result.src_tokens) else '?'
                    print(f"  ├─ src[{src_idx}] : '{src_tok}'")
                    candidates = filtered_candidates[src_idx]
                    if candidates:
                        for tgt_rank, (tgt_idx, score) in enumerate(candidates):
                            tgt_tok = result.tgt_tokens[tgt_idx].text if tgt_idx < len(result.tgt_tokens) else '?'
                            print(f"  │   ├─ rank {tgt_rank}: tgt[{tgt_idx}] (score: {score:.4f}) '{tgt_tok}'")
                    else:
                        print(f"  │   └─ No candidates")
                print(f"{'═'*60}")

            # 2. Estimate target region from content-token votes
            region = self._estimate_target_region(
                filtered_candidates, result.src_tokens, len(result.tgt_tokens)
            )

            if self.verbose:
                print(f"  │  Region estimate: tokens [{region[0]}..{region[1]}]")

            # 3. Alignment refinement (method selected by `refinement` parameter)
            if refinement == "path_dp":
                alignment, paths = self._path_dp_refinement(
                    filtered_candidates,
                    result.src_tokens,
                    result.tgt_tokens,
                    estimated_region=region,
                    max_paths=max_paths,
                )
            else:
                alignment, paths = self._sequence_aware_refinement(
                    filtered_candidates,
                    result.src_tokens,
                    result.tgt_tokens,
                    estimated_region=region
                )

            src_to_tgt = {}
            for src_idx, tgt_idx in alignment:
                src_to_tgt.setdefault(src_idx, []).append(tgt_idx)

            covered_src = [i for i, t in enumerate(result.src_tokens)
                           if max(span['start'], t.start) < min(span['end'], t.end)]

            mapped_tgt = []
            for s_idx in covered_src:
                mapped_tgt.extend(src_to_tgt.get(s_idx, []))

            if not mapped_tgt:
                if self.verbose:
                    print(f"  └─ No target tokens mapped.")
                projected.append(None)
                renderings.append(None)
                continue

            clusters = self._find_contiguous_clusters(mapped_tgt, max_gap)
            clusters = self._fill_gaps_in_clusters(clusters)
            expected_center = (region[0] + region[1]) / 2 if region else None

            # Compute importance weight for each aligned target index.
            # Uses the max importance of any source token mapping to that target,
            # so content tokens (importance=1.0) outweigh function words (0.1-0.3).
            alignment_weights: Dict[int, float] = {}
            for src_idx in covered_src:
                importance = self._compute_token_importance(
                    result.src_tokens[src_idx].text
                ) if src_idx < len(result.src_tokens) else 1.0
                for tgt_idx in src_to_tgt.get(src_idx, []):
                    alignment_weights[tgt_idx] = max(
                        alignment_weights.get(tgt_idx, 0.0), importance
                    )

            largest = self._cluster_merge(
                clusters,
                src_token_count=len(covered_src),
                base_gap=max_gap,
                mass_factor=0.0,
                expected_center=expected_center,
                alignment_weights=alignment_weights
            )

            if self.verbose:
                # print(self.visualize_alignment_text(
                #     filtered_candidates,
                #     result.src_tokens, 
                #     result.tgt_tokens,
                #     paths=paths,
                #     final_indices=largest
                # ))
                renderings.append(self.visualize_alignment_html(
                    filtered_candidates,
                    result.src_tokens, 
                    result.tgt_tokens,
                    paths=paths,
                    #final_indices=largest
                ))


                print(f"  │  Clusters: {clusters}")

            if not largest:
                if self.verbose:
                    print(f"  └─ No cluster selected.")
                projected.append(None)
                renderings.append(None)
                continue
                
            min_idx, max_idx = min(largest), max(largest)
            tgt_start = result.tgt_tokens[min_idx].start
            tgt_end = result.tgt_tokens[max_idx].end

            if self.verbose:
                print(f"  └─ Result: '{tgt_text[tgt_start:tgt_end]}'"
                      f"  [{tgt_start}:{tgt_end}]  (token indices {min_idx}..{max_idx})")

            if debugging:
                token_alignments = []
                for src_idx in covered_src:
                    best_tgt = src_to_tgt.get(src_idx, [None])[0]
                    # Get score from topk_candidates (reliable) rather than sim matrix (partial/BPE-indexed)
                    score = 0.0
                    if best_tgt is not None and src_idx in filtered_candidates:
                        for cand_tgt, cand_score in filtered_candidates[src_idx]:
                            if cand_tgt == best_tgt:
                                score = cand_score
                                break
                        
                    token_alignments.append({
                        'src_idx': src_idx,
                        'src_token': result.src_tokens[src_idx].text if src_idx < len(result.src_tokens) else '?',
                        'tgt_idx': best_tgt,
                        'tgt_token': result.tgt_tokens[best_tgt].text if best_tgt is not None else '?',
                        'score': float(score),
                        'in_cluster': best_tgt in largest if best_tgt is not None else False
                    })
                    
                projected.append({
                    "start": tgt_start,
                    "end": tgt_end,
                    "text": tgt_text[tgt_start:tgt_end],
                    "labels": span.get("labels", []),
                    "cluster_size": len(largest),
                    "total_aligned": len(mapped_tgt),
                    "num_clusters": len(clusters),
                    "token_alignments": token_alignments,
                    "all_clusters": clusters
                })
            else:
                projected.append({
                    "start": tgt_start,
                    "end": tgt_end,
                    "text": tgt_text[tgt_start:tgt_end],
                    "labels": span.get("labels", [])
                })
                
        return projected, renderings

    def project_tagged_text(self,
                            src_text: str,
                            tgt_text: str,
                            allowed_tags: Optional[List[str]] = None,
                            max_gap: int = 1,
                            debugging: bool = False) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Projects annotations from a tagged source text to a raw target text.

        This method first parses the tagged source text to extract spans, then
        projects those spans onto the target text, and finally returns the
        reconstructed tagged target text.

        Args:
            src_text: The source text containing XML-like tags (e.g., "<ORG>Oracle</ORG>").
            tgt_text: The raw target text to project annotations onto.
            allowed_tags: Ideally a List of strings containing the tags to extract.
                          If None, all tags found in the text are projected. Defaults to None.
            max_gap: Maximum allowed gap (in tokens) between aligned tokens. Defaults to 1.
            debugging: If True, includes detailed debug information in the output spans. Defaults to False.

        Returns:
            A tuple containing:
            1. str: The target text reconstructed with XML-like tags around projected spans.
            2. List[Dict[str, Any]]: The list of projected span dictionaries.
        """
        parsed = SpanAligner.tagged_text_to_task(src_text, allowed_tags=allowed_tags)
        raw_src_text = parsed["task"]["data"]["text"]
        src_spans = parsed["spans"] + parsed["entities"]

        projected = self.project_spans(raw_src_text, tgt_text, src_spans, max_gap=max_gap, debugging=debugging)
        projected_valid = [s for s in projected if s is not None]
        tagged_tgt, _ = SpanAligner.rebuild_tagged_text(tgt_text, projected_valid, [])
        
        return tagged_tgt, projected



    # Old methods of projecting based on the original sentence aligner (without top-K candidates or DP refinement). Kept for reference and debugging.
    # Finds the best allignment for each span based on token coverage and clustering, but without the enhanced candidate filtering and sequence-aware DP refinement steps of project_spans_topk.

    def project_spans_single_path(self,
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
        detailed = self.project_spans_single_path_with_debug_info(src_text, tgt_text, src_spans, max_gap)
        if debugging:
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

    def project_spans_single_path_with_debug_info(self,
                                       src_text: str,
                                       tgt_text: str,
                                       src_spans: List[Dict[str, Any]],
                                       max_gap: int) -> List[Dict[str, Any]]:
        
        """Project spans with detailed debug information."""
        if not src_text or not tgt_text or not src_spans:
            return []
            
        src_tokens, src_vectors = self.aligner.get_text_embeddings(src_text)
        tgt_tokens, tgt_vectors = self.aligner.get_text_embeddings(tgt_text)


        projected = []
        renderings = []
        for span in src_spans:
            if self.verbose:
                print(f"{'═'*60}")
                print(f"\n  ┌─ Span: '{span.get('text', '')}'  [{span['start']}:{span['end']}]")
        
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
                    print(f"  └─ Skipped (no token coverage): {e}")
                projected.append(None)
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
                if self.verbose:
                    print(f"  └─ No target tokens mapped.")
                projected.append(None)
                continue

            clusters = self._find_contiguous_clusters(mapped_tgt, max_gap)
            clusters = self._fill_gaps_in_clusters(clusters)
            largest = self._cluster_merge(clusters, src_token_count=len(covered_src), base_gap=max_gap, mass_factor=0.6)

            min_idx, max_idx = min(largest), max(largest)
            tgt_start = result.tgt_tokens[min_idx].start
            tgt_end = result.tgt_tokens[max_idx].end

            if self.verbose:
                topk = self._build_topk_from_similarity(result.similarity_matrix, k=5)
                # Filter to only covered source tokens
                topk_covered = {i: topk[i] for i in covered_src if i in topk}
                paths = self._alignment_to_paths(alignment, topk_covered)
                # print(self.visualize_alignment_text(
                #     topk_covered,
                #     result.src_tokens, result.tgt_tokens,
                #     paths=paths,
                #     #final_indices=largest
                # ))
                renderings.append(self.visualize_alignment_text(
                    topk_covered,
                    result.src_tokens, result.tgt_tokens,
                    paths=paths,
                    #final_indices=largest
                ))
                print(f"  │  Clusters: {clusters}")
                print(f"  └─ Result: '{tgt_text[tgt_start:tgt_end]}'  [{tgt_start}:{tgt_end}]"
                      f"  (token indices {min_idx}..{max_idx})")
            
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
    
    def project_spans_single_path_long(self,
                            src_text: str,
                            tgt_text: str,
                            src_spans: List[Dict[str, Any]],
                            window_size: int = 512,
                            max_gap: int = 1,
                            debugging: bool = False) -> List[Dict[str, Any]]:
        """
        Project spans from source text to target text using windowed alignment
        for long documents that exceed model token limits.

        Each span is projected within a local token window centered around its
        position, then mapped back to full-text coordinates.

        Args:
            src_text: Raw source text (untagged)
            tgt_text: Target text for projection
            src_spans: List of span dicts on source text
            window_size: Token window size for local alignment
            max_gap: Maximum gap in token clustering
            debugging: Whether to print debug visualization

        Returns:
            List of projected span dicts on target text
        """
        if not src_text or not tgt_text or not src_spans:
            return []

        import bisect

        src_tokens = self._get_char_tokens(src_text)
        tgt_tokens = self._get_char_tokens(tgt_text)

        if not src_tokens or not tgt_tokens:
            return []

        src_starts = [t[0] for t in src_tokens]
        half_window = window_size // 2
        projected = []

        for span in src_spans:
            # Locate the span centre in token space
            span_center = (span["start"] + span["end"]) / 2
            center_idx = bisect.bisect_left(src_starts, span_center)
            center_idx = max(0, min(len(src_tokens) - 1, center_idx))

            # Source window
            src_start_idx = max(0, center_idx - half_window)
            src_end_idx = min(len(src_tokens), center_idx + half_window)
            win_src_start_char = src_tokens[src_start_idx][0]
            win_src_end_char = src_tokens[src_end_idx - 1][1]
            src_window_text = src_text[win_src_start_char:win_src_end_char]

            # Corresponding target window (proportional position mapping)
            rel_pos = center_idx / max(len(src_tokens) - 1, 1)
            tgt_center_idx = int(rel_pos * len(tgt_tokens))
            tgt_start_idx = max(0, tgt_center_idx - half_window)
            tgt_end_idx = min(len(tgt_tokens), tgt_center_idx + half_window)
            win_tgt_start_char = tgt_tokens[tgt_start_idx][0]
            win_tgt_end_char = tgt_tokens[tgt_end_idx - 1][1]
            tgt_window_text = tgt_text[win_tgt_start_char:win_tgt_end_char]

            # Adjust span to window-relative coordinates
            rel_span = span.copy()
            rel_span["start"] -= win_src_start_char
            rel_span["end"] -= win_src_start_char

            # Sanity check: span must fit inside the extracted window
            if rel_span["start"] < 0 or rel_span["end"] > len(src_window_text):
                continue

            # Project within the window
            window_projections = self.project_spans_single_path(
                src_window_text, tgt_window_text, [rel_span],
                max_gap=max_gap, debugging=debugging
            )

            # Map back to full-text coordinates
            for p in window_projections:
                p["start"] += win_tgt_start_char
                p["end"] += win_tgt_start_char
                p["text"] = tgt_text[p["start"]:p["end"]]
                projected.append(p)

        return projected



   # -------------------------------------------------------------------------
    # Visualization helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _alignment_to_paths(
        alignment: List[Tuple[int, int]],
        topk_candidates: Dict[int, List[Tuple[int, float]]],
        score: float = 0.0
    ) -> List[Dict]:
        """
        Convert a flat alignment list to the paths format used by visualizers.

        Each path entry has:
          - 'path': List of (src_idx, rank, tgt_idx) tuples
          - 'score': cumulative score of the path

        Args:
            alignment: List of (src_idx, tgt_idx) pairs.
            topk_candidates: The candidate matrix used to look up ranks.
            score: Overall path score (default 0.0).

        Returns:
            A single-element list containing the path dict.
        """
        path_nodes = []
        for src_idx, tgt_idx in sorted(alignment):
            rank = 0
            if src_idx in topk_candidates:
                for r, (cand_tgt, _) in enumerate(topk_candidates[src_idx]):
                    if cand_tgt == tgt_idx:
                        rank = r
                        break
            path_nodes.append((src_idx, rank, tgt_idx))
        return [{'path': path_nodes, 'score': score}]

    @staticmethod
    def _build_topk_from_similarity(
        similarity_matrix,
        k: int = 5
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Build topk_candidates dict from a similarity matrix.

        Args:
            similarity_matrix: (n_src, n_tgt) array of similarity scores.
            k: Number of top candidates per source token.

        Returns:
            Dict mapping src_idx -> [(tgt_idx, score), ...] sorted by score desc.
        """
        import numpy as np
        topk: Dict[int, List[Tuple[int, float]]] = {}
        for src_idx in range(similarity_matrix.shape[0]):
            row = similarity_matrix[src_idx]
            top_indices = np.argsort(row)[::-1][:k]
            topk[src_idx] = [(int(idx), float(row[idx])) for idx in top_indices]
        return topk

    @staticmethod
    def _build_cell_paths(
        topk_candidates: Dict[int, List[Tuple[int, float]]],
        paths: List[Dict],
        top_n_paths: int
    ) -> Tuple[Dict[Tuple[int, int], List[int]], List[int], int]:
        """
        Shared logic for building the (src_idx, rank) -> path-indices mapping.

        Returns:
            cell_paths: Maps (src_idx, rank) to list of path indices containing that cell.
            src_indices: Sorted list of source indices in topk_candidates.
            max_k: Maximum number of candidates across all source positions.
        """
        src_indices = sorted(topk_candidates.keys())
        max_k = max((len(c) for c in topk_candidates.values()), default=0)

        cell_paths: Dict[Tuple[int, int], List[int]] = {}
        for path_idx, p in enumerate(paths[:top_n_paths]):
            for node in p['path']:
                src_i, rank, tgt_idx = node
                cell_paths.setdefault((src_i, rank), []).append(path_idx)

        return cell_paths, src_indices, max_k

    def visualize_alignment_text(
        self,
        topk_candidates: Dict[int, List[Tuple[int, float]]],
        src_tokens: List[TokenInfo],
        tgt_tokens: List[TokenInfo],
        paths: Optional[List[Dict]] = None,
        final_indices: Optional[List[int]] = None,
        top_n_paths: int = 5,
        label: str = ""
    ) -> str:
        """
        Text-based Rank x Src alignment matrix for console / verbose output.

        Layout mirrors ``visualize_alignment_html``:
        rows = ranks, columns = source token indices.

        Markers:
          P0..Pn  — the path that selected this cell
          #       — target token is in the final projected cluster

        Args:
            topk_candidates: Candidate matrix {src_idx: [(tgt_idx, score), ...]}.
            src_tokens: Source token list (for header labels).
            tgt_tokens: Target token list (for cell labels).
            paths: Path dicts with 'path' and 'score' keys. If None, no path markers.
            final_indices: Target token indices in the final cluster.
            top_n_paths: Max paths to display.
            label: Optional title line.
        """
        if not topk_candidates:
            return "  (no candidates)"

        paths = paths or []
        cell_paths, src_indices, max_k = self._build_cell_paths(
            topk_candidates, paths, top_n_paths
        )
        final_set = set(final_indices) if final_indices else set()

        col_w = 22
        idx_col_w = 6
        sep = "─" * (idx_col_w + 2 + col_w * len(src_indices))

        lines = []
        if label:
            lines.append(f"  {label}")

        # Header row: Rank \ Src | src_idx (token)
        hdr = f"  {'Rank':<{idx_col_w}}|"
        for i in src_indices:
            tok = src_tokens[i].text[:10] if i < len(src_tokens) else "?"
            hdr += f"  {f'{i}({tok})':<{col_w - 2}}"
        lines.append(sep)
        lines.append(hdr)
        lines.append(sep)

        # One row per rank
        for rank in range(max_k):
            row = f"  {rank:<{idx_col_w}}|"
            for i in src_indices:
                cands = topk_candidates[i]
                if rank < len(cands):
                    tgt_idx, sim = cands[rank]
                    tgt_tok = tgt_tokens[tgt_idx].text[:8] if tgt_idx < len(tgt_tokens) else "?"
                    p_indices = cell_paths.get((i, rank), [])
                    path_mark = ",".join(f"P{p}" for p in p_indices) if p_indices else ""
                    final_mark = "#" if tgt_idx in final_set else ""
                    cell = f"{tgt_idx}:{tgt_tok}({sim:.2f}){path_mark}{final_mark}"
                    row += f"  {cell:<{col_w - 2}}"
                else:
                    row += f"  {'·':<{col_w - 2}}"
            lines.append(row)

        lines.append(sep)

        # Path summary (compact)
        for path_idx, p in enumerate(paths[:top_n_paths]):
            parts = []
            last_src = -1
            for node in p['path']:
                src_i, rank, tgt_idx = node
                if last_src != -1 and src_i > last_src + 1:
                    parts.append(f"[GAP {src_i - last_src - 1}]")
                parts.append(f"{tgt_idx}(s:{src_i},r:{rank})")
                last_src = src_i
            lines.append(f"  P{path_idx} (score={p['score']:.2f}): {' -> '.join(parts)}")

        legend_parts = []
        if paths:
            legend_parts.append("Pn = path n")
        if final_set:
            legend_parts.append("# = in final cluster")
        if legend_parts:
            lines.append(f"  ({' | '.join(legend_parts)})")

        return "\n".join(lines)

    def visualize_alignment_html(
        self,
        topk_candidates: Dict[int, List[Tuple[int, float]]],
        src_tokens: List[TokenInfo],
        tgt_tokens: List[TokenInfo],
        paths: Optional[List[Dict]] = None,
        final_indices: Optional[List[int]] = None,
        top_n_paths: int = 5,
        label: str = ""
    ) -> str:
        """
        HTML Rank x Src alignment matrix for Jupyter notebooks.

        Renders the same data as ``visualize_alignment_text`` but as a styled
        HTML table with colour-coded path highlights.

        Args:
            topk_candidates: Candidate matrix {src_idx: [(tgt_idx, score), ...]}.
            src_tokens: Source token list (for header labels).
            tgt_tokens: Target token list (for cell labels).
            paths: Path dicts with 'path' and 'score' keys.
            final_indices: Target token indices in the final cluster.
            top_n_paths: Max paths to render.
            label: Optional title line.
        """
        paths = paths or []
        src_indices = sorted(topk_candidates.keys())
        if not src_indices:
            return "No data"

        max_k = max(len(cands) for cands in topk_candidates.values())
        colors = ['#ffb3ba', '#baffc9', '#bae1ff', '#ffffba', '#ffb3ff', '#e6b3ff', '#b3ffe6']
        final_set = set(final_indices) if final_indices else set()

        # Map (src_idx, rank) -> list of path indices
        cell_paths: Dict[Tuple[int, int], List[int]] = {}
        for path_idx, p in enumerate(paths[:top_n_paths]):
            for node in p['path']:
                src_i, rank, tgt_idx = node
                cell_paths.setdefault((src_i, rank), []).append(path_idx)

        # --- Table ---
        html = ""
        if label:
            html += f"<div style='font-family:sans-serif;font-weight:bold;margin-bottom:6px;'>{label}</div>"
        html += "<table border='1' style='border-collapse:collapse;text-align:center;font-family:sans-serif;'>"
        html += "<tr><th style='padding:5px;'>Rank \\ Src Idx</th>"
        for i in src_indices:
            tok = src_tokens[i].text[:10] if i < len(src_tokens) else "?"
            html += f"<th style='padding:5px;'>{i}<br><small>{tok}</small></th>"
        html += "</tr>"

        for rank in range(max_k):
            html += f"<tr><td style='padding:5px;'><b>{rank}</b></td>"
            for i in src_indices:
                cands = topk_candidates[i]
                if rank < len(cands):
                    tgt_idx, sim = cands[rank]
                    tgt_tok = tgt_tokens[tgt_idx].text[:10] if tgt_idx < len(tgt_tokens) else "?"
                    p_indices = cell_paths.get((i, rank), [])
                    in_final = tgt_idx in final_set
                    final_border = "border:3px double #d4af37;" if in_final else ""

                    if p_indices:
                        bg_color = colors[p_indices[0] % len(colors)]
                        path_labels = ",".join([f"P{p}" for p in p_indices])
                        cell_text = f"<span style='font-size: 1.2em;'>{tgt_idx}</span><br>{tgt_tok}<br><small>({sim:.2f})</small><br><b>[{path_labels}]</b>"
                        html += f"<td style='background-color: {bg_color}; color: black; padding: 5px; border: 2px solid #333; {final_border}'>{cell_text}</td>"
                    else:
                        cell_text = f"{tgt_idx}<br>{tgt_tok}<br><small style='color: #666;'>({sim:.2f})</small>"
                        style = f"padding: 5px; color: #555; {final_border}"
                        html += f"<td style='{style}'>{cell_text}</td>"
                else:
                    html += "<td style='background-color:#f9f9f9;'></td>"
            html += "</tr>"
        html += "</table>"

        # --- Path legend ---
        if paths:
            html += "<div style='margin-top:20px;font-family:sans-serif;'><h3>Top Paths:</h3>"
            html += "<ul style='list-style-type:none;padding:0;'>"
            for path_idx, p in enumerate(paths[:top_n_paths]):
                color = colors[path_idx % len(colors)]
                score = p['score']
                parts = []
                last_src = -1
                for node in p['path']:
                    src_i, rank, tgt_idx = node
                    if last_src != -1 and src_i > last_src + 1:
                        parts.append(f"<span style='color:red;'>[GAP {src_i - last_src - 1}]</span>")
                    parts.append(f"<b>{tgt_idx}</b>(s:{src_i},r:{rank})")
                    last_src = src_i
                path_str = " &rarr; ".join(parts)
                html += (f"<li style='margin-bottom:10px;'>"
                         f"<span style='background-color:{color};padding:2px 8px;border-radius:4px;"
                         f"font-weight:bold;border:1px solid #333;'>P{path_idx}</span> "
                         f"<b>Score: {score:.2f}</b> <br> {path_str}</li>")
            html += "</ul></div>"

        # --- Final cluster legend ---
        if final_set:
            html += ("<div style='margin-top:6px;font-family:sans-serif;font-size:0.9em;'>"
                     "&#9670; Gold double-border = in final cluster</div>")

        return html


