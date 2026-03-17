# coding=utf-8

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional
import re
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
try:
    import networkx as nx
    from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
except ImportError:
    nx = None
import torch

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TokenInfo:
    """Token with text and character offsets."""
    text: str
    start: int
    end: int
    idx: int


@dataclass 
class AlignmentResult:
    """Complete alignment result with all necessary data for projection."""
    alignments: Dict[str, List[Tuple[int, int]]]  # method -> [(src_idx, tgt_idx), ...]
    src_tokens: List[TokenInfo]
    tgt_tokens: List[TokenInfo]
    similarity_matrix: np.ndarray
    src_vectors: np.ndarray
    tgt_vectors: np.ndarray
    topk_candidates: Optional[Dict[int, List[Tuple[int, float]]]] = None


# =============================================================================
# EMBEDDING PROVIDERS - Pluggable embedding backends
# =============================================================================

class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    # Default word tokenization pattern
    WORD_PATTERN = re.compile(r'\b\w+\b|[^\s\w]')
    
    def tokenize_text(self, text: str) -> List[TokenInfo]:
        """Tokenize text into tokens with character offsets.
        
        This is the canonical tokenization used throughout the pipeline.
        """
        tokens = []
        for match in self.WORD_PATTERN.finditer(text):
            tokens.append(TokenInfo(
                text=match.group(),
                start=match.start(),
                end=match.end(),
                idx=len(tokens)
            ))
        return tokens
    
    @abstractmethod
    def get_embeddings(self, tokens: List[str]) -> np.ndarray:
        """Get embeddings for a list of tokens."""
        pass
    
    @abstractmethod
    def get_subword_to_word_map(self, words: List[str]) -> Tuple[List[str], List[int]]:
        """Get subword tokens and mapping to original word indices."""
        pass


class TransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using HuggingFace transformers."""
    
    _model_cache = {}
    
    MODEL_CONFIGS = {
        'bert': 'bert-base-multilingual-cased',
        'xlmr': 'xlm-roberta-base',
        'xlmr-large': 'xlm-roberta-large',
    }
    
    def __init__(self, model: str = "bert", device: str = "cpu", layer: int = 8):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        
        self.device = torch.device(device)
        self.layer = layer
        
        model_name = self.MODEL_CONFIGS.get(model, model)
        self.model_name = model_name
        
        cache_key = f"transformer_{model_name}_{device}"
        if cache_key not in self._model_cache:
            config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
            emb_model = AutoModel.from_pretrained(model_name, config=config)
            emb_model.eval()
            emb_model.to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model_cache[cache_key] = (emb_model, tokenizer)
        
        self.emb_model, self.tokenizer = self._model_cache[cache_key]
    
    def get_subword_to_word_map(self, words: List[str]) -> Tuple[List[str], List[int]]:
        subwords = []
        word_map = []
        for i, word in enumerate(words):
            tokens = self.tokenizer.tokenize(word)
            subwords.extend(tokens)
            word_map.extend([i] * len(tokens))
        return subwords, word_map
    
    def get_embeddings(self, tokens: List[str]) -> np.ndarray:
        """Get embeddings for a list of tokens, handling long sequences by chunking."""
        if not tokens:
            return np.empty((0, 0))
            
        max_length = self.tokenizer.model_max_length
        if max_length > 100000: # Some tokenizers report very large max lengths
            max_length = 512
            
        # We need to leave room for special tokens (e.g., [CLS], [SEP])
        max_subwords = max_length - 2
        
        # First, get subwords to know where to chunk
        subwords = []
        word_to_subword_counts = []
        for word in tokens:
            word_subwords = self.tokenizer.tokenize(word)
            subwords.extend(word_subwords)
            word_to_subword_counts.append(len(word_subwords))
            
        if len(subwords) <= max_subwords:
            # Fast path for short sequences
            with torch.no_grad():
                inputs = self.tokenizer(tokens, is_split_into_words=True, 
                                       padding=True, truncation=True, return_tensors="pt")
                hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
                if self.layer >= len(hidden):
                    raise ValueError(f"Layer {self.layer} requested but model has only {len(hidden)} layers.")
                outputs = hidden[self.layer][:, 1:-1, :]
                return outputs.cpu().numpy()[0]
                
        # Slow path for long sequences: chunking
        all_outputs = []
        current_chunk_words = []
        current_subword_count = 0
        
        for word, count in zip(tokens, word_to_subword_counts):
            if current_subword_count + count > max_subwords and current_chunk_words:
                # Process current chunk
                with torch.no_grad():
                    inputs = self.tokenizer(current_chunk_words, is_split_into_words=True, 
                                           padding=True, truncation=True, return_tensors="pt")
                    hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
                    outputs = hidden[self.layer][:, 1:-1, :]
                    all_outputs.append(outputs.cpu().numpy()[0])
                
                # Reset for next chunk
                current_chunk_words = [word]
                current_subword_count = count
            else:
                current_chunk_words.append(word)
                current_subword_count += count
                
        # Process final chunk
        if current_chunk_words:
            with torch.no_grad():
                inputs = self.tokenizer(current_chunk_words, is_split_into_words=True, 
                                       padding=True, truncation=True, return_tensors="pt")
                hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
                outputs = hidden[self.layer][:, 1:-1, :]
                all_outputs.append(outputs.cpu().numpy()[0])
                
        return np.concatenate(all_outputs, axis=0)
    
    def get_embeddings_batch(self, batch: List[List[str]]) -> List[np.ndarray]:
        """Get embeddings for a batch of token lists, handling long sequences."""
        # For simplicity and robustness with long texts, process individually
        # since different items in the batch might need different chunking
        return [self.get_embeddings(tokens) for tokens in batch]


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Ollama API (new /api/embed endpoint)."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required for Ollama provider")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self._requests = requests

    def get_subword_to_word_map(self, words: List[str]) -> Tuple[List[str], List[int]]:
        return words, list(range(len(words)))

    def get_embeddings(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.empty((0, 0))

        # Ollama has a context window limit (often 2048 or 8192 tokens depending on the model)
        # We chunk the input to be safe. 500 words is a safe conservative chunk size.
        chunk_size = 500
        all_embeddings = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            response = self._requests.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": chunk,
                },
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()
            all_embeddings.extend(data["embeddings"])

        return np.array(all_embeddings, dtype=np.float32)



class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers."""
    
    _model_cache = {}
    
    def __init__(self, model: str = "paraphrase-multilingual-MiniLM-L12-v2", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers package required")
        
        self.device = device
        cache_key = f"sbert_{model}"
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = SentenceTransformer(model, device=device)
        self.model = self._model_cache[cache_key]
    
    def get_subword_to_word_map(self, words: List[str]) -> Tuple[List[str], List[int]]:
        return words, list(range(len(words)))
    
    def get_embeddings(self, tokens: List[str]) -> np.ndarray:
        if not tokens:
            return np.empty((0, 0))
            
        # SentenceTransformers also has a max_seq_length (usually 128, 256, or 512)
        # We chunk the input to be safe.
        chunk_size = 250
        all_embeddings = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk = tokens[i:i + chunk_size]
            chunk_embeddings = self.model.encode(chunk, convert_to_numpy=True)
            all_embeddings.append(chunk_embeddings)
            
        return np.concatenate(all_embeddings, axis=0)


# =============================================================================
# SENTENCE ALIGNER - Core alignment logic
# =============================================================================

class SentenceAligner:
    """Word alignment using contextual embeddings and various matching algorithms."""
    
    MATCHING_METHODS = {"a": "inter", "m": "mwmf", "i": "itermax", "f": "fwd", "r": "rev",
                        "g": "greedy", "t": "threshold"}
    
    def __init__(self, 
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 model: str = "bert",
                 token_type: str = "bpe",
                 distortion: float = 0.0,
                 matching_methods: str = "mai",
                 device: str = "cpu",
                 layer: int = 8):
        """
        Initialize SentenceAligner.
        
        Args:
            embedding_provider: Optional pre-configured EmbeddingProvider instance.
            model: Model name (used if embedding_provider is None)
            token_type: "bpe" for subword alignment, "word" for word-level
            distortion: Position distortion factor (0.0 = no distortion)
            matching_methods: String of method codes
            device: Device for computation
            layer: Transformer layer to extract embeddings from
        """
        self.token_type = token_type
        self.distortion = distortion
        self.matching_methods = [self.MATCHING_METHODS[m] for m in matching_methods if m in self.MATCHING_METHODS]
        
        if embedding_provider is not None:
            self.embed_provider = embedding_provider
        else:
            self.embed_provider = TransformerEmbeddingProvider(model=model, device=device, layer=layer)

    # -------------------------------------------------------------------------
    # Static alignment algorithms
    # -------------------------------------------------------------------------
    
    @staticmethod
    def get_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return (cosine_similarity(X, Y) + 1.0) / 2.0

    @staticmethod
    def apply_distortion(sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        shape = sim_matrix.shape
        if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
            return sim_matrix
        pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])])
        pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])])
        distortion_mask = 1.0 - ((pos_x - np.transpose(pos_y)) ** 2) * ratio
        return np.multiply(sim_matrix, distortion_mask)

    @staticmethod
    def get_alignment_matrix(sim_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]
        return forward, backward.transpose()

    @staticmethod
    def get_max_weight_match(sim: np.ndarray) -> np.ndarray:
        if nx is None:
            raise ValueError("networkx must be installed to use mwmf algorithm.")
        def permute(edge):
            if edge[0] < sim.shape[0]:
                return edge[0], edge[1] - sim.shape[0]
            else:
                return edge[1], edge[0] - sim.shape[0]
        G = from_biadjacency_matrix(csr_matrix(sim))
        matching = nx.max_weight_matching(G, maxcardinality=True)
        matching = [permute(x) for x in matching]
        res_matrix = np.zeros_like(sim)
        for edge in matching:
            res_matrix[edge[0], edge[1]] = 1
        return res_matrix

    @staticmethod
    def iter_max(sim_matrix: np.ndarray, max_count: int = 2) -> np.ndarray:
        alpha_ratio = 0.9
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]
        inter = forward * backward.transpose()

        if min(m, n) <= 2:
            return inter

        count = 1
        while count < max_count:
            mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
            mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
            mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
            mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
            if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
                mask *= 0.0
                mask_zeros *= 0.0

            new_sim = sim_matrix * mask
            fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
            bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
            new_inter = fwd * bac

            if np.array_equal(inter + new_inter, inter):
                break
            inter = inter + new_inter
            count += 1
        return inter

    @staticmethod
    def greedy_match(sim_matrix: np.ndarray, one_to_one: bool = True) -> np.ndarray:
        m, n = sim_matrix.shape
        result = np.zeros((m, n))
        sim_flat = sim_matrix.flatten()
        indices = np.argsort(-sim_flat)
        
        used_src = set() if one_to_one else None
        used_tgt = set() if one_to_one else None
        
        for idx in indices:
            i, j = idx // n, idx % n
            if sim_matrix[i, j] <= 0:
                break
            if one_to_one:
                if i in used_src or j in used_tgt:
                    continue
                used_src.add(i)
                used_tgt.add(j)
            result[i, j] = 1
        return result

    @staticmethod
    def threshold_match(sim_matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (sim_matrix >= threshold).astype(float)

    # -------------------------------------------------------------------------
    # Main alignment methods
    # -------------------------------------------------------------------------

    def _prepare_tokens(self, sent: Union[str, List[str]]) -> Tuple[List[str], List[str], List[int]]:
        """Prepare tokens and get subword mapping."""
        if isinstance(sent, str):
            words = sent.split()
        else:
            words = sent
        subwords, word_map = self.embed_provider.get_subword_to_word_map(words)
        return words, subwords, word_map

    def _compute_alignments(self, sim: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute all requested alignment matrices."""
        all_mats = {}
        all_mats["fwd"], all_mats["rev"] = self.get_alignment_matrix(sim)
        all_mats["inter"] = all_mats["fwd"] * all_mats["rev"]
        
        if "mwmf" in self.matching_methods:
            all_mats["mwmf"] = self.get_max_weight_match(sim)
        if "itermax" in self.matching_methods:
            all_mats["itermax"] = self.iter_max(sim)
        if "greedy" in self.matching_methods:
            all_mats["greedy"] = self.greedy_match(sim)
        if "threshold" in self.matching_methods:
            all_mats["threshold"] = self.threshold_match(sim)
        
        return all_mats

    def _average_embeds_over_words(self, 
                                    bpe_vectors: List[np.ndarray],
                                    words: List[List[str]],
                                    subwords: List[List[str]]) -> List[np.ndarray]:
        """Average subword embeddings to get word-level embeddings."""
        new_vectors = []
        for lang_idx in range(2):
            word_list = words[lang_idx]
            _, word_map = self.embed_provider.get_subword_to_word_map(word_list)
            
            word_vectors = []
            for word_idx in range(len(word_list)):
                subword_indices = [i for i, w in enumerate(word_map) if w == word_idx]
                if subword_indices and max(subword_indices) < bpe_vectors[lang_idx].shape[0]:
                    word_vectors.append(bpe_vectors[lang_idx][subword_indices].mean(0))
                else:
                    word_vectors.append(np.zeros(bpe_vectors[lang_idx].shape[1]))
            
            new_vectors.append(np.array(word_vectors) if word_vectors else np.zeros((0, bpe_vectors[lang_idx].shape[1])))
        return new_vectors

    def get_text_embeddings(self, text: str) -> Tuple[List[TokenInfo], np.ndarray]:
        """
        Get embeddings for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (tokens, embeddings)
        """
        tokens = self.embed_provider.tokenize_text(text)
        words = [t.text for t in tokens]
        
        # Get subword mappings
        subwords, word_map = self.embed_provider.get_subword_to_word_map(words)
        
        # Get embeddings
        if hasattr(self.embed_provider, 'get_embeddings_batch'):
            vectors = self.embed_provider.get_embeddings_batch([words])[0]
        else:
            vectors = self.embed_provider.get_embeddings(words)
        
        # Truncate to subword lengths
        vectors = vectors[:len(subwords)]
        
        # Average over words if needed
        if self.token_type == "word":
            vectors = self._average_embeds_over_words(
                [vectors, vectors], # Hack: pass twice to satisfy list expectation
                [words, words],
                [subwords, subwords]
            )[0]
            
        return tokens, vectors


    # Helper method to get embeddings and similarity
    def _get_embeddings_and_similarity(self, src_words: List[str], tgt_words: List[str], 
                                       compute_sim: bool = True) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Get embeddings for source and target words, optionally compute similarity."""
        # Get subword mappings
        src_subwords, _ = self.embed_provider.get_subword_to_word_map(src_words)
        tgt_subwords, _ = self.embed_provider.get_subword_to_word_map(tgt_words)
        
        # Get embeddings
        if hasattr(self.embed_provider, 'get_embeddings_batch'):
            vectors = self.embed_provider.get_embeddings_batch([src_words, tgt_words])
        else:
            vectors = [
                self.embed_provider.get_embeddings(src_words),
                self.embed_provider.get_embeddings(tgt_words)
            ]
        
        # Truncate to subword lengths
        vectors = [vectors[0][:len(src_subwords)], vectors[1][:len(tgt_subwords)]]
        
        # Average over words if needed
        if self.token_type == "word":
            vectors = self._average_embeds_over_words(
                vectors, [src_words, tgt_words], [src_subwords, tgt_subwords]
            )
        
        # Compute similarity if requested
        sim = None
        if compute_sim:
            sim = self.get_similarity(vectors[0], vectors[1])
            sim = self.apply_distortion(sim, self.distortion)
        
        return vectors, sim

    def _get_subword_range(self, word_start: int, word_end: int, 
                          w2b: List[int]) -> Tuple[int, int]:
        """Convert word range to subword range."""
        if self.token_type == "bpe":
            sub_start = next((i for i, w in enumerate(w2b) if w >= word_start), 0)
            sub_end = next((i for i, w in enumerate(w2b) if w >= word_end), len(w2b))
            return sub_start, sub_end
        return word_start, word_end

    def _get_topk_alignments(self, sim: np.ndarray, src_w2b: List[int], tgt_w2b: List[int],
                             n_src_word: int, n_tgt_word: int,
                             top_k: int = 5, threshold: float = 0.5) -> Dict[int, List[Tuple[int, float]]]:
        """
        Convert similarity matrix to top K alignment candidates per source token.
        Returns a dict mapping src_idx to a list of (tgt_idx, score) tuples.
        
        Guarantees at least 1 candidate per source token (the best match),
        even if it falls below the threshold. This prevents the DP from
        having dead spots where source tokens are unaligned.
        """
        word_sim = np.zeros((n_src_word, n_tgt_word))
        if self.token_type == "bpe":
            for i in range(min(sim.shape[0], len(src_w2b))):
                for j in range(min(sim.shape[1], len(tgt_w2b))):
                    src_idx = src_w2b[i] if i < len(src_w2b) else i
                    tgt_idx = tgt_w2b[j] if j < len(tgt_w2b) else j
                    if src_idx < n_src_word and tgt_idx < n_tgt_word:
                        word_sim[src_idx, tgt_idx] = max(word_sim[src_idx, tgt_idx], sim[i, j])
        else:
            word_sim[:sim.shape[0], :sim.shape[1]] = sim
            
        candidates = {}
        for i in range(n_src_word):
            src_candidates = []
            best_j, best_score = -1, -1.0
            for j in range(n_tgt_word):
                score = word_sim[i, j]
                if score >= threshold:
                    src_candidates.append((j, float(score)))
                # Track global best regardless of threshold
                if score > best_score:
                    best_score = score
                    best_j = j
            src_candidates.sort(key=lambda x: x[1], reverse=True)
            src_candidates = src_candidates[:top_k]
            
            # Guarantee at least 1 candidate: include the best match even if below threshold
            if not src_candidates and best_j >= 0:
                src_candidates = [(best_j, float(best_score))]
                
            candidates[i] = src_candidates
            
        return candidates

    def _get_alignments_from_similarity(self, sim: np.ndarray, vectors: List[np.ndarray],
                                        src_w2b: List[int], tgt_w2b: List[int],
                                        n_src_sub: int, n_tgt_sub: int,
                                        n_src_word: int, n_tgt_word: int,
                                        tgt_offset: int = 0) -> Dict[str, List]:
        """
        Convert similarity matrix to alignment tuples.
        
        Args:
            sim: Similarity matrix
            vectors: Source and target embedding vectors
            src_w2b: Source subword-to-word mapping
            tgt_w2b: Target subword-to-word mapping
            n_src_sub: Number of source subwords
            n_tgt_sub: Number of target subwords
            n_src_word: Number of source words
            n_tgt_word: Number of target words
            tgt_offset: Offset for target indices (for partial alignment)
            
        Returns:
            Dictionary mapping method names to sorted alignment lists
        """
        all_mats = self._compute_alignments(sim)
        logger.debug(f"Computed alignment matrices for methods: {list(all_mats.keys())}")
        aligns = {method: set() for method in self.matching_methods}
        
        n_src = n_src_sub if self.token_type == "bpe" else n_src_word
        n_tgt = n_tgt_sub if self.token_type == "bpe" else n_tgt_word
        
        for i in range(min(vectors[0].shape[0], n_src)):
            for j in range(min(vectors[1].shape[0], n_tgt)):
                actual_tgt_idx = j + tgt_offset
                
                for method in self.matching_methods:
                    if method in all_mats and all_mats[method][i, j] > 0:
                        if self.token_type == "bpe":
                            src_idx = src_w2b[i] if i < len(src_w2b) else i
                            tgt_idx = tgt_w2b[actual_tgt_idx] if actual_tgt_idx < len(tgt_w2b) else actual_tgt_idx
                        else:
                            src_idx = i
                            tgt_idx = actual_tgt_idx
                        
                        aligns[method].add((src_idx, tgt_idx))
        
        # Convert sets to sorted lists
        return {method: sorted(align_set) for method, align_set in aligns.items()}


    # Alignment functions
    def get_word_aligns(self, 
                        src_sent: Union[str, List[str]], 
                        trg_sent: Union[str, List[str]]) -> Tuple[Dict[str, List], List[np.ndarray], np.ndarray]:
        """
        Get word alignments between source and target sentences.
        Legacy API - for backwards compatibility.
        """
        src_words, src_subwords, src_w2b = self._prepare_tokens(src_sent)
        tgt_words, tgt_subwords, tgt_w2b = self._prepare_tokens(trg_sent)
        
        vectors, sim = self._get_embeddings_and_similarity(src_words, tgt_words)
        
        aligns = self._get_alignments_from_similarity(sim, vectors, src_w2b, tgt_w2b,
                                                       len(src_subwords), len(tgt_subwords),
                                                       len(src_words), len(tgt_words))
        
        return aligns, vectors, sim


   
   # Main alignment entry points
    def align_texts(self, src_text: str, tgt_text: str) -> AlignmentResult:
        """
        Align two texts and return complete alignment result.
        
        This is the main entry point that handles tokenization, embedding, and alignment.
        All tokenization is done by the embedding provider for consistency.
        
        Args:
            src_text: Source text string
            tgt_text: Target text string
            
        Returns:
            AlignmentResult with alignments, tokens, and similarity matrix
        """
        logger.debug("Computing alignments...")
        result = self.align_texts_partial(src_text, tgt_text, src_char_start=0, src_char_end=None)
        logger.debug("Alignments computed.")
        return result

    def align_texts_partial(self, src_text: str, tgt_text: str,
                           src_char_start: int = 0,
                           src_char_end: Optional[int] = None,
                           top_k: int = 5) -> AlignmentResult:
        """
        Align two texts with partial source range defined by character positions.
        
        Args:
            src_text: Source text string
            tgt_text: Target text string
            src_char_start: Start index (char position) in source text
            src_char_end: End index (char position) in source text (None = end of text)
            top_k: Number of top alignments to return
            
        Returns:
            AlignmentResult with alignments for the partial source range.
        """
        # compute embeddings
        src_tokens, src_vectors = self.get_text_embeddings(src_text)
        tgt_tokens, tgt_vectors = self.get_text_embeddings(tgt_text)
        
        return self.align_texts_partial_with_embeddings(
            src_tokens, tgt_tokens, src_vectors, tgt_vectors, src_char_start, src_char_end, top_k=top_k
        )
        
    def align_texts_partial_with_embeddings(self, 
                                            src_tokens: List[TokenInfo],
                                            tgt_tokens: List[TokenInfo],
                                            src_vectors: np.ndarray,
                                            tgt_vectors: np.ndarray,
                                            src_char_start: int,
                                            src_char_end: Optional[int] = None,
                                            top_k: int = 5) -> AlignmentResult:
        """
        Align partial source text to target text using pre-computed embeddings and character positions.
        """
        src_words = [t.text for t in src_tokens]
        tgt_words = [t.text for t in tgt_tokens]
        
        # Handle default end
        if src_char_end is None:
            if src_tokens:
                src_char_end = src_tokens[-1].end
            else:
                src_char_end = 0

            
        # Map char range to token range
        src_start_idx = None
        src_end_idx = None
        
        # Simple mapping
        for i, token in enumerate(src_tokens):
            # Check for overlap
            t_start = token.start
            t_end = token.end
            
            # Start index: first token that ends after char_start
            if src_start_idx is None and t_end > src_char_start:
                src_start_idx = i
            
            # End index: last token that starts before char_end
            if t_start < src_char_end:
                src_end_idx = i + 1
        
        if src_start_idx is None:
            src_start_idx = 0
        if src_end_idx is None:
            src_end_idx = len(src_tokens) 

        # Logic from _align_partial_internal adapted:
        src_subwords, src_w2b = self.embed_provider.get_subword_to_word_map(src_words)
        tgt_subwords, tgt_w2b = self.embed_provider.get_subword_to_word_map(tgt_words)
        
        # Determine source subword range
        src_sub_start, src_sub_end = self._get_subword_range(src_start_idx, src_end_idx, src_w2b)
        
        # Handle out-of-bounds indices
        if src_sub_end > src_vectors.shape[0]:
            if src_sub_start >= src_vectors.shape[0]:
                if len(src_w2b) > 0:
                    logger.warning(
                        f"Source subword start index ({src_sub_start}) is beyond source vectors length ({src_vectors.shape[0]}). "
                        "Cannot align this text span as it has no embeddings (likely truncated by the model)."
                    )
                src_sub_start = src_vectors.shape[0]
                src_sub_end = src_vectors.shape[0]
                src_end_idx = src_start_idx
            else:
                logger.warning(
                    f"Source subword end index ({src_sub_end}) exceeds source vectors length ({src_vectors.shape[0]}). "
                    "Truncating the alignment range to match available embeddings."
                )
                src_sub_end = src_vectors.shape[0]
                # Adjust src_end_idx to match the truncated subwords
                if src_sub_end > src_sub_start:
                    last_available_word_idx = src_w2b[src_sub_end - 1]
                    src_end_idx = min(src_end_idx, last_available_word_idx + 1)
                else:
                    src_end_idx = src_start_idx

        # Slice source vectors
        src_partial_vecs = src_vectors[src_sub_start:src_sub_end]
        src_w2b_partial = [idx - src_start_idx for idx in src_w2b[src_sub_start:src_sub_end]]
        # Compute similarity
        sim_partial = self.get_similarity(src_partial_vecs, tgt_vectors)
        sim_partial = self.apply_distortion(sim_partial, self.distortion)
        # Compute alignments
        aligns = self._get_alignments_from_similarity(
            sim_partial, [src_partial_vecs, tgt_vectors], src_w2b_partial, tgt_w2b,
            len(src_w2b_partial), len(tgt_subwords),
            src_end_idx - src_start_idx, len(tgt_words),
            tgt_offset=0
        )
        
        topk_candidates = self._get_topk_alignments(
            sim_partial, src_w2b_partial, tgt_w2b,
            src_end_idx - src_start_idx, len(tgt_words),
            top_k=top_k, threshold=0.5
        )
        
        partial_src_tokens = src_tokens[src_start_idx:src_end_idx]
        
        return AlignmentResult(
            alignments=aligns,
            src_tokens=partial_src_tokens,
            tgt_tokens=tgt_tokens,
            similarity_matrix=sim_partial,
            src_vectors=src_partial_vecs,
            tgt_vectors=tgt_vectors,
            topk_candidates=topk_candidates
        )

    def align_texts_partial_substring(self, src_text: str, tgt_text: str,
                                      src_substring: str) -> AlignmentResult:
        """
        Align two texts using a substring to define the source range.
        
        This method finds the substring in the source text and aligns only that portion.
        
        Args:
            src_text: Source text string
            tgt_text: Target text string
            src_substring: Substring to find in source text
            
        Returns:
            AlignmentResult with alignments for the substring range
            
        Raises:
            ValueError: If substring is not found in source text
        """
        # Find substring position
        substring_start = src_text.find(src_substring)
        if substring_start == -1:
            raise ValueError(f"Substring '{src_substring}' not found in source text")
        
        substring_end = substring_start + len(src_substring)
        
        # Use align_texts_partial with computed indices
        return self.align_texts_partial(src_text, tgt_text, substring_start, substring_end)

 
    # Debugging and visualization
    def print_alignment(self, alignment_result: AlignmentResult, method: str = "inter") -> str:
        """print alignments in a human-readable format showing the source and target tokens with their indices and the aligned pairs according to the specified method."""
        if method not in alignment_result.alignments:
            return f"Method '{method}' not found in alignment results."
        
        src_tokens = alignment_result.src_tokens
        tgt_tokens = alignment_result.tgt_tokens
        aligns = alignment_result.alignments[method]
        
        for alignment in aligns:
            print(f" {alignment}: {src_tokens[alignment[0]].text} <-> {tgt_tokens[alignment[1]].text}")
