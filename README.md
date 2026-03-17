# span-aligner

[PyPI: span-aligner](https://pypi.org/project/span-aligner/)

## Span Projecting & Alignment

**span-aligner** is a Python utility for aligning and mapping text spans between different versions of a text, and for projecting annotations across languages using semantic alignment. It supports both monolingual (fuzzy/regex) and cross-lingual (embedding-based) span mapping, making it ideal for tasks like annotation transfer, text comparison, and multilingual NLP workflows.

## Features

The library consists of three main classes:

1.  **`SentenceAligner`**: Handles the low-level token alignment between two strings. It computes similarity matrices using embeddings and applies matching algorithms (like Max Weight Matching) to find the best token correspondence.
2.  **`SpanAligner`**: A utility class for converting between annotation formats. It handles the conversion between Label Studio-style JSON annotations and XML-tagged text strings (e.g., `<label>text</label>`).
3.  **`SpanProjector`**: The high-level orchestrator that uses `SentenceAligner` to project spans from source to target. It handles token clustering and span boundary reconstruction.

## Installation

Install:

```bash
pip install span-aligner
```

## Quick Start: Span Projection

The `SpanProjector` is the primary class for projecting annotations. It uses semantic alignment based on transformer embeddings (BERT, XLM-R) to map tokens between sentences.

```python
from span_aligner import SpanProjector
from IPython.display import display, HTML

# Initialize the projector
# model: 'bert' (multilingual cased) or 'xlmr' / 'xlmr-large'
# token_type: 'bpe' (recommended for transformers) or 'word'
projector = SpanProjector(
    src_lang="en", 
    tgt_lang="nl", 
    model="bert", 
    token_type="bpe",
    matching_method="inter"
)

text_src = """This is a long and winded sentence to test the mapping of 'Ghent, Belgium' to the target text in Dutch.
With some additional context to make it more complex and realistic for the span projection process."""

text_tgt = """Dit is een lange en omslachtige zin om de mapping van 'Gent, België' naar de doeltaal in het Nederlands te testen.
Met wat extra context om het complexer en realistischer te maken voor het spanprojectieproces."""

spans = [
    {"start": 59, "end": 73, "text": "Ghent, Belgium", "labels": ["LOC"]},
]

# 1. Project Spans with Visualization (for Jupyter Notebooks)
# Returns both the projected spans and a list of HTML strings visualizing the alignment process
projected_spans, renderings = projector.project_spans_with_renderings(text_src, text_tgt, spans)

print("Projected Spans:")
print(projected_spans)

# In a notebook, use display(HTML(...)) to see the alignment details
if renderings:
    display(HTML(renderings[0]))
```

**Output (Console):**

```python
Projected Spans:
[{'start': 55, 'end': 67, 'text': 'Gent, België', 'labels': ['LOC']}]
```

**Output (Visualization):**

The method `project_spans_with_renderings` returns rich HTML that visualizes the alignment candidates and the chosen path.

<table border='1' style='border-collapse:collapse;text-align:center;font-family:sans-serif;'><tr><th style='padding:5px;'>Rank \ Src Idx</th><th style='padding:5px;'>0<br><small>Ghent</small></th><th style='padding:5px;'>1<br><small>,</small></th><th style='padding:5px;'>2<br><small>Belgium</small></th></tr><tr><td style='padding:5px;'><b>0</b></td><td style='background-color: #ffb3ba; color: black; padding: 5px; border: 2px solid #333; '><span style='font-size: 1.2em;'>12</span><br>Gent<br><small>(0.88)</small><br><b>[P0,P1,P2,P3,P4]</b></td><td style='background-color: #ffb3ba; color: black; padding: 5px; border: 2px solid #333; '><span style='font-size: 1.2em;'>13</span><br>,<br><small>(0.94)</small><br><b>[P0,P1,P2,P3,P4]</b></td><td style='background-color: #ffb3ba; color: black; padding: 5px; border: 2px solid #333; '><span style='font-size: 1.2em;'>14</span><br>België<br><small>(0.95)</small><br><b>[P0]</b></td></tr><tr><td style='padding:5px;'><b>1</b></td><td style='padding: 5px; color: #555; '>14<br>België<br><small style='color: #666;'>(0.85)</small></td><td style='background-color:#f9f9f9;'></td><td style='background-color: #baffc9; color: black; padding: 5px; border: 2px solid #333; '><span style='font-size: 1.2em;'>12</span><br>Gent<br><small>(0.86)</small><br><b>[P1]</b></td></tr><tr><td style='padding:5px;'><b>2</b></td><td style='padding: 5px; color: #555; '>13<br>,<br><small style='color: #666;'>(0.84)</small></td><td style='background-color:#f9f9f9;'></td><td style='background-color: #bae1ff; color: black; padding: 5px; border: 2px solid #333; '><span style='font-size: 1.2em;'>13</span><br>,<br><small>(0.83)</small><br><b>[P2]</b></td></tr><tr><td style='padding:5px;'><b>3</b></td><td style='padding: 5px; color: #555; '>15<br>'<br><small style='color: #666;'>(0.79)</small></td><td style='background-color:#f9f9f9;'></td><td style='background-color: #ffb3ff; color: black; padding: 5px; border: 2px solid #333; '><span style='font-size: 1.2em;'>21</span><br>Nederlands<br><small>(0.79)</small><br><b>[P4]</b></td></tr><tr><td style='padding:5px;'><b>4</b></td><td style='padding: 5px; color: #555; '>9<br>mapping<br><small style='color: #666;'>(0.78)</small></td><td style='background-color:#f9f9f9;'></td><td style='background-color: #ffffba; color: black; padding: 5px; border: 2px solid #333; '><span style='font-size: 1.2em;'>15</span><br>'<br><small>(0.78)</small><br><b>[P3]</b></td></tr></table><div style='margin-top:20px;font-family:sans-serif;'><h3>Top Paths:</h3><ul style='list-style-type:none;padding:0;'><li style='margin-bottom:10px;'><span style='background-color:#ffb3ba;padding:2px 8px;border-radius:4px;font-weight:bold;border:1px solid #333;'>P0</span> <b>Score: 19.19</b> <br> <b>12</b>(s:0,r:0) &rarr; <b>13</b>(s:1,r:0) &rarr; <b>14</b>(s:2,r:0)</li><li style='margin-bottom:10px;'><span style='background-color:#baffc9;padding:2px 8px;border-radius:4px;font-weight:bold;border:1px solid #333;'>P1</span> <b>Score: 18.34</b> <br> <b>12</b>(s:0,r:0) &rarr; <b>13</b>(s:1,r:0) &rarr; <b>12</b>(s:2,r:1)</li><li style='margin-bottom:10px;'><span style='background-color:#bae1ff;padding:2px 8px;border-radius:4px;font-weight:bold;border:1px solid #333;'>P2</span> <b>Score: 16.99</b> <br> <b>12</b>(s:0,r:0) &rarr; <b>13</b>(s:1,r:0) &rarr; <b>13</b>(s:2,r:2)</li><li style='margin-bottom:10px;'><span style='background-color:#ffffba;padding:2px 8px;border-radius:4px;font-weight:bold;border:1px solid #333;'>P3</span> <b>Score: 16.50</b> <br> <b>12</b>(s:0,r:0) &rarr; <b>13</b>(s:1,r:0) &rarr; <b>15</b>(s:2,r:4)</li><li style='margin-bottom:10px;'><span style='background-color:#ffb3ff;padding:2px 8px;border-radius:4px;font-weight:bold;border:1px solid #333;'>P4</span> <b>Score: 10.59</b> <br> <b>12</b>(s:0,r:0) &rarr; <b>13</b>(s:1,r:0) &rarr; <b>21</b>(s:2,r:3)</li></ul></div>

For standard projection without visualization, you can use `projector.project_spans(text_src, text_tgt, spans)`.

## Usage

The package `span_aligner` provides two main classes: `SpanAligner` and `SpanProjector`.

*   **`SpanAligner`**:
    Uses regex and fuzzy search. It is highly efficient but restricted to **monolingual** tasks (same language). It serves as a strong baseline for correcting boundary offsets or mapping annotations between slightly different versions of a text.

*   **`SpanProjector`**:
    Uses **word embeddings** (Transformers) to align tokens semantically. It supports **cross-lingual** projection and handles significant paraphrasing. However, it is computationally more expensive.
    *   *Complexity*: The `mwmf` (Max Weight Matching) algorithm has a complexity of **O(n³)**, meaning execution time increases exponentially with text length. Default `inter` functions much faster. Works excellently for **short, distinct spans**.
    *   *Use Case*: Use when languages differ or when textual differences are too great for fuzzy matching.

## Optimization & Best Practices

To achieve the best results while managing computational cost, follow these guidelines:

### 1. Choose the Right Tool for the Job
If the source and target texts are in the same language, **always start with `SpanAligner`**. It is significantly faster and creates precise splits. Only switch to `SpanProjector` if fuzzy matching fails due to low textual overlap.

### 2. Manage Text Length (Chunking)
The `SpanProjector` (specifically with `mwmf`) struggles with very long sequences.
*   **Split Texts**: Break documents into logical segments (e.g., paragraphs, decisions, list items) before projection.
*   **Project Locally**: Align spans within their corresponding segments rather than projecting a small span against an entire document.

### 3. Select the Appropriate Algorithm
*   **`mwmf`** (Max Weight Matching): The gold standard. Finds the globally optimal alignment but is slow. Use for final, high-quality output on segmented text.
*   **`inter`** (Intersection): Much faster. Works excellently for **short, distinct spans** (e.g., named entities like persons, locations, dates) where context is less critical.
*   **`itermax`**: A balanced heuristic that offers better speed than `mwmf` with comparable quality for many tasks.

### 4. Translation-Assisted Projection (Hybrid Approach)
If direct cross-lingual projection yields subpar results, consider an intermediate translation step to simplify the alignment task:

1.  **Translate Source**: Use an LLM or NMT model to translate the annotated source text (or just the spans) into the target language.
2.  **Align Locally**: Use `SpanAligner` (or `SpanProjector` with `inter`) to map the *translated* spans onto the *actual* target text.

**Tip**: The translation should mimic the vocabulary of the target text as closely as possible.
*   *Workflow*: `annotated_source` + `target_text` → **LLM** → `rough_translated_source` → **SpanAligner** → `final_annotated_target`



### Span Aligner

Utilities for exact and fuzzy span mapping.

#### Get Annotations from Tagged Text

Extract structured spans and entities from a string with inline tags.

```python
from span_aligner import SpanAligner

tagged_input = "<administrative_body>Environmental Committee</administrative_body> discussed the <impact_location>central park</impact_location> renovation on <publication_date>2025-12-15</publication_date>."

ner_map = {
    "administrative_body": "ADMINISTRATIVE BODY",
    "publication_date": "PUBLICATION DATE",
    "impact_location": "PRIMARY LOCATION"
}

span_map ={
    "motivation" : "MOTIVATION"
}

annotations = SpanAligner.get_annotations_from_tagged_text(
    tagged_input,
    ner_map=ner_map,
    span_map=span_map
)

print(annotations["entities"])
# Output:
#[
#    {'start': 0, 'end': 23, 'text': 'Environmental Committee', 'labels': ['ADMINISTRATIVE BODY']},
#    {'start': 38, 'end': 50, 'text': 'central park', 'labels': ['PRIMARY LOCATION']},
#    {'start': 65, 'end': 75, 'text': '2025-12-15', 'labels': ['PUBLICATION DATE']}
#]
```

#### Rebuild Tagged Text

Reconstruct a string with XML-like tags from raw text and span/entity lists.

```python
from span_aligner import SpanAligner

text = "On 2026-01-12, the Budget Committee finalized the annual report."
# Entities corresponding to 'ADMINISTRATIVE BODY' label (indices skip "the ")
entities = [{"start": 19, "end": 35, "labels": ["administrative_body"]}]

tagged, stats = SpanAligner.rebuild_tagged_text(text, entities=entities)
print(tagged)
# Output: On 2026-01-12, the <administrative_body>Budget Committee</administrative_body> finalized the annual report.
```

#### Map Tags to Original

Align annotated spans from a tagged string back to their positions in the original text, allowing for noisy text or translation differences.

```python
from span_aligner import SpanAligner

original_text = "Budget Committee met on 2026-01-12 to view\n\n the central park prject."
tagged_text = "<administrative_body>Budget Committee</administrative_body> met on <publication_date>2026-01-12</publication_date> to review the <impact_location>central park</impact_location> project."

mapped_tagged_text = SpanAligner.map_tags_to_original(
    original_text=original_text,
    tagged_text=tagged_text,
    min_ratio=0.7
)
print(mapped_tagged_text)
# Output preserves original text errors:
# "<administrative_body>Budget Committee</administrative_body> met on <publication_date>2026-01-12</publication_date> to view
#  the <impact_location>central park</impact_location> prject."
```

### Span Projector

Project annotations from one text to another using semantic alignment (e.g., cross-lingual projection).

The process begins by generating embeddings for both source and target texts, creating a similarity matrix, and finding the optimal set of alignment pairs. Several algorithms are implemented for this matching phase, including `mwmf`, `inter`, `itermax`, `fwd`, `rev`, `greedy`, and `threshold`.



#### Project En -> En (Identity/Paraphrase)

Project annotations to a similar text in the same language. Functions similar to the `spanAligner` with improved fuzzy matching.

```python
from span_aligner import SpanProjector

# Initialize projector (uses BERT embeddings by default)
projector = SpanProjector(src_lang="en", tgt_lang="en")

src_text = "The <ent>cat</ent> sat on the mat."
tgt_text = "The cat sat\n\n on th.e mat."

tagged_tgt, spans = projector.project_tagged_text(src_text, tgt_text)
print(tagged_tgt)
# Output: The <ent>cat</ent>\n\n sat on th.e mat.
```

#### Project En -> Nl (Cross-Lingual)

Project annotations from an English source text to a Dutch target translation.

```python
from span_aligner import SpanProjector

# Initialize projector
projector = SpanProjector(src_lang="en", tgt_lang="nl")

src_text = """DECISION LIST <contextual_location>Municipality of Zele</contextual_location>
 <administrative_body>Standing Committee</administrative_body> | <contextual_date>June 28, 2021</contextual_date>
  <title>1. Acceptance of candidacies for the examination procedure coordinator of Welfare</title>
  <decision>Acceptance of candidacies for the examination procedure coordinator of Welfare</decision>
  <title>2. Establishment of valuation rules for the integrated entity Municipality and Public Social Welfare Center (OCMW)</title>
  <decision>Establishment of valuation rules for the integrated entity Municipality and OCMW</decision>"""

tgt_text = """BESLUITENLIJST Gemeente Zele Vast bureau | 28 juni 20211.
 1. Aanvaarden kandidaturen examenprocedure coördinator Welzijn
 Aanvaarden kandidaturen examenprocedure coördinator Welzijn
 2. Vaststelling waarderingsregels geïntegreerde entiteit Gemeente en OCMW
 Vaststelling waarderingsregels geïntegreerde entiteit Gemeente en OCMW"""

tagged_tgt, spans = projector.project_tagged_text(src_text, tgt_text)
print(tagged_tgt)
# Output: BESLUITENLIJST <contextual_location>Gemeente Zele</contextual_location>
# <administrative_body>Vast bureau</administrative_body> <contextual_date>| 28 juni 20211</contextual_date>.
# <title>1. Aanvaarden kandidaturen examenprocedure coördinator Welzijn
# Aanvaarden kandidaturen examenprocedure coördinator</title> Welzijn
# <title>2. Vaststelling waarderingsregels geïntegreerde entiteit Gemeente en OCMW</title>
# <decision>Vaststelling waarderingsregels geïntegreerde entiteit Gemeente en OCMW</decision>

```


## Pluggable Embedding Backends

The alignment and projection utilities support **pluggable embedding backends**. This means you can choose between different embedding providers depending on your requirements, hardware, or available APIs. The embedding backend is responsible for converting tokens or words into vector representations used for alignment.

### Available Embedding Providers

- **HuggingFace Transformers** (default):
    - Use the `TransformerEmbeddingProvider` for BERT, XLM-R, and similar models.
    - Example: `aligner = SentenceAligner(model="bert")`

- **Sentence-Transformers**:
    - Use the `SentenceTransformerProvider` for fast, high-quality sentence or token embeddings.
    - Example: `aligner = SentenceAligner(embedding_provider=SentenceTransformerProvider(model="paraphrase-multilingual-MiniLM-L12-v2"))`

- **Ollama API**:
    - Use the `OllamaEmbeddingProvider` to get embeddings from a local Ollama server (e.g., with `embeddinggemma`).
    - Example: `aligner = SentenceAligner(embedding_provider=OllamaEmbeddingProvider(model="embeddinggemma"))`

You can also implement your own embedding provider by subclassing `EmbeddingProvider` and implementing the required methods.

### How to Use a Custom Embedding Provider

```python
from span_aligner.sentence_aligner import SentenceAligner, EmbeddingProvider

class MyEmbeddingProvider(EmbeddingProvider):
        def get_embeddings(self, tokens):
                # Return a numpy array of embeddings for the tokens
                ...
        def get_subword_to_word_map(self, words):
                # Return (subwords, word_map)
                ...

aligner = SentenceAligner(embedding_provider=MyEmbeddingProvider())
```

### Switching Providers

You can switch embedding providers by passing the desired provider to `SentenceAligner` or `SpanProjector` via the `embedding_provider` argument. If not provided, the default is a HuggingFace transformer model.

---

### Sentence Aligner

Low-level class for aligning tokens between two texts (sentences or paragraphs) using transformer embeddings.  Based on the work of `simalign` but optimized for span mapping (partial alignment instead of full text) and customized for different embedding providers (Ollama, SaaS providers, Transformers, Sentence-Transformers).

#### Initialize Aligner

```python
from span_aligner import SentenceAligner

# Use bert embeddings (default) with BPE tokenization
aligner = SentenceAligner(model="bert", token_type="bpe") 

text_src = "1. Approval of the minutes of the previous meeting"
text_tgt = "1. Goedkeuring notulen van de voorgaande vergadering"
```

#### Get Text Embeddings

Retrieve tokens and embedding vectors for a string.

```python
tokens_src, vecs_src = aligner.get_text_embeddings(text_src)
print(f"Src tokens: {len(tokens_src)}, Vectors: {vecs_src.shape}")
# Output: Src tokens: 10, Vectors: (12, 768)
```

#### Align Partial Substring

Find the alignment of a specific substring from source to target.

```python
# Align "simple test"
res_sub = aligner.align_texts_partial_substring(text_src, text_tgt, "minutes of the previous meeting")
print("==============================")
for src, tgt in res_sub.alignments.get("inter"):
    print(f"Aligned: '{src}' {res_sub.src_tokens[src].text}-> '{tgt}' {res_sub.tgt_tokens[tgt].text}")
# Output:
# ==============================
# Aligned: '0' - 'minutes'-> '3' - 'notulen'
# Aligned: '1' - 'of'-> '4' - 'van'
# Aligned: '2' - 'the'-> '5' - 'de'
# Aligned: '3' - 'previous'-> '6' - 'voorgaande'
# Aligned: '4' - 'meeting'-> '7' - 'vergadering'
```

## Configuration & Advanced Usage

### Embedding Models

The `model` parameter supports common transformer models:

- `"bert"`: `bert-base-multilingual-cased` (Default, robust multilingual performance)
- `"xlmr"`: `xlm-roberta-base` (Strong cross-lingual transfer)
- `"xlmr-large"`: `xlm-roberta-large` (Higher accuracy, more resource intensive)

```python
# Use xlm-roberta-base
projector = SpanProjector(model="xlmr")
```

### Matching Algorithms

The `matching_method` parameter controls how the token similarity matrix is converted into an alignment.

- `"mwmf"` (**Max Weight Matching**): Finds the global optimal independent edge set. Best quality, O(n³) complexity.
- `"inter"` (**Intersection**): Intersection of forward and backward attention. High precision, lower recall, very fast.
- `"itermax"` (**Iterative Max**): Heuristic iterative maximization. Good speed/quality balance.
- `"greedy"` (**Greedy**): Selects best matches greedily. Fast but local optimum.

```python
# Trade accuracy for speed with 'inter'
projector = SpanProjector(matching_method="inter")
```

### Tokenization: BPE vs Word

- `token_type="bpe"` (Recommended): Uses the transformer's subword tokenizer (e.g. WordPiece). Handles rare words better and aligns closer to the model's internal representation.
- `token_type="word"`: Splits by whitespace/punctuation. Simpler, but can result in `[UNK]` tokens for transformers.

