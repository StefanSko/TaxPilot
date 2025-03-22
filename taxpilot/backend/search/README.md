# TaxPilot Search Component

The search component handles text processing, segmentation, and vector embedding for legal texts to enable semantic search functionality.

## Modules

### Segmentation

The segmentation module handles preparing legal text for vector embeddings by:

1. Breaking down legal texts into appropriate chunks for embedding
2. Considering semantic boundaries (paragraphs, sections)
3. Handling overlapping windows for better context preservation
4. Retaining metadata about source location

#### Segmentation Strategies

The module supports three main segmentation strategies:

- **Section-level segmentation**: Keeps entire sections together, best for broad search
- **Paragraph-level segmentation**: Splits text by paragraphs, best for detailed search
- **Sentence-level segmentation**: Groups sentences with context, best for precise matching

#### Usage

```python
from taxpilot.backend.search.segmentation import (
    SegmentationStrategy, SegmentationConfig, segment_text
)

# Configure the segmentation
config = SegmentationConfig(
    strategy=SegmentationStrategy.PARAGRAPH,
    chunk_size=512,
    chunk_overlap=128
)

# Segment the text
segments = segment_text(
    text="Your legal text here...",
    law_id="estg",
    section_id="section_1",
    config=config
)

# Each segment contains the text and metadata
for segment in segments:
    print(f"Segment ID: {segment.segment_id}")
    print(f"Text: {segment.text[:100]}...")
    print(f"Metadata: {segment.metadata}")
    print()
```

#### Optimization

The module provides utilities to optimize chunk sizes for different embedding models:

```python
from taxpilot.backend.search.segmentation import optimize_chunk_size

# For a model with 8k token context
optimal_size = optimize_chunk_size(model_max_tokens=8192)
```

## Design Decisions

### Text Segmentation

1. **Semantic Boundaries**: The segmentation respects natural semantic boundaries in the legal text, such as paragraphs and sections, which helps maintain context.

2. **German-specific Handling**: The module includes specialized handling for German legal text characteristics, including section references (§), paragraph numbering, and abbreviations.

3. **Overlapping Strategy**: Overlapping between segments ensures that content near segment boundaries isn't lost in embedding space, improving recall for searches that span these boundaries.

4. **Adaptive Chunking**: The paragraph segmenter can adaptively call the sentence segmenter for paragraphs that exceed the maximum chunk size.

5. **Chunk Size Optimization**: Segments are sized based on the token limitations of the target embedding model, with functions to optimize this relationship.

### Text Cleaning

The module performs several text normalization and cleaning steps:

1. Unicode normalization
2. OCR error correction for common German legal text issues
3. Standardization of section and paragraph references
4. Special formatting for enumerated lists and indentation

## Future Work

Planned enhancements include:

1. **Embedding generation**: Integration with embedding models for generating vector representations
2. **Optimized vector storage**: Efficient storage and retrieval of embeddings
3. **Semantic search endpoints**: API endpoints for searching laws by meaning
4. **Relevance ranking**: Improved ranking of search results based on semantic similarity