# TaxPilot Search Component

The search component handles text processing, segmentation, and vector embedding for legal texts to enable semantic search functionality.

## Modules

### 1. Segmentation

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

### 2. Embeddings

The embeddings module provides functionality to generate vector embeddings for legal text segments:

1. Loads German language models appropriate for legal text
2. Generates embeddings efficiently with batching
3. Stores embeddings with metadata for retrieval
4. Optimizes for Modal.com serverless environment with GPU acceleration

#### Embedding Models

The module supports several embedding models optimized for German legal text:

- **German BERT** (`deepset/gbert-base`): General-purpose German language model
- **German BERT Large** (`deepset/gbert-large`): Larger model with more parameters
- **Multilingual E5** (`intfloat/multilingual-e5-large`): Multilingual model with strong retrieval performance
- **Legal BERT** (`nlpaueb/legal-bert-base-uncased`): English legal domain model (transferable to German)

#### Usage

```python
from taxpilot.backend.search.embeddings import (
    EmbeddingConfig, TextEmbedder, EmbeddingProcessor
)
from taxpilot.backend.data_processing.database import DbConfig

# Configure the embedding process
config = EmbeddingConfig(
    model_name="deepset/gbert-base",
    batch_size=32,
    use_gpu=True
)

# Create an embedder for direct embedding
embedder = TextEmbedder(config)

# Generate embeddings for text
text = "§ 1 Persönlicher Anwendungsbereich..."
embedding = embedder.embed_text(text)
print(f"Embedding shape: {embedding.shape}")

# Process and store embeddings in the database
db_config = DbConfig(db_path="taxpilot.db")
embedding_config = EmbeddingConfig(db_config=db_config)
processor = EmbeddingProcessor(embedding_config)

# Process segments and store embeddings
embedding_ids = processor.process_segments(segments)
print(f"Generated and stored {len(embedding_ids)} embeddings")
```

## Modal.com Integration

TaxPilot uses Modal.com for serverless deployment of embedding generation:

```python
from taxpilot.infrastructure.embedding_config import app

# Generate embeddings for laws that need them
with modal.run():
    # Get laws needing embeddings
    laws = app.get_laws_needing_embeddings.remote()
    
    if laws:
        # Process laws in batches
        for batch in [laws[i:i+10] for i in range(0, len(laws), 10)]:
            app.generate_embeddings_batch.remote(batch)
```

## Design Decisions

### Text Segmentation

1. **Semantic Boundaries**: The segmentation respects natural semantic boundaries in the legal text, such as paragraphs and sections, which helps maintain context.

2. **German-specific Handling**: The module includes specialized handling for German legal text characteristics, including section references (§), paragraph numbering, and abbreviations.

3. **Overlapping Strategy**: Overlapping between segments ensures that content near segment boundaries isn't lost in embedding space, improving recall for searches that span these boundaries.

### Embedding Generation

1. **German Language Models**: The project uses German-specific embedding models like deepset/gbert-base that understand the nuances of German language, which is essential for legal text.

2. **GPU Acceleration**: The Modal.com configuration requests GPU acceleration when available, significantly speeding up embedding generation for large corpora.

3. **Caching and Batching**: Model loading is cached to avoid redundant initialization, and text processing is batched for efficiency.

4. **Incremental Updates**: The system tracks which laws already have embeddings, allowing for incremental updates when laws change without reprocessing the entire corpus.

## Why German BERT?

The default embedding model (`deepset/gbert-base`) was chosen for several reasons:

1. **Native German Understanding**: It was trained specifically on German text, understanding German grammar, compound words, and linguistic structures
2. **Legal Terminology Support**: The training corpus included German Wikipedia and news, which contain significant legal terminology
3. **Balance of Quality and Performance**: The base version provides good embeddings while being computationally efficient
4. **Semantic Similarity Capability**: It effectively captures semantic relationships between legal concepts and terms
5. **Common in German NLP**: Well-tested for German language tasks with established performance metrics

For premium performance, the larger model (`deepset/gbert-large`) can be used, which provides better semantic understanding at the cost of more computation time.