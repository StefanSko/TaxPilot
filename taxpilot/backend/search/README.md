# TaxPilot Search Component

The search component handles text processing, segmentation, and vector embedding for legal texts to enable semantic search functionality.

## Setup

By default, the search pipeline expects a Qdrant vector database server running and accessible at `http://localhost:6333` (configurable via the `QDRANT_URL` environment variable).

The easiest way to run Qdrant is using Docker:
```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### Local Storage Mode

Alternatively, you can configure the pipeline to use local on-disk storage for Qdrant, eliminating the need for a separate server. This is useful for development or simpler deployments.

To enable local mode, provide a path when configuring the `IndexingConfig`:

```python
from pathlib import Path
from taxpilot.backend.search.indexing_pipeline import IndexingConfig

config = IndexingConfig(
    # ... other settings
    qdrant_local_path=Path("./my_qdrant_data") # Set this path
)
```
The pipeline will then create and manage Qdrant data files within the specified directory (`./my_qdrant_data` in this example).

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

The module also supports hierarchical segmentation, which preserves information about the legal document structure:

- **Hierarchical segmentation**: Extracts metadata about article numbers, subsections, and paragraph structure
- **Article-based grouping**: Preserves relationships between segments and their parent articles

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

- **German BERT** (`deepset/gbert-base`): General-purpose German language model (Default)
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

## End-to-End Search Pipeline

The `indexing_pipeline.py` module provides an end-to-end solution for indexing laws and searching them. For complete working examples, see the [examples](./examples) directory.

### Indexing Laws

To index all laws in the database:

```bash
# Index all laws in the database (assumes Qdrant server running)
python -m taxpilot.backend.search.indexing_pipeline --db-path=data/processed/taxpilot.duckdb

# Index with specific configuration (e.g., using local Qdrant)
python -m taxpilot.backend.search.indexing_pipeline \
    --db-path=data/processed/taxpilot.duckdb \
    --segmentation-strategy=paragraph \
    --embedding-model=deepset/gbert-base \
    --use-gpu \
    --qdrant-local-path=./qdrant_data # Add this for local mode

# Index only specific laws
python -m taxpilot.backend.search.indexing_pipeline \
    --db-path=data/processed/taxpilot.duckdb \
    --laws estg ustg_1980
```

### Searching Laws with CLI

The `search_cli.py` module provides a command-line interface for searching indexed laws:

```bash
# Basic search (assumes Qdrant server running or uses local path if indexed with one)
python -m taxpilot.backend.search.search_cli "Einkommensteuer"

# Search in a specific law
python -m taxpilot.backend.search.search_cli "Steuerpflicht" --law estg

# Get more results
python -m taxpilot.backend.search.search_cli "Umsatzsteuer" --limit 10

# Output as JSON for further processing
python -m taxpilot.backend.search.search_cli "Steuerabzug" --json

# Disable highlighting
python -m taxpilot.backend.search.search_cli "Abgabenordnung" --no-highlights

# Use article-based search (group results by article)
python -m taxpilot.backend.search.search_cli "Umsatzsteuer" --group-by-article

# Specify article scoring strategy (max, average, weighted, count_boosted)
python -m taxpilot.backend.search.search_cli "Steuerabzug" --group-by-article --score-strategy weighted
```

### Using the Search API in Code

```python
from pathlib import Path
from taxpilot.backend.data_processing.database import DbConfig
from taxpilot.backend.search.indexing_pipeline import create_search_api, IndexingConfig

# Create search API configuration
# Default: Connects to Qdrant server
config = IndexingConfig(
    db_config=DbConfig(db_path="data/processed/taxpilot.duckdb")
)

# Alternative: Use local Qdrant storage
# config = IndexingConfig(
#     db_config=DbConfig(db_path="data/processed/taxpilot.duckdb"),
#     qdrant_local_path=Path("./qdrant_data") # Use the same path as during indexing
# )

search_api = create_search_api(config)

# Perform search
try:
    # Standard segment-based search
    results = search_api.search(
        query="Steuererklärung",
        filters={"law_id": "estg"},  # Optional filter
        limit=5,
        highlight=True
    )
    
    # Article-based search 
    article_results = search_api.search(
        query="Steuererklärung",
        filters={"law_id": "estg"},
        limit=5,
        highlight=True,
        group_by_article=True  # Enable article-based grouping
    )
    
    # Process segment-based results
    print("Segment-based results:")
    for result in results.results:
        print(f"Section: {result.section_number}")
        print(f"Title: {result.title}")
        print(f"Score: {result.relevance_score}")
        print(f"Content: {result.content[:200]}...")
        print()
        
    # Process article-based results
    print("Article-based results:")
    for result in article_results.results:
        print(f"Article: {result.section_number}")
        print(f"Title: {result.title}")
        print(f"Score: {result.relevance_score}")
        print(f"Matching segments: {result.metadata.get('matching_segments', 0)}")
        print(f"Content: {result.content[:200]}...")
        print()
finally:
    search_api.close()
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

4. **Hierarchical Structure**: The segmentation system preserves hierarchical information about the legal structure (articles, subsections, paragraphs), enabling more meaningful organization of search results.

### Article-based Search

1. **Result Organization**: Search results can be grouped by article rather than showing disconnected text segments, improving readability and context for legal professionals.

2. **Flexible Scoring**: Multiple article scoring strategies are available: 
   - **Max Score**: Uses the highest segment score within an article (default)
   - **Average Score**: Averages all segment scores within an article
   - **Weighted Score**: Weights scores based on segment position within the article
   - **Count Boosted**: Boosts scores based on the number of matching segments

3. **Metadata Enrichment**: Article results include metadata about how many segments matched the query and their position in the document, providing contextual information.

### Embedding Generation

1. **German Language Models**: The project uses German-specific embedding models like `deepset/gbert-base` that understand the nuances of German language, which is essential for legal text.

2. **GPU Acceleration**: The Modal.com configuration requests GPU acceleration when available, significantly speeding up embedding generation for large corpora.

3. **Caching and Batching**: Model loading is cached to avoid redundant initialization, and text processing is batched for efficiency.

4. **Incremental Updates**: The system tracks which laws already have embeddings, allowing for incremental updates when laws change without reprocessing the entire corpus.

## Why German BERT?

The default embedding model (`deepset/gbert-base`) was chosen for several reasons:

1. **Native German Understanding**: It was trained specifically on German text, understanding German grammar, compound words, and linguistic structures.
2. **Legal Terminology Support**: The training corpus included German Wikipedia and news, which contain significant legal terminology.
3. **Balance of Quality and Performance**: The base version provides good embeddings while being computationally efficient.
4. **Semantic Similarity Capability**: It effectively captures semantic relationships between legal concepts and terms.
5. **Common in German NLP**: Well-tested for German language tasks with established performance metrics.

For premium performance, the larger model (`deepset/gbert-large`) can be used, which provides better semantic understanding at the cost of more computation time.