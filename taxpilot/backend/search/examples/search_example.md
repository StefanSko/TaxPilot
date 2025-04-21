# Technical Documentation: TaxPilot Search Example

## 1. Introduction

The TaxPilot search functionality demonstrates end-to-end semantic search for German tax law. It processes raw legal text from a DuckDB database, segments it while preserving hierarchical structure, generates embeddings, stores them in a vector database (Qdrant), and provides both segment-based and article-based search capabilities.

This document details the data flow, component interactions, and both search approaches available in the system.

## 2. Configuration and Initialization

The script begins by setting up the necessary configuration for the entire pipeline.

1.  **Database Path:** It defines the path to the source DuckDB database (`germanlawfinder.duckdb`) containing the laws and sections.
2.  **Available Laws (Check):** It attempts to connect to the DuckDB database using `database.get_connection()` and fetches all available law IDs using `database.get_all_laws()`. This step helps verify the database connection and identify which laws can be indexed.
3.  **`IndexingConfig`:** An `IndexingConfig` object (from `indexing_pipeline.py`) is created. This is the central configuration object that dictates the behavior of the pipeline. Key settings include:
    *   `db_config`: Specifies the DuckDB database path.
    *   `segmentation_strategy`: How text should be chunked (e.g., `PARAGRAPH`).
    *   `embedding_model`: Which transformer model to use (e.g., `deepset/gbert-base`).
    *   `chunk_size`, `chunk_overlap`: Parameters for segmentation.
    *   `use_accelerator`: Whether to attempt using MPS (on Mac) or CUDA (if available).
    *   `laws_to_index`: A list of law IDs to process. If empty (as configured now), all laws found in the database will be indexed.
    *   `qdrant_local_path`: If set to a `Path` object (as it is now: `./qdrant_local_data`), Qdrant runs in local on-disk mode. If `None`, it attempts to connect to a Qdrant server specified by `QDRANT_URL` (defaulting to `http://localhost:6333`).
4.  **`SearchPipeline` Initialization:** An instance of `SearchPipeline` (from `indexing_pipeline.py`) is created using the `IndexingConfig`. During its initialization (`__init__`):
    *   It creates specific configuration objects for sub-components based on `IndexingConfig`:
        *   `SegmentationConfig`
        *   `EmbeddingConfig` (detects device: MPS/CUDA/CPU based on `use_accelerator`)
        *   `VectorDbConfig` (sets `local_path` if provided in `IndexingConfig`)
    *   It initializes the main processing components:
        *   `EmbeddingProcessor`: Handles embedding generation and storage in DuckDB (`section_embeddings` table).
        *   `TextEmbedder`: Used by `EmbeddingProcessor` and later `SearchService` to load models (via `ModelCache`) and generate embeddings.
        *   `VectorDatabaseManager`: Manages interaction with the Qdrant vector store (local or server). Initializes `VectorDatabase`.
        *   `SearchService`: Orchestrates the search functionality.

## 3. Step 1: Indexing Workflow (`pipeline.index_all_laws()`)

This is the core data processing and indexing phase.

1.  **Get Laws:** Retrieves all law records (or filters based on `config.laws_to_index`) from DuckDB using `database.get_all_laws()`.
2.  **Loop Through Laws:** Iterates through each selected law.
3.  **Get Sections:** For the current law, retrieves all its associated sections from DuckDB using `database.get_sections_by_law(law_id)`.
4.  **Accumulate Segments:**
    *   Initializes an empty list `law_segments` to hold all `TextSegment` objects for the current law.
    *   Iterates through each `section` of the law.
    *   If a section has content, it calls `segmentation.segment_text(text=section_content, ..., config=segmentation_config)`.
    *   `segment_text` applies the chosen strategy (e.g., paragraph splitting), cleans the text, and creates multiple `TextSegment` dataclass objects. Each `TextSegment` contains:
        *   `text`: The actual text chunk.
        *   `law_id`, `section_id`: Source identifiers.
        *   `segment_id`: A unique ID for this specific chunk (e.g., `BJNR010050934_1_p1`).
        *   `start_idx`, `end_idx`: Character indices within the original section content.
        *   `metadata`: Dictionary including strategy, length, etc.
    *   The generated `segments` list for the section is appended to the `law_segments` list.
5.  **Process Law Segments:** After processing all sections for the current law, if `law_segments` is not empty:
    *   It calls `embedding_processor.process_segments(law_segments)`.
    *   **Inside `EmbeddingProcessor.process_segments`:**
        *   Calls `self.embedder.embed_segments(law_segments)`.
        *   **Inside `TextEmbedder.embed_segments`:**
            *   Extracts the `text` field from each `TextSegment`.
            *   Calls `self.embed_texts(texts)` which in turn calls `self.encode(texts)`.
            *   **Inside `TextEmbedder.encode`:**
                *   Loads the transformer model and tokenizer using `ModelCache` (which caches loaded models/tokenizers in memory). The `ModelCache` handles device selection (CPU/MPS/CUDA) and potentially `torch_dtype` (like float16 on accelerators).
                *   Tokenizes the batch of texts using the loaded tokenizer (`self.tokenizer(...)`). Padding, truncation, and conversion to PyTorch tensors (`pt`) occur here.
                *   Moves the tokenized input to the target device (e.g., 'mps').
                *   Performs inference: Passes the tokenized input to the model (`self.model(**encoded_input)`) within a `torch.no_grad()` context.
                *   Applies mean pooling (`embeddings.mean_pooling`) to the model output's token embeddings using the attention mask to get sentence-level embeddings.
                *   Optionally normalizes the resulting embeddings (`torch.nn.functional.normalize`).
                *   Moves embeddings back to the CPU and converts them to a NumPy array (`np.vstack`).
            *   Creates a list of `TextEmbedding` objects, pairing each NumPy vector with its corresponding `TextSegment` metadata.
        *   Calls `self._store_embeddings(embeddings)`.
        *   **Inside `EmbeddingProcessor._store_embeddings`:**
            *   Iterates through the `TextEmbedding` objects.
            *   Connects to DuckDB using `database.get_connection()`.
            *   For each embedding:
                *   Generates a unique primary key `embedding_id` for the database row.
                *   Serializes the NumPy vector to bytes (`vector.tobytes()`).
                *   Serializes the metadata dictionary to a JSON string (`json.dumps(metadata)`).
                *   Checks if an embedding for this `segment_id`, `model`, and `version` already exists.
                *   If it exists, `UPDATE`s the `embedding` and `metadata`.
                *   If not, `INSERT`s a new row into the `section_embeddings` table with `id`, `law_id`, `section_id`, `segment_id`, `embedding_model`, `embedding_version`, `embedding` (blob), `metadata` (json).
            *   *(Connection is no longer closed here)*.
    *   **Inside `SearchPipeline.index_all_laws` (Continued):** The IDs returned by `process_segments` are collected for statistics.
6.  **Loop Continuation:** The process repeats for the next law.
7.  **Synchronization & Optimization:** After processing all laws:
    *   `vector_db.synchronize()` is called (currently, this iterates through DuckDB `section_embeddings` and tries to ensure corresponding entries exist in Qdrant - potentially redundant given the current flow).
    *   `vector_db.optimize()` is called, which triggers Qdrant's internal optimization if applicable (mainly for server mode).
8.  **Statistics:** Returns a dictionary containing counts of laws, sections, segments processed, and embeddings generated.

## 4. Step 2: Search Workflow (`pipeline.search()`)

After indexing, the script performs several example searches.

1.  **Search Call:** The `example.py` loop calls `pipeline.search(query=..., law_id=..., limit=...)`.
2.  **`SearchPipeline.search`:** This method primarily delegates to `SearchService.search`. It constructs a `filters` dictionary if `law_id` is provided.
3.  **`SearchService.search`:**
    *   Embeds the `query` text using `self.embedder.embed_text(query)`, which follows the same `encode` logic described in the indexing phase (model loading, tokenization, inference, pooling, normalization).
    *   Creates a `SearchParameters` object containing the `query_vector`, `top_k` (limit), `offset`, `min_score`, etc.
    *   Calls `self.vector_db.search_similar(search_params)`.
    *   **Inside `VectorDatabaseManager.search_similar`:** Delegates to `self.db.search(params)`.
    *   **Inside `VectorDatabase.search`:**
        *   Constructs Qdrant search keyword arguments (`search_kwargs`) including the vector, limit, offset, etc.
        *   *Crucially*, it checks if `self.config.local_path` is `None`. Only if it's running in server mode does it add the `filter` argument to `search_kwargs` (because local mode doesn't support the `filter` parameter in the same way via the client API call).
        *   Calls `self.client.search(**search_kwargs)` to execute the search against the Qdrant collection (local file or remote server). Qdrant performs an approximate nearest neighbor (ANN) search based on the configured distance metric (e.g., cosine).
        *   Receives a list of `ScoredPoint` objects from Qdrant.
        *   Converts each `ScoredPoint` into a `SearchResult` dataclass object, extracting `segment_id`, `law_id`, `section_id`, `score`, and `metadata` from the Qdrant payload.
        *   Returns the list of `SearchResult` objects.
    *   **Back in `SearchService.search`:**
        *   Calls `self._enrich_results(vector_results, query, highlight)` to get full text and apply highlighting.
        *   **Inside `SearchService._enrich_results`:**
            *   Extracts the list of unique `segment_id`s from the `vector_results`.
            *   Calls `self._get_segments_content(segment_ids)`.
            *   **Inside `SearchService._get_segments_content`:**
                *   Gets a DuckDB connection.
                *   Parses `section_id`s from the `segment_id`s.
                *   Executes a SQL query against the `sections` table (joining with `laws`) to get details (full content, title, etc.) for the relevant *sections*. Fetches results and converts them into a dictionary `section_details` mapped by `section_id`.
                *   Executes a second SQL query against the `section_embeddings` table to retrieve the `metadata` JSON for the specific *segments*. Fetches results and converts them into a dictionary `segment_metadata` mapped by `segment_id`.
                *   Iterates through the original `segment_ids`. For each one, it finds the corresponding section details and segment metadata.
                *   It reconstructs the original *segment text* by using the `start_idx` and `end_idx` from the segment's metadata and slicing the full `section_content`.
                *   Builds a dictionary `segments_data` mapping `segment_id` to its details (including the reconstructed segment text, full section text, title, law info, etc.).
                *   Returns `segments_data`.
            *   **Back in `_enrich_results`:** Iterates through the `vector_results` (from Qdrant). For each `result`, it looks up the corresponding entry in the `segments_data` dictionary.
            *   Calls `self._highlight_text(section_content, query)` if highlighting is enabled. This uses simple regex substitution to wrap query terms found in the text with `<mark>` tags.
            *   Creates a `QueryResult` dataclass object containing the enriched data (ID, law info, title, original content, highlighted content, score, metadata).
            *   Returns the list of `QueryResult` objects.
        *   **Back in `SearchService.search`:** Packages the `enriched_results` into a `SearchResults` object, which also includes total count (estimated), pagination info, the original query, and execution time.
        *   Returns the `SearchResults` object.
4.  **Display Results:** The `example.py` script receives the `SearchResults`, iterates through the `results` list (containing `QueryResult` objects), replaces `<mark>` tags with terminal color codes, and prints the formatted results (Section, Title, Score, Snippet).

## 5. Step 3: Statistics (`pipeline.get_index_stats()`)

1.  Calls `pipeline.get_index_stats()`, which delegates to `vector_db.get_stats()`.
2.  **Inside `VectorDatabaseManager.get_stats`:** Delegates to `self.db.get_collection_stats()`.
3.  **Inside `VectorDatabase.get_collection_stats`:**
    *   Calls `self.client.get_collection()` to get basic info from Qdrant.
    *   *Note: The current implementation attempts to scroll through all points to count vectors by model/law, which can be very inefficient for large collections. A better approach might use Qdrant's count API if available or rely on stats provided by `get_collection`.*
    *   Returns a dictionary with statistics like total vectors, dimensions, counts, etc.
4.  The `example.py` script prints selected statistics.

## 6. Cleanup (`pipeline.close()`)

1.  The `finally` block in `example.py` ensures `pipeline.close()` is called.
2.  **Inside `SearchPipeline.close`:**
    *   Calls `self.search_service.close()`.
    *   Calls `self.vector_db.close()`.
    *   Calls the global `database.close_connection()` to finally close the persistent DuckDB connection.

## 7. Article-Based Search

The TaxPilot search system provides an enhanced article-based search approach that groups search results by their parent articles, providing more complete and contextual results compared to disconnected text segments.

### Article Search Workflow

When using article-based search (via the `group_by_article=True` parameter):

1. **Initial Search:** The search process begins like standard segment-based search, generating embeddings for the query and searching the vector database.

2. **Result Aggregation:**
   * After retrieving segment-level results from Qdrant, the `ArticleSearchService` groups these results by their parent article using the `article_id` field stored in the vector database.
   * For each article, multiple matching segments are collected along with their relevance scores.

3. **Article Scoring:**
   The system offers multiple scoring strategies to determine article relevance:
   * **MAX** (default): Uses the highest segment score within each article
   * **AVERAGE**: Averages all segment scores for each article
   * **WEIGHTED**: Applies position-based weighting (typically giving more weight to segments near the beginning of articles)
   * **COUNT_BOOSTED**: Boosts scores based on the number of matching segments in each article

4. **Result Enrichment:**
   * The service fetches full article content and metadata
   * It attaches metadata about the number of matching segments
   * It finds the most relevant segments within each article for highlighting

5. **Result Presentation:**
   * Results are returned as complete articles (rather than fragments)
   * Metadata indicates how many segments matched and their positions
   * Articles are ranked according to the selected scoring strategy

### Usage Example

```python
# Create search API
search_api = create_search_api(config)

# Perform standard segment-based search
segment_results = search_api.search(
    query="Steuererklärung",
    limit=5
)

# Perform article-based search
article_results = search_api.search(
    query="Steuererklärung",
    limit=5,
    group_by_article=True,  # Enable article grouping
    score_strategy="weighted"  # Optional, defaults to "max"
)

# Article results provide complete context
for article in article_results.results:
    print(f"Article {article.section_number}: {article.title}")
    print(f"Score: {article.relevance_score}")
    print(f"Matching segments: {article.metadata.get('matching_segments', 0)}")
    print(f"Content: {article.content[:100]}...")
```

## 8. Overall Architecture & Data Flow Summary

1.  **Source Data:** Laws and Sections reside in DuckDB (`laws`, `sections` tables).
2.  **Configuration:** `IndexingConfig` drives the process (strategy, model, Qdrant mode).
3.  **Indexing:**
    *   `SearchPipeline` reads from DuckDB (`sections`).
    *   `segmentation.segment_text` creates `TextSegment` objects.
    *   When hierarchical segmentation is enabled, the system extracts article IDs, subsection numbers, and hierarchy paths.
    *   `TextEmbedder` generates embedding vectors using a transformer model (via `ModelCache`).
    *   `EmbeddingProcessor` stores segment metadata (including hierarchical information) into DuckDB.
    *   `VectorDatabaseManager` stores the embeddings and metadata into Qdrant.
4.  **Segment-Based Searching:**
    *   `SearchService` takes a text query.
    *   `TextEmbedder` converts the query to a vector.
    *   `VectorDatabaseManager` searches Qdrant using the query vector, returning segment data and scores.
    *   `SearchService` queries DuckDB to fetch full content and applies highlighting.
    *   Returns individual text segments as search results.
5.  **Article-Based Searching:**
    *   Starts with the same segment-based search process.
    *   `ArticleSearchService` groups segments by article ID.
    *   Applies article scoring strategy and ranking.
    *   Returns complete articles with information about matching segments.

This architecture enables both precision searching at the segment level and contextual reading at the article level, addressing the needs of different legal research scenarios.