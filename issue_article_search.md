## Problem Statement

The current search implementation returns individual text segments rather than complete legal articles, creating several user experience issues:

- Users see disconnected segments without proper context
- It's difficult to understand where segments fit within legal structure
- Relevance scores are calculated at segment level, not article level
- No clear boundaries between different legal articles in results

## Solution Overview

Enhance the search system to return complete articles/sections while preserving semantic search capabilities by implementing:

1. **Hierarchical Document Structure**
2. **Two-Phase Search**
3. **Result Aggregation**
4. **Improved Result Presentation**

## Implementation Checklist

### 1. Enhanced Segmentation Process ✅

- [x] Modify `TextSegment` class to include hierarchical information:
  - [x] Add `article_id` field alongside existing `section_id`
  - [x] Add `hierarchy_path` field to store full path (e.g., "estg/§13/abs2/satz1")
  - [x] Add `segment_type` field (article, paragraph, sentence)
  - [x] Track position within parent (e.g., paragraph number within article)

- [x] Update segmentation functions to preserve legal structure:
  - [x] Identify article boundaries during segmentation
  - [x] Recognize paragraph and subsection structures
  - [x] Maintain original numbering from legal text

### 2. Vector Database Enhancements ✅

- [x] Modify vector database payload structure:
  - [x] Add hierarchical identifiers to each vector entry
  - [x] Store article-level metadata alongside segment data
  - [x] Add structural position information

- [x] Create article-to-segment mapping:
  - [x] Build index of segments belonging to each article
  - [x] Store relationship between segments and articles

### 3. Two-Phase Search Implementation ✅

- [x] Modify `SearchService.search()` method:
  - [x] Phase 1: Execute semantic search on segments (existing approach)
  - [x] Phase 2: Group results by article/section
  - [x] Aggregate scores for segments within same article

- [x] Add score aggregation strategies:
  - [x] Implement max score (best matching segment determines article relevance)
  - [x] Implement weighted average (all segment matches contribute to article score)
  - [x] Implement count-based boosting (articles with more matching segments rank higher)

### 4. Result Processing Updates

- [ ] Modify `_enrich_results()` method:
  - [ ] Group segments by article/section ID
  - [ ] Fetch complete article content instead of just segments
  - [ ] Combine metadata from all matching segments

- [ ] Create new `ArticleResult` class:
  - [ ] Include complete article text and structure
  - [ ] Store relevance score for entire article
  - [ ] Track which segments matched and their individual scores
  - [ ] Support for highlighting multiple segments within article

### 5. Database Query Optimization

- [ ] Optimize database queries for article retrieval:
  - [ ] Create efficient query to get complete article content
  - [ ] Fetch article structure and metadata in single query
  - [ ] Implement caching for frequently accessed articles

- [ ] Add indexing for faster article lookup:
  - [ ] Create indexes on hierarchy fields
  - [ ] Optimize for article-level retrieval

### 6. Result Presentation Improvements

- [ ] Enhance `QueryResult` to support article-based results:
  - [ ] Include article number and title prominently
  - [ ] Show law abbreviation (e.g., "EStG § 13")
  - [ ] Present complete article with proper formatting
  - [ ] Highlight all matching segments within article
  - [ ] Indicate why article matched (which concepts matched)

- [ ] Implement smart snippet selection:
  - [ ] Show most relevant parts of article based on match strength
  - [ ] Provide context around matching segments
  - [ ] Include proper heading hierarchy

## Testing and Validation

- [x] Create test suite for basic hierarchical segmentation:
  - [x] Test article detection and structure extraction
  - [x] Test hierarchical path construction
  - [x] Test segmentation with hierarchy at different levels

- [ ] Create test suite for article-based search:
  - [ ] Test accurate grouping of segments to articles
  - [ ] Test relevance scoring at article level
  - [ ] Test highlighting of multiple matches within article
  - [ ] Compare precision/recall with segment-based approach

- [ ] Validate with real queries:
  - [ ] Select 20-30 representative legal queries
  - [ ] Compare current vs. new approach results
  - [ ] Evaluate user understanding and satisfaction

## Implementation Status

### Completed ✅
- Enhanced segmentation process with hierarchical information
- Created test suite for hierarchical segmentation

### In Progress 🔄
- Result Processing Updates

## Implementation Phases

1. **Phase 1: Data Structure Updates** (1-2 weeks) ✅
   - Enhance `TextSegment` and database schema
   - Update segmentation process

2. **Phase 2: Search Logic Modifications** (2-3 weeks)
   - Implement two-phase search
   - Add article grouping and score aggregation

3. **Phase 3: Result Processing** (1-2 weeks)
   - Create article-based result format
   - Implement highlighting for multiple segments

4. **Phase 4: Testing and Optimization** (1-2 weeks)
   - Test accuracy and performance
   - Optimize bottlenecks