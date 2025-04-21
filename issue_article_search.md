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

### 1. Enhanced Segmentation Process

- [ ] Modify `TextSegment` class to include hierarchical information:
  - [ ] Add `article_id` field alongside existing `section_id`
  - [ ] Add `hierarchy_path` field to store full path (e.g., "estg/§13/abs2/satz1")
  - [ ] Add `segment_type` field (article, paragraph, sentence)
  - [ ] Track position within parent (e.g., paragraph number within article)

- [ ] Update segmentation functions to preserve legal structure:
  - [ ] Identify article boundaries during segmentation
  - [ ] Recognize paragraph and subsection structures
  - [ ] Maintain original numbering from legal text

### 2. Vector Database Enhancements

- [ ] Modify vector database payload structure:
  - [ ] Add hierarchical identifiers to each vector entry
  - [ ] Store article-level metadata alongside segment data
  - [ ] Add structural position information

- [ ] Create article-to-segment mapping:
  - [ ] Build index of segments belonging to each article
  - [ ] Store relationship between segments and articles

### 3. Two-Phase Search Implementation

- [ ] Modify `SearchService.search()` method:
  - [ ] Phase 1: Execute semantic search on segments (existing approach)
  - [ ] Phase 2: Group results by article/section
  - [ ] Aggregate scores for segments within same article

- [ ] Add score aggregation strategies:
  - [ ] Implement max score (best matching segment determines article relevance)
  - [ ] Implement weighted average (all segment matches contribute to article score)
  - [ ] Implement count-based boosting (articles with more matching segments rank higher)

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

- [ ] Create test suite for article-based search:
  - [ ] Test accurate grouping of segments to articles
  - [ ] Test relevance scoring at article level
  - [ ] Test highlighting of multiple matches within article
  - [ ] Compare precision/recall with segment-based approach

- [ ] Validate with real queries:
  - [ ] Select 20-30 representative legal queries
  - [ ] Compare current vs. new approach results
  - [ ] Evaluate user understanding and satisfaction

## Implementation Phases

1. **Phase 1: Data Structure Updates** (1-2 weeks)
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