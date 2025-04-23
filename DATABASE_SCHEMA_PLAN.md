# Database Schema Alignment Plan

## Problem Analysis

The TaxPilot project has two different definitions of the `section_embeddings` table schema:

1. In `embeddings.py`, it expects columns including:
   - `id` (PRIMARY KEY)
   - `law_id` 
   - `section_id`
   - `segment_id`
   - `embedding_model`
   - `embedding_version`
   - `embedding` (vector array)
   - `vector_db_id`
   - `metadata` (JSON)
   - `created_at` (timestamp)

2. In `database.py`, it defines a simpler schema:
   - `section_id` (PRIMARY KEY)
   - `embedding` (vector array with size 384)
   - Foreign key relationship to sections table

This mismatch prevents the search examples from working correctly and causes errors when initializing the embedding system.

## Implementation Strategy

### Phase 1: Schema Unification

1. Create a new function `update_section_embeddings_schema()` in `database.py` to handle migrations.
2. Modify the `initialize_database()` function to use the extended schema that matches what `embeddings.py` expects.
3. Update the table creation SQL to include all required fields with appropriate types.
4. Update the existing method signatures for CRUD operations to maintain backward compatibility.

### Phase 2: Data Migration

1. Implement a migration function to upgrade existing databases by:
   - Creating a temporary backup of existing data
   - Recreating the table with the new schema
   - Migrating data back with the new schema fields
   - Cleaning up temporary tables

### Phase 3: Code Updates

1. Refactor any functions that interact with the section_embeddings table to use the extended schema.
2. Update the vector database synchronization logic to handle the consistent schema.
3. Ensure all example scripts work with the unified schema.

### Phase 4: Testing

1. Add unit tests for database schema validation
2. Create tests that verify successful migration of data
3. Test all search examples with the updated schema

## Risks and Mitigations

### Data Loss Risk
- **Risk**: Existing embeddings could be lost during schema migration
- **Mitigation**: Implement backup/restore functionality, test migration thoroughly

### Performance Risk
- **Risk**: The extended schema might affect performance
- **Mitigation**: Add appropriate indexes for optimized queries

### Compatibility Risk
- **Risk**: Various parts of the code may assume different schemas
- **Mitigation**: Implement utility functions to abstract schema details

## Future Recommendations

1. Centralize schema definitions in a single location
2. Implement versioned migrations system
3. Add schema validation at startup
4. Document schema requirements clearly for contributors