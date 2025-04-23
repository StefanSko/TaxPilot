"""
Tests for search utility functions.

This module tests the utility functions used for enhancing search results,
particularly text highlighting and context extraction.
"""

import pytest
from taxpilot.backend.api.search_utils import (
    highlight_text,
    extract_context,
    merge_overlapping_snippets
)


class TestHighlightText:
    """Tests for the highlight_text function."""

    def test_basic_highlighting(self):
        """Test basic highlighting of a single term."""
        text = "Dies ist ein Test für Steuererklärung"
        query = "Steuererklärung"
        
        highlighted = highlight_text(text, query)
        
        assert "<mark>Steuererklärung</mark>" in highlighted
        assert highlighted == "Dies ist ein Test für <mark>Steuererklärung</mark>"

    def test_multiple_terms_highlighting(self):
        """Test highlighting of multiple terms."""
        text = "Eine Steuererklärung für Homeoffice und Gewerbesteuer"
        query = "Steuererklärung Homeoffice"
        
        highlighted = highlight_text(text, query)
        
        assert "<mark>Steuererklärung</mark>" in highlighted
        assert "<mark>Homeoffice</mark>" in highlighted

    def test_case_insensitive_highlighting(self):
        """Test that highlighting is case-insensitive."""
        text = "STEUERERKLÄRUNG ist wichtig"
        query = "steuererklärung"
        
        highlighted = highlight_text(text, query)
        
        assert "<mark>STEUERERKLÄRUNG</mark>" in highlighted

    def test_word_boundary_highlighting(self):
        """Test that only whole words are highlighted."""
        text = "Steuer und Steuererklärung sind verschieden"
        query = "Steuer"
        
        highlighted = highlight_text(text, query)
        
        assert "<mark>Steuer</mark>" in highlighted
        assert "Steuererklärung" in highlighted  # Not highlighted
        assert "<mark>Steuer</mark>erklärung" not in highlighted  # Not part of the word

    def test_stop_words_ignored(self):
        """Test that common stop words are ignored."""
        text = "Der Test für die Steuer"
        query = "der für test"
        
        highlighted = highlight_text(text, query)
        
        assert "<mark>Test</mark>" in highlighted
        assert "<mark>der</mark>" not in highlighted  # Stop word
        assert "<mark>für</mark>" not in highlighted  # Stop word

    def test_empty_inputs(self):
        """Test behavior with empty inputs."""
        assert highlight_text("", "query") == ""
        assert highlight_text("text", "") == "text"
        assert highlight_text("", "") == ""


class TestExtractContext:
    """Tests for the extract_context function."""

    def test_basic_context_extraction(self):
        """Test basic context extraction around a match."""
        text = "Dies ist ein sehr langer Text, der eine Steuererklärung enthält. " * 10
        query = "Steuererklärung"
        
        context = extract_context(text, query, context_size=20)
        
        assert "Steuererklärung" in context
        assert len(context) < len(text)
        
    def test_multiple_matches_context(self):
        """Test context extraction with multiple matches."""
        text = "Homeoffice am Anfang. " + ("X " * 50) + "Homeoffice in der Mitte. " + ("Y " * 50) + "Homeoffice am Ende."
        query = "Homeoffice"
        
        # For each match, we extract the match and surrounding context
        context = extract_context(text, query, context_size=10)
        
        # The function should extract context around each match, which includes all three instances
        assert "Homeoffice" in context
        
        # Check that the context has all three matches (we're okay even if the full context isn't there)
        assert context.count("Homeoffice") == 3
    
    def test_short_text_returns_unchanged(self):
        """Test that short texts are returned unchanged."""
        text = "Kurzer Text mit Steuererklärung"
        query = "Steuererklärung"
        
        context = extract_context(text, query, context_size=100)
        
        assert context == text
    
    def test_no_matches_returns_beginning(self):
        """Test that text beginning is returned when no matches found."""
        text = "Dies ist ein sehr langer Text ohne relevante Begriffe. " * 10
        query = "Steuererklärung"
        
        context = extract_context(text, query, context_size=30)
        
        assert context.startswith("Dies ist ein sehr langer Text")
        assert context.endswith("...")
        assert len(context) < len(text)


class TestMergeOverlappingSnippets:
    """Tests for the merge_overlapping_snippets function."""

    def test_non_overlapping_snippets_unchanged(self):
        """Test that non-overlapping snippets are returned unchanged."""
        snippets = [
            "Dies ist der erste Abschnitt.",
            "Völlig andere Wörter hier."  # Make sure these have no common words
        ]
        
        merged = merge_overlapping_snippets(snippets)
        
        assert len(merged) == 2
        assert set(merged) == set(snippets)  # Order might change, check contents
    
    def test_overlapping_snippets_merged(self):
        """Test that overlapping snippets are merged."""
        snippets = [
            "Dies ist ein Abschnitt über Steuererklärung.",
            "Steuererklärung ist ein wichtiges Thema."
        ]
        
        merged = merge_overlapping_snippets(snippets)
        
        assert len(merged) == 1
        # The merging currently returns the longest snippet
        assert merged[0] == max(snippets, key=len)
    
    def test_empty_input_returns_empty_list(self):
        """Test that an empty input returns an empty list."""
        assert merge_overlapping_snippets([]) == []
    
    def test_single_snippet_returned_unchanged(self):
        """Test that a single snippet is returned unchanged."""
        snippet = "Dies ist ein einzelner Abschnitt."
        assert merge_overlapping_snippets([snippet]) == [snippet]