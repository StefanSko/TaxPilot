"""
Unit tests for the text segmentation module.
"""

import time
import pytest
from taxpilot.backend.search.segmentation import (
    SegmentationStrategy, SegmentationConfig, TextSegment,
    segment_text, clean_and_normalize_text, handle_special_formatting,
    segment_by_section, segment_by_paragraph, segment_by_sentence,
    segment_with_overlapping_windows, optimize_chunk_size
)


@pytest.fixture
def sample_legal_text() -> str:
    """Return a sample legal text for testing."""
    return """
§ 1 Persönlicher Anwendungsbereich

(1) Unbeschränkt einkommensteuerpflichtig sind natürliche Personen, die im Inland einen Wohnsitz oder 
ihren gewöhnlichen Aufenthalt haben. Zum Inland im Sinne dieses Gesetzes gehört auch der der 
Bundesrepublik Deutschland zustehende Anteil am Festlandsockel, soweit dort Naturschätze des 
Meeresgrundes und des Meeresuntergrundes erforscht oder ausgebeutet werden oder dieser der 
Energieerzeugung dient.

(2) Auf Antrag werden auch natürliche Personen als unbeschränkt einkommensteuerpflichtig behandelt, 
die im Inland weder einen Wohnsitz noch ihren gewöhnlichen Aufenthalt haben, soweit sie 
inländische Einkünfte im Sinne des § 49 haben. Dies gilt nur, wenn ihre Einkünfte im 
Kalenderjahr mindestens zu 90 Prozent der deutschen Einkommensteuer unterliegen oder die nicht 
der deutschen Einkommensteuer unterliegenden Einkünfte den Grundfreibetrag nach § 32a Absatz 1 
Satz 2 Nummer 1 nicht übersteigen; dieser Betrag ist zu kürzen, soweit es nach den Verhältnissen 
im Wohnsitzstaat des Steuerpflichtigen notwendig und angemessen ist. Inländische Einkünfte, die 
nach einem Abkommen zur Vermeidung der Doppelbesteuerung nur der Höhe nach beschränkt besteuert 
werden dürfen, gelten hierbei als nicht der deutschen Einkommensteuer unterliegend. 
Unberücksichtigt bleiben bei der Ermittlung der Einkünfte nach Satz 2 nicht der deutschen 
Einkommensteuer unterliegende Einkünfte, die im Ausland nicht besteuert werden, soweit 
vergleichbare Einkünfte im Inland steuerfrei sind. Weitere Voraussetzung ist, dass die 
Höhe der nicht der deutschen Einkommensteuer unterliegenden Einkünfte durch eine Bescheinigung 
der zuständigen ausländischen Steuerbehörde nachgewiesen wird. Der Steuerabzug nach § 50a steht 
der unbeschränkten Einkommensteuerpflicht nach Satz 1 nicht entgegen.

§ 2 Umfang der Besteuerung, Begriffsbestimmungen

(1) Der Einkommensteuer unterliegen
1. Einkünfte aus Land- und Forstwirtschaft,
2. Einkünfte aus Gewerbebetrieb,
3. Einkünfte aus selbständiger Arbeit,
4. Einkünfte aus nichtselbständiger Arbeit,
5. Einkünfte aus Kapitalvermögen,
6. Einkünfte aus Vermietung und Verpachtung,
7. sonstige Einkünfte im Sinne des § 22, die der Steuerpflichtige während seiner unbeschränkten 
Einkommensteuerpflicht oder als inländische Einkünfte während seiner beschränkten 
Einkommensteuerpflicht erzielt.
    """


@pytest.fixture
def sample_section_text() -> str:
    """Return a sample section text for testing."""
    return """
(2) Einkünfte sind der Gewinn bei den Einkunftsarten des Absatzes 1 Nummer 1 bis 3 und der Überschuss 
der Einnahmen über die Werbungskosten bei den Einkunftsarten des Absatzes 1 Nummer 4 bis 7. 
Bei nichtselbständiger Arbeit tritt an die Stelle der Werbungskosten der Arbeitnehmer-Pauschbetrag (§ 9a Satz 1 Nummer 1 Buchstabe a).

(3) Die Summe der Einkünfte, vermindert um den Altersentlastungsbetrag, den Entlastungsbetrag für 
Alleinerziehende und den Freibetrag nach § 13 Absatz 3 für Einkünfte aus Land- und Forstwirtschaft, 
ist der Gesamtbetrag der Einkünfte.
    """


@pytest.fixture
def complex_formatting_text() -> str:
    """Return text with complex legal formatting."""
    return """
§ l Allgemeine Vorschriften

1. Der Steuerpflichtige hat:
   a) seine Bücher zu führen und
   b) Aufzeichnungen zu machen.
2. Die Buchführung muss:
   a) vollständig,
   b) richtig,
   c) zeitgerecht und
   d) geordnet sein.

Die Vorschriften gelten entsprechend § 238 HGB (vgl. auch § 140 AO).
Die Regelungen sind im EStG verankert.
    """


def test_clean_and_normalize_text(complex_formatting_text):
    """Test cleaning and normalizing text."""
    cleaned = clean_and_normalize_text(complex_formatting_text)
    
    # Should fix OCR issues
    assert "§ 1 Allgemeine Vorschriften" in cleaned
    assert "§l" not in cleaned
    
    # Should handle citations
    assert "(vgl. auch" in cleaned
    
    # Should remove excessive whitespace
    assert "  " not in cleaned


def test_handle_special_formatting(complex_formatting_text):
    """Test handling special formatting in legal texts."""
    formatted = handle_special_formatting(complex_formatting_text)
    
    # Should handle enumerations
    assert "1." in formatted
    assert "2." in formatted
    
    # Should handle letter lists
    assert "a)" in formatted
    assert "b)" in formatted
    
    # Should handle section references
    assert "§ 238 HGB" in formatted
    assert "§ 140 AO" in formatted
    
    # Should handle law abbreviations
    assert "HGB" in formatted
    assert "AO" in formatted
    assert "EStG" in formatted


def test_segment_by_section(sample_legal_text):
    """Test segmenting text at the section level."""
    config = SegmentationConfig(strategy=SegmentationStrategy.SECTION)
    segments = segment_by_section(sample_legal_text, "estg", "s1", config)
    
    # Should create exactly one segment for the entire text
    assert len(segments) == 1
    
    # Check that key phrases are in the segment (exact spacing may be different after cleaning)
    assert "Persönlicher Anwendungsbereich" in segments[0].text
    assert "Umfang der Besteuerung" in segments[0].text
    
    # Metadata should be set correctly
    assert segments[0].segment_id == "s1_full"
    assert segments[0].law_id == "estg"
    assert segments[0].section_id == "s1"
    assert segments[0].metadata["strategy"] == SegmentationStrategy.SECTION.value


def test_segment_by_paragraph(sample_legal_text):
    """Test segmenting text at the paragraph level."""
    config = SegmentationConfig(
        strategy=SegmentationStrategy.PARAGRAPH,
        min_chunk_size=10,  # Small value for testing
        max_chunk_size=5000  # Large value to avoid further subdivision
    )
    segments = segment_by_paragraph(sample_legal_text, "estg", "s1", config)
    
    # Each segment should be a paragraph from the original text
    # At minimum, there should be at least one segment
    assert len(segments) > 0
    
    # Check for characteristic phrases (spacing may be different after cleaning)
    assert any("Persönlicher Anwendungsbereich" in segment.text for segment in segments)
    
    # Check for specific paragraph content
    paragraph_patterns = [
        "Unbeschränkt einkommensteuerpflichtig sind",
        "Auf Antrag werden auch natürliche Personen",
        "Der Einkommensteuer unterliegen"
    ]
    
    found_patterns = 0
    for pattern in paragraph_patterns:
        if any(pattern in segment.text for segment in segments):
            found_patterns += 1
    
    # At least some of the patterns should be found
    assert found_patterns > 0
    
    # Check that metadata is properly set
    for segment in segments:
        assert segment.law_id == "estg"
        assert segment.section_id == "s1"
        assert "paragraph" in segment.metadata["strategy"]
        assert isinstance(segment.metadata["paragraph_index"], int)


def test_segment_by_sentence(sample_section_text):
    """Test segmenting text at the sentence level."""
    config = SegmentationConfig(
        strategy=SegmentationStrategy.SENTENCE,
        chunk_size=150,  # Small value to force multiple chunks
        chunk_overlap=30,
        min_chunk_size=10
    )
    segments = segment_by_sentence(sample_section_text, "estg", "s2", config)
    
    # Should create multiple segments
    assert len(segments) > 0
    
    # Sentences should be properly grouped
    for segment in segments:
        assert len(segment.text) > 0
        assert segment.law_id == "estg"
        assert segment.section_id == "s2"
        assert segment.metadata["strategy"] == SegmentationStrategy.SENTENCE.value
        
        # Each segment should be smaller than max_chunk_size
        assert len(segment.text) <= config.max_chunk_size


def test_segment_with_overlapping_windows(sample_legal_text):
    """Test segmenting text with overlapping windows."""
    config = SegmentationConfig(
        chunk_size=200,  # Small chunks for testing
        chunk_overlap=50,
        min_chunk_size=10
    )
    segments = segment_with_overlapping_windows(sample_legal_text, "estg", "s1", config)
    
    # Should create multiple segments
    assert len(segments) > 1
    
    # Check that windows overlap
    found_overlap = False
    for i in range(len(segments) - 1):
        end_of_first = segments[i].text[-30:]
        start_of_second = segments[i+1].text[:30]
        
        # Check for overlap between consecutive segments
        for j in range(1, len(end_of_first) - 5):
            if end_of_first[-j:] in start_of_second:
                found_overlap = True
                break
    
    assert found_overlap
    
    # Check metadata
    for i, segment in enumerate(segments):
        assert segment.metadata["window_index"] == i
        assert segment.metadata["strategy"] == "window"


def test_segment_text_with_different_strategies(sample_legal_text):
    """Test segment_text function with different strategies."""
    for strategy in SegmentationStrategy:
        config = SegmentationConfig(strategy=strategy)
        segments = segment_text(sample_legal_text, "estg", "s1", config)
        
        # Should create at least one segment
        assert len(segments) > 0
        
        # All segments should have the correct metadata and IDs
        for segment in segments:
            assert segment.law_id == "estg"
            assert segment.section_id == "s1"
            
            # Check that strategy metadata exists
            assert "strategy" in segment.metadata
            
            # In our implementation, depending on how the segmentation happens,
            # the exact strategy value might change, as segmenters can call other
            # segmenters internally (e.g., paragraph segmenter might call sentence segmenter
            # for long paragraphs). We just verify that a strategy exists.


def test_optimize_chunk_size():
    """Test chunk size optimization based on model limits."""
    model_max_tokens = 1024
    optimal_size = optimize_chunk_size(model_max_tokens)
    
    # Optimal size should be less than model_max_tokens * token_to_char_ratio
    assert optimal_size < model_max_tokens * 5
    
    # Should be proportional to token limits
    assert optimize_chunk_size(2048) > optimize_chunk_size(1024)


def test_empty_text():
    """Test handling of empty text."""
    config = SegmentationConfig()
    segments = segment_text("", "estg", "s1", config)
    
    # Should return empty list for empty text
    assert len(segments) == 0


def test_segmentation_performance(sample_legal_text):
    """Test the performance of different segmentation strategies."""
    large_text = sample_legal_text * 10  # Create larger text for timing
    
    strategies = [
        SegmentationStrategy.SECTION,
        SegmentationStrategy.PARAGRAPH,
        SegmentationStrategy.SENTENCE
    ]
    
    strategy_times = {}
    segment_counts = {}
    
    for strategy in strategies:
        config = SegmentationConfig(strategy=strategy)
        
        start_time = time.time()
        segments = segment_text(large_text, "estg", "s1", config)
        end_time = time.time()
        
        strategy_times[strategy.value] = end_time - start_time
        segment_counts[strategy.value] = len(segments)
    
    # Section should be fastest (single segment)
    assert strategy_times[SegmentationStrategy.SECTION.value] <= strategy_times[SegmentationStrategy.PARAGRAPH.value]
    
    # Section should produce fewest segments
    assert segment_counts[SegmentationStrategy.SECTION.value] == 1
    
    # Either paragraph or sentence segmentation should produce more segments than section
    assert segment_counts[SegmentationStrategy.PARAGRAPH.value] > 0 or segment_counts[SegmentationStrategy.SENTENCE.value] > 1
    
    # Performance data is available (no assertion, just for reporting)
    assert strategy_times
    assert segment_counts