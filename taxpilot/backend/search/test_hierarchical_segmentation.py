"""
Test the hierarchical segmentation functionality.

This script tests the enhanced segmentation system that builds hierarchical
document structures for legal texts.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from taxpilot.backend.search.segmentation import (
    segment_text,
    SegmentationConfig,
    SegmentationStrategy,
    SegmentType,
    extract_legal_structure,
    build_hierarchy_path,
    derive_article_id
)

# Sample German legal text
SAMPLE_TEXT = """
§ 13 Einkünfte aus Land- und Forstwirtschaft

(1) Einkünfte aus Land- und Forstwirtschaft sind

1. Einkünfte aus dem Betrieb von Landwirtschaft, Forstwirtschaft, Weinbau, Gartenbau, Obstbau, Gemüsebau, Baumschulen und aus allen Betrieben, die Pflanzen und Pflanzenteile mit Hilfe der Naturkräfte gewinnen. Zu diesen Einkünften gehören auch die Einkünfte aus der Tierzucht und Tierhaltung, wenn im Wirtschaftsjahr

für die ersten 20 Hektar
    nicht mehr als 10 Vieheinheiten,
für die nächsten 10 Hektar
    nicht mehr als 7 Vieheinheiten,
für die nächsten 20 Hektar
    nicht mehr als 6 Vieheinheiten,
für die nächsten 50 Hektar
    nicht mehr als 3 Vieheinheiten,
und für die weitere Fläche
    nicht mehr als 1,5 Vieheinheiten

je Hektar der vom Inhaber des Betriebs regelmäßig landwirtschaftlich genutzten Flächen erzeugt oder gehalten werden. Die Tierbestände sind nach dem Futterbedarf in Vieheinheiten umzurechnen;

2. Einkünfte aus Binnenfischerei, Fischzucht und Teichwirtschaft;

3. Einkünfte aus Imkerei;

4. Einkünfte aus Wanderschäferei;

5. Einkünfte aus Saatzucht, Pflanzenzucht, Pflanzenvermehrung und Baumschulen.

(2) Zu den Einkünften im Sinne des Absatzes 1 gehören auch

1. Einkünfte aus einem land- und forstwirtschaftlichen Nebenbetrieb. Als Nebenbetrieb gilt ein Betrieb, der dem land- und forstwirtschaftlichen Hauptbetrieb zu dienen bestimmt ist;

2. Gewinne aus der Veräußerung oder Entnahme von Grund und Boden, der zum Anlagevermögen eines land- und forstwirtschaftlichen Betriebs gehört, wenn der Veräußerungspreis oder der an dessen Stelle tretende Wert nach Abzug der Veräußerungskosten den Buchwert übersteigt.

(3) Die Einkünfte aus Land- und Forstwirtschaft werden bei der Ermittlung des Gesamtbetrags der Einkünfte nur berücksichtigt, soweit sie den Betrag von 900 Euro übersteigen. Bei Ehegatten, die nach §§ 26, 26b zusammen veranlagt werden, verdoppelt sich der Betrag von 900 Euro.

(4) Werden einzelne Wirtschaftsgüter eines land- und forstwirtschaftlichen Betriebs auf einen anderen Betrieb desselben Steuerpflichtigen im Sinne des § 15 Absatz 1 Satz 1 Nummer 1 oder 2 oder einen Betrieb seines Ehegatten überführt, so ist der Wert, mit dem das einzelne Wirtschaftsgut nach § 6 Absatz 1 Nummer 5 anzusetzen ist, als Entnahme anzusetzen. Satz 1 gilt nicht, soweit einzelne Wirtschaftsgüter des Anlagevermögens im Rahmen der Veräußerung eines land- und forstwirtschaftlichen Teilbetriebs auf einen anderen eigenen Betrieb oder auf einen eigenen gewerblichen Betrieb im Sinne des § 15 Absatz 1 Satz 1 Nummer 1 überführt werden. In diesem Fall ist § 16 Absatz 5 entsprechend anzuwenden. 

(5) Die private Nutzung eines Wirtschaftsgutes, das zum land- und forstwirtschaftlichen Betriebsvermögen des Steuerpflichtigen gehört, gehört zu den Einkünften aus Land- und Forstwirtschaft. Wird ein zum land- und forstwirtschaftlichen Betriebsvermögen des Steuerpflichtigen gehörendes Wirtschaftsgut durch den Steuerpflichtigen für eigene Wohnzwecke genutzt, so ist der dadurch entstehende Nutzungswert nicht anzusetzen; damit zusammenhängende Aufwendungen dürfen den Gewinn nicht mindern. Satz 2 gilt entsprechend für die Nutzung durch eine zur Ehe des Steuerpflichtigen gehörige Person oder durch Personen, die zu seinen Kindern im Sinne des § 32 zu rechnen sind.

(6) Absatz 5 gilt entsprechend für Grundstücksteile, soweit sie dem Inhaber des Betriebs oder einer zu seinem Haushalt gehörigen Person als Wohnung dienen oder zu eigenen oder zu land- und forstwirtschaftsfremden Zwecken des Steuerpflichtigen genutzt werden. Satz 1 ist auch anzuwenden auf Gebäude oder Gebäudeteile, die nach ihrem Nutzungszusammenhang zum Wohngebäude oder zu land- und forstwirtschaftsfremden Zwecken genutzten Gebäude gehören oder der besonderen Zweckbestimmung dieser Gebäude dienen und untergeordnete Bedeutung haben.
"""

def test_extract_legal_structure():
    """Test extracting legal structure from text."""
    structure = extract_legal_structure(SAMPLE_TEXT)
    print("Extracted structure:")
    print(f"Article number: {structure['article_number']}")
    print(f"Is article: {structure['is_article']}")
    print(f"Hierarchy components: {structure['hierarchy_components']}")
    
    assert structure["article_number"] == "13"
    assert structure["is_article"] == True
    assert "§13" in structure["hierarchy_components"]
    
    # Test a subsection
    subsection = """
    (2) Zu den Einkünften im Sinne des Absatzes 1 gehören auch
    
    1. Einkünfte aus einem land- und forstwirtschaftlichen Nebenbetrieb. Als Nebenbetrieb gilt ein Betrieb, der dem land- und forstwirtschaftlichen Hauptbetrieb zu dienen bestimmt ist;
    """
    
    sub_structure = extract_legal_structure(subsection)
    print("\nSubsection structure:")
    print(f"Subsections: {sub_structure['subsections']}")
    
    assert "2" in sub_structure["subsections"]
    
    return True

def test_hierarchical_segmentation():
    """Test hierarchical segmentation of legal text."""
    # Configure segmentation
    config = SegmentationConfig(
        strategy=SegmentationStrategy.PARAGRAPH,
        build_hierarchy=True,
        detect_articles=True,
        identify_subsections=True,
        extract_section_numbers=True
    )
    
    # Perform segmentation
    segments = segment_text(SAMPLE_TEXT, "estg", "estg_13", config)
    
    print(f"\nGenerated {len(segments)} segments")
    
    # Test that article ID is correctly set
    for i, segment in enumerate(segments[:3]):  # Print first 3 segments
        print(f"\nSegment {i+1}:")
        print(f"Type: {segment.segment_type}")
        print(f"Article ID: {segment.article_id}")
        print(f"Hierarchy path: {segment.hierarchy_path}")
        print(f"Text excerpt: {segment.text[:50]}...")
        print(f"Metadata: {segment.metadata}")
    
    # Verify hierarchy information is present
    assert all(segment.article_id for segment in segments)
    assert all(segment.hierarchy_path for segment in segments)
    
    # Check that article ID is consistently the same for all segments
    article_ids = set(segment.article_id for segment in segments)
    assert len(article_ids) == 1, f"Expected 1 article ID, got {len(article_ids)}: {article_ids}"
    
    return True

def test_sentence_level_segmentation():
    """Test sentence-level segmentation with hierarchy."""
    # Configure segmentation
    config = SegmentationConfig(
        strategy=SegmentationStrategy.SENTENCE,
        build_hierarchy=True,
        detect_articles=True,
        identify_subsections=True,
        extract_section_numbers=True
    )
    
    # Use a smaller sample for sentence segmentation
    sample = SAMPLE_TEXT.split('\n\n')[0:3]
    sample_text = '\n\n'.join(sample)
    
    # Perform segmentation
    segments = segment_text(sample_text, "estg", "estg_13", config)
    
    print(f"\nGenerated {len(segments)} sentence-level segments")
    
    # Print details of the first few segments
    for i, segment in enumerate(segments[:2]):
        print(f"\nSentence segment {i+1}:")
        print(f"Type: {segment.segment_type}")
        print(f"Hierarchy path: {segment.hierarchy_path}")
        print(f"Text: {segment.text}")
    
    # Verify hierarchy paths are unique
    paths = [segment.hierarchy_path for segment in segments]
    assert len(paths) == len(set(paths)), "Hierarchy paths should be unique"
    
    return True

if __name__ == "__main__":
    print("Testing hierarchical segmentation...")
    
    # Run tests
    test_extract_legal_structure()
    test_hierarchical_segmentation()
    test_sentence_level_segmentation()
    
    print("\nAll tests passed!")