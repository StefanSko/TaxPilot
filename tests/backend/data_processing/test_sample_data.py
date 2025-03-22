"""
Sample test data for the GermanLawFinder database tests.

This module provides sample data fixtures derived from the actual XML files
in the data directory.
"""

from datetime import date
from typing import TypedDict, Any, cast, List, Dict

from taxpilot.backend.data_processing.database import Law, Section, SectionEmbedding


def get_sample_laws() -> list[Law]:
    """
    Get sample law data extracted from the XML files.
    
    Returns:
        A list of Law objects with sample data.
    """
    laws_data = [
        {
            "id": "estg",
            "full_name": "Einkommensteuergesetz",
            "abbreviation": "EStG",
            "last_updated": date(2024, 12, 23),  # From XML: "zuletzt geändert durch Art. 2 G v. 23.12.2024"
            "issue_date": date(1934, 10, 16),    # From XML: "ausfertigung-datum 1934-10-16"
            "status_info": "Neugefasst durch Bek. v. 8.10.2009 I 3366, 3862; zuletzt geändert durch Art. 2 G v. 23.12.2024",
            "metadata": {
                "jurisdiction": "Germany",
                "category": "Tax Law",
                "xml_file": "BJNR010050934.xml"
            }
        },
        {
            "id": "ao",
            "full_name": "Abgabenordnung",
            "abbreviation": "AO",
            "last_updated": date(2025, 1, 23),   # From XML: "Neugefasst durch Bek. v. 23.1.2025 I Nr. 24"
            "issue_date": date(1976, 3, 16),     # From XML: "ausfertigung-datum 1976-03-16"
            "status_info": "Neugefasst durch Bek. v. 23.1.2025 I Nr. 24",
            "metadata": {
                "jurisdiction": "Germany",
                "category": "Tax Law",
                "xml_file": "BJNR006130976.xml"
            }
        }
    ]
    
    return [cast(Law, law) for law in laws_data]


def get_sample_estg_sections() -> list[Section]:
    """
    Get sample sections from EStG (Income Tax Act).
    
    Returns:
        A list of Section objects with sample data from EStG.
    """
    sections = [
        {
            "id": "estg_1",
            "law_id": "estg",
            "section_number": "1",
            "title": "Steuerpflicht",
            "content": "Natürliche Personen, die im Inland einen Wohnsitz oder ihren gewöhnlichen Aufenthalt haben, sind unbeschränkt einkommensteuerpflichtig. Zum Inland im Sinne dieses Gesetzes gehört auch der der Bundesrepublik Deutschland zustehende Anteil am Festlandsockel, soweit dort Naturschätze des Meeresgrundes und des Meeresuntergrundes erforscht oder ausgebeutet werden oder dieser der Energieerzeugung unter Nutzung erneuerbarer Energien dient.",
            "parent_section_id": None,
            "hierarchy_level": 1,
            "path": "/estg/1",
            "metadata": {"type": "section"}
        },
        {
            "id": "estg_2",
            "law_id": "estg",
            "section_number": "2",
            "title": "Umfang der Besteuerung, Begriffsbestimmungen",
            "content": "(1) Der Einkommensteuer unterliegen\n1. Einkünfte aus Land- und Forstwirtschaft,\n2. Einkünfte aus Gewerbebetrieb,\n3. Einkünfte aus selbständiger Arbeit,\n4. Einkünfte aus nichtselbständiger Arbeit,\n5. Einkünfte aus Kapitalvermögen,\n6. Einkünfte aus Vermietung und Verpachtung,\n7. sonstige Einkünfte im Sinne des § 22,\nsoweit sie dem Steuerpflichtigen im Kalenderjahr zugeflossen sind.",
            "parent_section_id": None,
            "hierarchy_level": 1,
            "path": "/estg/2",
            "metadata": {"type": "section"}
        },
        {
            "id": "estg_5a",
            "law_id": "estg",
            "section_number": "5a",
            "title": "Gewinnermittlung bei Handelsschiffen im internationalen Verkehr",
            "content": "(1) Anstelle der Ermittlung des Gewinns nach § 4 Absatz 1 oder § 5 ist bei einem Gewerbebetrieb mit Geschäftsleitung im Inland der Gewinn, soweit er auf den Betrieb von Handelsschiffen im internationalen Verkehr entfällt, auf unwiderruflichen Antrag des Steuerpflichtigen nach der in seinem Betrieb geführten Tonnage zu ermitteln, wenn die Bereederung dieser Handelsschiffe im Inland durchgeführt wird...",
            "parent_section_id": None,
            "hierarchy_level": 1,
            "path": "/estg/5a",
            "metadata": {"type": "section"}
        }
    ]
    
    return [cast(Section, section) for section in sections]


def get_sample_ao_sections() -> list[Section]:
    """
    Get sample sections from AO (Fiscal Code).
    
    Returns:
        A list of Section objects with sample data from AO.
    """
    sections = [
        {
            "id": "ao_1",
            "law_id": "ao",
            "section_number": "1",
            "title": "Anwendungsbereich",
            "content": "Dieses Gesetz gilt für alle Steuern einschließlich der Steuervergütungen, die durch Bundesrecht oder Recht der Europäischen Union geregelt sind, soweit sie durch Bundes- oder Landesfinanzbehörden verwaltet werden.",
            "parent_section_id": None,
            "hierarchy_level": 1,
            "path": "/ao/1",
            "metadata": {"type": "section", "part": "Erster Teil", "chapter": "Erster Abschnitt"}
        },
        {
            "id": "ao_3",
            "law_id": "ao",
            "section_number": "3",
            "title": "Steuern, steuerliche Nebenleistungen",
            "content": "Steuern sind Geldleistungen, die nicht eine Gegenleistung für eine besondere Leistung darstellen und von einem öffentlich-rechtlichen Gemeinwesen zur Erzielung von Einnahmen allen auferlegt werden, bei denen der Tatbestand zutrifft, an den das Gesetz die Leistungspflicht knüpft.",
            "parent_section_id": None,
            "hierarchy_level": 1,
            "path": "/ao/3",
            "metadata": {"type": "section", "part": "Erster Teil", "chapter": "Zweiter Abschnitt"}
        }
    ]
    
    return [cast(Section, section) for section in sections]


def get_sample_embeddings(section_ids: list[str]) -> list[SectionEmbedding]:
    """
    Generate sample embeddings for the given section IDs.
    
    Args:
        section_ids: List of section IDs to generate embeddings for.
        
    Returns:
        A list of SectionEmbedding objects with sample data.
    """
    embeddings = []
    
    for i, section_id in enumerate(section_ids):
        # Create a unique vector for each section based on its index
        vector = [float(i + j)/1000 for j in range(384)]
        embeddings.append({
            "section_id": section_id,
            "embedding": vector
        })
    
    return [cast(SectionEmbedding, embedding) for embedding in embeddings]