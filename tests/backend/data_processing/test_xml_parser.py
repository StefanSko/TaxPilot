"""
Unit tests for the XML parser module.
"""

import os
from pathlib import Path
import pytest
from lxml import etree
from unittest.mock import patch, mock_open, MagicMock

from taxpilot.backend.data_processing.xml_parser import (
    clean_text, extract_tables, element_to_text, element_to_html,
    extract_law_metadata, extract_section_content, parse_section,
    extract_sections, parse_xml, parse_xml_file, parse_zip_file,
    process_law_file, process_directory,
    ParserConfig, LawMetadata, SectionContent, Section, Law
)


@pytest.fixture
def sample_xml():
    """Return a sample XML string for testing."""
    return """<?xml version="1.0" encoding="UTF-8" ?><!DOCTYPE dokumente SYSTEM "http://www.gesetze-im-internet.de/dtd/1.01/gii-norm.dtd">
<dokumente builddate="20240101000000" doknr="BJNR010050934">
  <norm builddate="20240101000000" doknr="BJNR010050934">
    <metadaten>
      <jurabk>EStG</jurabk>
      <amtabk>EStG</amtabk>
      <ausfertigung-datum manuell="ja">1934-10-16</ausfertigung-datum>
      <fundstelle typ="amtlich"><periodikum>RGBl I</periodikum><zitstelle>1934, 1005</zitstelle></fundstelle>
      <langue>Einkommensteuergesetz</langue>
      <standangabe checked="ja"><standtyp>Neuf</standtyp><standkommentar>Neugefasst durch Bek. v. 8.10.2009 I 3366, 3862;</standkommentar></standangabe>
      <standangabe checked="ja"><standtyp>Stand</standtyp><standkommentar>zuletzt geändert durch Art. 2 G v. 23.12.2024 I Nr. 449</standkommentar></standangabe>
    </metadaten>
    <textdaten>
      <fussnoten>
        <Content>
          <P>Test content</P>
        </Content>
      </fussnoten>
    </textdaten>
  </norm>
  <norm builddate="20240101000000" doknr="BJNR010050934_section_1">
    <metadaten>
      <jurabk>EStG</jurabk>
      <enbez>§ 1</enbez>
      <titel>Steuerpflicht</titel>
    </metadaten>
    <textdaten>
      <text format="XML">
        <P>Unbeschränkt einkommensteuerpflichtig sind:</P>
        <DL>
          <DT>1.</DT>
          <DD>natürliche Personen, die im Inland einen Wohnsitz oder ihren gewöhnlichen Aufenthalt haben.</DD>
        </DL>
      </text>
    </textdaten>
  </norm>
  <norm builddate="20240101000000" doknr="BJNR010050934_section_2">
    <metadaten>
      <jurabk>EStG</jurabk>
      <enbez>§ 2</enbez>
      <titel>Umfang der Besteuerung</titel>
    </metadaten>
    <textdaten>
      <text format="XML">
        <P>
          Die Einkommensteuer bemisst sich nach dem Einkommen.
          <table frame="none">
            <tgroup cols="2">
              <tbody>
                <row>
                  <entry>Spalte 1</entry>
                  <entry>Spalte 2</entry>
                </row>
              </tbody>
            </tgroup>
          </table>
        </P>
      </text>
    </textdaten>
  </norm>
</dokumente>
"""


@pytest.fixture
def parser_config():
    """Return a default parser configuration."""
    return ParserConfig()


@pytest.fixture
def xml_element(sample_xml):
    """Return a parsed XML element."""
    return etree.fromstring(sample_xml.encode('utf-8'))


def test_clean_text():
    """Test cleaning and normalizing text."""
    # Test normalization
    text = "This  is a   test   with    multiple spaces."
    config = ParserConfig(normalize_whitespace=True)
    cleaned = clean_text(text, config)
    assert "  " not in cleaned
    assert cleaned == "This is a test with multiple spaces."
    
    # Test without normalization
    config = ParserConfig(normalize_whitespace=False)
    cleaned = clean_text(text, config)
    assert cleaned == text.strip()
    
    # Test with max length
    config = ParserConfig(max_text_length=10)
    cleaned = clean_text(text, config)
    assert len(cleaned) == 10
    assert cleaned.endswith("...")


def test_extract_tables(xml_element):
    """Test extracting tables from XML."""
    # Find an element with a table
    element = xml_element.find(".//norm[@doknr='BJNR010050934_section_2']//text")
    tables = extract_tables(element)
    
    assert len(tables) == 1
    assert "<table" in tables[0]
    assert "<entry>Spalte 1</entry>" in tables[0]


def test_element_to_text(xml_element, parser_config):
    """Test converting an XML element to plain text."""
    # Test with a simple paragraph
    element = xml_element.find(".//norm[@doknr='BJNR010050934_section_1']//P")
    text = element_to_text(element, parser_config)
    assert text == "Unbeschränkt einkommensteuerpflichtig sind:"
    
    # Test with a more complex element
    element = xml_element.find(".//norm[@doknr='BJNR010050934_section_1']//text")
    text = element_to_text(element, parser_config)
    assert "Unbeschränkt einkommensteuerpflichtig sind:" in text
    assert "1. natürliche Personen" in text


def test_element_to_html(xml_element, parser_config):
    """Test converting an XML element to HTML."""
    element = xml_element.find(".//norm[@doknr='BJNR010050934_section_1']//text")
    html = element_to_html(element, parser_config)
    
    assert "<P>" in html
    assert "<DL>" in html
    assert "<DT>1.</DT>" in html
    assert "natürliche Personen" in html


def test_extract_law_metadata(xml_element):
    """Test extracting law metadata from XML."""
    metadata = extract_law_metadata(xml_element)
    
    assert isinstance(metadata, LawMetadata)
    assert metadata.law_id == "BJNR010050934"
    assert metadata.abbreviation == "EStG"
    assert metadata.full_title == "Einkommensteuergesetz"
    assert metadata.issue_date == "1934-10-16"
    assert metadata.publication_info == "RGBl I 1934, 1005"
    assert "geändert durch Art. 2 G v. 23.12.2024" in metadata.last_changed


def test_extract_section_content(xml_element, parser_config):
    """Test extracting section content from XML."""
    element = xml_element.find(".//norm[@doknr='BJNR010050934_section_1']/textdaten")
    content = extract_section_content(element, parser_config)
    
    assert isinstance(content, SectionContent)
    assert "Unbeschränkt einkommensteuerpflichtig sind" in content.text
    assert content.html is not None
    assert "<P>" in content.html
    
    # Test with tables
    element = xml_element.find(".//norm[@doknr='BJNR010050934_section_2']/textdaten")
    content = extract_section_content(element, parser_config)
    
    assert "Die Einkommensteuer bemisst sich nach dem Einkommen" in content.text
    assert content.tables is not None
    assert len(content.tables) == 1
    assert "<table" in content.tables[0]


def test_parse_section(xml_element, parser_config):
    """Test parsing a section from XML."""
    element = xml_element.find(".//norm[@doknr='BJNR010050934_section_1']")
    section = parse_section(
        element=element,
        law_id="BJNR010050934",
        parent_id=None,
        level=1,
        order_index=1,
        config=parser_config
    )
    
    assert isinstance(section, Section)
    assert section.section_id == "BJNR010050934_section_1"
    assert section.law_id == "BJNR010050934"
    assert section.section_number == "§ 1"
    assert section.title == "Steuerpflicht"
    assert "Unbeschränkt einkommensteuerpflichtig sind" in section.content.text
    assert section.level == 1
    assert section.order_index == 1


def test_extract_sections(xml_element, parser_config):
    """Test extracting all sections from XML."""
    sections = extract_sections(xml_element, "BJNR010050934", parser_config)
    
    assert len(sections) == 2  # We should have 2 sections (§1 and §2), skipping the first norm
    assert sections[0].section_number == "§ 1"
    assert sections[1].section_number == "§ 2"
    assert "Umfang der Besteuerung" in sections[1].title


def test_parse_xml(sample_xml, parser_config):
    """Test parsing a complete XML document."""
    # Convert to bytes since lxml doesn't accept unicode strings with encoding declarations
    law = parse_xml(sample_xml.encode('utf-8'), parser_config)
    
    assert isinstance(law, Law)
    assert law.metadata.abbreviation == "EStG"
    assert len(law.sections) == 2
    assert law.sections[0].section_number == "§ 1"
    assert law.sections[1].section_number == "§ 2"


@patch("builtins.open", new_callable=mock_open, read_data=b"<sample_xml/>")
@patch("taxpilot.backend.data_processing.xml_parser.parse_xml")
def test_parse_xml_file(mock_parse_xml, mock_file):
    """Test parsing an XML file."""
    mock_parse_xml.return_value = Law(
        metadata=LawMetadata(
            law_id="TEST",
            abbreviation="TEST",
            full_title="Test Law",
            issue_date="2023-01-01"
        ),
        sections=[]
    )
    
    law = parse_xml_file(Path("/test/path.xml"))
    
    assert isinstance(law, Law)
    assert law.metadata.law_id == "TEST"
    mock_file.assert_called_once_with(Path("/test/path.xml"), 'rb')


@patch("zipfile.ZipFile")
@patch("taxpilot.backend.data_processing.xml_parser.parse_xml")
def test_parse_zip_file(mock_parse_xml, mock_zipfile):
    """Test parsing a ZIP file containing XML."""
    # Set up mock
    mock_zip_instance = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = mock_zip_instance
    mock_zip_instance.namelist.return_value = ["file.xml"]
    
    mock_xml_file = MagicMock()
    mock_zip_instance.open.return_value.__enter__.return_value = mock_xml_file
    mock_xml_file.read.return_value = b"<sample_xml/>"
    
    mock_parse_xml.return_value = Law(
        metadata=LawMetadata(
            law_id="TEST",
            abbreviation="TEST",
            full_title="Test Law",
            issue_date="2023-01-01"
        ),
        sections=[]
    )
    
    law = parse_zip_file(Path("/test/path.zip"))
    
    assert isinstance(law, Law)
    assert law.metadata.law_id == "TEST"
    mock_zipfile.assert_called_once_with(Path("/test/path.zip"), 'r')


@patch("taxpilot.backend.data_processing.xml_parser.parse_xml_file")
@patch("taxpilot.backend.data_processing.xml_parser.parse_zip_file")
def test_process_law_file(mock_parse_zip, mock_parse_xml):
    """Test processing various file types."""
    # Set up mocks
    mock_law = Law(
        metadata=LawMetadata(
            law_id="TEST",
            abbreviation="TEST",
            full_title="Test Law",
            issue_date="2023-01-01"
        ),
        sections=[]
    )
    mock_parse_xml.return_value = mock_law
    mock_parse_zip.return_value = mock_law
    
    # Test XML file
    law = process_law_file(Path("/test/path.xml"))
    assert law.metadata.law_id == "TEST"
    mock_parse_xml.assert_called_once()
    
    # Test ZIP file
    mock_parse_xml.reset_mock()
    law = process_law_file(Path("/test/path.zip"))
    assert law.metadata.law_id == "TEST"
    mock_parse_zip.assert_called_once()
    
    # Test unsupported file
    with pytest.raises(ValueError):
        process_law_file(Path("/test/path.txt"))


@patch("taxpilot.backend.data_processing.xml_parser.process_law_file")
@patch("pathlib.Path.glob")
def test_process_directory(mock_glob, mock_process_file):
    """Test processing a directory of law files."""
    # Set up mocks - prevent duplicate files from multiple globs
    mock_glob.side_effect = lambda pattern: [
        Path("/test/law1.xml")
    ] if pattern.endswith(".xml") else (
        [Path("/test/law2.zip")] if pattern.endswith(".zip") else []
    )
    
    mock_process_file.side_effect = [
        Law(
            metadata=LawMetadata(
                law_id="LAW1",
                abbreviation="L1",
                full_title="Law One",
                issue_date="2023-01-01"
            ),
            sections=[]
        ),
        Law(
            metadata=LawMetadata(
                law_id="LAW2",
                abbreviation="L2",
                full_title="Law Two",
                issue_date="2023-01-01"
            ),
            sections=[]
        )
    ]
    
    laws = process_directory(Path("/test"))
    
    assert len(laws) == 2
    assert "LAW1" in laws
    assert "LAW2" in laws
    assert mock_process_file.call_count == 2