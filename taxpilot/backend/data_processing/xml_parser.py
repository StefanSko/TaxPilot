"""
XML parser module for German legal documents.

This module provides functionality to parse German law XML files
downloaded from gesetze-im-internet.de according to the gii-norm.dtd structure.
"""

import logging
import re
from typing import Self, TypedDict, NotRequired, Literal, cast
from pathlib import Path
from datetime import datetime
import zipfile
from pydantic import BaseModel, Field, ConfigDict
from lxml import etree


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("xml_parser.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("xml_parser")


# Type definitions
class LawMetadataDict(TypedDict):
    """Dictionary representation of law metadata."""
    law_id: str
    abbreviation: str
    full_title: str
    issue_date: str
    publication_info: NotRequired[str]
    last_changed: NotRequired[str]
    status_info: NotRequired[str]


class SectionContentDict(TypedDict):
    """Dictionary representation of section content."""
    text: str
    html: NotRequired[str]
    tables: NotRequired[list[str]]


class SectionDict(TypedDict):
    """Dictionary representation of a law section."""
    section_id: str
    law_id: str
    section_number: str
    title: NotRequired[str]
    content: SectionContentDict
    parent_id: NotRequired[str]
    level: int
    order_index: int


# Pydantic models
class LawMetadata(BaseModel):
    """Metadata about a German law."""
    law_id: str = Field(description="Unique ID for the law")
    abbreviation: str = Field(description="Official abbreviation of the law (e.g., 'EStG')")
    full_title: str = Field(description="Full title of the law (e.g., 'Einkommensteuergesetz')")
    issue_date: str = Field(description="Date when the law was issued")
    publication_info: str | None = Field(
        default=None, description="Publication information (e.g., 'BGBl I S. 3366')")
    last_changed: str | None = Field(
        default=None, description="Date and reference of last change")
    status_info: str | None = Field(
        default=None, description="Status information about the law")

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True
    )


class SectionContent(BaseModel):
    """Content of a law section with both plain text and HTML versions."""
    text: str = Field(description="Plain text content of the section")
    html: str | None = Field(
        default=None, description="HTML formatted content of the section")
    tables: list[str] | None = Field(
        default=None, description="Tables contained in the section")

    model_config = ConfigDict(
        str_strip_whitespace=True
    )


class Section(BaseModel):
    """A section of a German law (e.g., a paragraph or article)."""
    section_id: str = Field(description="Unique ID for the section")
    law_id: str = Field(description="ID of the law this section belongs to")
    section_number: str = Field(description="Section number (e.g., '§ 1' or 'Art. 1')")
    title: str | None = Field(
        default=None, description="Title of the section")
    content: SectionContent = Field(description="Content of the section")
    parent_id: str | None = Field(
        default=None, description="ID of the parent section (for hierarchical structure)")
    level: int = Field(
        default=1, description="Hierarchical level of the section (1 = top level)")
    order_index: int = Field(
        description="Index for ordering sections within their level")

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True
    )


class Law(BaseModel):
    """A complete German law with metadata and sections."""
    metadata: LawMetadata = Field(description="Metadata about the law")
    sections: list[Section] = Field(description="Sections of the law")

    model_config = ConfigDict(
        populate_by_name=True
    )


class ParserConfig(BaseModel):
    """Configuration for the XML parser."""
    normalize_whitespace: bool = Field(
        default=True, 
        description="Whether to normalize whitespace in text content"
    )
    strip_xml_comments: bool = Field(
        default=True,
        description="Whether to remove XML comments from parsed content"
    )
    include_html: bool = Field(
        default=True,
        description="Whether to include HTML formatted content"
    )
    max_text_length: int | None = Field(
        default=None,
        description="Maximum length for text content (None for unlimited)"
    )

    model_config = ConfigDict(
        populate_by_name=True
    )


def clean_text(text: str, config: ParserConfig) -> str:
    """
    Clean and normalize text content from XML.
    
    Args:
        text: The raw text from XML.
        config: Parser configuration.
        
    Returns:
        Cleaned text.
    """
    if config.normalize_whitespace:
        # Replace multiple whitespace with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove unnecessary spaces before punctuation
        text = re.sub(r'\s+([.,;:!?)])', r'\1', text)
        # Add a space after punctuation if not followed by a space
        text = re.sub(r'([.,;:!?(])(?!\s)', r'\1 ', text)
    
    text = text.strip()
    
    if config.max_text_length and len(text) > config.max_text_length:
        text = text[:config.max_text_length - 3] + "..."
        
    return text


def extract_tables(element: etree._Element) -> list[str]:
    """
    Extract tables from XML element.
    
    Args:
        element: The XML element containing tables.
        
    Returns:
        List of table HTML strings.
    """
    tables = []
    for table_elem in element.xpath('.//table'):
        # Convert the table to an HTML string
        table_html = etree.tostring(table_elem, encoding='unicode', pretty_print=True)
        tables.append(table_html)
    
    return tables


def element_to_text(element: etree._Element, config: ParserConfig) -> str:
    """
    Convert an XML element to plain text.
    
    Args:
        element: The XML element to convert.
        config: Parser configuration.
        
    Returns:
        Plain text content.
    """
    if element is None:
        return ""
    
    # Extract all text content recursively
    text_parts = []
    
    if element.text:
        text_parts.append(element.text)
    
    for child in element:
        # Process text content of this child
        if child.tag in ("BR", "br"):
            text_parts.append("\n")
        elif child.tag in ("P", "p"):
            child_text = element_to_text(child, config)
            if child_text:
                text_parts.append(child_text)
                text_parts.append("\n\n")
        else:
            child_text = element_to_text(child, config)
            if child_text:
                text_parts.append(child_text)
        
        # Add tail text (text after closing tag)
        if child.tail:
            text_parts.append(child.tail)
    
    text = "".join(text_parts)
    return clean_text(text, config)


def element_to_html(element: etree._Element, config: ParserConfig) -> str:
    """
    Convert an XML element to HTML.
    
    Args:
        element: The XML element to convert.
        config: Parser configuration.
        
    Returns:
        HTML content.
    """
    if element is None:
        return ""
    
    # Convert the element to HTML string
    html = etree.tostring(element, encoding='unicode', pretty_print=True)
    
    # Clean up HTML if needed
    if config.strip_xml_comments:
        html = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)
    
    return html


def extract_law_metadata(root: etree._Element) -> LawMetadata:
    """
    Extract law metadata from the XML root element.
    
    Args:
        root: The XML root element.
        
    Returns:
        LawMetadata object.
    """
    metadata_dict: dict[str, str | None] = {
        "law_id": "",
        "abbreviation": "",
        "full_title": "",
        "issue_date": "",
        "publication_info": None,
        "last_changed": None,
        "status_info": None
    }
    
    # Find the first norm element and its metadata
    norm = root.find(".//norm")
    if norm is None:
        raise ValueError("No 'norm' element found in XML")
    
    metadata_elem = norm.find("./metadaten")
    if metadata_elem is None:
        raise ValueError("No 'metadaten' element found in XML")
    
    # Basic metadata elements
    jurabk = metadata_elem.findtext("./jurabk")
    metadata_dict["law_id"] = norm.get("doknr", "") if norm.get("doknr") else ""
    metadata_dict["abbreviation"] = jurabk if jurabk else ""
    
    langue = metadata_elem.findtext("./langue")
    metadata_dict["full_title"] = langue if langue else ""
    
    # Issue date
    issue_date = metadata_elem.find("./ausfertigung-datum")
    if issue_date is not None:
        metadata_dict["issue_date"] = issue_date.text if issue_date.text else ""
    
    # Publication info
    fundstelle = metadata_elem.find("./fundstelle")
    if fundstelle is not None:
        periodikum = fundstelle.findtext("./periodikum", "")
        zitstelle = fundstelle.findtext("./zitstelle", "")
        if periodikum and zitstelle:
            metadata_dict["publication_info"] = f"{periodikum} {zitstelle}"
    
    # Last change and status
    stand_elements = metadata_elem.findall("./standangabe")
    for stand in stand_elements:
        standtyp = stand.findtext("./standtyp", "")
        standkommentar = stand.findtext("./standkommentar", "")
        
        if standtyp == "Stand" and standkommentar:
            metadata_dict["last_changed"] = standkommentar
        elif standkommentar:
            # Store any other status information
            if metadata_dict["status_info"] is None:
                metadata_dict["status_info"] = standkommentar
            else:
                metadata_dict["status_info"] += f"; {standkommentar}"
    
    return LawMetadata(**metadata_dict)


def extract_section_content(element: etree._Element, config: ParserConfig) -> SectionContent:
    """
    Extract section content from XML element.
    
    Args:
        element: The XML element containing the section content.
        config: Parser configuration.
        
    Returns:
        SectionContent object.
    """
    content_dict: dict[str, str | list[str] | None] = {
        "text": "",
        "html": None,
        "tables": None
    }
    
    # Find text content
    text_element = element.find(".//text")
    if text_element is None:
        text_element = element
    content_dict["text"] = element_to_text(text_element, config)
    
    # Extract tables if present
    tables = extract_tables(element)
    if tables:
        content_dict["tables"] = tables
    
    # Generate HTML if configured
    if config.include_html:
        content_dict["html"] = element_to_html(text_element, config)
    
    return SectionContent(**content_dict)


def parse_section(element: etree._Element, law_id: str, parent_id: str | None, 
                 level: int, order_index: int, config: ParserConfig) -> Section:
    """
    Parse a section element from XML.
    
    Args:
        element: The XML element for the section.
        law_id: ID of the law.
        parent_id: ID of the parent section.
        level: Hierarchical level of the section.
        order_index: Index for ordering within the level.
        config: Parser configuration.
        
    Returns:
        Section object.
    """
    # Extract section ID and metadata
    section_id = element.get("doknr", "") or f"{law_id}_section_{order_index}"
    
    metadata = element.find("./metadaten")
    
    # Extract section number
    section_number = ""
    enbez = metadata.findtext("./enbez") if metadata is not None else None
    gliederungseinheit = metadata.find("./gliederungseinheit") if metadata is not None else None
    
    if enbez:
        section_number = enbez
    elif gliederungseinheit is not None:
        gliederungskennung = gliederungseinheit.findtext("./gliederungskennzeichen")
        gliederungsbez = gliederungseinheit.findtext("./gliederungsbezzeichnung")
        if gliederungskennung:
            section_number = gliederungskennung
            if gliederungsbez:
                section_number += f" {gliederungsbez}"
    
    # Extract title if available
    title = None
    if metadata is not None:
        title_elem = metadata.find("./titel")
        if title_elem is not None:
            title = element_to_text(title_elem, config)
        else:
            title = metadata.findtext("./enbez") or metadata.findtext("./gliederungsbezzeichnung")
    
    # Extract content
    textdaten = element.find("./textdaten")
    content = extract_section_content(textdaten if textdaten is not None else element, config)
    
    # Create the section
    section = Section(
        section_id=section_id,
        law_id=law_id,
        section_number=section_number,
        title=title,
        content=content,
        parent_id=parent_id,
        level=level,
        order_index=order_index
    )
    
    return section


def extract_sections(root: etree._Element, law_id: str, config: ParserConfig) -> list[Section]:
    """
    Extract all sections from the XML document.
    
    Args:
        root: The XML root element.
        law_id: ID of the law.
        config: Parser configuration.
        
    Returns:
        List of Section objects.
    """
    sections: list[Section] = []
    order_index = 0
    
    # Find all norm elements
    norm_elements = root.findall(".//norm")
    
    for norm_elem in norm_elements:
        # Skip the first norm as it typically contains only metadata
        if order_index == 0 and norm_elem.find("./metadaten/jurabk") is not None:
            order_index += 1
            continue
        
        # Process regular section
        section = parse_section(
            element=norm_elem,
            law_id=law_id,
            parent_id=None,
            level=1,
            order_index=order_index,
            config=config
        )
        
        sections.append(section)
        order_index += 1
    
    return sections


def parse_xml(xml_content: str | bytes, config: ParserConfig | None = None) -> Law:
    """
    Parse the XML content into a Law object.
    
    Args:
        xml_content: The XML content to parse.
        config: Optional parser configuration.
        
    Returns:
        Law object.
    """
    if config is None:
        config = ParserConfig()
    
    try:
        # Parse XML
        parser = etree.XMLParser(recover=True, remove_comments=config.strip_xml_comments)
        root = etree.fromstring(xml_content, parser)
        
        # Extract metadata
        metadata = extract_law_metadata(root)
        
        # Extract sections
        sections = extract_sections(root, metadata.law_id, config)
        
        return Law(metadata=metadata, sections=sections)
        
    except Exception as e:
        logger.error(f"Error parsing XML: {e}", exc_info=True)
        raise


def parse_xml_file(file_path: Path, config: ParserConfig | None = None) -> Law:
    """
    Parse an XML file into a Law object.
    
    Args:
        file_path: Path to the XML file.
        config: Optional parser configuration.
        
    Returns:
        Law object.
    """
    try:
        with open(file_path, 'rb') as f:
            xml_content = f.read()
        return parse_xml(xml_content, config)
    
    except Exception as e:
        logger.error(f"Error reading or parsing XML file {file_path}: {e}", exc_info=True)
        raise


def parse_zip_file(zip_path: Path, config: ParserConfig | None = None) -> Law:
    """
    Extract and parse an XML file from a ZIP archive.
    
    Args:
        zip_path: Path to the ZIP file.
        config: Optional parser configuration.
        
    Returns:
        Law object.
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Find the XML file in the archive
            xml_files = [name for name in zip_file.namelist() if name.lower().endswith('.xml')]
            
            if not xml_files:
                raise ValueError(f"No XML files found in ZIP archive {zip_path}")
            
            # Use the first XML file (typically there's only one)
            xml_filename = xml_files[0]
            logger.info(f"Parsing XML file {xml_filename} from ZIP archive {zip_path}")
            
            # Extract and parse the XML
            with zip_file.open(xml_filename) as xml_file:
                xml_content = xml_file.read()
                return parse_xml(xml_content, config)
    
    except Exception as e:
        logger.error(f"Error extracting or parsing XML from ZIP file {zip_path}: {e}", exc_info=True)
        raise


def process_law_file(file_path: Path, config: ParserConfig | None = None) -> Law:
    """
    Process a law file (either XML or ZIP containing XML).
    
    Args:
        file_path: Path to the file.
        config: Optional parser configuration.
        
    Returns:
        Law object.
    """
    if config is None:
        config = ParserConfig()
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.xml':
        return parse_xml_file(file_path, config)
    elif suffix in ('.zip', '.gz'):
        return parse_zip_file(file_path, config)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .xml, .zip, or .gz")


def process_directory(directory: Path, config: ParserConfig | None = None) -> dict[str, Law]:
    """
    Process all law files in a directory.
    
    Args:
        directory: Directory containing law files.
        config: Optional parser configuration.
        
    Returns:
        Dictionary mapping law IDs to Law objects.
    """
    if config is None:
        config = ParserConfig()
    
    result: dict[str, Law] = {}
    
    # Process all XML and ZIP files in the directory
    xml_files = list(directory.glob("**/*.xml"))
    zip_files = list(directory.glob("**/*.zip"))
    
    for file_path in xml_files + zip_files:
        try:
            law = process_law_file(file_path, config)
            result[law.metadata.law_id] = law
            logger.info(f"Successfully processed {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    return result