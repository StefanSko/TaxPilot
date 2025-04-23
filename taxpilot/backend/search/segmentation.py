"""
Text segmentation module for legal documents.

This module provides functionality to segment legal texts into chunks
appropriate for vector embeddings, considering semantic boundaries
and preserving context.
"""

import re
import unicodedata
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class SegmentationStrategy(Enum):
    """Strategies for text segmentation."""
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class SegmentType(Enum):
    """Types of segments in legal documents."""
    ARTICLE = "article"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SUBSECTION = "subsection"
    SENTENCE = "sentence"


@dataclass
class TextSegment:
    """A segment of text with metadata about its source and hierarchical position."""
    text: str
    law_id: str
    section_id: str
    segment_id: str
    start_idx: int
    end_idx: int
    article_id: str = ""  # ID of the article this segment belongs to
    hierarchy_path: str = ""  # Full path in the legal hierarchy (e.g., "estg/§13/abs2/satz1")
    segment_type: SegmentType = SegmentType.PARAGRAPH  # Type of this segment
    position_in_parent: int = 0  # Position within parent (e.g., paragraph number)
    metadata: dict[str, Any] = None  # Additional metadata
    
    def __post_init__(self):
        """Initialize metadata if not provided"""
        if self.metadata is None:
            self.metadata = {}
            
        # Ensure hierarchy information is included in metadata
        self.metadata["article_id"] = self.article_id
        self.metadata["hierarchy_path"] = self.hierarchy_path
        self.metadata["segment_type"] = self.segment_type.value if isinstance(self.segment_type, Enum) else self.segment_type
        self.metadata["position_in_parent"] = self.position_in_parent


class SegmentationConfig:
    """Configuration for text segmentation."""
    
    def __init__(
        self,
        strategy: SegmentationStrategy = SegmentationStrategy.PARAGRAPH,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        max_chunk_size: int = 1024,
        include_metadata: bool = True,
        clean_text: bool = True,
        preserve_references: bool = True,
        build_hierarchy: bool = True,  # Whether to build hierarchical structure
        detect_articles: bool = True,  # Whether to detect articles in sections
        identify_subsections: bool = True,  # Whether to identify subsections
        link_to_parent_article: bool = True,  # Whether to link segments to parent articles
        extract_section_numbers: bool = True,  # Whether to extract section numbers from text
    ):
        """
        Initialize the segmentation configuration.
        
        Args:
            strategy: Segmentation strategy to use
            chunk_size: Target size for text chunks in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size to keep
            max_chunk_size: Maximum chunk size allowed
            include_metadata: Whether to include metadata in segments
            clean_text: Whether to clean and normalize text
            preserve_references: Whether to preserve legal references
            build_hierarchy: Whether to build hierarchical structure
            detect_articles: Whether to detect articles in sections
            identify_subsections: Whether to identify subsections
            link_to_parent_article: Whether to link segments to parent articles
            extract_section_numbers: Whether to extract section numbers from text
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.include_metadata = include_metadata
        self.clean_text = clean_text
        self.preserve_references = preserve_references
        self.build_hierarchy = build_hierarchy
        self.detect_articles = detect_articles
        self.identify_subsections = identify_subsections
        self.link_to_parent_article = link_to_parent_article
        self.extract_section_numbers = extract_section_numbers


def clean_and_normalize_text(text: str) -> str:
    """
    Clean and normalize legal text for better segmentation.
    
    Args:
        text: The input text to clean
        
    Returns:
        Cleaned and normalized text
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Remove excessive newlines but preserve paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Fix common OCR/formatting errors in German legal texts
    text = text.replace('§ l', '§ 1')
    text = text.replace('§l', '§1')
    
    # Ensure consistent handling of section symbols
    text = re.sub(r'§\s*(\d+)', r'§ \1', text)
    
    # Clean up citation references
    text = re.sub(r'\(\s*vgl\.', r'(vgl.', text)
    
    # Handle hyphenation at line breaks
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    
    return text.strip()


def handle_special_formatting(text: str) -> str:
    """
    Handle special legal text formatting.
    
    Args:
        text: The input text with potential special formatting
        
    Returns:
        Text with properly handled special formatting
    """
    # Handle enumeration and indentation
    text = re.sub(r'^(\s*\d+\.\s+)', r'\n\1', text, flags=re.MULTILINE)
    
    # Handle lettered lists
    text = re.sub(r'^(\s*[a-zäöüß]\)\s+)', r'\n\1', text, flags=re.MULTILINE)
    
    # Handle section and paragraph references
    text = re.sub(r'(§+\s*\d+[a-z]?(\s*Abs\.\s*\d+)?(\s*Satz\s*\d+)?)', r' \1 ', text)
    
    # Handle law abbreviations
    text = re.sub(r'(\b[A-ZÄÖÜ]{2,}\b)', r' \1 ', text)
    
    # Remove excessive spaces that might have been introduced
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()


def segment_by_section(text: str, law_id: str, section_id: str, config: SegmentationConfig) -> list[TextSegment]:
    """
    Segment text at the section level.
    
    Args:
        text: The legal text to segment
        law_id: The ID of the law
        section_id: The ID of the section
        config: Segmentation configuration
        
    Returns:
        List of text segments at section level
    """
    if config.clean_text:
        text = clean_and_normalize_text(text)
        text = handle_special_formatting(text)
    
    # Extract article information if configured
    article_id = section_id
    hierarchy_path = f"{law_id}/{section_id.split('_')[-1]}" if '_' in section_id else f"{law_id}"
    segment_type = SegmentType.SECTION
    
    if config.build_hierarchy:
        # Extract structural information
        structure = extract_legal_structure(text)
        
        # Determine if this is an article
        if structure["is_article"]:
            segment_type = SegmentType.ARTICLE
            
        # Get article ID if possible
        if config.detect_articles and structure["article_number"]:
            article_id = derive_article_id(section_id, structure["article_number"])
        
        # Build hierarchical path
        if config.extract_section_numbers:
            hierarchy_path = build_hierarchy_path(law_id, structure)
    
    # For section-level, we keep the entire section as one segment
    return [
        TextSegment(
            text=text,
            law_id=law_id,
            section_id=section_id,
            segment_id=f"{section_id}_full",
            start_idx=0,
            end_idx=len(text),
            article_id=article_id,
            hierarchy_path=hierarchy_path,
            segment_type=segment_type,
            position_in_parent=0,
            metadata={
                "strategy": SegmentationStrategy.SECTION.value,
                "length": len(text)
            }
        )
    ]


def segment_by_paragraph(
    text: str, 
    law_id: str, 
    section_id: str, 
    config: SegmentationConfig,
    article_id: str = "",
    parent_path: str = ""
) -> list[TextSegment]:
    """
    Segment text at the paragraph level.
    
    Args:
        text: The legal text to segment
        law_id: The ID of the law
        section_id: The ID of the section
        config: Segmentation configuration
        article_id: ID of the parent article (optional)
        parent_path: Hierarchical path of the parent (optional)
        
    Returns:
        List of text segments at paragraph level
    """
    if config.clean_text:
        text = clean_and_normalize_text(text)
        text = handle_special_formatting(text)
    
    # Use provided article_id and parent_path if given, otherwise extract from text
    if not article_id:
        article_id = section_id
        
        if config.build_hierarchy:
            # Extract structural information
            section_structure = extract_legal_structure(text)
            
            # Get article ID if possible
            if config.detect_articles and section_structure["article_number"]:
                article_id = derive_article_id(section_id, section_structure["article_number"])
    
    if not parent_path:
        parent_path = f"{law_id}/{section_id.split('_')[-1]}" if '_' in section_id else f"{law_id}"
        
        if config.build_hierarchy and config.extract_section_numbers:
            # Extract structural information if not done already
            if not 'section_structure' in locals():
                section_structure = extract_legal_structure(text)
                
            # Build base hierarchical path
            parent_path = build_hierarchy_path(law_id, section_structure)
    
    # Split by paragraphs using blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    segments = []
    for i, para in enumerate(paragraphs):
        # Skip paragraphs that are too short
        if len(para) < config.min_chunk_size:
            continue
            
        # Extract paragraph-specific structure if configured
        paragraph_hierarchy_path = f"{parent_path}/p{i+1}"
        paragraph_type = SegmentType.PARAGRAPH
        
        if config.build_hierarchy and config.identify_subsections:
            para_structure = extract_legal_structure(para)
            
            # Check if this paragraph is a subsection
            if para_structure["subsections"]:
                paragraph_type = SegmentType.SUBSECTION
                
            # Extend the hierarchical path if subsection information is found
            if config.extract_section_numbers and para_structure["hierarchy_components"]:
                # Add paragraph index if no specific subsection was found
                if not para_structure["hierarchy_components"]:
                    para_structure["hierarchy_components"].append(f"p{i+1}")
                    
                paragraph_hierarchy_path = f"{parent_path}/{'/'.join(para_structure['hierarchy_components'])}"
        
        # Handle paragraphs that are too long by splitting them further
        if len(para) > config.max_chunk_size:
            # Recursively segment this paragraph using sentence-level segmentation
            # Pass the hierarchical information down to sentence segmentation
            sentence_config = config
            
            para_segments = segment_by_sentence(
                para, law_id, section_id, sentence_config,
                article_id=article_id, 
                parent_path=paragraph_hierarchy_path
            )
            segments.extend(para_segments)
            continue
        
        segment = TextSegment(
            text=para,
            law_id=law_id,
            section_id=section_id,
            segment_id=f"{section_id}_p{i+1}",
            start_idx=text.find(para),
            end_idx=text.find(para) + len(para),
            article_id=article_id,
            hierarchy_path=paragraph_hierarchy_path,
            segment_type=paragraph_type,
            position_in_parent=i+1,
            metadata={
                "strategy": SegmentationStrategy.PARAGRAPH.value,
                "paragraph_index": i,
                "length": len(para)
            }
        )
        segments.append(segment)
    
    return segments


def segment_by_sentence(
    text: str, 
    law_id: str, 
    section_id: str, 
    config: SegmentationConfig,
    article_id: str = "",
    parent_path: str = ""
) -> list[TextSegment]:
    """
    Segment text at the sentence level.
    
    Args:
        text: The legal text to segment
        law_id: The ID of the law
        section_id: The ID of the section
        config: Segmentation configuration
        article_id: ID of the parent article (optional)
        parent_path: Hierarchical path of the parent (optional)
        
    Returns:
        List of text segments at sentence level
    """
    if config.clean_text:
        text = clean_and_normalize_text(text)
        text = handle_special_formatting(text)
    
    # Set default values for hierarchical information if not provided
    if not article_id and config.link_to_parent_article:
        article_id = section_id
        
        if config.build_hierarchy and config.detect_articles:
            # Try to extract article number from text
            section_structure = extract_legal_structure(text)
            if section_structure["article_number"]:
                article_id = derive_article_id(section_id, section_structure["article_number"])
    
    if not parent_path:
        parent_path = f"{law_id}/{section_id.split('_')[-1]}" if '_' in section_id else f"{law_id}"
    
    # Complex regex for German legal sentence splitting
    # Matches end of sentences while handling abbreviations, numbers, etc.
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-ZÄÖÜ][a-zäöü]\.)(?<=\.|\?|\!)\s+(?=[A-ZÄÖÜ§\d])'
    
    # Split into raw sentences
    raw_sentences = re.split(sentence_pattern, text)
    
    # Process and group sentences to meet target chunk size
    segments = []
    current_chunk = ""
    sentence_indices: list[int] = []
    start_sentence_idx = 0
    
    for i, sentence in enumerate(raw_sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Get position in original text
        start_idx = text.find(sentence, 0 if not sentence_indices else sentence_indices[-1])
        if start_idx == -1:  # Fallback if exact match isn't found
            start_idx = 0 if not sentence_indices else sentence_indices[-1]
        
        sentence_indices.append(start_idx)
        
        # If adding this sentence would exceed max_chunk_size, finalize current chunk
        if len(current_chunk) + len(sentence) > config.max_chunk_size and current_chunk:
            # Build the hierarchical path for this sentence group
            hierarchy_path = f"{parent_path}/s{start_sentence_idx+1}-{i}"
            
            segment = TextSegment(
                text=current_chunk,
                law_id=law_id,
                section_id=section_id,
                segment_id=f"{section_id}_s{start_sentence_idx+1}",
                start_idx=sentence_indices[0],
                end_idx=sentence_indices[-1],
                article_id=article_id,
                hierarchy_path=hierarchy_path,
                segment_type=SegmentType.SENTENCE,
                position_in_parent=start_sentence_idx+1,
                metadata={
                    "strategy": SegmentationStrategy.SENTENCE.value,
                    "sentence_count": len(sentence_indices),
                    "sentence_range": f"{start_sentence_idx+1}-{i}",
                    "length": len(current_chunk)
                }
            )
            segments.append(segment)
            
            # Update the start index for the next chunk
            start_sentence_idx = i
            
            # Start a new chunk with overlap if configured
            if config.chunk_overlap > 0 and len(sentence_indices) > 1:
                # Find sentences that should be included in the overlap
                overlap_size = 0
                overlap_sentences = []
                overlapped_indices = []
                for s_idx in range(len(sentence_indices) - 1, -1, -1):
                    s = raw_sentences[s_idx].strip()
                    if overlap_size + len(s) > config.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlapped_indices.insert(0, sentence_indices[s_idx])
                    overlap_size += len(s)
                
                current_chunk = " ".join(overlap_sentences)
                sentence_indices = overlapped_indices
                start_sentence_idx = i - len(overlap_sentences)
            else:
                current_chunk = ""
                sentence_indices = []
        
        # Add the current sentence to the chunk
        if current_chunk:
            current_chunk += " " + sentence
        else:
            current_chunk = sentence
    
    # Add the last chunk if it's not empty and meets minimum size
    if current_chunk and len(current_chunk) >= config.min_chunk_size:
        # Build the hierarchical path for the last sentence group
        hierarchy_path = f"{parent_path}/s{start_sentence_idx+1}-{len(raw_sentences)}"
        
        segment = TextSegment(
            text=current_chunk,
            law_id=law_id,
            section_id=section_id,
            segment_id=f"{section_id}_s{start_sentence_idx+1}",
            start_idx=sentence_indices[0] if sentence_indices else 0,
            end_idx=sentence_indices[-1] + len(raw_sentences[-1]) if sentence_indices else len(current_chunk),
            article_id=article_id,
            hierarchy_path=hierarchy_path,
            segment_type=SegmentType.SENTENCE,
            position_in_parent=start_sentence_idx+1,
            metadata={
                "strategy": SegmentationStrategy.SENTENCE.value,
                "sentence_count": len(sentence_indices),
                "sentence_range": f"{start_sentence_idx+1}-{len(raw_sentences)}",
                "length": len(current_chunk)
            }
        )
        segments.append(segment)
    
    return segments


def segment_with_overlapping_windows(
    text: str, 
    law_id: str, 
    section_id: str, 
    config: SegmentationConfig,
    article_id: str = "",
    parent_path: str = ""
) -> list[TextSegment]:
    """
    Segment text using overlapping fixed-size windows.
    
    Args:
        text: The legal text to segment
        law_id: The ID of the law
        section_id: The ID of the section
        config: Segmentation configuration
        article_id: ID of the parent article (optional)
        parent_path: Hierarchical path of the parent (optional)
        
    Returns:
        List of text segments with overlapping windows
    """
    if config.clean_text:
        text = clean_and_normalize_text(text)
        text = handle_special_formatting(text)
    
    # Set default values for hierarchical information if not provided
    if not article_id and config.link_to_parent_article:
        article_id = section_id
        
        if config.build_hierarchy and config.detect_articles:
            # Try to extract article number from text
            section_structure = extract_legal_structure(text)
            if section_structure["article_number"]:
                article_id = derive_article_id(section_id, section_structure["article_number"])
    
    if not parent_path:
        parent_path = f"{law_id}/{section_id.split('_')[-1]}" if '_' in section_id else f"{law_id}"
    
    segments = []
    text_length = len(text)
    
    # If text is shorter than min_chunk_size, return it as a single segment
    if text_length < config.min_chunk_size:
        hierarchy_path = f"{parent_path}/window1"
        
        return [
            TextSegment(
                text=text,
                law_id=law_id,
                section_id=section_id,
                segment_id=f"{section_id}_window1",
                start_idx=0,
                end_idx=text_length,
                article_id=article_id,
                hierarchy_path=hierarchy_path,
                segment_type=SegmentType.PARAGRAPH,
                position_in_parent=1,
                metadata={
                    "strategy": "window",
                    "window_index": 0,
                    "length": text_length
                }
            )
        ]
    
    # Create overlapping windows
    start_idx = 0
    window_idx = 0
    
    while start_idx < text_length:
        # Calculate end index for this window
        end_idx = min(start_idx + config.chunk_size, text_length)
        
        # Adjust to avoid cutting in the middle of words
        if end_idx < text_length:
            next_space = text.find(' ', end_idx)
            if next_space != -1 and next_space - end_idx < 20:  # Limit adjustment to 20 chars
                end_idx = next_space
        
        chunk = text[start_idx:end_idx].strip()
        
        # Only add chunks that meet minimum size
        if len(chunk) >= config.min_chunk_size:
            hierarchy_path = f"{parent_path}/window{window_idx+1}"
            
            segment = TextSegment(
                text=chunk,
                law_id=law_id,
                section_id=section_id,
                segment_id=f"{section_id}_window{window_idx+1}",
                start_idx=start_idx,
                end_idx=end_idx,
                article_id=article_id,
                hierarchy_path=hierarchy_path,
                segment_type=SegmentType.PARAGRAPH,
                position_in_parent=window_idx+1,
                metadata={
                    "strategy": "window",
                    "window_index": window_idx,
                    "length": len(chunk)
                }
            )
            segments.append(segment)
        
        # Move to next position with overlap
        start_idx += config.chunk_size - config.chunk_overlap
        window_idx += 1
        
        # Ensure we're making progress
        if start_idx <= 0:
            start_idx = config.chunk_size
    
    return segments


def segment_text(text: str, law_id: str, section_id: str, config: SegmentationConfig) -> list[TextSegment]:
    """
    Segment text according to the configured strategy.
    
    Args:
        text: The legal text to segment
        law_id: The ID of the law
        section_id: The ID of the section
        config: Segmentation configuration
        
    Returns:
        List of text segments
    """
    if not text:
        return []
    
    # Pre-analyze text to extract article information if hierarchical features enabled
    article_id = ""
    parent_path = ""
    
    if config.build_hierarchy:
        # Extract structural information
        structure = extract_legal_structure(text)
        
        # Get article ID if possible
        if config.detect_articles and structure["article_number"]:
            article_id = derive_article_id(section_id, structure["article_number"])
        else:
            article_id = section_id
            
        # Build hierarchical path
        if config.extract_section_numbers:
            parent_path = build_hierarchy_path(law_id, structure)
        else:
            parent_path = f"{law_id}/{section_id.split('_')[-1]}" if '_' in section_id else f"{law_id}"
    
    # Apply the selected segmentation strategy
    if config.strategy == SegmentationStrategy.SECTION:
        return segment_by_section(text, law_id, section_id, config)
    elif config.strategy == SegmentationStrategy.PARAGRAPH:
        # Clone config and disable hierarchy building for segment_by_paragraph
        # to avoid duplicate hierarchy detection
        paragraph_config = config
        if config.build_hierarchy:
            # Create a copy of the config with hierarchy detection disabled
            paragraph_config = SegmentationConfig(
                strategy=config.strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                min_chunk_size=config.min_chunk_size,
                max_chunk_size=config.max_chunk_size,
                include_metadata=config.include_metadata,
                clean_text=config.clean_text,
                preserve_references=config.preserve_references,
                build_hierarchy=False,  # Disable duplicate hierarchy detection
                detect_articles=False,
                identify_subsections=False,
                link_to_parent_article=True,
                extract_section_numbers=False
            )
        
        return segment_by_paragraph(
            text, law_id, section_id, paragraph_config,
            article_id=article_id,
            parent_path=parent_path
        )
    elif config.strategy == SegmentationStrategy.SENTENCE:
        # Clone config and disable hierarchy building for segment_by_sentence
        sentence_config = config
        if config.build_hierarchy:
            # Create a copy of the config with hierarchy detection disabled
            sentence_config = SegmentationConfig(
                strategy=config.strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                min_chunk_size=config.min_chunk_size,
                max_chunk_size=config.max_chunk_size,
                include_metadata=config.include_metadata,
                clean_text=config.clean_text,
                preserve_references=config.preserve_references,
                build_hierarchy=False,  # Disable duplicate hierarchy detection
                detect_articles=False,
                identify_subsections=False,
                link_to_parent_article=True,
                extract_section_numbers=False
            )
        
        return segment_by_sentence(
            text, law_id, section_id, sentence_config, 
            article_id=article_id, 
            parent_path=parent_path
        )
    else:
        # Default to overlapping windows
        # Clone config and disable hierarchy building
        window_config = config
        if config.build_hierarchy:
            # Create a copy of the config with hierarchy detection disabled
            window_config = SegmentationConfig(
                strategy=config.strategy,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                min_chunk_size=config.min_chunk_size,
                max_chunk_size=config.max_chunk_size,
                include_metadata=config.include_metadata,
                clean_text=config.clean_text,
                preserve_references=config.preserve_references,
                build_hierarchy=False,  # Disable duplicate hierarchy detection
                detect_articles=False,
                identify_subsections=False,
                link_to_parent_article=True,
                extract_section_numbers=False
            )
            
        return segment_with_overlapping_windows(
            text, law_id, section_id, window_config,
            article_id=article_id,
            parent_path=parent_path
        )


def extract_legal_structure(text: str) -> dict[str, Any]:
    """
    Extract legal structure information from text.
    
    Identifies article numbers, paragraphs, and subsections in German legal texts.
    
    Args:
        text: Legal text to analyze
        
    Returns:
        Dictionary containing extracted structure information
    """
    # Initialize result structure
    structure = {
        "article_number": None,
        "section_numbers": [],
        "subsections": [],
        "paragraphs": [],
        "is_article": False,
        "hierarchy_components": []
    }
    
    # Look for section symbol followed by number (§ 1, §1, § 1a)
    section_pattern = r'§\s*(\d+[a-z]?)'
    section_matches = re.findall(section_pattern, text[:200])  # Check first 200 chars
    
    if section_matches:
        structure["article_number"] = section_matches[0]
        structure["is_article"] = True
        structure["hierarchy_components"].append(f"§{section_matches[0]}")
    
    # Look for subsection numbers in parentheses at the beginning of the text (e.g., "(2) Zu den Einkünften...")
    # This is common in German legal texts for indicating subsections
    subsection_pattern = r'^\s*\(\s*(\d+[a-z]?)\s*\)'
    subsection_matches = re.findall(subsection_pattern, text, re.MULTILINE)
    
    if subsection_matches:
        structure["subsections"] = subsection_matches
        structure["hierarchy_components"].extend([f"abs{num}" for num in subsection_matches])
    
    # Look for "Absatz" or "Abs." followed by number
    absatz_pattern = r'(?:Absatz|Abs\.)\s*(\d+[a-z]?)'
    absatz_matches = re.findall(absatz_pattern, text)
    
    if absatz_matches:
        # Add to existing subsections
        structure["subsections"].extend(absatz_matches)
        structure["hierarchy_components"].extend([f"abs{num}" for num in absatz_matches])
    
    # Look for "Satz" followed by number
    satz_pattern = r'Satz\s*(\d+)'
    satz_matches = re.findall(satz_pattern, text)
    
    if satz_matches:
        structure["sentences"] = satz_matches
        structure["hierarchy_components"].extend([f"satz{num}" for num in satz_matches])
    
    # Look for numbered lists like (1), (2), etc.
    # But not at the beginning of a line, as those are likely subsections
    list_pattern = r'(?<!^)\s+\(\s*(\d+)\s*\)'  # Negative lookbehind for start of line
    list_matches = re.findall(list_pattern, text, re.MULTILINE)
    
    if list_matches:
        structure["list_items"] = list_matches
        structure["hierarchy_components"].extend([f"nr{num}" for num in list_matches])
    
    return structure


def derive_article_id(section_id: str, article_number: str | None) -> str:
    """
    Derive an article ID from a section ID and article number.
    
    Args:
        section_id: Section ID (e.g., "estg_2")
        article_number: Article number (e.g., "2", "2a")
        
    Returns:
        Article ID (e.g., "estg_2")
    """
    # If no article number is found, use the section ID as the article ID
    if not article_number:
        return section_id
    
    # Extract law ID from section ID (assuming format like "estg_2")
    law_id = section_id.split('_')[0] if '_' in section_id else ""
    
    # Construct article ID
    return f"{law_id}_{article_number}"


def build_hierarchy_path(law_id: str, structure: dict[str, Any]) -> str:
    """
    Build a hierarchical path from structure components.
    
    Args:
        law_id: ID of the law
        structure: Structure information from extract_legal_structure
        
    Returns:
        Hierarchical path string (e.g., "estg/§2/abs3/satz2")
    """
    # Start with law ID
    path_components = [law_id]
    
    # Create a set to track added components to avoid duplicates
    added_components = set()
    
    # Add article number if present
    if structure["article_number"]:
        component = f"§{structure['article_number']}"
        if component not in added_components:
            path_components.append(component)
            added_components.add(component)
    
    # Add subsections (up to 2) if present
    for i, subsection in enumerate(structure.get("subsections", [])[:2]):
        component = f"abs{subsection}"
        if component not in added_components:
            path_components.append(component)
            added_components.add(component)
    
    # Add numbered items (up to 2) if present
    for i, item in enumerate(structure.get("list_items", [])[:2]):
        component = f"nr{item}"
        if component not in added_components:
            path_components.append(component)
            added_components.add(component)
    
    # Add sentences (up to 2) if present
    for i, sentence in enumerate(structure.get("sentences", [])[:2]):
        component = f"satz{sentence}"
        if component not in added_components:
            path_components.append(component)
            added_components.add(component)
    
    # Join with slashes
    return "/".join(path_components)


def optimize_chunk_size(model_max_tokens: int, token_to_char_ratio: float = 4.0) -> int:
    """
    Calculate optimal chunk size based on model token limits.
    
    Args:
        model_max_tokens: Maximum token limit for the target model
        token_to_char_ratio: Estimated ratio of tokens to characters (default: 4 chars ≈ 1 token)
        
    Returns:
        Optimal character count for chunks
    """
    # Reserve tokens for metadata and model overhead (prompt tokens, etc.)
    usable_tokens = int(model_max_tokens * 0.9)  # 90% of max tokens
    
    # Convert tokens to approximate character count
    optimal_chars = int(usable_tokens * token_to_char_ratio)
    
    return optimal_chars