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


class SegmentationStrategy(Enum):
    """Strategies for text segmentation."""
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


@dataclass
class TextSegment:
    """A segment of text with metadata about its source."""
    text: str
    law_id: str
    section_id: str
    segment_id: str
    start_idx: int
    end_idx: int
    metadata: dict[str, any]


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
        preserve_references: bool = True
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
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.include_metadata = include_metadata
        self.clean_text = clean_text
        self.preserve_references = preserve_references


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
    
    # For section-level, we keep the entire section as one segment
    return [
        TextSegment(
            text=text,
            law_id=law_id,
            section_id=section_id,
            segment_id=f"{section_id}_full",
            start_idx=0,
            end_idx=len(text),
            metadata={
                "strategy": SegmentationStrategy.SECTION.value,
                "length": len(text)
            }
        )
    ]


def segment_by_paragraph(text: str, law_id: str, section_id: str, config: SegmentationConfig) -> list[TextSegment]:
    """
    Segment text at the paragraph level.
    
    Args:
        text: The legal text to segment
        law_id: The ID of the law
        section_id: The ID of the section
        config: Segmentation configuration
        
    Returns:
        List of text segments at paragraph level
    """
    if config.clean_text:
        text = clean_and_normalize_text(text)
        text = handle_special_formatting(text)
    
    # Split by paragraphs using blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    segments = []
    for i, para in enumerate(paragraphs):
        # Skip paragraphs that are too short
        if len(para) < config.min_chunk_size:
            continue
            
        # Handle paragraphs that are too long by splitting them further
        if len(para) > config.max_chunk_size:
            # Recursively segment this paragraph using sentence-level segmentation
            para_segments = segment_by_sentence(
                para, law_id, section_id, config
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
            metadata={
                "strategy": SegmentationStrategy.PARAGRAPH.value,
                "paragraph_index": i,
                "length": len(para)
            }
        )
        segments.append(segment)
    
    return segments


def segment_by_sentence(text: str, law_id: str, section_id: str, config: SegmentationConfig) -> list[TextSegment]:
    """
    Segment text at the sentence level.
    
    Args:
        text: The legal text to segment
        law_id: The ID of the law
        section_id: The ID of the section
        config: Segmentation configuration
        
    Returns:
        List of text segments at sentence level
    """
    if config.clean_text:
        text = clean_and_normalize_text(text)
        text = handle_special_formatting(text)
    
    # Complex regex for German legal sentence splitting
    # Matches end of sentences while handling abbreviations, numbers, etc.
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-ZÄÖÜ][a-zäöü]\.)(?<=\.|\?|\!)\s+(?=[A-ZÄÖÜ§\d])'
    
    # Split into raw sentences
    raw_sentences = re.split(sentence_pattern, text)
    
    # Process and group sentences to meet target chunk size
    segments = []
    current_chunk = ""
    sentence_indices: list[int] = []
    
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
            segment = TextSegment(
                text=current_chunk,
                law_id=law_id,
                section_id=section_id,
                segment_id=f"{section_id}_s{i-len(sentence_indices)+1}",
                start_idx=sentence_indices[0],
                end_idx=sentence_indices[-1],
                metadata={
                    "strategy": SegmentationStrategy.SENTENCE.value,
                    "sentence_count": len(sentence_indices),
                    "length": len(current_chunk)
                }
            )
            segments.append(segment)
            
            # Start a new chunk with overlap if configured
            if config.chunk_overlap > 0 and len(sentence_indices) > 1:
                # Find sentences that should be included in the overlap
                overlap_size = 0
                overlap_sentences = []
                for s_idx in range(len(sentence_indices) - 1, -1, -1):
                    s = raw_sentences[s_idx].strip()
                    if overlap_size + len(s) > config.chunk_overlap:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_size += len(s)
                
                current_chunk = " ".join(overlap_sentences)
                sentence_indices = sentence_indices[-len(overlap_sentences):]
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
        segment = TextSegment(
            text=current_chunk,
            law_id=law_id,
            section_id=section_id,
            segment_id=f"{section_id}_s{len(raw_sentences)-len(sentence_indices)+1}",
            start_idx=sentence_indices[0] if sentence_indices else 0,
            end_idx=sentence_indices[-1] + len(raw_sentences[-1]) if sentence_indices else len(current_chunk),
            metadata={
                "strategy": SegmentationStrategy.SENTENCE.value,
                "sentence_count": len(sentence_indices),
                "length": len(current_chunk)
            }
        )
        segments.append(segment)
    
    return segments


def segment_with_overlapping_windows(text: str, law_id: str, section_id: str, config: SegmentationConfig) -> list[TextSegment]:
    """
    Segment text using overlapping fixed-size windows.
    
    Args:
        text: The legal text to segment
        law_id: The ID of the law
        section_id: The ID of the section
        config: Segmentation configuration
        
    Returns:
        List of text segments with overlapping windows
    """
    if config.clean_text:
        text = clean_and_normalize_text(text)
        text = handle_special_formatting(text)
    
    segments = []
    text_length = len(text)
    
    # If text is shorter than min_chunk_size, return it as a single segment
    if text_length < config.min_chunk_size:
        return [
            TextSegment(
                text=text,
                law_id=law_id,
                section_id=section_id,
                segment_id=f"{section_id}_window1",
                start_idx=0,
                end_idx=text_length,
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
            segment = TextSegment(
                text=chunk,
                law_id=law_id,
                section_id=section_id,
                segment_id=f"{section_id}_window{window_idx+1}",
                start_idx=start_idx,
                end_idx=end_idx,
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
    
    if config.strategy == SegmentationStrategy.SECTION:
        return segment_by_section(text, law_id, section_id, config)
    elif config.strategy == SegmentationStrategy.PARAGRAPH:
        return segment_by_paragraph(text, law_id, section_id, config)
    elif config.strategy == SegmentationStrategy.SENTENCE:
        return segment_by_sentence(text, law_id, section_id, config)
    else:
        # Default to overlapping windows
        return segment_with_overlapping_windows(text, law_id, section_id, config)


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