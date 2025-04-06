"""
Utility functions for the PDF processing pipeline.
"""

import os
import re
import uuid
from typing import List, Dict, Any, Optional, Tuple


def generate_unique_id() -> str:
    """
    Generate a unique ID for documents and charts.
    
    Returns:
        str: A UUID string
    """
    return str(uuid.uuid4())


def extract_chart_markers(text: str) -> List[str]:
    """
    Extract chart reference markers from text.
    
    Args:
        text (str): The text to analyze for chart references
        
    Returns:
        List[str]: List of extracted chart markers
    """
    # Look for patterns like [CHART:identifier]
    return re.findall(r'\[CHART:(.*?)\]', text)


def extract_figure_references(text: str) -> List[str]:
    """
    Extract figure references from text.
    
    Args:
        text (str): The text to analyze for figure references
        
    Returns:
        List[str]: List of extracted figure references
    """
    # Look for patterns like "Figure X", "Chart X", etc.
    references = re.findall(r'(Figure|Chart|Graph)\s+(\d+)', text, re.IGNORECASE)
    return [ref[1] for ref in references]


def ensure_directory_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File extension (including the dot)
    """
    return os.path.splitext(file_path)[1].lower()


def is_pdf_file(file_path: str) -> bool:
    """
    Check if a file is a PDF based on its extension.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        bool: True if the file is a PDF, False otherwise
    """
    return get_file_extension(file_path) == '.pdf'


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace characters that are not alphanumeric, dash, underscore, or dot
    return re.sub(r'[^\w\-\.]', '_', filename)


def text_to_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs based on double newlines.
    
    Args:
        text (str): The text to split
        
    Returns:
        List[str]: List of paragraphs
    """
    # Split by double newlines and filter out empty paragraphs
    return [p.strip() for p in text.split('\n\n') if p.strip()]


def is_likely_chart_paragraph(text: str) -> bool:
    """
    Determine if a paragraph likely refers to a chart.
    
    Args:
        text (str): The paragraph text
        
    Returns:
        bool: True if the paragraph likely refers to a chart
    """
    # Check for common chart reference patterns
    chart_keywords = [
        'chart', 'figure', 'graph', 'diagram', 'plot', 
        'candle', 'candlestick', 'pattern'
    ]
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check for explicit chart markers
    if re.search(r'\[CHART:.*?\]', text):
        return True
    
    # Check for figure references
    if re.search(r'(figure|chart|graph)\s+\d+', text_lower):
        return True
    
    # Check for keywords
    keyword_count = sum(1 for keyword in chart_keywords if keyword in text_lower)
    return keyword_count >= 2  # At least two keywords should be present
