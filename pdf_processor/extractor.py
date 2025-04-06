"""
Module for extracting text and charts from PDF documents.
"""

import os
import fitz  # PyMuPDF
import uuid
from typing import Dict, List, Tuple, Optional, Any
from pydantic import BaseModel, Field

from .chart_detector import ChartDetector


class Paragraph(BaseModel):
    """Model representing a paragraph extracted from a PDF."""
    text: str
    page_num: int
    position: Tuple[float, float, float, float]  # x0, y0, x1, y1
    
    class Config:
        """Pydantic config for the Paragraph model."""
        arbitrary_types_allowed = True


class Chart(BaseModel):
    """Model representing a chart image extracted from a PDF."""
    image_data: bytes
    page_num: int
    position: Tuple[float, float, float, float]  # x0, y0, x1, y1
    position_marker: str  # Must be UUID format for database storage
    
    class Config:
        """Pydantic config for the Chart model."""
        arbitrary_types_allowed = True


class PDFExtractor:
    """
    Extracts text and chart images from PDF documents with paragraph-level granularity.
    """
    
    def __init__(
        self, 
        min_image_size: int = 100, 
        chart_detection_threshold: float = 0.7,
        force_chart_extraction: bool = True,
        contour_area_threshold: int = 5000,
        trading_chart_detection: bool = True
    ):
        """
        Initialize the PDFExtractor with configuration parameters.
        
        Args:
            min_image_size (int): Minimum width/height for an image to be considered a chart
            chart_detection_threshold (float): Confidence threshold for chart detection
            force_chart_extraction (bool): If True, use aggressive chart extraction for trading docs
            contour_area_threshold (int): Minimum area for contour detection (lower = more sensitive)
            trading_chart_detection (bool): Enable specialized trading chart detection
        """
        self.chart_detector = ChartDetector(
            min_image_size=min_image_size,
            chart_detection_threshold=chart_detection_threshold,
            force_chart_extraction=force_chart_extraction,
            contour_area_threshold=contour_area_threshold,
            trading_chart_detection=trading_chart_detection
        )
    
    def extract_paragraphs(self, pdf_path: str) -> List[Paragraph]:
        """
        Extract paragraphs from a PDF document with position data.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Paragraph]: List of extracted paragraphs with position data
        """
        paragraphs = []
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                # Extract text blocks (paragraphs)
                blocks = page.get_text("blocks")
                
                for block in blocks:
                    # Filter out non-text blocks
                    if not block[4].strip():
                        continue
                    
                    # Create paragraph with position data
                    para = Paragraph(
                        text=block[4].strip(),
                        page_num=page_num,
                        position=(block[0], block[1], block[2], block[3])
                    )
                    paragraphs.append(para)
            
            doc.close()
            return paragraphs
        
        except Exception as e:
            print(f"Error extracting paragraphs from PDF: {e}")
            return []
    
    def generate_uuid_position_marker(self, chart_type: str, page_num: int, index: int) -> str:
        """
        Generate a UUID-based position marker for the chart.
        
        Args:
            chart_type (str): Type of chart (e.g., 'embedded', 'vector', 'trading')
            page_num (int): Page number of the chart
            index (int): Index of the chart on the page
            
        Returns:
            str: UUID-based position marker with chart metadata
        """
        # Generate a unique UUID first - this ensures we're using valid UUIDs
        # that will work with the database's UUID column type
        chart_uuid = str(uuid.uuid4())
        
        # Return the UUID without any modifications
        return chart_uuid
    
    def detect_charts(self, pdf_path: str) -> List[Chart]:
        """
        Detect and extract chart images from a PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            List[Chart]: List of extracted chart images with position data
        """
        charts = []
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num, page in enumerate(doc):
                page_charts = []
                
                # Get chart detection results
                chart_results = self.chart_detector.detect_all_charts(doc, page, page_num)
                
                # Convert to Chart objects with proper UUIDs
                for i, result in enumerate(chart_results):
                    chart_type = result.get('type', 'unknown')
                    chart_data = result.get('image_data')
                    chart_position = result.get('position')
                    
                    # Generate a proper UUID for database storage
                    position_marker = self.generate_uuid_position_marker(chart_type, page_num, i)
                    
                    chart = Chart(
                        image_data=chart_data,
                        page_num=page_num,
                        position=chart_position,
                        position_marker=position_marker
                    )
                    
                    page_charts.append(chart)
                
                charts.extend(page_charts)
            
            doc.close()
            return charts
        
        except Exception as e:
            print(f"Error detecting charts from PDF: {e}")
            return []
    
    def process_pdf(self, pdf_path: str) -> Tuple[List[Paragraph], List[Chart]]:
        """
        Process a PDF document to extract both paragraphs and charts.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            Tuple[List[Paragraph], List[Chart]]: Extracted paragraphs and charts
        """
        paragraphs = self.extract_paragraphs(pdf_path)
        charts = self.detect_charts(pdf_path)
        
        return paragraphs, charts
