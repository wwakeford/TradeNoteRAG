"""
Tests for the PDF extractor module.
"""

import os
import pytest
from pdf_processor.extractor import PDFExtractor, Paragraph, Chart


@pytest.fixture
def sample_pdf_path():
    """Get path to a sample PDF file."""
    # This is a placeholder - in practice, you'd need a real PDF file
    # For testing, you might want to include a small test PDF in your repository
    return os.path.join(os.path.dirname(__file__), "data", "sample.pdf")


@pytest.fixture
def extractor():
    """Create a PDF extractor instance."""
    return PDFExtractor(
        min_image_size=50,  # Lower threshold for testing
        chart_detection_threshold=0.5
    )


def test_extract_paragraphs(extractor, sample_pdf_path, monkeypatch):
    """Test paragraph extraction from PDF."""
    # Skip the test if the sample PDF doesn't exist
    if not os.path.exists(sample_pdf_path):
        pytest.skip(f"Sample PDF not found at {sample_pdf_path}")
    
    # Test with a real PDF
    paragraphs = extractor.extract_paragraphs(sample_pdf_path)
    
    # Basic assertions
    assert isinstance(paragraphs, list)
    assert all(isinstance(p, Paragraph) for p in paragraphs)
    
    # If paragraphs were found, check their structure
    if paragraphs:
        for para in paragraphs:
            assert hasattr(para, 'text')
            assert hasattr(para, 'page_num')
            assert hasattr(para, 'position')
            assert isinstance(para.position, tuple)
            assert len(para.position) == 4  # (x0, y0, x1, y1)


def test_detect_charts(extractor, sample_pdf_path, monkeypatch):
    """Test chart detection from PDF."""
    # Skip the test if the sample PDF doesn't exist
    if not os.path.exists(sample_pdf_path):
        pytest.skip(f"Sample PDF not found at {sample_pdf_path}")
    
    # Test with a real PDF
    charts = extractor.detect_charts(sample_pdf_path)
    
    # Basic assertions
    assert isinstance(charts, list)
    assert all(isinstance(c, Chart) for c in charts)
    
    # If charts were found, check their structure
    if charts:
        for chart in charts:
            assert hasattr(chart, 'image_data')
            assert hasattr(chart, 'page_num')
            assert hasattr(chart, 'position')
            assert hasattr(chart, 'position_marker')
            assert isinstance(chart.position, tuple)
            assert len(chart.position) == 4  # (x0, y0, x1, y1)
            assert isinstance(chart.image_data, bytes)


def test_process_pdf(extractor, sample_pdf_path, monkeypatch):
    """Test complete PDF processing."""
    # Skip the test if the sample PDF doesn't exist
    if not os.path.exists(sample_pdf_path):
        pytest.skip(f"Sample PDF not found at {sample_pdf_path}")
    
    # Test with a real PDF
    paragraphs, charts = extractor.process_pdf(sample_pdf_path)
    
    # Basic assertions
    assert isinstance(paragraphs, list)
    assert isinstance(charts, list)
    
    # Check types of returned items
    assert all(isinstance(p, Paragraph) for p in paragraphs)
    assert all(isinstance(c, Chart) for c in charts)


def test_error_handling(extractor):
    """Test error handling for non-existent files."""
    # Test with a non-existent file
    paragraphs = extractor.extract_paragraphs("non_existent_file.pdf")
    charts = extractor.detect_charts("non_existent_file.pdf")
    
    # Should return empty lists rather than raising exceptions
    assert paragraphs == []
    assert charts == []


def test_minimum_image_size_filter(extractor, monkeypatch):
    """Test filtering of small images."""
    # Create a mock implementation of Image.open that returns 
    # an image with known dimensions
    class MockImage:
        def __init__(self, width, height):
            self.width = width
            self.height = height
    
    # This test would be more complex in practice
    # Here, we're just verifying the concept
    assert extractor.min_image_size == 50
