"""
Tests for the UUID position marker functionality.
"""

import os
import uuid
import pytest
from pdf_processor.extractor import PDFExtractor


def test_position_marker_uuid_format():
    """Test that generated position markers are valid UUIDs."""
    extractor = PDFExtractor()
    
    # Test generating UUIDs for different types of charts
    for chart_type in ['embedded', 'vector', 'trading', 'forced']:
        for page_num in range(3):
            for index in range(5):
                # Generate a position marker
                marker = extractor.generate_uuid_position_marker(chart_type, page_num, index)
                
                # Verify it's a valid UUID
                try:
                    uuid_obj = uuid.UUID(marker)
                    assert str(uuid_obj) == marker, "UUID string representation should match the original"
                except ValueError:
                    pytest.fail(f"Generated marker '{marker}' is not a valid UUID")
