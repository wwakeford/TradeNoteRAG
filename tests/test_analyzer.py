"""
Tests for the relationship analyzer module.
"""

import pytest
from pdf_processor.extractor import Paragraph, Chart
from pdf_processor.analyzer import RelationshipAnalyzer


@pytest.fixture
def analyzer():
    """Create a relationship analyzer instance."""
    return RelationshipAnalyzer(proximity_threshold=50)


@pytest.fixture
def sample_paragraphs():
    """Create sample paragraphs for testing."""
    return [
        Paragraph(
            text="This is paragraph 1 with a reference to [CHART:chart1].",
            page_num=0,
            position=(10, 10, 100, 30)
        ),
        Paragraph(
            text="This is paragraph 2 with no chart references.",
            page_num=0,
            position=(10, 40, 100, 60)
        ),
        Paragraph(
            text="This is paragraph 3 with a reference to Figure 2.",
            page_num=0,
            position=(10, 70, 100, 90)
        ),
        Paragraph(
            text="This paragraph is on page 1 and near chart3.",
            page_num=1,
            position=(10, 20, 100, 40)
        )
    ]


@pytest.fixture
def sample_charts():
    """Create sample charts for testing."""
    return [
        Chart(
            image_data=b"sample_image_data_1",
            page_num=0,
            position=(120, 10, 200, 50),
            position_marker="chart1"
        ),
        Chart(
            image_data=b"sample_image_data_2",
            page_num=0,
            position=(120, 70, 200, 110),
            position_marker="chart2"
        ),
        Chart(
            image_data=b"sample_image_data_3",
            page_num=1,
            position=(120, 15, 200, 55),
            position_marker="chart3"
        )
    ]


def test_analyze_text_chart_relationships(analyzer, sample_paragraphs, sample_charts):
    """Test analysis of text-based relationships between paragraphs and charts."""
    relationships = analyzer.analyze_text_chart_relationships(
        sample_paragraphs, sample_charts
    )
    
    # Check if relationships dict is created
    assert isinstance(relationships, dict)
    
    # Check if paragraph with explicit chart reference is linked
    assert "0" in relationships
    assert "chart1" in relationships["0"]
    
    # Check if paragraph with figure reference is linked
    assert "2" in relationships
    assert "2" in relationships["2"]  # Figure 2 reference
    
    # Check if paragraph without references is not linked
    assert "1" not in relationships


def test_analyze_spatial_relationships(analyzer, sample_paragraphs, sample_charts):
    """Test analysis of spatial relationships between paragraphs and charts."""
    spatial_relationships = analyzer.analyze_spatial_relationships(
        sample_paragraphs, sample_charts
    )
    
    # Check if relationships dict is created
    assert isinstance(spatial_relationships, dict)
    
    # The test paragraphs and charts are positioned so there should be
    # spatial relationships between paragraphs and nearby charts
    
    # On the same page, nearby in position
    # This test can be expanded based on your actual spatial logic


def test_merge_relationships(analyzer):
    """Test merging of text and spatial relationships."""
    text_relationships = {
        "0": ["chart1"],
        "2": ["chart2"]
    }
    
    spatial_relationships = {
        "1": ["chart1"],
        "2": ["chart3"]
    }
    
    merged = analyzer.merge_relationships(text_relationships, spatial_relationships)
    
    # Check if all paragraph indices are included
    assert "0" in merged
    assert "1" in merged
    assert "2" in merged
    
    # Check if chart references are merged without duplicates
    assert "chart1" in merged["0"]
    assert "chart1" in merged["1"]
    assert "chart2" in merged["2"]
    assert "chart3" in merged["2"]
    
    # Check if merged relationships have no duplicates
    for para_idx, chart_refs in merged.items():
        assert len(chart_refs) == len(set(chart_refs))


def test_create_bidirectional_mapping(analyzer):
    """Test creation of bidirectional mapping from paragraphs to charts and vice versa."""
    para_to_charts = {
        "0": ["chart1"],
        "1": ["chart1", "chart2"],
        "2": ["chart2", "chart3"]
    }
    
    chart_to_paras = analyzer.create_bidirectional_mapping(para_to_charts)
    
    # Check if all chart references are included
    assert "chart1" in chart_to_paras
    assert "chart2" in chart_to_paras
    assert "chart3" in chart_to_paras
    
    # Check if paragraph indices are correctly mapped
    assert "0" in chart_to_paras["chart1"]
    assert "1" in chart_to_paras["chart1"]
    assert "1" in chart_to_paras["chart2"]
    assert "2" in chart_to_paras["chart2"]
    assert "2" in chart_to_paras["chart3"]


def test_analyze_document_structure(analyzer, sample_paragraphs):
    """Test analysis of document structure."""
    structure = analyzer.analyze_document_structure(sample_paragraphs)
    
    # Check if structure dict is created
    assert isinstance(structure, dict)
    assert "sections" in structure
    assert "section_to_paragraphs" in structure
    
    # This is mostly a placeholder test since the actual implementation
    # of document structure analysis will be more complex
