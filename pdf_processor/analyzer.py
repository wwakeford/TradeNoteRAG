"""
Module for analyzing relationships between text and charts in trading documents.
"""

import re
from typing import Dict, List, Tuple, Optional, Any, Set
from .extractor import Paragraph, Chart
from .utils import extract_chart_markers, extract_figure_references, is_likely_chart_paragraph


class RelationshipAnalyzer:
    """
    Analyzes the relationships between paragraphs and charts in trading documents.
    """
    
    def __init__(self, proximity_threshold: int = 50):
        """
        Initialize the RelationshipAnalyzer.
        
        Args:
            proximity_threshold (int): Threshold in points for spatial proximity detection
        """
        self.proximity_threshold = proximity_threshold
    
    def analyze_text_chart_relationships(
        self, 
        paragraphs: List[Paragraph], 
        charts: List[Chart]
    ) -> Dict[str, List[str]]:
        """
        Determine which paragraphs reference which charts based on text analysis.
        
        Args:
            paragraphs (List[Paragraph]): List of extracted paragraphs
            charts (List[Chart]): List of extracted charts
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping paragraph indices to chart markers
        """
        relationships = {}
        
        # Create a mapping of position markers to chart indices
        chart_markers = {chart.position_marker: i for i, chart in enumerate(charts)}
        
        for i, para in enumerate(paragraphs):
            # Extract explicit chart references
            refs = extract_chart_markers(para.text)
            
            # Extract figure references
            figure_refs = extract_figure_references(para.text)
            
            # Combine all references
            all_refs = refs + figure_refs
            
            # Store any found relationships
            if all_refs:
                relationships[str(i)] = all_refs
        
        return relationships
    
    def analyze_spatial_relationships(
        self, 
        paragraphs: List[Paragraph], 
        charts: List[Chart]
    ) -> Dict[str, List[str]]:
        """
        Analyze spatial relationships between paragraphs and charts.
        
        Args:
            paragraphs (List[Paragraph]): List of extracted paragraphs
            charts (List[Chart]): List of extracted charts
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping paragraph indices to nearby chart markers
        """
        spatial_relationships = {}
        
        for i, para in enumerate(paragraphs):
            nearby_charts = []
            
            for chart in charts:
                # Only consider charts and paragraphs on the same page
                if para.page_num == chart.page_num:
                    para_y_mid = (para.position[1] + para.position[3]) / 2
                    chart_y_mid = (chart.position[1] + chart.position[3]) / 2
                    
                    # Check vertical proximity
                    if abs(para_y_mid - chart_y_mid) < self.proximity_threshold * 2:
                        nearby_charts.append(chart.position_marker)
                        continue
                    
                    # Check if paragraph is just above the chart
                    para_bottom = para.position[3]
                    chart_top = chart.position[1]
                    if 0 < (chart_top - para_bottom) < self.proximity_threshold:
                        nearby_charts.append(chart.position_marker)
                        continue
                    
                    # Check if paragraph is just below the chart
                    para_top = para.position[1]
                    chart_bottom = chart.position[3]
                    if 0 < (para_top - chart_bottom) < self.proximity_threshold:
                        nearby_charts.append(chart.position_marker)
                        continue
            
            if nearby_charts and is_likely_chart_paragraph(para.text):
                spatial_relationships[str(i)] = nearby_charts
        
        return spatial_relationships
    
    def merge_relationships(
        self, 
        text_relationships: Dict[str, List[str]], 
        spatial_relationships: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Merge text-based and spatial relationships, removing duplicates.
        
        Args:
            text_relationships (Dict[str, List[str]]): Relationships from text analysis
            spatial_relationships (Dict[str, List[str]]): Relationships from spatial analysis
            
        Returns:
            Dict[str, List[str]]: Merged relationships
        """
        merged = {}
        
        # Include all paragraph indices from both dictionaries
        all_para_indices = set(text_relationships.keys()) | set(spatial_relationships.keys())
        
        for para_idx in all_para_indices:
            text_refs = text_relationships.get(para_idx, [])
            spatial_refs = spatial_relationships.get(para_idx, [])
            
            # Combine and de-duplicate references
            merged[para_idx] = list(set(text_refs + spatial_refs))
        
        return merged
    
    def create_bidirectional_mapping(
        self, 
        para_to_charts: Dict[str, List[str]]
    ) -> Dict[str, List[str]]:
        """
        Create a bidirectional mapping from charts to paragraphs.
        
        Args:
            para_to_charts (Dict[str, List[str]]): Mapping of paragraphs to charts
            
        Returns:
            Dict[str, List[str]]: Mapping of charts to paragraphs
        """
        chart_to_paras = {}
        
        for para_idx, chart_refs in para_to_charts.items():
            for chart_ref in chart_refs:
                if chart_ref not in chart_to_paras:
                    chart_to_paras[chart_ref] = []
                
                chart_to_paras[chart_ref].append(para_idx)
        
        return chart_to_paras
    
    def analyze_document_structure(
        self, 
        paragraphs: List[Paragraph]
    ) -> Dict[str, Any]:
        """
        Analyze the structure of the document to identify sections and subsections.
        
        Args:
            paragraphs (List[Paragraph]): List of extracted paragraphs
            
        Returns:
            Dict[str, Any]: Document structure information
        """
        # This is a placeholder for future implementation
        # In a full implementation, we'd analyze font sizes, spacing, etc.
        structure = {
            "sections": [],
            "section_to_paragraphs": {}
        }
        
        # Simple heuristic: paragraphs with fewer than 10 words might be headings
        current_section = None
        
        for i, para in enumerate(paragraphs):
            words = para.text.split()
            
            if len(words) < 10 and not any(c in para.text for c in "[]{}<>()"):
                # Likely a heading
                current_section = para.text
                structure["sections"].append({
                    "title": current_section,
                    "index": i
                })
                structure["section_to_paragraphs"][current_section] = []
            elif current_section is not None:
                structure["section_to_paragraphs"][current_section].append(i)
        
        return structure
