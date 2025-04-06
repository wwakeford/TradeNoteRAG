"""
Module for enhancing extracted content using LLM-based analysis.
"""

import openai
from typing import Dict, List, Tuple, Optional, Any, Set
from pydantic import BaseModel, Field
from .extractor import Paragraph, Chart
from .utils import is_likely_chart_paragraph


class EnhancedParagraph(BaseModel):
    """Model representing an enhanced paragraph with metadata."""
    original_text: str
    enhanced_text: Optional[str] = None
    trading_terms: List[str] = Field(default_factory=list)
    chunk_id: Optional[int] = None
    chart_references: List[str] = Field(default_factory=list)


class EnhancedChart(BaseModel):
    """Model representing an enhanced chart with metadata."""
    position_marker: str
    caption: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    related_paragraphs: List[int] = Field(default_factory=list)


class ContentEnhancer:
    """
    Enhances extracted content with LLM-powered metadata and analysis.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the ContentEnhancer.
        
        Args:
            api_key (str): OpenAI API key
            model (str): The model to use for LLM operations
        """
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
    
    def extract_trading_terms(self, text: str) -> List[str]:
        """
        Extract technical trading terms from text using LLM.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: Extracted trading terms
        """
        if not text:
            return []
            
        prompt = f"""
        Extract technical trading terms and concepts from the following text about trading.
        Return only a comma-separated list of terms without explanations.
        
        Text: {text}
        
        Technical trading terms:
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a trading analysis assistant that extracts technical trading terms from text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.0
            )
            
            result = response.choices[0].message.content.strip()
            terms = [term.strip() for term in result.split(',') if term.strip()]
            return terms
        
        except Exception as e:
            print(f"Error extracting trading terms: {e}")
            return []
    
    def generate_chart_caption(self, chart: Chart, surrounding_text: str) -> Tuple[str, List[str]]:
        """
        Generate a caption and tags for a chart based on surrounding text.
        
        Args:
            chart (Chart): The chart to generate a caption for
            surrounding_text (str): Text surrounding the chart
            
        Returns:
            Tuple[str, List[str]]: Generated caption and tags
        """
        if not surrounding_text:
            return "", []
            
        prompt = f"""
        Based on the following text that surrounds a chart in a trading document, 
        generate a concise caption for the chart and extract relevant trading tags.
        
        Text surrounding the chart: {surrounding_text}
        
        Format your response exactly as follows:
        Caption: [concise description of what the chart likely shows]
        Tags: [comma-separated list of relevant trading concepts, patterns, or indicators]
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a trading analysis assistant that generates captions and tags for trading charts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            result = response.choices[0].message.content.strip()
            
            caption = ""
            tags = []
            
            for line in result.split('\n'):
                if line.startswith('Caption:'):
                    caption = line[8:].strip()
                elif line.startswith('Tags:'):
                    tags_text = line[5:].strip()
                    tags = [tag.strip() for tag in tags_text.split(',') if tag.strip()]
            
            return caption, tags
        
        except Exception as e:
            print(f"Error generating chart caption: {e}")
            return "", []
    
    def determine_chunk_boundaries(
        self, 
        paragraphs: List[Paragraph],
        relationships: Dict[str, List[str]]
    ) -> List[List[int]]:
        """
        Determine optimal chunking boundaries based on semantic coherence.
        
        Args:
            paragraphs (List[Paragraph]): List of paragraphs
            relationships (Dict[str, List[str]]): Paragraph-chart relationships
            
        Returns:
            List[List[int]]: List of paragraph indices for each chunk
        """
        chunks = []
        current_chunk = []
        max_chunk_size = 5  # Maximum paragraphs per chunk
        
        # Process paragraphs in order
        for i, para in enumerate(paragraphs):
            # Start a new chunk if the current one is empty
            if not current_chunk:
                current_chunk.append(i)
                continue
                
            # Check if this paragraph has chart references
            has_chart_refs = str(i) in relationships and len(relationships[str(i)]) > 0
            
            # Check if the previous paragraph in this chunk has chart references
            prev_i = current_chunk[-1]
            prev_has_chart_refs = str(prev_i) in relationships and len(relationships[str(prev_i)]) > 0
            
            # Start a new chunk if:
            # 1. Current chunk is at max size, or
            # 2. This paragraph has chart refs but the previous didn't, or
            # 3. This paragraph has no chart refs but the previous did
            if (len(current_chunk) >= max_chunk_size or
                (has_chart_refs and not prev_has_chart_refs) or
                (not has_chart_refs and prev_has_chart_refs)):
                
                # Finish the current chunk
                chunks.append(current_chunk)
                current_chunk = [i]
            else:
                # Add to the current chunk
                current_chunk.append(i)
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def enhance_paragraphs(
        self, 
        paragraphs: List[Paragraph], 
        relationships: Dict[str, List[str]]
    ) -> List[EnhancedParagraph]:
        """
        Enhance paragraphs with metadata and chunking.
        
        Args:
            paragraphs (List[Paragraph]): List of paragraphs
            relationships (Dict[str, List[str]]): Paragraph-chart relationships
            
        Returns:
            List[EnhancedParagraph]: Enhanced paragraphs
        """
        enhanced_paragraphs = []
        chunk_boundaries = self.determine_chunk_boundaries(paragraphs, relationships)
        
        # Assign chunk IDs to each paragraph
        chunk_mapping = {}
        for chunk_id, indices in enumerate(chunk_boundaries):
            for idx in indices:
                chunk_mapping[idx] = chunk_id
        
        for i, para in enumerate(paragraphs):
            chart_refs = relationships.get(str(i), [])
            
            # Extract trading terms if:
            # 1. The paragraph is longer than 100 characters, or
            # 2. It contains chart references
            trading_terms = []
            if len(para.text) > 100 or chart_refs:
                trading_terms = self.extract_trading_terms(para.text)
            
            enhanced = EnhancedParagraph(
                original_text=para.text,
                trading_terms=trading_terms,
                chunk_id=chunk_mapping.get(i),
                chart_references=chart_refs
            )
            
            enhanced_paragraphs.append(enhanced)
        
        return enhanced_paragraphs
    
    def enhance_charts(
        self, 
        charts: List[Chart], 
        paragraphs: List[Paragraph], 
        relationships: Dict[str, List[str]]
    ) -> List[EnhancedChart]:
        """
        Enhance charts with metadata generated from related paragraphs.
        
        Args:
            charts (List[Chart]): List of charts
            paragraphs (List[Paragraph]): List of paragraphs
            relationships (Dict[str, List[str]]): Paragraph-chart relationships
            
        Returns:
            List[EnhancedChart]: Enhanced charts
        """
        enhanced_charts = []
        
        # Create a mapping from chart markers to paragraph indices
        chart_to_paras = {}
        for para_idx, chart_refs in relationships.items():
            for chart_ref in chart_refs:
                if chart_ref not in chart_to_paras:
                    chart_to_paras[chart_ref] = []
                chart_to_paras[chart_ref].append(int(para_idx))
        
        for chart in charts:
            related_para_indices = chart_to_paras.get(chart.position_marker, [])
            
            # Get surrounding text for caption generation
            surrounding_text = ""
            if related_para_indices:
                # Concatenate the related paragraphs
                for idx in related_para_indices:
                    if 0 <= idx < len(paragraphs):
                        surrounding_text += paragraphs[idx].text + " "
            
            # If no explicit references, try to find paragraphs nearby in the PDF
            if not surrounding_text:
                for i, para in enumerate(paragraphs):
                    if para.page_num == chart.page_num:
                        # Check if paragraph is within reasonable distance
                        para_bottom = para.position[3]
                        para_top = para.position[1]
                        chart_bottom = chart.position[3]
                        chart_top = chart.position[1]
                        
                        if (abs(para_bottom - chart_top) < 100 or 
                            abs(chart_bottom - para_top) < 100):
                            surrounding_text += para.text + " "
                            related_para_indices.append(i)
            
            caption, tags = "", []
            if surrounding_text:
                caption, tags = self.generate_chart_caption(chart, surrounding_text)
            
            enhanced = EnhancedChart(
                position_marker=chart.position_marker,
                caption=caption,
                tags=tags,
                related_paragraphs=related_para_indices
            )
            
            enhanced_charts.append(enhanced)
        
        return enhanced_charts
    
    def combine_paragraphs_into_chunks(
        self,
        enhanced_paragraphs: List[EnhancedParagraph]
    ) -> Dict[int, str]:
        """
        Combine paragraphs into coherent chunks based on assigned chunk IDs.
        
        Args:
            enhanced_paragraphs (List[EnhancedParagraph]): Enhanced paragraphs
            
        Returns:
            Dict[int, str]: Mapping of chunk IDs to combined text
        """
        chunks = {}
        
        # Group paragraphs by chunk ID
        for para in enhanced_paragraphs:
            if para.chunk_id is not None:
                if para.chunk_id not in chunks:
                    chunks[para.chunk_id] = ""
                
                chunks[para.chunk_id] += para.original_text + "\n\n"
        
        # Trim extra whitespace
        for chunk_id in chunks:
            chunks[chunk_id] = chunks[chunk_id].strip()
        
        return chunks
