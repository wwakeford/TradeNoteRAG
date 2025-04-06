"""
Main module for the PDF processing pipeline.
"""

import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from pydantic import BaseModel, Field
from .extractor import PDFExtractor, Paragraph, Chart
from .analyzer import RelationshipAnalyzer
from .enhancer import ContentEnhancer, EnhancedParagraph, EnhancedChart
from .database import VectorDBClient
from .utils import ensure_directory_exists, is_pdf_file, generate_unique_id


class ProcessingResult(BaseModel):
    """Model representing the result of processing a PDF."""
    note_id: str
    title: str
    section: str
    subsection: str
    chunk_ids: List[str] = Field(default_factory=list)
    chart_ids: List[str] = Field(default_factory=list)
    paragraph_count: int = 0
    chart_count: int = 0
    chunk_count: int = 0
    error: Optional[str] = None


class PDFProcessor:
    """
    Main processor that coordinates the PDF processing pipeline.
    """
    
    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        openai_api_key: str,
        min_image_size: int = 100,
        chart_detection_threshold: float = 0.7,
        proximity_threshold: int = 50,
        temp_dir: str = "./temp"
    ):
        """
        Initialize the PDF processor with all required components.
        
        Args:
            supabase_url (str): Supabase URL
            supabase_key (str): Supabase API key
            openai_api_key (str): OpenAI API key
            min_image_size (int): Minimum size for chart detection
            chart_detection_threshold (float): Confidence threshold for chart detection
            proximity_threshold (int): Threshold for spatial proximity
            temp_dir (str): Directory for temporary files
        """
        # Initialize all components
        self.extractor = PDFExtractor(
            min_image_size=min_image_size,
            chart_detection_threshold=chart_detection_threshold
        )
        
        self.analyzer = RelationshipAnalyzer(
            proximity_threshold=proximity_threshold
        )
        
        self.enhancer = ContentEnhancer(
            api_key=openai_api_key
        )
        
        self.db_client = VectorDBClient(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            openai_api_key=openai_api_key
        )
        
        self.temp_dir = temp_dir
        ensure_directory_exists(temp_dir)
    
    def _extract_title_and_metadata(self, paragraphs: List[Paragraph]) -> Tuple[str, str, str]:
        """
        Extract title and metadata from paragraphs.
        
        Args:
            paragraphs (List[Paragraph]): List of paragraphs
            
        Returns:
            Tuple[str, str, str]: Title, section, and subsection
        """
        title = "Untitled Trading Note"
        section = ""
        subsection = ""
        
        if paragraphs:
            # First paragraph is often the title or contains section info
            first_para = paragraphs[0].text
            
            # If it's short, it's likely a title
            if len(first_para.split()) < 15:
                title = first_para
                
                # Check if the second paragraph might be a subsection
                if len(paragraphs) > 1 and len(paragraphs[1].text.split()) < 15:
                    subsection = paragraphs[1].text
            
            # Extract potential section information
            for para in paragraphs[:3]:  # Check first few paragraphs
                text_lower = para.text.lower()
                
                # Look for common section indicators
                if "technical analysis" in text_lower:
                    section = "Technical analysis"
                elif "fundamental analysis" in text_lower:
                    section = "Fundamental analysis"
                elif "day trading" in text_lower:
                    section = "Day trading"
                elif "swing trading" in text_lower:
                    section = "Swing trading"
                elif "long-term" in text_lower:
                    if "technical" in text_lower:
                        section = "Long-term Technical analysis"
                    elif "fundamental" in text_lower:
                        section = "Long-term Fundamental analysis"
                elif "short-term" in text_lower:
                    if "technical" in text_lower:
                        section = "Short-term Technical analysis"
        
        return title, section, subsection
    
    def process_pdf(
        self, 
        pdf_path: str,
        section: str = "",
        subsection: str = ""
    ) -> ProcessingResult:
        """
        Process a PDF document through the entire pipeline.
        
        Args:
            pdf_path (str): Path to the PDF file
            section (str): Optional manual section override
            subsection (str): Optional manual subsection override
            
        Returns:
            ProcessingResult: Result of the processing
        """
        try:
            # Check if file exists and is a PDF
            if not os.path.exists(pdf_path):
                return ProcessingResult(
                    note_id="",
                    title="",
                    section="",
                    subsection="",
                    error=f"File not found: {pdf_path}"
                )
                
            if not is_pdf_file(pdf_path):
                return ProcessingResult(
                    note_id="",
                    title="",
                    section="",
                    subsection="",
                    error=f"Not a PDF file: {pdf_path}"
                )
            
            # Step 1: Extract paragraphs and charts
            print(f"Extracting content from {pdf_path}...")
            paragraphs, charts = self.extractor.process_pdf(pdf_path)
            
            if not paragraphs:
                return ProcessingResult(
                    note_id="",
                    title="",
                    section="",
                    subsection="",
                    error="No text content extracted from PDF"
                )
            
            # Step 2: Analyze relationships
            print("Analyzing text-chart relationships...")
            text_relationships = self.analyzer.analyze_text_chart_relationships(paragraphs, charts)
            spatial_relationships = self.analyzer.analyze_spatial_relationships(paragraphs, charts)
            relationships = self.analyzer.merge_relationships(text_relationships, spatial_relationships)
            
            # Step 3: Enhance content
            print("Enhancing content with LLM analysis...")
            enhanced_paragraphs = self.enhancer.enhance_paragraphs(paragraphs, relationships)
            enhanced_charts = self.enhancer.enhance_charts(charts, paragraphs, relationships)
            chunks = self.enhancer.combine_paragraphs_into_chunks(enhanced_paragraphs)
            
            # Step 4: Extract metadata
            auto_title, auto_section, auto_subsection = self._extract_title_and_metadata(paragraphs)
            
            # Use provided values if available, otherwise use auto-detected ones
            title = auto_title
            final_section = section if section else auto_section
            final_subsection = subsection if subsection else auto_subsection
            
            # Step 5: Store in database
            print("Storing content in database...")
            
            # Create full content for the note
            full_content = "\n\n".join([p.text for p in paragraphs])
            
            # Store the note and get ID
            note_id = self.db_client.store_note(
                title=title,
                content=full_content,
                section=final_section,
                subsection=final_subsection,
                tags=[]  # We could extract global tags in the future
            )
            
            # Store chunks
            chunk_ids = self.db_client.store_chunks(
                note_id=note_id,
                chunks=chunks,
                enhanced_paragraphs=enhanced_paragraphs
            )
            
            # Store charts
            chart_ids = []
            for i, (chart, enhanced_chart) in enumerate(zip(charts, enhanced_charts)):
                chart_id = self.db_client.store_chart(
                    note_id=note_id,
                    chart=chart,
                    enhanced_chart=enhanced_chart
                )
                if chart_id:
                    chart_ids.append(chart_id)
                
                # Add delay to prevent rate limiting
                if i < len(charts) - 1:
                    time.sleep(0.2)
            
            # Return processing result
            return ProcessingResult(
                note_id=note_id,
                title=title,
                section=final_section,
                subsection=final_subsection,
                chunk_ids=chunk_ids,
                chart_ids=chart_ids,
                paragraph_count=len(paragraphs),
                chart_count=len(charts),
                chunk_count=len(chunks)
            )
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ProcessingResult(
                note_id="",
                title="",
                section="",
                subsection="",
                error=str(e)
            )
