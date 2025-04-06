"""
Example usage of the PDF processing pipeline.
"""

import os
import sys
import time
from pdf_processor.processor import PDFProcessor
from pdf_processor.config import get_settings


def example_process_pdf(pdf_path):
    """
    Example of processing a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
    """
    # Get application settings
    settings = get_settings()
    
    # Initialize the processor
    processor = PDFProcessor(
        supabase_url=settings.SUPABASE_URL,
        supabase_key=settings.SUPABASE_KEY,
        openai_api_key=settings.OPENAI_API_KEY,
        min_image_size=settings.MIN_IMAGE_SIZE,
        chart_detection_threshold=settings.CHART_DETECTION_THRESHOLD,
        proximity_threshold=settings.PROXIMITY_THRESHOLD,
        temp_dir=settings.TEMP_DIR
    )
    
    # Process the PDF
    print(f"Processing {pdf_path}...")
    start_time = time.time()
    
    result = processor.process_pdf(
        pdf_path=pdf_path,
        section="",  # Auto-detect
        subsection=""  # Auto-detect
    )
    
    elapsed_time = time.time() - start_time
    
    # Display results
    if result.error:
        print(f"Error: {result.error}")
    else:
        print("\nProcessing Results:")
        print("-" * 50)
        print(f"Note ID: {result.note_id}")
        print(f"Title: {result.title}")
        print(f"Section: {result.section}")
        print(f"Subsection: {result.subsection}")
        print(f"Paragraphs extracted: {result.paragraph_count}")
        print(f"Charts extracted: {result.chart_count}")
        print(f"Chunks created: {result.chunk_count}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print("-" * 50)
        
        print("\nChunk IDs:")
        for i, chunk_id in enumerate(result.chunk_ids):
            print(f"  {i+1}. {chunk_id}")
        
        print("\nChart IDs:")
        for i, chart_id in enumerate(result.chart_ids):
            print(f"  {i+1}. {chart_id}")
    
    print("\nProcessing complete!")


def main():
    """Main function."""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python example.py <path_to_pdf>")
        return
    
    pdf_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return
    
    # Check if file is a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"Error: Not a PDF file: {pdf_path}")
        return
    
    # Process the PDF
    example_process_pdf(pdf_path)


if __name__ == "__main__":
    main()
