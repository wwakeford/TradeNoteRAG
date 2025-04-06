"""
Command-line interface for PDF processing.
"""

import os
import argparse
import time
from typing import List, Optional
from .processor import PDFProcessor
from .config import get_settings


def get_pdf_files(directory: str) -> List[str]:
    """
    Get all PDF files in a directory.
    
    Args:
        directory (str): Directory path
        
    Returns:
        List[str]: List of PDF file paths
    """
    pdf_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    return pdf_files


def process_single_pdf(
    pdf_path: str,
    section: str = "",
    subsection: str = "",
    verbose: bool = False
) -> None:
    """
    Process a single PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        section (str): Optional section override
        subsection (str): Optional subsection override
        verbose (bool): Enable verbose output
    """
    settings = get_settings()
    
    processor = PDFProcessor(
        supabase_url=settings.SUPABASE_URL,
        supabase_key=settings.SUPABASE_KEY,
        openai_api_key=settings.OPENAI_API_KEY,
        min_image_size=settings.MIN_IMAGE_SIZE,
        chart_detection_threshold=settings.CHART_DETECTION_THRESHOLD,
        proximity_threshold=settings.PROXIMITY_THRESHOLD,
        temp_dir=settings.TEMP_DIR
    )
    
    print(f"Processing {pdf_path}...")
    start_time = time.time()
    
    result = processor.process_pdf(
        pdf_path=pdf_path,
        section=section,
        subsection=subsection
    )
    
    elapsed_time = time.time() - start_time
    
    if result.error:
        print(f"Error: {result.error}")
    else:
        print(f"Successfully processed {pdf_path}")
        print(f"Note ID: {result.note_id}")
        print(f"Title: {result.title}")
        print(f"Section: {result.section}")
        print(f"Subsection: {result.subsection}")
        print(f"Paragraphs: {result.paragraph_count}")
        print(f"Charts: {result.chart_count}")
        print(f"Chunks: {result.chunk_count}")
        
        if verbose:
            print(f"Chunk IDs: {result.chunk_ids}")
            print(f"Chart IDs: {result.chart_ids}")
    
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print("-" * 50)


def process_directory(directory: str, verbose: bool = False) -> None:
    """
    Process all PDF files in a directory.
    
    Args:
        directory (str): Directory path
        verbose (bool): Enable verbose output
    """
    pdf_files = get_pdf_files(directory)
    
    if not pdf_files:
        print(f"No PDF files found in {directory}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    for i, pdf_path in enumerate(pdf_files):
        print(f"Processing file {i+1}/{len(pdf_files)}")
        process_single_pdf(pdf_path, verbose=verbose)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Process trading PDFs for analysis")
    
    # Define arguments
    parser.add_argument(
        "path",
        help="Path to a PDF file or directory containing PDFs"
    )
    
    parser.add_argument(
        "--section",
        help="Override the section category",
        default=""
    )
    
    parser.add_argument(
        "--subsection",
        help="Override the subsection category",
        default=""
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if path exists
    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}")
        return
    
    # Process PDF file or directory
    if os.path.isfile(args.path):
        process_single_pdf(
            pdf_path=args.path,
            section=args.section,
            subsection=args.subsection,
            verbose=args.verbose
        )
    else:
        process_directory(
            directory=args.path,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
