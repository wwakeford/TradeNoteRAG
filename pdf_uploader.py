"""
Streamlit web app for uploading and processing trading PDFs.
"""

import os
import tempfile
import time
import streamlit as st
from pdf_processor.processor import PDFProcessor
from pdf_processor.config import get_settings


def main():
    """Main function for the Streamlit app."""
    # Page title and description
    st.title("Trading Notes Analyzer")
    st.write("Upload your trading PDFs for processing and analysis")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Get settings
    settings = get_settings()
    
    # Settings form
    with st.sidebar.form("settings_form"):
        supabase_url = st.text_input(
            "Supabase URL",
            value=settings.SUPABASE_URL
        )
        
        supabase_key = st.text_input(
            "Supabase Key",
            value=settings.SUPABASE_KEY,
            type="password"
        )
        
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=settings.OPENAI_API_KEY,
            type="password"
        )
        
        min_image_size = st.slider(
            "Minimum Chart Size (px)",
            min_value=50,
            max_value=300,
            value=settings.MIN_IMAGE_SIZE,
            step=10
        )
        
        chart_detection_threshold = st.slider(
            "Chart Detection Threshold",
            min_value=0.1,
            max_value=1.0,
            value=settings.CHART_DETECTION_THRESHOLD,
            step=0.05
        )
        
        submitted = st.form_submit_button("Update Settings")
    
    # File uploader
    st.header("Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        accept_multiple_files=False
    )
    
    # Form for metadata
    with st.form("metadata_form"):
        st.write("Optional metadata (leave blank for auto-detection)")
        
        # Section selection
        section = st.selectbox(
            "Section",
            options=[
                "",  # Auto-detect
                "Long-term Fundamental analysis",
                "Long-term Technical analysis",
                "Short-term Technical analysis - day trading the S&P500",
                "Swing trading strategies",
                "Market psychology",
                "Trading indicators"
            ]
        )
        
        # Subsection input
        subsection = st.text_input(
            "Subsection",
            placeholder="E.g.: Price action patterns, Support and resistance, etc."
        )
        
        process_button = st.form_submit_button("Process PDF")
    
    # Handle PDF processing
    if uploaded_file is not None and process_button:
        # Display progress information
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                # Write uploaded file to temp file
                tmp_file.write(uploaded_file.getvalue())
                temp_path = tmp_file.name
            
            # Initialize processor
            progress_text.text("Initializing processor...")
            progress_bar.progress(10)
            
            processor = PDFProcessor(
                supabase_url=supabase_url,
                supabase_key=supabase_key,
                openai_api_key=openai_api_key,
                min_image_size=min_image_size,
                chart_detection_threshold=chart_detection_threshold,
                proximity_threshold=settings.PROXIMITY_THRESHOLD,
                temp_dir=settings.TEMP_DIR
            )
            
            # Process the PDF
            progress_text.text("Processing PDF... This may take a few minutes.")
            progress_bar.progress(20)
            
            start_time = time.time()
            result = processor.process_pdf(
                pdf_path=temp_path,
                section=section,
                subsection=subsection
            )
            elapsed_time = time.time() - start_time
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Display results
            progress_bar.progress(100)
            
            if result.error:
                st.error(f"Error processing PDF: {result.error}")
            else:
                progress_text.text("Processing complete!")
                
                # Display result in expandable sections
                st.subheader("Processing Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Processing Time", f"{elapsed_time:.2f} seconds")
                    st.metric("Paragraphs Extracted", result.paragraph_count)
                
                with col2:
                    st.metric("Charts Extracted", result.chart_count)
                    st.metric("Chunks Created", result.chunk_count)
                
                # Note details
                with st.expander("Note Details", expanded=True):
                    st.write(f"**Note ID:** {result.note_id}")
                    st.write(f"**Title:** {result.title}")
                    st.write(f"**Section:** {result.section}")
                    st.write(f"**Subsection:** {result.subsection}")
                
                # Chunk IDs
                with st.expander("Chunk IDs"):
                    for i, chunk_id in enumerate(result.chunk_ids):
                        st.write(f"{i+1}. `{chunk_id}`")
                
                # Chart IDs
                with st.expander("Chart IDs"):
                    for i, chart_id in enumerate(result.chart_ids):
                        st.write(f"{i+1}. `{chart_id}`")
                
                # Success message
                st.success("PDF processed successfully!")
                
                # Add a button to process another PDF
                if st.button("Process Another PDF"):
                    st.experimental_rerun()
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            
            # Clean up temporary file if it exists
            if 'temp_path' in locals():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
