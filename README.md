# Trading Notes Analysis System

An automated system that extracts, processes, and indexes trading notes from PDFs, enabling intelligent query and analysis through a RAG system that understands trading methodology and properly references charts alongside explanatory text.

## Features

- PDF processing pipeline for text and chart extraction
- Semantic analysis of trading content
- Chart-paragraph relationship detection 
- LLM-assisted metadata generation
- Vector database integration for retrieval
- Content chunking for optimal RAG performance
- Streamlit interface for PDF uploading and processing

## Architecture

The system consists of five main components:

1. **PDF Processing Pipeline**: Extracts text and images from PDFs
   - Paragraph-level text extraction
   - Chart image extraction with positional data
   - Structure preservation for document coherence

2. **Content Enhancement System**: Adds metadata and creates semantic chunks
   - Paragraph-level semantic analysis and chunking
   - Chart-paragraph linking and bidirectional references
   - LLM-assisted metadata generation

3. **Vector Database Integration**: Stores and indexes content for retrieval
   - Embedding generation for text and chart descriptions
   - Relationship preservation between notes, chunks, and charts
   - Vector search optimization

4. **RAG Query Engine**: Handles queries and generates responses (future enhancement)
   - Retrieval system for relevant chunks and charts
   - Response generation incorporating retrieved content
   - Trading methodology-based analysis

5. **User Interface**: Provides interaction points for document uploading and querying
   - Document upload and processing workflow
   - Query interface for trading analysis (future enhancement)
   - Response visualization with reference charts (future enhancement)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/wwakeford/TradeNoteRAG.git
   cd trading_notes_analyzer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Command Line Interface

Process a single PDF file:
```
python -m pdf_processor.cli /path/to/your/trading_note.pdf
```

Process a directory of PDF files:
```
python -m pdf_processor.cli /path/to/your/pdf_directory
```

Optional arguments:
```
--section "Technical analysis"    # Specify a section category
--subsection "Price action"       # Specify a subsection
--verbose                         # Enable verbose output
```

### Streamlit Interface

Run the Streamlit web app:
```
streamlit run pdf_uploader.py
```

This will open a web interface where you can:
- Upload PDF files
- Configure processing parameters
- View processing results

### Python API

```python
from pdf_processor.processor import PDFProcessor
from pdf_processor.config import get_settings

# Get default settings
settings = get_settings()

# Initialize the processor
processor = PDFProcessor(
    supabase_url=settings.SUPABASE_URL,
    supabase_key=settings.SUPABASE_KEY,
    openai_api_key=settings.OPENAI_API_KEY
)

# Process a PDF
result = processor.process_pdf(
    pdf_path="path/to/your/trading_note.pdf",
    section="Technical analysis",  # Optional
    subsection="Price action"      # Optional
)

# Access the results
print(f"Note ID: {result.note_id}")
print(f"Paragraphs: {result.paragraph_count}")
print(f"Charts: {result.chart_count}")
```

## Database Schema

The system uses Supabase with the following tables:

1. `trading_notes`: Stores original PDFs with metadata
2. `note_chunks`: Stores semantic chunks with embeddings
3. `chart_images`: Stores extracted chart images with metadata

## Testing

Run the test suite:
```
pytest
```

## Future Enhancements

- OCR integration for chart text extraction
- Multi-modal retrieval with text and image embedding
- Advanced chart pattern recognition
- User feedback loop for retrieval improvement
- Fine-tuned embedding models for trading content

## License

MIT
