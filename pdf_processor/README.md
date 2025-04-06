# PDF Processor Module

This module is responsible for extracting, processing, and analyzing text and charts from trading PDFs.

## Key Components

- **PDFExtractor**: Main class for extracting content from PDFs
- **ChartDetector**: Specialized class for detecting and extracting charts from PDFs
- **Analyzer**: Analyzes relationships between text and charts
- **Enhancer**: Enhances content with metadata
- **Database**: Integrates with Supabase for storage

## Chart Detection Improvements

The chart detection system now uses multiple methods to identify charts in PDFs:

1. **Embedded Images**: Extracts images directly embedded in the PDF
2. **Vectorized Charts**: Uses contour detection to identify chart-like regions
3. **Trading-Specific Detection**: Special algorithms optimized for trading charts
4. **Forced Extraction**: As a fallback, extracts the middle section of pages that might contain charts

## UUID Position Markers

Chart position markers are now proper UUIDs to ensure compatibility with the database's UUID column type.

## Usage

```python
from pdf_processor.extractor import PDFExtractor

# Initialize the extractor with custom settings
extractor = PDFExtractor(
    min_image_size=80,  # Minimum size in pixels
    force_chart_extraction=True,  # Force chart extraction for trading PDFs
    trading_chart_detection=True  # Enable trading-specific detection
)

# Process a PDF
paragraphs, charts = extractor.process_pdf("path/to/trading_note.pdf")

# Work with the extracted content
for chart in charts:
    print(f"Chart found on page {chart.page_num} with UUID: {chart.position_marker}")
```

## Testing

Run the test suite using pytest:

```bash
pytest tests/
```
