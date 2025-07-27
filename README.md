# Adobe Challenge 1A - PDF Heading Extraction System

A high-accuracy PDF parsing system for extracting titles and hierarchical headings (H1, H2, H3) from multilingual PDFs.

## 🎯 Features

- **Title Extraction**: Automatically detects document titles from the first page
- **Hierarchical Heading Detection**: Identifies H1, H2, and H3 headings based on font properties
- **Multilingual Support**: Handles documents with mixed scripts (English, Devanagari, etc.)
- **Noise Filtering**: Removes headers, footers, and other non-content elements
- **Font Clustering**: Uses machine learning to dynamically classify heading tiers
- **Docker Ready**: Containerized solution for easy deployment

## 🚀 Quick Start
### Docker Deployment

```bash
# Build Docker image
docker build -t adobe-challenge-1a .

# Run with mounted volumes
docker run -v /path/to/input:/app/input -v /path/to/output:/app/output adobe-challenge-1a
# For example: docker run -v "d:/adobe-final/Adobe_1A/sample_dataset/pdfs:/app/input" -v "d:/adobe-final/Adobe_1A/sample_dataset/outputs:/app/output" adobe-challenge-1a
```

## 📁 Project Structure

```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # JSON files provided as outputs.
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile           # Docker container configuration
├── process_pdfs.py      # Sample processing script
└── README.md           # This file

```

## 🔧 How It Works

### 1. Font Analysis
- Extracts all text spans with font properties (size, boldness, position)
- Uses K-means clustering to identify font size tiers
- Calculates median body text size for relative comparison

### 2. Title Detection
- Analyzes first page for largest font sizes
- Prefers bold text for title candidates
- Concatenates multiple title lines in correct order
- Filters out common non-title patterns

### 3. Heading Detection
- **H1**: Font size ≥ 1.4× median, bold, ≥14pt
- **H2**: Font size ≥ 1.2× median, bold, ≥12pt  
- **H3**: Font size ≥ 1.1× median, (bold or ≥11pt)

### 4. Post-Processing
- Merges split headings across multiple spans
- Removes duplicates and noise
- Ensures proper hierarchical structure
- Filters out headers/footers

## 📊 Output Format

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Main Heading",
      "page": 1
    },
    {
      "level": "H2", 
      "text": "Subheading",
      "page": 2
    }
  ]
}
```



### Font Detection Criteria

- **Bold Detection**: Checks font name and PyMuPDF flags
- **Size Thresholds**: Relative to median body text size
- **Clustering**: Uses K-means with 3 clusters max

## 🔍 Performance

- **Speed**: < 10 seconds per 50-page PDF
- **Memory**: < 200MB model size
- **Accuracy**: High precision for well-structured documents
- **Robustness**: Handles various PDF formats and layouts

## 🛠️ Dependencies

- **PyMuPDF**: PDF text extraction and font analysis
- **scikit-learn**: Font size clustering
- **numpy**: Numerical operations
- **pandas**: Data manipulation (optional)

## 🐳 Docker Constraints

- **Architecture**: AMD64 compatible
- **Memory**: 16GB RAM limit
- **CPU**: 8-core limit
- **No Internet**: Offline operation required
- **Input/Output**: `/app/input` and `/app/output` directories
