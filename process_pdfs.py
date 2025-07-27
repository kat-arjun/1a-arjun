#!/usr/bin/env python3
import fitz  # PyMuPDF
import os
import json
import re
import time
import logging
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions for Classification ---

def get_common_font_style(page_blocks):
    """Calculates the most common font size and family on a page."""
    if not page_blocks:
        return 10.0, "Helvetica" # Default values
    
    font_sizes = defaultdict(int)
    font_faces = defaultdict(int)
    
    for block in page_blocks:
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                font_sizes[round(span['size'])] += 1
                font_faces[span['font']] += 1
                
    if not font_sizes:
        return 10.0, "Helvetica"
        
    common_size = float(max(font_sizes, key=font_sizes.get))
    common_face = max(font_faces, key=font_faces.get)
    return common_size, common_face

def classify_block(block_text, span, common_font_size, page_blocks_count):
    """
    Assigns a 'heading_score' to a text block based on a lightweight scoring system.
    Returns a score and a potential heading level.
    """
    score = 0
    level = 1
    
    # 1. Font Size: Larger than common size is a strong indicator.
    if span['size'] > common_font_size + 0.5:  # Reduced threshold
        score += 30 * (span['size'] / common_font_size)
        level = max(1, min(6, int(6 - (span['size'] - common_font_size) / 1.5)))

    # 2. Font Weight: Bold text is a strong indicator.
    if "bold" in span['font'].lower():
        score += 25

    # 3. Capitalization: All caps or title case.
    if block_text.isupper() and len(block_text) > 3:  # Reduced length requirement
        score += 20
    elif block_text.istitle() and len(block_text) > 3:
        score += 15

    # 4. Brevity: Headers are short.
    if len(block_text) < 150:  # Increased threshold
        score += 10
    
    # 5. Numbering Schemes (Rule-based pattern recognition)
    # Matches "1.", "1.1", "1.1.1", "Chapter 1", "Section A", "Appendix A" etc.
    match = re.match(r'^(?:Chapter|Section|Appendix)\s+([A-Z0-9]+)|^(\d+\.?\d*\.?)\s+', block_text, re.IGNORECASE)
    if match:
        score += 40
        # Determine hierarchy level from numbering
        num_str = match.group(2) or "1"
        level = num_str.count('.') + 1
    
    # 6. Check for common heading patterns
    heading_patterns = [
        r'^(Summary|Background|Timeline|Overview|Introduction|Conclusion)\s*:?\s*$',
        r'^(Phase\s+[IVX]+|Part\s+\d+)',
        r'^[A-Z][a-z]+\s*:$',  # Single word followed by colon
        r'^[A-Z\s]+:$'  # All caps followed by colon
    ]
    
    for pattern in heading_patterns:
        if re.match(pattern, block_text, re.IGNORECASE):
            score += 30
            break
    
    # 7. Position-based scoring (first few blocks on a page are more likely headings)
    if page_blocks_count <= 3:
        score += 10

    # Heuristic: A block is likely a heading if its score is high.
    # Reduced threshold for better detection
    if score > 35:  # Reduced from 50
        return 'heading', score, max(1, min(6, level))
    else:
        return 'paragraph', score, 0


def structure_content(classified_blocks):
    """
    Analyzes classified blocks to build document hierarchy.
    Handles lists and basic table-like structures.
    Page numbering: Title page (page 0) is not counted, content starts from page 1
    """
    structured_data = []
    current_section = None
    
    for block in classified_blocks:
        block_type = block['type']
        block_text = block['text'].strip()
        # Page numbering: PDF page 0 = title (not numbered), PDF page 1 = page 1, PDF page 2 = page 2, etc.
        pdf_page = block.get('page', 0)  # This is the actual PDF page number (0-based)
        display_page = pdf_page  # PDF page 1 becomes display page 1, PDF page 2 becomes display page 2, etc.

        if not block_text:
            continue
            
        # Regex for list items (e.g., *, -, 1., a))
        is_list_item = re.match(r'^\s*([*\-•]|\d+\.|\w\))\s+', block_text)
        
        if block_type == 'heading':
            # Finalize the previous section before starting a new one
            if current_section:
                structured_data.append(current_section)
                
            current_section = {
                "heading": block_text,
                "level": block['level'],
                "page_number": display_page,
                "content": []
            }
        elif current_section:
            if is_list_item:
                 # Simple list handling: append as distinct content items
                current_section["content"].append({"type": "list_item", "text": block_text})
            else:
                # Append paragraph to the current section
                # Coalesce consecutive paragraphs for better readability
                if current_section["content"] and current_section["content"][-1]["type"] == "paragraph":
                    current_section["content"][-1]["text"] += " " + block_text
                else:
                    current_section["content"].append({"type": "paragraph", "text": block_text})
        else:
            # Create a default section for content without headings
            if not structured_data or structured_data[-1]["heading"] != "Document Content":
                structured_data.append({
                    "heading": "Document Content",
                    "level": 1,
                    "page_number": display_page,
                    "content": []
                })
            
            if is_list_item:
                structured_data[-1]["content"].append({"type": "list_item", "text": block_text})
            else:
                structured_data[-1]["content"].append({"type": "paragraph", "text": block_text})
        
    if current_section:
        structured_data.append(current_section)
        
    return structured_data

def extract_document_title(doc):
    """
    Attempts to extract document title from the first page.
    Combines multiple text spans to form complete title.
    """
    if len(doc) == 0:
        return "Untitled Document"
    
    first_page = doc[0]
    page_content = first_page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)
    
    # Collect all text from the first page with font sizes
    text_blocks = []
    
    for block in page_content.get('blocks', []):
        if block['type'] == 0:  # Text block
            for line in block['lines']:
                line_text = ""
                max_font_size = 0
                for span in line['spans']:
                    line_text += span['text']
                    max_font_size = max(max_font_size, span['size'])
                
                if line_text.strip() and len(line_text.strip()) > 3:
                    text_blocks.append({
                        'text': line_text.strip(),
                        'font_size': max_font_size,
                        'y_pos': line['bbox'][1]  # Y position for ordering
                    })
    
    if not text_blocks:
        return "Untitled Document"
    
    # Sort by Y position (top to bottom)
    text_blocks.sort(key=lambda x: x['y_pos'])
    
    # Find the largest font size in the top portion of the page
    top_blocks = text_blocks[:min(10, len(text_blocks))]  # Consider first 10 blocks
    max_font_size = max(block['font_size'] for block in top_blocks)
    
    # Collect all text with the largest font size or close to it
    title_parts = []
    for block in top_blocks:
        if block['font_size'] >= max_font_size - 1:  # Allow slight variation
            title_parts.append(block['text'])
    
    # Combine title parts
    title = " ".join(title_parts).strip()
    
    # Clean up the title
    title = re.sub(r'\s+', ' ', title)  # Remove extra whitespace
    
    return title if title else "Untitled Document"

# --- Main Processing Function ---

# Add these imports at the top
import psutil
import jsonschema
from datetime import datetime

# Add these validation functions
def validate_processing_time(start_time, page_count, max_time_per_50_pages=10):
    """Validate processing time meets requirements."""
    processing_time = time.time() - start_time
    expected_max_time = (page_count / 50) * max_time_per_50_pages
    
    if processing_time > expected_max_time:
        logger.warning(f"Processing time {processing_time:.2f}s exceeds limit for {page_count} pages")
        return False
    return True

def validate_memory_usage(max_memory_gb=16):
    """Validate memory usage stays within limits."""
    process = psutil.Process()
    memory_usage_gb = process.memory_info().rss / (1024**3)
    
    if memory_usage_gb > max_memory_gb:
        logger.warning(f"Memory usage {memory_usage_gb:.2f}GB exceeds {max_memory_gb}GB limit")
        return False
    return True

def validate_output_schema(output_data, schema_path="sample_dataset/schema/output_schema.json"):
    """Validate output against schema."""
    try:
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            jsonschema.validate(output_data, schema)
            return True
    except Exception as e:
        logger.warning(f"Schema validation failed: {e}")
        return False
    return True

# Modify the process_pdf function to include validations
def process_pdf(pdf_path):
    try:
        logger.info(f"Processing {os.path.basename(pdf_path)}...")
        start_time = time.time()
        
        # Memory check before processing
        if not validate_memory_usage():
            logger.error("Memory limit exceeded before processing")
            return
        
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        
        if len(doc) > 100: # Early termination for excessively large docs if needed
            logger.warning(f"Skipping very large document: {os.path.basename(pdf_path)} with {len(doc)} pages.")
            doc.close()
            return

        # Extract document title
        document_title = extract_document_title(doc)
        
        all_blocks = []
        for page_num, page in enumerate(doc):
            # Use 'dict' format for rich metadata including font, size, color
            page_content = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)
            common_size, _ = get_common_font_style(page_content.get('blocks', []))
            
            block_count = 0
            for block in page_content.get('blocks', []):
                if block['type'] == 0: # Text block
                    block_text = ""
                    # The first span determines the style for the whole block for simplicity
                    first_span = block['lines'][0]['spans'][0] if block.get('lines') and block['lines'][0].get('spans') else {}
                    
                    for line in block['lines']:
                        for span in line['spans']:
                            block_text += span['text'] + " "
                    
                    block_text = block_text.strip()
                    if block_text and first_span:
                        block_type, _, level = classify_block(block_text, first_span, common_size, block_count)
                        all_blocks.append({
                            "text": block_text,
                            "type": block_type,
                            "level": level,
                            "page": page_num  # Keep 0-based for now, will convert in structure_content
                        })
                        block_count += 1
        
        # Structure the extracted and classified content
        structured_content = structure_content(all_blocks)
        
        # Calculate processing time and file size
        processing_time = round(time.time() - start_time, 2)
        file_size = os.path.getsize(pdf_path)
        
        # Generate outline from structured content
        outline = []
        for section in structured_content:
            if section['heading'] != "Document Content":  # Skip the default section
                # Map level numbers to H1, H2, H3, etc.
                level_map = {1: "H1", 2: "H2", 3: "H3", 4: "H4", 5: "H5", 6: "H6"}
                heading_level = level_map.get(section['level'], "H1")
                
                outline.append({
                    "level": heading_level,
                    "text": section['heading'],
                    "page": section['page_number']
                })
        
        # Generate JSON output in the new format (matching process_pdfs.py)
        output_data = {
            "title": document_title,
            "outline": outline
        }

        # Determine output directory
        if os.path.exists("/app/output"):
            # Docker environment
            output_dir = "/app/output"
        else:
            # Local environment
            output_dir = "sample_dataset/outputs"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write JSON output
        output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Finished {os.path.basename(pdf_path)} in {processing_time} seconds.")
        
        # Validate processing time
        if not validate_processing_time(start_time, page_count):
            logger.error(f"Processing time validation failed for {pdf_path}")
        
        # Validate output schema
        if not validate_output_schema(output_data):
            logger.error(f"Schema validation failed for {pdf_path}")
        
        # Memory check after processing
        if not validate_memory_usage():
            logger.warning("Memory limit exceeded after processing")
        
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
    finally:
        if 'doc' in locals():
            doc.close()

def main():
    """Main function to process PDFs in input directory."""
    # Check if running in Docker or locally
    if os.path.exists("/app/input"):
        # Docker environment
        input_dir = "/app/input"
        output_dir = "/app/output"
    else:
        # Local environment - use sample dataset
        input_dir = "sample_dataset/pdfs"
        output_dir = "sample_dataset/outputs"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        logger.warning("No PDF files found in the input directory.")
    else:
        # Leverage all available CPU cores
        num_cores = min(8, cpu_count())
        logger.info(f"Starting PDF processing with {num_cores} CPU cores.")
        
        with Pool(processes=num_cores) as pool:
            pool.map(process_pdf, pdf_files)
            
        logger.info("All PDF files have been processed.")

# --- Orchestrator ---

if __name__ == "__main__":
    main()