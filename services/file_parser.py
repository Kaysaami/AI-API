# services/file_parser.py
import spacy
import re
import logging
import unicodedata
from dateutil.parser import parse as parse_date
from datetime import datetime
from dateutil.relativedelta import relativedelta
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader
import pdfplumber 
from werkzeug.datastructures import FileStorage
from typing import Dict, Tuple, List
import io
from lxml import etree
import textwrap
import docx.oxml
from services.shared_utilities import (
    SECTION_SYNONYMS,
    HEADER_KEYWORDS,
    SECTION_PATTERNS,
    STANDARD_SECTIONS,
    TECH_SYNONYMS
)
from contextlib import contextmanager
from docx.oxml import OxmlElement

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ENCODING_FALLBACKS = ['utf-8', 'latin-1', 'windows-1252']

try:
    nlp = spacy.load("en_core_web_lg")
except OSError:
    raise RuntimeError("Language model required. Run: python -m spacy download en_core_web_lg")

# Optional custom skill patterns for the entity ruler.
skill_patterns = [
    {"label": "SKILL", "pattern": [{"LOWER": "python"}]},
    {"label": "SKILL", "pattern": [{"LOWER": {"IN": ["javascript", "js"]}}]}
]

if not nlp.has_pipe("entity_ruler"):
    ruler = nlp.add_pipe("entity_ruler", after="ner")
    ruler.add_patterns(skill_patterns)

def extract_text(file: FileStorage) -> Dict[str, str]:
    """
    Reads the file, extracts and cleans text, then enriches metadata.
    Returns a dictionary with keys: 'full_text', 'metadata', 'file_type'.
    """
    validate_file(file)
    try:
        file.stream.seek(0)
        file_content = file.stream.read()
        file.stream.seek(0)
        if file.filename.lower().endswith('.pdf'):
            raw_text, metadata = extract_pdf_text(file_content)
        elif file.filename.lower().endswith('.docx'):
            raw_text, metadata = extract_docx_text(file_content)
        else:
            raise ValueError("Unsupported file format")
        cleaned_text = clean_text(raw_text)
        enriched_metadata = parse_structured_data(cleaned_text, metadata)
        return {
            'full_text': cleaned_text,
            'metadata': enriched_metadata,
            'file_type': file.content_type
        }
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}", exc_info=True)
        raise ValueError(f"Text extraction error: {str(e)}")

def validate_file(file: FileStorage):
    """
    Validates that a file is provided, is non-empty, does not exceed the maximum size,
    and has an allowed file extension.
    """
    if not file or not file.stream:
        raise ValueError("No file provided")
    file.stream.seek(0, 2)
    file_size = file.stream.tell()
    file.stream.seek(0)
    if file_size == 0:
        raise ValueError("Empty file")
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"File size exceeds {MAX_FILE_SIZE//1024//1024}MB limit")
    if not allowed_file(file.filename):
        raise ValueError("Unsupported file type")

def allowed_file(filename: str) -> bool:
    """
    Checks if the file extension is among the supported types.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx'}


def normalize_special_chars(text: str) -> str:
    """
    Replaces common PDF special-character artifacts.
    """
    return text.replace(u"\ufffd", "'").replace(u"\xad", "")

def extract_pdf_text(file_content: bytes) -> tuple:
    """
    Extracts text from a PDF file by reconstructing lines and detecting sections.
    Returns a tuple of (extracted_text, metadata).
    """
    metadata = {'tables': []}
    section_texts = {}
    current_section = "raw"  # Default section
    section_texts[current_section] = []  # initialize raw section

    try:
        with pdfplumber.open(io.BytesIO(file_content)) as pdf:
            if pdf.metadata:
                metadata.update(pdf.metadata)
                
            for page in pdf.pages:
                words = page.extract_words(use_text_flow=True)
                if not words:
                    continue

                # Group words into lines based on vertical position
                lines = []
                current_line = []
                current_y = None
                y_tolerance = 3
                
                for word in words:
                    word_top = float(word.get('top', 0))
                    if current_y is None:
                        current_y = word_top
                        current_line.append(word)
                    else:
                        if abs(word_top - current_y) <= y_tolerance:
                            current_line.append(word)
                        else:
                            current_line_sorted = sorted(current_line, key=lambda w: float(w.get('x0', 0)))
                            line_text = " ".join(w['text'] for w in current_line_sorted)
                            lines.append(line_text)
                            current_line = [word]
                            current_y = word_top
                if current_line:
                    current_line_sorted = sorted(current_line, key=lambda w: float(w.get('x0', 0)))
                    line_text = " ".join(w['text'] for w in current_line_sorted)
                    lines.append(line_text)

                # Process each line for section detection
                for line in lines:
                    stripped_line = line.strip()
                    if not stripped_line:
                        continue

                    if _is_section_header(stripped_line):
                        # Normalize the header text
                        new_section = _normalize_section_name(stripped_line)
                        logger.debug("Detected section header: '%s' normalized to '%s'", stripped_line, new_section)
                        # If this section already exists, we don't overwrite it—just switch current section
                        if new_section not in section_texts:
                            section_texts[new_section] = []
                        current_section = new_section
                    else:
                        # Append non-header lines to the current section
                        section_texts.setdefault(current_section, []).append(stripped_line)

                # Process tables (append table text to current section)
                tables = page.extract_tables()
                if tables:
                    metadata['tables'].extend(tables)
                    for table in tables:
                        table_text = "\n".join(" | ".join(str(cell) for cell in row if cell) for row in table)
                        section_texts.setdefault(current_section, []).append(table_text)

        # Fallback: if no text was extracted, use PdfReader as a backup
        if not any(section_texts.values()):
            reader = PdfReader(io.BytesIO(file_content))
            raw_text = "\n".join(page.extract_text() for page in reader.pages)
            section_texts = {"raw": [raw_text]}

        # Join the sections into a final text block with markers for non-raw sections
        final_text = ""
        for section, lines in section_texts.items():
            if lines:  # Only add non-empty sections
                section_content = "\n".join(lines).strip()
                if section != "raw":
                    final_text += f"\n## {section.capitalize()} ##\n{section_content}\n"
                else:
                    final_text += f"{section_content}\n"

        return final_text.strip(), metadata

    except Exception as e:
        logger.error("PDF extraction error: %s", e)
        raise

def _is_section_header(line: str) -> bool:
    """
    Determines if a line is likely a section header based on heuristics.
    Uses HEADER_KEYWORDS, SECTION_PATTERNS, and checks for uppercase.
    """
    # Skip empty lines or lines that are too long to be headers.
    if not line or len(line.split()) > 10:
        return False

    line_lower = line.lower()

    # Heuristics:
    # 1. The line is all uppercase (and short)
    is_uppercase = line.isupper()
    is_short = len(line.split()) <= 5
    # 2. The line contains any known header keyword or starts with one
    has_keyword = any(keyword.lower() in line_lower for keyword in HEADER_KEYWORDS)
    starts_with_keyword = any(line_lower.startswith(keyword.lower()) for keyword in HEADER_KEYWORDS)
    # 3. The line matches any known section pattern
    has_pattern_match = any(pattern.match(line) for pattern, _ in SECTION_PATTERNS)
    
    return (is_uppercase and is_short) or has_keyword or starts_with_keyword or has_pattern_match

def _normalize_section_name(header: str) -> str:
    """
    Normalizes section header text to a standard format using SECTION_SYNONYMS.
    Also attempts partial matching and pattern matching.
    """
    # Remove special characters and extra whitespace
    clean_header = re.sub(r'[^\w\s]', '', header.lower()).strip()
    
    # First, try an exact match in SECTION_SYNONYMS.
    if clean_header in SECTION_SYNONYMS:
        return SECTION_SYNONYMS[clean_header]
    
    # Then try partial matches
    for syn, standard in SECTION_SYNONYMS.items():
        if clean_header in syn or syn in clean_header:
            return standard
            
    # Check if any SECTION_PATTERNS match
    for pattern, name in SECTION_PATTERNS:
        if pattern.match(header):
            return name
            
    # Remove common filler words
    words_to_remove = {'section', 'and', 'the', 'my'}
    header_words = [word for word in clean_header.split() if word not in words_to_remove]
    
    if not header_words:
        return clean_header
        
    normalized = ' '.join(header_words)
    
    # Final check: if the normalized header is in the standard list, return it.
    if normalized in STANDARD_SECTIONS:
        return normalized
        
    return normalized

@contextmanager
def secure_docx_parsing():
    """
    Context manager that temporarily sets up a secure XML parser for DOCX files.
    """
    original_parse_xml = docx.oxml.parse_xml
    secure_parser = etree.XMLParser(resolve_entities=False)
    docx.oxml.parse_xml = lambda xml: etree.fromstring(xml, parser=secure_parser)
    try:
        yield
    finally:
        docx.oxml.parse_xml = original_parse_xml

def extract_docx_text(file_content: bytes) -> Tuple[str, dict]:
    """
    Extracts text from a DOCX file using python-docx with secure XML parsing.
    Returns the extracted text along with metadata.
    """
    metadata = {'author': None, 'created': None, 'modified': None, 'tables': []}
    text_parts = []
    with secure_docx_parsing():
        try:
            doc = Document(io.BytesIO(file_content))
            metadata.update({
                'author': doc.core_properties.author,
                'created': doc.core_properties.created.isoformat() if doc.core_properties.created else None,
                'modified': doc.core_properties.modified.isoformat() if doc.core_properties.modified else None,
            })
            for element in iter_block_items(doc):
                if isinstance(element, Paragraph):
                    text = process_paragraph(element)
                    if text:
                        text_parts.append(text)
                elif isinstance(element, Table):
                    structured_table = [[cell.text.strip() for cell in row.cells] for row in element.rows]
                    metadata['tables'].append(structured_table)
                    text_parts.append(process_docx_table(element))
        except Exception as e:
            logger.error(f"DOCX parsing failed: {str(e)}")
            raise ValueError("Invalid DOCX file structure")
    return '\n'.join(text_parts), metadata

def process_paragraph(para: Paragraph) -> str:
    """
    Processes a DOCX paragraph and returns its text.
    If the paragraph is detected as a section header based on various signals,
    the text is wrapped with '##' markers for later section extraction.
    """
    text = para.text.strip()
    if not text:
        return ""
    
    # If the paragraph is part of a bullet list, return it with a bullet.
    if para.style.name.startswith('List'):
        return f"• {text}"
    
    # Get the style name in lowercase (if available).
    style_name = para.style.name.lower() if para.style.name else ""
    
    # --- Condition 1: Explicit Heading Styles ---
    # If the style name indicates a heading or title, mark it as a header.
    if "heading" in style_name or "title" in style_name:
        return f"\n## {text} ##\n"
    
    # --- Condition 2: Known Section Header Terms ---
    # Normalize the text and check against known section header synonyms.
    normalized_text = text.lower().strip()
    for synonym in SECTION_SYNONYMS.keys():
        if synonym in normalized_text:
            return f"\n## {text} ##\n"
    
    # --- Condition 3: Formatting Heuristics ---
    # Check if the text is short (e.g., 8 words or less) and either bold or large font.
    is_bold = any(getattr(run, "bold", False) for run in para.runs) if para.runs else False
    font_size = para.style.font.size.pt if para.style.font.size else 0
    word_count = len(text.split())
    if (is_bold or font_size > 11) and word_count <= 8:
        return f"\n## {text} ##\n"
    
    # --- Condition 4: All Uppercase and Short ---
    # Sometimes headers are written entirely in uppercase.
    if text.isupper() and word_count <= 8:
        return f"\n## {text} ##\n"
    
    # Otherwise, return the text as-is.
    return text


def process_docx_table(table: Table) -> str:
    """
    Converts a DOCX table to a structured text format.
    """
    return '\n'.join(' | '.join(cell.text.strip() for cell in row.cells) for row in table.rows)

def process_pdf_table(table_data: list) -> str:
    """
    Converts extracted PDF table data to a structured text format.
    """
    return '\n'.join('\t'.join(str(cell) for cell in row) for row in table_data)

def iter_block_items(parent) -> List:
    """
    Yields each paragraph and table child within a DOCX document (or element),
    in document order.
    """
    items = []
    for child in parent.element.body.iterchildren():
        if child.tag.endswith('}p'):
            from docx.text.paragraph import Paragraph
            items.append(Paragraph(child, parent))
        elif child.tag.endswith('}tbl'):
            from docx.table import Table
            items.append(Table(child, parent))
    return items

def normalize_unicode_text(text: str) -> str:
    # Use our shared normalization function and further remove extra spaces and symbols.
    text = normalize_unicode(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fix_hyphenation(text: str) -> str:
    # Fix broken-word hyphenation across line breaks.
    return re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1 \2\n', text)

def remove_header_footer(text: str) -> str:
    lines = text.split('\n')
    if len(lines) < 4:
        return text
    region_size = max(2, len(lines) // 10)
    start_region = [line.strip() for line in lines[:region_size] if line.strip()]
    end_region = [line.strip() for line in lines[-region_size:] if line.strip()]
    common_elements = set(start_region) & set(end_region)
    filtered = []
    for line in lines:
        if line.strip() in common_elements:
            continue
        filtered.append(line)
    return '\n'.join(filtered)

def clean_whitespace(text: str) -> str:
    text = re.sub(r'\t+', ' ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def clean_text(text: str) -> str:
    # Run through a series of cleaning functions
    text = normalize_unicode_text(text)  # Update 1
    text = fix_hyphenation(text)
    text = remove_header_footer(text)
    text = clean_whitespace(text)
    return text

def normalize_unicode(text: str) -> str:
    """
    Normalizes Unicode characters and replaces special quotes, dashes, and ellipses.
    """
    text = unicodedata.normalize('NFKC', text)
    replacements = {
        '“': '"', '”': '"', "‘": "'", "’": "'",
        '–': '-', '—': '-', '…': '...'
    }
    for char, repl in replacements.items():
        text = text.replace(char, repl)
    return text


def standardize_bullets(text: str) -> str:
    """
    Normalizes various bullet characters into a standard bullet symbol.
    """
    return re.sub(r'(^|\n)\s*[\•▪∙◦⁃→]\s*', '\n• ', text)

def detect_sections(text: str) -> str:
    """
    Detects section headers in the text by marking them with '## ... ##'.
    """
    text = re.sub(r'\nSECTION_HEADER:(.+?)\n', r'\n## \1 ##\n', text)
    text = re.sub(r'(?i)(\n\s*)(education|experience|skills)(\s*\n)', r'\n## \2 ##\n', text)
    return re.sub(r'(?i)(\n)([A-Z][A-Z\s]+[A-Z])(\n)', r'\n## \2 ##\n', text)

def replace_section_synonyms(text: str) -> str:
    """
    Replaces section header synonyms with their canonical forms.
    """
    for synonym, standard in SECTION_SYNONYMS.items():
        text = re.sub(re.escape(synonym), standard, text, flags=re.IGNORECASE)
    return text

def normalize_skills(text: str) -> str:
    """
    Standardizes technology skill terms using synonyms.
    """
    for syn, canon in TECH_SYNONYMS.items():
        text = re.sub(rf'\b{re.escape(syn)}\b', canon, text, flags=re.I)
    return text

def normalize_dates(text: str) -> str:
    """
    Normalizes date formats, converting various date patterns into a standard format.
    """
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', r'\1/\2/\3', text)
    months = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    return re.sub(rf'\b{months}\s+(\d{{4}})\b', r'\1 \2', text, flags=re.IGNORECASE)

def remove_isolated_numbers(text: str) -> str:
    """
    Removes lines that consist solely of numbers (often page numbers).
    """
    return re.sub(r'\n\s*\d+\s*\n', '\n', text)

def is_encrypted_pdf(file_content: bytes) -> bool:
    """
    Checks whether the provided PDF file content is encrypted.
    """
    try:
        reader = PdfReader(io.BytesIO(file_content))
        return reader.is_encrypted
    except Exception as e:
        logger.warning(f"Encryption check failed: {e}")
        return False


def parse_structured_data(text: str, metadata: dict) -> dict:
    """
    Parses the full text using spaCy to extract entities and experience sections.
    Adds an "entities" key and an "experience" key to the metadata.
    """
    doc = nlp(text)
    metadata["entities"] = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]
    metadata["experience"] = extract_experience_sections(text)
    return metadata

def extract_experience_sections(text: str) -> list:
    """
    Extracts experience entries from text.
    Looks for bullet points and associated durations.
    """
    experiences = []
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    pattern = re.compile(
        r'(?:^|\n)\s*•\s*(.+?)\s*\(([^)]+)\)\s*((?:\n\s*[-•*]\s*.+)+)',
        flags=re.IGNORECASE | re.DOTALL
    )
    for match in pattern.finditer(text):
        title = match.group(1).strip()
        duration_str = match.group(2).strip()
        bullets_text = match.group(3).strip()
        duration = parse_duration(duration_str) if duration_str else None
        bullets = extract_achievements(bullets_text)
        experiences.append({
            "title": title,
            "duration": duration,
            "bullets": bullets
        })
    return experiences

def parse_duration(duration_str: str) -> dict:
    """
    Attempts to extract start and end dates from a duration string.
    Returns a dictionary with the raw duration and parsed dates.
    """
    try:
        dates = re.findall(r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})[\w\s,.-]*\d{4}\b)', duration_str)
        start = parse_date(dates[0], fuzzy=True) if dates else None
        end = parse_date(dates[1], fuzzy=True) if len(dates) > 1 else "Present"
        return {
            "raw": duration_str,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if isinstance(end, datetime) else end
        }
    except Exception:
        return {"raw": duration_str}

def extract_achievements(text: str) -> list:
    """
    Extracts bullet-point achievements from a block of text.
    """
    return [line.strip().lstrip('-•*') for line in text.split('\n') if re.match(r'^[\-\•\*]', line.strip())]
