import re
import spacy
import logging
import json  # Import for json
from typing import Dict, List
from spacy.matcher import PhraseMatcher
import numpy as np
from spacy.tokens import Doc, Span  # For type annotations

# Import missing constants from shared_utilities
from .shared_utilities import (
    load_config,
    SKILL_CATEGORIES,
    TECH_SYNONYMS,
    KNOWN_SKILLS,
    HEADER_KEYWORDS,
    STANDARD_SECTIONS,
    SECTION_SYNONYMS
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def cosine_similarity(vec1, vec2) -> float:
    import numpy as np
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return (vec1 @ vec2) / (norm1 * norm2)

class SectionExtractor:
    def __init__(self, nlp: spacy.language.Language = None, config: dict = None):
        self.nlp = nlp or spacy.load("en_core_web_trf")
        self.config = config or {}
        # Load standard sections and synonyms from config or defaults.
        self.standard_sections = [s.lower() for s in self.config.get("standard_sections", [
            "career objective", "summary", "qualifications", "professional experience", 
            "education", "certifications", "skills", "projects"
        ])]
        # The synonyms dictionary maps alternate header strings to the canonical names.
        self.section_synonyms = {
            k.lower(): v.lower() for k, v in self.config.get("section_synonyms", {
                "education & certifications": "education",
                "certifications and licenses": "education",
                "work experience": "professional experience",
                "experience": "professional experience",
                "profile": "summary"
            }).items()
        }
        # Precompute header vectors for cosine similarity fallback.
        self.header_vectors = {sec: self.nlp(sec).vector for sec in self.standard_sections}

    def _normalize_header(self, text: str) -> str:
        """
        Normalize header text by stripping extra characters, punctuation, and common stopwords.
        """
        text = text.strip()
        # Remove common bullet characters and dashes.
        text = re.sub(r'[\*\-]+', '', text)
        # Remove punctuation (except for ampersands) and extra spaces.
        text = re.sub(r'[^\w\s&]', '', text)
        return text.lower().strip()

    def _is_header_line(self, line: str) -> bool:
        """
        Returns True if a given line is likely a section header.
        Heuristics include:
        - Short lines (<=10 words) that are all uppercase or in title case
        - The normalized line exactly matches a known section or synonym (from shared_utilities)
        - Fuzzy matching using cosine similarity against the precomputed header vectors.
        """
        stripped = line.strip()
        if not stripped:
            return False

        words = stripped.split()
        # Use known keywords from shared_utilities for additional checks.
        known_keywords = {kw.lower() for kw in HEADER_KEYWORDS}
        
        # Heuristic: if the line is short and is all uppercase or title case,
        # and contains one of the known keywords, then treat it as a header.
        if len(words) <= 10 and (stripped.isupper() or stripped.istitle()):
            lower_line = stripped.lower()
            if any(keyword in lower_line for keyword in known_keywords):
                return True

        normalized = self._normalize_header(stripped)
        if normalized in self.standard_sections or normalized in self.section_synonyms:
            return True

        # Fallback: compare vector similarity.
        line_vec = self.nlp(normalized).vector
        for canon in self.standard_sections:
            sim = cosine_similarity(line_vec, self.header_vectors[canon])
            if sim > 0.85:
                return True
        return False


    def _canonical_header(self, line: str) -> str:
        """
        Given a header line, return its canonical name using exact and fuzzy matching.
        """
        normalized = self._normalize_header(line)
        # Check direct mapping via synonyms.
        if normalized in self.section_synonyms:
            return self.section_synonyms[normalized]
        # Check if normalized header is one of the standard sections.
        if normalized in self.standard_sections:
            return normalized
        # Fallback: use cosine similarity to pick the best match.
        line_vec = self.nlp(normalized).vector
        best_match, best_sim = None, 0.0
        for canon in self.standard_sections:
            sim = cosine_similarity(line_vec, self.header_vectors[canon])
            if sim > best_sim:
                best_sim = sim
                best_match = canon
        return best_match if best_sim > 0.8 else normalized

    def detect_sections(self, text: str) -> Dict[str, str]:
        """
        Detect sections in the resume text by looking for lines that start with "##"
        and then using heuristics to decide if that line is a true header or not.
        If the line is not a header (for example, it’s part of the content), the marker is removed
        and the text is appended to the current section.
        """
        # Force every "##" marker onto its own line.
        text = re.sub(r'\s*##\s*', r'\n## ', text).strip()
        if not text:
            return {"raw": ""}
        lines = text.splitlines()
        sections = {}
        current_header = "raw"
        current_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("##"):
                # Remove the marker.
                candidate = stripped_line[2:].strip()
                # Use heuristics to decide if this candidate is a header.
                if self._is_header_line(candidate):
                    # Before switching sections, save accumulated content (if any).
                    if current_lines:
                        sections[current_header] = "\n".join(current_lines).strip()
                    # Use the canonical name for known headers.
                    header = self._canonical_header(candidate)
                    current_header = header
                    current_lines = []
                    logger.debug("Detected header: '%s' mapped to '%s'", candidate, current_header)
                else:
                    # If it does not pass as a header, treat it as content (but without the marker).
                    current_lines.append(candidate)
            else:
                current_lines.append(line)
        if current_lines:
            sections[current_header] = "\n".join(current_lines).strip()
        # If all sections are empty, return the whole text under "raw"
        non_empty = {k: v for k, v in sections.items() if v}
        return non_empty if non_empty else {"raw": text}


    # The remaining methods are kept from your original file.

    def _resolve_section_header(self, header: str) -> str:
        header_clean = self._normalize_header(header)
        if header_clean in self.standard_sections:
            return header_clean
        if header_clean in self.section_synonyms:
            return self.section_synonyms[header_clean]
        header_vec = self.nlp(header_clean).vector
        best_match, best_sim = None, 0.0
        for canon in self.standard_sections:
            sim = cosine_similarity(header_vec, self.header_vectors[canon])
            if sim > best_sim:
                best_sim = sim
                best_match = canon
        return best_match if best_sim > 0.8 else "other"

    def _validate_constants(self):
        for syn, canonical in self.section_synonyms.items():
            if canonical not in self.standard_sections:
                logger.warning(f"Synonym '{syn}' maps to '{canonical}', which is not in the standard sections: {self.standard_sections}")

    def _initialize_matcher(self):
        all_headers = list(set(self.standard_sections + [kw.lower() for kw in HEADER_KEYWORDS]))
        all_headers_sorted = sorted(all_headers, key=lambda x: len(x.split()), reverse=True)
        patterns = [self.nlp.make_doc(header) for header in all_headers_sorted]
        self.matcher.add("SECTION_HEADERS", patterns)
        logger.debug("Initialized matcher with %d unique patterns", len(patterns))

    def _initialize_patterns(self):
        def pattern_length(item):
            pat = item[0]
            if hasattr(pat, 'pattern'):
                return len(pat.pattern)
            else:
                return len(pat)
        sorted_patterns = sorted(
            self.config.get('section_patterns', []),
            key=pattern_length,
            reverse=True
        )
        self.compiled_patterns = []
        for pat, name in sorted_patterns:
            if hasattr(pat, 'pattern'):
                self.compiled_patterns.append((pat, name))
            else:
                self.compiled_patterns.append((re.compile(pat, re.IGNORECASE), name))

    def validate_sections(self, sections: dict) -> tuple:
        """
        Validates a sections dictionary structure and contents.
        
        Args:
            sections: Dictionary with section names as keys and section text as values
            
        Returns:
            Tuple of (is_valid, normalized_sections, error_message)
            - is_valid: Boolean indicating if sections are valid
            - normalized_sections: Dictionary with normalized section keys
            - error_message: String with error message if invalid, None otherwise
        """
        if not isinstance(sections, dict):
            return False, None, "Sections must be a dictionary"
        
        # Use the class's own standard sections and synonyms for validation
        expected_sections = self.standard_sections + list(self.section_synonyms.keys())
        
        # Normalize section keys to lowercase
        normalized = {}
        for key, value in sections.items():
            if not isinstance(key, str):
                return False, None, f"Section key must be a string: {key}"
            if not isinstance(value, str):
                return False, None, f"Section value must be a string: {key}"
                
            # For each key, check if it maps to a canonical header
            # This uses the class's existing header normalization logic
            norm_key = self._normalize_header(key)
            canonical_key = self._canonical_header(norm_key)
            normalized[canonical_key] = value.strip()
        
        # Check if any recognized section names exist
        found_expected = any(section in normalized for section in expected_sections)
        if not found_expected:
            # This is just a warning, not an error
            warning = f"No standard section names found. Sections provided: {list(normalized.keys())}"
            return True, normalized, warning
        
        return True, normalized, None

    def _detect_header_spans(self, doc: Doc) -> List[Span]:
        headers = []
        for sent in doc.sents:
            text = sent.text.strip()
            for pattern, name in self.compiled_patterns:
                match_obj = pattern.match(text)
                if match_obj:
                    headers.append(sent)
                    logger.debug(f"Regex header detected: {sent.text}")
                    break
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            if self._is_valid_header(span.text) and span not in headers:
                headers.append(span)
                logger.debug(f"Matcher header detected: {span.text}")
        unique_headers = []
        for span in sorted(headers, key=lambda x: (x.start, -x.end)):
            if not any(span.start >= other.start and span.end <= other.end for other in unique_headers):
                unique_headers.append(span)
        return sorted(unique_headers, key=lambda x: x.start)

    def _is_valid_header(self, text: str) -> bool:
        text_clean = self._preprocess_header(text)
        return text_clean in self.standard_sections or text_clean in self.section_synonyms

    def _preprocess_header(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r'[\*\-]+', '', text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _clean_content(self, lines: List[str]) -> str:
        cleaned = []
        for line in lines:
            line_clean = re.sub(r'\s+', ' ', line).strip()
            if len(line_clean.split()) < 2 or re.match(r'^[-=]{2,}$', line_clean):
                continue
            cleaned.append(line_clean)
        return '\n'.join(cleaned)

    def _post_process_sections(self, sections: Dict[str, str]) -> Dict[str, str]:
        merged = {}
        for name, content in sections.items():
            canonical = self.section_synonyms.get(name.lower(), name.lower())
            merged.setdefault(canonical, []).append(content.strip())
        for key in merged:
            merged[key] = "\n".join(merged[key]).strip()
        return {k: v for k, v in merged.items() if v}

    def load_config(self, config_path: str):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.standard_sections = [section.lower() for section in config.get('standard_sections', STANDARD_SECTIONS)]
            self.section_synonyms = {k.lower(): v.lower() for k, v in config.get('section_synonyms', SECTION_SYNONYMS).items()}
            self.section_vectors = {section: self.nlp(section).vector for section in self.standard_sections}
            self._initialize_matcher()
            self._initialize_patterns()
            logger.info("Loaded configuration from %s", config_path)
        except Exception as e:
            logger.error("Config load failed: %s", str(e))
