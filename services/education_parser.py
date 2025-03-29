# services/education_parser.py
import re
import spacy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EducationParser:
    # Revised regex: Use alternation for delimiters (comma or pipe) rather than a character class.
    DEGREE_REGEX = re.compile(
        r'(?i)\b(?P<degree>'
            r'Bachelor(?:\s+of\s+(?:Science|Arts|Engineering|Commerce))|'
            r'Master(?:\s+of\s+(?:Science|Arts|Business Administration|Engineering))|'
            r'MBA|'
            r'Ph\.?\s*D\.?|'
            r'Doctorate'
        r')\b'
        r'(?:\s*(?:,|\|)\s*(?P<field1>[\w\s&]+))?'
        r'(?:\s*(?:in|of)\s*(?:,|\|)?\s*(?P<field2>[\w\s&]+))?'
    )

    # Keywords for school names
    SCHOOL_KEYWORDS = ["university", "college", "institute", "academy", "school"]

    def __init__(self, nlp: spacy.language.Language):
        self.nlp = nlp

    def parse(self, education_text: str) -> list:
        """
        Parses the education section text to extract:
          - degree (e.g. "Bachelor of Science")
          - field (e.g. "Computer Science")
          - year (the first 4-digit number found, if any)
          - school (using adjacent lines and spaCy NER)
        Preprocesses the text by removing extraneous bullet characters.
        """
        # Remove common bullet characters and extra dashes.
        cleaned_text = education_text.replace("•", "").replace("-", " ").strip()
        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        records = []
        n_lines = len(lines)
        for i, line in enumerate(lines):
            match = self.DEGREE_REGEX.search(line)
            if match:
                degree = match.group("degree").strip()
                field1 = match.group("field1").strip() if match.group("field1") else ""
                field2 = match.group("field2").strip() if match.group("field2") else ""
                field = (field1 + " " + field2).strip()
                # Fallback: if field is still empty and the line contains a pipe, try to use the text after the first pipe.
                if not field and "|" in line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        field = parts[1].strip()
                # Look for a 4-digit year in the same line.
                year_match = re.search(r'(\d{4})', line)
                year = year_match.group(1) if year_match else ""
                # Use adjacent lines as candidates for the school name.
                candidates = []
                if i > 0:
                    candidates.append(lines[i - 1])
                if i < n_lines - 1:
                    candidates.append(lines[i + 1])
                school = next((cand for cand in candidates 
                               if any(kw in cand.lower() for kw in self.SCHOOL_KEYWORDS)), "Unknown")
                # Use spaCy NER on the entire block to search for organization names.
                doc = self.nlp(education_text)
                orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                if orgs:
                    # Prefer an organization name that contains a school keyword.
                    for org in orgs:
                        if any(kw in org.lower() for kw in self.SCHOOL_KEYWORDS):
                            school = org
                            break
                    else:
                        school = orgs[0]
                record = {
                    "degree": degree,
                    "field": field,
                    "year": year,
                    "school": school
                }
                records.append(record)
                logger.debug("Extracted education record: %s", record)
        return records
