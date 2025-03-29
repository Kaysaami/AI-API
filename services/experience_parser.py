import re
import spacy
import logging
from datetime import datetime
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ExperienceParser:
    def __init__(self, nlp: spacy.language.Language, config: dict):
        self.nlp = nlp
        self.config = config
        # Lookup table for job titles from configuration (if needed)
        self.job_title_lookup = {title.lower() for title in config.get("job_title_lookup", [])}
        # Company suffix whitelist
        self.company_suffixes = ["inc", "corp", "llc", "ltd", "co"]

    def _extract_date_ranges(self, text: str) -> list:
        date_ranges = []
        # Patterns for dates (e.g., "Feb 2021 - Present" or "2018 - 2020")
        patterns = [
            r'(\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|'
            r'Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\s*(?:-|–|to)\s*([\w\s]+?\d{4}|Present)',
            r'(\d{4})\s*(?:-|–|to)\s*(\d{4}|Present)'
        ]
        for pattern in patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                part1 = m.group(1)
                part2 = m.group(2)
                try:
                    # Update 5: Use fuzzy=True with try/except
                    start_date = date_parser.parse(part1, fuzzy=True)
                except Exception as e:
                    logger.debug("Date parsing failed for '%s': %s", part1, e)
                    start_date = None
                try:
                    if re.search(r'present', part2, re.IGNORECASE):
                        end_date = datetime.now()
                    else:
                        end_date = date_parser.parse(part2, fuzzy=True)
                except Exception as e:
                    logger.debug("Date parsing failed for '%s': %s", part2, e)
                    end_date = None
                # Update 6: Validate dates – if end_date < start_date, swap them.
                if start_date and end_date and end_date < start_date:
                    start_date, end_date = end_date, start_date
                if start_date and end_date:
                    date_ranges.append({"start": start_date, "end": end_date, "text": m.group(0)})
        return date_ranges

    def _calculate_total_years(self, date_ranges: list) -> float:
        if not date_ranges:
            return 0.0
        # Merge overlapping ranges and calculate duration in years.
        sorted_ranges = sorted(date_ranges, key=lambda x: x["start"])
        merged = []
        current = sorted_ranges[0]
        for r in sorted_ranges[1:]:
            if r["start"] <= current["end"]:
                current["end"] = max(current["end"], r["end"])
            else:
                merged.append(current)
                current = r
        merged.append(current)
        total_days = sum((r["end"] - r["start"]).days for r in merged)
        return round(total_days / 365.25, 2)

    def _extract_companies(self, doc: spacy.tokens.Doc) -> list:
        companies = set()
        non_company_keywords = {"iso", "erp", "mrp"}
        for ent in doc.ents:
            if ent.label_ == "ORG":
                comp = ent.text.strip()
                if re.search(r'\d', comp):
                    continue
                if any(kw in comp.lower() for kw in non_company_keywords):
                    continue
                # Update 11: Validate by checking for common suffixes or title case
                if not any(comp.lower().endswith(suffix) for suffix in self.company_suffixes) and comp != comp.title():
                    continue
                companies.add(comp)
        return sorted(companies)

    def _extract_positions(self, text: str, doc: spacy.tokens.Doc) -> list:
        positions = set()
        lines = text.splitlines()
        for line in lines:
            if re.search(r'\d{4}', line):
                for word in line.split():
                    if word.lower() in self.job_title_lookup:
                        positions.add(word.title())
        return sorted(positions)

    def parse(self, experience_text: str) -> dict:
        doc = self.nlp(experience_text)
        date_ranges = self._extract_date_ranges(experience_text)
        total_years = self._calculate_total_years(date_ranges)
        companies = self._extract_companies(doc)
        positions = self._extract_positions(experience_text, doc)
        return {
            "total_years": total_years,
            "companies": companies,
            "positions": positions
        }
