# services/job_analyzer.py
"""
Job Description Analyzer

Parses job description documents to extract key information such as skills,
salary details, company culture, experience requirements, and more.
Utilizes spaCy and transformer models for NLP tasks.
"""

import logging
import re
from collections import defaultdict
from typing import Dict, List, Set, Optional
import numpy as np
import spacy
from Levenshtein import distance
from sentence_transformers import SentenceTransformer, util
from .industry_config import IndustryAnalysisConfig, IndustryDetector
from .shared_utilities import (
    TECH_SYNONYMS,
    TECH_TERMS,
    KNOWN_SKILLS,
    INDUSTRY_SYNONYMS,
    STANDARD_SECTIONS,
    INDUSTRY_CONFIGS,
    SECTION_SYNONYMS
)

logger = logging.getLogger(__name__)


class JobDescriptionAnalyzer:
    def __init__(self):
        # Load spaCy transformer model (excluding lemmatizer and textcat for performance)
        self.nlp = spacy.load('en_core_web_trf', exclude=["lemmatizer", "textcat"])
        self._add_custom_entities()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.known_skills = set(skill.lower() for skill in KNOWN_SKILLS)
        self.section_embeddings = self._precompute_section_embeddings()
        self.synonym_map = self._build_synonym_mappings()
        # Cache embeddings for known skills to avoid repeated computation
        self.cached_embeddings = {skill: self._encode_text(skill.lower()) for skill in KNOWN_SKILLS}

    def _encode_text(self, text_or_tokens) -> np.ndarray:
        emb = self.model.encode(text_or_tokens)
        if isinstance(emb, np.ndarray) and emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        return emb

    def _add_custom_entities(self) -> None:
        """Add custom entity patterns for technology and industry terms."""
        ruler = self.nlp.add_pipe("entity_ruler", after="ner")
        tech_patterns = [{"label": "TECH", "pattern": [{"LOWER": t.lower()}]} for t in TECH_TERMS]
        industry_patterns = [{"label": "INDUSTRY", "pattern": [{"LOWER": i.lower()}]} for i in INDUSTRY_SYNONYMS.keys()]
        ruler.add_patterns(tech_patterns + industry_patterns)

    def _build_synonym_mappings(self) -> Dict[str, str]:
        synonym_map = {}
        # Map technology synonyms
        for variant, canonical in TECH_SYNONYMS.items():
            synonym_map[variant.lower()] = canonical.lower()
        # Map industry synonyms
        for variant, canonical in INDUSTRY_SYNONYMS.items():
            synonym_map[variant.lower()] = canonical.lower()
        # Map section synonyms
        for variant, canonical in SECTION_SYNONYMS.items():
            synonym_map[variant.lower()] = canonical.lower()
        return synonym_map

    def _precompute_section_embeddings(self) -> Dict[str, np.ndarray]:
        sections_to_embed = [s.lower().strip(":") for s in STANDARD_SECTIONS] + list(SECTION_SYNONYMS.keys())
        return {s: self._encode_text(s) for s in sections_to_embed}

    def _initialize_analysis_template(self) -> Dict:
        """Returns a template dictionary to hold the analysis results."""
        return {
            "requirements": {
                "must_have": set(),
                "nice_to_have": set(),
                "experience": set()
            },
            "preferred_qualifications": set(),
            "responsibilities": [],
            "salary_info": {"min": None, "max": None, "currency": "USD"},
            "job_location": None,
            "experience_level": "",
            "employment_type": "unknown",
            "industry_type": "general",
            "company_culture": {"values": [], "benefits": []}
        }

    def _detect_sections_with_embeddings(self) -> Dict[str, str]:
        """
        Using sentence embeddings, detect section headers in the job description.
        Returns a dictionary mapping canonical section names to the concatenated text of that section.
        """
        doc = self.nlp(self.current_text)
        sections = {}
        current_section = None
        for sent in doc.sents:
            sent_emb = self._encode_text(sent.text)
            similarities = {sec: util.pytorch_cos_sim(sent_emb, emb) for sec, emb in self.section_embeddings.items()}
            best_match = max(similarities.items(), key=lambda x: x[1])
            if best_match[1] > 0.5:
                canonical_section = self.synonym_map.get(best_match[0].lower(), best_match[0].lower())
                current_section = canonical_section
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(sent.text)
        return {k: " ".join(v) for k, v in sections.items()}

    def _get_section_handler(self, section_name: str):
        handlers = {
            "requirements": self._process_requirements,
            "preferred_qualifications": self._process_preferred,
            "responsibilities": self._process_responsibilities,
            "company_culture": self._process_culture
        }
        return handlers.get(section_name)

    def _process_requirements(self, analysis: Dict, text: str) -> None:
        doc = self.nlp(text)
        analysis["requirements"]["experience"].update(self._extract_experience_requirements(doc))
        tskills = self._extract_tech_skills(doc)
        sskills = self._extract_soft_skills(doc)
        analysis["requirements"]["must_have"].update(tskills)
        analysis["requirements"]["nice_to_have"].update(sskills)


    def _process_preferred(self, analysis: Dict, text: str) -> None:
        doc = self.nlp(text)
        tskills = self._extract_tech_skills(doc)
        sskills = self._extract_soft_skills(doc)
        analysis["requirements"]["nice_to_have"].update(tskills)
        analysis["requirements"]["nice_to_have"].update(sskills)

    def _process_responsibilities(self, analysis: Dict, text: str) -> None:
        doc = self.nlp(text)
        responsibilities = []
        for sent in doc.sents:
            if any(t.pos_ == "VERB" for t in sent):
                cleaned = " ".join((t.lemma_ if t.pos_ == "VERB" else t.text) for t in sent if not t.is_stop)
                responsibilities.append(cleaned.capitalize())
        analysis["responsibilities"].extend(responsibilities)

    def _process_culture(self, analysis: Dict, text: str) -> None:
        doc = self.nlp(text)
        c = self._analyze_company_culture(doc)
        analysis["company_culture"]["values"].extend(c["values"])
        analysis["company_culture"]["benefits"].extend(c["benefits"])

    def _extract_skills_no_section(self, analysis: Dict, doc: spacy.tokens.Doc) -> None:
        analysis["requirements"]["experience"].update(self._extract_experience_requirements(doc))
        tskills = self._extract_tech_skills(doc)
        sskills = self._extract_soft_skills(doc)
        if tskills or sskills:
            analysis["requirements"]["must_have"].add("requirements")
        analysis["requirements"]["must_have"].update(tskills)
        analysis["requirements"]["nice_to_have"].update(sskills)

    def _extract_salary_no_section(self, analysis: Dict, doc: spacy.tokens.Doc) -> None:
        salary_info = self._extract_salary_info(doc.text)
        if salary_info["min"] is not None:
            analysis["salary_info"]["min"] = salary_info["min"]
        if salary_info["max"] is not None:
            analysis["salary_info"]["max"] = salary_info["max"]
        if salary_info["currency"] != "USD":
            analysis["salary_info"]["currency"] = salary_info["currency"]

    def _extract_culture_no_section(self, analysis: Dict, doc: spacy.tokens.Doc) -> None:
        c = self._analyze_company_culture(doc)
        analysis["company_culture"]["values"].extend(c["values"])
        analysis["company_culture"]["benefits"].extend(c["benefits"])

    def _extract_experience_requirements(self, doc: spacy.tokens.Doc) -> Set[str]:
        exp_matches = set()
        for sent in doc.sents:
            if "experience" in sent.text.lower() or "residency" in sent.text.lower() or "fellowship" in sent.text.lower():
                exp_matches.add(sent.text.strip())
        return exp_matches


    def _extract_tech_skills(self, doc: spacy.tokens.Doc) -> Set[str]:
        found_skills = set()
        
        # Process tokens
        for token in doc:
            tok_lower = token.text.lower()
            
            # Check direct matches
            if tok_lower in self.known_skills:
                found_skills.add(tok_lower)
                continue
                
            # Check synonyms
            if tok_lower in self.synonym_map:
                mapped = self.synonym_map[tok_lower]
                if mapped in self.known_skills:
                    found_skills.add(mapped)
                continue
                
            # Check fuzzy matches
            for skill in self.known_skills:
                if distance(skill, tok_lower) <= 2:
                    found_skills.add(skill)
                    break

        # Process entities
        for ent in doc.ents:
            if ent.label_ == "TECH":
                ent_lower = ent.text.lower()
                canonical = self.synonym_map.get(ent_lower, ent_lower)
                found_skills.add(canonical)

        return found_skills

    def _extract_soft_skills(self, doc: spacy.tokens.Doc) -> Set[str]:
        soft_list = ["communication", "leadership", "teamwork", "problem-solving", "adaptability"]
        found = set()
        for chunk in doc.noun_chunks:
            ctext = chunk.text.lower()
            for s in soft_list:
                if s in ctext:
                    found.add(s)
        return found

    def _extract_salary_info(self, text: str) -> Dict:
        salary_info = {"min": None, "max": None, "currency": "USD"}
        patterns = [
            r'(\$|£|€)?(\d{1,3}(?:,\d{3})*k?)(?:\s*[-–]\s*|\s+to\s+)(\$|£|€)?(\d{1,3}(?:,\d{3})*k?)',
            r'(\d+\s*[kK])\s*[-–]\s*(\d+\s*[kK])'
        ]
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                left_symbol = groups[0] or ""
                right_symbol = groups[2] if len(groups) > 2 else ""
                symbol = left_symbol if left_symbol else right_symbol
                min_str = groups[1]
                max_str = groups[3] if len(groups) > 3 else None
                if symbol:
                    salary_info["currency"] = self._determine_currency(symbol)
                min_val = self._parse_salary_number(min_str)
                max_val = self._parse_salary_number(max_str) if max_str else None
                salary_info["min"] = min_val
                salary_info["max"] = max_val
                break
        return salary_info

    def _parse_salary_number(self, raw: str) -> Optional[float]:
        if not raw:
            return None
        s = raw.lower().replace(",", "")
        if "k" in s:
            try:
                base = float(re.sub(r'[^\d]', '', s))
                return base * 1000
            except:
                return None
        try:
            return float(re.sub(r'[^\d.]', '', s))
        except:
            return None

    def _determine_currency(self, text: str) -> str:
        if "£" in text:
            return "GBP"
        if "€" in text:
            return "EUR"
        if "$" in text:
            return "USD"
        t = text.lower()
        if "gbp" in t or "pound" in t:
            return "GBP"
        if "eur" in t or "euro" in t:
            return "EUR"
        if "usd" in t or "dollar" in t:
            return "USD"
        return "USD"

    def _extract_location(self, doc: spacy.tokens.Doc) -> Optional[str]:
        locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]
        if locations:
            return " and ".join(locations)
        return None

    def _detect_industry(self, doc: spacy.tokens.Doc) -> str:
        industry_scores = defaultdict(float)
        text_lower = doc.text.lower()

        # Process industry configurations dynamically
        for industry_id, config in INDUSTRY_CONFIGS.items():
            # Score based on keywords
            for kw in config.get("keywords", []):
                if kw.lower() in text_lower:
                    industry_scores[industry_id] += 0.5
                    
            # Score based on experience patterns
            for pattern in config.get("experience_patterns", []):
                matches = re.findall(pattern, text_lower, flags=re.IGNORECASE)
                industry_scores[industry_id] += len(matches) * 0.3

        # Score based on entities
        for ent in doc.ents:
            if ent.label_ == "INDUSTRY":
                norm_term = self.synonym_map.get(ent.text.lower(), ent.text.lower())
                for industry_id, config in INDUSTRY_CONFIGS.items():
                    if norm_term in config.get("keywords", []):
                        industry_scores[industry_id] += 1.0

        if industry_scores:
            best_match = max(industry_scores.items(), key=lambda x: x[1])
            return INDUSTRY_CONFIGS.get(best_match[0], {}).get("name", best_match[0])
        return "general"


    def _determine_experience_level(self, doc: spacy.tokens.Doc) -> str:
        text = doc.text.lower()
        level_keywords = {
            "junior": [r"junior", r"entry[- ]?level", r"entry"],
            "mid": [r"mid[- ]?level", "intermediate"],
            "senior": ["senior", "lead", "principal", "staff"]
        }
        for level, patterns in level_keywords.items():
            for pat in patterns:
                if re.search(rf"\b{pat}\b", text):
                    return level
        match = re.search(r"(\d+)\+?\s*years?\s*experience", text)
        if match:
            years = int(match.group(1))
            if years >= 5:
                return "senior"
            elif years <= 2:
                return "junior"
            else:
                return "mid"
        return "not_specified"

    def _determine_employment_type(self, doc: spacy.tokens.Doc) -> str:
        return "unknown"

    def _analyze_company_culture(self, doc: spacy.tokens.Doc) -> Dict[str, List[str]]:
        result = {"values": [], "benefits": []}
        value_keywords = ["innovation", "diversity", "collaboration", "integrity"]
        benefit_keywords = ["401k", "401k matching", "health insurance", "paid time off", "unlimited pto", "remote work"]
        for sent in doc.sents:
            stext = sent.text.lower()
            if any(bk in stext for bk in benefit_keywords):
                result["benefits"].append(sent.text)
            elif any(vk in stext for vk in value_keywords):
                result["values"].append(sent.text)
        return result

    def _normalize_analysis(self, analysis: Dict) -> Dict:
        for key in ["must_have", "nice_to_have", "experience"]:
            normalized = set()
            for item in analysis["requirements"][key]:
                low_item = item.lower()
                canonical = self.synonym_map.get(low_item, low_item)
                normalized.add(canonical)
            analysis["requirements"][key] = list(normalized)
        itype = analysis["industry_type"].lower()
        analysis["industry_type"] = self.synonym_map.get(itype, analysis["industry_type"])
        return analysis

    def analyze_job_description(self, job_description: str) -> Dict:
        """
        Main entry: parse doc, detect sections, do extractions, return structured analysis.
        
        Note: This method uses IndustryDetector indirectly through IndustryAnalysisConfig
        to maintain consistency with industry detection across the codebase.
        """
        self.current_text = job_description
        doc = self.nlp(job_description)
        analysis = self._initialize_analysis_template()

        # Create an industry detector through IndustryAnalysisConfig
        industry_config = IndustryAnalysisConfig()
        industry_detector = industry_config.industry_detector
        
        # First, detect the industry to ensure proper context for all other analyses
        industry, confidence = industry_detector.detect_industry(job_description)
        analysis["industry_type"] = industry

        # Continue with section detection and processing as before
        sections = self._detect_sections_with_embeddings()
        for section_name, section_text in sections.items():
            handler = self._get_section_handler(section_name)
            if handler:
                handler(analysis, section_text)

        # When extracting skills, pass the detected industry
        self._extract_skills_with_industry(analysis, doc, industry, industry_detector)
        self._extract_salary_no_section(analysis, doc)
        self._extract_culture_no_section(analysis, doc)
        analysis["job_location"] = self._extract_location(doc)
        analysis["experience_level"] = self._determine_experience_level(doc)
        analysis["employment_type"] = self._determine_employment_type(doc)

        # Normalize the final analysis
        analysis = self._normalize_analysis(analysis)
        return analysis
    
    def _extract_skills_with_industry(self, analysis: Dict, doc: spacy.tokens.Doc, industry: str, industry_detector) -> None:
        """
        Extract skills based on the detected industry.
        This ensures we only extract skills relevant to the job's domain.
        
        Note: This method receives the industry_detector as a parameter to avoid
        recreating it, as it was already instantiated in the calling method.
        """
        analysis["requirements"]["experience"].update(self._extract_experience_requirements(doc))
        
        # Get required skills for this industry
        industry_skills = industry_detector.get_required_skills_for_industry(industry, doc.text)
        
        # Extract from text
        found_skills = set()
        for token in doc:
            if token.text.lower() in industry_skills:
                found_skills.add(token.text.lower())
        
        # Add some common skills for this industry if we found few
        if len(found_skills) < 3:
            common_skills = list(industry_skills)[:5]  # Take up to 5 common skills
            found_skills.update(common_skills)
        
        # Extract soft skills (these are often cross-industry)
        soft_skills = self._extract_soft_skills(doc)
        
        analysis["requirements"]["must_have"].update(found_skills)
        analysis["requirements"]["nice_to_have"].update(soft_skills)

# Instantiate the analyzer for easy import and use
analyzer = JobDescriptionAnalyzer()
