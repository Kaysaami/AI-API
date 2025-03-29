# services/resume_analyzer.py
import re
import spacy
import logging
from datetime import datetime
from dateutil.parser import parse as parse_date
from typing import Optional, Dict, List

# Import specialized modules for section, education, experience, skills, and job analysis.
from .section_extractor import SectionExtractor
from .education_parser import EducationParser
from .experience_parser import ExperienceParser
from .skills_extractor import SkillsExtractor
from .job_analyzer import JobDescriptionAnalyzer
from .industry_config import IndustryAnalysisConfig, IndustryDetector
from .shared_utilities import load_config, INDUSTRY_CONFIGS, HEADER_KEYWORDS 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ResumeAnalyzer:
    def __init__(self):
        # Load the spaCy transformer model with a configuration that limits batch size.
        self.nlp = spacy.load("en_core_web_trf", config={"nlp": {"batch_size": 1}})
        self.nlp.max_length = 20000  # Adjust if necessary

        # Load configuration settings.
        self.config = load_config()

        # Initialize specialized modules.
        self.section_extractor = SectionExtractor(self.nlp, self.config)
        self.education_parser = EducationParser(self.nlp)  # (Use your updated EducationParser here)
        self.experience_parser = ExperienceParser(self.nlp, self.config)
        self.job_analyzer = JobDescriptionAnalyzer()
        # For skills extraction, load a transformer-based model (excluding lemmatizer and NER for speed).
        self.skills_extractor = SkillsExtractor(spacy.load("en_core_web_trf", exclude=["lemmatizer", "ner"]), self.config)

    def robust_insert_section_markers(text: str) -> str:
        """
        Scans the text and inserts explicit section markers ("## ... ##") for lines that are likely headers.
        If the text is flat (has very few newlines), it first inserts newlines before any occurrence of a known
        header keyword (from HEADER_KEYWORDS).
        """
        # If the text is flat (e.g., extracted from a PDF), insert newlines before known section keywords.
        if text.count('\n') < 5:
            for header in HEADER_KEYWORDS:
                # This regex looks for the header keyword not already preceded by a newline or marker.
                pattern = re.compile(r'(?<!\n)(?<!## )\b(' + re.escape(header) + r')\b', re.IGNORECASE)
                # Insert a newline and markers before the header keyword.
                text = pattern.sub(r'\n## \1 ##', text)

        # Now split the text into lines and process each line.
        lines = text.splitlines()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            # Preserve empty lines.
            if not stripped:
                new_lines.append(line)
                continue
            # If already marked as a header, preserve.
            if stripped.startswith("##") and stripped.endswith("##"):
                new_lines.append(line)
                continue
            # If the line is short (<=10 words), check if it contains any known header keyword.
            if len(stripped.split()) <= 10:
                lower_line = stripped.lower()
                for header in HEADER_KEYWORDS:
                    if header in lower_line:
                        new_lines.append(f"## {stripped} ##")
                        break
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        return "\n".join(new_lines)


    def _preprocess_headers(self, text: str) -> str:
        """
        Standardizes keywords (e.g. "present") and ensures that every "##" marker is on its own line.
        If no markers are found (common with PDF extraction), insert them using robust_insert_section_markers.
        """
        text = re.sub(r'\b(present|current)\b', "Present", text, flags=re.IGNORECASE)
        # Force a newline before every "##" so headers are isolated.
        text = re.sub(r'\s*##\s*', r'\n## ', text)
        # If no markers exist, insert them robustly.
        if "##" not in text:
            text = self.robust_insert_section_markers(text)
        return text
    
    def analyze_resume(self, text: str, job_description: str = None, sections: dict = None) -> dict:
        """
        Analyzes the resume by:
        1. Preprocessing header text (ensuring keywords are standardized and "##" markers are on new lines).
        2. Handling pre-extracted sections or splitting the resume into sections if needed.
        3. Identifying the Education and Experience sections by keyword.
        4. Computing basic text statistics.
        5. Extracting skills, education, experience, and contact info.
        6. Optionally performing job matching analysis.
        
        Args:
            text: The full text of the resume
            job_description: Optional job description to match against
            sections: Optional pre-extracted sections dictionary with section names as keys
            
        Returns:
            Dictionary containing comprehensive resume analysis
        """
        # Preprocess the text to standardize and isolate header markers.
        text = self._preprocess_headers(text)

        # Validate and normalize sections if provided
        if sections is not None:
            # Use section_extractor's validate_sections method
            is_valid, normalized_sections, warning_msg = self.section_extractor.validate_sections(sections)
            
            if not is_valid:
                logger.warning(f"Invalid sections: {warning_msg}. Falling back to extraction.")
                sections = None
            else:
                sections = normalized_sections
                if warning_msg:
                    logger.warning(warning_msg)
                    
            # If somehow we ended up with empty sections dict, set to None to trigger extraction
            if sections is not None and not sections:
                logger.warning("Provided sections were empty. Falling back to extraction.")
                sections = None

        # Use provided sections if available; otherwise, extract from the raw text.
        if sections is None:
            if "##" in text:
                sections = self._split_sections_by_markers(text)
            else:
                sections = self.section_extractor.detect_sections(text)
            if not sections:
                # If no sections were detected, at least keep the raw text
                sections = {"raw": text}

        # Add the raw text to sections if not already present
        if "raw" not in sections:
            sections["raw"] = text

        # Identify the Education section by scanning for headers that contain "educat"
        education_section_txt = ""
        for key, value in sections.items():
            if "educat" in key.lower():
                education_section_txt = value.strip()
                break
        if not education_section_txt:
            education_section_txt = text

        # Identify the Experience section by scanning for headers containing "experien" or "work"
        experience_section_txt = ""
        for key, value in sections.items():
            if any(term in key.lower() for term in ["experien", "work", "professional"]):
                experience_section_txt = value.strip()
                break
        if not experience_section_txt:
            experience_section_txt = text

        # Look for skills section
        skills_section_txt = ""
        for key, value in sections.items():
            if "skill" in key.lower():
                skills_section_txt = value.strip()
                break

        # Compute basic statistics
        basic_stats = self._get_basic_stats(sections)

        # Initialize the analysis result dictionary
        analysis = {
            "sections": sections,
            "raw_text": text,
            "basic_stats": basic_stats,
        }

        # Determine industry configuration (and run job analysis if provided)
        job_analysis = None
        if job_description:
            job_analysis = self.job_analyzer.analyze_job_description(job_description)
            industry = job_analysis.get("industry_type", "general").lower()
            industry_cfg = INDUSTRY_CONFIGS.get(industry, INDUSTRY_CONFIGS["general"])
        else:
            industry = "general"
            industry_cfg = INDUSTRY_CONFIGS["general"]

        # --- Skills Extraction ---
        if skills_section_txt:
            skills = self.skills_extractor.extract_skills(skills_section_txt, industry=industry)
        else:
            skills = self.skills_extractor.extract_skills(text, industry=industry)
        skills = self._filter_skills_extraction(skills)
        analysis["skills"] = skills

        # --- Experience Extraction ---
        experience_data = self.experience_parser.parse(experience_section_txt)
        if not experience_data.get("positions"):
            experience_data["positions"] = self._extract_positions_from_experience(experience_section_txt)
        if not experience_data.get("total_years") or experience_data.get("total_years") == 0.0:
            # Use our fallback method to sum durations from date ranges
            experience_data["total_years"] = self._calculate_total_experience_years(experience_section_txt)
        analysis["experience"] = experience_data

        # --- Education Extraction ---
        education_data = self._extract_education_fixed(education_section_txt)
        analysis["education"] = education_data
        analysis["education_analysis"] = education_data

        # --- Contact Information Extraction ---
        contact_info = self._extract_contact_info(text)
        analysis["contact_info"] = contact_info

        # --- Job Matching Analysis (if applicable) ---
        if job_description and job_analysis:
            resume_skills_set = self._flatten_skills_dict(skills)
            job_required = {s.lower() for s in job_analysis["requirements"].get("must_have", [])}
            job_preferred = {s.lower() for s in job_analysis["requirements"].get("nice_to_have", [])}
            
            # Pass the industry to _analyze_skills_match
            skills_match = self._analyze_skills_match(resume_skills_set, job_required, job_preferred, industry)
            experience_match = self._analyze_experience_match(experience_data, job_analysis, industry_cfg)
            requirements_match = self._analyze_requirements_match(text, job_analysis)
            analysis["job_match"] = {
                "description": job_description,
                "industry": industry,
                "requirements_match": requirements_match,
                "skills_match": skills_match,
                "experience_match": experience_match,
                "overall_score": self._calculate_overall_match(skills_match, experience_match, job_analysis, industry_cfg)
            }
        return analysis

    def _split_sections_by_markers(self, text: str) -> Dict[str, str]:
        """
        Splits the text into sections using explicit "##" markers.
        Assumes that each header marker is at the start of a line.
        """
        pattern = re.compile(r'\n##\s*(.*?)\s*##\s*')
        parts = pattern.split(text)
        sections = {}
        if len(parts) > 1:
            if parts[0].strip():
                sections["raw"] = parts[0].strip()
            # Process pairs: header followed by content.
            for i in range(1, len(parts) - 1, 2):
                header = parts[i].strip().lower()
                content = parts[i + 1].strip()
                sections[header] = content
        else:
            sections["raw"] = text
        return sections

    def _auto_detect_sections(self, text: str) -> Dict[str, str]:
        """
        As a fallback, automatically detect section headers by scanning for short, all-uppercase lines.
        """
        sections = {}
        current_header = "raw"
        current_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and stripped == stripped.upper() and len(stripped.split()) < 10:
                if current_lines:
                    sections[current_header] = "\n".join(current_lines).strip()
                current_header = stripped.lower()
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sections[current_header] = "\n".join(current_lines).strip()
        return sections

    def _get_basic_stats(self, text) -> dict:
        """
        Computes basic statistics (word count, average sentence length, section count)
        from the provided resume text. If a dictionary of sections is provided, all values
        are concatenated into a single text for analysis.
        """
        # If text is a dict (i.e., a sections dictionary), join its values.
        if isinstance(text, dict):
            combined_text = "\n".join(text.values())
        else:
            combined_text = text

        words = combined_text.split()
        word_count = len(words)
        # Split on period, but only consider non-empty parts.
        sentences = [s for s in combined_text.split('.') if s.strip()]
        avg_sentence_length = word_count / len(sentences) if sentences else 0
        # Count sections by splitting on newline; this is just one measure.
        section_count = len(combined_text.split('\n'))
        return {
            'word_count': word_count,
            'avg_sentence_length': round(avg_sentence_length, 2),
            'section_count': section_count
        }

    def _extract_education_fixed(self, text: str) -> List[dict]:
        records = self.education_parser.parse(text)
        fixed = []
        for rec in records:
            school = rec.get("school", "").strip()
            if len(school) < 4:
                candidates = [line for line in text.splitlines()
                              if any(kw in line.lower() for kw in ["university", "college", "institute", "academy", "school"])]
                if candidates:
                    rec["school"] = candidates[0].strip()
            fixed.append(rec)
        return fixed

    def _extract_positions_from_experience(self, text: str) -> list:
        """
        Enhanced extraction of positions from the experience section.
        Tries two regex patterns:
          Pattern A: "Company, Position, Location, Date..."
          Pattern B: "Position, Company, Location, Date..."
        Then selects the candidate that contains common job-title keywords.
        """
        positions = []
        lines = text.splitlines()
        # Define two regex patterns.
        pattern_a = re.compile(r'^(?P<company>.+?),\s*(?P<position>.+?),\s*(?P<location>.+?)\s+(?P<date_range>.+)$')
        pattern_b = re.compile(r'^(?P<position>.+?),\s*(?P<company>.+?),\s*(?P<location>.+?)\s+(?P<date_range>.+)$')
        # List of common job-title keywords.
        job_keywords = {"associate", "intern", "consultant", "manager", "engineer", "lead", "analyst", "director", "officer", "specialist"}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            candidate_a = candidate_b = ""
            m_a = pattern_a.match(line)
            m_b = pattern_b.match(line)
            if m_a:
                candidate_a = m_a.group("position").strip()
            if m_b:
                candidate_b = m_b.group("position").strip()
            # Decide which candidate seems more plausible:
            # If candidate_b contains a keyword and candidate_a doesn't, use candidate_b.
            if candidate_b and any(kw in candidate_b.lower() for kw in job_keywords):
                positions.append(candidate_b)
            elif candidate_a and any(kw in candidate_a.lower() for kw in job_keywords):
                positions.append(candidate_a)
            elif candidate_a:
                # As a fallback, if only candidate_a is available, use it.
                positions.append(candidate_a)
        return positions

    def _calculate_total_experience_years(self, text: str) -> float:
        """
        Fallback method: scans the experience text for date ranges using a regex that accepts full month names
        (or common abbreviations) for both start and end dates. It then parses these dates and sums the durations.
        """
        # Updated regex:
        pattern = re.compile(
            r'\b(?P<start>(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b\s*(?:-|–|to)\s*\b(?P<end>(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|Present)\b',
            re.IGNORECASE
        )
        total_days = 0
        for m in pattern.finditer(text):
            start_str = m.group("start")
            end_str = m.group("end")
            try:
                start_date = parse_date(start_str, fuzzy=True)
            except Exception as e:
                logger.debug("Error parsing start date '%s': %s", start_str, e)
                start_date = None
            try:
                if "present" in end_str.lower():
                    end_date = datetime.now()
                else:
                    end_date = parse_date(end_str, fuzzy=True)
            except Exception as e:
                logger.debug("Error parsing end date '%s': %s", end_str, e)
                end_date = None
            if start_date and end_date:
                delta = (end_date - start_date).days
                total_days += delta
        total_years = total_days / 365.25 if total_days else 0.0
        return round(total_years, 2)

    def _extract_contact_info(self, text: str, doc: Optional[spacy.tokens.Doc] = None) -> dict:
        if doc is None:
            doc = self.nlp(text)
        contact_info = {'email': None, 'phone': None, 'linkedin': None, 'location': None}
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
        if email_match:
            contact_info['email'] = email_match.group(0)
        phone_match = re.search(r'(?:\+\d{1,2}\s)?\(?\d{2,3}\)?[\s.-]?\d{3}[\s.-]?\d{4}', text)
        if phone_match:
            contact_info['phone'] = phone_match.group(0)
        linkedin_match = re.search(r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+', text, re.IGNORECASE)
        if linkedin_match:
            contact_info['linkedin'] = linkedin_match.group(0)
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                contact_info['location'] = ent.text
                break
        return contact_info

    def _flatten_skills_dict(self, skills: dict) -> set:
        flattened = set()
        for cat_name, skill_list in skills.items():
            if cat_name == "errors":
                continue
            for item in skill_list:
                if isinstance(item, dict) and 'skill' in item:
                    flattened.add(item['skill'].lower())
                elif isinstance(item, str):
                    flattened.add(item.lower())
        return flattened

    def _analyze_skills_match(self, resume_skills: set, job_required: set, job_preferred: set, industry: str) -> dict:
        """
        Match resume skills against job requirements, considering industry context.
        
        Note: This method uses IndustryDetector indirectly through IndustryAnalysisConfig
        to maintain consistency with industry skill retrieval across the codebase.
        
        Args:
            resume_skills: Set of skills found in the resume
            job_required: Set of required skills from the job description
            job_preferred: Set of preferred skills from the job description
            industry: The detected industry name
        
        Returns:
            A dictionary with skill matching analysis
        """
        # Get an instance of IndustryAnalysisConfig and use its industry_detector
        industry_config = IndustryAnalysisConfig()
        industry_detector = industry_config.industry_detector
        
        # Get industry-specific skills to use as context
        industry_skills = industry_detector.get_required_skills_for_industry(industry.lower())
        
        # Normalize skills to lowercase for comparison
        resume_set = {s.lower() for s in resume_skills}
        required_set = {s.lower() for s in job_required}
        preferred_set = {s.lower() for s in job_preferred}
        
        # Filter required and preferred skills to ensure they're relevant to this industry
        required_set = required_set & industry_skills
        preferred_set = preferred_set & industry_skills
        
        # If filtering left us with too few skills, ensure there are some industry basics
        if len(required_set) < 3:
            common_skills = set(list(industry_skills)[:5])
            required_set.update(common_skills)
        
        # Continue with matching logic as before
        required_matched = resume_set & required_set
        preferred_matched = resume_set & preferred_set
        
        req_pct = (len(required_matched) / len(required_set) * 100) if required_set else 100
        pref_pct = (len(preferred_matched) / len(preferred_set) * 100) if preferred_set else 0
        
        return {
            'required': {
                'matched': sorted(required_matched),
                'missing': sorted(required_set - resume_set),
                'match_percentage': round(req_pct, 1)
            },
            'preferred': {
                'matched': sorted(preferred_matched),
                'missing': sorted(preferred_set - resume_set),
                'match_percentage': round(pref_pct, 1)
            }
        }

    def _analyze_requirements_match(self, resume_text: str, job_analysis: dict) -> dict:
        import string
        translator = str.maketrans('', '', string.punctuation)
        normalized_resume = resume_text.lower().translate(translator)
        matches = {'must_have': [], 'nice_to_have': [], 'missing_requirements': []}
        req_must = [req.lower().translate(translator).strip() for req in job_analysis.get('requirements', {}).get('must_have', [])]
        req_nice = [req.lower().translate(translator).strip() for req in job_analysis.get('requirements', {}).get('nice_to_have', [])]
        for req in req_must:
            if req in normalized_resume:
                matches['must_have'].append(req)
            else:
                matches['missing_requirements'].append(req)
        for req in req_nice:
            if req in normalized_resume:
                matches['nice_to_have'].append(req)
        return matches

    def _analyze_experience_match(self, resume_experience: dict, job_analysis: dict, industry_cfg: dict) -> dict:
        job_level = job_analysis.get('experience_level', 'entry').lower()
        resume_years = resume_experience.get('total_years', 0.0)
        required_years_map = industry_cfg.get('experience_levels', {'entry': 0, 'junior': 1, 'mid': 3, 'senior': 5})
        req_years = required_years_map.get(job_level, 0)
        meets_req = resume_years >= req_years
        exp_weight = industry_cfg.get('experience_weight', 0.3)
        score = round(min(resume_years / req_years, 1.0) * exp_weight, 2) if req_years > 0 else exp_weight
        return {
            'meets_requirements': meets_req,
            'years_required': req_years,
            'years_actual': resume_years,
            'experience_match_score': score
        }

    def _calculate_overall_match(self, skills_match: dict, experience_match: dict, job_analysis: dict, industry_cfg: dict = None) -> float:
        weights = industry_cfg.get('weights', {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.3
        }) if industry_cfg else {
            'skills': 0.4,
            'experience': 0.3,
            'education': 0.3
        }
        skills_req_score = skills_match.get('required', {}).get('match_percentage', 0.0) / 100
        skills_pref_score = skills_match.get('preferred', {}).get('match_percentage', 0.0) / 100
        skills_score = (skills_req_score * 0.7) + (skills_pref_score * 0.3)
        exp_score = experience_match.get('experience_match_score', 0.0)
        edu_score = 1.0 if job_analysis.get('education_requirements') else 0.5
        final_score = (skills_score * weights['skills'] +
                       exp_score * weights['experience'] +
                       edu_score * weights['education'])
        return round(max(0.0, min(1.0, final_score)), 2)

    def _filter_skills_extraction(self, skills: dict) -> dict:
        filtered = skills.copy()
        errors = filtered.get("errors", [])
        general = filtered.get("general", [])
        new_errors = []
        non_tech_terms = {"conflict resolution", "leadership", "communication"}
        for entry in errors:
            skill_text = entry.get("skill", "").lower()
            if skill_text in non_tech_terms:
                general.append(entry)
            else:
                new_errors.append(entry)
        filtered["errors"] = new_errors
        filtered["general"] = general
        return filtered

# Instantiate the analyzer so that other modules can import and use it.
analyzer = ResumeAnalyzer()
