import re
import spacy
import logging
from typing import Dict, List, Set, Optional
from fuzzywuzzy import fuzz
from spacy.matcher import PhraseMatcher
from spacy.util import filter_spans
from .shared_utilities import load_config, SKILL_CATEGORIES, TECH_SYNONYMS, KNOWN_SKILLS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

class SkillsExtractor:
    def __init__(self, nlp: spacy.language.Language = None, config: dict = None):
        """
        Initializes the SkillsExtractor.
        
        :param nlp: A spaCy language model. If not provided, a default model is loaded.
        :param config: A configuration dictionary. Defaults to an empty dict if not provided.
        """
        # Use provided NLP model or load a default transformer-based model.
        self.nlp = nlp or spacy.load("en_core_web_trf", exclude=["lemmatizer", "ner"])
        self.config = config or {}  # You can extend this later if needed.
        
        # Build the PhraseMatcher for skill extraction.
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        # Combine known skills and synonyms, then normalize them.
        all_syns = list(TECH_SYNONYMS.keys()) + list(TECH_SYNONYMS.values())
        self.normalized_skills = set(self.normalize_skill(s) for s in (KNOWN_SKILLS + all_syns))
        patterns = [self.nlp.make_doc(skill) for skill in self.normalized_skills if skill]
        self.matcher.add("SKILLS", patterns)
        
        # Build an internal skill database for quick lookup.
        self._build_skill_db()

    def _build_skill_db(self):
        self.skill_terms = set()
        all_terms = list(TECH_SYNONYMS.keys()) + KNOWN_SKILLS
        all_terms.sort(key=lambda x: len(x.split()), reverse=True)
        for term in all_terms:
            self.skill_terms.add(self.normalize_skill(term))

    def normalize_skill(self, skill: str) -> str:
        """
        Normalizes a skill string to its canonical form.
        Steps:
          1. Strip whitespace.
          2. Remove trailing version numbers (e.g. "Python 3.8" -> "Python").
          3. Replace with canonical form if a synonym exists.
          4. Remove unwanted punctuation (while preserving symbols like '+' and '#').
          5. Return in title case.
        """
        raw = skill.strip()
        if not raw:
            return ""
        # Remove trailing version numbers.
        no_version = re.sub(r"\s+\d+(\.\d+)?", "", raw).strip()
        lower_no_version = no_version.lower()
        if lower_no_version in TECH_SYNONYMS:
            # For example, "node js" -> "Node.js"
            return TECH_SYNONYMS[lower_no_version].title()
        skill_clean = re.sub(r"[^\w\s#+]", "", no_version).strip()
        return skill_clean.title()

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text, replacing synonyms with their canonical forms.
        """
        sorted_synonyms = sorted(TECH_SYNONYMS.keys(), key=len, reverse=True)
        processed = text
        for syn in sorted_synonyms:
            canon = TECH_SYNONYMS[syn]
            pattern = re.compile(r"\b" + re.escape(syn) + r"\b", re.IGNORECASE)
            def replace_with_canonical(match):
                if " " in canon and not any(c.isupper() for c in canon[1:]):
                    return canon.title()
                return canon
            new_processed = pattern.sub(replace_with_canonical, processed)
            if new_processed != processed:
                logger.debug(f"Replaced '{syn}' -> '{canon}'")
                processed = new_processed
        return processed

    def extract_skill_context(self, text: str) -> List[dict]:
        doc = self.nlp(text)
        matches = self.matcher(doc)
        spans = filter_spans([doc[start:end] for _, start, end in matches])
        results = []
        for span in spans:
            # Get the sentence in which the skill appears.
            context = span.sent.text.strip()
            results.append({
                "skill": span.text,
                "context": context
            })
        return results

    def _find_duration(self, doc, start: int, end: int) -> Optional[str]:
        window_size = 5
        search_start = max(0, start - window_size)
        search_end = min(len(doc), end + window_size)
        durations = []
        for i in range(search_start, search_end - 1):
            tok1 = doc[i]
            tok2 = doc[i + 1]
            if tok1.like_num and tok2.text.lower() in {"year", "years", "month", "months"}:
                durations.append((tok1.text, i))
        if not durations:
            return None
        # Pick the closest to the skill
        skill_center = (start + end) / 2.0
        best_dur = None
        best_dist = None
        for num_str, idx in durations:
            try:
                num = float(num_str)
                dist = abs(idx - skill_center)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_dur = num_str
            except ValueError:
                continue
        return best_dur

    def match_skill(self, skill: str, known_skills: List[str], threshold: int = 80) -> Optional[str]:
        normalized_skill = self.normalize_skill(skill)
        exact_matches = [ks for ks in known_skills if self.normalize_skill(ks) == normalized_skill]
        if exact_matches:
            return exact_matches[0]
        best_match = None
        highest_score = 0
        for ks in known_skills:
            score = fuzz.ratio(normalized_skill, self.normalize_skill(ks))
            if score > highest_score:
                best_match = ks
                highest_score = score
        if highest_score >= threshold:
            return best_match
        logger.debug(f"No strong match found for '{skill}' (score: {highest_score}).")
        return None

    def _get_industry_skills(self, industry: str) -> Set[str]:
        """
        Returns the set of normalized skill strings for the given industry.
        If the industry is not found, defaults to 'technology'.
        """
        industry = industry.lower()
        categories = SKILL_CATEGORIES.get(industry, SKILL_CATEGORIES.get("technology", {}))
        skills = set()
        for cat_skills in categories.values():
            for skill in cat_skills:
                normalized = self.normalize_skill(skill).lower()
                if normalized:
                    skills.add(normalized)
        for syn in TECH_SYNONYMS.keys():
            normalized_syn = self.normalize_skill(syn).lower()
            if normalized_syn:
                skills.add(normalized_syn)
        logger.debug(f"Normalized industry skills for '{industry}': {skills}")
        return skills

    def get_skill_category(self, skill: str, industry: str) -> Optional[str]:
        for category, cat_skills in SKILL_CATEGORIES.get(industry.lower(), {}).items():
            normalized_cat_skills = [self.normalize_skill(s) for s in cat_skills]
            if self.normalize_skill(skill) in normalized_cat_skills:
                return category
        return None

    def extract_skills(self, text: str, industry: str = "technology") -> Dict[str, List[dict]]:
        """
        Extracts skills from the provided text, categorizes them using SKILL_CATEGORIES.
        In case of errors, an "errors" key is added with relevant messages.
        """
        if not isinstance(text, str) or not text.strip():
            logger.error("Skill extraction failed: Input must be a non-empty string")
            cats = SKILL_CATEGORIES.get(industry.lower(), SKILL_CATEGORIES["general"])
            result = {cat: [] for cat in cats}
            result["general"] = []
            result["errors"] = ["Skill extraction failed due to invalid input"]
            return result

        cats = SKILL_CATEGORIES.get(industry.lower(), SKILL_CATEGORIES["technology"])
        result = {cat: [] for cat in cats}
        result["general"] = []
        result["errors"] = []

        try:
            pre_text = self._preprocess_text(text)
            raw_skills = self.extract_skill_context(pre_text)
            logger.debug(f"Raw extracted skills: {raw_skills}")
        except Exception as e:
            logger.error(f"Skill extraction error: {str(e)}")
            result["errors"].append("Skill extraction process encountered an issue")
            return result

        industry_skills = self._get_industry_skills(industry)
        for sk in raw_skills:
            skill_name = sk.get("skill", "").strip()
            if not skill_name:
                logger.debug("Skipping empty skill name.")
                continue

            lower_sn = skill_name.lower()
            if lower_sn in industry_skills:
                normalized_skill = self.normalize_skill(skill_name)
                sk["skill"] = normalized_skill
                placed = False

                for cat_name, cat_skills in cats.items():
                    normalized_cat_skills = {self.normalize_skill(s).lower() for s in cat_skills}
                    if lower_sn in normalized_cat_skills:
                        result[cat_name].append(sk)
                        placed = True
                        break

                if not placed:
                    result["general"].append(sk)
            else:
                # If not recognized in industry, put it in errors
                result["errors"].append(sk)

        # Deduplicate
        for cat_name in result:
            seen = set()
            unique_skills = []
            for skill_entry in result[cat_name]:
                norm_low = skill_entry["skill"].lower()
                if norm_low not in seen:
                    seen.add(norm_low)
                    unique_skills.append(skill_entry)
            result[cat_name] = unique_skills

        logger.debug(f"Final extracted skills: {result}")
        return result
