# services/industry_config.py
from typing import Dict, List, Set, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging
import re
from .shared_utilities import (
    SKILL_CATEGORIES, 
    SECTION_SYNONYMS, 
    INDUSTRY_SYNONYMS, 
    INDUSTRY_CONFIGS
)

logger = logging.getLogger(__name__)

class IndustryConfig:
    def __init__(
        self,
        name: str,
        required_sections: List[str],
        section_weights: Dict[str, float],
        keywords: List[str],
        skill_categories: Dict[str, List[str]],
        experience_patterns: List[str]
    ):
        self.name = name
        # Normalize required sections to canonical form using SECTION_SYNONYMS
        self.required_sections = [SECTION_SYNONYMS.get(s.lower(), s.lower()) for s in required_sections]
        self.section_weights = section_weights
        self.keywords = keywords
        self.skill_categories = skill_categories
        self.experience_patterns = experience_patterns


class IndustryDetector:
    """
    A comprehensive industry detection system that uses multiple strategies
    to accurately classify job descriptions and resumes.
    
    Strategies include:
    1. Job title pattern matching
    2. Industry-specific keyword counting
    3. TF-IDF similarity calculation
    4. Experience pattern matching
    """
    
    def __init__(self):
        try:
            # Try to import sentence transformers, but don't require it
            from sentence_transformers import SentenceTransformer, util
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            self.util = util
            self.use_embeddings = True
        except (ImportError, Exception) as e:
            logger.warning(f"Sentence transformer not available, falling back to TF-IDF: {str(e)}")
            self.use_embeddings = False
            
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Initialize industry profiles
        self.industry_profiles = {}
        self.job_title_patterns = self._compile_job_title_patterns()
        self._initialize_industry_profiles()
    
    def _compile_job_title_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regular expressions for job title matching."""
        patterns = {
            'technology': re.compile(r'\b(software|web|frontend|backend|full.?stack|devops|cloud|data)\s+(developer|engineer|architect|programmer|scientist|analyst)\b', re.IGNORECASE),
            'engineering': re.compile(r'\b(civil|structural|mechanical|electrical|chemical|biomedical|environmental|industrial|aerospace|systems)\s+(engineer|technician|designer|drafter)\b|\b(engineering|CAD)\b', re.IGNORECASE),
            'healthcare': re.compile(r'\b(doctor|physician|nurse|rn|medical|clinical|healthcare|patient|pharma|biotech|hospital)\b', re.IGNORECASE),
            'finance': re.compile(r'\b(financial|accountant|auditor|banking|investment|trader|analyst|portfolio|risk|compliance|underwriter|actuary)\b', re.IGNORECASE),
            'legal': re.compile(r'\b(attorney|lawyer|legal|paralegal|counsel|compliance|regulatory|contract)\b', re.IGNORECASE),
            'education': re.compile(r'\b(teacher|professor|instructor|educator|curriculum|education|teaching|academic|faculty|school|university|tutor)\b', re.IGNORECASE),
            'creative': re.compile(r'\b(designer|artist|writer|creative|content|ux|ui|graphic|brand|marketing|media|production)\b', re.IGNORECASE),
            'business': re.compile(r'\b(business|marketing|sales|operations|management|strategy|product|project|program|executive|director|manager|consultant)\b', re.IGNORECASE),
        }
        return patterns
        
    def _initialize_industry_profiles(self):
        """Initialize industry profiles for similarity comparisons."""
        for industry, config in INDUSTRY_CONFIGS.items():
            # Combine keywords and experience patterns for a rich industry profile
            keywords = config.get('keywords', [])
            experience_patterns = config.get('experience_patterns', [])
            
            # Get all associated skills for this industry
            industry_skills = []
            skill_categories = SKILL_CATEGORIES.get(industry, {})
            for category, skills in skill_categories.items():
                if isinstance(skills, list):
                    industry_skills.extend(skills)
                elif isinstance(skills, dict):
                    for subcategory, subcategory_skills in skills.items():
                        industry_skills.extend(subcategory_skills)
            
            # Create the profile text
            profile_text = ' '.join(keywords + experience_patterns + industry_skills)
            
            # Store the profile text for later use
            self.industry_profiles[industry] = profile_text
    
    # Modified detect_industry method for IndustryDetector class

    def detect_industry(self, text: str) -> Tuple[str, float]:
        """
        Detect the most likely industry for the given text.
        Returns a tuple of (industry_name, confidence_score)
        """
        text = text.lower()
        
        # Initialize industry scores for all known industries from INDUSTRY_CONFIGS
        industry_scores = {industry: 0.0 for industry in INDUSTRY_CONFIGS.keys()}
        
        # Add all industries from job title patterns to ensure they exist in the dictionary
        for industry in self.job_title_patterns.keys():
            if industry not in industry_scores:
                industry_scores[industry] = 0.0
        
        # Strategy 1: Job title pattern matching (highest weight)
        for industry, pattern in self.job_title_patterns.items():
            matches = pattern.findall(text)
            if matches:
                industry_scores[industry] += len(matches) * 2.0  # Higher weight for job title matches
        
        # Strategy 2: Industry-specific keyword counting
        for industry, config in INDUSTRY_CONFIGS.items():
            keywords = config.get('keywords', [])
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text):
                    industry_scores[industry] += 0.5
        
        # Strategy 3: Experience pattern matching
        for industry, config in INDUSTRY_CONFIGS.items():
            patterns = config.get('experience_patterns', [])
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                industry_scores[industry] += len(matches) * 0.3
        
        # Strategy 4: TF-IDF similarity 
        try:
            # Convert industry profiles and input text to TF-IDF vectors
            profile_texts = list(self.industry_profiles.values())
            all_texts = profile_texts + [text]
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            
            # Calculate similarities between input text and industry profiles
            text_vector = tfidf_matrix[-1]
            for i, industry in enumerate(self.industry_profiles.keys()):
                profile_vector = tfidf_matrix[i]
                similarity = cosine_similarity(text_vector, profile_vector)[0][0]
                industry_scores[industry] += similarity * 1.5  # Good weight for TF-IDF similarity
        except Exception as e:
            logger.warning(f"TF-IDF similarity calculation failed: {str(e)}")
        
        # Get the best match and confidence score
        if not industry_scores:
            return "general", 0.0
            
        best_industry = max(industry_scores.items(), key=lambda x: x[1])
        industry_name = best_industry[0]
        # Normalize confidence to 0-1 range
        confidence = min(best_industry[1] / 10.0, 1.0)  
        
        # Apply industry synonym mapping if needed
        normalized_industry = INDUSTRY_SYNONYMS.get(industry_name.lower(), industry_name)
        
        # If confidence is too low, fall back to "general"
        if confidence < 0.2:
            logger.info(f"Low confidence ({confidence:.2f}) for industry detection, using 'general'")
            return "general", confidence
            
        logger.info(f"Detected industry: {normalized_industry} with confidence {confidence:.2f}")
        return normalized_industry, confidence
        
    def get_industry_skill_categories(self, industry: str) -> Dict:
        """Get skill categories for a specific industry."""
        industry = INDUSTRY_SYNONYMS.get(industry.lower(), industry.lower())
        return SKILL_CATEGORIES.get(industry, SKILL_CATEGORIES.get("general", {}))
        
    def get_required_skills_for_industry(self, industry: str, job_description: str = None) -> Set[str]:
        """
        Get a set of skills that are typically required for a specific industry.
        If job_description is provided, it is used to refine the skill set.
        """
        industry = INDUSTRY_SYNONYMS.get(industry.lower(), industry.lower())
        
        # Start with industry-specific skills
        skill_categories = self.get_industry_skill_categories(industry)
        all_skills = set()
        
        # Extract skills from all categories
        for category, skills in skill_categories.items():
            if isinstance(skills, list):
                all_skills.update([s.lower() for s in skills])
            elif isinstance(skills, dict):
                for subcategory, subcategory_skills in skills.items():
                    all_skills.update([s.lower() for s in subcategory_skills])
        
        # If job description is provided, refine skills based on it
        if job_description:
            refined_skills = set()
            
            # Only keep skills that are mentioned in the job description
            for skill in all_skills:
                # Check for direct mentions
                if skill.lower() in job_description.lower():
                    refined_skills.add(skill)
                    continue
                
                # Check for partial matches (e.g., "Python" in "Python programming")
                skill_words = skill.lower().split()
                if any(word in job_description.lower() for word in skill_words if len(word) > 3):
                    refined_skills.add(skill)
            
            # If we have enough refined skills, use those
            if len(refined_skills) >= 5:
                return refined_skills
        
        return all_skills


class IndustryAnalysisConfig:
    # A default configuration to use if no matching industry is found.
    DEFAULT_CONFIG = IndustryConfig(
        name="general",
        required_sections=["skills", "experience"],
        section_weights={"skills": 0.5, "experience": 0.5},
        keywords=["general"],
        skill_categories={"management": ["leadership"]},
        experience_patterns=[r"experience"]
    )

    def __init__(self):
        # Build configurations from the global INDUSTRY_CONFIGS constant.
        self.configs = {
            industry: IndustryConfig(
                name=industry,
                required_sections=config["required_sections"],
                section_weights=config["section_weights"],
                keywords=config["keywords"],
                skill_categories=config["skill_categories"],
                experience_patterns=config["experience_patterns"]
            )
            for industry, config in INDUSTRY_CONFIGS.items()
        }
        
        # Initialize the industry detector
        self.industry_detector = IndustryDetector()

    def _calculate_weights(self, job_description: str, required_sections: List[str]) -> Dict[str, float]:
        """
        Dynamically calculates section weights from the job description using TF-IDF.
        Returns a dictionary mapping each required section to a normalized weight.
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([job_description])
        except ValueError:
            return {section: 1.0 / len(required_sections) for section in required_sections}
            
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = dict(zip(feature_names, tfidf_matrix.toarray()[0]))
        section_scores = defaultdict(float)
        for section in required_sections:
            # Gather section synonyms
            section_terms = [term for term, canon in SECTION_SYNONYMS.items() if canon == section]
            # For the skills section, add additional terms from the skill taxonomy (using "technology" as default)
            if section == "skills":
                skill_terms = []
                for cat in SKILL_CATEGORIES.get("technology", {}):
                    skill_terms.extend([term.lower() for term in SKILL_CATEGORIES["technology"].get(cat, [])])
                section_terms += skill_terms
            section_scores[section] = sum(tfidf_scores.get(term, 0) for term in section_terms)
        total = sum(section_scores.values()) or 1.0
        return {section: section_scores[section] / total for section in required_sections}

    def _enhance_patterns(self, base_patterns: List[str], industry: str) -> List[str]:
        """
        Enhances the provided regex patterns by appending additional keywords
        from the industry's skill categories and synonyms.
        """
        enhanced = list(base_patterns)
        industry_key = industry.lower()
        if industry_key in SKILL_CATEGORIES:
            skill_terms = []
            for cat in SKILL_CATEGORIES[industry_key].values():
                skill_terms.extend(cat)
            if skill_terms:
                enhanced.append(fr'\b({"|".join(map(re.escape, [s.lower() for s in skill_terms]))})\b')
        for syn, canon in INDUSTRY_SYNONYMS.items():
            if canon.lower() == industry_key:
                enhanced.append(fr'\b{re.escape(syn.lower())}\b')
        return enhanced

    def _identify_industry(self, job_description: str) -> str:
        """
        Use the IndustryDetector to identify the industry from a job description.
        """
        industry, confidence = self.industry_detector.detect_industry(job_description)
        logger.info(f"Industry detection: {industry} (confidence: {confidence:.2f})")
        return industry

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes the text by converting to lowercase and replacing industry synonyms.
        """
        text = text.lower()
        for syn, canon in INDUSTRY_SYNONYMS.items():
            text = re.sub(rf'\b{re.escape(syn)}\b', canon, text)
        return text

    def get_config(self, job_description: str) -> IndustryConfig:
        """
        Returns an IndustryConfig instance based on the job description.
        Uses the IndustryDetector for more accurate industry identification.
        """
        industry = self._identify_industry(job_description)
        config_data = INDUSTRY_CONFIGS.get(industry, None)
        if not config_data:
            return self.DEFAULT_CONFIG
            
        section_weights = self._calculate_weights(job_description, config_data["required_sections"])
        return IndustryConfig(
            name=config_data['name'],
            keywords=config_data['keywords'] + self._get_skill_keywords(config_data),
            required_sections=config_data['required_sections'],
            section_weights=section_weights,
            skill_categories=config_data['skill_categories'],
            experience_patterns=self._enhance_patterns(config_data['experience_patterns'], config_data['name'].lower())
        )

    def validate_section_weights(self, weights: Dict[str, float]) -> None:
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Section weights sum to {total:.2f}, must be 1.0")
        if any(weight < 0 or weight > 1 for weight in weights.values()):
            raise ValueError("All weights must be between 0 and 1")

    def get_custom_config(self, 
                          name: str,
                          keywords: List[str],
                          required_sections: List[str],
                          section_weights: Dict[str, float],
                          skill_categories: List[str],
                          experience_patterns: List[str]) -> IndustryConfig:
        """
        Creates and returns a custom IndustryConfig instance after validating the provided parameters.
        """
        self.validate_required_sections(required_sections)
        self.validate_section_weights(section_weights)
        all_categories = set()
        for industry in SKILL_CATEGORIES.values():
            all_categories.update(industry.keys())
        invalid_cats = set(skill_categories) - all_categories
        if invalid_cats:
            raise ValueError(f"Invalid skill categories: {invalid_cats}")
        config_for_skills = {
            'name': name,
            'skill_categories': skill_categories
        }
        return IndustryConfig(
            name=name,
            keywords=keywords + self._get_skill_keywords(config_for_skills),
            required_sections=[SECTION_SYNONYMS.get(s.lower(), s) for s in required_sections],
            section_weights=section_weights,
            skill_categories=config_for_skills['skill_categories'],
            experience_patterns=self._enhance_patterns(experience_patterns, name.lower())
        )

    def validate_required_sections(self, sections: List[str]) -> None:
        """
        Validates that every required section (after canonical mapping) is in the list of valid sections.
        """
        canonical_sections = [
            SECTION_SYNONYMS.get(s.lower().strip(), s.lower().strip()) 
            for s in sections
        ]
        valid_sections = set(SECTION_SYNONYMS.values())
        invalid = set(canonical_sections) - valid_sections
        if invalid:
            raise ValueError(f"Invalid sections: {invalid}. Valid options: {valid_sections}")
        
    def _get_skill_keywords(self, config: dict) -> list:
        """
        Extracts all lower-case skill keywords from the provided configuration.
        """
        industry_key = config['name'].lower()
        if industry_key not in SKILL_CATEGORIES:
            return []
        return [
            term.lower()
            for category in config['skill_categories']
            for term in SKILL_CATEGORIES[industry_key].get(category, [])
        ]