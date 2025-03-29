# services/feedback_generator.py

"""
Resume Feedback Generator
Generates resume feedback using a combination of rule-based analysis and
LLM-based skill/action generation approach.
"""

import os
import re
import logging
import random
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
import openai
import spacy
import json
from spacy.matcher import PhraseMatcher
from transformers import pipeline
from nltk.corpus import wordnet
from .industry_config import IndustryAnalysisConfig, IndustryDetector, IndustryConfig
from .shared_utilities import (
    SKILL_CATEGORIES,
    INDUSTRY_CONFIGS,
    INDUSTRY_SYNONYMS,
    SEED_VERBS,
    WEAK_VERBS,
    GENERAL_FEEDBACK_RULES,
)

logger = logging.getLogger(__name__)


class ResumeFeedbackGenerator:
    """
    Generates resume feedback using a combination of rule-based analysis and,
    if available, a pre-trained or LLM-based approach for skills and achievements.
    
    Key features:
      - Optional model for zero-shot classification (Transformers).
      - Industry configurations to extract required skills and adjust weighting.
      - NLP analysis (using spaCy) of the resume text for style and keyword density.
      - Dynamic generation of recommendations (action verbs, achievements, etc.)
        using OpenAI's ChatCompletion, if an API key is available.
    """

    def __init__(self):
        # Validate required constants before initializing components
        self._validate_constants()

        # Initialize optional ML components
        try:
            self.skill_predictor = pipeline('zero-shot-classification', model='facebook/bart-large-mnli', revision="main", force_download=True)
        except Exception as e:
            logger.warning(
                "Transformer model for skill prediction not loaded. "
                "Proceeding without ML. Error: %s", str(e)
            )
            self.skill_predictor = None

        # Initialize core NLP components
        try:
            self.nlp = spacy.load("en_core_web_lg")
        except OSError:
            logger.error("spaCy model 'en_core_web_lg' not found. "
                         "Install with: python -m spacy download en_core_web_lg")
            raise
        except Exception as e:
            logger.error("Error loading spaCy model: %s", str(e))
            raise

        # Initialize industry configuration analyzer
        self.industry_config = IndustryAnalysisConfig()


    def _validate_constants(self) -> None:
        """
        Ensure required configuration constants are properly set.
        Removed reference to ACHIEVEMENT_TEMPLATES, as we now use a dynamic approach.
        """
        required_constants = [
            SKILL_CATEGORIES,
            SEED_VERBS,
            WEAK_VERBS,
            GENERAL_FEEDBACK_RULES
        ]
        # If any of them is empty or None, raise an error
        if any(not c for c in required_constants):
            logger.error("Missing required constants in configuration")
            raise ValueError("Incomplete constants configuration")
        
    def _normalize_position(self, position) -> dict:
        """
        Ensure that a position is represented as a dictionary with
        at least a 'title' key and a 'bullets' list.
        
        If the input is a string, it is assumed to be the title.
        If it is a dict, it is returned as is.
        Otherwise, the method converts the input to a string for the title.
        """
        if isinstance(position, dict):
            # Optionally, you can check if required keys are missing and fill them in.
            normalized = {
                "title": position.get("title", str(position)),
                "bullets": position.get("bullets", [])
            }
            return normalized
        elif isinstance(position, str):
            return {"title": position, "bullets": []}
        else:
            return {"title": str(position), "bullets": []}


    def _get_industry_context(self, job_description: str) -> Tuple[Optional[IndustryConfig], str]:
        """
        Determine industry context from job description using IndustryAnalysisConfig.
        
        Note: This method uses IndustryDetector indirectly through IndustryAnalysisConfig
        to maintain consistency with other industry detection methods and to reuse
        any configuration that happens during IndustryAnalysisConfig initialization.
        
        Args:
            job_description: The text to analyze for industry detection
            
        Returns:
            A tuple containing (industry_config, industry_name)
        """
        # Get an instance of IndustryAnalysisConfig and use its industry_detector
        industry_config = IndustryAnalysisConfig()
        industry_detector = industry_config.industry_detector
        
        # Detect industry with confidence score
        if job_description:
            detected_industry, confidence = industry_detector.detect_industry(job_description)
            if confidence > 0.3:  # Use detected industry if confidence is reasonable
                industry_name = detected_industry.lower()
                logger.info(f"Industry detection: {industry_name} (confidence: {confidence:.2f})")
            else:
                # Fall back to the original method if confidence is low
                cfg = self.industry_config.get_config(job_description)
                industry_name = cfg.name.lower() if cfg else 'general'
        else:
            industry_name = 'general'
        
        # Map through synonyms
        industry_name = INDUSTRY_SYNONYMS.get(industry_name, industry_name)
        
        # Get the config for this industry
        industry_cfg = None
        if industry_name in INDUSTRY_CONFIGS:
            industry_cfg = self.industry_config.get_config(job_description)
        
        return industry_cfg, industry_name

    def _get_industry_required_skills(self, industry: str) -> Set[str]:
        """
        Get normalized set of required skills for an industry.
        
        Note: This method uses IndustryDetector indirectly through IndustryAnalysisConfig
        to maintain consistency with industry skill retrieval across the codebase.
        
        Args:
            industry: The industry to get required skills for
            
        Returns:
            A set of required skills for the specified industry
        """
        # Get an instance of IndustryAnalysisConfig and use its industry_detector
        industry_config = IndustryAnalysisConfig()
        industry_detector = industry_config.industry_detector
        
        return industry_detector.get_required_skills_for_industry(industry)

    @staticmethod
    def get_wordnet_synonyms(base_verb: str, max_synonyms: int = 5) -> List[str]:
        """
        Returns a list of synonyms for a given verb from WordNet.
        """
        synonyms = set()
        for syn in wordnet.synsets(base_verb, pos=wordnet.VERB):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_', ' ')
                if lemma_name.lower() != base_verb.lower():
                    synonyms.add(lemma_name)
        return list(synonyms)[:max_synonyms]

    @staticmethod
    def suggest_action_verbs_hybrid_wordnet(industry: str, max_seeds: int = 3, max_synonyms: int = 3) -> List[str]:
        """
        Merges a small curated set of seed verbs with synonyms from WordNet.
        """
        base_verbs = SEED_VERBS.get(industry.lower(), SEED_VERBS['general'])[:max_seeds]
        suggestions = set()

        for verb in base_verbs:
            suggestions.add(verb)
            # Expand with WordNet synonyms
            synonyms = ResumeFeedbackGenerator.get_wordnet_synonyms(verb, max_synonyms)
            suggestions.update(synonyms)

        return list(suggestions)

    @staticmethod
    def expand_verbs_with_llm(base_verbs: List[str],
                              job_description: str = "",
                              max_synonyms_per_verb: int = 3) -> List[str]:
        """
        Use OpenAI GPT-3.5 or GPT-4 to generate synonyms/related action verbs 
        for each verb in base_verbs, optionally tailored to a job description.
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            # fallback if no key
            return base_verbs

        openai.api_key = openai_api_key
        new_verbs = set()

        for verb in base_verbs:
            prompt = f"""
            The job description is: {job_description}
            Suggest {max_synonyms_per_verb} synonyms or alternate action verbs 
            closely matching '{verb}' and suitable for a resume in this context.
            Only list the verbs, comma-separated.
            """

            try:
                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.7
                )
                synonyms_str = response.choices[0].message.content.strip()
                synonyms = [s.strip() for s in synonyms_str.split(',') if s.strip()]
                new_verbs.add(verb)
                new_verbs.update(synonyms)
            except Exception as e:
                logger.warning("LLM expansion failed for verb '%s': %s", verb, str(e))
                new_verbs.add(verb)

        return list(new_verbs)

    @staticmethod
    def suggest_action_verbs_hybrid_llm(industry: str,
                                        job_description: str,
                                        max_seeds: int = 3) -> List[str]:
        """
        Hybrid approach:
          1) Take a small curated set of domain-specific seed verbs
          2) Expand them using LLM synonyms
        """
        base_verbs = SEED_VERBS.get(industry.lower(), SEED_VERBS['general'])[:max_seeds]
        verbs_llm = ResumeFeedbackGenerator.expand_verbs_with_llm(base_verbs, job_description)
        return verbs_llm


    def _generate_dynamic_achievements(self,
                                       position: dict,
                                       analysis: dict,
                                       industry: str) -> List[str]:
        """
        Dynamically generate achievements using an LLM (e.g., OpenAI).
        
        Inputs:
          - position: a dict with keys like 'title', 'duration', 'bullets', etc.
          - analysis: the entire analysis dict (can be used for skill sets, etc.)
          - industry: a string for industry context
        
        Returns:
          A list of bullet-point achievement statements.
        """

        role = position.get("title", "Unknown Role")
        bullets = position.get("bullets", [])  # existing partial achievements from parsing
        # Optionally gather relevant skills from analysis
        # Flatten them for demonstration:
        skill_dict = analysis.get('skills', {})
        relevant_skills = []
        for cat_list in skill_dict.values():
            if isinstance(cat_list, list):
                # Each cat_list is something like [{'skill': 'Python', ...}, ...]
                # or it could be a list of skill dictionaries
                for item in cat_list:
                    if isinstance(item, dict) and 'skill' in item:
                        relevant_skills.append(item['skill'])
                    elif isinstance(item, str):
                        relevant_skills.append(item)

        # If we have a job_description stored:
        job_description = analysis.get('job_match', {}).get('description', '') or ''

        # Construct the prompt
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            # No key available, fallback to a minimal approach or return existing bullets
            logger.info("No OPENAI_API_KEY found. Returning original bullets as fallback.")
            return bullets[:2] if bullets else []

        openai.api_key = openai_key
        bullet_snippets = "\n- " + "\n- ".join(bullets) if bullets else "(no existing bullets)"
        joined_skills = ", ".join(list(set(relevant_skills))[:10]) if relevant_skills else "None"

        prompt = f"""
        You are an expert resume writer. 
        The candidate has a role: {role}.
        They have these existing bullet points: {bullet_snippets}
        They have these key skills: {joined_skills}
        Here is the job description for context (if available): {job_description}

        Generate 2 concise bullet-point achievements that highlight key impacts, 
        using strong action verbs and measurable results if possible. 
        Write them in a professional resume style (no extra commentary).
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            text_output = response.choices[0].message.content.strip()
            # Split on newlines, ignoring blank lines
            lines = [line.strip() for line in text_output.split('\n') if line.strip()]
            return lines
        except Exception as e:
            logger.warning("LLM achievement generation failed: %s", str(e))
            # Fallback: return original bullets
            return bullets[:2] if bullets else []

    def _add_achievement_feedback(self, feedback: Dict, analysis: Dict, industry: str) -> None:
        """
        Add achievement-based recommendations using dynamic LLM generation.
        """
        positions = analysis.get('experience', {}).get('positions')
        if not positions:
            logger.info("No experience data for achievement feedback.")
            return

        for pos in positions:
            try:
                position = self._normalize_position(pos)
                suggestions = self._generate_dynamic_achievements(position, analysis, industry)
                # Add only up to 2 suggestions to keep it concise
                feedback['recommendations'].extend(suggestions[:2])
            except Exception as e:
                pos_title = self._normalize_position(pos).get('title', 'Unknown')
                logger.warning("Dynamic achievement generation failed for position '%s': %s", pos_title, str(e))


    def _add_verb_suggestions(self, feedback: Dict, industry: str, job_description: str = "") -> None:
        """
        Suggest industry-specific action verbs using a hybrid approach 
        (seed verbs + LLM synonyms).
        """
        # Option 1) WordNet approach (commented out):
        # verbs = self.suggest_action_verbs_hybrid_wordnet(industry)

        # Option 2) LLM approach:
        verbs = self.suggest_action_verbs_hybrid_llm(industry, job_description)
        if not verbs:
            return

        recommended = ', '.join(verbs[:5])
        feedback['recommendations'].append(
            f"Use stronger, domain-focused action verbs (e.g., {recommended})."
        )

    @staticmethod
    def find_weak_verbs_and_suggestions(doc, max_synonyms=3) -> dict:
        """
        Given a spaCy doc, find all verbs that appear in WEAK_VERBS 
        and suggest stronger replacements using WordNet or default synonyms.
        Return a dict: {weak_verb_in_text: [list_of_suggestions]}
        """
        suggestions_map = {}

        for token in doc:
            if token.pos_ == "VERB":
                lemma = token.lemma_.lower()
                if lemma in WEAK_VERBS:
                    # Try WordNet synonyms first
                    better_options = ResumeFeedbackGenerator.get_wordnet_synonyms(lemma, max_synonyms=max_synonyms)
                    if not better_options:
                        better_options = ["improved", "organized"]  # fallback
                    suggestions_map[token.text] = better_options
        return suggestions_map

    def _add_weak_verb_feedback(self, feedback: Dict, analysis: Dict) -> None:
        raw_text = analysis.get("raw_text", "")
        if not raw_text.strip():
            return

        doc = self.nlp(raw_text)
        suggestions_map = {}

        for token in doc:
            if token.pos_ == "VERB":
                lemma = token.lemma_.lower()
                text_lower = token.text.lower()
                if lemma in WEAK_VERBS or text_lower in WEAK_VERBS:
                    better_options = self.get_wordnet_synonyms(lemma, max_synonyms=3)
                    if not better_options:
                        better_options = ["enhanced", "improved", "led"]  # fallback
                    suggestions_map[token.text] = better_options

        if suggestions_map:
            lines = []
            for weak_verb, replacements in suggestions_map.items():
                lines.append(f"'{weak_verb}' -> try: {', '.join(replacements)}")

            feedback['improvements'].append(
                "Replace weak verbs with stronger ones:\n  " + "\n  ".join(lines)
            )

    def _add_education_feedback(self, feedback: Dict, analysis: Dict) -> None:
        """Provide feedback on the education section."""
        if not analysis.get("education"):
            feedback['improvements'].append("Include relevant education/certifications.")
        else:
            feedback['strengths'].append("Strong educational background noted.")

    def _add_job_match_feedback(self, feedback: Dict, job_match: Dict) -> None:
        """
        Add job matching analysis feedback (skills matched/missing, etc.).
        """
        skills_match = job_match.get('skills_match', {})
        missing_required = skills_match.get('required', {}).get('missing', [])
        matched_required = skills_match.get('required', {}).get('matched', [])
        if missing_required:
            feedback['improvements'].append(f"Missing requirements: {', '.join(missing_required[:3])}")
        if matched_required:
            feedback['strengths'].append(f"Matched requirements: {', '.join(matched_required[:3])}")

    def _finalize_feedback(self, feedback: Dict) -> Dict:
        """Deduplicate and limit feedback items."""
        for category in feedback:
            seen = set()
            filtered = []
            for item in feedback[category]:
                if item not in seen:
                    filtered.append(item)
                    seen.add(item)
            feedback[category] = filtered[:5]  # keep only up to 5 unique items
        return feedback

    def _detect_passive_voice(self, text: str) -> int:
        """Count passive voice constructions using spaCy."""
        doc = self.nlp(text)
        return sum(1 for token in doc if token.dep_ in ("auxpass", "nsubjpass"))

    def _add_style_feedback(self, feedback: Dict, analysis: Dict) -> None:
        """Provide writing style feedback (e.g., detect passive voice)."""
        raw_text = analysis.get("raw_text", "")
        if not raw_text.strip():
            return

        passive_count = self._detect_passive_voice(raw_text)
        threshold = GENERAL_FEEDBACK_RULES["passive_voice"]["threshold"]
        if passive_count > threshold:
            message = GENERAL_FEEDBACK_RULES["passive_voice"]["message"].format(count=passive_count)
            feedback["improvements"].append(message)

    def _validate_sections(self,
                           feedback: Dict,
                           analysis: Dict,
                           industry_config: Optional[IndustryConfig]) -> None:
        """Ensure required industry-specific sections are present."""
        base_sections = ["experience", "skills", "education"]
        required_sections = industry_config.required_sections if industry_config else base_sections

        missing = [s for s in required_sections if s not in analysis.get('sections', {})]
        if missing:
            feedback['improvements'].append(f"Missing sections: {', '.join(missing)}")

    def _analyze_skills(self, feedback: Dict, analysis: Dict, industry: str) -> None:
        """Rule-based skill analysis (keyword density, presence of industry terms, etc.)."""
        try:
            raw_text = analysis.get("raw_text", "")
            if not raw_text.strip():
                feedback["improvements"].append("Insufficient content for skill analysis.")
                return

            doc = self.nlp(raw_text.lower())
            total_words = len(doc)
            if total_words == 0:
                feedback["improvements"].append("Insufficient content for skill analysis.")
                return

            industry_terms = self._get_industry_required_skills(industry)

            tokens = [token.lemma_.strip() for token in doc if not token.is_punct and not token.is_stop]
            found_terms = set(tokens) & industry_terms

            from spacy.vocab import Vocab
            if isinstance(doc.vocab, Vocab):
                from spacy.matcher import PhraseMatcher
                matcher = PhraseMatcher(doc.vocab)
                patterns = [self.nlp.make_doc(phrase) for phrase in industry_terms if ' ' in phrase]
                matcher.add("INDUSTRY_PHRASES", patterns)
                matches = matcher(doc)
                found_phrases = {doc[start:end].text for _, start, end in matches}
                found_terms.update(found_phrases)

            keyword_count = len(found_terms)
            density = keyword_count / total_words
            threshold = GENERAL_FEEDBACK_RULES["keyword_density"]

            if density < threshold:
                missing_skills = sorted(industry_terms - found_terms)[:3]
                feedback["improvements"].append(
                    f"Increase industry keywords ({density:.1%} < {threshold:.0%}): {', '.join(missing_skills)}"
                )
            if found_terms:
                feedback["strengths"].append(
                    "Relevant skills: " + ", ".join(s.capitalize() for s in sorted(found_terms)[:5])
                )
        except Exception as e:
            logger.error("Skill analysis error: %s", str(e))
            feedback["improvements"].append("Skill analysis failed - check section formatting.")

    def _add_general_feedback(self, feedback: Dict, analysis: Dict) -> None:
        """Add general resume quality feedback (e.g., word count, skill balance)."""
        general_rules = GENERAL_FEEDBACK_RULES
        stats = analysis.get('basic_stats', {})
        word_count = stats.get('word_count', 0)

        # Word count feedback
        if word_count > general_rules['word_count']['high']:
            feedback['improvements'].append(general_rules['word_count']['high_message'])
        elif word_count < general_rules['word_count']['low']:
            feedback['improvements'].append(general_rules['word_count']['low_message'])

        # Skill balance feedback
        skills = analysis.get('skills', {})
        tech_count = len(skills.get('technical', []))
        soft_count = len(skills.get('soft_skills', []))

        if tech_count == 0:
            feedback['improvements'].append("No technical skills listed.")
        elif tech_count > 0 and (soft_count / tech_count) < 0.6:
            feedback['improvements'].append(general_rules['skill_balance'])

    def _generate_dynamic_feedback_for_industry(self, analysis: dict, industry: str, job_description: str = "") -> dict:
        """
        Use an LLM (e.g., OpenAI GPT) to generate dynamic feedback for the resume.
        The prompt explicitly requests three lists:
        1. Strengths
        2. Improvements (actionable advice for enhancing the resume)
        3. Recommendations (specific steps to take, not just rehashed achievements)
        """
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return {
                "strengths": [f"No OpenAI key found. Static fallback for {industry}."],
                "improvements": [],
                "recommendations": []
            }

        openai.api_key = openai_api_key

        # Gather candidate info from the analysis.
        skills_found = analysis.get("skills", {})
        skill_snippets = []
        for _cat, skill_list in skills_found.items():
            for item in skill_list[:5]:
                if isinstance(item, dict):
                    skill_snippets.append(item.get('skill', ''))
                elif isinstance(item, str):
                    skill_snippets.append(item)
        skill_text = ", ".join(list(set(skill_snippets))) if skill_snippets else "None found"

        experience = analysis.get("experience", {})
        positions = experience.get("positions", [])
        roles_text = ", ".join(positions) if positions else "Not specified"

        prompt = f"""
    You are an expert resume consultant specializing in the {industry.capitalize()} domain.
    The candidate's resume details include:
    - Positions: {roles_text}
    - Key Skills: {skill_text}
    - Job Description Context (if available): {job_description[:300]}

    Please generate feedback in three distinct categories as valid JSON:
    1. "strengths": Bullet points that highlight the candidate's accomplishments and what they do well.
    2. "improvements": Bullet points that provide specific, actionable advice for improving the resume (e.g., clarifying responsibilities, adding missing certifications, improving bullet point structure).
    3. "recommendations": Bullet points that outline concrete next steps the candidate can take to enhance their resume (for example, reorganizing sections or using more impactful language). Avoid listing achievements here.

    The JSON output must have exactly these keys: "strengths", "improvements", and "recommendations", and each key must map to a list of concise bullet-point strings.
    For example:
    {{
    "strengths": ["Strong clinical expertise demonstrated in patient care", "..."],
    "improvements": ["Clarify responsibilities in the 'Experience' section", "..."],
    "recommendations": ["Revise bullet points to include quantifiable metrics", "..."]
    }}
        """

        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            content = response.choices[0].message.content.strip()
            try:
                feedback_json = json.loads(content)
            except json.JSONDecodeError:
                feedback_json = {
                    "strengths": [f"Unable to parse JSON from LLM: {content[:100]}..."],
                    "improvements": [],
                    "recommendations": []
                }
            for key in ["strengths", "improvements", "recommendations"]:
                if key not in feedback_json or not isinstance(feedback_json[key], list):
                    feedback_json[key] = []
            return feedback_json
        except Exception as e:
            return {
                "strengths": [],
                "improvements": [f"LLM call failed: {str(e)}"],
                "recommendations": []
            }


    def generate_feedback(self, analysis_results: Dict, job_description: str = None) -> Dict:
        """
        Generate comprehensive resume feedback based on analysis and job description.
        
        1) Runs a series of rule-based checks: validating sections, skill analysis, style feedback, etc.
        2) Optionally uses LLM-based steps: achievements generation, dynamic industry feedback, verb suggestions.
        3) Consolidates all feedback into strengths/improvements/recommendations, then deduplicates via _finalize_feedback.
        """

        # Initialize the feedback structure
        feedback = {
            'strengths': [],
            'improvements': [],
            'recommendations': []
        }

        # Optional: we can capture raw_text if needed for style checks or passive voice detection
        # (We use it in _add_style_feedback → _detect_passive_voice)
        # If you don't need it, you can remove or comment this out.
        raw_text = analysis_results.get("raw_text", "")

        # 1) Determine the industry context from the job description
        industry_cfg, industry_name = self._get_industry_context(job_description or "")

        # 2) Validate required sections (based on industry)
        self._validate_sections(feedback, analysis_results, industry_cfg)

        # 3) Perform skill-related analysis (keyword density, missing skills, etc.)
        self._analyze_skills(feedback, analysis_results, industry_name)

        # 4) Provide writing style feedback (passive voice, etc.)
        self._add_style_feedback(feedback, analysis_results)

        # 5) Add general resume quality feedback (word count, skill balance)
        self._add_general_feedback(feedback, analysis_results)

        # 6) Education-based suggestions
        self._add_education_feedback(feedback, analysis_results)

        # 7) If there's a job match analysis, provide missing requirement feedback
        self._add_job_match_feedback(feedback, analysis_results.get('job_match', {}))

        # 8) Identify weak verbs and recommend replacements
        self._add_weak_verb_feedback(feedback, analysis_results)

        # 9) Dynamically generate achievements for each job/position (LLM-based if key available)
        self._add_achievement_feedback(feedback, analysis_results, industry_name)

        # 10) Generate an LLM-based block of feedback for the given industry
        #     (combining strengths, improvements, recommendations)
        llm_feedback = self._generate_dynamic_feedback_for_industry(
            analysis_results,
            industry_name,
            job_description or ""
        )
        feedback["strengths"].extend(llm_feedback.get("strengths", []))
        feedback["improvements"].extend(llm_feedback.get("improvements", []))
        feedback["recommendations"].extend(llm_feedback.get("recommendations", []))

        # 11) Suggest stronger action verbs (hybrid approach: seed verbs + LLM synonyms)
        self._add_verb_suggestions(feedback, industry_name, job_description or "")

        # If no feedback was generated at all, insert a fallback note
        if not (feedback["strengths"] or feedback["improvements"] or feedback["recommendations"]):
            feedback["improvements"].append(
                "No significant feedback generated – check resume content."
            )

        # 12) Deduplicate and limit items in each feedback category, then return
        return self._finalize_feedback(feedback)

