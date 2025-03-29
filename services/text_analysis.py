# services/text_analysis.py
import logging
import re
from typing import Dict, List, Set, Optional
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util
from spacy.lang.en.stop_words import STOP_WORDS
from .shared_utilities import TECH_TERMS, TECH_SYNONYMS, INDUSTRY_SYNONYMS, INDUSTRY_CONFIGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TextAnalyzer:
    """
    Enhanced text analysis with:
      - Embedding-based similarity scoring
      - Industry-aware keyword/entity extraction
    """
    def __init__(self):
        # Load the transformer-based spaCy model (excluding tagger and lemmatizer for speed)
        self.nlp = spacy.load('en_core_web_lg', exclude=["tagger", "lemmatizer"])
        self._add_custom_entities()  # Add patterns for TECH and INDUSTRY terms
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vectorizer = CountVectorizer(stop_words='english')
        self.tech_terms = TECH_TERMS
        self._tech_synonyms = TECH_SYNONYMS
        self.industry_synonyms = INDUSTRY_SYNONYMS

    def _add_custom_entities(self) -> None:
        """Add custom entity patterns for technology and industry terms."""
        ruler = self.nlp.add_pipe("entity_ruler", after="ner")
        patterns = (
            [{"label": "TECH", "pattern": term} for term in TECH_TERMS] +
            [{"label": "INDUSTRY", "pattern": term} for term in INDUSTRY_SYNONYMS]
        )
        ruler.add_patterns(patterns)

    def analyze_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0

        text1_norm = self._normalize_text(text1)
        text2_norm = self._normalize_text(text2)

        # Calculate embedding similarity.
        emb1 = self.model.encode(text1_norm, convert_to_tensor=True)
        emb2 = self.model.encode(text2_norm, convert_to_tensor=True)
        embedding_sim = util.pytorch_cos_sim(emb1, emb2).item()

        # Calculate tech-term Jaccard similarity.
        terms1 = self.extract_tech_terms(text1_norm)
        terms2 = self.extract_tech_terms(text2_norm)
        intersection = terms1.intersection(terms2)
        union = terms1.union(terms2)
        tech_sim = len(intersection) / len(union) if union else 0.0

        # Domain boost: if both texts have healthcare keywords, add extra weight.
        healthcare_keywords = {"physician", "clinical", "hospital", "emr", "ehr", "patient"}
        if healthcare_keywords.intersection(set(text1_norm.split())) and healthcare_keywords.intersection(set(text2_norm.split())):
            domain_boost = 0.1  # add a 10% boost
        else:
            domain_boost = 0.0

        final_score = (embedding_sim * 0.6) + (tech_sim * 0.4) + domain_boost
        return min(final_score, 1.0)


    def _compare_sections(self, section1: str, section2: str, industry: str = None) -> float:
        """
        Compare two sections using embedding similarity and optionally boost
        the score based on industry-specific keywords.
        """
        base_score = self.analyze_similarity(section1, section2)
        if industry:
            boost = self._industry_boost(section1, section2, industry)
            return min(base_score + boost, 1.0)
        return base_score

    def _industry_boost(self, text1: str, text2: str, industry: str) -> float:
        """
        Boosts the similarity score based on the presence of industry-specific keywords.
        """
        industry_key = industry.lower()
        if industry_key not in INDUSTRY_CONFIGS:
            return 0.0
        keywords = INDUSTRY_CONFIGS[industry_key].get('keywords', [])
        industry_terms = set(keywords + [k for k, v in self.industry_synonyms.items() if v.lower() == industry_key])
        text1_tokens = set(self._normalize_text(text1).split())
        text2_tokens = set(self._normalize_text(text2).split())
        matches = len(text1_tokens & industry_terms) + len(text2_tokens & industry_terms)
        max_len = max(len(text1_tokens), len(text2_tokens))
        return 0.3 * (matches / max_len) if max_len > 0 else 0.0

    def _normalize_text(self, text: str) -> str:
        """
        Normalizes text: lowercases, converts specific variants (like "node.js") and
        replaces synonyms from TECH_SYNONYMS.
        """
        text = text.lower()
        text = re.sub(r'\bnode\.js\b', 'nodejs', text)
        for variant, normalized in self._tech_synonyms.items():
            text = re.sub(rf'\b{re.escape(variant)}\b', normalized.lower(), text)
        text = re.sub(r'[^\w\s-]', ' ', text)
        return ' '.join(text.split())

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extracts keywords from text using a combination of TF-IDF and noun chunk extraction.
        Also includes any detected tech terms.
        """
        if not text:
            return []
        try:
            tfidf_keywords = self._extract_keywords_tfidf(text)
            phrase_keywords = self._extract_keywords_phrases(text)
            tech_terms = list(self.extract_tech_terms(text))
            return list(set(tfidf_keywords + phrase_keywords + tech_terms))
        except Exception as e:
            logger.error(f"Keyword extraction failed: {str(e)}")
            return []

    def _extract_keywords_tfidf(self, text: str) -> List[str]:
        if not text.strip():
            return []
        tfidf_matrix = self.vectorizer.fit_transform([text])
        sums = np.array(tfidf_matrix.sum(axis=0)).ravel()
        sorted_indices = np.argsort(sums)
        top_indices = sorted_indices[-20:][::-1]
        features = self.vectorizer.get_feature_names_out()
        return [features[idx] for idx in top_indices]

    def _extract_keywords_phrases(self, text: str) -> List[str]:
        """
        Extracts noun phrases from the text using spaCy's noun chunker.
        """
        doc = self.nlp(text)
        phrases = [chunk.text.lower() for chunk in doc.noun_chunks if chunk.text.lower() not in STOP_WORDS]
        return list(set(phrases))

    def extract_entities(self, text: str) -> List[str]:
        """
        Extracts entities from the text, including any custom tech or industry terms.
        """
        doc = self.nlp(text)
        entities = {ent.text.lower() for ent in doc.ents}
        entities.update(self.extract_tech_terms(text))
        for ent in doc.ents:
            if ent.label_ == "INDUSTRY":
                term = ent.text.lower()
                normalized_term = self.industry_synonyms.get(term, term)
                entities.add(normalized_term)
        return list(entities)

    def extract_tech_terms(self, text: str) -> Set[str]:
        """
        Extracts technical terms from the text using a regex search and noun chunk analysis.
        """
        normalized = self._normalize_text(text)
        terms = set()
        # Regex match for each tech term.
        for term in self.tech_terms:
            if re.search(rf'\b{re.escape(term)}\b', normalized):
                terms.add(term)
        # Check noun chunks for phrases containing known tech terms.
        doc = self.nlp(normalized)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if any(t in self.tech_terms for t in chunk_text.split()):
                terms.add(chunk_text)
        return terms
