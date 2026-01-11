"""NLP-based event class classifier.

Infers event classes from natural language descriptions using keyword matching
and simple rule-based heuristics. Can be enhanced with ML models later.
"""

import re
from typing import Dict, List, Optional, Tuple

from .registry import EventClassRegistry, get_registry


class ClassificationResult:
    """Result of event class classification."""

    def __init__(
        self,
        inferred_class: str,
        confidence: float,
        alternatives: Optional[List[Tuple[str, float]]] = None,
        matched_keywords: Optional[List[str]] = None,
    ):
        """Initialize classification result.

        Args:
            inferred_class: The inferred event class path
            confidence: Confidence score (0-1)
            alternatives: List of (class_path, confidence) tuples for alternatives
            matched_keywords: Keywords that were matched
        """
        self.inferred_class = inferred_class
        self.confidence = confidence
        self.alternatives = alternatives or []
        self.matched_keywords = matched_keywords or []

    def to_dict(self) -> Dict:
        """Convert to dictionary format.

        Returns:
            Dictionary representation
        """
        return {
            "inferred_class": self.inferred_class,
            "confidence": self.confidence,
            "alternatives": [
                {"class": alt_class, "confidence": alt_conf}
                for alt_class, alt_conf in self.alternatives
            ],
            "matched_keywords": self.matched_keywords,
        }


class EventClassifier:
    """Classifies natural language input into event classes."""

    def __init__(self, registry: Optional[EventClassRegistry] = None):
        """Initialize the classifier.

        Args:
            registry: Event class registry to use. If None, uses global registry.
        """
        self.registry = registry or get_registry()

        # Common stopwords to filter out
        self.stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "will", "with", "after", "during", "before", "near"
        }

    def classify(
        self,
        natural_language: str,
        top_k: int = 3
    ) -> Optional[ClassificationResult]:
        """Classify natural language input into an event class.

        Args:
            natural_language: Natural language event description
            top_k: Number of top alternatives to include

        Returns:
            ClassificationResult if classification successful, None otherwise
        """
        # Extract keywords from input
        keywords = self._extract_keywords(natural_language)

        if not keywords:
            return None

        # Search for matching classes
        match_scores = self.registry.search_by_keywords(keywords)

        if not match_scores:
            return None

        # Convert raw match counts to confidence scores
        scored_classes = self._compute_confidence_scores(match_scores, keywords)

        if not scored_classes:
            return None

        # Get top result and alternatives
        top_class, top_confidence = scored_classes[0]
        alternatives = scored_classes[1:top_k] if len(scored_classes) > 1 else []

        # Find which keywords actually matched
        matched_keywords = self._find_matched_keywords(top_class, keywords)

        return ClassificationResult(
            inferred_class=top_class,
            confidence=top_confidence,
            alternatives=alternatives,
            matched_keywords=matched_keywords,
        )

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from natural language text.

        Args:
            text: Input text

        Returns:
            List of extracted keywords
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters except spaces and hyphens
        text = re.sub(r'[^a-z0-9\s-]', ' ', text)

        # Split into words
        words = text.split()

        # Remove stopwords and short words
        keywords = [
            word for word in words
            if word not in self.stopwords and len(word) > 2
        ]

        # Also extract common multi-word phrases
        multi_word_keywords = self._extract_multi_word_phrases(text)
        keywords.extend(multi_word_keywords)

        return keywords

    def _extract_multi_word_phrases(self, text: str) -> List[str]:
        """Extract common multi-word phrases from text.

        Args:
            text: Input text

        Returns:
            List of multi-word phrases
        """
        # Define common multi-word patterns for hazards
        common_phrases = [
            "storm surge",
            "flash flood",
            "wild fire",
            "wildland urban interface",
            "crown fire",
            "tornado damage",
            "hurricane flooding",
            "river flood",
            "coastal flood",
            "urban flood",
            "burn scar",
            "wind damage",
            "tropical cyclone",
            "severe thunderstorm",
        ]

        found_phrases = []
        for phrase in common_phrases:
            if phrase in text:
                found_phrases.append(phrase)

        return found_phrases

    def _compute_confidence_scores(
        self,
        match_scores: Dict[str, int],
        keywords: List[str]
    ) -> List[Tuple[str, float]]:
        """Compute confidence scores from raw match counts.

        Args:
            match_scores: Dictionary of class_path -> match_count
            keywords: Original keywords

        Returns:
            List of (class_path, confidence) tuples, sorted by confidence
        """
        if not match_scores:
            return []

        max_matches = max(match_scores.values())
        num_keywords = len(keywords)

        scored_classes = []
        for class_path, match_count in match_scores.items():
            # Base confidence from proportion of matches
            base_confidence = match_count / max(max_matches, num_keywords)

            # Boost confidence for more specific classes (longer paths)
            depth_bonus = class_path.count(".") * 0.02
            confidence = min(base_confidence + depth_bonus, 0.99)

            # Reduce confidence if only partial match
            if match_count < num_keywords / 2:
                confidence *= 0.7

            scored_classes.append((class_path, confidence))

        # Sort by confidence (descending)
        scored_classes.sort(key=lambda x: x[1], reverse=True)

        # Normalize confidences so they sum to <= 1.0
        total_confidence = sum(conf for _, conf in scored_classes)
        if total_confidence > 1.0:
            scored_classes = [
                (class_path, conf / total_confidence)
                for class_path, conf in scored_classes
            ]

        return scored_classes

    def _find_matched_keywords(self, class_path: str, keywords: List[str]) -> List[str]:
        """Find which keywords matched for a given class.

        Args:
            class_path: Event class path
            keywords: List of keywords

        Returns:
            List of matched keywords
        """
        event_class = self.registry.get_class(class_path)
        if not event_class:
            return []

        class_keywords_lower = {kw.lower() for kw in event_class.keywords}
        matched = []

        for keyword in keywords:
            if keyword.lower() in class_keywords_lower:
                matched.append(keyword)

        return matched


# Global classifier instance
_classifier: Optional[EventClassifier] = None


def get_classifier() -> EventClassifier:
    """Get the global event classifier instance.

    Returns:
        EventClassifier singleton
    """
    global _classifier
    if _classifier is None:
        _classifier = EventClassifier()
    return _classifier


def classify_text(text: str) -> Optional[ClassificationResult]:
    """Classify natural language text into an event class.

    Args:
        text: Natural language event description

    Returns:
        ClassificationResult if successful, None otherwise
    """
    return get_classifier().classify(text)


if __name__ == "__main__":
    import sys

    # Demo usage
    test_inputs = [
        "flooding in coastal areas after hurricane",
        "forest fire with severe burn scars",
        "tornado damage assessment",
        "storm surge from tropical cyclone",
        "river overflow causing flood",
        "wildfire in grasslands",
    ]

    if len(sys.argv) > 1:
        # Use command line argument
        test_inputs = [" ".join(sys.argv[1:])]

    classifier = get_classifier()

    for text in test_inputs:
        print(f"\nInput: '{text}'")
        result = classifier.classify(text)

        if result:
            print(f"  Inferred class: {result.inferred_class}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Matched keywords: {result.matched_keywords}")

            if result.alternatives:
                print("  Alternatives:")
                for alt_class, alt_conf in result.alternatives:
                    print(f"    - {alt_class}: {alt_conf:.2f}")
        else:
            print("  No classification found")
