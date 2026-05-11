"""Keyword extraction for the retrieval pipeline."""

from __future__ import annotations

import re

from my_agent.core.retrieval.synonym_mapper import SynonymMapper


class KeywordExtractor:
    """Extract stable lookup tokens from a natural-language question."""

    STOP_WORDS = {
        "show",
        "me",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "by",
        "for",
        "to",
        "from",
        "with",
        "is",
        "are",
        "was",
        "were",
        "what",
        "how",
        "many",
        "much",
        "do",
        "does",
        "did",
        "can",
        "could",
        "would",
        "should",
        "will",
        "i",
        "my",
        "our",
        "their",
        "vs",
        "versus",
        "compared",
        "last",
        "this",
        "all",
        "each",
        "every",
        "give",
        "get",
        "find",
        "list",
        "display",
        "tell",
        "please",
        "want",
        "need",
        "see",
        "look",
        "at",
        "up",
        "on",
        "it",
        "its",
    }

    def __init__(self, synonym_mapper: SynonymMapper | None = None) -> None:
        self._synonym_mapper = synonym_mapper or SynonymMapper()

    def extract(self, question: str) -> list[str]:
        """Tokenise a question, remove stop words, and expand synonyms."""
        expanded_question = self._synonym_mapper.expand_question(question)
        tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", expanded_question.lower())
        keywords = [
            token
            for token in tokens
            if token not in self.STOP_WORDS and len(token) > 1
        ]
        return self._synonym_mapper.expand_keywords(keywords)
