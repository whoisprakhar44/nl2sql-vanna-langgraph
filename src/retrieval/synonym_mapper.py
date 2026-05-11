"""Business synonym normalisation for keyword retrieval."""

from __future__ import annotations

import re


class SynonymMapper:
    """
    Expand user language into canonical business terms.

    The mapper keeps the original words and adds canonical terms. This improves
    recall without hiding the user's phrasing from later retrieval stages.
    """

    def __init__(self, synonym_groups: dict[str, list[str]] | None = None) -> None:
        self._alias_to_canonical: dict[str, str] = {}
        self._canonical_to_aliases: dict[str, set[str]] = {}

        for canonical, aliases in (synonym_groups or {}).items():
            self.add_group(canonical, aliases)

    def add_group(self, canonical: str, aliases: list[str]) -> None:
        """Register aliases for a canonical term."""
        canonical_norm = self._normalise_phrase(canonical)
        if not canonical_norm:
            return

        self._canonical_to_aliases.setdefault(canonical_norm, set()).add(canonical_norm)
        self._alias_to_canonical[canonical_norm] = canonical_norm

        for alias in aliases:
            alias_norm = self._normalise_phrase(alias)
            if not alias_norm:
                continue
            self._canonical_to_aliases[canonical_norm].add(alias_norm)
            self._alias_to_canonical[alias_norm] = canonical_norm

    def expand_question(self, question: str) -> str:
        """
        Append canonical terms whose aliases appear as words or phrases.

        Example: "sales by region" becomes text containing both "sales" and
        "revenue" if "sales" is configured as an alias for "revenue".
        """
        question_norm = self._normalise_phrase(question)
        additions: list[str] = []

        for alias, canonical in self._alias_to_canonical.items():
            if self._contains_phrase(question_norm, alias) and canonical not in additions:
                additions.append(canonical)

        if not additions:
            return question
        return f"{question} {' '.join(additions)}"

    def expand_keywords(self, keywords: list[str]) -> list[str]:
        """Return original keywords plus canonical synonym tokens."""
        expanded: list[str] = []
        seen: set[str] = set()

        keyword_text = self._normalise_phrase(" ".join(keywords))

        for keyword in keywords:
            for token in self._phrase_tokens(keyword):
                if token not in seen:
                    seen.add(token)
                    expanded.append(token)

            canonical = self._alias_to_canonical.get(self._normalise_phrase(keyword))
            if canonical:
                for token in self._phrase_tokens(canonical):
                    if token not in seen:
                        seen.add(token)
                        expanded.append(token)

        for alias, canonical in self._alias_to_canonical.items():
            if self._contains_phrase(keyword_text, alias):
                for token in self._phrase_tokens(canonical):
                    if token not in seen:
                        seen.add(token)
                        expanded.append(token)

        return expanded

    def canonical_for(self, phrase: str) -> str | None:
        """Return the canonical term for a phrase, if configured."""
        return self._alias_to_canonical.get(self._normalise_phrase(phrase))

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))

    @staticmethod
    def _normalise_phrase(value: str) -> str:
        value = value.lower().replace("_", " ")
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _phrase_tokens(value: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", value.lower()) if token]
