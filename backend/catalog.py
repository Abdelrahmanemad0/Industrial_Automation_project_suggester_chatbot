"""Loads the project + component catalog and normalizes free-text hardware
input from users into canonical component ids.

Kept dependency-free (stdlib only) so it works even when the ML stack
(torch/transformers/peft) isn't installed - this module backs the
always-available fallback path.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Component:
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)


class Catalog:
    def __init__(self, components_path: Path = None, projects_path: Path = None):
        components_path = components_path or DATA_DIR / "components.json"
        projects_path = projects_path or DATA_DIR / "projects.json"

        raw_components = json.loads(components_path.read_text())
        self.components: Dict[str, Component] = {}
        # lookup_text -> component id, for exact/alias matching
        self._alias_index: Dict[str, str] = {}
        for group in raw_components.values():
            for c in group:
                comp = Component(id=c["id"], name=c["name"], aliases=c.get("aliases", []))
                self.components[comp.id] = comp
                self._alias_index[comp.id.replace("_", " ")] = comp.id
                self._alias_index[comp.name.lower()] = comp.id
                for alias in comp.aliases:
                    self._alias_index[alias.lower()] = comp.id

        self.projects: List[dict] = json.loads(projects_path.read_text())

    def component_name(self, comp_id: str) -> str:
        c = self.components.get(comp_id)
        return c.name if c else comp_id

    def normalize_component(self, text: str) -> str | None:
        """Best-effort match of free-text hardware mention to a canonical id.

        Returns None if nothing reasonably close is found (rather than
        guessing wrong - a wrong "human proof" match is worse than none).
        """
        text = text.strip().lower()
        if not text:
            return None
        if text in self._alias_index:
            return self._alias_index[text]
        # substring containment either direction
        for key, comp_id in self._alias_index.items():
            if key in text or text in key:
                return comp_id
        # fuzzy fallback for typos
        close = difflib.get_close_matches(text, self._alias_index.keys(), n=1, cutoff=0.75)
        if close:
            return self._alias_index[close[0]]
        return None

    def find_mentioned_component_ids(self, text: str) -> set[str]:
        """Scans free text for any known component name/alias, using word
        boundaries so short aliases (e.g. "uno") don't false-positive inside
        unrelated words (e.g. "unofficial"). Used to catch LLM hallucinations
        of components that weren't actually offered to it.
        """
        text_lower = (text or "").lower()
        found: set[str] = set()
        # longest keys first so multi-word aliases match before short substrings do
        for key in sorted(self._alias_index.keys(), key=len, reverse=True):
            if len(key) < 3:
                continue
            if re.search(r"(?<![a-z0-9])" + re.escape(key) + r"(?![a-z0-9])", text_lower):
                found.add(self._alias_index[key])
        return found


    def normalize_many(self, texts: List[str]) -> List[str]:
        out = []
        for t in texts:
            # allow comma/newline separated blobs too
            for piece in str(t).replace("\n", ",").split(","):
                cid = self.normalize_component(piece)
                if cid and cid not in out:
                    out.append(cid)
        return out


_catalog_singleton: Catalog | None = None


def get_catalog() -> Catalog:
    global _catalog_singleton
    if _catalog_singleton is None:
        _catalog_singleton = Catalog()
    return _catalog_singleton
