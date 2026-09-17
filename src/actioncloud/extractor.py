from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .llm import LLM, get_llm
from .schema import Experience

log = logging.getLogger(__name__)


SYSTEM_EXTRACTION_PROMPT = """You are a technical knowledge extraction engine for an AI agent system.
Given a raw episode trajectory, extract:
1. A generalized reusable step-by-step workflow (JSON object with 'steps', 'prerequisites', and 'pitfalls').
2. Knowledge triples (list of JSON objects with 'subject', 'predicate', and 'object').

Output ONLY valid JSON matching this schema:
{
  "workflow": {
    "title": "Short title",
    "prerequisites": ["..."],
    "steps": ["Step 1", "Step 2"],
    "pitfalls": ["..."]
  },
  "knowledge_triples": [
    {"subject": "Docker", "predicate": "uses_port", "object": "5433"}
  ]
}
"""


class ExperienceExtractor:
    """
    Extracts structured procedural workflows and knowledge triples from experiences.
    """

    def __init__(self, llm: Optional[LLM] = None) -> None:
        self.llm = llm or get_llm()

    def extract(self, exp: Experience) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract workflow dict and list of triple dicts from an Experience.
        """
        user_prompt = (
            f"Task: {exp.task}\n"
            f"Problem: {exp.problem or 'None'}\n"
            f"Action: {exp.action}\n"
            f"Solution: {exp.solution or 'None'}\n"
            f"Result: {exp.result}\n"
            f"Technologies: {', '.join(exp.technologies)}\n"
            f"Tools Used: {', '.join(exp.tools_used)}\n"
        )

        try:
            resp = self.llm.complete(user_prompt, system=SYSTEM_EXTRACTION_PROMPT)
            # Try parsing JSON response
            text = resp.text.strip()
            if text.startswith("```json"):
                text = text.split("```json", 1)[1].rsplit("```", 1)[0].strip()
            elif text.startswith("```"):
                text = text.split("```", 1)[1].rsplit("```", 1)[0].strip()

            parsed = json.loads(text)
            workflow = parsed.get("workflow")
            triples = parsed.get("knowledge_triples", [])
            return workflow, triples
        except Exception as e:
            log.warning("Extraction LLM call failed or failed to parse JSON (%s); using heuristic fallback", e)
            # Fallback heuristic workflow generation
            fallback_workflow = {
                "title": f"Workflow for: {exp.task[:50]}",
                "prerequisites": exp.technologies,
                "steps": [exp.action],
                "pitfalls": [exp.problem] if exp.problem else [],
            }
            fallback_triples = [
                {"subject": tech, "predicate": "used_in_task", "object": exp.task[:40]}
                for tech in exp.technologies
            ]
            return fallback_workflow, fallback_triples
