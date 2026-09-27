from __future__ import annotations

import difflib
import uuid
from typing import Any, Dict, List, Tuple
from .policy import MemorySelectionPolicy


def count_tokens(text: str) -> int:
    """
    Approximate token counter (replaceable tokenizer abstraction).
    Approximates tokens as max(1, len(text) // 4).
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _relevance(c: Dict[str, Any]) -> float:
    r = c.get("relevance")
    return 0.0 if r is None else float(r)


def compute_text_similarity(text_a: str, text_b: str) -> float:
    """
    Lightweight string sequence matcher similarity ratio [0.0, 1.0].
    Used for redundancy filtering between candidate memory workflows.
    """
    if not text_a or not text_b:
        return 0.0
    return difflib.SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()


def filter_redundant_memories(
    candidates: List[Dict[str, Any]], redundancy_threshold: float = 0.85
) -> List[Dict[str, Any]]:
    """
    Filter out candidate memories that share high procedural similarity or identical task keys
    with an already selected candidate.
    """
    selected: List[Dict[str, Any]] = []

    for item in candidates:
        task_text = item.get("task", "")
        task_key = item.get("task_key")
        solution_text = item.get("solution") or item.get("result") or ""
        content = f"{task_text} {solution_text}"

        is_redundant = False
        for s in selected:
            s_task_key = s.get("task_key")
            s_content = f"{s.get('task', '')} {s.get('solution') or s.get('result') or ''}"

            # Check task key match
            if task_key and s_task_key and task_key == s_task_key:
                is_redundant = True
                break

            # Check textual workflow similarity
            sim = compute_text_similarity(content, s_content)
            if sim >= redundancy_threshold:
                is_redundant = True
                break

        if not is_redundant:
            selected.append(item)

    return selected


class CompactContextBuilder:
    """
    Deterministic compact context builder. Transforms selected experiences
    and procedural workflows into a token-budgeted prompt context.
    """

    @staticmethod
    def format_single_memory(idx: int, exp: Dict[str, Any]) -> str:
        """
        Compact procedural rendering. When an extracted workflow exists its
        steps ARE the reusable content, so the raw solution text is not
        repeated (it used to be, doubling the tokens for the same steps).
        """
        lines = []
        success = exp.get("success", True)
        marker = "WORKED" if success else "FAILED — avoid this approach"
        tier = str(exp.get("tier", "shared")).upper()
        lines.append(f"### {idx}. [{marker}] [{tier}] {exp.get('task', '')}")

        wf = exp.get("workflow")
        steps = wf.get("steps", []) if isinstance(wf, dict) else []
        if steps:
            if wf.get("prerequisites"):
                lines.append(f"- Prerequisites: {', '.join(wf['prerequisites'])}")
            lines.append("- Procedure:")
            for step_idx, st in enumerate(steps, 1):
                lines.append(f"  {step_idx}. {st}")
            if wf.get("pitfalls"):
                lines.append(f"- Pitfalls: {'; '.join(wf['pitfalls'])}")
        else:
            if exp.get("problem"):
                lines.append(f"- Problem: {exp['problem']}")
            if exp.get("solution"):
                lines.append(f"- Solution: {exp['solution']}")
            elif exp.get("result"):
                lines.append(f"- Outcome: {exp['result']}")
        return "\n".join(lines)

    @classmethod
    def build_context(
        cls, candidates: List[Dict[str, Any]], policy: MemorySelectionPolicy
    ) -> Tuple[str, List[uuid.UUID], int, int]:
        """
        Builds a compact memory context.
        Returns: (context_string, injected_experience_ids, candidate_count, final_count)
        """
        if not candidates:
            return "", [], 0, 0

        # 1. Relevance threshold. A missing score is treated as 0.0 (unknown =
        #    not relevant). If nothing clears the bar we inject NOTHING: an
        #    irrelevant memory costs prompt tokens and can mislead the agent,
        #    which is exactly what the threshold exists to prevent.
        relevant = [
            c for c in candidates
            if _relevance(c) >= policy.similarity_threshold
            and float(c.get("confidence", 1.0) if c.get("confidence") is not None else 1.0)
            >= policy.min_confidence
        ]
        if not relevant:
            return "", [], len(candidates), 0

        # 1b. Rank by relevance with a small trust bonus, so between two
        #     similarly relevant memories the better-governed one wins.
        relevant.sort(
            key=lambda c: _relevance(c) + policy.trust_weight * float(c.get("confidence") or 0.0),
            reverse=True,
        )

        # 2. Redundancy Filtering
        non_redundant = filter_redundant_memories(relevant, redundancy_threshold=policy.redundancy_threshold)

        # 3. Token Budget Allocator & Context Assembly
        selected_ids: List[uuid.UUID] = []
        formatted_blocks: List[str] = []
        current_tokens = count_tokens("## Relevant prior experience from other agents\n\n")

        for c in non_redundant:
            if len(selected_ids) >= policy.max_context_memories:
                break

            block = cls.format_single_memory(len(selected_ids) + 1, c)
            block_tokens = count_tokens(block)

            if current_tokens + block_tokens > policy.context_token_budget and selected_ids:
                break  # stop if adding block exceeds budget (unless first memory)

            formatted_blocks.append(block)
            current_tokens += block_tokens
            exp_id = c["id"] if isinstance(c["id"], uuid.UUID) else uuid.UUID(str(c["id"]))
            selected_ids.append(exp_id)

        if not formatted_blocks:
            return "", [], len(candidates), 0

        header = "## Relevant prior experience from other agents\n\n"
        footer = "\nUse the above if it applies. If it does not, solve the task directly and ignore it."
        context_str = header + "\n\n".join(formatted_blocks) + footer

        return context_str, selected_ids, len(candidates), len(selected_ids)
