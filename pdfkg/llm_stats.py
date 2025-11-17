"""Simple in-memory counters for LLM call usage."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, Tuple

_CALL_COUNTS: Counter[Tuple[str, str, str]] = Counter()


def extract_token_usage(usage: Any) -> Tuple[int, int, int]:
    """
    Extract (input, output, total) token counts from a provider-dependent usage object.
    Returns zeros if not available.
    """
    if not usage:
        return 0, 0, 0

    # Mistral returns usage as dict-like with "prompt_tokens" and "completion_tokens"
    if isinstance(usage, dict):
        tokens_in = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or (tokens_in + tokens_out))
        return tokens_in, tokens_out, total

    # Gemini response.usage_metadata has explicit counts
    if hasattr(usage, "prompt_token_count") or hasattr(usage, "candidates_token_count"):
        tokens_in = int(getattr(usage, "prompt_token_count", 0) or 0)
        tokens_out = int(getattr(usage, "candidates_token_count", 0) or 0)
        total = int(getattr(usage, "total_token_count", tokens_in + tokens_out))
        return tokens_in, tokens_out, total

    return 0, 0, 0


def record_call(
    provider: str,
    phase: str,
    label: str,
    *,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    total_tokens: int | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Record a single LLM call.
    Extra fields are accepted for compatibility with upstream callers but currently only count occurrences.
    """
    key = (provider, phase, label)
    _CALL_COUNTS[key] += 1


def summary() -> Dict[str, Dict[str, Dict[str, int]]]:
    """Return nested dict provider -> phase -> label -> count."""
    report: Dict[str, Dict[str, Dict[str, int]]] = {}
    for (provider, phase, label), count in _CALL_COUNTS.items():
        report.setdefault(provider, {}).setdefault(phase, {})[label] = count
    return report


def summary_lines() -> Iterable[str]:
    """Return a list of formatted strings summarizing counters."""
    if not _CALL_COUNTS:
        return ["No LLM calls recorded yet."]

    lines = ["LLM Call Summary:"]
    for provider, phase_map in summary().items():
        lines.append(f"- Provider: {provider}")
        for phase, label_map in phase_map.items():
            lines.append(f"  • Phase: {phase}")
            for label, count in sorted(label_map.items()):
                lines.append(f"    - {label}: {count}")
    return lines


def reset() -> None:
    """Reset counters to zero."""
    _CALL_COUNTS.clear()


def log_extraction_io(id_short: str, prompt: str, output: str, metadata: Dict[str, Any] | None = None) -> None:
    """
    Placeholder hook for logging extraction I/O. Currently no-op to keep compatibility
    with batch extractor; can be extended to persist prompts/outputs if needed.
    """
    return None
