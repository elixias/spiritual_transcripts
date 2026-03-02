from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, RootModel

from .config import StageLLMConfig
from .model_manager import ModelManager
from .segmentation import validate_modules_payload
from .transcript import format_transcript_lines, parse_transcript_text


logger = logging.getLogger(__name__)


OPENAI_PRICING_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.0},
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.0},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.4},
}


class SegmentWorkflowState(TypedDict, total=False):
    transcript_text: str
    editorial_rules: str
    user_idea_override: str | None
    context_analysis: dict[str, Any]
    story_ideas: list[dict[str, Any]]
    current_idea_index: int
    drafted_modules: list[dict[str, Any]]
    selection_failures: list[dict[str, Any]]
    final_json: str


class LectureContextOutput(BaseModel):
    lecture_context: str = Field(..., description="Overall lecture context inferred from transcript")
    lecture_arc: str = Field(..., description="Summary of the lecture's teaching progression")
    dominant_topics: list[str] = Field(default_factory=list)
    audience_context: str | None = None
    editorial_notes: str | None = None


class StoryOrchestratorOutput(BaseModel):
    main_point: str
    opening: str
    important_fact: str
    conclusion: str


class StoryIdeaOutput(BaseModel):
    title: str
    category: str
    subcategory: str | None = None
    context: str | None = None
    story_orchestrator: StoryOrchestratorOutput


class StoryIdeasOutput(BaseModel):
    ideas: list[StoryIdeaOutput] = Field(default_factory=list)


class ModuleSegmentOutput(BaseModel):
    start: float
    end: float
    text: str


class ModuleSelectionOutput(BaseModel):
    title: str
    category: str
    subcategory: str | None = None
    context: str | None = None
    normalized_terms: dict[str, Any] | None = None
    categorization_confidence: float | None = Field(
        default=None, description="Confidence in category/subcategory assignment on a 0..1 scale"
    )
    segments: list[ModuleSegmentOutput] = Field(default_factory=list)


class ModuleSelectionFlexibleOutput(RootModel[ModuleSelectionOutput | list[ModuleSelectionOutput]]):
    pass


def _normalize_model_name(model_name: str | None) -> str | None:
    if not model_name:
        return None
    return model_name.strip().lower()


def _lookup_openai_pricing(model_name: str | None) -> dict[str, float] | None:
    normalized = _normalize_model_name(model_name)
    if not normalized:
        return None
    for prefix, pricing in sorted(OPENAI_PRICING_PER_1M_TOKENS.items(), key=lambda item: len(item[0]), reverse=True):
        if normalized.startswith(prefix):
            return pricing
    return None


def _resolve_usage_provider(provider: str | None, configured_model: str | None, seen_models: list[str]) -> str | None:
    if provider:
        return provider

    candidates = [configured_model, *seen_models]
    for model_name in candidates:
        normalized = _normalize_model_name(model_name)
        if not normalized:
            continue
        if normalized.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            return "openai"
        if normalized.startswith("claude"):
            return "anthropic"
        if normalized.startswith("gemini"):
            return "google"
    return None


def _build_usage_summary(
    usage_callback: UsageMetadataCallbackHandler, *, provider: str | None, configured_model: str | None
) -> dict[str, Any]:
    resolved_provider = _resolve_usage_provider(
        provider,
        configured_model,
        list(usage_callback.usage_metadata.keys()),
    )
    models: dict[str, Any] = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_cached_input_tokens = 0
    total_estimated_cost_usd = 0.0
    estimated_cost_available = False

    for model_name, usage in usage_callback.usage_metadata.items():
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        total = int(usage.get("total_tokens", input_tokens + output_tokens))
        input_details = usage.get("input_token_details") or {}
        cached_input_tokens = int(input_details.get("cache_read", 0))

        model_summary: dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "cached_input_tokens": cached_input_tokens,
            "usage_metadata": usage,
        }

        pricing = _lookup_openai_pricing(model_name) if (resolved_provider or "").lower() == "openai" else None
        if pricing is not None:
            noncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
            estimated_cost = (
                (noncached_input_tokens / 1_000_000.0) * pricing["input"]
                + (cached_input_tokens / 1_000_000.0) * pricing["cached_input"]
                + (output_tokens / 1_000_000.0) * pricing["output"]
            )
            model_summary["pricing_usd_per_1m_tokens"] = pricing
            model_summary["estimated_cost_usd"] = estimated_cost
            total_estimated_cost_usd += estimated_cost
            estimated_cost_available = True
        else:
            model_summary["pricing_usd_per_1m_tokens"] = None
            model_summary["estimated_cost_usd"] = None

        models[model_name] = model_summary
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_tokens += total
        total_cached_input_tokens += cached_input_tokens

    summary: dict[str, Any] = {
        "provider": resolved_provider,
        "configured_model": configured_model,
        "models": models,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "cached_input_tokens": total_cached_input_tokens,
        "estimated_cost_usd": total_estimated_cost_usd if estimated_cost_available else None,
    }
    if (resolved_provider or "").lower() == "openai":
        summary["pricing_source"] = "https://platform.openai.com/docs/models/gpt-5.1/"
    return summary


def _dominant_topics_json(context_analysis: dict[str, Any]) -> str:
    topics = context_analysis.get("dominant_topics")
    if not isinstance(topics, list):
        topics = []
    clean_topics = [str(topic).strip() for topic in topics if str(topic).strip()]
    return json.dumps(clean_topics, ensure_ascii=False, indent=2)


def _should_retry_user_idea_split(raw_idea_draft: str, normalized_ideas: list[dict[str, Any]], dominant_topics: list[str]) -> bool:
    if len(normalized_ideas) != 1:
        return False
    if len(dominant_topics) < 3:
        return False

    draft = raw_idea_draft.strip()
    if "\n" in draft:
        nonempty_lines = [line.strip() for line in draft.splitlines() if line.strip()]
        if len(nonempty_lines) >= 2:
            return True

    separators = [";", "•", "-", "*", "|", ":"]
    if sum(draft.count(sep) for sep in separators) >= 3:
        return True

    return len(draft) > 180


def _load_editorial_rules(template_path: Path) -> str:
    template = template_path.read_text(encoding="utf-8")
    marker = "Now process the transcript:"
    if marker in template:
        return template.split(marker, 1)[0].strip()
    return template.strip()


def _pydantic_chain(template: str, model: Any, parser: PydanticOutputParser):
    prompt = PromptTemplate(
        template=template,
        input_variables=sorted(
            {
                field_name
                for field_name in PromptTemplate.from_template(template).input_variables
                if field_name != "format_instructions"
            }
        ),
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    return prompt | model | parser


def _pydantic_prompt(template: str, parser: PydanticOutputParser) -> PromptTemplate:
    return PromptTemplate(
        template=template,
        input_variables=sorted(
            {
                field_name
                for field_name in PromptTemplate.from_template(template).input_variables
                if field_name != "format_instructions"
            }
        ),
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


def run_segment_generation_workflow(
    *,
    transcript_text: str,
    prompt_file: Path,
    stage_cfg: StageLLMConfig,
    temperature: float,
    idea_override: str | None = None,
) -> str:
    editorial_rules = _load_editorial_rules(prompt_file)
    transcript_lines = parse_transcript_text(transcript_text)
    normalized_transcript_text = format_transcript_lines(transcript_lines)
    model = ModelManager(stage_cfg).get_chat_model(temperature=temperature)
    usage_callback = UsageMetadataCallbackHandler()
    invoke_config = {"callbacks": [usage_callback]}

    context_parser = PydanticOutputParser(pydantic_object=LectureContextOutput)
    context_chain = _pydantic_chain(
        """
You are a lecture analyst for advanced spiritual teachings.

Read the ENTIRE transcript and infer the overall lecture context and teaching arc.

Return STRICT JSON object with:
- lecture_context (string)
- lecture_arc (string)
- dominant_topics (array of strings)
- audience_context (string|null)
- editorial_notes (string|null)

Rules:
- Infer context only from transcript clues.
- Do not add teachings not present.
- Keep it concise but useful for downstream clip planning.

Output format instructions:
{format_instructions}

Transcript:
<<<TRANSCRIPT>>>
{transcript}
<<<END>>>
        """.strip(),
        model,
        context_parser,
    )

    ideas_parser = PydanticOutputParser(pydantic_object=StoryIdeasOutput)
    ideas_chain = _pydantic_chain(
        """
You are a story orchestrator for educational video modules.

Use the full transcript and the lecture context analysis to propose a small set of coherent sub-video ideas.
Each idea should describe:
- the main point
- how the video should begin
- the most important fact/instruction
- the conclusion

Return a STRICT JSON object with key `ideas`, where `ideas` is an array of items.
Each item must contain:
- title
- category
- subcategory
- context
- story_orchestrator (main_point, opening, important_fact, conclusion)

Constraints:
- Ideas must be grounded in transcript content only.
- Prefer 2-8 ideas depending on transcript breadth.
- Ideas should be conceptually distinct and suitable for 3-20 minute modules.
- Use the dominant topics list as the primary source of idea candidates.
- Cover the major distinct dominant topics instead of collapsing the whole lecture into one umbrella idea.
- Avoid merging clearly different dominant topics into one idea unless the transcript treats them as a single uninterrupted teaching unit.
- Use the closest taxonomy category from the editorial rules when applicable.

Output format instructions:
{format_instructions}

Editorial rules:
{editorial_rules}

Context analysis JSON:
{context_analysis_json}

Dominant topics:
{dominant_topics_json}

Transcript:
<<<TRANSCRIPT>>>
{transcript}
<<<END>>>
        """.strip(),
        model,
        ideas_parser,
    )

    user_ideas_parser = PydanticOutputParser(pydantic_object=StoryIdeasOutput)
    user_ideas_template = """
You are converting a user-provided idea draft into one or more internal structured story ideas.

Given:
 - a user idea draft string (may contain one idea or multiple ideas)
 - editorial rules / taxonomy
 - inferred lecture context
 - the full transcript

Produce a STRICT JSON object with key `ideas`, where `ideas` is an array containing one or more items.
Each item must contain:
- title
- category
- subcategory
- context
- story_orchestrator {{ main_point, opening, important_fact, conclusion }}

Requirements:
- Ground everything in transcript content.
- Preserve the user's intended topic/scope from the draft.
- If the draft contains multiple distinct module ideas, return multiple items.
- If the draft contains one idea, return exactly one item.
- Use the dominant topics list to decide whether the draft is actually covering multiple distinct ideas.
- If the draft is a heading list, summary list, or broad lecture outline that spans multiple dominant topics, split it into multiple ideas instead of collapsing it into one umbrella idea.
- Only keep it as one idea when the draft is clearly focused on one coherent topic.
- Choose the closest taxonomy category from editorial rules if possible.
- If taxonomy mapping is unclear, use category "user_provided".
- Make story_orchestrator fields specific and useful for segment selection (not duplicates).

Output format instructions:
{format_instructions}

User idea draft:
{user_idea}

Editorial rules:
{editorial_rules}

Lecture context analysis:
{context_analysis_json}

Dominant topics:
{dominant_topics_json}

Transcript:
<<<TRANSCRIPT>>>
{transcript}
<<<END>>>
    """.strip()
    user_ideas_prompt = _pydantic_prompt(user_ideas_template, user_ideas_parser)
    user_ideas_chain = user_ideas_prompt | model | user_ideas_parser
    split_user_ideas_parser = PydanticOutputParser(pydantic_object=StoryIdeasOutput)
    split_user_ideas_template = """
You are expanding a broad user-provided lecture summary into multiple internal module ideas.

The first normalization collapsed the draft into one idea, but the lecture context shows multiple dominant topics.
Re-split the draft into multiple ideas when the draft spans multiple dominant topics.

Return a STRICT JSON object with key `ideas`, where `ideas` is an array containing two or more items whenever the draft clearly covers multiple dominant topics.
Each item must contain:
- title
- category
- subcategory
- context
- story_orchestrator {{ main_point, opening, important_fact, conclusion }}

Requirements:
- Ground everything in transcript content.
- Preserve the user draft's scope, but split broad summaries into distinct ideas.
- Use dominant topics as the main splitting seeds.
- Do not collapse unrelated dominant topics into one umbrella idea.
- If two dominant topics are part of the same teaching unit, you may combine them.
- If the draft truly supports only one coherent idea, you may return one item, but only if the dominant topics clearly point to a single unit.

Output format instructions:
{format_instructions}

Original user idea draft:
{user_idea}

First collapsed idea:
{first_idea_json}

Editorial rules:
{editorial_rules}

Lecture context analysis:
{context_analysis_json}

Dominant topics:
{dominant_topics_json}

Transcript:
<<<TRANSCRIPT>>>
{transcript}
<<<END>>>
    """.strip()
    split_user_ideas_prompt = _pydantic_prompt(split_user_ideas_template, split_user_ideas_parser)
    split_user_ideas_chain = split_user_ideas_prompt | model | split_user_ideas_parser
    module_parser = PydanticOutputParser(pydantic_object=ModuleSelectionFlexibleOutput)
    select_segments_chain = _pydantic_chain(
        """
You are an editor assembling one video module from a raw timestamped transcript.

Your task:
- Given ONE module idea and the full transcript, select the exact transcript segments that support the idea.
- Use exact timestamps from transcript boundaries.
- Merge adjacent lines conceptually when useful.
- Build a final module JSON object in the required output shape.

Return a STRICT JSON object for one module (NOT an array).

Important:
- Use the idea's story orchestrator as guidance for selection only.
- Do not fabricate content or paraphrase beyond conservative cleanup.
- Do not overlap segments with previously selected time ranges.
- Prefer coherent module flow: opening -> explanation -> important fact -> conclusion (when present in transcript).
- Include `categorization_confidence` as a number between 0 and 1 representing confidence in category/subcategory.

Output format instructions:
{format_instructions}

Already selected time ranges (avoid overlap):
{used_ranges_json}

Lecture context analysis:
{context_analysis_json}

Module idea:
{idea_json}

Editorial rules:
{editorial_rules}

Transcript:
<<<TRANSCRIPT>>>
{transcript}
<<<END>>>
        """.strip(),
        model,
        module_parser,
    )

    retry_select_segments_chain = _pydantic_chain(
        """
You are repairing ONE invalid module selection produced from a raw timestamped transcript.

The previous output failed validation. Produce a corrected STRICT JSON object for one module (NOT an array).

Required fixes:
- Resolve the exact validation error.
- Keep the same intended idea and scope.
- Use exact transcript boundary timestamps.
- Keep segments chronologically ordered and non-overlapping.
- Avoid overlap with already selected time ranges.
- Include `categorization_confidence` as a number between 0 and 1.

Validation error:
{validation_error}

Previous invalid module attempt:
{previous_attempt_json}

Already selected time ranges (avoid overlap):
{used_ranges_json}

Lecture context analysis:
{context_analysis_json}

Module idea:
{idea_json}

Editorial rules:
{editorial_rules}

Transcript:
<<<TRANSCRIPT>>>
{transcript}
<<<END>>>

Output format instructions:
{format_instructions}
        """.strip(),
        model,
        module_parser,
    )

    def analyze_context(state: SegmentWorkflowState) -> SegmentWorkflowState:
        context_payload = context_chain.invoke({"transcript": state["transcript_text"]}, config=invoke_config)
        return {"context_analysis": context_payload.model_dump()}

    def _snap_value_to_boundary(value: Any, valid_points: set[float]) -> tuple[float, bool]:
        numeric = round(float(value), 2)
        if numeric in valid_points:
            return numeric, False
        nearest = min(valid_points, key=lambda point: abs(point - numeric))
        return nearest, True

    def _fallback_user_idea(raw_idea: str, *, reason: str | None = None) -> dict[str, Any]:
        context = "User-provided idea override from CLI."
        if reason:
            context = f"{context} Fallback used because normalization failed: {reason}"
        return {
            "title": raw_idea[:120],
            "category": "user_provided",
            "subcategory": None,
            "context": context,
            "story_orchestrator": {
                "main_point": raw_idea,
                "opening": f"Introduce the topic: {raw_idea}",
                "important_fact": f"Select the transcript passages that best support: {raw_idea}",
                "conclusion": f"Close with the key takeaway about: {raw_idea}",
            },
        }

    def _normalize_story_idea_dict(item_dict: dict[str, Any], raw_idea: str) -> dict[str, Any]:
        if not item_dict.get("title"):
            item_dict["title"] = raw_idea[:120]
        if not item_dict.get("category"):
            item_dict["category"] = "user_provided"

        # Ensure story_orchestrator exists and is usable for downstream selection prompts.
        so = item_dict.get("story_orchestrator")
        if not isinstance(so, dict):
            return _fallback_user_idea(raw_idea, reason="missing story_orchestrator")

        for key in ("main_point", "opening", "important_fact", "conclusion"):
            value = so.get(key)
            if not isinstance(value, str) or not value.strip():
                so[key] = raw_idea
        item_dict["story_orchestrator"] = so
        return item_dict

    def normalize_user_idea(state: SegmentWorkflowState) -> SegmentWorkflowState:
        raw_idea_draft = (state.get("user_idea_override") or "").strip()
        if not raw_idea_draft:
            raise ValueError("`--idea` was provided but is empty.")
        logger.info(
            "Normalizing user-provided idea draft into structured story idea(s)",
        )
        try:
            dominant_topics_json = _dominant_topics_json(state["context_analysis"])
            invoke_inputs = {
                "user_idea": raw_idea_draft,
                "editorial_rules": state["editorial_rules"],
                "context_analysis_json": json.dumps(state["context_analysis"], ensure_ascii=False, indent=2),
                "dominant_topics_json": dominant_topics_json,
                "transcript": state["transcript_text"],
            }
            try:
                rendered_prompt = user_ideas_prompt.format(**invoke_inputs)
                logger.info(
                    "Rendered user idea normalization prompt (%s chars)\n<<<USER_IDEA_NORMALIZATION_PROMPT>>>\n%s\n<<<END_USER_IDEA_NORMALIZATION_PROMPT>>>",
                    len(rendered_prompt),
                    rendered_prompt,
                )
            except Exception as render_exc:  # noqa: BLE001
                logger.warning("Failed to render user idea normalization prompt for logging: %s", render_exc)
            try:
                ideas_payload = user_ideas_chain.invoke(invoke_inputs, config=invoke_config)
            except Exception as first_exc:  # noqa: BLE001
                logger.warning(
                    "User idea draft normalization attempt 1 failed; retrying same normalization chain: %s",
                    first_exc,
                )
                ideas_payload = user_ideas_chain.invoke(invoke_inputs, config=invoke_config)
            normalized_ideas: list[dict[str, Any]] = []
            for idx, item in enumerate(ideas_payload.ideas, start=1):
                item_dict = _normalize_story_idea_dict(item.model_dump(), raw_idea_draft)
                if not item_dict.get("title"):
                    raise ValueError(f"User idea item {idx} missing title.")
                if not item_dict.get("category"):
                    raise ValueError(f"User idea item {idx} missing category.")
                normalized_ideas.append(item_dict)
            dominant_topics = json.loads(dominant_topics_json)
            if _should_retry_user_idea_split(raw_idea_draft, normalized_ideas, dominant_topics):
                logger.info(
                    "User idea normalization collapsed to one idea despite multiple dominant topics; retrying with explicit split instructions."
                )
                split_inputs = {
                    **invoke_inputs,
                    "first_idea_json": json.dumps(normalized_ideas[0], ensure_ascii=False, indent=2),
                }
                split_payload = split_user_ideas_chain.invoke(split_inputs, config=invoke_config)
                split_ideas: list[dict[str, Any]] = []
                for idx, item in enumerate(split_payload.ideas, start=1):
                    item_dict = _normalize_story_idea_dict(item.model_dump(), raw_idea_draft)
                    if not item_dict.get("title"):
                        raise ValueError(f"User idea split item {idx} missing title.")
                    if not item_dict.get("category"):
                        raise ValueError(f"User idea split item {idx} missing category.")
                    split_ideas.append(item_dict)
                if split_ideas:
                    normalized_ideas = split_ideas
            if not normalized_ideas:
                raise ValueError("User idea draft normalization returned zero ideas.")

            logger.info("User idea draft normalized into %s idea(s)", len(normalized_ideas))
            return {"story_ideas": normalized_ideas}
        except Exception as exc:  # noqa: BLE001
            logger.warning("User idea draft normalization failed; using fallback structure: %s", exc)
            return {"story_ideas": [_fallback_user_idea(raw_idea_draft, reason=str(exc))]}

    def generate_story_ideas(state: SegmentWorkflowState) -> SegmentWorkflowState:
        ideas_payload = ideas_chain.invoke(
            {
                "editorial_rules": state["editorial_rules"],
                "context_analysis_json": json.dumps(state["context_analysis"], ensure_ascii=False, indent=2),
                "dominant_topics_json": _dominant_topics_json(state["context_analysis"]),
                "transcript": state["transcript_text"],
            },
            config=invoke_config,
        )
        ideas: list[dict[str, Any]] = []
        for idx, item in enumerate(ideas_payload.ideas, start=1):
            item_dict = item.model_dump()
            if not item_dict.get("title"):
                raise ValueError(f"Idea {idx} missing title.")
            if not item_dict.get("category"):
                raise ValueError(f"Idea {idx} missing category.")
            ideas.append(item_dict)
        if not ideas:
            raise ValueError("Idea generation returned zero ideas.")
        return {"story_ideas": ideas}

    def init_selection_loop(_: SegmentWorkflowState) -> SegmentWorkflowState:
        return {"current_idea_index": 0, "drafted_modules": [], "selection_failures": []}

    def _used_ranges_json(drafted_modules: list[dict[str, Any]]) -> str:
        ranges: list[dict[str, float]] = []
        for module in drafted_modules:
            for seg in module.get("segments", []):
                try:
                    ranges.append({"start": float(seg["start"]), "end": float(seg["end"])})
                except Exception:  # noqa: BLE001
                    continue
        ranges.sort(key=lambda r: (r["start"], r["end"]))
        return json.dumps(ranges, ensure_ascii=False)

    def _used_ranges_list(drafted_modules: list[dict[str, Any]]) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        for module in drafted_modules:
            for seg in module.get("segments", []):
                try:
                    ranges.append((float(seg["start"]), float(seg["end"])))
                except Exception:  # noqa: BLE001
                    continue
        ranges.sort(key=lambda x: (x[0], x[1]))
        return ranges

    def _normalize_module_from_flexible(
        payload: ModuleSelectionFlexibleOutput, idea_index: int
    ) -> ModuleSelectionOutput:
        parsed = payload.root
        if isinstance(parsed, list):
            if len(parsed) == 0:
                raise ValueError(f"Idea {idea_index + 1} selection returned no module.")
            if len(parsed) != 1:
                raise ValueError(
                    f"Idea {idea_index + 1} selection node returned {len(parsed)} modules; expected exactly 1."
                )
            return parsed[0]
        return parsed

    def _validate_candidate_module(module_dict: dict[str, Any], drafted_modules: list[dict[str, Any]]) -> None:
        validate_modules_payload(
            [module_dict],
            transcript_lines=transcript_lines,
            enforce_duration=False,
            enforce_global_overlap=False,
        )

    def _collect_candidate_issues(module_dict: dict[str, Any], drafted_modules: list[dict[str, Any]]) -> list[str]:
        issues: list[str] = []
        total_duration = sum(float(seg["end"]) - float(seg["start"]) for seg in module_dict.get("segments", []))
        if total_duration < 180 or total_duration > 1200:
            issues.append(
                f"Suggested duration range is 3-20 minutes, but module duration is {total_duration:.1f}s."
            )

        used_ranges = _used_ranges_list(drafted_modules)
        for seg_idx, seg in enumerate(module_dict.get("segments", []), start=1):
            start = float(seg["start"])
            end = float(seg["end"])
            for used_start, used_end in used_ranges:
                if start < used_end and used_start < end:
                    issues.append(
                        "Module overlaps a previously selected range "
                        f"at segment {seg_idx} ({start:.2f}-{end:.2f} overlaps {used_start:.2f}-{used_end:.2f})."
                    )
        return issues

    def _snap_candidate_to_transcript_boundaries(module_dict: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        adjusted = dict(module_dict)
        raw_segments = module_dict.get("segments") or []
        if not isinstance(raw_segments, list):
            return adjusted, []

        valid_starts = {round(line.start, 2) for line in transcript_lines}
        valid_ends = {round(line.end, 2) for line in transcript_lines}
        snapped_segments: list[dict[str, Any]] = []
        logs: list[str] = []

        for seg_idx, seg in enumerate(raw_segments, start=1):
            snapped = dict(seg)
            start, start_changed = _snap_value_to_boundary(seg["start"], valid_starts)
            raw_end = round(float(seg["end"]), 2)
            candidate_ends = {point for point in valid_ends if point > start}
            if candidate_ends:
                end, end_changed = _snap_value_to_boundary(raw_end, candidate_ends)
            else:
                end, end_changed = _snap_value_to_boundary(raw_end, valid_ends)
            snapped["start"] = start
            snapped["end"] = end
            snapped_segments.append(snapped)
            if start_changed:
                logs.append(f"segment_{seg_idx}_start_snapped:{float(seg['start']):.2f}->{start:.2f}")
            if end_changed:
                logs.append(f"segment_{seg_idx}_end_snapped:{float(seg['end']):.2f}->{end:.2f}")

        adjusted["segments"] = snapped_segments
        return adjusted, logs

    def _reorder_module_segments(module_dict: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        reordered = dict(module_dict)
        raw_segments = module_dict.get("segments") or []
        if not isinstance(raw_segments, list):
            return reordered, False
        sorted_segments = sorted(raw_segments, key=lambda s: (float(s["start"]), float(s["end"])))
        changed = any(a is not b for a, b in zip(raw_segments, sorted_segments)) or len(raw_segments) != len(
            sorted_segments
        )
        reordered["segments"] = [dict(seg) for seg in sorted_segments]
        return reordered, changed

    def _attach_processing_metadata(
        module_dict: dict[str, Any],
        *,
        idea_index: int,
        retried: bool,
        reordered: bool,
        attempt_count: int,
        logs: list[str],
        errors_encountered: list[str],
    ) -> dict[str, Any]:
        out = dict(module_dict)
        out["retried"] = retried
        out["reordered"] = reordered
        out["processing_metadata"] = {
            "pipeline": "langgraph_segment_selection",
            "idea_index": idea_index + 1,
            "selection_attempts": attempt_count,
            "model": stage_cfg.model,
            "provider": stage_cfg.provider,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "errors_encountered": errors_encountered,
            "logs": logs,
        }
        return out

    def select_segments_for_current_idea(state: SegmentWorkflowState) -> SegmentWorkflowState:
        idx = state["current_idea_index"]
        idea = state["story_ideas"][idx]
        drafted = list(state.get("drafted_modules", []))
        selection_failures = list(state.get("selection_failures", []))
        logs: list[str] = [f"idea_{idx + 1}:initial_selection_started"]
        logger.info("Selecting segments for idea %s (%s)", idx + 1, idea.get("title", "untitled"))
        try:
            module_payload = select_segments_chain.invoke(
                {
                    "used_ranges_json": _used_ranges_json(drafted),
                    "context_analysis_json": json.dumps(state["context_analysis"], ensure_ascii=False, indent=2),
                    "idea_json": json.dumps(idea, ensure_ascii=False, indent=2),
                    "editorial_rules": state["editorial_rules"],
                    "transcript": state["transcript_text"],
                },
                config=invoke_config,
            )
            parsed_module = _normalize_module_from_flexible(module_payload, idx)
            candidate = parsed_module.model_dump()
            candidate, snap_logs = _snap_candidate_to_transcript_boundaries(candidate)
            logs.extend(snap_logs)
            errors_encountered: list[str] = []

            retried = False
            reordered = False
            attempt_count = 1

            try:
                _validate_candidate_module(candidate, drafted)
                logs.append("initial_selection_valid")
            except Exception as first_exc:  # noqa: BLE001
                first_err = str(first_exc)
                logs.append(f"initial_selection_invalid:{first_err}")
                logger.info("Idea %s failed validation on first attempt: %s", idx + 1, first_err)
                retried = True
                attempt_count = 2

                retry_payload = retry_select_segments_chain.invoke(
                    {
                        "validation_error": first_err,
                        "previous_attempt_json": json.dumps(candidate, ensure_ascii=False, indent=2),
                        "used_ranges_json": _used_ranges_json(drafted),
                        "context_analysis_json": json.dumps(state["context_analysis"], ensure_ascii=False, indent=2),
                        "idea_json": json.dumps(idea, ensure_ascii=False, indent=2),
                        "editorial_rules": state["editorial_rules"],
                        "transcript": state["transcript_text"],
                    },
                    config=invoke_config,
                )
                retry_module = _normalize_module_from_flexible(retry_payload, idx)
                candidate = retry_module.model_dump()
                candidate, retry_snap_logs = _snap_candidate_to_transcript_boundaries(candidate)
                logs.extend(retry_snap_logs)

                try:
                    _validate_candidate_module(candidate, drafted)
                    logs.append("retry_selection_valid")
                except Exception as second_exc:  # noqa: BLE001
                    second_err = str(second_exc)
                    logs.append(f"retry_selection_invalid:{second_err}")
                    logger.info("Idea %s failed validation on retry: %s", idx + 1, second_err)
                    candidate, reordered = _reorder_module_segments(candidate)
                    if reordered:
                        logs.append("segments_reordered_by_start")
                    else:
                        logs.append("reorder_attempt_no_change")
                    try:
                        _validate_candidate_module(candidate, drafted)
                        logs.append("reordered_selection_valid")
                    except Exception as final_exc:  # noqa: BLE001
                        logs.append(f"reordered_selection_invalid:{final_exc}")
                        raise ValueError(
                            f"Idea {idx + 1} failed after retry and reorder fallback: {final_exc}"
                        ) from final_exc

            errors_encountered = _collect_candidate_issues(candidate, drafted)
            logs.extend(f"nonfatal_issue:{issue}" for issue in errors_encountered)
            candidate = _attach_processing_metadata(
                candidate,
                idea_index=idx,
                retried=retried,
                reordered=reordered,
                attempt_count=attempt_count,
                logs=logs,
                errors_encountered=errors_encountered,
            )
            drafted.append(candidate)
            return {"drafted_modules": drafted, "selection_failures": selection_failures}
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
            logs.append(f"idea_skipped:{err}")
            logger.warning("Skipping idea %s (%s): %s", idx + 1, idea.get("title", "untitled"), err)
            selection_failures.append(
                {
                    "idea_index": idx + 1,
                    "title": idea.get("title"),
                    "category": idea.get("category"),
                    "error": err,
                    "logs": logs,
                }
            )
            return {"drafted_modules": drafted, "selection_failures": selection_failures}

    def advance_idea_index(state: SegmentWorkflowState) -> SegmentWorkflowState:
        return {"current_idea_index": state["current_idea_index"] + 1}

    def route_next(state: SegmentWorkflowState) -> str:
        if state.get("current_idea_index", 0) < len(state.get("story_ideas", [])):
            return "select_segments"
        return "finalize"

    def route_after_context(state: SegmentWorkflowState) -> str:
        if (state.get("user_idea_override") or "").strip():
            return "normalize_user_idea"
        return "generate_story_ideas"

    def finalize_output(state: SegmentWorkflowState) -> SegmentWorkflowState:
        modules = list(state.get("drafted_modules", []))
        failures = list(state.get("selection_failures", []))
        usage_summary = _build_usage_summary(
            usage_callback,
            provider=stage_cfg.provider,
            configured_model=stage_cfg.model,
        )

        def first_start(module: dict[str, Any]) -> float:
            segments = module.get("segments") or []
            if not segments:
                return float("inf")
            try:
                return min(float(seg["start"]) for seg in segments)
            except Exception:  # noqa: BLE001
                return float("inf")

        modules.sort(key=first_start)
        if not modules:
            if failures:
                summary = "; ".join(
                    f"idea {f.get('idea_index')}: {f.get('error')}" for f in failures[:3]
                )
                if len(failures) > 3:
                    summary += f"; ... ({len(failures) - 3} more)"
                raise RuntimeError(f"No valid modules were produced. All idea selections failed. {summary}")
            raise RuntimeError("No modules were produced.")
        if failures:
            logger.warning(
                "Segmentation completed with %s skipped idea(s); %s module(s) produced.",
                len(failures),
                len(modules),
            )
        else:
            logger.info("Segmentation completed successfully with %s module(s).", len(modules))
        output_payload = {
            "modules": modules,
            "workflow_metadata": {
                "llm_usage": usage_summary,
            },
        }
        return {"final_json": json.dumps(output_payload, ensure_ascii=False, indent=2)}

    graph = StateGraph(SegmentWorkflowState)
    graph.add_node("analyze_context", analyze_context)
    graph.add_node("normalize_user_idea", normalize_user_idea)
    graph.add_node("generate_story_ideas", generate_story_ideas)
    graph.add_node("init_selection_loop", init_selection_loop)
    graph.add_node("select_segments", select_segments_for_current_idea)
    graph.add_node("advance_idea_index", advance_idea_index)
    graph.add_node("finalize", finalize_output)

    graph.add_edge(START, "analyze_context")
    graph.add_conditional_edges(
        "analyze_context",
        route_after_context,
        {"normalize_user_idea": "normalize_user_idea", "generate_story_ideas": "generate_story_ideas"},
    )
    graph.add_edge("normalize_user_idea", "init_selection_loop")
    graph.add_edge("generate_story_ideas", "init_selection_loop")
    graph.add_conditional_edges("init_selection_loop", route_next, {"select_segments": "select_segments", "finalize": "finalize"})
    graph.add_edge("select_segments", "advance_idea_index")
    graph.add_conditional_edges("advance_idea_index", route_next, {"select_segments": "select_segments", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    app = graph.compile()
    result = app.invoke(
        {
            "transcript_text": normalized_transcript_text,
            "editorial_rules": editorial_rules,
            "user_idea_override": idea_override,
        }
    )
    final_json = result.get("final_json")
    if not isinstance(final_json, str) or not final_json.strip():
        raise RuntimeError("LangGraph workflow did not produce final JSON output.")
    return final_json
