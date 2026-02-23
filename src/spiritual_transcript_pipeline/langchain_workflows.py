from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import StageLLMConfig
from .model_manager import ModelManager


class SegmentWorkflowState(TypedDict, total=False):
    transcript_text: str
    editorial_rules: str
    context_analysis: dict[str, Any]
    story_ideas: list[dict[str, Any]]
    current_idea_index: int
    drafted_modules: list[dict[str, Any]]
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
    segments: list[ModuleSegmentOutput] = Field(default_factory=list)


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


def run_segment_generation_workflow(
    *,
    transcript_text: str,
    prompt_file: Path,
    stage_cfg: StageLLMConfig,
    temperature: float,
) -> str:
    editorial_rules = _load_editorial_rules(prompt_file)
    model = ModelManager(stage_cfg).get_chat_model(temperature=temperature)

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
- Use the closest taxonomy category from the editorial rules when applicable.

Output format instructions:
{format_instructions}

Editorial rules:
{editorial_rules}

Context analysis JSON:
{context_analysis_json}

Transcript:
<<<TRANSCRIPT>>>
{transcript}
<<<END>>>
        """.strip(),
        model,
        ideas_parser,
    )

    module_parser = PydanticOutputParser(pydantic_object=ModuleSelectionOutput)
    select_segments_chain = _pydantic_chain(
        """
You are an editor assembling one video module from a raw timestamped transcript.

Your task:
- Given ONE module idea and the full transcript, select the exact transcript segments that support the idea.
- Use exact timestamps from transcript boundaries.
- Merge adjacent lines conceptually when useful.
- Build a final module JSON object in the required output shape.

Return a STRICT JSON object for one module.

Important:
- Use the idea's story orchestrator as guidance for selection only.
- Do not fabricate content or paraphrase beyond conservative cleanup.
- Do not overlap segments with previously selected time ranges.
- Prefer coherent module flow: opening -> explanation -> important fact -> conclusion (when present in transcript).

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

    def analyze_context(state: SegmentWorkflowState) -> SegmentWorkflowState:
        context_payload = context_chain.invoke({"transcript": state["transcript_text"]})
        return {"context_analysis": context_payload.model_dump()}

    def generate_story_ideas(state: SegmentWorkflowState) -> SegmentWorkflowState:
        ideas_payload = ideas_chain.invoke(
            {
                "editorial_rules": state["editorial_rules"],
                "context_analysis_json": json.dumps(state["context_analysis"], ensure_ascii=False, indent=2),
                "transcript": state["transcript_text"],
            },
        )
        ideas: list[dict[str, Any]] = []
        for idx, item in enumerate(ideas_payload.ideas, start=1):
            item_dict = item.model_dump()
            if not item_dict.get("title"):
                raise ValueError(f"Idea {idx} missing title.")
            if not item_dict.get("category"):
                raise ValueError(f"Idea {idx} missing category.")
            ideas.append(item_dict)
        return {"story_ideas": ideas}

    def init_selection_loop(_: SegmentWorkflowState) -> SegmentWorkflowState:
        return {"current_idea_index": 0, "drafted_modules": []}

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

    def select_segments_for_current_idea(state: SegmentWorkflowState) -> SegmentWorkflowState:
        idx = state["current_idea_index"]
        idea = state["story_ideas"][idx]
        drafted = list(state.get("drafted_modules", []))
        module_payload = select_segments_chain.invoke(
            {
                "used_ranges_json": _used_ranges_json(drafted),
                "context_analysis_json": json.dumps(state["context_analysis"], ensure_ascii=False, indent=2),
                "idea_json": json.dumps(idea, ensure_ascii=False, indent=2),
                "editorial_rules": state["editorial_rules"],
                "transcript": state["transcript_text"],
            },
        )
        drafted.append(module_payload.model_dump())
        return {"drafted_modules": drafted}

    def advance_idea_index(state: SegmentWorkflowState) -> SegmentWorkflowState:
        return {"current_idea_index": state["current_idea_index"] + 1}

    def route_next(state: SegmentWorkflowState) -> str:
        if state.get("current_idea_index", 0) < len(state.get("story_ideas", [])):
            return "select_segments"
        return "finalize"

    def finalize_output(state: SegmentWorkflowState) -> SegmentWorkflowState:
        modules = list(state.get("drafted_modules", []))

        def first_start(module: dict[str, Any]) -> float:
            segments = module.get("segments") or []
            if not segments:
                return float("inf")
            try:
                return min(float(seg["start"]) for seg in segments)
            except Exception:  # noqa: BLE001
                return float("inf")

        modules.sort(key=first_start)
        return {"final_json": json.dumps(modules, ensure_ascii=False, indent=2)}

    graph = StateGraph(SegmentWorkflowState)
    graph.add_node("analyze_context", analyze_context)
    graph.add_node("generate_story_ideas", generate_story_ideas)
    graph.add_node("init_selection_loop", init_selection_loop)
    graph.add_node("select_segments", select_segments_for_current_idea)
    graph.add_node("advance_idea_index", advance_idea_index)
    graph.add_node("finalize", finalize_output)

    graph.add_edge(START, "analyze_context")
    graph.add_edge("analyze_context", "generate_story_ideas")
    graph.add_edge("generate_story_ideas", "init_selection_loop")
    graph.add_conditional_edges("init_selection_loop", route_next, {"select_segments": "select_segments", "finalize": "finalize"})
    graph.add_edge("select_segments", "advance_idea_index")
    graph.add_conditional_edges("advance_idea_index", route_next, {"select_segments": "select_segments", "finalize": "finalize"})
    graph.add_edge("finalize", END)

    app = graph.compile()
    result = app.invoke(
        {
            "transcript_text": transcript_text,
            "editorial_rules": editorial_rules,
        }
    )
    final_json = result.get("final_json")
    if not isinstance(final_json, str) or not final_json.strip():
        raise RuntimeError("LangGraph workflow did not produce final JSON output.")
    return final_json
