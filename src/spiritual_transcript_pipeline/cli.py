from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from .config import PipelineConfig
from .cutter import cut_modules_to_clips
from .export_transcript_as_pdf import export_modules_pdf
from .ffmpeg_utils import extract_audio_to_wav
from .langchain_workflows import run_segment_generation_workflow
from .llm_clients import transcribe_audio_with_timestamps
from .segmentation import (
    get_workflow_metadata,
    modules_to_jsonable,
    parse_llm_json_response,
    validate_modules_payload,
)
from .transcript import parse_transcript_txt, save_json, save_transcript_txt


logger = logging.getLogger(__name__)


def _resolve_idea_override(idea_arg: str | None) -> str | None:
    if idea_arg is None:
        return None
    candidate = Path(idea_arg).expanduser()
    if candidate.is_file():
        text = candidate.read_text(encoding="utf-8")
        logger.info(
            "Resolved --idea from file %s (%s chars, %s lines)\n<<<IDEA_FILE_CONTENT>>>\n%s\n<<<END_IDEA_FILE_CONTENT>>>",
            candidate,
            len(text),
            len(text.splitlines()),
            text,
        )
        return text
    return idea_arg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stp",
        description="Spiritual transcript pipeline: extract audio, transcribe, segment, and cut clips.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract-audio", help="Extract mono WAV audio from a video file.")
    p_extract.add_argument("input_video", type=Path)
    p_extract.add_argument("-o", "--output", type=Path, required=True)
    p_extract.add_argument("--sample-rate", type=int, default=16000)

    p_transcribe = sub.add_parser(
        "transcribe", help="Transcribe audio to timestamped transcript text using the configured model."
    )
    p_transcribe.add_argument("input_audio", type=Path)
    p_transcribe.add_argument("--transcript-out", type=Path, required=True)
    p_transcribe.add_argument("--raw-json-out", type=Path)

    p_segment = sub.add_parser("segment", help="Generate strict JSON modules from a timestamped transcript.")
    p_segment.add_argument("transcript_txt", type=Path)
    p_segment.add_argument("-o", "--output", type=Path, required=True)
    p_segment.add_argument("--prompt-file", type=Path)
    p_segment.add_argument("--raw-response-out", type=Path)
    p_segment.add_argument("--no-validate", action="store_true")
    p_segment.add_argument(
        "--idea",
        type=str,
        help=(
            "User-provided module idea text, or a path to a text file containing one or more ideas "
            "(for example, one idea per line/paragraph). If set, skips automatic idea generation."
        ),
    )

    p_cut = sub.add_parser("cut", help="Cut module clips from source video using modules JSON.")
    p_cut.add_argument("input_video", type=Path)
    p_cut.add_argument("modules_json", type=Path)
    p_cut.add_argument("--out-dir", type=Path, required=True)

    p_export_pdf = sub.add_parser("export-pdf", help="Render modules JSON into a formatted PDF transcript.")
    p_export_pdf.add_argument("modules_json", type=Path)
    p_export_pdf.add_argument("-o", "--output", type=Path, required=True)
    p_export_pdf.add_argument("--title", type=str)

    p_all = sub.add_parser("run-all", help="Run extract -> transcribe -> segment -> cut.")
    p_all.add_argument("input_video", type=Path)
    p_all.add_argument("--work-dir", type=Path)
    p_all.add_argument("--prompt-file", type=Path)
    p_all.add_argument("--sample-rate", type=int, default=16000)
    p_all.add_argument("--no-validate", action="store_true")
    p_all.add_argument(
        "--idea",
        type=str,
        help=(
            "User-provided module idea text, or a path to a text file containing one or more ideas "
            "(for example, one idea per line/paragraph). If set, skips automatic idea generation."
        ),
    )

    return parser


def _print(msg: str) -> None:
    print(msg, file=sys.stderr)


def cmd_extract_audio(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    _print(f"[extract-audio] Extracting audio from {args.input_video} -> {args.output}")
    extract_audio_to_wav(cfg.ffmpeg_bin, args.input_video, args.output, sample_rate=args.sample_rate)
    _print("[extract-audio] Done")
    return 0


def cmd_transcribe(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    _print(f"[transcribe] Transcribing {args.input_audio} with model={cfg.transcribe.model}")
    lines, raw_payload = transcribe_audio_with_timestamps(args.input_audio, cfg.transcribe)
    save_transcript_txt(lines, args.transcript_out)
    _print(f"[transcribe] Wrote transcript -> {args.transcript_out}")
    if args.raw_json_out:
        save_json(raw_payload, args.raw_json_out)
        _print(f"[transcribe] Wrote raw transcription JSON -> {args.raw_json_out}")
    return 0


def cmd_segment(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    transcript_lines = parse_transcript_txt(args.transcript_txt)
    transcript_text = args.transcript_txt.read_text(encoding="utf-8")
    prompt_path = args.prompt_file or cfg.segment_prompt_file
    _print(f"[segment] Using prompt={prompt_path} model={cfg.segment.model}")
    raw_response = run_segment_generation_workflow(
        transcript_text=transcript_text,
        prompt_file=prompt_path,
        stage_cfg=cfg.segment,
        temperature=cfg.segment_temperature,
        idea_override=_resolve_idea_override(args.idea),
    )
    if args.raw_response_out:
        args.raw_response_out.parent.mkdir(parents=True, exist_ok=True)
        args.raw_response_out.write_text(raw_response, encoding="utf-8")
        _print(f"[segment] Wrote raw LLM response -> {args.raw_response_out}")

    payload = parse_llm_json_response(raw_response)
    if args.no_validate:
        output_payload = payload
    else:
        modules = validate_modules_payload(
            payload,
            transcript_lines=transcript_lines,
            enforce_duration=False,
            enforce_global_overlap=False,
        )
        output_payload = modules_to_jsonable(modules, workflow_metadata=get_workflow_metadata(payload))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print(f"[segment] Wrote modules JSON -> {args.output}")
    return 0


def cmd_cut(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    _print(f"[cut] Cutting clips from {args.input_video} using {args.modules_json}")
    created = cut_modules_to_clips(cfg.ffmpeg_bin, args.input_video, args.modules_json, args.out_dir)
    _print(f"[cut] Created {len(created)} clips in {args.out_dir}")
    return 0


def cmd_export_pdf(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    del cfg
    _print(f"[export-pdf] Rendering {args.modules_json} -> {args.output}")
    export_modules_pdf(args.modules_json, args.output, document_title=args.title)
    _print(f"[export-pdf] Wrote PDF -> {args.output}")
    return 0


def cmd_run_all(args: argparse.Namespace, cfg: PipelineConfig) -> int:
    video_path: Path = args.input_video
    if args.work_dir:
        work_dir = args.work_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work_dir = cfg.output_root / f"{video_path.stem}_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)

    audio_path = work_dir / "audio.wav"
    transcript_path = work_dir / "transcript.txt"
    raw_transcribe_path = work_dir / "transcript_verbose.json"
    modules_path = work_dir / "modules.json"
    raw_llm_path = work_dir / "segment_raw_response.txt"
    clips_dir = work_dir / "clips"

    _print(f"[run-all] Working directory: {work_dir}")
    extract_audio_to_wav(cfg.ffmpeg_bin, video_path, audio_path, sample_rate=args.sample_rate)
    _print(f"[run-all] Audio extracted -> {audio_path}")

    lines, raw_payload = transcribe_audio_with_timestamps(audio_path, cfg.transcribe)
    save_transcript_txt(lines, transcript_path)
    save_json(raw_payload, raw_transcribe_path)
    _print(f"[run-all] Transcript written -> {transcript_path}")

    prompt_path = args.prompt_file or cfg.segment_prompt_file
    raw_response = run_segment_generation_workflow(
        transcript_text=transcript_path.read_text(encoding="utf-8"),
        prompt_file=prompt_path,
        stage_cfg=cfg.segment,
        temperature=cfg.segment_temperature,
        idea_override=_resolve_idea_override(args.idea),
    )
    raw_llm_path.write_text(raw_response, encoding="utf-8")

    payload = parse_llm_json_response(raw_response)
    if args.no_validate:
        modules_jsonable = payload
    else:
        modules = validate_modules_payload(
            payload,
            transcript_lines=lines,
            enforce_duration=False,
            enforce_global_overlap=False,
        )
        modules_jsonable = modules_to_jsonable(modules, workflow_metadata=get_workflow_metadata(payload))
    modules_path.write_text(json.dumps(modules_jsonable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _print(f"[run-all] Modules JSON written -> {modules_path}")

    created = cut_modules_to_clips(cfg.ffmpeg_bin, video_path, modules_path, clips_dir)
    _print(f"[run-all] Created {len(created)} clips in {clips_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
        logging.getLogger("spiritual_transcript_pipeline").setLevel(logging.INFO)
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = PipelineConfig.from_env()

    try:
        if args.command == "extract-audio":
            return cmd_extract_audio(args, cfg)
        if args.command == "transcribe":
            return cmd_transcribe(args, cfg)
        if args.command == "segment":
            return cmd_segment(args, cfg)
        if args.command == "cut":
            return cmd_cut(args, cfg)
        if args.command == "export-pdf":
            return cmd_export_pdf(args, cfg)
        if args.command == "run-all":
            return cmd_run_all(args, cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
