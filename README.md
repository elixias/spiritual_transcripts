# Spiritual Transcript Pipeline

## Command to transcribe audio
stp segment "/Users/elixander.tan/Downloads/Module 1 Intro to Shakti Kundalini Yoga_whisper_1.txt" -o "/Users/elixander.tan/Desktop/raw_text/processed/raw_text/M1_whisper_v5_with_user_idea_input.json" --idea "/Users/elixander.tan/Desktop/raw_text/2012 06 19 Sadhak Induction Modules/ideas.txt"

stp export-pdf "/Users/elixander.tan/Desktop/raw_text/processed/raw_text/Module 2 Divine Healing & Karma.json" \
  -o "/Users/elixander.tan/Desktop/raw_text/processed/raw_text/Module 2 Divine Healing & Karma.pdf" \
  --title "Module 2 - Divine Healing & Karma"

## CLI Pipeline - About

CLI pipeline to:

1. Extract audio from a video
2. Transcribe audio into `[start-end] text` lines (timestamped transcript)
3. Send transcript to an LLM to produce strict JSON modules
4. Cut module clips from the source video using the JSON timestamps

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` installed on PATH
- API credentials for your transcription model and segmentation model

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configure

```bash
cp .env.example .env
```

Set your keys/models in `.env`. You can use different providers/models for:

- `TRANSCRIBE_*` for audio transcription
- `SEGMENT_*` for transcript -> JSON segmentation

You do not need to repeat API keys per stage in the normal setup:
- Transcription uses `OPENAI_API_KEY` by default
- Segmentation uses the provider key inferred from `SEGMENT_MODEL`
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`)

`TRANSCRIBE_API_KEY` and `SEGMENT_API_KEY` are optional overrides only (for example, if you want
different billing/account keys for specific stages).

Segmentation now uses a LangGraph workflow (with LangChain chat model wrappers) and a model manager that infers the provider from
`SEGMENT_MODEL` (for example `gpt-*` -> OpenAI, `claude-*` -> Anthropic, `gemini-*` -> Google).
If `SEGMENT_API_KEY` is not set, the provider-specific env key (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`) is used automatically.

Transcription remains on the OpenAI SDK because this pipeline requires timestamped segment output
(`verbose_json` with segment timestamps), which LangChain does not provide as a stable abstraction.

### Cut rendering defaults

Clipping now has a little extra rendering pass that overlays your logo and burns subtitles. The appearance and encoder are driven by cut-specific keys in `.env`:

- `CUT_LOGO_PATH`: path to a PNG/JPG that will be placed in the top-right corner (margin controlled by `CUT_LOGO_MARGIN`).
- `CUT_ENABLE_SUBTITLES`: set to `0`/`false` to disable the automatic subtitles that are generated from each module's segments.
- `CUT_SUBTITLE_FONT_PATH`: optional font directory for ffmpeg (`fontsdir`). Use `CUT_SUBTITLE_FONT_SIZE`, `CUT_SUBTITLE_COLOR`, `CUT_SUBTITLE_OUTLINE_COLOR`, `CUT_SUBTITLE_OUTLINE_WIDTH`, and `CUT_SUBTITLE_MARGIN` to tweak the caption styling.
- `CUT_VIDEO_ENCODER`: default `libx264`; change to `h264_nvenc`/`hevc_nvenc` to keep the pass GPU-accelerated.
- `CUT_VIDEO_ENCODER_ARGS`: encoder arguments (default `-preset medium -crf 20`). For fast GPU output try `-preset fast -rc:v vbr_hq -cq 20 -b:v 0`.

The same defaults are used by `stp cut` and the `run-all` workflow, but you can override them per invocation with the new CLI flags.

## Commands

### 1) Extract audio

```bash
stp extract-audio input.mp4 -o work/audio.wav
```

### 2) Transcribe audio to timestamped text

```bash
stp transcribe work/audio.wav \
  --transcript-out work/transcript.txt \
  --raw-json-out work/transcript_verbose.json
```

### 3) Generate module JSON from transcript

```bash
stp segment work/transcript.txt -o work/modules.json
```

### 4) Cut clips from the source video

```bash
stp cut input.mp4 work/modules.json --out-dir work/clips
pipenv run stp cut "C:\Users\elixa\Desktop\7 Modules\M1 Intro to Shakti Kundalini Yoga.mp4" "C:\Users\elixa\Desktop\archive\M1_whisper_v5_with_user_idea_input.json" --out-dir "C:\Users\elixa\Desktop\archive\processed_clips" --logo "C:/Users/elixa/Desktop/archive/sscf_logohead.png" 
```

By default this command overlays the configured logo and burns subtitles from the module segments. Use `--logo`/`--skip-logo`/`--logo-margin` to control the placement and `--no-subtitles`, `--subtitle-font`, or `--subtitle-font-size` to adjust the captions.

### 5) Export modules JSON to PDF

```bash
stp export-pdf work/modules.json -o work/modules.pdf
```

### Full pipeline

```bash
stp run-all input.mp4 --work-dir work/run1
```

This creates:

- `work/run1/audio.wav`
- `work/run1/transcript.txt`
- `work/run1/transcript_verbose.json`
- `work/run1/modules.json`
- `work/run1/clips/*.mp4`

## Notes

- The `segment` command validates JSON structure, non-overlapping segments, and approximate module duration (3-20 min by summed segment duration).
- The `cut` command creates one output clip per module by cutting each module segment and concatenating them.
- The `export-pdf` command renders the modules JSON into a readable PDF with metadata and timestamped transcript segments.
- The prompt template is stored in `prompts/segment_modules_prompt.txt`.
- Segmentation orchestration uses LangGraph (context analysis -> story idea orchestration -> per-idea segment selection).
