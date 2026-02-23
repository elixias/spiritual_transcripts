# Spiritual Transcript Pipeline

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
- The prompt template is stored in `prompts/segment_modules_prompt.txt`.
- Segmentation orchestration uses LangGraph (context analysis -> story idea orchestration -> per-idea segment selection).
