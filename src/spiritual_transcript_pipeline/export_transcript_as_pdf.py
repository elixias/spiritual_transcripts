from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .segmentation import extract_modules_payload, get_workflow_metadata


def _format_timestamp(seconds: float) -> str:
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}.{milliseconds:03d}"
    return f"{mins:02d}:{secs:02d}.{milliseconds:03d}"


def _format_optional_number(value: object, *, decimals: int = 6) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.{decimals}f}"
    return str(value) if value is not None else "-"


def export_modules_pdf(modules_json_path: Path, output_path: Path, *, document_title: str | None = None) -> Path:
    payload = json.loads(modules_json_path.read_text(encoding="utf-8"))
    modules = extract_modules_payload(payload)
    workflow_metadata = get_workflow_metadata(payload) or {}
    llm_usage = workflow_metadata.get("llm_usage") if isinstance(workflow_metadata, dict) else {}
    if not isinstance(llm_usage, dict):
        llm_usage = {}

    title_text = document_title or f"{modules_json_path.stem} - Modules Transcript"
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=LETTER,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Heading1"]
    h2_style = styles["Heading2"]
    h3_style = styles["Heading3"]
    body_style = styles["BodyText"]
    meta_style = ParagraphStyle(
        "meta",
        parent=body_style,
        fontSize=10.2,
        leading=13,
        spaceAfter=6,
    )
    segment_style = ParagraphStyle(
        "segment",
        parent=body_style,
        fontSize=10.5,
        leading=14,
        spaceAfter=8,
    )
    timestamp_style = ParagraphStyle(
        "timestamp",
        parent=body_style,
        fontSize=9.8,
        textColor=colors.HexColor("#444444"),
        leading=12,
        spaceAfter=2,
    )

    elements: list[object] = []
    elements.append(Paragraph(escape(title_text), title_style))
    elements.append(Spacer(1, 0.15 * inch))
    elements.append(
        Paragraph(
            "This document is generated from the JSON output and includes module metadata and timestamped transcript segments.",
            body_style,
        )
    )
    elements.append(Spacer(1, 0.2 * inch))

    summary_rows = [
        ["Provider", str(llm_usage.get("provider", "-"))],
        ["Configured model", str(llm_usage.get("configured_model", "-"))],
        ["Input tokens", str(llm_usage.get("input_tokens", "-"))],
        ["Output tokens", str(llm_usage.get("output_tokens", "-"))],
        ["Total tokens", str(llm_usage.get("total_tokens", "-"))],
        ["Estimated cost (USD)", _format_optional_number(llm_usage.get("estimated_cost_usd"))],
    ]
    table = Table(summary_rows, colWidths=[1.6 * inch, 4.6 * inch])
    table.setStyle(
        TableStyle(
            [
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ]
        )
    )
    elements.append(table)

    if modules:
        elements.append(PageBreak())

    for idx, module in enumerate(modules, start=1):
        title = str(module.get("title") or "(untitled)")
        elements.append(Paragraph(f"Module {idx}: {escape(title)}", h2_style))
        elements.append(Spacer(1, 0.08 * inch))

        category = escape(str(module.get("category") or "-"))
        subcategory = escape(str(module.get("subcategory") or "-"))
        confidence = module.get("categorization_confidence")
        confidence_text = f"{float(confidence):.2f}" if isinstance(confidence, (int, float)) else "-"
        context = escape(str(module.get("context") or "-"))

        elements.append(
            Paragraph(
                f"<b>Category:</b> {category} &nbsp;&nbsp; "
                f"<b>Subcategory:</b> {subcategory} &nbsp;&nbsp; "
                f"<b>Confidence:</b> {confidence_text}",
                meta_style,
            )
        )
        elements.append(Paragraph(f"<b>Context:</b> {context}", meta_style))

        normalized_terms = module.get("normalized_terms") or {}
        if isinstance(normalized_terms, dict) and normalized_terms:
            elements.append(Paragraph("<b>Key terms (normalized):</b>", meta_style))
            items = [
                ListItem(
                    Paragraph(
                        f"{escape(str(source))} -&gt; {escape(str(target))}",
                        meta_style,
                    )
                )
                for source, target in normalized_terms.items()
            ]
            elements.append(ListFlowable(items, bulletType="bullet", leftIndent=14, bulletFontSize=8))
            elements.append(Spacer(1, 0.12 * inch))
        else:
            elements.append(Spacer(1, 0.06 * inch))

        elements.append(Paragraph("Transcript segments (timestamped):", h3_style))
        elements.append(Spacer(1, 0.06 * inch))

        segments = module.get("segments") or []
        for segment in segments:
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
            text = escape(str(segment.get("text") or "-")).replace("\n", "<br/>")
            elements.append(
                Paragraph(
                    f"[{_format_timestamp(start)} -&gt; {_format_timestamp(end)}]",
                    timestamp_style,
                )
            )
            elements.append(Paragraph(text, segment_style))

        if idx != len(modules):
            elements.append(PageBreak())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.build(elements)
    return output_path
