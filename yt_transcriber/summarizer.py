"""Video summarization using Gemini AI.

This module provides AI-powered video summarization functionality,
generating executive summaries with key points, timestamps, and action items.
"""

import logging
import re
from datetime import datetime

from core.llm import call_gemini_with_cache
from core.models import TimestampedSection, VideoSummary
from core.settings import settings

logger = logging.getLogger(__name__)


class SummarizationError(Exception):
    """Raised when summarization fails."""

    pass


def generate_summary(
    transcript: str,
    video_title: str,
    video_url: str,
    video_id: str,
) -> VideoSummary:
    """Generate comprehensive video summary using Gemini AI.

    Args:
        transcript: Full video transcript text
        video_title: Title of the video
        video_url: YouTube URL
        video_id: YouTube video ID

    Returns:
        VideoSummary object with all fields populated

    Raises:
        SummarizationError: If Gemini API fails or response invalid
    """
    logger.info(f"Generating summary for video: {video_title}")

    # 1. Detect language from transcript
    language = _detect_language(transcript)
    logger.info(f"Detected language: {language}")

    # 2. Calculate statistics
    word_count = len(transcript.split())
    duration_minutes = word_count / 150  # Average speaking rate

    # 3. Build prompt
    prompt = _build_prompt(
        transcript=transcript,
        video_title=video_title,
        word_count=word_count,
        duration_minutes=duration_minutes,
        language=language,
    )

    # 4. Call Gemini API with caching
    inputs = {
        "transcript": transcript[:1000],  # First 1000 chars for uniqueness
        "video_title": video_title,
        "language": language,
        "task": "video_summarization",
    }

    summary_text = call_gemini_with_cache(
        prompt=prompt,
        model_name=settings.SUMMARIZER_MODEL,
        prompt_version=settings.SUMMARIZER_PROMPT_VERSION,
        inputs=inputs,
        temperature=0.7,
        error_class=SummarizationError,
    )

    logger.info(f"Summary response received ({len(summary_text)} chars)")

    # 5. Parse markdown response
    try:
        summary = _parse_summary_response(
            summary_text=summary_text,
            video_url=video_url,
            video_title=video_title,
            video_id=video_id,
            word_count=word_count,
            duration_minutes=duration_minutes,
            language=language,
        )
        logger.info("✅ Summary generated successfully")
        return summary

    except Exception as e:
        logger.error(f"Failed to parse summary: {e}", exc_info=True)
        raise SummarizationError(f"Failed to parse Gemini response: {e}") from e


def _detect_language(transcript: str) -> str:
    """Detect language from transcript text.

    Simple heuristic: count common Spanish vs English words.
    """
    # Sample first 500 words for faster detection
    sample = " ".join(transcript.split()[:500]).lower()

    # Common Spanish words
    spanish_indicators = [
        "el ",
        "la ",
        "los ",
        "las ",
        "de ",
        "que ",
        "es ",
        "en ",
        "un ",
        "una ",
        "para ",
        "con ",
        "por ",
        "está ",
        "son ",
        "pero ",
        "cómo ",
        "qué ",
        "español",
    ]

    # Common English words
    english_indicators = [
        "the ",
        "is ",
        "are ",
        "and ",
        "or ",
        "but ",
        "in ",
        "on ",
        "at ",
        "to ",
        "for ",
        "with ",
        "this ",
        "that ",
        "how ",
        "what ",
        "english",
    ]

    spanish_count = sum(sample.count(word) for word in spanish_indicators)
    english_count = sum(sample.count(word) for word in english_indicators)

    # Default to Spanish if inconclusive
    return "es" if spanish_count >= english_count else "en"


def _build_prompt(
    transcript: str,
    video_title: str,
    word_count: int,
    duration_minutes: float,
    language: str,
) -> str:
    """Build Gemini prompt based on detected language."""
    if language == "es":
        return SUMMARY_PROMPT_ES.format(
            video_title=video_title,
            transcript=transcript,
            word_count=word_count,
            duration=f"{duration_minutes:.1f}",
        )
    else:
        return SUMMARY_PROMPT_EN.format(
            video_title=video_title,
            transcript=transcript,
            word_count=word_count,
            duration=f"{duration_minutes:.1f}",
        )


def _parse_summary_response(
    summary_text: str,
    video_url: str,
    video_title: str,
    video_id: str,
    word_count: int,
    duration_minutes: float,
    language: str,
) -> VideoSummary:
    """Parse Gemini markdown response into VideoSummary object."""
    # Extract sections using regex
    exec_summary = _extract_section(
        summary_text, r"## 🎯 Resumen Ejecutivo|## 🎯 Executive Summary"
    )
    key_points = _extract_list_items(summary_text, r"## 🔑 Puntos Clave|## 🔑 Key Points")
    timestamps = _extract_timestamps(summary_text)
    conclusion = _extract_section(summary_text, r"## 💡 Conclusión|## 💡 Conclusion")
    action_items = _extract_list_items(summary_text, r"## ✅ Action Items")

    return VideoSummary(
        video_url=video_url,
        video_title=video_title,
        video_id=video_id,
        executive_summary=exec_summary.strip(),
        key_points=key_points,
        timestamps=timestamps,
        conclusion=conclusion.strip(),
        action_items=action_items,
        word_count=word_count,
        estimated_duration_minutes=duration_minutes,
        language=language,
        generated_at=datetime.now(),
    )


def _extract_section(text: str, header_pattern: str) -> str:
    """Extract content between section headers."""
    # Match header followed by content until next header (##) or end of string
    # Using negative lookahead to stop at next header (with optional leading whitespace)
    pattern = rf"(?:{header_pattern})\s*\n((?:(?!^\s*##).)+)"
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    if match:
        return match.group(1).strip()
    return ""


def _extract_list_items(text: str, header_pattern: str) -> list[str]:
    """Extract numbered or bulleted list items."""
    section = _extract_section(text, header_pattern)
    if not section:
        return []

    # Match numbered lists (1. Item) or bullets (- Item or * Item)
    items = re.findall(r"^\s*(?:\d+\.|[-*])\s+(.+)$", section, re.MULTILINE)
    return [item.strip() for item in items if item.strip()]


def _extract_timestamps(text: str) -> list[TimestampedSection]:
    """Extract timestamps with descriptions."""
    section = _extract_section(text, r"## ⏱️ Momentos Importantes|## ⏱️ Important Moments")
    if not section:
        return []

    timestamps = []
    # Match patterns like: - **05:30** - Description (one per line)
    pattern = r"-\s+\*\*(\d{1,2}:\d{2}(?::\d{2})?)\*\*\s+-\s+(.+)"
    matches = re.findall(pattern, section, re.MULTILINE)

    for timestamp, description in matches:
        timestamps.append(
            TimestampedSection(
                timestamp=timestamp.strip(),
                description=description.strip(),
                importance=3,  # Default importance
            )
        )

    return timestamps


# ============================================================================
# Prompt Templates
# ============================================================================

SUMMARY_PROMPT_ES = """
Analiza la siguiente transcripción de video de YouTube y genera un resumen ejecutivo completo.

**VIDEO**: {video_title}
**DURACIÓN**: ~{duration} minutos
**PALABRAS**: {word_count}

**TRANSCRIPCIÓN**:
{transcript}

---

Genera un resumen estructurado siguiendo EXACTAMENTE este formato Markdown:

# 📹 Resumen: {video_title}

## 🎯 Resumen Ejecutivo
[Escribe 2-3 líneas describiendo de qué trata el video, qué temas cubre y qué se aprende. Sé conciso pero completo.]

## 🔑 Puntos Clave
1. **[Tema 1]**: [Explicación breve del primer punto principal con contexto]
2. **[Tema 2]**: [Explicación del segundo punto importante]
3. **[Tema 3]**: [Tercer punto clave]
4. **[Tema 4]**: [Cuarto punto]
5. **[Tema 5]**: [Quinto punto]
6. **[Tema 6]**: [Sexto punto (opcional)]
7. **[Tema 7]**: [Séptimo punto (opcional)]

[Incluir 5-7 puntos totales. Cada punto debe tener el tema en **negrita** seguido de explicación.]

## ⏱️ Momentos Importantes
- **00:00** - [Descripción breve del tema de inicio]
- **MM:SS** - [Descripción de momento clave 2]
- **MM:SS** - [Descripción de momento clave 3]
- **MM:SS** - [Descripción de momento clave 4]
- **MM:SS** - [Descripción de momento clave 5]
- **MM:SS** - [Descripción de momento clave 6 (opcional)]
- **MM:SS** - [Descripción de momento clave 7 (opcional)]
- **MM:SS** - [Descripción de momento clave 8 (opcional)]

[Incluir 5-8 timestamps. Si la transcripción tiene timestamps explícitos, úsalos. Si no, infiere los momentos basándote en la secuencia del contenido. Formato: **MM:SS** en negrita.]

## 💡 Conclusión
[Escribe 1-2 líneas con el mensaje principal o takeaway más importante del video. ¿Qué debe recordar el espectador?]

## ✅ Action Items
1. [Primera acción específica y práctica que puede tomar el espectador después de ver el video]
2. [Segunda acción concreta relacionada con el contenido]
3. [Tercera acción aplicable]
4. [Cuarta acción (opcional)]
5. [Quinta acción (opcional)]

[Incluir 3-5 action items totales. Cada uno debe ser específico, accionable y relacionado directamente con el contenido del video.]

---
**📊 Estadísticas**: {word_count} palabras | ~{duration} minutos de contenido

**INSTRUCCIONES CRÍTICAS**:
- Usa EXACTAMENTE el formato markdown mostrado arriba
- Todos los emojis (📹, 🎯, 🔑, ⏱️, 💡, ✅, 📊) deben estar incluidos
- Los timestamps deben estar en formato **MM:SS** en negrita
- Los temas de puntos clave deben estar en **negrita**
- Mantén un tono profesional pero accesible
- Enfócate en información práctica y útil
- NO inventes información que no esté en la transcripción
"""

SUMMARY_PROMPT_EN = """
Analyze the following YouTube video transcript and generate a comprehensive executive summary.

**VIDEO**: {video_title}
**DURATION**: ~{duration} minutes
**WORDS**: {word_count}

**TRANSCRIPT**:
{transcript}

---

Generate a structured summary following EXACTLY this Markdown format:

# 📹 Summary: {video_title}

## 🎯 Executive Summary
[Write 2-3 lines describing what the video is about, what topics it covers, and what viewers learn. Be concise but complete.]

## 🔑 Key Points
1. **[Topic 1]**: [Brief explanation of the first main point with context]
2. **[Topic 2]**: [Explanation of the second important point]
3. **[Topic 3]**: [Third key point]
4. **[Topic 4]**: [Fourth point]
5. **[Topic 5]**: [Fifth point]
6. **[Topic 6]**: [Sixth point (optional)]
7. **[Topic 7]**: [Seventh point (optional)]

[Include 5-7 total points. Each point should have the topic in **bold** followed by explanation.]

## ⏱️ Important Moments
- **00:00** - [Brief description of opening topic]
- **MM:SS** - [Description of key moment 2]
- **MM:SS** - [Description of key moment 3]
- **MM:SS** - [Description of key moment 4]
- **MM:SS** - [Description of key moment 5]
- **MM:SS** - [Description of key moment 6 (optional)]
- **MM:SS** - [Description of key moment 7 (optional)]
- **MM:SS** - [Description of key moment 8 (optional)]

[Include 5-8 timestamps. If the transcript has explicit timestamps, use them. If not, infer moments based on content sequence. Format: **MM:SS** in bold.]

## 💡 Conclusion
[Write 1-2 lines with the main message or most important takeaway from the video. What should viewers remember?]

## ✅ Action Items
1. [First specific, practical action viewers can take after watching the video]
2. [Second concrete action related to the content]
3. [Third applicable action]
4. [Fourth action (optional)]
5. [Fifth action (optional)]

[Include 3-5 total action items. Each should be specific, actionable, and directly related to the video content.]

---
**📊 Statistics**: {word_count} words | ~{duration} minutes of content

**CRITICAL INSTRUCTIONS**:
- Use EXACTLY the markdown format shown above
- All emojis (📹, 🎯, 🔑, ⏱️, 💡, ✅, 📊) must be included
- Timestamps must be in **MM:SS** format in bold
- Key point topics must be in **bold**
- Maintain a professional but accessible tone
- Focus on practical and useful information
- DO NOT invent information not present in the transcript
"""
