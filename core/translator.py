"""Script Translator - Translates generated scripts to Spanish with context preservation.

Shared translator module used by both yt_transcriber and yt_script_generator.
"""

import logging
import re
from datetime import datetime

from core.llm import call_llm
from core.models import TimestampedSection, VideoSummary
from core.settings import settings

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when translation fails."""

    pass


class ScriptTranslator:
    """Translates generated scripts and summaries to Spanish with intelligent term preservation."""

    def __init__(self, use_translation_model: bool = True):
        """Initialize the translator.

        Args:
            use_translation_model: If True, uses TRANSLATOR_MODEL (haiku for summaries).
                                   If False, uses SUMMARIZER_MODEL (sonnet for scripts).
        """
        self.model_name = (
            settings.TRANSLATOR_MODEL if use_translation_model else settings.SUMMARIZER_MODEL
        )

        logger.info(f"ScriptTranslator initialized with model: {self.model_name}")

    def translate_to_spanish(self, script):
        """Translate script from English to Spanish with context awareness.

        Args:
            script: Generated script in English (GeneratedScript object)

        Returns:
            New GeneratedScript object with Spanish translation

        Raises:
            TranslationError: If translation fails
        """
        # Import here to avoid circular dependency
        from yt_script_generator.models import GeneratedScript

        logger.info(f"Translating script: {script.seo_title}")

        try:
            # Translate main script content
            translated_content = self._translate_content(
                script.script_markdown,
                script.seo_title,
            )

            # Translate SEO metadata
            translated_title = self._translate_seo_title(script.seo_title)
            translated_description = self._translate_seo_description(script.seo_description)

            # Keep original tags + add Spanish variants
            translated_tags = self._adapt_seo_tags(script.seo_tags)

            # Create new script object with translations
            translated_script = GeneratedScript(
                user_idea=script.user_idea,
                script_markdown=translated_content,
                estimated_duration_minutes=script.estimated_duration_minutes,
                word_count=len(translated_content.split()),
                seo_title=translated_title,
                seo_description=translated_description,
                seo_tags=translated_tags,
                synthesis_used=script.synthesis_used,
                num_reference_videos=script.num_reference_videos,
                generation_timestamp=script.generation_timestamp,
                cost_usd=script.cost_usd,
                estimated_quality_score=script.estimated_quality_score,
            )

            logger.info("Translation completed successfully")
            return translated_script

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(f"Failed to translate script: {e}") from e

    def translate_summary(self, summary: VideoSummary) -> VideoSummary:
        """Translate video summary from English to Spanish with context awareness.

        Args:
            summary: VideoSummary object in English

        Returns:
            New VideoSummary object with Spanish translation

        Raises:
            TranslationError: If translation fails
        """
        logger.info(f"Translating summary: {summary.video_title}")

        try:
            # Translate executive summary
            translated_executive = self._translate_text_block(
                summary.executive_summary,
                "executive summary",
                summary.video_title,
            )

            # Translate key points (list of strings)
            translated_key_points = [
                self._translate_text_block(point, "key point", summary.video_title)
                for point in summary.key_points
            ]

            # Translate timestamps (preserve timestamp format, translate descriptions)
            translated_timestamps = [
                TimestampedSection(
                    timestamp=ts.timestamp,  # Keep timestamp unchanged
                    description=self._translate_text_block(
                        ts.description, "timestamp description", summary.video_title
                    ),
                    importance=ts.importance,  # Keep importance score
                )
                for ts in summary.timestamps
            ]

            # Translate conclusion
            translated_conclusion = self._translate_text_block(
                summary.conclusion,
                "conclusion",
                summary.video_title,
            )

            # Translate action items
            translated_action_items = [
                self._translate_text_block(item, "action item", summary.video_title)
                for item in summary.action_items
            ]

            # Create new summary object with Spanish translations
            translated_summary = VideoSummary(
                video_url=summary.video_url,
                video_title=summary.video_title,  # Keep original title
                video_id=summary.video_id,
                executive_summary=translated_executive,
                key_points=translated_key_points,
                timestamps=translated_timestamps,
                conclusion=translated_conclusion,
                action_items=translated_action_items,
                word_count=summary.word_count,  # Keep original count
                estimated_duration_minutes=summary.estimated_duration_minutes,
                language="es",  # Mark as Spanish
                generated_at=datetime.now(),  # New timestamp for translation
            )

            logger.info("Summary translation completed successfully")
            return translated_summary

        except Exception as e:
            logger.error(f"Summary translation failed: {e}")
            raise TranslationError(f"Failed to translate summary: {e}") from e

    def _translate_text_block(self, text: str, block_type: str, video_title: str) -> str:
        """Translate a text block with context awareness.

        Args:
            text: Text to translate
            block_type: Type of block (for context)
            video_title: Video title for additional context

        Returns:
            Translated text in Spanish
        """
        prompt = f"""Translate the following text to Spanish with these requirements:

CONTEXT: This is a {block_type} from a video summary about: "{video_title}"

TRANSLATION RULES:
1. Use NATURAL Spanish (neutral for Spain/Latin America)
2. Preserve technical terms in English when appropriate:
   - Software/tool names (Claude, Gemini, VS Code, Chrome, etc.)
   - Programming terms (API, SDK, plugin, extension, etc.)
   - Product names (Photoshop, Docker, npm, etc.)
3. Adapt expressions idiomatically, not literally
4. Maintain professional but accessible tone
5. Keep it concise and clear
6. If text has bold (**text**), preserve markdown formatting

TEXT TO TRANSLATE:
{text}

OUTPUT: Only the translated text in Spanish, nothing else."""

        try:
            translated = call_llm(prompt=prompt, model=self.model_name)

            if not translated:
                logger.warning(f"Empty translation for {block_type}, using original")
                return text

            return translated.strip()

        except Exception as e:
            logger.warning(f"Translation failed for {block_type}, using original: {e}")
            return text

    def _translate_content(self, content: str, title: str) -> str:
        """Translate script content with context preservation.

        Args:
            content: Original script content in English
            title: Script title for context

        Returns:
            Translated content in Spanish
        """
        prompt = f"""Translate the following YouTube script to Spanish with these requirements:

CONTEXT: This is a technical tutorial script about: "{title}"

TRANSLATION RULES:
1. Use NATURAL Spanish (Spain/Latin America neutral, mainly Spain)
2. Preserve technical terms in English when appropriate:
   - Software names (n8n, npm, Docker, etc.)
   - Programming terms (workflow, API, endpoint, deployment)
   - Commands and code blocks (keep unchanged)
3. Adapt expressions idiomatically, not literally
4. Maintain markdown formatting EXACTLY:
   - Headers (#, ##, ###)
   - Bold (**text**)
   - Lists (-, *)
   - Code blocks (```...```)
5. Adapt CTAs to Spanish style:
   - "Subscribe" → "Suscríbete"
   - "Like this video" → "Dale like si te gustó"
   - "Comment below" → "Déjame un comentario"
6. Keep professional but friendly tone
7. Preserve ALL timestamps if present

SCRIPT TO TRANSLATE:
{content}

OUTPUT: Only the translated script in Spanish, maintaining exact markdown structure."""

        try:
            translated = call_llm(prompt=prompt, model=self.model_name)

            if not translated:
                raise TranslationError("Empty translation response")

            return translated.strip()

        except Exception as e:
            logger.error(f"Content translation failed: {e}")
            # Fallback: return original with warning
            return f"⚠️ Translation failed. Original script below:\n\n{content}"

    def _translate_seo_title(self, title: str) -> str:
        """Translate SEO title to Spanish.

        Args:
            title: Original title in English

        Returns:
            Translated title in Spanish
        """
        prompt = f"""Translate this YouTube video title to Spanish.

RULES:
1. Keep it SEO-friendly and engaging
2. Preserve technical terms (software names, tools)
3. Maximum 100 characters
4. Natural Spanish (neutral for Spain/Latin America)

ORIGINAL TITLE:
{title}

OUTPUT: Only the translated title, nothing else."""

        try:
            translated = call_llm(prompt=prompt, model=self.model_name)

            # Remove quotes if LLM added them
            translated = re.sub(r'^["\'](.*)["\']$', r"\1", translated.strip())

            return translated if translated else title

        except Exception as e:
            logger.warning(f"Title translation failed, using original: {e}")
            return title

    def _translate_seo_description(self, description: str) -> str:
        """Translate SEO description to Spanish.

        Args:
            description: Original description in English

        Returns:
            Translated description in Spanish
        """
        prompt = f"""Translate this YouTube video description to Spanish.

RULES:
1. Keep it engaging and SEO-friendly
2. Preserve technical terms and tool names
3. Natural Spanish (neutral)
4. Maintain similar length (~150-300 chars)

ORIGINAL DESCRIPTION:
{description}

OUTPUT: Only the translated description, nothing else."""

        try:
            translated = call_llm(prompt=prompt, model=self.model_name)

            return translated.strip() if translated else description

        except Exception as e:
            logger.warning(f"Description translation failed, using original: {e}")
            return description

    def _adapt_seo_tags(self, tags: list[str]) -> list[str]:
        """Adapt SEO tags for Spanish audience.

        Args:
            tags: Original tags in English

        Returns:
            Combined list of original + Spanish translated tags
        """
        # Keep all original tags (technical terms are often searched in English)
        adapted_tags = tags.copy()

        # Common translations for generic terms
        translations = {
            "tutorial": "tutorial",
            "guide": "guía",
            "beginner": "principiante",
            "beginners": "principiantes",
            "installation": "instalación",
            "setup": "configuración",
            "how to": "cómo",
            "step by step": "paso a paso",
            "quick start": "inicio rápido",
            "getting started": "primeros pasos",
            "automation": "automatización",
            "workflow": "flujo de trabajo",
            "free": "gratis",
            "local": "local",
            "self-hosted": "auto-alojado",
        }

        # Add Spanish variants for translatable tags
        for tag in tags:
            tag_lower = tag.lower()
            if tag_lower in translations and translations[tag_lower] not in adapted_tags:
                adapted_tags.append(translations[tag_lower])

        # Limit to 30 tags total (YouTube limit)
        return adapted_tags[:30]


__all__ = ["ScriptTranslator", "TranslationError"]
