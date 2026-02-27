"""Generate LinkedIn and Twitter posts from video summaries.

This module extends the summarization pipeline to create social media
ready content for LinkedIn and Twitter/X.
"""

import logging
import re

from core.llm import call_llm
from core.models import VideoSummary
from core.settings import settings
from yt_transcriber.models import LinkedInPost, PostKits, TwitterThread

logger = logging.getLogger(__name__)


class PostKitsError(Exception):
    """Raised when post kits generation fails."""

    pass


def generate_post_kits(
    summary: VideoSummary,
    video_title: str,
    video_url: str,
) -> PostKits:
    """Generate LinkedIn post and Twitter thread from video summary.

    Args:
        summary: VideoSummary object from summarizer
        video_title: Original video title
        video_url: YouTube URL

    Returns:
        PostKits object with LinkedIn and Twitter content

    Raises:
        PostKitsError: If generation or validation fails
    """
    logger.info(f"Generating post kits for: {video_title}")

    # 1. Generate LinkedIn post (English)
    linkedin_post_en = _generate_linkedin_post(summary, video_title)

    # 2. Generate Twitter thread (English)
    twitter_thread_en = _generate_twitter_thread(summary, video_title)

    # 3. Translate to Spanish
    logger.info("Translating post kits to Spanish...")
    try:
        linkedin_post_es = _translate_linkedin_post(linkedin_post_en, video_title)
    except PostKitsError as e:
        logger.warning(f"LinkedIn translation failed, using English version: {e}")
        linkedin_post_es = linkedin_post_en  # Fallback to English

    try:
        twitter_thread_es = _translate_twitter_thread(twitter_thread_en, video_title)
    except PostKitsError as e:
        logger.warning(f"Twitter translation failed, using English version: {e}")
        twitter_thread_es = twitter_thread_en  # Fallback to English

    # 4. Create PostKits object (Spanish version if available, else English)
    post_kits = PostKits(
        linkedin=linkedin_post_es,
        twitter=twitter_thread_es,
        video_title=video_title,
        video_url=video_url,
    )

    # 5. Validate (just warnings, don't fail)
    is_valid, errors = post_kits.validate()
    if not is_valid:
        error_msg = "; ".join(errors)
        logger.warning(f"Post kits validation warnings: {error_msg}")
        logger.info("⚠️ Post kits generated but may need manual review")
    else:
        logger.info("✅ Post kits generated, translated, and validated successfully")

    return post_kits


def _generate_linkedin_post(
    summary: VideoSummary,
    video_title: str,
) -> LinkedInPost:
    """Generate LinkedIn post using Claude CLI.

    Args:
        summary: VideoSummary object
        video_title: Video title

    Returns:
        LinkedInPost object
    """
    logger.info("Generating LinkedIn post...")

    # Build prompt
    prompt = _build_linkedin_prompt(summary, video_title)

    linkedin_text = call_llm(prompt=prompt, model=settings.SUMMARIZER_MODEL)

    logger.debug(f"LinkedIn response: {len(linkedin_text)} chars")

    # Parse response into LinkedInPost
    linkedin_post = _parse_linkedin_response(linkedin_text)

    return linkedin_post


def _generate_twitter_thread(
    summary: VideoSummary,
    video_title: str,
) -> TwitterThread:
    """Generate Twitter thread using Claude CLI.

    Args:
        summary: VideoSummary object
        video_title: Video title

    Returns:
        TwitterThread object
    """
    logger.info("Generating Twitter thread...")

    # Build prompt
    prompt = _build_twitter_prompt(summary, video_title)

    twitter_text = call_llm(prompt=prompt, model=settings.SUMMARIZER_MODEL)

    logger.debug(f"Twitter response: {len(twitter_text)} chars")

    # Parse response into TwitterThread
    twitter_thread = _parse_twitter_response(twitter_text)

    return twitter_thread


def _build_linkedin_prompt(summary: VideoSummary, video_title: str) -> str:
    """Build prompt for LinkedIn post generation."""

    # Extract key info from summary
    executive_summary = summary.executive_summary
    key_points = "\n".join(f"- {point}" for point in summary.key_points)

    prompt = f"""You are a LinkedIn content creator. Generate a post about this video.

Length: 800-1200 characters total (critical)

Video Title: {video_title}

Executive Summary:
{executive_summary}

Key Points:
{key_points}

===== OUTPUT FORMAT (FOLLOW EXACTLY) =====

You MUST use this exact format:

Hook: [One clear statement establishing relevance]

Intro: [1-2 sentences explaining context]

Insight1: 🔹 [Mini subtitle]: [brief explanation]
Insight2: 🔹 [Mini subtitle]: [brief explanation]
Insight3: 🔹 [Mini subtitle]: [brief explanation]
Insight4: 🔹 [Mini subtitle]: [brief explanation] (optional)

WhyItMatters: [Personal reflection or industry context]

CTA: [Natural closing question or reflection]

Tags: #Tag1 #Tag2 #Tag3

===== EXAMPLE OUTPUT =====

Hook: OpenAI Dev Day 2025 has made one thing clear: we're entering a new era for AI creation

Intro: The focus this year was straightforward: make AI accessible to everyone, from developers to non-technical users. And what they've presented promises to change how we work.

Insight1: 🔹 Apps SDK for ChatGPT: now you can create interactive interfaces and connect your products directly inside ChatGPT. A real "app store" for AI is coming
Insight2: 🔹 Visual Agent Builder: a no-code constructor to create agents and intelligent flows, drag-and-drop style, like n8n or Zapier but with AI models
Insight3: 🔹 Codex (GPT-5 version): OpenAI's engineering assistant is now available. Brings integrations, SDK, and team tools. A valuable help for coding faster
Insight4: 🔹 GPT-5 Pro & Real Time Mini: Pro = power and precision for complex tasks. Mini = fluid, natural real-time voice. Two APIs that expand what apps can do

WhyItMatters: This marks the start of Phase 4 of AI: we don't just use it anymore, now we build on it. Developers, creators, and companies can start designing truly intelligent experiences without as many technical barriers

CTA: What tool would you like to try first?

Tags: #OpenAIDevDay #ArtificialIntelligence #AIDevelopment

===== TONE & LENGTH GUIDELINES =====

1. PROFESSIONAL BUT APPROACHABLE:
   - Use affirmative statements ("has made clear" not "just dropped a bomb!")
   - Write like a tech professional, not a press release
   - Balance authority with accessibility
   - Avoid: "OMG!", "This is HUGE!", corporate jargon

2. STRUCTURE & LENGTH:
   - Hook: Clear, confident statement (not question)
   - Context: Broad audience, explain significance
   - Insights: 3-5 bullets with 🔹 + subtitle + explanation
   - Closing: Industry context or personal reflection (NOT "Why it matters:")
   - CTA: Natural, light question or reflection
   - Keep the WHOLE post between 800-1200 characters

3. EXAMPLES OF GOOD TONE:
   ✅ "has clearly set a new benchmark"
   ✅ "the focus was straightforward"
   ✅ "this marks the beginning"
   ✅ "a few days ago we were talking about Claude, but OpenAI isn't staying behind"

   ❌ "just blew my mind!"
   ❌ "Company X has announced"
   ❌ "The future is full of promises"

Now generate the LinkedIn post using the EXACT format above:"""

    return prompt


def _build_twitter_prompt(summary: VideoSummary, video_title: str) -> str:
    """Build prompt for Twitter thread generation."""

    executive_summary = summary.executive_summary
    key_points = "\n".join(f"- {point}" for point in summary.key_points)

    prompt = f"""You are a tech content creator on Twitter/X. Generate a thread about this video.

Target length: 8-12 tweets (critical)

Video Title: {video_title}

Executive Summary:
{executive_summary}

Key Points:
{key_points}

===== OUTPUT FORMAT (FOLLOW EXACTLY) =====

You MUST use this exact format (numbered tweets with "/"):

1. [hook with clear value + emoji]
2. [insight 1 with specific example]
3. [insight 2 with concrete data]
4. [insight 3]
5. [insight 4]
6. [insight 5]
7. [insight 6]
8. [insight 7]
9. [personal take or industry context]
10. [natural CTA + light question]

Hashtags: tag1, tag2, tag3

===== EXAMPLE OUTPUT =====

1. Just watched OpenAI Dev Day 2025. Some updates here could reshape how we build with AI 🚀
2. The no-code agent builder is a big deal. Visual drag-and-drop for complex workflows, like n8n for AI
3. GPT-4 Turbo now has 128K token context. That's the entire Bible or 300 pages processed at once
4. Example: An agent that processes 1000+ files in parallel. What took 20 min now takes 30 sec
5. API costs dropped 50%. This makes ambitious AI projects accessible to startups and enterprises alike
6. New vision capabilities are impressive. GPT can analyze images and understand them deeply
7. Persistent memory is here. Your AI will remember you between conversations, learning preferences
8. 30% faster response times. Your apps and agents will feel snappier and more responsive
9. This feels like Phase 4 of AI: moving from consumption to active building with intelligent agents
10. What tool are you most excited to try? I'm going with the Agent Builder 👇

Hashtags: OpenAIDevDay, AI, GPT4

===== TONE GUIDELINES =====

1. CONVERSATIONAL BUT CREDIBLE:
   - Use first person naturally ("just watched", "this stands out")
   - Include genuine reactions ("impressive", "big deal") without excess
   - Avoid: "OMG!", "🤯🤯🤯", overly hyped language
   - Balance enthusiasm with professionalism

2. EACH TWEET:
   - Length: 200-250 characters (max 280)
   - ONE point per tweet
   - Use "1/", "2/", etc. format
   - Strategic emoji use (1-2 per tweet, relevant only)

3. STORYTELLING:
   - Build coherent narrative, not random bullets
   - Use smooth transitions
   - Include specific data when available
   - Tweet 9-10: Personal take + industry context

4. EXAMPLES OF GOOD TONE:
   ✅ "could reshape how we build"
   ✅ "is a big deal"
   ✅ "this feels like Phase 4"
   ✅ "what tool are you most excited to try?"

   ❌ "Thread about announcements 🧵"
   ❌ "In this thread we'll see..."
   ❌ "Conclusion: The future..."

Now generate the Twitter thread using the EXACT format above:"""

    return prompt


def _parse_linkedin_response(response_text: str) -> LinkedInPost:
    """Parse LLM response into LinkedInPost object.

    Args:
        response_text: Raw text from LLM

    Returns:
        LinkedInPost object

    Raises:
        PostKitsError: If parsing fails completely
    """
    lines = response_text.strip().split("\n")

    data = {}
    insights = []
    raw_content = []  # Fallback: collect all non-label lines

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Hook:"):
            data["hook"] = line.replace("Hook:", "").strip()
        elif line.startswith("Intro:"):
            data["intro"] = line.replace("Intro:", "").strip()
        elif line.startswith("Insight"):
            # Extract insight text after the colon
            if ":" in line:
                insight = line.split(":", 1)[1].strip()
                insights.append(insight)
        elif (
            line.startswith("•")
            or line.startswith("-")
            or line.startswith("*")
            or line.startswith("🔹")
        ):
            # Handle bullet points for insights (used in translations and some responses)
            insight = line.lstrip("•-*🔹").strip()
            if insight and not any(
                insight.lower().startswith(prefix)
                for prefix in ["hook", "intro", "whyitmatters", "cta", "tags"]
            ):
                insights.append(insight)
                raw_content.append(insight)
        elif line.startswith("WhyItMatters:"):
            data["why_it_matters"] = line.replace("WhyItMatters:", "").strip()
        elif line.startswith("CTA:"):
            data["cta"] = line.replace("CTA:", "").strip()
        elif line.startswith("Tags:") or line.startswith("#"):
            tags = line.replace("Tags:", "").strip()
            if tags:
                data["tags"] = tags
        else:
            # Collect non-label content for fallback
            if not any(
                label in line
                for label in ["Hook:", "Intro:", "Insight", "WhyItMatters:", "CTA:", "Tags:"]
            ):
                raw_content.append(line)

    # Strict parsing: require minimum fields and insights
    required_fields = ["hook", "intro", "why_it_matters", "cta"]
    missing = [field for field in required_fields if field not in data]
    if missing:
        raise PostKitsError(f"LinkedIn parsing failed - missing fields: {', '.join(missing)}")

    min_insights = settings.POST_KITS_LINKEDIN_MIN_INSIGHTS
    max_insights = settings.POST_KITS_LINKEDIN_MAX_INSIGHTS
    if len(insights) < min_insights:
        raise PostKitsError(
            f"LinkedIn parsing failed - only {len(insights)} insights found (min {min_insights})"
        )
    if len(insights) > max_insights:
        logger.warning(f"Too many insights ({len(insights)}), truncating to {max_insights}")
        insights = insights[:max_insights]

    # Create LinkedInPost object with fallback values
    return LinkedInPost(
        hook=data.get("hook", "LinkedIn Post"),
        intro=data.get("intro", ""),
        insights=insights,
        why_it_matters=data.get("why_it_matters", ""),
        cta=data.get("cta", "¿Qué opinas?"),
        tags=data.get("tags"),
    )


def _parse_twitter_response(response_text: str) -> TwitterThread:
    """Parse LLM response into TwitterThread object.

    Args:
        response_text: Raw text from LLM

    Returns:
        TwitterThread object

    Raises:
        PostKitsError: If parsing fails
    """
    logger.debug(f"Parsing Twitter response ({len(response_text)} chars)")

    lines = response_text.strip().split("\n")

    tweets = []
    hashtags = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse tweet lines - try multiple patterns
        # Pattern 1: "1. tweet text" (numbered list with dot)
        match = re.match(r"^\d+\.\s*(.+)$", line)
        if match:
            tweet_text = match.group(1).strip()
            tweets.append(tweet_text)
            continue

        # Pattern 2: "1/ tweet text" (Twitter thread style with slash)
        match = re.match(r"^\d+/\s*(.+)$", line)
        if match:
            tweet_text = match.group(1).strip()
            tweets.append(tweet_text)
            continue

        # Pattern 3: "Tweet 1: tweet text"
        match = re.match(r"^Tweet\s+\d+:\s*(.+)$", line, re.IGNORECASE)
        if match:
            tweet_text = match.group(1).strip()
            tweets.append(tweet_text)
            continue

        # Parse hashtags - Format 1: "Hashtags: tag1, tag2"
        if line.lower().startswith("hashtags:"):
            hashtag_text = line.split(":", 1)[1].strip()
            # Split by comma or space
            hashtags = [
                tag.strip().replace("#", "")
                for tag in re.split(r"[,\s]+", hashtag_text)
                if tag.strip() and tag.strip() != "#"
            ]
            continue

        # Parse hashtags - Format 2: "#Python #FastAPI #Scalable"
        if line.startswith("#"):
            hashtag_tags = re.findall(r"#(\w+)", line)
            if hashtag_tags:
                hashtags = hashtag_tags

    logger.debug(f"Parsed {len(tweets)} tweets and {len(hashtags)} hashtags")

    # Validate
    if len(tweets) < 8:
        logger.error(f"Twitter parsing failed - response text:\n{response_text}")
        raise PostKitsError(f"Twitter parsing failed - only {len(tweets)} tweets found (min 8)")

    # Create TwitterThread object
    return TwitterThread(tweets=tweets, hashtags=hashtags)


# ========================================
# Translation Functions (EN → ES)
# ========================================


def _translate_linkedin_post(
    post_en: LinkedInPost,
    video_title: str,
) -> LinkedInPost:
    """Translate LinkedIn post from English to Spanish.

    Args:
        post_en: LinkedInPost in English
        video_title: Video title for context

    Returns:
        LinkedInPost in Spanish

    Raises:
        PostKitsError: If translation fails
    """
    logger.info("Translating LinkedIn post to Spanish...")

    # Build translation prompt
    prompt = f"""Translate this LinkedIn post from English to Spanish with a PROFESSIONAL yet CONVERSATIONAL tone suitable for LinkedIn.

**VIDEO TITLE**: {video_title}

**ORIGINAL POST (ENGLISH)**:
Hook: {post_en.hook}

Intro: {post_en.intro}

Insights:
{chr(10).join(f"• {insight}" for insight in post_en.insights)}

Why it matters: {post_en.why_it_matters}

CTA: {post_en.cta}

Tags: {post_en.tags or ""}

**CRITICAL TRANSLATION REQUIREMENTS**:

1. PROFESSIONAL LINKEDIN TONE (NOT casual influencer):
   - Use natural Spanish suitable for tech professionals
   - Balance authority with approachability
   - Avoid excessive colloquialisms ("flipado", "brutal", "me ha volado la cabeza")
   - Prefer: "ha dejado claro", "destaca", "promete cambiar", "una ayuda valiosa"

2. PRESERVE STRUCTURE:
   - Keep exact same structure (Hook, Intro, Insights with 🔹, Closing reflection, CTA)
   - Maintain bullet points with 🔹 emoji
   - Keep mini subtitles format: "🔹 Subtitle: explanation"

3. TECHNICAL TERMS:
   - Preserve in English when standard: API, SDK, app store, no-code, drag-and-drop
   - Translate when natural: "desarrolladores", "equipos", "herramientas"

4. HASHTAGS:
   - Translate to Spanish equivalents
   - Use clear, professional tags: #InteligenciaArtificial #DesarrolloIA

5. CONCISENESS (Spanish tends to be longer):
   - Maximum 1000 characters total
   - Use shorter synonyms: "ahora" instead of "en este momento"
   - Remove filler words: "realmente", "básicamente", "prácticamente"

6. STYLE EXAMPLES:
   ❌ AVOID: "¡Acabo de ver el DevDay y me ha flipado!"
   ✅ USE: "OpenAI Dev Day 2025 ha dejado claro algo: estamos entrando en una nueva era"

   ❌ AVOID: "Esto es brutal para la productividad"
   ✅ USE: "Una ayuda valiosa para programar más rápido y mejor"

   ❌ AVOID: "¿Qué te ha parecido más increíble?"
   ✅ USE: "¿Qué herramienta te gustaría probar primero?"

7. CLOSING REFLECTION (CRITICAL - NO "Why it matters:" label):
   - NEVER use "Why it matters:" as a label
   - Integrate the reflection naturally into the text
   - Prefer industry context: "Hace unos días hablábamos de Claude, pero OpenAI no se quiere quedar atrás. Esto marca un punto de inflexión en la accesibilidad de la IA"
   - Or personal take: "Creo que esto marca el inicio de una nueva fase: ya no solo usamos la IA, ahora construimos sobre ella"
   - Keep it conversational and professional, not preachy

**OUTPUT FORMAT** (exact format required):
Hook: [Spanish translation - clear, confident statement]

Intro: [Spanish translation - 1-2 sentences with context]

Insights:
• [First insight in Spanish - 🔹 Subtitle: explanation]
• [Second insight in Spanish - 🔹 Subtitle: explanation]
• [Continue for all insights]

WhyItMatters: [Spanish translation - BUT NEVER write "Why it matters:" - just the text naturally]

CTA: [Spanish translation - natural closing, light question or reflection]

Tags: [Spanish hashtags separated by commas]

CRITICAL: Preserve professional-yet-approachable tone. Avoid sounding like press release OR overly casual influencer."""

    linkedin_text = call_llm(prompt=prompt, model=settings.SUMMARIZER_MODEL)

    logger.debug(f"LinkedIn translation response: {len(linkedin_text)} chars")

    # Parse response
    try:
        post_es = _parse_linkedin_response(linkedin_text)
        logger.info("✅ LinkedIn post translated successfully")
        return post_es
    except PostKitsError as e:
        logger.error(f"LinkedIn translation parsing failed: {e}")
        raise


def _translate_twitter_thread(
    thread_en: TwitterThread,
    video_title: str,
) -> TwitterThread:
    """Translate Twitter thread from English to Spanish.

    Args:
        thread_en: TwitterThread in English
        video_title: Video title for context

    Returns:
        TwitterThread in Spanish

    Raises:
        PostKitsError: If translation fails
    """
    logger.info("Translating Twitter thread to Spanish...")

    # Build translation prompt
    numbered_tweets = "\n".join(f"{i + 1}. {tweet}" for i, tweet in enumerate(thread_en.tweets))

    prompt = f"""Translate this Twitter thread from English to Spanish with a CONVERSATIONAL yet PROFESSIONAL tone suitable for Twitter/X.

**VIDEO TITLE**: {video_title}

**ORIGINAL THREAD (ENGLISH)**:
{numbered_tweets}

Hashtags: {", ".join("#" + tag for tag in thread_en.hashtags)}

**CRITICAL TRANSLATION REQUIREMENTS**:

1. TWITTER-APPROPRIATE TONE (professional but engaging):
   - Use natural, engaging Spanish suitable for tech community
   - Balance enthusiasm with credibility
   - Avoid excessive colloquialisms ("flipado", "me ha volado la cabeza", "esto es una locura")
   - Prefer: "destaca", "impresionante", "cambia las reglas", "vale la pena"

2. CHARACTER LIMITS:
   - **CRITICAL**: Keep each tweet under 250 characters in Spanish
   - Spanish is longer than English - be concise
   - Remove filler words: "realmente", "básicamente", "literalmente"
   - Use contractions where natural

3. TECHNICAL TERMS:
   - Preserve in English when standard: API, SDK, no-code, drag-and-drop, real-time
   - Translate: "desarrolladores" → "devs" (Twitter style)

4. EMOJIS:
   - Maintain in same positions
   - Don't add extra emojis
   - Keep professional (avoid 🤯🔥😅 overuse)

5. HASHTAGS:
   - Translate to Spanish equivalents
   - Keep professional: #OpenAI #InteligenciaArtificial #DesarrolloIA

6. NARRATIVE FLOW:
   - Preserve thread's story arc
   - Use natural connectors: "y lo mejor", "además", "aquí viene lo interesante"
   - Maintain momentum and interest

7. STYLE EXAMPLES:
   ❌ AVOID: "Acabo de ver el DevDay de OpenAI y mi cabeza ha explotado 🤯"
   ✅ USE: "OpenAI Dev Day 2025 trae novedades que pueden cambiar cómo construimos con IA"

   ❌ AVOID: "Esto es BRUTAL para proyectos masivos"
   ✅ USE: "Ideal para proyectos grandes con gran ahorro y eficiencia"

   ❌ AVOID: "¿Cuál te ha volado más la cabeza? Yo todavía estoy procesando..."
   ✅ USE: "¿Qué novedad te gustaría probar primero? Yo voy con el Agent Builder"

8. CLOSING TWEET (professional but friendly):
   - Use natural CTAs: "¿Qué opinas?", "¿Lo usarías en producción?"
   - Light personal touch: "Me leo 👇", "Comparte tu experiencia"
   - Avoid generic: "Comparte tus pensamientos"

**OUTPUT FORMAT** (exact format required):
1. [First tweet in Spanish - under 250 chars]
2. [Second tweet in Spanish - under 250 chars]
3. [Continue for all {len(thread_en.tweets)} tweets]

Hashtags: [Spanish hashtags separated by commas]

CRITICAL: Maintain professional tech community tone. Avoid sounding overly hyped OR too formal."""

    twitter_text = call_llm(prompt=prompt, model=settings.SUMMARIZER_MODEL)

    logger.debug(f"Twitter translation response: {len(twitter_text)} chars")

    # Parse response
    try:
        thread_es = _parse_twitter_response(twitter_text)
        logger.info("✅ Twitter thread translated successfully")
        return thread_es
    except PostKitsError as e:
        logger.error(f"Twitter translation parsing failed: {e}")
        raise
