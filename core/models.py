"""Shared data models for YouTube Content Suite.

Cross-pipeline models used by multiple modules (transcriber, script generator).
These models represent core concepts that are shared across different features.
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TimestampedSection:
    """Sección del video con timestamp y descripción."""

    timestamp: str  # Formato: "MM:SS" o "HH:MM:SS"
    description: str
    importance: int = 3  # 1-5, donde 5 es más importante

    def __str__(self) -> str:
        """Format as markdown list item."""
        return f"- **{self.timestamp}** - {self.description}"


@dataclass
class VideoSummary:
    """Resumen completo de un video de YouTube generado con IA."""

    # Video metadata
    video_url: str
    video_title: str
    video_id: str

    # Summary content
    executive_summary: str  # 2-3 líneas
    key_points: list[str]  # 5-7 bullets
    timestamps: list[TimestampedSection]  # 5-10 momentos clave
    conclusion: str  # 1-2 líneas
    action_items: list[str]  # 3-5 acciones

    # Statistics
    word_count: int
    estimated_duration_minutes: float

    # Metadata
    language: str  # 'es' or 'en'
    generated_at: datetime

    def to_markdown(self) -> str:
        """Convert summary to formatted markdown document."""
        # Header
        md = f"# 📹 Resumen: {self.video_title}\n\n"
        md += f"**🔗 Video**: {self.video_url}\n"
        md += f"**📅 Generado**: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md += "---\n\n"

        # Executive Summary
        md += "## 🎯 Resumen Ejecutivo\n\n"
        md += f"{self.executive_summary}\n\n"

        # Key Points
        md += "## 🔑 Puntos Clave\n\n"
        for i, point in enumerate(self.key_points, 1):
            md += f"{i}. {point}\n"
        md += "\n"

        # Timestamps
        if self.timestamps:
            md += "## ⏱️ Momentos Importantes\n\n"
            for ts in self.timestamps:
                md += f"{ts}\n"
            md += "\n"

        # Conclusion
        md += "## 💡 Conclusión\n\n"
        md += f"{self.conclusion}\n\n"

        # Action Items
        if self.action_items:
            md += "## ✅ Action Items\n\n"
            for i, item in enumerate(self.action_items, 1):
                md += f"{i}. {item}\n"
            md += "\n"

        # Footer statistics
        md += "---\n\n"
        md += f"**📊 Estadísticas**: {self.word_count:,} palabras | "
        md += f"~{self.estimated_duration_minutes:.1f} minutos de contenido\n"

        return md

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "video_url": self.video_url,
            "video_title": self.video_title,
            "video_id": self.video_id,
            "executive_summary": self.executive_summary,
            "key_points": self.key_points,
            "timestamps": [
                {
                    "timestamp": ts.timestamp,
                    "description": ts.description,
                    "importance": ts.importance,
                }
                for ts in self.timestamps
            ],
            "conclusion": self.conclusion,
            "action_items": self.action_items,
            "word_count": self.word_count,
            "estimated_duration_minutes": self.estimated_duration_minutes,
            "language": self.language,
            "generated_at": self.generated_at.isoformat(),
        }


__all__ = ["TimestampedSection", "VideoSummary"]
