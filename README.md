# YouTube Video Transcriber & Summarizer

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.8-red.svg)](https://pytorch.org/)

**Transform YouTube videos into searchable text, AI summaries, and social media content.**

## Features

- **CUDA-accelerated transcription** with OpenAI Whisper
- **Multi-language support** with auto-detection
- **AI-powered summaries** (EN + ES) with Gemini 2.5 Flash:
  - Executive summary (2-3 sentences)
  - Key points (5-10 bullets)
  - Smart timestamps (inferred from content)
  - Action items (when applicable)
- **Social Media Post Kits**: Auto-generate LinkedIn posts + Twitter threads
- **Multiple input sources**: YouTube, Google Drive, local files

## Quick Start

```bash
# Clone the repository
git clone https://github.com/sergiomarquezdev/yt-transcriber.git
cd yt-transcriber

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

# Transcribe your first video
yt-transcriber transcribe --url "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Prerequisites

- **Python 3.13+**
- **FFmpeg** - Required for audio processing
- **CUDA 12.8** (Optional) - For GPU acceleration

### FFmpeg Installation

**Windows:**
```powershell
# Download from https://github.com/BtbN/FFmpeg-Builds/releases
# Extract to C:\ffmpeg and add to PATH
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\ffmpeg\bin", [EnvironmentVariableTarget]::User)
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
```

## Usage

```bash
# Only transcription (DEFAULT)
yt-transcriber transcribe --url "https://www.youtube.com/watch?v=VIDEO_ID"

# Transcription + AI summaries (EN + ES)
yt-transcriber transcribe --url "URL" --summarize

# Transcription + summaries + Post Kits (LinkedIn + Twitter)
yt-transcriber transcribe --url "URL" --post-kits

# Force Spanish transcription
yt-transcriber transcribe --url "URL" --language es

# Local file
yt-transcriber transcribe --url "path/to/video.mp4" --summarize
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--url` | `-u` | YouTube URL, Google Drive URL, or local file path |
| `--language` | `-l` | Language code (`en`, `es`) - auto-detect if omitted |
| `--summarize` | | Generate AI summaries (EN + ES) |
| `--post-kits` | | Generate LinkedIn + Twitter content (implies --summarize) |
| `--ffmpeg-location` | | Custom FFmpeg path |

## Output

Files are saved to `output/`:

```
output/
├── transcripts/
│   └── {title}_vid_{id}.txt          # Raw transcription
└── summaries/
    ├── {title}_vid_{id}_summary_EN.md  # English summary
    ├── {title}_vid_{id}_summary_ES.md  # Spanish summary
    └── {title}_vid_{id}_post_kits.md   # LinkedIn + Twitter content
```

### AI Summary Contents

- Executive summary
- Key points (5-7 bullets)
- Timestamps (5-8 important moments)
- Action items

### Post Kits Contents

- **LinkedIn post** (800-1200 chars): Professional hook, insights, CTA, hashtags
- **Twitter thread** (8-12 tweets): Numbered tweets, max 280 chars each

## Configuration

Create `.env` from template:

```bash
# Whisper Model
WHISPER_MODEL_NAME=base    # tiny, base, small, medium, large
WHISPER_DEVICE=cuda        # cuda or cpu

# AI Summarizer (required for summaries)
GOOGLE_API_KEY=your_key    # Get from https://aistudio.google.com/apikey

# Directories
TEMP_DOWNLOAD_DIR=temp_files/
OUTPUT_TRANSCRIPTS_DIR=output/transcripts/
SUMMARY_OUTPUT_DIR=output/summaries/

# Logging
LOG_LEVEL=INFO
```

### Model Selection

| Model | Speed | Accuracy | VRAM | Use Case |
|-------|-------|----------|------|----------|
| `tiny` | Fast | Low | ~1GB | Quick drafts |
| `base` | Good | Medium | ~1GB | **Default - Balanced** |
| `small` | Medium | Good | ~2GB | Better quality |
| `medium` | Slow | High | ~5GB | High accuracy |
| `large` | Slowest | Best | ~10GB | Best quality |

## Programmatic Usage

```python
from yt_transcriber.cli import run_transcribe_command

# Only transcription (default)
transcript, _, _, _ = run_transcribe_command(url="path/to/video.mp4")

# With summaries
transcript, summary_en, summary_es, _ = run_transcribe_command(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    generate_summary=True,
)

# With post kits (implies summary)
transcript, summary_en, summary_es, post_kits = run_transcribe_command(
    url="https://www.youtube.com/watch?v=VIDEO_ID",
    generate_post_kits=True,
)
```

## Troubleshooting

**FFmpeg not found:**
```bash
# Use direct path
yt-transcriber transcribe --url "URL" --ffmpeg-location "C:\ffmpeg\bin\ffmpeg.exe"
```

**CUDA not available:**
```bash
# Check installation
python -c "import torch; print(torch.cuda.is_available())"

# Fall back to CPU in .env
WHISPER_DEVICE=cpu
```

**Out of memory:**
```bash
# Use smaller model in .env
WHISPER_MODEL_NAME=tiny
```

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - AI transcription
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - Video downloading
- [Google Gemini](https://ai.google.dev/) - AI summarization

---

Made with care by [Sergio Marquez](https://github.com/sergiomarquezdev)
