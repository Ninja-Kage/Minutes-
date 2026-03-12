# Recall AI

**Intelligent meeting transcription and retrieval system with speaker diarization and natural language querying.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-PEP8-black.svg)](https://peps.python.org/pep-0008/)

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Recall AI is a local-first meeting intelligence system that captures audio, identifies individual speakers, transcribes speech to text, and enables natural language querying across all past meetings. It is designed for teams that need a private, cost-effective alternative to cloud meeting tools like Otter.ai or Fireflies.

The system runs entirely on CPU, requires no GPU, and stores meeting data persistently in Pinecone for cross-session retrieval.

---

## Demo

![Recall AI Screenshot](assets/demo.png)

**Example queries you can run after a meeting:**
- *"What action items were assigned and to whom?"*
- *"What did the team decide about the product launch?"*
- *"Summarize everything SPEAKER_01 said."*
- *"Were there any blockers mentioned?"*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Recall AI                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Microphone Input                                          │
│        │                                                    │
│        ▼                                                    │
│   ┌─────────────┐                                          │
│   │ recorder.py │  sounddevice → 16kHz mono WAV            │
│   └──────┬──────┘                                          │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────────┐                                     │
│   │ transcriber.py   │                                     │
│   │                  │                                     │
│   │  WhisperX        │ → Speech-to-text (CPU, int8)        │
│   │  Pyannote 3.1    │ → Speaker diarization               │
│   │  Overlap Match   │ → Assign speakers to segments       │
│   └──────┬───────────┘                                     │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────────┐                                     │
│   │    rag.py        │                                     │
│   │                  │                                     │
│   │  MiniLM-L6-v2    │ → Text embeddings (local)           │
│   │  Pinecone        │ → Vector storage (namespace/meeting)│
│   │  Groq / Llama3   │ → LLM answer generation             │
│   └──────┬───────────┘                                     │
│          │                                                  │
│          ▼                                                  │
│   ┌──────────────────┐                                     │
│   │    app.py        │ → Streamlit UI                      │
│   └──────────────────┘                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Audio Capture → WAV File → Transcription → Diarization → 
Labeled Segments → Chunking → Embedding → Pinecone Storage → 
User Query → Embedding → Vector Search → LLM → Answer
```

---

## Features

| Feature | Description |
|---|---|
| 🎤 Live Recording | Capture meeting audio directly from microphone |
| 👥 Speaker Diarization | Automatically identify and separate speakers |
| 📝 Transcription | Accurate speech-to-text via WhisperX |
| 🔍 Semantic Search | Query meetings using natural language |
| 🗂️ Meeting History | All meetings persisted and queryable across sessions |
| ✏️ Speaker Renaming | Map SPEAKER_00 labels to real names |
| ☁️ Cloud Storage | Pinecone vector DB with per-meeting namespaces |
| 💬 Conversational Q&A | Chat-style interface for asking questions |

---

## Tech Stack

| Layer | Technology | Reason |
|---|---|---|
| UI | Streamlit | Rapid prototyping, Python-native |
| Audio Capture | sounddevice | Low-latency microphone access |
| Speech-to-Text | WhisperX | Faster-whisper backend, CPU-optimized |
| Speaker Diarization | Pyannote Audio 3.1 | State-of-the-art diarization |
| Embeddings | all-MiniLM-L6-v2 | Fast, local, no API cost |
| Vector Database | Pinecone | Persistent, scalable, managed |
| LLM | Groq (Llama 3.1 8B) | Fast inference, free tier |
| RAG Framework | LangChain | Modular retrieval pipeline |
| Environment | python-dotenv | Secure credential management |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- `pip` or `uv` package manager
- A HuggingFace account with model access granted for:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- API keys for:
  - [Groq](https://console.groq.com) — free tier available
  - [Pinecone](https://app.pinecone.io) — free tier available

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/recall-ai.git
cd recall-ai

# 2. Create a virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\activate

# Activate — Mac/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root. **Never commit this file.**

```bash
cp .env.example .env
```

Fill in your credentials:

```env
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
HF_TOKEN=your_huggingface_token_here
```

> ⚠️ The `.env` file is listed in `.gitignore` and will not be pushed to GitHub.

---

## Usage

### Starting the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Recording a Meeting

1. Click **⏺ Start Recording** — audio capture begins immediately
2. Conduct your meeting normally
3. Click **⏹ Stop & Transcribe** when finished
4. Wait for transcription and diarization to complete (~30–60s for a 10-minute meeting on CPU)

### Viewing the Transcript

- Each speaker is color-coded and labeled (`SPEAKER_00`, `SPEAKER_01`, etc.)
- Timestamps are shown per segment
- Use the **sidebar** to rename speakers to real names

### Querying the Meeting

Type any natural language question in the chat panel:

```
What tasks were assigned?
Who is responsible for the backend?
What was decided about the Q3 deadline?
Summarize the key discussion points.
```

Toggle **All Meetings** to search across your entire meeting history.

---

## Project Structure

```
recall-ai/
├── app.py              # Streamlit UI and application logic
├── recorder.py         # Audio capture using sounddevice
├── transcriber.py      # WhisperX transcription + Pyannote diarization
├── rag.py              # Pinecone vector store + Groq RAG pipeline
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

---

## Design Decisions

**Why WhisperX over vanilla Whisper?**
WhisperX uses faster-whisper under the hood which provides 4x speedup on CPU via CTranslate2 quantization. `int8` compute type further reduces memory usage with minimal accuracy loss.

**Why Pyannote for diarization?**
Pyannote 3.1 is the current state-of-the-art for open-source speaker diarization. It outperforms alternatives like SpeechBrain on standard benchmarks (CALLHOME, AMI corpus).

**Why Pinecone over FAISS?**
FAISS requires managing local index files and doesn't support multi-session persistence out of the box. Pinecone provides a managed, cloud-hosted solution with namespace support — allowing all meetings to live in a single index while remaining independently queryable.

**Why namespaces instead of separate indexes?**
Pinecone's free tier limits users to 5 serverless indexes. Using a single `meetings` index with one namespace per meeting removes this constraint entirely and scales to unlimited meetings.

**Why all-MiniLM-L6-v2 for embeddings?**
It runs locally with no API cost, produces 384-dimensional vectors suitable for semantic similarity tasks, and has strong performance on sentence-level retrieval benchmarks (MTEB). For a meeting transcript use case, the accuracy is more than sufficient.

---

## Known Limitations

- **CPU only** — transcription is slower without GPU (approx. 0.3x real-time on base model)
- **Short recordings** — Pyannote requires 30+ seconds of audio for reliable diarization
- **Speaker accuracy** — degrades with 5+ speakers or overlapping speech
- **Language** — optimized for English; other languages supported by Whisper but untested
- **Audio quality** — background noise significantly impacts transcription accuracy

---

## Roadmap

- [ ] GPU support via `device="cuda"` flag
- [ ] Upload existing audio files for transcription
- [ ] Auto-generated meeting summary on transcription complete
- [ ] Export transcript as formatted PDF
- [ ] Speaker recognition across meetings (same person, different meetings)
- [ ] REST API layer using FastAPI
- [ ] Docker container for one-command setup
- [ ] Evaluation pipeline with WER benchmarking

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure your code follows PEP8 style guidelines and includes relevant comments.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">Built with Python · WhisperX · Pyannote · LangChain · Pinecone · Groq</p>
