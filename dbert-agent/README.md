[![DBERT Internship & Fellowship Program](https://img.shields.io/badge/DBERT-Internship%20%26%20Fellowship%20Program-1F3B2C?style=for-the-badge)](https://internship.dbert.online)

# DBERT Agent

**A local-first, privacy-first AI assistant that runs entirely on your laptop** — chat, voice, document RAG, web research, file tools, and task automation, all powered by a local model through [LM Studio](https://lmstudio.ai/) (with optional cloud fallback). No data leaves the machine unless you explicitly turn on a web-search tool call.

This project was built end-to-end by a fellow of the **[DBERT Internship & Fellowship Program](https://internship.dbert.online)** — DBERT's paid, proof-of-work-based track that puts engineers on real, deployable systems instead of tutorials and sandbox exercises. It's shared here as a real example of what fellows ship.

---

## About the DBERT Internship & Fellowship Program

Most internships hand out certificates for watching videos. DBERT does the opposite: fellows spend 10–20 hours a week shipping live, production-style software — real repos, real URLs, real accountability — with stipends tied to completed sprints, not attendance.

This repository is one such output: a full local AI-agent stack (provider abstraction, RAG pipeline, voice I/O, web research, scheduling/automation, and a desktop UI) built by a fellow during the program.

- **Apply for the Launchpad or Fellowship:** [internship.dbert.online](https://internship.dbert.online)
- **See DBERT's own production AI product this pipeline informed:** [DBERT Chat](https://dbert.online/ai-solutions/products/dbert-chat)

If you're evaluating whether this program produces real engineers, this codebase is the evidence — read the code, run it, judge it on its own merits.

---

## What it does

- **Local-first chat** against any model loaded in LM Studio (or Ollama-style OpenAI-compatible endpoints), with optional OpenAI / Anthropic / Gemini cloud fallback via [LiteLLM](https://github.com/BerriAI/litellm).
- **Document RAG** — ingest PDFs/text files, chunk + embed them into a local vector store, and query them with citation-grounded answers.
- **Persistent local memory** — every session is stored in SQLite and semantically searchable later ("what did I decide about X last week?").
- **Web & local deep research** — multi-step research mode that decomposes a query, gathers sources, and synthesizes a cited report; a local variant scopes the same pipeline to your own documents/history instead of the web.
- **Voice mode** — offline speech-to-text (Whisper) and text-to-speech (Piper) for a hands-free loop.
- **Tool layer with permissions** — every file write, shell command, or web call goes through an explicit per-tool permission level (off / ask every time / ask once / always allow), plus MCP client support to plug in external tool servers.
- **Automation** — scheduled jobs and URL monitors that run the agent loop headlessly on a timer.
- **Desktop GUI** (Tkinter) alongside a full CLI/REPL mode for terminal-first use.

Full product/architecture rationale is in [`DBERT_Agent_Spec.md`](DBERT_Agent_Spec.md); the module-by-module build log is in [`DBERT_Development_Guide.md`](DBERT_Development_Guide.md).

## Status

Verified before publishing: every module compiles and imports cleanly, and the CLI runs end-to-end — including its no-provider-configured path, which drops into a working offline console with clear setup guidance rather than crashing. RAG ingestion and chat naturally require a live model provider (see Setup below); without one they fail with a clear error message, as designed. There is currently no automated test suite (`pytest` is listed as a dependency for future use).

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running locally with a model loaded (recommended default; any OpenAI-compatible local server works), **or** an API key for OpenAI/Anthropic/Gemini
- OS packages for voice mode: `portaudio` (e.g. `apt install libportaudio2` / `brew install portaudio`)
- OS packages for the desktop GUI: `tkinter` (e.g. `apt install python3-tk`; bundled with most Python installers on macOS/Windows)

## Setup

```bash
git clone https://github.com/DBERT-INDIA/dbert-agent.git
cd dbert-agent
python -m venv venv
source venv/bin/activate      # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Start LM Studio, load a model (e.g. `Qwen2.5-0.5B-Instruct`), and start its local server on port 1234 — this is DBERT's default provider and needs no API key. Alternatively, register a cloud provider from inside DBERT with `/provider add <name> <api_key>`.

## Running it

```bash
python -m src.main
```

Run from the repository root (not from inside `src/`) — the codebase uses absolute `src.*` imports.

- Launches the desktop GUI automatically when run interactively.
- Pass `--cli` to force the terminal REPL instead.
- With no model provider configured, it drops into an offline console where you can still inspect `/help`, `/permissions`, and `/provider`.

### Key commands

| Command | Description |
|---|---|
| `/model` | Switch the active model |
| `/provider add <name> <key>` | Register a cloud provider |
| `/ingest <file>` | Ingest a PDF/text file into the local vector store |
| `/rag <query>` | Query the local vector store directly |
| `/research web <query>` | Multi-step web deep-research |
| `/voice` | Enter hands-free voice mode |
| `/schedule add <name> <url> <interval>` | Add a monitoring/automation job |
| `/mcp add <command>` | Connect a local MCP tool server |
| `/permissions` | View current tool permission levels |
| `/exit` | Save and quit |

## Project structure

```
src/
├── main.py                  # CLI/GUI entry point, command loop
├── core/                    # config, hardware profiling, providers, model registry, sessions
├── rag/                     # PDF parsing, ingestion, embeddings, vector store, summarization
├── research/                # web and local deep-research pipelines
├── memory/                  # semantic chat-history search
├── voice/                   # Whisper STT, Piper TTS, voice loop controller
├── tools/                   # tool registry, permission levels, MCP client
├── automation/               # scheduler and URL monitor jobs
└── ui/                       # Tkinter desktop GUI
```

## Local-first by design

All inference, embeddings, vector storage, and chat history stay on-device by default. The only network calls are: (1) your chosen LLM provider's API, if you configure a cloud one instead of local LM Studio, and (2) explicit web-search/deep-research tool calls, which are gated behind the permission system like any other tool.

## License

MIT — see [LICENSE](LICENSE).

## Program links

- Internship & Fellowship applications: **[internship.dbert.online](https://internship.dbert.online)**
- DBERT Chat product page: **[dbert.online/ai-solutions/products/dbert-chat](https://dbert.online/ai-solutions/products/dbert-chat)**
- Contact: contactus@dbert.online
