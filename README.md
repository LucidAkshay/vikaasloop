<div align="center">

[![GitHub stars](https://img.shields.io/github/stars/LucidAkshay/vikaasloop?style=social)](https://github.com/LucidAkshay/vikaasloop/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/LucidAkshay/vikaasloop)](https://github.com/LucidAkshay/vikaasloop/issues)
[![GitHub forks](https://img.shields.io/github/forks/LucidAkshay/vikaasloop?style=social)](https://github.com/LucidAkshay/vikaasloop/network/members)
[![GitHub license](https://img.shields.io/github/license/LucidAkshay/vikaasloop)](https://github.com/LucidAkshay/vikaasloop/blob/main/LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat)](https://github.com/LucidAkshay/vikaasloop/blob/main/CONTRIBUTING.md)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-%23db61a2.svg?&logo=github&logoColor=white)](https://github.com/sponsors/LucidAkshay)
[![Reddit](https://img.shields.io/badge/Reddit-AkshayCodes-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/user/AkshayCodes/)

<br/>

```text
 ██╗   ██╗██╗██╗  ██╗ █████╗  █████╗ ███████╗██╗      ██████╗  ██████╗ ██████╗
 ██║   ██║██║██║ ██╔╝██╔══██╗██╔══██╗██╔════╝██║      ██╔═══██╗██╔═══██╗██╔══██╗
 ██║   ██║██║█████╔╝ ███████║███████║███████╗██║      ██║   ██║██║   ██║██████╔╝
 ╚██╗ ██╔╝██║██╔═██╗ ██╔══██║██╔══██║╚════██║██║      ██║   ██║██║   ██║██╔═══╝
  ╚████╔╝ ██║██║  ██╗██║  ██║██║  ██║███████║███████╗╚██████╔╝╚██████╔╝██║
   ╚═══╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝
```

**VikaasLoop — The Self-Improving LLM Fine-Tuning Engine**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-00539B?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?style=flat&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=flat&logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Google Gemini](https://img.shields.io/badge/Google_Gemini-4285F4?style=flat&logo=google&logoColor=white)](https://ai.google.dev/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow)](https://huggingface.co/)

The first open-source tool that closes the full loop between data generation, model training, quality evaluation, and strategy learning — autonomously, iteratively, and for free.

*"Every other fine-tuning tool is a one-shot instrument. VikaasLoop is a research institution that fits on a laptop."*

</div>

## Table of Contents
- [What Is VikaasLoop?](#what-is-vikaasloop)
- [The Problem We Solve](#the-problem-we-solve)
- [Why Nothing Else Does This](#why-nothing-else-does-this)
- [The Core Innovation — Skills Library](#the-core-innovation--skills-library)
- [How It Works — The 5-Agent Loop](#how-it-works--the-5-agent-loop)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Zero Cost Infrastructure](#zero-cost-infrastructure)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Roadmap](#roadmap)
- [Impact — Who Benefits](#impact--who-benefits)
- [Contributing](#contributing)
- [License](#license)

## What Is VikaasLoop?

VikaasLoop (विकास = growth / development in Hindi and Sanskrit) is an autonomous, self-improving LLM fine-tuning engine that runs entirely on your local machine.

You give it three things:

```plaintext
1. A task description  →  "Make this model better at explaining Rust concepts"
2. A base model        →  microsoft/phi-2
3. A quality target    →  75% win rate vs base model
```

VikaasLoop does the rest — automatically, in a loop, getting smarter with each iteration:

```plaintext
┌─────────────────────────────────────────────────────────────────┐
│                     THE VIKAASLOOP CYCLE                        │
│                                                                 │
│   Skills Library        DataGen Agent        Training Agent     │
│   ┌──────────┐          ┌──────────┐         ┌──────────┐       │
│   │ What     │ ──hint──▶│ Generate │ ──data──▶│ Fine-   │       │
│   │ worked   │          │ training │         │ tune with│       │
│   │ before   │          │ data     │         │ LoRA     │       │
│   └──────────┘          └──────────┘         └──────────┘       │
│        ▲                                           │            │
│        │ update score                              adapter      │
│        │                                           │            │
│   ┌──────────┐                              ┌──────────┐        │
│   │ Gemini   │◀──── score ───────────────── │ Eval     │        │
│   │ Judge    │                              │ Agent    │        │
│   │ (LLM)    │                              │          │        │
│   └──────────┘                              └──────────┘        │
│                                                                 │
│   Repeat until win rate ≥ target OR max iterations reached      │
└─────────────────────────────────────────────────────────────────┘
```

The loop runs until your model reaches the quality you want — or until you stop it.

## The Problem We Solve

Fine-tuning a language model today requires:

| Step | Who does it today | Time cost |
| :--- | :--- | :--- |
| Curate training data | You, manually | Days to weeks |
| Write quality training examples | You or contractors | Hours per batch |
| Decide if training worked | You, subjectively | Per run |
| Figure out why it didn't work | Trial and error | Weeks |
| Try again with a new strategy | You, from scratch | Repeat everything |
| Remember what worked last time | Spreadsheets, if you're lucky | Organizational debt |

The result: A PhD student runs the same fine-tuning experiment 200 times with minor variations. A startup hires a machine learning engineer just to run this manual loop. A researcher in an emerging market simply cannot participate — the tooling assumes you have a team.

**VikaasLoop automates the entire loop.** You press start once. You walk away. You come back to a fine-tuned model and a record of exactly what strategies improved it.

## Why Nothing Else Does This

We have studied every major fine-tuning tool available as of 2026. Not one of them does all four of the things VikaasLoop does simultaneously:

| Tool | Auto data gen | LLM-as-judge eval | Strategy memory | Autonomous loop |
| :--- | :---: | :---: | :---: | :---: |
| Axolotl | ❌ | ❌ | ❌ | ❌ |
| LLaMA Factory | ❌ | ❌ | ❌ | ❌ |
| HF AutoTrain | ❌ | ❌ | ❌ | ❌ |
| Unsloth | ❌ | ❌ | ❌ | ❌ |
| OpenPipe | ❌ | ❌ | ❌ | ❌ |
| Predibase | ❌ | ❌ | ❌ | ❌ |
| Ludwig | ❌ | ❌ | Partial | ❌ |
| Microsoft RD-Agent | Partial | ❌ | Partial | ✅ |
| **VikaasLoop** | ✅ | ✅ | ✅ | ✅ |

The top-right position on the automation/self-improvement axis was empty before VikaasLoop.

## The Core Innovation — Skills Library

Every other fine-tuning tool treats each training run as a stateless operation. Run it. Get a model. The system forgets everything.

The Skills Library is VikaasLoop's institutional memory. It is a highly optimized SQLite database (WAL mode) paired with vectorized mathematical operations that stores:

```plaintext
For every iteration:
  task_description   → What were we trying to improve?
  strategy_name      → What data generation approach did we use?
  win_rate           → Did the fine-tuned model beat the base model?
  task_embedding     → Vector representation for similarity search
```

Before each new iteration, the Orchestrator queries the Skills Library:

```python
# "What strategies worked best on tasks similar to this one?"
top_strategies = skills_library.get_top_strategies(
    task_description="Explain Rust ownership concepts",
    top_k=3
)
# Returns: ["Chain-of-thought with code examples", "Socratic Q&A pairs", ...]
```

This means:
- Iteration 1 uses a general strategy
- Iteration 5 uses a strategy informed by 4 rounds of real results
- Iteration 10 is qualitatively smarter than iteration 1

The Skills Library is the difference between a person running an experiment once and a research institution that accumulates knowledge across thousands of experiments. It can be exported as JSON and shared with the community.

## How It Works — The 5-Agent Loop

### Agent 1 — DataGen Agent
Calls Gemini Flash to generate diverse, high-quality instruction-response training pairs guided by:
- The task description you provided
- The strategy hint retrieved from the Skills Library
- A few-shot example of what a good training pair looks like

**Output:** `data/generated/{run_id}.jsonl` — a JSONL file of training pairs, each quality-scored 1–5.

### Agent 2 — Training Agent
Loads a fresh base model and applies QLoRA (4-bit quantization + LoRA adapters) using HuggingFace TRL's SFTTrainer. Strictly manages VRAM by leveraging Gradient Checkpointing and dynamic precision scaling to prevent OOM crashes. Streams per-step loss values to the dashboard in real-time via WebSocket.

**Output:** `models/{run_id}/adapter/` — a LoRA adapter that can be loaded on top of the base model.

### Agent 3 — Eval Agent
Loads both the base model and the fine-tuned adapter. Runs both on 50 held-out test prompts (carved from the training data before training). Sends both responses to Gemini as a judge:

```plaintext
"Which response better achieves [task goal]? Answer A, B, or Tie."
```
All 50 judging calls run in parallel (asyncio + semaphore-controlled client pool). Returns a win rate between 0.0 and 1.0.

**Output:** Structured result dict with win rate, sample comparisons, and per-verdict breakdown.

### Agent 4 — Skills Library
Stores the result of this iteration. Uses a sentence-transformer embedding of the task description for semantic similarity search paired with NumPy matrix multiplication for high-performance querying. Implements UPSERT so repeated strategies accumulate a single, up-to-date win rate record.

### Agent 5 — Orchestrator
Coordinates the full loop. Owns the ModelManager lifecycle (models are loaded once per loop, not once per iteration). Manages WebSocket message queues authenticated via short-lived JWTs so the frontend receives secure, real-time updates.

## Features

### Core Loop
- Natural language task description input — no config files, no YAML.
- Fully autonomous loop: DataGen → Train → Eval → Learn → Repeat.
- Configurable target win rate (50% – 95%) and max iterations (1 – 20).
- Pause, resume, or stop at any time from the dashboard.

### Data Generation
- Gemini Flash generates diverse instruction-response pairs.
- Exact-match deduplication (O(n), no latency spikes).
- Quality scoring 1–5 per pair before training.
- JSONL output compatible with any HuggingFace dataset loader.

### Training
- QLoRA (4-bit) training via HuggingFace TRL + PEFT.
- Supports: `microsoft/phi-2`, `meta-llama/Llama-3.2-1B`, `google/gemma-2-2b`.
- Per-model LoRA target modules automatically selected.
- Live loss streaming to dashboard via WebSocket.
- Tokenizer cached across iterations — only adapter reloads between runs.

### Evaluation
- LLM-as-judge (Gemini Flash) with task-aware judge prompts.
- 50 parallel judge calls (semaphore-controlled client pool, ~3–5s per eval).
- Robust verdict parsing: handles "Response A", "Option A", "the first one".
- Sample comparison storage in SQLite for the Eval Dashboard.

### Security & Architecture
- **Zero-Trust File Operations:** Path traversal prevention on all file exports.
- **WebSocket Auth:** Rotating JWT authentication for streaming endpoints.
- **Non-Blocking I/O:** Heavy GPU and disk operations offloaded to thread pools to keep FastAPI event loops pristine.

### Dashboards
- **Engine UI (`index.html`):** Enterprise-styled React 18 frontend featuring live Chart.js trajectory tracking, terminal-style execution logs, and one-click HuggingFace Hub deployment.
- **Evaluation Studio (`eval_dashboard.html`):** Cryptographic-grade visual diffing for evaluating LLM outputs side-by-side.

## Technology Stack

| Layer | Technology | Why |
| :--- | :--- | :--- |
| Web framework | FastAPI 0.110+ | Async, WebSocket support, auto OpenAPI docs |
| Frontend | React 18 via CDN | No build step, runs anywhere |
| Styling | Tailwind CSS via CDN | Enterprise-grade UI without npm configuration |
| Charts | Chart.js 4 | Lightweight, streams well |
| LLM API | Google Gemini Flash | Free tier: 1M tokens/day |
| LLM SDK | google-genai | The correct, modern Python SDK |
| Fine-tuning | HuggingFace TRL + PEFT | Industry standard, LoRA support |
| Quantization | bitsandbytes | 4-bit QLoRA — runs on consumer GPUs |
| Embeddings | sentence-transformers | Fast semantic similarity for Skills Library |
| Database | SQLite (WAL mode) | Zero infrastructure, concurrent access |
| Auth | PyJWT | Rotating short-lived tokens for WebSockets |
| Model hosting | HuggingFace Hub | Free model publishing |

## Zero Cost Infrastructure

VikaasLoop runs entirely on free infrastructure. Here is every external service used and its cost:

| Service | What it does | Free tier |
| :--- | :--- | :--- |
| Gemini Flash API | Data generation + evaluation judging | 1,000,000 tokens/day, 15 RPM |
| HuggingFace Hub | Download base models + publish adapters | Unlimited public models |
| GitHub | Source code + CI/CD | Free for public repos |
| Your GPU | Training | Already yours |
| SQLite | Skills Library + eval results | Built into Python |

**Total monthly infrastructure cost: ₹0 / $0 / £0**

The only cost is your electricity bill for GPU training time.

## Quick Start

### Prerequisites

```bash
# Python 3.11 or higher
python --version  # Should print Python 3.11.x

# NVIDIA GPU (strongly recommended)
nvidia-smi        # Should show your GPU name and VRAM

# Git
git --version
```

### 1 — Clone

```bash
git clone https://github.com/LucidAkshay/vikaasloop.git
cd vikaasloop
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

**CUDA / Windows Note:** PyTorch installs CPU-only by default via standard pip. For local GPU training, ensure you install the CUDA build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

If you are on Windows and encounter bitsandbytes GPU detection errors, use the pre-compiled Windows wheel:

```bash
pip uninstall bitsandbytes -y
python -m pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

### 3 — Get your free Gemini API key
1. Go to https://aistudio.google.com/apikey
2. Click **Create API Key**
3. Copy the key

### 4 — Configure

```bash
# Copy the example env file
cp .env.example .env

# Open .env and add your key
GEMINI_API_KEY=your_key_here
```

### 5 — Run

```bash
python main.py
```

### 6 — Open the dashboard
Navigate to http://localhost:8000

You should see the VikaasLoop Engine dashboard. Enter a task description, select a model, set your target score, and click **Initialize Autonomous Loop**.

## Supported Models

| Model | Parameters | VRAM required | Speed | Recommended for |
| :--- | :--- | :--- | :--- | :--- |
| microsoft/phi-2 | 2.7B | ~6 GB | Fast | Default choice, great quality/speed ratio |
| meta-llama/Llama-3.2-1B | 1B | ~4 GB | Fastest | Low-VRAM machines, quick experiments |
| google/gemma-2-2b | 2B | ~6 GB | Fast | Strong reasoning tasks |

No GPU? VikaasLoop falls back to CPU training automatically. Training will be significantly slower but will complete. Recommended only for testing with 10–20 training pairs.

## Roadmap

### v1.1 — Community Edition
- [ ] Community Skills Library sync — share your `skills.db` with the world
- [ ] Multi-model tournament — pit 3 fine-tuned variants against each other
- [ ] Constitutional AI data generation mode — RLHF-ready preference datasets
- [ ] CLI mode — headless server operation, no browser required

### v1.2 — Enterprise Edition
- [ ] Docker container with GPU passthrough
- [ ] Scheduled loops — run experiments overnight on a cron schedule
- [ ] Discord/Slack webhook notifications on loop completion

### v2.0 — Research Edition
- [ ] FAISS-powered Skills Library — scales to millions of strategy records
- [ ] Automated hyperparameter search — LoRA rank and alpha optimization
- [ ] Integration with VikaasLoop Software Factory pipeline

## Impact — Who Benefits

### Individual Developers
Run a proper model improvement research loop on your laptop with no cloud bills. A developer in Jalandhar, Lagos, or Jakarta now has the same self-improving research capability that a 20-person ML team at a big lab has.

### Students and Researchers
Run 100 fine-tuning experiments while you sleep. Wake up to a Skills Library that tells you exactly which data strategies worked and by how much. Publish your Skills Library as a research artifact alongside your paper.

### Startups
Build a domain-specific model for your product without hiring an ML engineer. Your data never leaves your machine. The Skills Library you build becomes a competitive moat — institutional knowledge about what training approaches work for your specific domain.

## Contributing

VikaasLoop is built for the community. Contributions are deeply welcome.

### How to contribute

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_GITHUB_USERNAME/vikaasloop.git
cd vikaasloop

# Create a branch for your feature
git checkout -b feature/add-mistral-support

# Make your changes, then run the smoke tests
python verify_implementation.py

# Commit and push
git add .
git commit -m "feat: add Mistral-7B LoRA target modules"
git push origin feature/add-mistral-support

# Open a Pull Request on GitHub
```

### Code style
- **Python:** Black formatter (`black .`) + isort (`isort .`)
- **Security:** Any path construction must use `os.path.join()` and pass through `sanitize_run_id()`.

## License

VikaasLoop is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

This means:
- ✅ Free to use, modify, and distribute
- ✅ Free for personal, research, and commercial use
- ✅ You can build products on top of VikaasLoop
- ⚠️ If you deploy a modified version as a service (SaaS), you must open-source your modifications
- ⚠️ All derivative works must carry the same AGPL license

See [LICENSE](LICENSE) for the full text.

Built with love in India 🇮🇳 for the global open-source community

*"The best model improvements come from better data, not better hyperparameters."*

---

**About the Creator**

**Akshay Sharma**
Creator of VikaasLoop and the open-source Kavach Application (Tactical Zero-Trust Firewall for Autonomous AI). Brand Owner at Amrutya Essence. Passionate about building AI tools that solve real problems people didn't know they had.

🌐 **Personal Website:** [https://lucidakshay.dev](https://lucidakshay.dev)
