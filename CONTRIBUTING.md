# Contributing to the VikaasLoop Engine

First, thank you for considering contributing to VikaasLoop! Whether you are fixing a bug, adding support for a new base model, or improving the React dashboard, your help makes this open-source research tool better for everyone.

## 🚀 Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/LucidAkshay/vikaasloop.git
   cd vikaasloop
   ```
3. Install dependencies (we highly recommend using a virtual environment):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Upgrade pip and install with dev dependencies
   python -m pip install --upgrade pip
   pip install -e ".[dev]"
   ```

## 🧠 Architectural Guardrails

VikaasLoop is a highly orchestrated 5-agent system. To keep the codebase stable and secure, please adhere to these design principles:

1. **Separation of Concerns**: Agents must not cross boundaries. Do not put HuggingFace/PyTorch code in the `EvalAgent`. Do not put Gemini API calls in the `ModelManager` or `TrainingAgent`.
2. **Zero-Trust Security**: Any code that handles file paths (especially for model exports or datasets) MUST use `os.path.join()` and pass through the `sanitize_run_id()` utility to prevent directory traversal attacks.
3. **Non-Blocking I/O**: The FastAPI event loop must remain fast. Any heavy GPU operations, large file writes, or synchronous CPU-bound tasks must be offloaded to a background thread (`asyncio.to_thread` or `ProcessPoolExecutor`).

## 💻 Making Changes

### Branching Strategy
Always create a new branch for your feature or bug fix:
`git checkout -b feature/add-mistral-support` or `git checkout -b fix/websocket-auth`

Write clear, descriptive commit messages explaining why a change was made.

### Coding Guidelines (Python)
We enforce strict style guidelines to keep the engine codebase clean:

*   **Formatting**: We use `black`. Always run `black .` before committing.
*   **Imports**: We use `isort`. Always run `isort .` before committing.
*   **Type Hinting**: All functions and methods MUST include Python type hints (`typing` module).
*   **Linting**: We enforce `flake8` checks to catch undefined variables and syntax errors.

### Testing (Crucial Step)
We do not use standard `pytest` for core architecture validation. We use a custom, asynchronous smoke-test suite that validates the JWT security, deduplication logic, and vectorized Skills Library math without requiring a GPU or API key.

Before opening a Pull Request, you must ensure this suite passes:

```bash
python verify_implementation.py
```
*(If this script fails locally, the GitHub Actions CI pipeline will automatically block your Pull Request).*

## 📤 Opening a Pull Request

When you are ready to submit your code:

1. Push your branch to your fork.
2. Open a Pull Request against the `main` branch of the upstream repository.
3. Fill out the PR template completely. Link to any relevant open issues.
4. Ensure the automated CI pipeline passes (it will run formatting checks and the verification suite).
5. A maintainer will review your code. Please be responsive to feedback!

## 🐛 Reporting Bugs

If you find a bug, please check the existing GitHub Issues first. If it is new, open an Issue and provide:

*   Your OS and Python version.
*   Your GPU model and VRAM (e.g., RTX 3060 12GB).
*   The exact error message and stack trace from the terminal.
*   Steps to reproduce the failure.