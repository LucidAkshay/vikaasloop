---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ''
assignees: ''

---

---
name: "🐛 Bug Report"
about: "Report a technical failure or edge case in the VikaasLoop engine"
title: "[BUG] <Describe the issue briefly>"
labels: "bug"
assignees: ""
---

### 🛑 Failure Description
Provide a clear / concise description of what went wrong. Include any crash logs or terminal output from the Orchestrator or the specific Agent that failed.

### 💻 Environment & Hardware
VikaasLoop is hardware sensitive. Provide these details :
• **OS :** [e.g. Windows 11 / WSL2 / Ubuntu 22.04]
• **GPU :** [e.g. RTX 3060 12GB]
• **VRAM Usage at Failure :** [e.g. 11.2GB / 12GB]
• **Python Version :** [e.g. 3.11.5]
• **CUDA Version :** [e.g. 12.1]

### 👣 Reproduction Steps
How can we trigger this bug again?
1. Start the loop with `python main.py`
2. Configure the `DataGen` agent with these seed prompts : [Insert Prompts]
3. Observe the crash at the [Training / Eval / Orchestrator] stage

### 📉 Expected vs Actual Behavior
• **Expected :** What should have happened?
• **Actual :** What actually happened (e.g. OOM error / JSON hallucination / Database lock)?

### 📂 Logs
Paste the relevant logs from the `logs/` directory. Focus on the traceback leading to the exit code.

### 🛠️ Potential Fix / Investigation
If you are a senior engineer / have you identified the root cause? Mention if this is a logic unit failure or a dependency mismatch (e.g. bitsandbytes versioning).
