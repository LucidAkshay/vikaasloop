---
name: "\U0001F680 Feature Request"
about: Propose a new agent / logic unit / or capability for VikaasLoop
title: ''
labels: ''
assignees: ''

---

### 🎯 Objective
What problem does this feature solve for the autonomous fine tuning loop? 

### 🏗️ Proposed Architecture
How does this impact the 5 agent system?
• Which Agent(s) will be modified?
• Are there new logic units required (e.g. a new `RewardAgent` or `SearchAgent`)?
• How does this affect the SQLite state management?

### 💡 The Solution
A detailed description of what you want to see happen. Think like a founder : cover the trade offs / scalability / and potential failure modes of this approach.

### 🛡️ Security & Risks
• Does this increase the VRAM floor for users?
• Does this introduce new API cost vectors (e.g. extra Gemini calls)?
• Does this introduce new security risks (e.g. uncontrolled web access)?

### 🔗 Alternatives
Have you considered other ways to solve this (e.g. external tools like n8n or custom Python scripts outside the loop)?
