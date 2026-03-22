# Security Policy : VikaasLoop

## 🛡️ Our Philosophy
VikaasLoop is designed with a **Security First** mindset, acknowledging that autonomous agents have access to powerful compute and sensitive API keys. This project follows the **Zero Trust** principles established in the Kavach Application. We assume every input is a potential injection and every local weight file is a potential vector for code execution.

---

## 🛑 Reporting a Vulnerability
If you discover a security hole or a way to break the agent sandbox, do **not** open a public issue. Public disclosure puts the entire user base at risk.

• **Private Disclosure:** Email the maintainer directly at the address listed on the GitHub profile.
• **Response Time:** You will receive an initial response within 48 hours.
• **Vulnerability Handling:** We use a coordinated disclosure process. A fix will be developed / tested / and merged before the details are made public.

---

## ✅ Supported Versions
We only provide security patches for the following versions :

| Version | Status |
| :--- | :--- |
| 1.0.x | Mainstream Support |
| < 1.0.0 | End of Life / Upgrade Immediately |

---

## 🏗️ Threat Model : The Founder's Reality Check
As the creator of VikaasLoop, you must be aware of these high / risk failure modes :

### 1. API Key Compromise
• **The Risk:** Your `GEMINI_API_KEY` or HuggingFace tokens are the keys to your financial kingdom.
• **Mitigation:** Never check in your `.env` file. The repository includes a `.gitignore` specifically tuned for this. Use GitHub Secrets for CI / CD runs.

### 2. Model Weight Poisoning (RCE)
• **The Risk:** Loading untrusted `.pth` or `.bin` files via `torch.load()` is a massive security risk because these formats can execute arbitrary Python code.
• **Mitigation:** VikaasLoop prioritizes **Safetensors**. If the `ModelManager` detects a non / safe format / it should be treated as a high / risk event.

### 3. Prompt Injection in DataGen
• **The Risk:** If an attacker can influence the "Seed Prompts" fed to the `DataGen Agent` / they can force the model to generate biased / toxic / or malicious training data. This "poisoned" data then becomes part of the fine tuned weights.
• **Mitigation:** Input sanitization is enforced by the `Orchestrator Agent` before any data is passed to the generation loop.

### 4. Database Injection
• **The Risk:** We use SQLite for state management. While the current implementation uses an ORM / raw queries could lead to data corruption or exfiltration.
• **Mitigation:** All queries are parameterized. Direct database access via the dashboard is restricted by JWT authentication.

---

## 🔒 Security Hardening Guide
To run VikaasLoop in a production environment / follow these 4 steps :

1. **Isolation:** Run the engine inside a Docker container with restricted network access.
2. **Least Privilege:** The Linux user running the training loop should only have write permissions for the `logs/` / `data/` / and `models/` directories.
3. **JWT Rotation:** Change your `JWT_SECRET` every 30 days and never use the default "test_secret" in production.
4. **Monitoring:** Regularly audit the `logs/security.log` for failed auth attempts or unauthorized `run_id` access attempts.

---

## 📦 Dependency Security
The VikaasLoop stack relies on heavy libraries like `transformers` / `bitsandbytes` / and `fastapi`.
• We run `safety` and `bandit` checks in the CI pipeline to catch known vulnerabilities in the supply chain.
• If a dependency is flagged / the build will fail immediately / preventing a "dirty" version from being deployed.

---

Akshay Sharma / this document ensures that any contributor or user treats VikaasLoop with the professional caution it deserves. 

**Would you like me to now generate the `CONTRIBUTING.md` file to set the coding standards for future VikaasLoop developers?**
