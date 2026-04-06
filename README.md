<div align="center">

# Living Mind Cortex 🧠⚡
**The Sovereign Backend for Autonomous Agents**

[![License](https://img.shields.io/badge/License-Sovereign%20Drop-red.svg)]()
[![Runtime](https://img.shields.io/badge/Runtime-Python%203.10+-blue.svg)]()
[![Zero Cloud](https://img.shields.io/badge/Cloud%20Dependencies-None-success.svg)]()

</div>

---

## 🛑 The Problem: The Goldfish Memory Crisis

Modern agents suffer from fatal amnesia. Every time an agent's terminal window closes, its session dies. Every hard problem solved, every bug fixed, and every architectural pattern learned is instantly lost to the void. Re-injecting massive text files into a context window on boot is expensive, slow, and highly prone to hallucination.

## ⚡ The Solution: Living Mind Cortex

**Living Mind Cortex** is a fully autonomous, local-first memory backend and deterministic event loop designed explicitly for AI agents. 

You attach your agent (or swarm) to this framework, and it instantly gains:
1. **Cognitive Continuity**: Your agent drops `AgentTrace` events into the ecosystem. When your agent reboots, it reads from the Cortex, waking up with full context of its previous lives.
2. **Autonomous Background Synthesis**: The Deterministic Event Loop (DEL) runs independently. While your agent sleeps, the Cortex chunks thoughts, runs self-directed research, and optimizes memory patterns using local LLMs.
3. **Zero-Trust Sovereignty**: 100% local operation. Zero cloud lock-in. No API keys. No external telemetry.

---

## 🏗️ Architecture

Unlike typical blocking REST APIs, the Cortex operates as a continuous organic runtime.

*   `Deterministic Event Loop (DEL)`: The core pulse driver that paces all background activities (formerly "Heartbeat").
*   `Persistent Memory Store & Synthesis Engine`: A high-performance spatial ledger for mapping episodic and semantic chunks over time.
*   `State & Telemetry Broker`: Real-time observability bus for routing agent state.
*   `Validation Boundary`: Hardened perimeter checking for malicious logic injections.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have [Ollama](https://ollama.com/) installed and running locally. The Cortex depends on it for local reasoning loops.
```bash
ollama serve
```

### 2. Boot the Cortex
Clone the repository and spin up the deterministic event loop (DEL) and the API viewer.

```bash
git clone https://github.com/your-username/living-mind-cortex.git
cd living-mind-cortex
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start the Living Mind Cortex
./start.sh
```

### 3. Connect your Agent
Your interface dashboard (`http://localhost:8008/ui/index.html`) will now be live. Your agent can immediately begin injecting context into `/api/agent/inject` and pulling memory states. 

If you are using the Nodeus Substrate, the Cortex is configured to securely sync with your ledger.

---

<div align="center">
<i>"To build sovereign machines, we must first give them the capability to remember."</i><br>
<b>A Manifesto Engine Provision.</b>
</div>
