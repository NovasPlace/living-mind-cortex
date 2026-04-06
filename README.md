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

## 🏛️ Sub-System Architecture Deep-Dive

The Living Mind Cortex is not a monolithic script; it is a modular, living topology composed of three primary macro-systems and their respective operational components.

### 1. `core/` — The Deterministic Runtime
The biological clock and security perimeter of the organism. This system drives execution and ensures safe operation.
*   **`runtime.py` (The Event Loop):** The primary 16-phase deterministic pulse loop that sequences all internal actions rather than relying on blocking REST API requests.
*   **`orchestrator.py` (The Brain):** The decision-making core. Interprets sensory input and queries the local LLM (`gemma4-auditor`) to decide on immediate Actions vs. Explorations.
*   **`security_perimeter.py` (The Immune System):** Registers all active subsystems, runs cyclic background health checks, and quarantines components that fail consecutively or trigger fatal exceptions.
*   **`execution_engine.py` (Motor Cortex):** A strictly isolated tool registry for executing highly trusted OS commands, filesystem traversals, and codebase AST generation.
*   **`research_engine.py` (Autodidact):** Spawns detached, non-blocking DDG/Ollama worker threads to autonomously crawl documentation or investigate topics deemed "ambiguous" by the Orchestrator without blocking the main event loop.

### 2. `cortex/` — Persistent Cognitive Memory
The long-term and working memory center, inspired directly by human cognitive psychology.
*   **`engine.py`:** The primary PostgreSQL/SQLite ingestion pipeline for new knowledge chunks.
*   **`working_memory.py`:** Short-term cache that decays quickly. Used for immediate context handling before chunks are consolidated to disk.
*   **`seed_axioms.py`:** Immutable core directives. These act as "Flashbulb Memories" that define the organism's baseline identity and can never decay.
*   **`cognitive_biases.py`:** An algebraic scoring engine. Promotes memories based on recentness (Ebbinghaus curve), emotional salience (fear/joy markers), and rehearsal frequency.
*   **`priming.py`:** Graph engine mapping that tracks overlapping neural pathways. If a memory is activated, closely linked sibling memories have their retrieval probability boosted.
*   **`imagination.py` (Dreams):** An offline background processor that reorganizes semantic memory chunks overnight to discover novel correlations or solve previously dead-locked logic puzzles.

### 3. `state/` — Telemetry & Internal Status
The chemical signaling bus that dictates how the organism *feels* and behaves mechanically over time.
*   **`telemetry_broker.py` (Hormone Bus):** Event-driven MQTT-style bus handling massive simulated chemical spikes. A spike in "Cortisol" drastically lowers the orchestrator's risk tolerance, while a spike in "Dopamine" reinforces successful actions.
*   **`health_monitor.py` (Homeostasis):** Watches CPU/Memory usage and adjusts the `pulse_interval` of the DEL up or down, ensuring the machine learning models don't crash the host OS.
*   **`circadian.py`:** Manages sleep/wake cycles. Triggers low-power "Dream" states during prolonged inactivity.

### 4. `dashboard/` — Visual Cortex
A bundled Svelte/Vanilla UI allowing operators to monitor the inner workings of the runtime.
*   **`viewer.html` The Cortex Dashboard:** Visually maps the `cortex/` memories and logs the live output of the `state/` broker in a realtime topology.

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
