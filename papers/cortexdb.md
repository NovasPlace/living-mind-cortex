# 🔥 FEED TO AGENT

## CORTEXDB — BIOLOGICALLY-INSPIRED COGNITIVE MEMORY ENGINE
SQLite-backed memory engine with 4 memory types, Ebbinghaus forgetting curves, emotional encoding, source monitoring, flashbulb detection, salience-gated working memory, and cognitive biases. Implements the Cortex Memory Complex architecture (Frost & Nirvash, 2026).

---

# MANIFESTO ENGINE — EXECUTION BLUEPRINT

## 1. SYSTEM ARCHITECTURE

### FILE MANIFEST
| File | Purpose |
|------|---------|
| cortex/engine.py | Core Cortex class — SQLite persistence, remember/recall/emotional_recall/decay/consolidate |
| cortex/working_memory.py | Fixed-capacity sliding window with salience-based eviction (not persisted) |
| cortex/cognitive_biases.py | Cognitive bias simulations — recency, confirmation, availability biases |
| cortex/priming.py | Semantic priming — pre-activation of related memories for faster retrieval |
| cortex/autobio.py | Autobiographical memory — narrative construction from episodic memories |
| cortex/trace.py | Execution tracing decorator for handler observability |
| cortex/api.py | HTTP API layer for memory operations |
| storage/engine.py | Storage abstraction — SQLite connection pooling and migration |
| storage/api.py | Storage HTTP endpoints |
| auth.py | API key authentication and access control |
| config.py | Configuration loader — database path, capacity limits, decay parameters |
| cli.py | Command-line interface for memory operations |
| main.py | Application entry point — FastAPI server |
| watchdog.py | Memory health monitoring — capacity alerts, decay scheduling |
| test_cortex.py | Real-input verification tests for memory operations |
| test_trace.py | Trace ledger verification tests |

### DATA MODELS

**Memory** (dataclass — 17 fields)
- id: str — UUID identifier
- content: str — Memory content (text)
- type: str — One of "episodic", "procedural", "semantic", "relational" (default: "episodic")
- tags: list[str] — Categorization tags
- importance: float — 0.0 to 1.0 (default 0.5), boosted by emotional encoding
- created_at: float — Creation timestamp
- last_accessed: float — Most recent recall timestamp
- access_count: int — Number of times recalled (drives spaced repetition)
- embedding: bytes | None — Vector embedding for semantic similarity search
- emotion: str — Emotional valence: "neutral", "joy", "fear", "anger", "surprise", "sadness", "disgust"
- confidence: float — Source monitoring confidence (1.0 for experienced, 0.65 for inferred)
- context: str — Contextual metadata for retrieval
- source: str — Origin type: "experienced", "told", "generated", "inferred"
- linked_ids: list[str] — Cross-references to related memories
- metadata: dict — Arbitrary structured metadata
- Computed: is_flashbulb — True if emotion is high-valence and importance > 0.8
- Computed: is_identity — True if tagged "identity" or "self"
- Relationships: Linked to other Memory via linked_ids list

**WorkingMemoryItem** (dataclass)
- id: str — Sequential ID (wm-1, wm-2, ...)
- content: str — Item content
- category: str — One of "event", "decision", "outcome", "reflex", "alert", "goal", "dream", "metabolism"
- salience: float — 0.0 to 1.0, determines eviction priority
- added_at: float — Timestamp
- metadata: dict — Additional data

---

### COGNITIVE CONSTANTS

**Emotional Encoding Boosts** (importance multipliers):
- fear: 1.5× — biological threat response
- surprise: 1.3× — novelty detection
- anger: 1.2× — social threat
- joy: 1.1× — reward signal
- sadness: 1.0× — no boost
- disgust: 0.9× — mild suppression
- neutral: 1.0× — baseline

**Source Monitoring Penalties** (confidence multipliers):
- experienced: 1.00 — first-hand experience
- told: 0.85 — second-hand information
- generated: 0.75 — AI-generated content
- inferred: 0.65 — logical deduction

**Ebbinghaus Decay Parameters** (§4.1):
- Base stability: 3600s (1 hour half-life)
- Formula: R(t) = e^{-t/S} where S = S_base × (1+n)^1.5 × (1+I×2.0)
- n = access_count, I = importance
- Consolidation check interval: 3600s
- Reconsolidation factor: 0.95 (5% confidence loss per recall)
- Reconsolidation floor: 0.1

---

### DATABASE SCHEMA (SQLite with FTS5)

### memories
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT PRIMARY KEY | Unique memory identifier |
| content | TEXT NOT NULL | Memory content |
| type | TEXT DEFAULT 'episodic' | Memory type: episodic/procedural/semantic/relational |
| tags | TEXT DEFAULT '[]' | JSON array of tags |
| importance | REAL DEFAULT 0.5 | Importance score 0.0-1.0 |
| created_at | REAL NOT NULL | Creation timestamp |
| last_accessed | REAL DEFAULT 0.0 | Last recall timestamp |
| access_count | INTEGER DEFAULT 0 | Recall count |
| embedding | BLOB | Vector embedding bytes |
| emotion | TEXT DEFAULT 'neutral' | Emotional valence |
| confidence | REAL DEFAULT 1.0 | Source monitoring confidence |
| context | TEXT DEFAULT '' | Contextual metadata |
| source | TEXT DEFAULT 'session' | Origin type |
| linked_ids | TEXT DEFAULT '[]' | JSON array of linked memory IDs |
| metadata | TEXT DEFAULT '{}' | JSON metadata |

### memories_fts (FTS5 virtual table)
- Content-synced with memories table via INSERT trigger
- Indexes: content, tags
- Used for full-text recall with BM25 ranking

---

## 2. HANDLER FUNCTIONS

**1. Handler: `Cortex.remember`**
- **Purpose**: Store a new memory with cognitive metadata.
- **Inputs**: content (str), type (str), tags, importance, emotion, source, confidence, context, linked_ids, metadata
- **Behavior**:
  1. Validate memory type (must be in MEMORY_TYPES).
  2. Apply source monitoring confidence penalty.
  3. Detect flashbulb conditions (high emotion + high importance).
  4. Apply emotional encoding boost to importance.
  5. Generate UUID, insert into SQLite.
  6. Sync to FTS5 index.
  7. Check capacity — trigger consolidation if at max_memory_count.
- **Returns**: Memory ID string.

**2. Handler: `Cortex.recall`**
- **Purpose**: FTS5 search ranked by relevance × importance.
- **Inputs**: query (str), limit (int), min_importance (float)
- **Behavior**:
  1. Sanitize FTS5 query (escape special chars).
  2. Execute FTS5 MATCH with BM25 ranking.
  3. Apply importance filter.
  4. Update access metadata (last_accessed, access_count += 1).
  5. Apply reconsolidation degradation (5% confidence loss per recall).
- **Returns**: list[Memory]

**3. Handler: `Cortex.emotional_recall`**
- **Purpose**: Retrieve memories with emotional valence boosting.
- **Behavior**: Fear memories surface first. If emotion specified, boosts that emotion. Applies same access tracking.

**4. Handler: `Cortex.decay`**
- **Purpose**: Run Ebbinghaus forgetting curves on all non-protected memories.
- **Formula**: R(t) = e^{-t/S} where S = S_base × (1+n)^1.5 × (1+I×2.0)
- **Behavior**: Calculates retention for each memory. Below threshold → prune. Flashbulb and identity memories protected.
- **Returns**: Count of pruned memories.

**5. Handler: `Cortex.consolidate`**
- **Purpose**: Convert old episodic memories to semantic type.
- **Behavior**: Memories older than consolidation threshold with high access count are converted. Content may be truncated.

**6. Handler: `WorkingMemory.add`**
- **Purpose**: Add item to salience-gated short-term buffer.
- **Behavior**: At capacity, evicts lowest-salience item. If incoming item has lower salience than all existing items, it's rejected.

---

## 3. VERIFICATION GATE & HARD CONSTRAINTS

### VERIFICATION TESTS

**Test 1: HAPPY PATH — Remember and Recall**
- Input: Store memory "PostgreSQL index optimization" with tags ["database", "performance"].
- Expected: recall("database optimization") returns the stored memory with BM25 score.

**Test 2: ERROR PATH — Invalid Memory Type**
- Input: remember(type="fantasy")
- Expected: Rejected or coerced to "episodic".

**Test 3: EDGE CASE — Ebbinghaus Decay**
- Input: Store low-importance memory, advance time past half-life.
- Expected: decay() prunes the memory. Flashbulb memories survive.

**Test 4: ADVERSARIAL — FTS Injection**
- Input: recall("DROP TABLE memories")
- Expected: Query sanitized, no SQL injection, returns empty results.

**Test 5: EMOTIONAL ENCODING — Fear Boost**
- Input: remember(content="production server down", emotion="fear", importance=0.5)
- Expected: importance boosted to 0.75 (fear × 1.5).

### HARD CONSTRAINTS
- All queries parameterized — no string interpolation in SQL
- FTS5 special characters escaped via _sanitize_fts_query
- Working memory is ephemeral — not persisted to disk
- Consolidation interval: 3600s
- Max memory count configurable (default 10,000)
- Reconsolidation floor: 0.1 (memories can't lose all confidence)
