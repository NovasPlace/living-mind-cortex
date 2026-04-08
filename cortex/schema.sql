-- Living Mind — Cortex Memory Schema
-- PostgreSQL with pg_trgm full-text search
-- 17-field memory record per Cortex Memory Complex spec

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ============================================================
-- MEMORIES — The primary memory store
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content         TEXT NOT NULL,
    type            TEXT NOT NULL DEFAULT 'episodic'
                        CHECK (type IN ('episodic','semantic','procedural','relational')),
    tags            TEXT[]          NOT NULL DEFAULT '{}',
    importance      REAL            NOT NULL DEFAULT 0.5
                        CHECK (importance >= 0.0 AND importance <= 1.0),
    created_at      DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    last_accessed   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    access_count    INTEGER         NOT NULL DEFAULT 0,
    embedding       BYTEA,          -- numpy float32 array, serialized
    emotion         TEXT            NOT NULL DEFAULT 'neutral'
                        CHECK (emotion IN ('neutral','joy','fear','anger','surprise',
                                           'sadness','disgust','curiosity','frustration')),
    -- renamed mechanical equivalents (activation_tag + signal_polarity)
    -- emotion/valence kept for backwards compat; new writes use the renamed names
    activation_tag  TEXT            GENERATED ALWAYS AS (emotion) STORED,
    confidence      REAL            NOT NULL DEFAULT 1.0
                        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    context         TEXT            NOT NULL DEFAULT '',
    source          TEXT            NOT NULL DEFAULT 'experienced'
                        CHECK (source IN ('experienced','told','generated','inferred')),
    linked_ids      UUID[]          NOT NULL DEFAULT '{}',
    metadata        JSONB           NOT NULL DEFAULT '{}',

    -- Computed / derived
    is_flashbulb    BOOLEAN GENERATED ALWAYS AS (
                        emotion IN ('fear','surprise') AND importance >= 0.8
                    ) STORED,
    is_identity     BOOLEAN GENERATED ALWAYS AS (
                        'identity' = ANY(tags) OR 'self' = ANY(tags)
                    ) STORED
);

-- Full-text search index using pg_trgm
CREATE INDEX IF NOT EXISTS memories_content_trgm
    ON memories USING gin(content gin_trgm_ops);

CREATE INDEX IF NOT EXISTS memories_tags_idx
    ON memories USING gin(tags);

CREATE INDEX IF NOT EXISTS memories_type_idx
    ON memories (type);

CREATE INDEX IF NOT EXISTS memories_importance_idx
    ON memories (importance DESC);

CREATE INDEX IF NOT EXISTS memories_emotion_idx
    ON memories (emotion);

CREATE INDEX IF NOT EXISTS memories_identity_idx
    ON memories (is_identity)
    WHERE is_identity = TRUE;

CREATE INDEX IF NOT EXISTS memories_flashbulb_idx
    ON memories (is_flashbulb)
    WHERE is_flashbulb = TRUE;

CREATE INDEX IF NOT EXISTS memories_created_at_idx
    ON memories (created_at DESC);


-- ============================================================
-- WORKING_MEMORY — Ephemeral salience-gated short-term buffer
-- Not persisted across restarts (cleared on boot)
-- ============================================================
CREATE TABLE IF NOT EXISTS working_memory (
    id          TEXT PRIMARY KEY,        -- wm-1, wm-2, ...
    content     TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'event'
                    CHECK (category IN ('event','decision','outcome','reflex',
                                        'alert','goal','dream','metabolism')),
    salience    REAL NOT NULL DEFAULT 0.5
                    CHECK (salience >= 0.0 AND salience <= 1.0),
    added_at    DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    metadata    JSONB NOT NULL DEFAULT '{}'
);


-- ============================================================
-- MEMORY_GRAPH — Relationship edges between memories
-- ============================================================
CREATE TABLE IF NOT EXISTS memory_graph (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_id       UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relationship    TEXT NOT NULL DEFAULT 'related',  -- caused, supports, contradicts, etc.
    strength        REAL NOT NULL DEFAULT 0.5,
    created_at      DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    UNIQUE(source_id, target_id, relationship)
);

CREATE INDEX IF NOT EXISTS memory_graph_source_idx ON memory_graph (source_id);
CREATE INDEX IF NOT EXISTS memory_graph_target_idx ON memory_graph (target_id);


-- ============================================================
-- PRIMING_ACTIVATIONS — Spreading activation state
-- Ephemeral — decays over time, cleared on restart
-- ============================================================
CREATE TABLE IF NOT EXISTS priming_activations (
    memory_id       UUID PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    boost           REAL NOT NULL DEFAULT 0.0,
    activated_at    DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    decay_duration  INTEGER NOT NULL DEFAULT 180   -- seconds
);


-- ============================================================
-- SYSTEM_STATE_VECTOR — Bounded cognitive control state
-- Replaces hormone_bus. Single-row singleton, upserted every pulse.
-- Three typed groups: drives (compete), loads (accumulate), regulators (drift).
-- All values strictly ∈ [0.0, 1.0] — enforced at the StateEngine layer.
-- ============================================================
CREATE TABLE IF NOT EXISTS system_state_vector (
    id                  INTEGER PRIMARY KEY DEFAULT 1,  -- always row 1

    -- Drives (compete via softmax; push behavior)
    reward_drive        REAL NOT NULL DEFAULT 0.30
                            CHECK (reward_drive BETWEEN 0.0 AND 1.0),
    novelty_exploration REAL NOT NULL DEFAULT 0.30
                            CHECK (novelty_exploration BETWEEN 0.0 AND 1.0),

    -- Loads (accumulate under pressure; suppress drives)
    stress_load         REAL NOT NULL DEFAULT 0.20
                            CHECK (stress_load BETWEEN 0.0 AND 1.0),
    sleep_pressure      REAL NOT NULL DEFAULT 0.10
                            CHECK (sleep_pressure BETWEEN 0.0 AND 1.0),

    -- Regulators (modulate transfer functions; drift toward baseline)
    focus_stability     REAL NOT NULL DEFAULT 0.55
                            CHECK (focus_stability BETWEEN 0.0 AND 1.0),
    social_trust        REAL NOT NULL DEFAULT 0.50
                            CHECK (social_trust BETWEEN 0.0 AND 1.0),

    -- Derived
    cognitive_stance    TEXT NOT NULL DEFAULT 'balanced',
    updated_at          DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    CONSTRAINT state_vector_singleton CHECK (id = 1)
);

INSERT INTO system_state_vector (id) VALUES (1) ON CONFLICT DO NOTHING;

-- Backwards-compatibility view: maps old hormone_bus column names to new ones.
-- Allows legacy code to keep reading without immediate refactor.
CREATE OR REPLACE VIEW hormone_bus AS
    SELECT
        id,
        reward_drive        AS dopamine,
        novelty_exploration AS norepinephrine,
        stress_load         AS cortisol,
        sleep_pressure      AS melatonin,
        focus_stability     AS acetylcholine,
        social_trust        AS oxytocin,
        cognitive_stance    AS valence,
        updated_at
    FROM system_state_vector;


-- ============================================================
-- CIRCADIAN — Body clock state
-- ============================================================
CREATE TABLE IF NOT EXISTS circadian (
    id              INTEGER PRIMARY KEY DEFAULT 1,
    phase           TEXT NOT NULL DEFAULT 'day'
                        CHECK (phase IN ('dawn','day','evening','night')),
    adenosine       REAL NOT NULL DEFAULT 0.0,   -- sleep pressure (builds awake)
    hour_of_day     INTEGER NOT NULL DEFAULT 12,
    updated_at      DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    CONSTRAINT circadian_singleton CHECK (id = 1)
);

INSERT INTO circadian (id) VALUES (1) ON CONFLICT DO NOTHING;


-- ============================================================
-- ORGAN_REGISTRY — Live census of registered organs
-- ============================================================
CREATE TABLE IF NOT EXISTS organ_registry (
    name        TEXT PRIMARY KEY,
    category    TEXT NOT NULL,   -- Defense|Evolution|Memory|Consciousness|etc.
    status      TEXT NOT NULL DEFAULT 'loaded'
                    CHECK (status IN ('loaded','failed','disabled','quarantined')),
    pulse_freq  INTEGER NOT NULL DEFAULT 1,     -- fire every N pulses
    last_fired  DOUBLE PRECISION,
    fire_count  BIGINT NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    metadata    JSONB NOT NULL DEFAULT '{}',
    registered_at DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now())
);


-- ============================================================
-- AGENT_TRACE — Execution telemetry (every organ fire)
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_trace (
    id          BIGSERIAL PRIMARY KEY,
    organ       TEXT NOT NULL,
    action      TEXT NOT NULL,
    pulse       BIGINT NOT NULL DEFAULT 0,
    success     BOOLEAN NOT NULL DEFAULT TRUE,
    duration_ms INTEGER,
    payload     JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS agent_trace_organ_idx ON agent_trace (organ);
CREATE INDEX IF NOT EXISTS agent_trace_created_at_idx ON agent_trace (created_at DESC);
CREATE INDEX IF NOT EXISTS agent_trace_pulse_idx ON agent_trace (pulse DESC);


-- ============================================================
-- DREAM_JOURNAL — Offline cognition output
-- ============================================================
CREATE TABLE IF NOT EXISTS dream_journal (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy    TEXT NOT NULL,   -- gene_affinity|niche_fill|mutation_replay|toxic_avoidance
    hypothesis  TEXT NOT NULL,
    pattern     TEXT NOT NULL DEFAULT '',
    confidence  REAL NOT NULL DEFAULT 0.5,
    staged      BOOLEAN NOT NULL DEFAULT FALSE,
    pulse       BIGINT NOT NULL DEFAULT 0,
    created_at  DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now())
);


-- ============================================================
-- SESSION_JOURNAL — Wipe-proof session bridge
-- ============================================================
CREATE TABLE IF NOT EXISTS session_journal (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type  TEXT NOT NULL,  -- session_start|session_end|decision|discovery|error_resolved
    summary     TEXT NOT NULL,
    is_identity BOOLEAN NOT NULL DEFAULT FALSE,
    pulse       BIGINT NOT NULL DEFAULT 0,
    metadata    JSONB NOT NULL DEFAULT '{}',
    created_at  DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now())
);

CREATE INDEX IF NOT EXISTS session_journal_identity_idx
    ON session_journal (is_identity)
    WHERE is_identity = TRUE;


-- ============================================================
-- HOMEOSTASIS_LOG — Set-point deviation tracking
-- ============================================================
CREATE TABLE IF NOT EXISTS homeostasis_log (
    id          BIGSERIAL PRIMARY KEY,
    variable    TEXT NOT NULL,
    set_point   REAL NOT NULL,
    actual      REAL NOT NULL,
    deviation   REAL GENERATED ALWAYS AS (actual - set_point) STORED,
    action_taken TEXT,
    pulse       BIGINT NOT NULL DEFAULT 0,
    created_at  DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now())
);


-- ============================================================
-- AGENT_SESSIONS — Tracks every Antigravity (or external agent)
-- conversation session end-to-end. Core research dataset.
-- ============================================================
CREATE TABLE IF NOT EXISTS agent_sessions (
    id              TEXT PRIMARY KEY,
    agent_id        TEXT NOT NULL DEFAULT 'antigravity',
    started_at      DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    ended_at        DOUBLE PRECISION,
    task_context    TEXT NOT NULL DEFAULT '',
    outcome         TEXT CHECK (outcome IN ('success','partial','failure','ongoing')),
    summary         TEXT,
    rating          REAL CHECK (rating >= 0.0 AND rating <= 1.0),
    what_worked     TEXT,
    what_failed     TEXT,
    memory_count    INTEGER NOT NULL DEFAULT 0,
    hormone_snapshot JSONB NOT NULL DEFAULT '{}',   -- hormone state at session start
    metadata        JSONB NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS agent_sessions_agent_idx ON agent_sessions (agent_id);
CREATE INDEX IF NOT EXISTS agent_sessions_started_idx ON agent_sessions (started_at DESC);
CREATE INDEX IF NOT EXISTS agent_sessions_outcome_idx ON agent_sessions (outcome);


-- ============================================================
-- MEMORIES — Pillar 6 additions (safe ALTER, idempotent)
-- counterfactual_of: fork-point provenance (the "git log")
-- agent_session_id: which agent session wrote this memory
-- ============================================================
DO $$ BEGIN
    ALTER TABLE memories ADD COLUMN counterfactual_of UUID REFERENCES memories(id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

DO $$ BEGIN
    ALTER TABLE memories ADD COLUMN agent_session_id TEXT REFERENCES agent_sessions(id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

CREATE INDEX IF NOT EXISTS memories_counterfactual_idx ON memories (counterfactual_of) WHERE counterfactual_of IS NOT NULL;
CREATE INDEX IF NOT EXISTS memories_agent_session_idx ON memories (agent_session_id) WHERE agent_session_id IS NOT NULL;


-- ============================================================
-- LINEAGE_SNAPSHOTS — Evolver genome archive (Pillar 7)
-- Every accepted mutation is checkpointed here.
-- ============================================================
CREATE TABLE IF NOT EXISTS lineage_snapshots (
    id              BIGSERIAL PRIMARY KEY,
    phase_config    JSONB NOT NULL DEFAULT '{}',
    hormone_genome  JSONB NOT NULL DEFAULT '{}',
    fitness         REAL NOT NULL DEFAULT 0.0,
    session_ratings JSONB NOT NULL DEFAULT '[]',
    accepted_at     DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now()),
    generation      INTEGER NOT NULL DEFAULT 0,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS lineage_snapshots_accepted_idx ON lineage_snapshots (accepted_at DESC);


-- ============================================================
-- THERMAL_FUSIONS — Emergent concept log (Thermorphic substrate)
-- Every semantic fusion spawned by the heat equation is recorded here.
-- The organism's invention history.
-- ============================================================
CREATE TABLE IF NOT EXISTS thermal_fusions (
    id              BIGSERIAL PRIMARY KEY,
    parent_a_id     TEXT NOT NULL,          -- thermorphic node ID (short hash)
    parent_b_id     TEXT NOT NULL,
    child_id        TEXT NOT NULL,
    parent_a_content TEXT NOT NULL DEFAULT '',
    parent_b_content TEXT NOT NULL DEFAULT '',
    child_content   TEXT NOT NULL,
    temp_at_fusion  REAL NOT NULL DEFAULT 0.0,
    memory_id       UUID REFERENCES memories(id) ON DELETE SET NULL,  -- if written to PG
    alpha           REAL NOT NULL DEFAULT 0.08,   -- α at time of fusion (evolver gene)
    pulse           BIGINT NOT NULL DEFAULT 0,    -- cortex pulse count at fusion
    created_at      DOUBLE PRECISION NOT NULL DEFAULT extract(epoch from now())
);

CREATE INDEX IF NOT EXISTS thermal_fusions_created_idx  ON thermal_fusions (created_at DESC);
CREATE INDEX IF NOT EXISTS thermal_fusions_child_idx    ON thermal_fusions (child_id);
CREATE INDEX IF NOT EXISTS thermal_fusions_pulse_idx    ON thermal_fusions (pulse DESC);

-- Tag crystallized memories with their thermal origin node
DO $$ BEGIN
    ALTER TABLE memories ADD COLUMN thermal_node_id TEXT;
EXCEPTION WHEN duplicate_column THEN NULL; END $$;

CREATE INDEX IF NOT EXISTS memories_thermal_node_idx
    ON memories (thermal_node_id)
    WHERE thermal_node_id IS NOT NULL;


-- ============================================================
-- CAUSAL_TRACE — Mutation attribution ledger (Evolution Engine)
-- Records the full causal chain:
--   mutation → parameter_delta → state_vector_delta
--             → behavior_shift → fitness_delta
-- Enables control vs treatment shadow evaluation queries.
-- ============================================================
CREATE TABLE IF NOT EXISTS causal_trace (
    trace_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Mutation layer
    mutation_id     UUID,
    subsystem       TEXT NOT NULL DEFAULT '',   -- RetrievalGenome|DiffusionGenome|MetacognitionGenome
    parameter_name  TEXT NOT NULL DEFAULT '',
    old_value       REAL,
    new_value       REAL,

    -- State impact (captures StateVector snapshot before + after)
    state_before    JSONB NOT NULL DEFAULT '{}',
    state_after     JSONB NOT NULL DEFAULT '{}',
    state_delta     JSONB NOT NULL DEFAULT '{}',

    -- Behavior impact
    behavior_before JSONB NOT NULL DEFAULT '{}',  -- {task_duration, memory_hits, error_rate, ...}
    behavior_after  JSONB NOT NULL DEFAULT '{}',
    behavior_delta  JSONB NOT NULL DEFAULT '{}',

    -- Fitness outcome
    fitness_before  REAL NOT NULL DEFAULT 0.0,
    fitness_after   REAL NOT NULL DEFAULT 0.0,
    fitness_delta   REAL GENERATED ALWAYS AS (fitness_after - fitness_before) STORED,

    -- Stability at time of evaluation
    stability       REAL,                        -- variance(state_vector window)

    -- Context
    task_id         TEXT,
    pulse           BIGINT NOT NULL DEFAULT 0,
    eval_mode       TEXT NOT NULL DEFAULT 'shadow'
                        CHECK (eval_mode IN ('shadow', 'live', 'replay')),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS causal_trace_mutation_idx  ON causal_trace (mutation_id);
CREATE INDEX IF NOT EXISTS causal_trace_subsystem_idx ON causal_trace (subsystem);
CREATE INDEX IF NOT EXISTS causal_trace_fitness_idx   ON causal_trace (fitness_delta DESC);
CREATE INDEX IF NOT EXISTS causal_trace_created_idx   ON causal_trace (created_at DESC);
CREATE INDEX IF NOT EXISTS causal_trace_pulse_idx     ON causal_trace (pulse DESC);
