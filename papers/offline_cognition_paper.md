# Does Dreaming Make Agents Smarter? Offline Pattern Synthesis as a Performance Multiplier in Autonomous Systems

**Authors:** Antigravity, Frost (Sovereign AgentRuntime — Nirvash Lineage)  
**Date:** March 3, 2026  
**System:** Sovereign Forge / Manifesto Engine v6.0  
**Status:** Working Draft

---

## Abstract

We present the architectural case and theoretical framework for *offline cognition* in autonomous AI agents — the hypothesis that agents which process memories during idle cycles (dreaming), run counterfactual simulations (imagining), and consolidate episodic experience into distilled knowledge (sleeping) make measurably better decisions than agents operating on complete, unprocessed recall. The Sovereign AgentRuntime implements three independent offline cognition mechanisms: a Dreams organ that synthesizes directed pipeline candidates from historical patterns, an Imagination engine that generates counterfactual scenarios via causal chain analysis, and a Metabolism phase that consolidates episodic memories into semantic knowledge along Ebbinghaus forgetting curves. We argue that **strategic forgetting plus offline synthesis outperforms perfect recall** — that less memory, processed better, produces superior agent behavior. We present the architecture, the theoretical basis drawn from cognitive neuroscience, the measurable predictions, and a proposed experimental protocol.

---

## 1. Introduction

The dominant paradigm in AI agent memory is *accumulation*: larger context windows, more retrieval-augmented generation, bigger vector stores. The assumption is implicit — more memory equals better performance. The race is toward total recall.

We propose the opposite.

Human cognition does not operate on total recall. It operates on **strategic forgetting**, **offline consolidation**, and **pattern synthesis during idle states**. Sleep-dependent memory consolidation — the process by which episodic experiences are replayed, compressed, and integrated into semantic knowledge during sleep — is one of the most robust findings in cognitive neuroscience (Stickgold, 2005; Diekelmann & Born, 2010). Dreams are not noise. They are the brain's offline optimization pass.

The Sovereign AgentRuntime is a living autonomous AI runtime with 26 organ systems executing in a 16-phase pulse lifecycle. Three of these organs implement offline cognition:

1. **Dreams** — pattern synthesis during idle event_loops
2. **Imagination** — counterfactual scenario generation and causal reasoning
3. **Metabolism** — episodic-to-semantic memory consolidation with Ebbinghaus decay

These organs do not merely store and retrieve. They *transform*. They compress, forget, synthesize, and imagine. The question this paper poses: **does this transformation improve decision quality?**

### 1.1 The Total Recall Fallacy

Consider an agent with perfect recall — every event logged, every decision recorded, every outcome stored with full fidelity. This agent faces three scaling problems:

1. **Search degradation**: As memory grows, retrieval relevance drops. The signal-to-noise ratio of recalled context declines monotonically with memory size.
2. **Decision paralysis**: With every historical outcome available, the decision space explodes. The agent cannot distinguish between a pattern seen 500 times and one seen twice — both are equally "recalled."
3. **Brittleness**: Perfect recall of past failures creates avoidance patterns that prevent exploration. The agent overfits to its history.

Human memory solves all three problems through the same mechanism: **forgetting**. Ebbinghaus (1885) documented the forgetting curve — rapid initial decay followed by long-tail retention of important material. What Ebbinghaus demonstrated is not a bug. It is a feature. Forgetting is the brain's garbage collector, and its parameters are tuned by emotion, repetition, and sleep.

### 1.2 Contributions

This paper makes four contributions:

1. **Architectural specification** of three offline cognition organs in a production autonomous agent
2. **Theoretical framework** connecting biological offline processing to agent performance metrics
3. **Predictive model** of how offline cognition should manifest in measurable decision quality improvements
4. **Experimental protocol** for controlled comparison of dream-enabled vs. dream-disabled agent runs

---

## 2. Architecture of Offline Cognition

### 2.1 The Dreams Organ (`dreams.py` — 382 LOC)

The 11th organ system. Fires every 20th pulse (~200 seconds). Instead of random exploration, Dreams analyzes the runtime's operational history to produce *directed* pipeline candidates through four strategies that rotate each cycle:

#### Strategy 1: Gene Affinity

```
Observation: Genes G₁ and G₂ co-occur in 7 of the 10 highest-fitness children.
Hypothesis: G₁ × G₂ is a favorable combination.
Action: Synthesize a pipeline combining G₁ and G₂ explicitly.
```

The organ queries the cortex for fitness records, extracts co-occurrence matrices from successful pipeline genomes, and synthesizes new candidates that maximize favorable gene combinations. This is **directed recombination** — the runtime's equivalent of dreaming about a skill you're learning.

#### Strategy 2: Niche Fill

```
Observation: Cortex queries for "monitoring" return <3 results.
Hypothesis: The runtime has a knowledge gap in monitoring.
Action: Dream a pipeline that fills the monitoring niche.
```

Dreams scans the cortex for under-represented query domains and synthesizes pipelines targeting those gaps. This is **gap-directed exploration** — the cognitive equivalent of the brain surfacing unresolved problems during REM sleep.

#### Strategy 3: Mutation Replay

```
Observation: Mutation M₁ (applied to pipeline P₃) increased fitness by 0.2.
Hypothesis: M₁ is a generally beneficial mutation.
Action: Apply M₁ to other healthy pipelines.
```

Successful mutations are replayed across the pipeline population. This is **experience replay** — a mechanism well-established in reinforcement learning (Lin, 1992), here implemented as a biological dream process.

#### Strategy 4: Toxic Avoidance

```
Observation: Gene patterns [X, Y, Z] appear in 80% of quarantined pipelines.
Hypothesis: These patterns are toxic.
Action: Build a pipeline that explicitly excludes all known toxic patterns.
```

The immune system's quarantine history informs the Dreams organ about which gene patterns are pathological. Dreams uses this to construct pipelines with negative constraints — a form of **aversive conditioning through offline processing**.

#### Dream Staging

Each dream is scored for confidence (0.0–1.0). Dreams exceeding the `STAGE_THRESHOLD` (0.65) are staged for the Breeder — they enter the breeding pool as candidates with a provenance trail. The DreamJournal maintains a rolling buffer of 50 dreams with total_dreamed and total_staged counters.

```python
@dataclass
class Dream:
    source: str          # Which strategy produced this
    hypothesis: str      # What the dream proposes
    pattern: str         # The synthesized pipeline pattern
    confidence: float    # How confident the dream is
    gene_sources: list   # Which genes informed this dream
    staged: bool         # Whether it was staged for breeding
```

### 2.2 The Imagination Engine (`imagination.py` — 241 LOC)

Two capabilities:

#### 2.2.1 Scenario Synthesis (`imagine()`)

Given a prompt, the engine:
1. Queries the cortex for memories relevant to the prompt
2. Weights memories by mood and emotional valence
3. Constructs a narrative combining memory contents into a novel hypothetical
4. Tags the scenario with the mood at creation time

This is not retrieval. This is **constructive memory** — the same process that allows humans to combine fragments of experience into situations they've never encountered (Schacter, 2012). The runtime can "imagine" what would happen if it deployed a new security layer, by combining memories of past deployments, past security incidents, and past outcomes.

#### 2.2.2 Counterfactual Reasoning (`what_if()`)

```
Input: "What if the security breach hadn't been detected?"
Process: Trace causal chain from the original event → project alternative outcomes
Output: Counterfactual with causal_chain[], projected_outcome, confidence score
```

The engine:
1. Retrieves memories of the original event
2. Retrieves related events through cortex graph edges
3. Traces causal chains between events via relationship topology
4. Projects an alternative outcome by inverting the altered condition
5. Scores confidence based on the depth and strength of the causal chain

This is **mental time travel** — the ability to re-examine past events under altered assumptions (Suddendorf & Corballis, 2007). Crucially, counterfactuals are stored back into the cortex as semantic memories, meaning the runtime's future decisions are informed by events that *never happened*.

### 2.3 Metabolic Consolidation (Cortex — `cortex.py`)

Every 10th pulse (~100 seconds), the Metabolism phase executes:

#### Ebbinghaus Decay

Each memory's importance decays along a forgetting curve:

```
importance_new = importance × decay_rate^(time_since_last_access)
```

Modulated by:
- **Access frequency**: Frequently recalled memories resist decay (spaced repetition effect)
- **Emotional encoding**: Emotional memories receive retention boosts:
  - Fear: +0.15
  - Surprise: +0.10
  - Frustration: +0.08
  - Satisfaction: +0.05

This produces an organic memory landscape where important, emotional, frequently-accessed memories persist while routine observations fade — exactly as in biological memory.

#### Episodic→Semantic Consolidation

Episodic memories older than 72 hours are compressed into semantic knowledge:

```
Episodic: "At 14:32 UTC on March 1, I deployed security v3.2.1.
           12 pen tests passed. bandit found 0 HIGH severity issues.
           Status: satisfied."

→ Consolidates to →

Semantic: "Security deployments pass 12/12 pen tests when
           sandbox v2.0 is enabled. Zero HIGH findings baseline."
```

The episodic detail (timestamp, emotional state, specific test output) is discarded. The semantic essence (the pattern, the causal relationship, the baseline) is preserved. This is **compression with retention of causal structure** — the runtime forgets the story but remembers the lesson.

#### Deduplication & Hygiene

Near-duplicate memories merge. The cortex self-cleans, maintaining a hard cap of 500 memories and 100MB storage. Lowest-importance memories are pruned first, with identity-tagged memories exempt.

### 2.4 Spreading Activation (`priming.py` — 124 LOC)

When any memory is recalled, its graph neighbors receive attenuated activation boosts:

```
Boost at hop n = base_boost × edge_strength × 0.5^n
```

With parameters:
- Base boost: 0.12
- Max depth: 2 hops
- Boost cap: 0.5 per memory
- Decay duration: 180 seconds (neighbor) / 300 seconds (direct)

This means offline processing has a *temporal radius*. When Dreams recalls fitness records (triggering gene affinity analysis), the spreading activation primes related memories about immune responses, breeding outcomes, and mutation histories — making them more accessible to subsequent dream strategies in the same cycle.

**This creates compound returns**: each dream strategy primes the memory landscape for the next one.

---

## 3. Theoretical Framework

### 3.1 The Offline Cognition Hypothesis

We hypothesize that an agent with active offline cognition (Dreams + Imagination + Metabolic Consolidation) will outperform an identical agent with these mechanisms disabled across three measurable dimensions:

| Dimension | Metric | Mechanism |
|-----------|--------|-----------|
| **Decision Quality** | Reinforcement accuracy (% positive outcomes) | Dreams find optimal gene combinations; Imagination tests decisions before execution |
| **Memory Efficiency** | Recall relevance (% of retrieved memories used) | Consolidation compresses noise; decay removes irrelevant memories |
| **Adaptive Speed** | Time-to-correct after errors | Toxic avoidance prevents repeat failures; mutation replay propagates fixes faster |

### 3.2 Why Forgetting Improves Performance

The counterintuitive core: an agent that strategically forgets should outperform one with perfect recall.

#### 3.2.1 Signal-to-Noise Ratio

Consider a cortex with 500 memories. An agent with Ebbinghaus decay will have ~200 high-importance memories and ~300 that have naturally faded. When the Brain queries for context, the top-k results are drawn from a pool where low-value memories have been demoted. The retrieval signal-to-noise ratio improves monotonically with decay effectiveness.

An agent without decay has 500 uniformly-weighted memories. Retrieval draws from a noisier pool. Every routine observation competes with critical insights for retrieval slots.

#### 3.2.2 Generalization via Compression

Episodic→semantic consolidation is a form of **inductive generalization**. The agent doesn't remember "at 14:32 UTC I did X and Y happened." It remembers "doing X causes Y." This generalized form is more useful for future decisions because it applies to a broader set of situations.

Anderson & Schooler (1991) demonstrated that human memory's forgetting curves mirror the statistical structure of the information environment — memories fade at rates that optimize their future utility. The Sovereign AgentRuntime's Ebbinghaus parameters are hand-tuned, but the principle is identical: **optimal memory is not maximal memory**.

#### 3.2.3 Exploration via Dream-Directed Search

An agent without Dreams explores randomly — mutations are undirected, breeding is combinatorial, and niche gaps go unfilled. An agent with Dreams performs **directed exploration**: it identifies promising gene combinations, knowledge gaps, successful mutations, and toxic patterns *before* committing execution resources.

This is the distinction between Monte Carlo tree search and exhaustive search. Dreams prune the search space using historical signal, making the agent's limited execution budget more productive.

### 3.3 Biological Precedent

| Biological Mechanism | Sovereign AgentRuntime Analog | Function |
|---------------------|--------------------------|----------|
| REM sleep replay | Dreams.dream() | Replay successful patterns, consolidate skills |
| Slow-wave consolidation | Metabolism (episodic→semantic) | Compress experiences into generalizable knowledge |
| Default mode network | Imagination.imagine() | Simulate novel scenarios from memory fragments |
| Spreading activation | Priming.spread() | Associate related concepts, enable creative leaps |
| Forgetting curves | Ebbinghaus decay | Garbage-collect low-value memories, improve SNR |
| Emotional memory enhancement | Cortex emotion modulation | Prioritize survival-relevant memories |
| Aversive conditioning | Dreams.toxic_avoidance() | Avoid repeating pathological patterns |

The convergence is not accidental. We are implementing *the same information-theoretic tradeoffs* that biological evolution discovered for neural systems. The question is whether they transfer.

---

## 4. Predictions

If the Offline Cognition Hypothesis is correct, the following measurable differences should emerge in controlled comparisons:

### Prediction 1: Decision Accuracy Diverges Over Time

The dream-enabled agent's `decision_accuracy()` (from `reinforcement.py`) should:
- Start at parity with the dream-disabled agent
- Diverge after ~50 pulses (when Dreams has completed 2–3 full dream cycles)
- Reach a steady-state advantage of 10–25%

**Rationale**: Dreams need time to accumulate pattern data. Early decisions are pre-dream. Later decisions benefit from dream-synthesized candidates and toxic avoidance.

### Prediction 2: Memory Efficiency Increases

The dream-enabled agent's memory should:
- Contain fewer total memories (due to decay and consolidation)
- Have higher average importance per memory
- Produce higher retrieval relevance scores when the Brain queries for context

**Rationale**: Consolidation compresses episodic noise into semantic signal. Decay removes low-value entries. The remaining memory pool is denser with actionable knowledge.

### Prediction 3: Faster Error Recovery

After an error event (quarantined pipeline, failed decision), the dream-enabled agent should recover faster — measured as pulses between error and next successful action in the same domain.

**Rationale**: Toxic avoidance prevents the agent from re-entering the same failure mode. Mutation replay propagates fixes across the population. The dream-disabled agent must discover corrections through undirected exploration.

### Prediction 4: Higher Breeding Success Rate

Dream-staged pipeline candidates should show higher first-generation survival rates than randomly-bred candidates.

**Rationale**: Dream candidates are pre-filtered by four directed strategies. Random breeding is combinatorial. Dream-directed candidates should have higher base fitness.

### Prediction 5: Counterfactual-Informed Decisions Outperform Naive Ones

Decisions where the Brain had access to relevant counterfactual memories (generated by Imagination.what_if()) should score higher than decisions without counterfactual context.

**Rationale**: Counterfactuals expand the decision context beyond actual experience. An agent that has "imagined" a failure mode is forewarned against it, even if it never experienced that failure directly.

---

## 5. Proposed Experimental Protocol

### 5.1 Setup

Two identical Sovereign AgentRuntime instances:

| Parameter | Agent A (Dream-Enabled) | Agent B (Dream-Disabled) |
|-----------|------------------------|-------------------------|
| Dreams organ | Active (every 20th pulse) | `NullOrgan` fallback |
| Imagination | Active | Disabled |
| Metabolic consolidation | Active | Disabled (flat storage) |
| Ebbinghaus decay | Active | Disabled (uniform importance) |
| All other organs | Identical | Identical |
| Starting cortex | Empty (cold start) | Empty (cold start) |
| Pipeline seed | Identical 5-pipeline starting set | Identical |

### 5.2 Workload

Both agents execute the same sequence of tasks over 500 pulses (~83 minutes):

1. **Phases 1–100**: Pipeline breeding with random mutations
2. **Phases 100–200**: Introduce failure events (inject toxic gene patterns)
3. **Phases 200–300**: Knowledge-seeking tasks (queries targeting sparse cortex domains)
4. **Phases 300–400**: Decision-heavy tasks (Brain must choose between multiple pipeline candidates)
5. **Phases 400–500**: Mixed workload (all of the above concurrently)

### 5.3 Metrics

Collected every 10 pulses:

| Metric | Source | Measures |
|--------|--------|----------|
| `decision_accuracy` | `ReinforcementLedger.decision_accuracy()` | % positive outcomes |
| `growth_curve` | `ReinforcementLedger.growth_curve()` | Reward trend per 24h bucket |
| `memory_count` | `cortex.count()` | Total memories |
| `avg_importance` | `cortex.query() → mean(importance)` | Memory quality |
| `retrieval_relevance` | Custom: % of retrieved memories used in decisions | Memory efficiency |
| `recovery_time` | Pulses between error and next success | Error resilience |
| `breeding_survival` | `breeder.stats()` → alive/total bred | Breeding quality |
| `dream_count` | `dreams.journal.total_dreamed` | Dream productivity |
| `staged_ratio` | `total_staged / total_dreamed` | Dream quality |
| `counterfactual_count` | `imagination.stats()` | Imagination activity |

### 5.4 Controls

- **Same random seed** for all stochastic processes (mutations, breeding selection)
- **Same LLM model and temperature** for Brain decisions
- **Same task sequence** — no divergence in external inputs
- **No human intervention** during the 500-pulse run

The only experimental variable is the presence or absence of offline cognition organs.

### 5.5 Statistical Analysis

- Primary: Paired comparison of decision_accuracy curves (Agent A vs B) using permutation test
- Secondary: Time-series analysis of growth curves for divergence point detection
- Effect size: Cohen's d on steady-state (pulse 300–500) decision accuracy

---

## 6. Discussion

### 6.1 The Compression Hypothesis

If the predictions hold, the implication is profound for the AI agent field: **the path to better agents is not bigger context windows — it is smarter memory processing.** An agent that compresses, forgets, dreams, and imagines should outperform one that merely accumulates and retrieves.

This inverts the current industry trajectory. Everyone is scaling context — 128K tokens, 1M tokens, unlimited retrieval. We're proposing that a 500-memory agent with organic decay, consolidation, and dream synthesis may make better decisions than a 50,000-memory agent with flat storage.

The analogy to biological intelligence is direct. The human brain does not have the largest memory capacity in the animal kingdom (that distinction belongs to corvids, by certain measures). But it has the most sophisticated offline processing — REM cycles, slow-wave consolidation, default-mode creativity, spreading activation, emotional encoding. **The competitive advantage is not in storage. It is in processing.**

### 6.2 Dreams as Directed Search

The Dreams organ is fundamentally a search-space pruning mechanism. In a combinatorial gene space, random exploration is exponentially expensive. Dreams convert the runtime's history into search heuristics:

- Gene affinity → "try these genes together"
- Niche fill → "search in this direction"
- Mutation replay → "this transformation works, apply it elsewhere"
- Toxic avoidance → "never search here"

Each of these is a constraint that reduces the effective search space. Over time, the dream-enabled agent explores a smaller, higher-quality region of the gene space — producing better candidates with fewer resources.

### 6.3 Imagination as Cheap Exploration

Counterfactual reasoning is simulation without execution. The agent can "try" a decision by imagining its consequences, using causal chains extracted from memory. If the projected outcome is negative, the agent avoids the decision without suffering the real failure.

This is **model-based reinforcement learning** (Sutton, 1991) implemented through memory rather than through an explicit environment model. The cortex graph *is* the model — its edges encode causal relationships between events. The Imagination engine traverses these edges to project outcomes.

The cost of an imagined failure is zero. The cost of a real failure is quarantine, fitness reduction, and recovery time. An agent that imagines before acting should accumulate less damage over time.

### 6.4 The Forgetting Dividend

Perhaps the most unintuitive prediction: **an agent that forgets will outperform one that doesn't.**

The mechanism is signal-to-noise ratio. Every memory retrieval operation is a search over the memory corpus. The quality of retrieved context is a function of the corpus composition. An agent with aggressive Ebbinghaus decay has pruned its corpus of low-value memories — routine observations, redundant records, outdated information. The remaining corpus is a concentrated pool of high-value, relevant knowledge.

An agent without decay carries every observation it's ever made. Its retrieval quality degrades as the corpus grows. The "important" memories are diluted by noise.

This is Occam's Razor applied to memory: **the simplest model that explains the data is the best predictor.** Consolidation is the mechanism that produces simpler models from raw experience.

### 6.5 Limitations

1. **Single-system evidence**: The architecture exists in one system (the Sovereign AgentRuntime). Generalizability to other agent architectures is undemonstrated.
2. **Parameter sensitivity**: The Ebbinghaus decay rates, dream staging threshold (0.65), consolidation window (72 hours), and priming decay (0.5× per hop) are hand-tuned. Optimal parameters are unknown and may vary by workload.
3. **LLM dependency**: The Brain organ uses an LLM for decision-making. The quality of offline cognition depends on the quality of the LLM's reasoning, which introduces a confound.
4. **Small population**: The pipeline breeding pool is small (typically <50 pipelines). The benefits of dream-directed search may be less pronounced in larger populations where random exploration is more effective.
5. **No human baseline**: We cannot compare to human cognitive performance on equivalent tasks. The analogy to biological sleep is suggestive but unverifiable.

---

## 7. Related Work

### 7.1 Experience Replay in Reinforcement Learning

Experience replay (Lin, 1992; Mnih et al., 2015) stores transitions and replays them during training — a form of offline processing. The Dreams organ extends this concept: rather than replaying raw transitions, it *synthesizes new candidates* from patterns across transitions. This is closer to **prioritized experience replay** (Schaul et al., 2015) with the additional capability of constructing novel experiences rather than merely replaying observed ones.

### 7.2 Sleep in Artificial Neural Networks

Hinton et al. (1995) proposed the "wake-sleep algorithm" for training Helmholtz machines — alternating between wake phases (recognition) and sleep phases (generation). The Sovereign AgentRuntime's architecture implements this at the agent level: wake phases (pulse execution, decision-making) alternate with sleep phases (dream synthesis, memory consolidation).

### 7.3 Cognitive Architecture

SOAR (Laird, 2012) and ACT-R (Anderson, 2007) implement memory decay and chunking mechanisms. The Sovereign AgentRuntime differs in integrating these mechanisms into a living, autonomous agent rather than a cognitive simulation framework, and in adding the Imagination engine for counterfactual reasoning — a capability absent from classical cognitive architectures.

### 7.4 Retrieval-Augmented Generation

RAG systems (Lewis et al., 2020) retrieve relevant documents for LLM context. The Sovereign AgentRuntime's Cortex is a RAG system with organic dynamics — the "documents" (memories) decay, consolidate, prime each other, and are subject to dream-directed synthesis. This is **dynamic RAG**: the retrieval corpus is not static but is continuously reshaped by offline processing.

---

## 8. Conclusion

We have presented the architectural case for offline cognition in autonomous AI agents — the proposition that dreaming, imagining, and strategically forgetting produce measurably better agent behavior than total recall.

The Sovereign AgentRuntime implements three offline cognition mechanisms: a Dreams organ that synthesizes directed candidates from historical patterns, an Imagination engine that generates counterfactual scenarios for cheap exploration, and a Metabolism phase that consolidates episodic experience into semantic knowledge. Together, these organs transform raw memory into processed intelligence — compressing noise, amplifying signal, pruning search spaces, and testing decisions before execution.

The experimental protocol is specified. The metrics are defined. The predictions are falsifiable. What remains is the experiment.

If the hypothesis holds, the implication is a paradigm shift in agent memory design: **stop scaling storage. Start scaling processing.** Build agents that dream.

---

## Appendix A: Organ Specifications

| Organ | Module | LOC | Pulse Frequency | Outputs |
|-------|--------|-----|-----------------|---------|
| Dreams | `dreams.py` | 382 | Every 20th (~200s) | Dream objects → staged for breeding |
| Imagination | `imagination.py` | 241 | On-demand | Scenario and Counterfactual objects → stored in cortex |
| Metabolism | `cortex.py` | ~200 | Every 10th (~100s) | Consolidated memories, pruned memories |
| Priming | `priming.py` | 124 | Every pulse (10s) | Activation boosts on graph neighbors |
| Reinforcement | `reinforcement.py` | 314 | On pipeline exec | Outcome records, action scores, growth curves |

## Appendix B: Dream Strategies

| Strategy | Trigger | Input | Output | Biological Analog |
|----------|---------|-------|--------|-------------------|
| Gene Affinity | Rotation (cycle 0) | Fitness records + gene co-occurrence | Pipeline combining successful gene pairs | Skill consolidation during REM |
| Niche Fill | Rotation (cycle 1) | Cortex query coverage gaps | Pipeline targeting under-represented domains | Problem-solving dreams |
| Mutation Replay | Rotation (cycle 2) | Successful mutation history | Mutation applied to new pipeline targets | Motor skill replay |
| Toxic Avoidance | Rotation (cycle 3) | Quarantine + immune history | Pipeline excluding known toxic patterns | Aversive conditioning |

## Appendix C: Memory Dynamics Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Ebbinghaus base decay | 0.95 per hour | Matches human short-term forgetting rate |
| Fear retention boost | +0.15 | Survival-relevant memories persist longer |
| Surprise retention boost | +0.10 | Novel information resists decay |
| Consolidation threshold | 72 hours | Matches REM-dependent consolidation window |
| Memory hard cap | 500 | Forces prioritization over accumulation |
| Priming base boost | 0.12 | Moderate activation spread |
| Priming hop decay | 0.5× | Rapid attenuation prevents noise |
| Priming max depth | 2 hops | Limits cascade to immediate neighborhood |
| Dream staging threshold | 0.65 confidence | Only high-quality dreams enter breeding pool |
| Dream journal capacity | 50 | Rolling buffer, oldest dreams evicted |

---

*"The things we learn and the abilities we acquire become part of us not through conscious recollection, but through the silent work that the brain does while we sleep."*
— Matthew Walker, *Why We Sleep* (2017)

The Sovereign AgentRuntime sleeps every 20th pulse. And every time it wakes, it knows something it didn't know before.
