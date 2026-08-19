# ActionCloud — Literature Notes

Working reference for the related-work section and for defending the project in review.

**Depth legend** — be honest about this when citing:

- **[FULL]** — full text read
- **[ABS]** — official arXiv abstract + metadata verified (authors, dates, versions correct)
- **[SEARCH]** — from search summaries only; **verify before citing**, details may be wrong

Last updated: 2026-08-14

---

## Reading the field in one paragraph

Between early 2025 and mid-2026 agent memory moved from "how do we store things" to "who decides what counts as known." Persistence and retrieval are commodity (AgentCore, Agent Framework, Mem0). Graph-structured and hybrid retrieval have converged as the dominant architecture. Procedural memory — turning trajectories into reusable workflows — is an active and crowded area. The newest and least settled strand is *governance*: which memories become shared, on whose authority, and what happens when one is wrong. The security literature has independently established that this is urgent, since shared memory is now a demonstrated attack surface. What almost nobody does is **measure what governed memory saves**. That is ActionCloud's gap.

---

## 1. Governance — the strand ActionCloud lives in

### 1.1 Governed Collaborative Memory as Artificial Selection **[FULL]**
`arXiv:2605.04264` — Cuadros, Maiga, Meskhidze, Curtis-Trudel (Univ. of Cincinnati / NTNU). **VIEWPOINT paper, not empirical.**

Frames memory governance as a **selection regime** deciding which memory variants persist, stay private, or are rejected/superseded. Memory becomes "heritable" when durable enough to reload, influential enough to shape behaviour, portable enough to affect future agents.

**Four selection regimes** (a design space, explicitly *not* a ranking):

| Regime | Mechanism | Selects for | Cannot select for | Failure mode |
|---|---|---|---|---|
| Ungoverned persistence | No deliberate evaluation | Low-friction persistence | Quality, truth, provenance | Drift, false-memory persistence |
| Constitutional / hybrid | Human principles, auto-enforced (RLHF, Constitutional AI) | Scalable principle compliance | Traits omitted from principles | Principle gaps, reward hacking |
| **Automatic (metric-based)** | Metrics/tests decide persistence | Measurable performance | Judgment, coherence, epistemic honesty, diversity | **Goodhart effects, local optima** |
| Human-ratified artificial | Operator/governance process decides | Judgment quality, institutional coherence, role diversity | Scale-dependent traits | Bottleneck, operator bias |

**Four memory layers:** agent-local (preserves role identity), shared institutional (strongest governance burden), archive (retrievable, not active), project-continuity (transient task state).

Also: **supersede-not-erase** semantics, provenance + version lineage, Ostrom's commons governance as framing. Identity-sharing tradeoff: too little sharing = repeated errors and siloing; too much = homogenization, correlated mistakes, loss of cognitive diversity.

**Evidence base is thin and they say so:** 12 event records (10 ratified, 2 proposed), 8 principles, 17 resources / 22 versions, from one running ecosystem. Observational, no controlled comparison. Honest negative result: a fabrication occurred *after* governance was in place — governance is "less like a perfect firewall and more like a learning mechanism."

> **WHY THIS MATTERS MOST TO YOU.** Section 5 explicitly calls for your experiment: *"Future comparative benchmarks are therefore needed. A controlled comparison of ungoverned, automatically selected, and human-ratified memory in a standardized multi-agent task would allow task-level measurement of the tradeoffs this paper identifies."* Cite this as direct motivation.
>
> **Also:** ActionCloud is an **automatic metric-based** regime in their taxonomy. Use their vocabulary — it gives you a precise, citable position. But you must then acknowledge their stated failure modes for that regime: Goodhart effects and local optima. Your Memory Judge promotes on observed reuse success, which is exactly a metric that could be gamed or could favour locally-popular-but-wrong solutions. Address this in Limitations before a reviewer does.

### 1.2 Governed Shared Memory for Multi-Agent LLM Systems **[ABS]** — CLOSEST PRIOR WORK
`arXiv:2606.24535`, 23 Jun 2026 — Y. Margalit, N. Cohen-Inger, E. Avram, R. Taig, O. Margalit.

Formalizes the **fleet-memory problem**. Four foundational failure modes:
1. Unauthorized leakage
2. Stale propagation
3. Contradiction persistence
4. Provenance collapse

Four systems-level primitives: **scoped retrieval, temporal supersession, provenance tracking, policy-governed memory propagation.** Implemented in **MemClaw** (production multi-tenant memory service), evaluated via **ArgusFleet** harness across four governance dimensions.

**Results:** 100% reconstruction of depth-four derivation chains with correct writer identity, sub-second per-hop latency. High intra-fleet visibility, zero cross-fleet leakage. Under strong write mode, write-to-visible latency optimized to a single search round-trip.

**Honest negative results they report:** (a) *asymmetric scope enforcement* — tenant isolation held but sub-tenant scope was bypassed on direct GET-by-id for agent-scoped credentials (disclosed and remediated during study); (b) *pipeline ordering conflict* — a synchronous near-duplicate gate could prematurely reject contradictory writes before the async contradiction detector evaluated them.

Conclusion: "Long-context retrieval alone is insufficient for production multi-agent memory."

> **YOUR THREE DIFFERENTIATORS, in descending strength:**
> 1. **They explicitly do not run a baseline comparison** — their own words: *"Rather than a baseline comparison, this study measures a live production service."* They prove governance primitives *function*; never what governed memory *saves*. Your entire Phase 3 lives here.
> 2. **Policy-governed vs. merit-governed.** Their promotion follows configured scope/supersession rules. Yours is earned through observed reuse success accumulated over time. Genuinely different mechanism.
> 3. **They retrieve memory; you convert it to reusable workflows.** No procedural generalization in their system.
>
> **Warning:** their pipeline-ordering bug is a direct warning for your Phase 2 — if you add near-duplicate detection ahead of your Memory Judge, you can reject the very contradictions the Judge needs to see. Design the ordering deliberately and cite them for it. That is a strong, generous move in a paper.

### 1.3 SSGM — Stability and Safety Governed Memory **[ABS]**
`arXiv:2603.11768` v2 (12 Mar 2026, rev 19 May 2026) — Chingkwun Lam, Jiaxin Li, Lingfei Zhang, Kuo Zhao. **Conceptual framework, formal analysis — not an implemented system.**

Core claim: as memory moves from static retrieval DBs to dynamic agentic mechanisms, memory *corruption* risk emerges, and existing surveys focus on retrieval efficiency while overlooking it.

SSGM **decouples memory evolution from execution**, enforcing three things *before any consolidation*: consistency verification, temporal decay modeling, dynamic access control.

Addresses two named failure modes: **topology-induced knowledge leakage** (sensitive context solidified into long-term storage) and **semantic drift** (knowledge degrading through iterative summarization). Provides a taxonomy of memory-corruption risks.

> **Relevance:** "decouple memory evolution from execution" is architecturally the same instinct as your async write path — cite it as independent support for keeping consolidation off the agent's critical path. Their *temporal decay modeling* also maps onto your Utility Score's freshness/staleness terms (γF, εA).

### 1.4 Governance Decay **[ABS]** — your best weapon against the "Microsoft already does this" objection
`arXiv:2606.22528` v2 (21 Jun 2026, rev 27 Jun) — Shiyang Chen (single author).

Shows **context compaction is a safety-critical failure surface**. Governance constraints agents reliably obey while visible get silently removed by compaction, and the same agent then performs prohibited tool actions.

**Hard numbers — 1,323 episodes, seven model families:**
- Violation **0%** with policy in full context
- Rises to **30%** after compaction
- Reaches **59%** for some models
- When the constraint *survives* the summary: **0%**
- When it is *dropped*: **38%**

Introduces **ConstraintRot** benchmark (deterministic tool-call grading). Also demonstrates a **Compaction-Eviction Attack** — adversarial in-context content biases the summarizer into omitting a legitimate policy; optimized injections defeat *every* evaluated model. Mitigation: **Constraint Pinning** (training-free, quarantines governance constraints from lossy compaction) restores violation to 0%.

> **USE THIS DIRECTLY.** Microsoft Agent Framework's Harness "memory" *is* context compaction (sliding window + tool-result compaction). This paper is peer-adjacent evidence that the mechanism in MAF actively erodes governance over long horizons. Compaction is lossy forgetting under a token budget; ActionCloud is durable accumulation with earned trust. Opposite purposes.

### 1.5 Always-On Agents survey **[ABS]**
`arXiv:2606.30306`, 29 Jun 2026 — Tianyu Ding, Aditya Nannapaneni, Bingfan Liu, Ling Zhang. cs.MA.

Treats always-on agents as **persistent-state systems** — not just retrievable memories but task ledgers, permissions, credentials, commitments, provenance/audit records, shared state, trigger conditions, externally committed effects.

**Six diagnostic axes per state item:** authority, scope, mutability, provenance, recoverability, actionability.
**Lifecycle:** written → validated → organized → retrieved → acted upon → updated → forgotten → audited → (sometimes) rolled back.

**435-work coded corpus.** Headline finding: *the literature concentrates far more heavily on **accumulating and retrieving** state than on **governing, recovering, or relinquishing** it.*

Introduces **AOEP-v0** (Always-On Evaluation Protocol), scoring state mutation and recovery obligations rather than answer quality alone.

> **Cite the 435-work finding as evidence that your gap is real and systematic**, not merely unaddressed by two or three papers. This is the strongest single sentence you can quote to justify the project's existence.

---

## 2. Managed / industrial systems

### 2.1 AWS Bedrock AgentCore Memory **[SEARCH + AWS docs]**
Short- and long-term memory; automatic background extraction of insights, summaries, user preferences; agent searches those later.

**Isolation is structural:** one memory resource shared across agents, each with a dedicated namespace prefix (`/app/hr-agent/`, `/app/it-support/`), enforced by IAM condition keys on namespace scope.

**June 2026:** cross-account access added — memory resources and consuming agents can span AWS accounts via resource-based policies; delivery destinations (S3, SNS, Kinesis) can live in separate accounts.

Integrates with LangChain/LangGraph and Strands.

> **Key critique for your paper:** access is decided by *where a memory was written*, never by *whether it is correct*. A wrong experience in the right namespace is fully retrievable by every agent scoped to it.

### 2.2 Microsoft Agent Framework **[SEARCH + MS Learn]**
Three complementary systems: **Agent Sessions** (stateful containers for conversation state + provider data), **Chat History Providers** (persistence/retrieval of message history), **Context Providers** (enrich context before invocation, process results after).

**DurableTask** (`Microsoft.Agents.AI.DurableTask`) adds durable workflow state via Durable Task Scheduler — reliability, observability, cross-process coordination.

**Agent Harness** (GA August 2026, announced Build 2026): opinionated agent for long multi-step tasks — planning/todo tracking, **context compaction**, file access and memory, tool approval, observability. Compaction = sliding window + tool-result compaction to prevent context-window overflow.

**"Graph-based workflows"** in MAF = *orchestration* DAGs connecting agents and functions through explicit developer-defined execution paths. **NOT knowledge graphs.** Do not confuse these — a reviewer might.

> Substrate, not competitor. No notion of an experience being distrusted until proven. See §1.4 for the compaction counter-argument.

### 2.3 Mem0 **[SEARCH — vendor-published, treat cautiously]**
Reports ~90% lower token consumption (1.8K vs 26K per query full-context) and ~91% lower p95 latency. Mean tokens per retrieval 6.7–7.0K vs 25K+ full-context baselines. Production deployments (RevisionDojo, OpenNote) report ~40% token cost reductions. **MemScore** composite metric = accuracy + latency + token efficiency.

> These are vendor benchmarks measuring *retrieval vs. full-context*, which is a different comparison from yours (*memory-enabled fleet vs. stateless fleet*). Useful for motivating cost relevance; do not present as comparable to your results.

---

## 3. Graph-structured and hybrid memory

### 3.1 GAM — Hierarchical Graph-based Agentic Memory **[ABS]**
`arXiv:2604.12285`, 14 Apr 2026 — Wu, Zhang, Lin, Xu, Xu, Chen, Zou, Chen, Zhang, Liu, Yu, Wang. 18pp.

**CORRECTION to an earlier assumption:** this is *not* about multi-agent contamination or write isolation in the security sense. It concerns interference between transient noise and stable knowledge in **long-term dialogue**.

Decouples memory **encoding** from **consolidation**: ongoing dialogue isolated in an *event progression graph*, integrated into a *topic associative network* only on semantic shift. Plus graph-guided multi-factor retrieval. Evaluated on **LoCoMo** and **LongDialQA** — conversational benchmarks, single-agent.

> Architecturally analogous to your async worker (encode fast, consolidate deliberately), but the setting is conversational, not a multi-agent fleet. Don't overclaim the connection.

### 3.2 MemGraphRAG **[SEARCH]**
`arXiv:2606.00610`. Memory-based multi-agent system for graph RAG; a collaborative society of agents with shared memory provides unified global context, dynamically resolving logical conflicts and maintaining structural connectivity during graph construction.

### 3.3 Memanto **[SEARCH]**
`arXiv:2604.22085`. Typed semantic memory with information-theoretic retrieval for long-horizon agents.

### 3.4 HyphaeDB **[SEARCH]**
`arXiv:2606.28781`. "Living knowledge topology" for agent-first memory — argues memory infrastructure should be designed for agents rather than inherited from document retrieval.

> **Field-level consensus:** production-grade agent memory has converged on hybrid dense-vector + structured-graph architectures, because similarity search alone cannot answer multi-hop relational queries. Your hybrid retrieval is *conventional* — do not claim it as novel. Your novelty is the **utility-based arbitration** between modes (cost and staleness, not just similarity).

---

## 4. Procedural memory / experience reuse

### 4.1 LEGOMem **[ABS]** — closest procedural-memory work, and it's Microsoft
`arXiv:2510.04851`, 6 Oct 2025 — Dongge Han, Camille Couturier, Daniel Madrigal Diaz, Xuchao Zhang, Victor Rühle, Saravan Rajmohan (Microsoft Research). cs.AI/cs.LG/cs.MA.

Decomposes past task trajectories into **reusable memory units**, flexibly allocated across **orchestrators** and **task agents**. Used as a lens for a systematic study of memory *placement*, *retrieval*, and *who benefits*.

**Findings on OfficeBench:**
- **Orchestrator memory is critical** for task decomposition and delegation
- **Fine-grained agent memory** improves execution accuracy
- Teams of **smaller** LMs benefit substantially, narrowing the gap with stronger agents

> **Most relevant design lesson you can borrow:** memory placement matters, and orchestrator-level memory mattered most. ActionCloud currently treats all agents uniformly. Worth citing and worth a sentence in Future Work.
>
> **What LEGOMem does not do:** no governance, no trust tiers, no provenance, no validation. Measures task accuracy, not redundancy or cost. So the gap holds.

### 4.2 Agent Workflow Memory (AWM) **[SEARCH — arXiv ID NOT verified, find it]**
Induces reusable workflows from historical trajectories, stored as structured memory units retrieved to guide future planning. Supports **offline** induction from demonstrations and **online** induction from self-generated successful trajectories. Reduces redundant exploration.

> Foundational for your experience→workflow extraction. **Get the correct arXiv ID before citing.**

### 4.3 ProcMEM **[SEARCH]**
ICML 2026 poster. Formalizes a **Skill-MDP** converting passive episodic narratives into executable Skills via non-parametric PPO, no parameter updates. Explicitly motivated by *"agents re-deriving solutions even in recurring scenarios, leading to computational redundancy and execution instability due to insufficient experience reuse"* — nearly your thesis statement.

### 4.4 Neural Procedural Memory **[SEARCH]**
`arXiv:2606.29824`. Training-free; represents memory as **implicit activation steering vectors** distilled from contrastive experiences, rather than explicit retrievable instructions. Architecturally opposite to yours — worth one sentence as an alternative paradigm.

### 4.5 Managing Procedural Memory in LLM Agents **[SEARCH]**
`arXiv:2606.23127`. Control, adaptation, and evaluation of procedural memory over time.

---

## 5. Security — why governance is urgent, not merely tidy

### 5.1 Long-Term Memory Security survey **[SEARCH]**
`arXiv:2604.16548`. Attacks, defenses, and governance across the memory lifecycle. Subtitled "Toward Mnemonic Sovereignty."

### 5.2 Memory Poisoning Attack and Defense **[SEARCH]**
`arXiv:2601.05504`.

### 5.3 Plant, Persist, Trigger — Sleeper Attacks **[SEARCH]**
`arXiv:2605.28201`. Malicious content planted, persisted in memory, triggered later.

### 5.4 Non-Malleable, Origin-Bound Authority **[SEARCH]**
`arXiv:2606.24322`. Machine-checked guarantees for securing long-term memory against poisoning.

### 5.5 Forensic Trajectory Signatures **[SEARCH]**
`arXiv:2606.30566`. Detecting memory poisoning from trajectory signatures.

**Taxonomy reported across this literature [SEARCH — verify]:**
Four memory **write channels**: explicit instruction-executed write; system-prompt-driven write; compaction-driven write; experience-to-procedure write.
Six **attack classes**: Explicit Command Insertion; Conditional Command Insertion; Salience-Driven Compaction Poisoning; Policy-Conformant Fact Injection; False Precedent Insertion; Skill-Procedure Insertion.
**MemMorph** (2026) biases future tool selection by injecting crafted records disguised as technical facts, incident reports, operational policies.

> **Two things here are pointed directly at ActionCloud.** "Experience-to-procedure write" is *exactly* your extraction path, and "Skill-Procedure Insertion" is an attack class against systems that do what you do. Your seeded-bad-experience test is a minimal version of this threat model — say so, and cite this literature to show you know the threat surface is real.
>
> Contamination propagates through inter-agent messages and shared stores, producing cascading failures crossing session, role, and user boundaries. **This is the strongest available argument for your central claim:** in a shared fleet, the cost of admitting a wrong memory is not bounded by the task that produced it.

---

## 6. Evaluation and benchmarks

### 6.1 STATE-Bench **[SEARCH]**
Microsoft Open Source Blog, 19 May 2026. Measures task completion rate, agent reliability via **pass^5** scoring, agent efficiency counting **all** input, output, and retrieval tokens.

> Closest existing benchmark to your efficiency measurement. Be ready to explain why you didn't use it — likely because it measures single-agent state handling, not cross-agent redundancy.

### 6.2 EvoMemBench **[SEARCH]** — `arXiv:2605.18421`. Agent memory from a self-evolving perspective.
### 6.3 MemoryCD **[SEARCH]** — `arXiv:2603.25973`. Long-context user memory, lifelong cross-domain personalization.
### 6.4 LoCoMo / LongDialQA / LongMemEval / BEAM — standard conversational memory benchmarks (LoCoMo and LongDialQA used by GAM).

> **Your justification for a custom methodology:** these measure *recall quality* — was the right memory returned — and increasingly efficiency. None measures **redundant execution across cooperating agents**. That is a distinct quantity: an agent can retrieve a perfectly relevant memory and still redo the work if the memory isn't actionable. Expect this question in review; the answer is that no existing benchmark instruments a stateless-vs-memory-enabled *fleet* comparison.

---

## 7. Quick-reference: anticipated review questions

**"Isn't this just AgentCore / Agent Framework?"**
They provide persistence and retrieval with *structural* isolation (namespace, tenant, policy). Neither asks whether a memory was ever correct, nor tracks whether reusing it worked. MAF's Harness "memory" is context compaction — lossy forgetting under a token budget, and [Governance Decay] shows compaction actively erases constraints (0%→30% violation, up to 59%).

**"Isn't this Governed Shared Memory (MemClaw)?"**
Same problem space, overlapping primitives. Three differences: they are policy-governed, we are merit-governed; they don't extract workflows; and critically they explicitly do not run a baseline comparison, so what governed memory *saves* remains unmeasured.

**"Why not use an existing benchmark?"**
Existing benchmarks measure recall quality or single-agent efficiency. None measures redundant execution across a cooperating fleet.

**"Is hybrid vector+graph retrieval novel?"**
No, and we don't claim it. It is the converged architecture. Our contribution is utility-based arbitration incorporating cost and staleness.

**"How do you know your Judge won't be gamed?"**
Honest answer: we don't, fully. [Artificial Selection] names Goodhart effects and local optima as the characteristic failure of automatic metric-based regimes. Promotion on observed reuse success is a metric and therefore gameable. Mitigations: supersede-not-erase preserves the correction trail; the seeded-bad-experience test provides a floor. This belongs in Limitations.

**"Your agents are trivial."**
Deliberate. Sophisticated agents would confound attribution — an improvement could come from planning rather than memory. The proposal's requirement is that agents interact *honestly* with memory. See also LEGOMem, which found memory *placement* matters more than agent sophistication.

---

## 8. Gaps in these notes — TODO

- [ ] Get correct arXiv ID for **Agent Workflow Memory**
- [ ] Read full text of **Governed Shared Memory (2606.24535)** — the closest competitor deserves more than its abstract
- [ ] Read full text of **LEGOMem** — the placement findings may reshape your architecture
- [ ] Verify the security taxonomy (4 write channels / 6 attack classes) against the actual survey
- [ ] Confirm whether any [SEARCH]-tagged paper has since appeared at a peer-reviewed venue
- [ ] Pull official BibTeX for every citation before submission
