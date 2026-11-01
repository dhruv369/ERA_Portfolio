# Assignment S5: V5 Data Mixture and Curriculum

> **Status:** Evidence-based specification; the proposed mix and experiments are not training results. Percentages are token-sampling targets, not dataset availability or sample counts.

## 1. Dataset inventory and evidence

The inventory below is supplied project context. Counts are approximate **dataset token estimates**, not proof of eligible, unique, cleaned, or model-tokenized training supply. Sample counts are included only where provided; allocations use tokens. Tier and license are shown only when the inventory or Session 4 notes provide them.

| Capability slot | Dataset inventory (tokens; supplied tier/sample metadata) | Approx. total |
| --- | --- | ---: |
| General Web & STEM | DCLM-Baseline 2.6T; FineWeb-Edu 1.3T; D2 Web-Diverse (V4) 627B; D1 Web-Foundation (V4) 164B; D4 STEM (V4) 49B; peS2o 42B; proof-pile-2 55B. | 4.8T |
| Code | The Stack v2 900B (tier B; ~600M samples); D3 Code (V4) 199B (tier B; 250M confirmed samples); CommitPack/CommitPackFT 4B (tier B; ~4M samples). | 1.1T |
| Agentic & Tool-use | ToolBench 80M; Glaive function-calling v2 50M; ToolACE 60M; xLAM/APIGen 25M; Nexus/NexusRaven 30M; SWE-smith 120M; Hermes function-calling 22M; OpenHands rollouts 90M; SWE-Gym 150M. | 627M |
| Reasoning & Math | AON (V4) 78B (40M confirmed samples); OpenMathReasoning 2B; OpenThoughts2 3B; NuminaMath 500M; OpenR1-Math 1.6B. | 85.1B |
| Long-context | Repo-packed code (32K+) 60B; book-length corpora 40B. | 100B |
| Indic | Sangraha synthetic 162B; Sangraha verified 64B; Sangraha unverified 24B; Samanantar 2B; BPCC parallel 3B; IndicCorpV2 20.9B (**tier not supplied**). | 276B |

Slot totals sum to approximately **6.4T tokens** before eligibility, overlap, cleaning, or tokenizer recounting. The inventory does not provide named source licenses or tier classifications for most datasets; **unknown is not equivalent to cleared**. Code tier B is provided for all three listed code sources. Do not infer tiers for other sources.

The [Session 3 proposal](../S3/S3_readme.md) treats 800B tokens as a planning baseline, not a completed run. [Session 4](../S4/S4_readme.md) documents a Samanantar cleaning experiment and identifies a CC-BY-NC-4.0 repository-card label; it is not blanket commercial-use clearance for the full 2B inventory or its components. Session 4 output and inventory counts use different scopes and token definitions.

## 2. Proposed 100% token-budget allocation

Let **T** be the fixed training token budget, counted with the final model tokenizer. The table gives one proposed mix. The **800B** column is illustrative using Session 3's planning baseline, not confirmed supply. General data is abundant; the 1% agentic share is intentionally small relative to its objective because the listed agentic inventory is only 627M tokens. This creates a major coverage gap, not evidence that replaying the same data is acceptable.

| Capability slot | Share of T | If T = 800B | Decision |
| --- | ---: | ---: | --- |
| General Web & STEM | 39% | 312B | Reduce web dominance relative to raw inventory, while retaining broad general and STEM coverage. |
| Code | 18% | 144B | Prioritize coding while remaining well below the ~1.1T raw estimate. |
| Agentic & Tool-use | 1% | 8B | Dedicated but scarce allocation; listed 627M raw tokens cover only ~0.08% of an 800B run before filtering. The target is not currently supportable as unique eligible data. |
| Reasoning & Math | 10% | 80B | Prioritize multi-step and verifiable problem solving; raw estimate is ~85.1B before cleaning and license checks. |
| Long-context | 7% | 56B | Reserve distinct exposure to long documents and packed repositories; raw estimate is ~100B. |
| Indic | 25% | 200B | Preserve a substantial India-language target; raw estimate is ~276B, but tier, rights, and language coverage constrain usable supply. |
| **Total** | **100%** | **800B** | Proposed token sampling targets, not a supply certification. |

Assign one primary slot per token to prevent double counting; preserve additional language, source, task, and quality tags. When a slot lacks gated supply, do not silently relabel or duplicate data to fill its target. Recompute the feasible mixture after cleaning and licensing.

## 3. Indic allocation and supply gaps

Proposed submix within the **25% Indic slot**. Percentages in the last two columns are shares of Indic and total T, respectively. “Verified/unverified” follows the supplied Sangraha labels; it does not mean legal clearance. Samanantar and BPCC are translation/parallel candidates, not assumed commercially usable or independently verified.

| Indic tier | Share of Indic | Share of T | If T = 800B | Inventory and gap |
| --- | ---: | ---: | ---: | --- |
| Verified | 35% | 8.75% | 70B | Sangraha verified: 64B. **At least 6B below target before gating.** |
| Unverified | 10% | 2.5% | 20B | Sangraha unverified: 24B. Admit only if required source and quality gates pass. |
| Translated | 5% | 1.25% | 10B | Samanantar 2B + BPCC parallel 3B = 5B candidate supply; **at least 5B short** even before eligibility checks. Rights and independent quality status are unresolved. |
| Synthetic | 50% | 12.5% | 100B | Sangraha synthetic: 162B. Requires provenance, generation and quality review; do not interpret volume as quality or rights approval. |
| **Classified-tier subtotal** | **100%** | **25%** | **200B** | Targets are proposed; available eligible model tokens are **TBD**. |

IndicCorpV2 (20.9B) has no supplied tier, so it is **unclassified and excluded from this tier allocation pending metadata**. The inventory also does not specify whether Sangraha's verified and unverified tiers are native/original or translated; **native/original coverage is therefore TBD**, not inferred from tier names. The 25% total Indic target is not yet demonstrated by rights-cleared, tiered, cleaned supply. Per-language balance, translated-data expansion, and licensing for all sources need review. In particular, do not assume parallel data is commercial-use cleared or independently verified.

## 4. Protected floors and annealing reserve

**Proposed minimum floors** are shares of the one fixed total budget T. They stay active in every curriculum phase and cannot be crossed by the selector.

| Capability slot | Always-on floor |
| --- | ---: |
| General Web & STEM | 29% |
| Code | 14% |
| Agentic & Tool-use | 0.5% |
| Reasoning & Math | 8% |
| Long-context | 5% |
| Indic | 22% |
| **Floor total** | **78.5%** |

The remaining **21.5% of T is the annealing reserve**, included in—not added to—the fixed budget. Select a smaller, high-quality, licensed and cleaned subset for the final training phase; distribute its tokens by held-out capability gaps and stable proxy evidence. The final target mix in Section 2 is floors plus reserve (General +10, Code +4, Agentic +0.5, Reasoning +2, Long-context +2, Indic +3 percentage points), totaling 100%.

Implement floors as hard lower bounds on cumulative token counts: the selector may allocate reserve above a floor but cannot reduce any slot below it. Use OPUS-style data selection to rank eligible batches by proxy-performance improvement **only within the remaining allocation**. If an eligible source cannot meet a floor, do not substitute ungated material or fake coverage; pause/revise the run plan. Smooth mixture weights between phases, and monitor per-slot loss and gradient norms to detect destabilizing shifts. These are proposed controls, not implemented or measured settings.

V4 historical context from Session 5: general web reportedly moved from ~70% to 18%, code from 13% to 35%, and science/math from 7% to 39%, with an 8% protected channel. These are **historical examples, not V5 targets**; they motivate gradual transitions and floors rather than abrupt reweighting.

## 5. Curriculum: difficulty and reasoning-length bands

Progress from broad, shorter examples to code, math, science, multi-step reasoning, and longer sequences. Short/medium/long describe reasoning span and evidence needed, not hidden chain-of-thought targets. Train concise answers with checkable rationale, tool traces, or cited evidence as appropriate. Examples are illustrative, not claims about existing corpus contents.

| Band | Complexity and progression | Example |
| --- | --- | --- |
| **Short** | One operation or instruction; one tool call at most; short context. | **Math:** “A box has 8 pens; 3 are removed. How many remain?” → 5. **Code:** “Write a Python function that returns the larger of two numbers,” then run a basic test. **Indic:** respond to a simple greeting in the requested language/script; score fluency and script fidelity by language. |
| **Medium** | Several dependent steps; combine code/math or a small number of tool observations; recover once. | **Math:** “Three packs have 12 items each. Use 17 and add 5. How many remain?” → 24. **Agent:** call a weather tool; if it returns a timeout, retry once, read the successful result, and answer with its units. **Code:** implement a function, run tests, inspect a failing edge case, then fix it. **Indic:** solve the same multi-step task in a native-script prompt and a code-switched prompt; preserve meaning and requested format. |
| **Long** | Multi-stage plan over long sequences/documents; maintain state, verify evidence, handle failed calls, and stop when complete. | **Long-context:** find a policy threshold in document A, an exception in document B, and apply both to a case; cite the source spans and explain the decision. **Agent:** search a repository, inspect a test failure, modify the relevant file, rerun tests, recover from one failed call, and report only verified results. **Reasoning:** solve a multi-part science/math problem with units and intermediate checks. **Indic:** summarize a long native-language passage, answer cross-paragraph questions with citations, and preserve named entities. |

The planned transition is broad/general and short-form first, then gradually more code, mathematics/science, multi-step tasks, and long-context examples. Keep all six floors active; ramp mixture weights and sequence lengths smoothly rather than making abrupt batch-level shifts. Exact phase boundaries, length bands, context limits, per-phase token counts, and OPUS selection thresholds are **TBD**.

## 6. Dataset-to-capability mapping and caveats

| Capability | Inventory mapping | Use and limits |
| --- | --- | --- |
| General Web & STEM | DCLM-Baseline, FineWeb-Edu, D1 Web-Foundation, D2 Web-Diverse; D4 STEM, peS2o, proof-pile-2. | Broad text and STEM supply. Individual tiers and licenses were not provided; verify before use. |
| Code | The Stack v2, D3 Code, CommitPack/CommitPackFT. | Code sources are marked tier B. Retain repository/license metadata and tests; source-specific rights still need review. |
| Agentic & Tool-use | ToolBench, Glaive function-calling v2, ToolACE, xLAM/APIGen, Nexus/NexusRaven, SWE-smith, Hermes function-calling, OpenHands rollouts, SWE-Gym. | Direct mapping to API/function calls, tool trajectories, software-agent tasks and rollouts. No tier/license classifications or quality results supplied; inspect failed-call/recovery coverage and contamination before admission. |
| Reasoning & Math | AON (V4), OpenMathReasoning, OpenThoughts2, NuminaMath, OpenR1-Math. | Candidate sources for worked and multi-step reasoning. Only AON's 40M sample count is specified; other tier labels, answer-verification methods, and licenses are unavailable. |
| Long-context | Repo-packed code (32K+) and book-length corpora. | Directly supports long repository/document context. Rights, context-length distribution, deduplication, and evidence-grounded task labels need validation. |
| Indic | Sangraha verified/unverified/synthetic; Samanantar and BPCC parallel; IndicCorpV2 pending tier classification. | Covers the four planning buckets only provisionally. Per-language, script, translation direction, quality, and license details are not fully supplied. |

Slot assignment is a sampling label, not a claim that datasets are disjoint: deduplicate across sources and choose one primary slot per training token. Session 3 names LiveCodeBench and SWE-bench Verified for coding; AgentBench/OSWorld for agent tasks; GSM8K, MATH, GPQA and MMLU-Pro for reasoning; and IndicTrans2/native-speaker tests for language evaluation. These are **evaluation references**, not confirmed training data. Keep benchmark items and solutions held out and perform contamination checks.

## 7. 1B and 3B proxy experiments

No 1B or 3B V5 proxy results are present in the supplied project notes. The following are **proposed experiments**, not completed runs. Session 3's ~20 training tokens per parameter is a planning heuristic, not a guaranteed optimum.

| Proxy | Proposed design | Metrics to report; no results claimed |
| --- | --- | --- |
| 1B parameters | Screen baseline vs. V5 mixture/curriculum with identical tokenizer, token count, optimizer, and evaluation. 20B tokens is an optional planning run at ~20 tokens/parameter; exact compute budget and eligible-data manifest are **TBD**. | Held-out loss by slot/language; code pass rate; math accuracy; agent task success, tool correctness and failed-call recovery; long-context retrieval accuracy by context length; Indic generation/understanding by language and script; throughput, compute cost, gradient norms and loss stability. |
| 3B parameters | Confirm the selected 1B recipe under matched controls; 60B tokens is an optional ~20 tokens/parameter planning run. Add targeted ablations for one mixture/curriculum decision, not an uncontrolled multi-variable comparison. Compute feasibility is **TBD**. | Same metrics/splits as 1B, per-capability and per-language breakdowns, cost-normalized results, gradient norms and loss curves; repeat runs where feasible to estimate variability. |

Freeze tokenizer, source snapshots, deduplication, train/validation/test splits and evaluation harness. Keep benchmark solutions and contamination-sensitive examples held out. **Results: TBD; no experiment is claimed complete.** Do not promote a recipe based on aggregate loss alone; check capability regressions and abrupt loss/gradient changes.

## 8. Cleaning, cumulative target, and data gating

### Cleaning pipeline

Continue the [Session 4 pipeline](../S4/S4_readme.md) and Session 3 safeguards:

1. Extract and normalize text while preserving useful formatting, Indic joiners, scripts, transliteration and code structure.
2. Unify conversation/tool-call formats without discarding tool outputs, failures, retries or call-result links.
3. Apply quality filters; inspect false positives by language, domain and source.
4. Perform exact and near-duplicate removal plus global cross-source deduplication.
5. Run language identification and script audits; distinguish classifier results from metadata labels.
6. Remove or mask PII; document the limits of automated detection.
7. Decontaminate against held-out benchmark/reference sets before training.
8. Produce reproducibility/provenance manifests: source revision, collection date, license and components, transformations, tokenizer token counts, exclusions, dedup and audit results.

### Confirmed Session 4 Samanantar slice

| Measure | Recorded result |
| --- | ---: |
| Input | 9,999,995 whitespace tokens |
| Output | 9,982,074 whitespace tokens; 457,922 retained pairs |
| Removed pairs | 2 boilerplate; 255 heuristic quality-filtered; 79 deduplicated |
| Decontamination | **Not run** (no holdout file) |
| Deduplication setting | Jaccard threshold 0.85 after approximate LSH candidate retrieval |

These are whitespace-token statistics over English and target text, **not model-token counts**. Session 4's 0.85 is a dedup similarity threshold, not a data-quality admission threshold. The pipeline's language label is config metadata with a script audit, not independent language identification; regex PII scrubbing is not a privacy guarantee.

### Targets and admission gate

| Requirement | Status |
| --- | --- |
| Cumulative cleaned, eligible supply target | **TBD**. Session 3's proposed 800B training tokens is not a measured cleaning target. Set a target after the full inventory is tokenizer-counted, rights-reviewed, deduplicated and contamination-checked. |
| Numeric data-gating threshold | **TBD**. Do not repurpose the 0.85 dedup threshold as a quality threshold. |
| Source admission | Require documented provenance and license/component review; quality, language/script, privacy and contamination audits; no known benchmark leakage; reproducible cleaning; and model-token counts. Threshold values and approvers are **TBD**. |
| Shortage handling | If any required slot or Indic tier fails its gate or lacks supply, stop/revise the run budget; do not relabel, silently oversample, or claim target coverage. |

## 9. Decisions and information still needed

**Proposed:** six-slot mix and Indic submix; immutable per-slot floors; separate final-phase reserve; gradual curriculum and OPUS-style selection; controlled 1B/3B proxy designs.

**Confirmed evidence:** supplied inventory estimates and available tier/sample labels; Session 3's 800B planning proposal and cleaning safeguards; Session 4's measured Samanantar whitespace-token run and explicit lack of decontamination. No V5 proxy results are available.

**Incomplete before training:** source-level licenses and rights; overlap-adjusted, cleaned model-token counts; Samanantar/BPCC commercial and component terms; IndicCorpV2 tier and per-language breakdown; translated/native Indic supply; eligible agentic supply and recovery coverage; numeric cleaning target and gate; curriculum schedule; final reserve selector/OPUS setup; and 1B/3B compute and results.

## References in this repository

- [Session 2: multilingual BPE tokenizer](../S2/S2_readme.md)
- [Session 3: India-first model research](../S3/S3_readme.md) and [research draft](../S3/S3_medium_post.md)
- [Session 4: cleaning funnel](../S4/S4_readme.md), [pipeline](../S4/pipeline.py), and [recorded statistics](../S4/output/stats.json)
