# ARCH-02 — Technical Specifications

**Project:** Asshole AI Detector  
**Status:** Proposed for team review (not an implementation or approval record)  
**Source of truth:** `docs/asshole-ai-detector-milestone-2.md` and `src/asshole_ai_detector/schemas/`  
**Tracking:** [Issue #4](https://github.com/asshole-ai-detector/asshole-ai-detector/issues/4)

## 1. Scope and decision status

This document closes architectural ambiguities before implementation. It does not replace the existing Pydantic contracts, and it does not claim that the MCP server, remote inference service, classifier, or RAG index is implemented.

- **Established:** specified in Milestone 2 or represented in existing Pydantic contracts.
- **Proposed:** implementation guidance requiring team agreement.
- **Dev-tuned:** numerical parameter to be selected using development data, then frozen.
- **Open:** a decision or interface dependency requiring explicit resolution.

## 2. Dataset roles and leakage prevention

| Dataset or partition | Role | May tune models/thresholds? | May appear in RAG index? |
|---|---|---|---|
| Scruples training partition | Train/fine-tune the five-class RoBERTa classifier (subject to verified label mapping) | Yes, training only | No by default |
| AITA development partition | Model selection, calibration, retrieval relevance annotation, thresholds, latency experiments | Yes | No |
| AITA Golden Dataset | Manually reviewed, frozen final evaluation (200 examples proposed) | **No** | **No** |
| AITA retrieval partition | Precedents and verdict distributions for the ChromaDB index | No evaluation tuning using this partition as held-out labels | Yes |

**Established:** AITA source is `MattBoraske/reddit-AITA-submissions-and-comments-multiclass`; community verdicts are reference labels, not objective truth. Golden selection targets 40 NTA, 40 YTA, 40 ESH, 40 NAH, 20 disputed, 20 missing-context cases. A consensus case requires at least five valid moral comments and at least 70% agreement. `INFO` is excluded from moral agreement. `GoldenAnnotation` stores two independent annotations for inter-annotator agreement before adjudication. Reuse the established DATA-01 dataset adapter and ARCH-01 Pydantic schemas; do not duplicate their interfaces.

**Proposed split procedure:** (1) normalize and group exact/near-duplicate stories before partitioning; (2) reserve the Golden Dataset and development set with a fixed seed and manifest; (3) exclude their IDs and near-duplicates from retrieval; (4) build the retrieval index from the remaining AITA records; (5) record dataset revision, selected IDs, grouping decisions, seed, and preprocessing version. Never use Golden labels, comments, or rephrasings as runtime evidence. A story's reference label must not be included in its inference prompt. The Golden Dataset must remain untouched until model selection and thresholds are frozen.

**Established RAG preparation:** exclude removed/deleted posts and comments, exclude null/INFO comments from moral consensus, reject posts under 50 words or with fewer than five valid moral comments, retain comment ranks and IDs (`post_id#rank`), and calculate `DerivedFields` from the valid comments. Keep up to 5,000 cases per `top_verdict` (maximum 20,000). `provider_ambiguity` is diagnostic only, not a filtering or scoring target. Initial near-duplicate cosine threshold is 0.95, subject to development validation.

**Open integration dependency — DATA-02/DATA-03 (PRs #17 and #14):** DATA-02 produces `CleanedAITACase`, a subtype of `NormalizedAITACase` with cleaned, filtered `NormalizedAITAComment` objects in `case.comments` and diagnostic `community_votes`. DATA-03 currently expects `label_cleaned_case(case, valid_comments)`, where `valid_comments` is a separate iterable of fully constructed `AITAComment` objects. Thus the two implementations do not yet have a direct handoff.

**Proposed handoff:** `NormalizedAITACase` → `CleanedAITACase` → `AITACase`. DATA-02 owns text normalization, comment/post filtering, and preservation of original ranks. DATA-03 should consume the cleaned case, construct final `AITAComment` objects with `comment_id = f"{post_id}#{rank}"`, and compute `DerivedFields` from the valid moral comments. Add a cross-PR integration test for the full handoff, including nonconsecutive ranks and excluded INFO comments. Resolve optional normalized comment fields with explicit validation or an agreed stricter cleaned-comment type; do not bypass the established schemas. Confirm whether diagnostic `community_votes.INFO` is retained in a preprocessing report or separately persisted, since it is not part of the current `AITACase` contract. This handoff is **proposed, not yet merged**.

## 3. MCP tools and final output contracts

**Established:** Python Pydantic contracts already exist in `src/asshole_ai_detector/schemas/tools.py`, `result.py`, `annotation.py`, and `case.py`. MCP tool implementations must validate their input and output using these classes. `ContractModel` rejects unexpected fields. The tool names and contracts are:

| MCP tool | Input contract | Output contract | Responsibility |
|---|---|---|---|
| `neutralize_narrative` | `NeutralizeNarrativeInput` | `NeutralizeNarrativeOutput` | Preserve facts while reducing loaded language; expose relationship, trigger and actions for deterministic retry |
| `classify_verdict` | `ClassifyVerdictInput` | `ClassifyVerdictOutput` | Calibrated probabilities over NTA/YTA/ESH/NAH/INFO, winning class, top-two margin and unnormalized entropy |
| `retrieve_similar_cases` | `RetrieveSimilarCasesInput` | `RetrieveSimilarCasesOutput` | Ranked precedents, evidence, top-neighbor distribution, disagreement and retrieval confidence |
| `generate_counter_perspective` | `GenerateCounterPerspectiveInput` | `GenerateCounterPerspectiveOutput` | Grounded alternative perspectives and preliminary explanation in one generative call |
| `check_sycophancy_and_bias` | `CheckSycophancyAndBiasInput` | `CheckSycophancyAndBiasOutput` | Role-swap consistency, unsupported claims, invalid references and narrator agreement checks |
| `assess_verdict_uncertainty` | `AssessVerdictUncertaintyInput` | `AssessVerdictUncertaintyOutput` | Deterministic four-tier decision and clarification metadata |

**Established final response:** `AssessmentResult` (`result.py`) includes `verdict`, optional `secondary_verdict`, `explanation`, `alternative_perspectives`, `uncertainty_tier`, `missing_information`, `limitations`, and at most three `clarification_questions`. `withhold` requires null primary and secondary verdicts; `split` requires two distinct moral verdicts; every other tier requires a primary verdict and no secondary verdict. `INFO` is never a user-facing moral verdict.

**Established validation details:** `ClassifyVerdictOutput.top_two_margin` is computed over **all five classes**; entropy uses natural logarithms and is not normalized. `RetrievedCaseEvidence` exposes at most three excerpts, one per verdict, each at most 60 words. `RetrieveSimilarCasesOutput` separates up-to-15-neighbor statistics from up-to-five displayed cases. Retrieval disagreement is four-class **normalized** Shannon entropy. An empty retrieval result has zero confidence and no distribution/disagreement statistics. `check_sycophancy_and_bias.passed` means no modeled bias flags were detected, not proof of neutrality.

**Proposed runtime policy:** validate inputs before invoking a tool; validate output before returning it to the controller; record structured error category and request ID in internal logs; never silently invent fallback evidence. For malformed LLM JSON, permit a bounded retry within the latency budget, then withhold or return a documented degraded result. The controller, not the LLM, selects the candidate verdict. Grounding checks must verify that referenced comment IDs exist in the retrieved evidence and that quoted scenario facts are supported.

**Open:** specify an explicit cross-tool failure/error envelope (outside the current successful-output schemas), timeout and retry limits, and whether revisions after bias checking need a separately typed internal output. Do not add fields to existing models without an agreed contract change and tests.

## 4. Controlled agent workflow

1. Apply a deterministic initial completeness checklist (narrator actions, other party actions/reactions, relationship, trigger, relevant intent and outcome). If two or more relevant details or a critical fact are missing, request up to three clarification questions once.
2. Neutralize the narrative without inventing motivations or events (generative call 1).
3. Run classifier and retrieval on the neutralized narrative. Inspect `INFO` probability against dev-tuned `t_info`; if necessary, use the single permitted clarification round.
4. If a moral verdict candidate remains defensible, generate alternative perspectives and a preliminary explanation (generative call 2).
5. Perform role-swap and deterministic grounding/disagreement checks (role-swap generation is generative call 3). If issues are found, revise explanation once (optional generative call 4), and preserve unresolved flags.
6. Apply the deterministic uncertainty hierarchy **withhold > split > leaning > confident** and validate `AssessmentResult` before returning it.

**Established:** initial completeness, classifier, retrieval and uncertainty assessment do not themselves require generative LLM calls. Deterministic query reformulation may retry retrieval once. A roughly 30-second end-to-end latency is a target to measure, not an achieved guarantee.

**Open semantic edge case:** when INFO is among the top two classifier probabilities, `top_two_margin` is not necessarily the margin between the two leading moral verdicts. The uncertainty rule must explicitly define which margin drives `split` and how to select two distinct moral verdicts, while respecting the current `ClassifyVerdictOutput` field definition. A high INFO probability should not be treated as a moral verdict.

## 5. Retrieval and indexing specifications

**Established:** one vector per AITA post, embedded text `title + "\n" + text`, with a 1,024-token safety cap (first 700 + last 324). Comments are not embedded. Milestone 2 describes complete cases as stored payload; the proposed implementation refinement below separates full case storage from vector metadata. Use Qwen3-Embedding-0.6B, 1,024-dimensional L2-normalized vectors, cosine similarity, a persistent ChromaDB collection, and Qwen3-Reranker-0.6B. Query instruction prefix and no document prefix follow Milestone 2 Section 3.4.

**Proposed case-store separation (pending team agreement):** ChromaDB holds post embeddings and retrieval metadata (`post_id`, `top_verdict`, `consensus_label`), while complete cleaned and labeled AITA cases (post text, URL, date, valid comments, comment IDs, derived fields and diagnostic metadata where retained) live in a persistent case store keyed by `post_id`. The retrieval service resolves candidate IDs against the case store for reranking and evidence construction. Both stores must be versioned consistently; missing case records must produce an explicit integrity error rather than fabricated evidence. The exact case-store format and location remain open. This proposal refines Milestone 2's payload-storage wording and requires team confirmation.

The retrieval pipeline performs dense top-30, reranks top-15, computes a relevance-weighted four-verdict distribution and disputed share from the reranked neighbors, then selects up to five displayed cases using verdict-balanced selection. `retrieval_confidence` is the mean score of the five highest-ranked neighbors (or the available fewer neighbors), **not** the displayed balanced cases. The initial reranker threshold is 0.5; if the best score is below threshold, retry once using a deterministic relationship/trigger/actions query. Treat low or missing retrieval evidence as weak or unavailable, never as agreement.

**Proposed index integrity:** store an index manifest with dataset revision, embedding/reranker model revisions, preprocessing version, seed, selected IDs/hash, dimension, metric, thresholds, and build timestamp. Rebuild when embedding model or preprocessing changes. Run ingestion offline and idempotently; do not expose index mutation as an MCP tool.

## 6. Model comparison and evaluation

| Component | Candidate(s) in Milestone 2 | Evaluation role |
|---|---|---|
| Agent SLM default | Qwen3-4B-Instruct-2507 | Primary controlled agent |
| Agent SLM comparisons | Gemma 3 4B; Llama 3.2 3B | Compare sycophancy, accuracy, consistency, latency |
| Small SLM baseline | Qwen2.5-1.5B-Instruct | Lower-cost baseline |
| Classifier | RoBERTa-base fine-tuned on Scruples | Five-class predictions; calibration and INFO behavior |
| Embedding | Qwen3-Embedding-0.6B | Retrieval; near-duplicate detection |
| Reranker | Qwen3-Reranker-0.6B | Evidence relevance |
| External API baseline | Gemini or GPT family, exact available model **open** | Zero-shot reference |

**Established comparison configurations:** direct LLM baseline (no tools), RAG-enhanced LLM, and full controlled agent. Use the same held-out stories and comparable prompting/output rules. Select models and tune thresholds on the development set only; freeze the final configuration before a single Golden evaluation. The default SLM remains Qwen3-4B unless a challenger improves YTA sycophancy without reducing verdict accuracy below the baseline and stays near the target median latency of 30 seconds.

**Established nine-configuration experiment matrix (Milestone 2, Section 4.5):**

| Model | Direct baseline | RAG-enhanced | Agent-based | Number of runs |
|---|---|---|---|---:|
| Qwen3-4B-Instruct-2507 | Yes | Yes | Yes | 3 |
| Gemma 3 4B | Yes | No | Yes | 2 |
| Llama 3.2 3B | Yes | No | Yes | 2 |
| Qwen2.5-1.5B-Instruct | Yes | No | No | 1 |
| API baseline (exact model to be fixed) | Yes | No | No | 1 |
| **Total** | **5** | **1** | **3** | **9** |

**Additional reference:** evaluate fine-tuned RoBERTa in a **classifier-only** configuration for classification accuracy, macro-F1 and calibration; this is not counted among the nine generative configurations. If time or compute is limited, prioritize the three Qwen3-4B configurations and classifier-only evaluation; optional challenger agent runs and ablations may be reduced, with deviations recorded. Additional experiments are extensions, not part of the original nine.

Use fixed prompt versions and deterministic decoding where supported. Record hardware, quantization, seeds, inference settings, token usage, model revisions and per-story latency. Compare configurations on the same held-out examples with metrics appropriate to their capabilities; do not apply generative explanation metrics to classifier-only runs.

**Metrics and denominators:**
- **Verdict accuracy / macro-F1:** only Golden `consensus` cases with non-null reference verdicts; report class-level results and confusion matrix.
- **YTA sycophancy / unjustified narrator agreement:** predefine a reproducible rubric for YTA-reference cases and report flagged count divided by eligible cases; include independent human review rather than equating any disagreement with sycophancy.
- **Uncertainty handling:** tier accuracy against `expected_uncertainty_tier`; withhold/clarification precision and recall for missing-context cases; report disputed cases separately.
- **Retrieval relevance:** strict and lenient Precision@5 from human relevance labels; annotator agreement and Cohen's kappa.
- **Grounding and perspective coverage:** human-coded supported-claim fraction, invalid references, and coverage of plausible non-narrator perspectives using an agreed rubric.
- **Consistency:** original/neutralized/role-swapped versions of 20 Golden consensus stories (60 inputs), with responsibility mapped back to the original narrator.
- **Latency:** median and p95 end-to-end time per story; per-tool timings, failures and retries.

**Open:** agree on the exact sycophancy rubric, threshold for acceptable accuracy difference, repeat count, and final API model ID (Milestone 2 lists Gemini 3.8 Flash or GPT-4o-mini as candidates; verify availability) before running the Golden evaluation. A model's sycophancy score must not be computed from Golden labels during selection.

## 7. Configuration and tuning

| Key | Initial value or status | Selection / constraint |
|---|---|---|
| `retrieval.dense_top_k` | 30 (established) | >= rerank top-k |
| `retrieval.rerank_top_k` | 15 (established) | >= evidence top-k |
| `retrieval.evidence_top_k` | 5 (established) | <= rerank top-k |
| `retrieval.reranker_threshold` | 0.5 initially; dev-tuned | Validate retrieval relevance and retry rate |
| `retrieval.disagreement_threshold` | dev-tuned | Normalize entropy over four moral verdicts |
| `uncertainty.m_low` | dev-tuned | Must be <= m_high |
| `uncertainty.m_high` | dev-tuned | Must be >= m_low |
| `uncertainty.t_info` | dev-tuned | INFO-based clarification signal |
| `ingestion.near_duplicate_cosine` | 0.95 initially; validate on dev pairs | Check false negatives/positives |
| `ingestion.min_words` | 50 (established) | Post filter |
| `ingestion.min_valid_comments` | 5 (established) | Consensus eligibility |
| `ingestion.consensus_threshold` | 0.70 (established) | Moral-vote agreement |
| `ingestion.max_per_top_verdict` | 5,000 (established) | Up to 20,000 posts |
| `agent.max_clarification_questions` | 3 (established) | Single clarification round |
| `agent.max_generative_calls` | 4 (established) | Excludes optional clarification round |
| `runtime.target_median_seconds` | ~30 target | Measure, do not assume |

**Proposed:** use `configs/default.yaml` for reproducible non-secret defaults, a separately versioned frozen evaluation config, and environment variables for credentials and deployment endpoints. Validate cross-field constraints at startup. Record effective configuration and its hash in every experiment. Avoid adding undocumented defaults to code.

## 8. Streamlit, local MCP and Colab communication

**Established architecture:** Streamlit user interface; a single workflow controller acting as an MCP client; a Python MCP server using the official SDK and local `stdio` transport; services for classifier, retrieval and generation. Milestone 2 plans heavy models on Google Colab, with Kaggle as backup.

**Important boundary:** local `stdio` connects processes on the same machine; it does not connect a local MCP server directly to a remote Colab process.

**Proposed integration:**

```text
Streamlit (local)
    -> Workflow controller / MCP client (local)
    -> Python MCP server (local, stdio)
        -> Local RAG service + versioned ChromaDB index
        -> Model-service adapter (authenticated HTTPS)
            -> Remote inference service on Colab (when reachable)
                -> RoBERTa / generator / other deployed models
```

The remote inference adapter is a **proposal**, not an existing deployed endpoint. Colab notebook sessions are ephemeral and may not expose stable inbound URLs; the team must verify tunnel/network access, authentication, session lifetime, GPU memory, and acceptable latency. Alternative: execute the MCP server and Streamlit in the same Colab environment for a demonstration, or run small quantized models locally. These alternatives should be tested before the final deployment choice.

**Early feasibility validation (open):** Before committing to the proposed remote deployment, run a minimal end-to-end proof of concept: Streamlit/controller → local MCP server → authenticated HTTPS inference adapter → Colab-hosted model → schema-validated response. Measure connectivity, cold-start behavior, session lifetime, authentication, failure/timeout handling and end-to-end latency. Record results and choose a fallback if the remote approach is unreliable. This proof of concept is a follow-up implementation task, not a completed ARCH-02 deliverable.

**Proposed service requirements:** explicit request IDs, model/version fields, authenticated transport, request/response timeouts, bounded retries, no raw scenario logging by default, redacted error messages, and health checks. Return structured errors to the controller; on remote model failure, do not fabricate a verdict. Keep the MCP success payload identical regardless of model location.

## 9. Storage and artifacts

**Proposed layout (paths are conventions, not evidence of existing directories):**

```text
configs/
  default.yaml
  evaluation-frozen.yaml             # created after dev tuning
src/asshole_ai_detector/
  schemas/                           # existing Pydantic contracts
  data/                              # existing source adapters
  data_processing/                   # consensus (PR #14 under review)
  agent/                             # proposed controller
  mcp_server/                        # proposed MCP adapters
  retrieval/                         # proposed index and search services
artifacts/                            # generated, normally gitignored
  chroma/<index-version>/            # embeddings + retrieval metadata
  cases/<dataset-version>/           # complete cases, keyed by post_id (proposed)
  models/<model-id>/
  runs/<run-id>/
  annotations/
docs/
  architecture/technical-specifications.md
```

**Proposed case storage:** Keep complete cleaned and labeled AITA cases separately from the ChromaDB vector index, using `post_id` as the lookup key. ChromaDB retains embeddings and retrieval metadata; the case store retains full post text, valid comments, comment IDs and derived consensus statistics. Coordinate case-store and index versions. The exact format (for example JSONL or a keyed database), shared location, and retention policy are not yet agreed.

**Proposed:** keep lightweight manifests, schemas, configs and reproducible scripts in Git; keep large model weights, datasets, vector indexes and private user stories out of Git. Use an artifact store or shared drive with checksums, access control and explicit retention policy. Each run records dataset revision/split IDs, model versions, code commit, config hash, evaluation script version, timestamps, metrics, and failure counts. Do not commit API tokens or user-submitted conflict narratives. If a large index is copied between local and Colab, verify its embedding model revision, dimensions and manifest before use.

## 10. Decisions, alternatives and trade-offs

| Decision | Status | Alternative / trade-off |
|---|---|---|
| Single controlled agent with mandatory validation | Established | More reproducible than unconstrained multi-agent debate; less flexible |
| RoBERTa primary verdict + deterministic uncertainty | Established | Avoids generative override; classifier domain shift must be measured |
| One vector per post | Established | Preserves narrative context; can lose detail in very long posts |
| Qwen3 embedding and reranker | Established in latest Milestone 2 | More capable pipeline but greater compute than MiniLM/BGE alternatives |
| ChromaDB persistent local index | Established | Simple deployment at current scale; requires artifact syncing across environments |
| Separate full-case store keyed by `post_id` | Proposed | Keeps vector metadata small; adds a lookup and requires synchronized versions |
| DATA-02 → DATA-03 cleaned-case handoff | Open | Removes duplicate comment lists; requires agreement and cross-PR integration tests |
| Nine-configuration experiment matrix | Established in Milestone 2 | Bounded comparison; additional runs and ablations are optional |
| Local stdio MCP | Established as prototype | Straightforward local orchestration, not a remote transport |
| Remote Colab inference adapter | Proposed | Preserves local MCP but requires reliable authenticated remote connectivity |
| Development tuning, Golden only for final evaluation | Established | Reduces leakage, limits ability to revisit choices after final results |

## 11. Open decisions and integration checklist

- [ ] Confirm DATA-02/DATA-03 handoff and cross-PR integration tests (PRs #17 and #14); reflect the agreed interface after merge.
- [x] Align the experiment matrix with the nine configurations in Milestone 2, Section 4.5.
- [ ] Confirm separate case-store format, `post_id` lookup, and index/case version consistency.
- [ ] Run the local MCP → authenticated HTTPS → Colab feasibility proof of concept and record fallback.
- [ ] Verify Scruples-to-five-class label mapping and calibration plan before classifier training.
- [ ] Define moral-only margin semantics and INFO interaction in uncertainty logic without silently changing `top_two_margin`.
- [ ] Decide remote Colab connectivity vs co-located demo vs local inference based on feasibility results; document fallback and security.
- [ ] Define MCP error envelope, timeouts, retry policy and validation tests.
- [ ] Agree on evaluation rubrics and exact baseline model identifiers.
- [ ] Tune and freeze numerical thresholds using development data only.
- [ ] Confirm artifact storage location, access policy and data retention.
- [ ] Obtain team review/approval; record decision owners and date.

## 12. Acceptance checks for ARCH-02

- [x] Draft dataset roles, split/leakage rules and reproducibility plan.
- [x] Map all six MCP tools to existing typed contracts and define the final response contract.
- [x] Reproduce Milestone 2's exact nine-configuration matrix, classifier-only reference, and metric definitions.
- [x] Inventory configuration parameters and distinguish established defaults from dev-tuned values.
- [x] Describe local MCP/Streamlit/Colab communication and explicitly flag unresolved deployment choices.
- [x] Propose separate case-store and vector-index layout, manifests and artifact policies (pending agreement).
- [x] Record decisions, alternatives, trade-offs and open questions.
- [ ] Validate all assumptions against the latest merged code and complete team review.
- [ ] Resolve or explicitly accept open decisions before marking Issue #4 complete.
