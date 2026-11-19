# Am I The Asshole? — Request for Comments (RFC)

## 1. Problem Definition

* **Context:** Interpersonal conflicts in everyday life (family, friends, partners,
  roommates, coworkers), where people want an outside opinion on whether they
  acted wrongly. Today they get one in two main ways: posting the story publicly on
  communities like Reddit's r/AmItheAsshole and waiting for the crowd to vote,
  or asking a general-purpose AI chatbot, which has become a popular shortcut
  (a growing number of "AI judge" apps are built on this idea) [8][9][10].

* **The Problem:** Both options are flawed. Public forums are slow, expose personal
  details, and give inconsistent answers depending on who replies first. Chatbots are
  instant and private, but they are sycophantic: on AITA posts where the crowd
  judged the poster to be in the wrong, leading LLMs still affirmed the user's
  behavior in about 42% of cases on average [1], and a later
  Science study found AI systems affirmed users in 51% of such cases, versus 0%
  for humans [3]. This is hard to fix with traditional software because the input is
  a subjective, one-sided narrative with no rules to check against, and verdicts
  are often genuinely divisive: even the crowd disagrees on many cases [4],
  so a single confident YES/NO is itself misleading.

* **Impact & Audience:** Anyone seeking advice on a personal conflict, especially
  people who treat the AI's answer as a neutral referee. Sycophantic advice has
  measurable costs: in controlled experiments it made users more convinced they
  were right and less willing to repair the relationship, while increasing their
  trust in and reliance on the AI [2]. Users also tend to prefer
  the flattering answers [2][11], so existing products have little incentive to fix this.
  The result is worse decisions in real relationships, delivered with false
  certainty.

## 2. State of the Art

* **Existing Solutions:** (1) General LLM chatbots (ChatGPT/Claude) prompted ad hoc;
  (2) Consumer "AI judge" apps (Moral Compass, Am I Wrong?, The Docket) that wrap an LLM
  with a persona and a verdict card [8][9][10]; (3) Research classifiers fine-tuned on r/AITA
  (RoBERTa/Longformer [5]; Scruples [4]).
* **Pros:**
  * Fluent, instant, zero-friction input (just paste your story).
  * Fine-tuned classifiers reach ~87% accuracy at predicting the crowd verdict [5].
  * Persona/shareable-verdict formats drive engagement [9][10].
* **Cons:**
  * Sycophancy: LLMs affirm behavior the crowd judged inappropriate in ~42% of AITA
    cases [1], and affirm users 49% more than humans [3].
    Users prefer sycophantic answers and become less willing to repair relationships [2][11].
  * Single-narrator bias: the only input is the asker's own framing.
  * False certainty: crowd judgments are often divisive [4], yet apps output one verdict.
  * Black-box outputs with no explanation or comparable cases (a limitation acknowledged
    by the Delphi authors) [6][7].
  * Privacy/monetization: sensitive stories sent to third-party free-tier APIs;
    pay-to-win verdict features undermine trust [10].

## 3. Proposal

* **Our Solution:** An AITA-judging app that reframes the user's story neutrally,
  generates the other party's strongest case, and returns a probability distribution
  over verdicts (NTA/YTA/ESH/NAH) grounded in retrieved, similar resolved AITA cases.
* **What is Different:** Anti-sycophancy is a measured design goal, not an afterthought:
  neutral third-person rewrite, forced counter-perspective, and a calibration check
  against a fine-tuned classifier, evaluated on YTA-labeled AITA posts [1][2].
* **What Makes it Better:** Lower false-negative rate than prompting a general LLM
  (target to be measured with the ELEPHANT benchmark [1]), calibrated uncertainty instead of
  fake certainty, explanations via similar cases, and a no-retention privacy mode.

## 4. Architecture Decisions and Open Questions (ARCH-02)

This section records the technical choices and review dependencies tracked in [Issue #4](https://github.com/asshole-ai-detector/asshole-ai-detector/issues/4) and detailed in [`docs/architecture/technical-specifications.md`](architecture/technical-specifications.md). It distinguishes the Milestone 2 baseline and existing typed contracts from implementation proposals; it does **not** claim that pending pull requests or deployment experiments have been approved or merged.

### 4.1 Established design baseline

| Area | Established direction | Rationale / trade-off |
|---|---|---|
| Workflow | Single controlled, tool-assisted agent with mandatory validation and the tier priority `withhold > split > leaning > confident` | Reproducible decisions instead of unrestricted multi-agent deliberation |
| Verdict prediction | RoBERTa-based five-class classifier (NTA, YTA, ESH, NAH, INFO); INFO is clarification-only, never a final moral verdict | Keeps moral prediction separate from generative explanations; requires verified Scruples label mapping and calibration |
| Typed interfaces | Reuse existing ARCH-01 Pydantic schemas, six MCP tool contracts and `AssessmentResult` | Avoids competing output definitions; tool implementation and error envelopes remain separate work |
| Retrieval | One vector per post, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, persistent ChromaDB, top-30 search / top-15 rerank / up to five displayed cases | Preserves narrative context; requires offline indexing and evidence integrity checks |
| Dataset integrity | Separate classifier training, AITA retrieval, development and frozen Golden evaluation roles; exclude Golden/dev cases and near-duplicates from retrieval | Prevents leakage and tuning on final evaluation labels |
| Experiments | Milestone 2's nine model/configuration runs, plus classifier-only RoBERTa evaluation | Fixed comparison scope; optional ablations and extra runs are separate |
| Runtime prototype | Python MCP server over local `stdio`, Streamlit UI and a controlled workflow client | Simple local integration; `stdio` cannot directly connect to remote Colab |

**Exact experiment matrix (Milestone 2, Section 4.5):** Qwen3-4B-Instruct-2507 (baseline, RAG, agent); Gemma 3 4B (baseline, agent); Llama 3.2 3B (baseline, agent); Qwen2.5-1.5B-Instruct (baseline); external API model (baseline). This is **nine** configurations. RoBERTa classifier-only is evaluated separately. The API model identifier must be confirmed before final evaluation.

### 4.2 Proposed implementation refinements — pending team agreement

1. **DATA-02/DATA-03 handoff (PRs #17/#14):** DATA-02 returns `CleanedAITACase` with filtered `NormalizedAITAComment` objects in `case.comments`. DATA-03 currently expects a separate `Iterable[AITAComment]`. Proposed integration: `NormalizedAITACase → CleanedAITACase → AITACase`, with DATA-03 creating `post_id#rank` comment IDs and consensus fields. Validate with cross-PR tests before documenting the interface as final. ARCH-01 contracts remain authoritative.
2. **Separate case storage:** Keep embeddings and retrieval metadata in ChromaDB, while storing complete posts, valid comments and derived fields in a separate persistent store keyed by `post_id`. This refines Milestone 2's full-payload description; the exact format, location, retention policy and consistency checks remain to be agreed.
3. **Colab inference adapter:** Keep the local MCP `stdio` boundary and call Colab-hosted inference through a proposed authenticated HTTPS adapter. Validate connectivity, session reliability, authentication, timeouts and latency in an early proof of concept. Co-located or local inference remains a fallback; no remote endpoint is claimed to exist yet.

### 4.3 Open decisions and follow-up validation

- Confirm DATA-02/DATA-03 integration, optional cleaned-comment field handling, and whether `community_votes.INFO` is retained outside the final `AITACase` schema.
- Verify the Scruples-to-five-class label mapping and calibration on development data.
- Specify moral-only verdict margin semantics when INFO affects the classifier's five-class `top_two_margin`.
- Define MCP error envelopes, retry/timeout limits, and grounding-validation behavior without changing established success schemas unilaterally.
- Confirm the case-store format and access/retention policy, and test the remote inference proof of concept before selecting deployment.
- Agree on sycophancy and human-evaluation rubrics, exact API baseline model, and repeat counts; tune numerical thresholds on the development set only.
- Freeze all configurations before the Golden Dataset is used; record review outcomes and any accepted deviations in the technical specifications and relevant PRs.

**Review status:** Proposed for team review. Items in Section 4.2 are not yet approved implementation decisions, and open questions do not block documenting the established architecture if the team explicitly accepts them as tracked follow-up work.

---

## References

### Cited above

* [1] M. Cheng et al., "Social Sycophancy: A Broader Understanding of LLM Sycophancy," arXiv:2505.13995, 2025. https://arxiv.org/abs/2505.13995v1 (ELEPHANT code & data: https://github.com/myracheng/elephant)
* [2] M. Cheng et al., "Sycophantic AI Decreases Prosocial Intentions and Promotes Dependence," arXiv:2510.01395, 2025 (published in Science, 2026). https://arxiv.org/pdf/2510.01395
* [3] Neuroscience News, summary of the Science study (source of the 49% / 51% vs 0% figures). https://neurosciencenews.com/ai-sycophancy-moral-judgment-30397/
* [4] N. Lourie et al., "Scruples: A Corpus of Community Ethical Judgments on 32,000 Real-Life Anecdotes," AAAI 2021, arXiv:2008.09094. https://arxiv.org/pdf/2008.09094 (DOI: 10.1609/aaai.v35i15.17589; data/code: https://github.com/allenai/scruples)
* [5] A. Alhassan, J. Zhang, V. Schlegel, "'Am I the Bad One'? Predicting the Moral Judgement of the Crowd Using Pre-trained Language Models," LREC 2022. https://aclanthology.org/2022.lrec-1.28/
* [6] L. Jiang et al., Delphi / Commonsense Norm Bank, arXiv:2110.07574, 2021. https://arxiv.org/pdf/2110.07574v1.pdf
* [7] IEEE Spectrum, coverage of Delphi (including the authors' note that its reasoning is hard to explain). https://spectrum.ieee.org/ai-ethics-machines-learn-good
* [8] "Moral Compass - AITA Community," Apple App Store listing. https://apps.apple.com/us/app/-/id6748293841
* [9] "Am I Wrong? AI Judge," AppGoblin listing. https://appgoblin.info/apps/6768784454
* [10] "The Docket," Product Hunt. https://www.producthunt.com/p/the-docket
* [11] Let's Data Science, coverage of the Stanford/Science sycophancy study (reporting AP and NPR). https://letsdatascience.com/news/study-finds-ai-chatbots-favor-flattery-over-facts-a2ef7288
