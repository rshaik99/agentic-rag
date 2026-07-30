# Design: multi-department enterprise RAG (bank context)

Status: draft, not implemented. Written to align on architecture before
building anything. Scoped against the four gaps identified for a bank
deployment: department-level access control, huge-file ingestion, images
at scale, and a Confluence connector -- plus audit logging, which any bank
context needs even though it wasn't explicitly asked for.

## 1. Why the current design doesn't carry over as-is

`rag-multimodal` and `agentic-rag` are both single-tenant: one Qdrant
collection, no concept of "who is asking," batch ingestion sized for a
2-page demo PDF, and no audit trail. Every one of those assumptions breaks
in a multi-department bank context. This doc proposes what changes and
what doesn't.

## 2. Access control (the part that reshapes retrieval itself)

**The core principle: access control must be enforced at retrieval time,
not as a UI filter.** An LLM handed the wrong context in its prompt will
use it -- filtering results *after* generation is too late, and hiding a
button in the UI doesn't stop an API caller. Every retrieval call must be
scoped to what the requesting user is allowed to see, before anything
reaches the LLM.

**Proposed shape:**
- Every indexed chunk gets an `access_groups: list[str]` metadata field at
  ingest time (e.g. `["retail-banking", "compliance"]`).
- Every retrieval call carries the requesting user's `access_groups` and
  applies it as a **mandatory Qdrant filter** (`must: [{key: access_groups,
  match: {any: [...]}}]`) -- not optional, not applied downstream.
- `rag_search` (agentic-rag) and `step_c_retrieve_rerank` (rag-multimodal)
  both need a `user_context` parameter threaded through from the entry
  point (CLI/API caller) down to the actual Qdrant query. Currently neither
  accepts one -- this is the biggest code-shape change, not just a config
  addition.
- Open question: **one Qdrant collection with a filter field, or one
  collection per department?** Filter-based is simpler operationally (one
  ingest pipeline, one index to maintain) but a compromised filter (a bug,
  not just an attack) leaks across departments in the same collection.
  Separate collections are stronger isolation but multiply ingestion/
  reindexing work per department. For a bank, I'd lean separate
  collections for anything regulatory-sensitive (compliance, legal) and
  filter-based for lower-sensitivity general departments -- a hybrid, not
  an all-or-nothing choice.
- Where does `access_groups` come from per-user? Needs to hook into
  whatever identity system the bank already has (likely SSO/LDAP/Active
  Directory group membership) -- this is an integration point, not
  something to build from scratch.

## 3. Ingestion at scale (huge files + incremental updates)

**Current state:** `step_a_ingest.py` is a batch script -- point it at
`data/`, it processes everything, `step_b_index.py --recreate` drops and
rebuilds the whole collection. Fine for a demo corpus, wrong model for a
bank's actual document volume.

**Proposed changes:**
- **Incremental ingestion**, not recreate-on-change. Each chunk already
  gets a deterministic ID (`chunk_uuid` hashing a readable key) -- that's
  the right foundation; upsert-by-ID already works, it's the *pipeline*
  that needs to stop assuming "ingest everything from scratch."
- **A real ingestion queue**, not a one-shot script: new/changed documents
  land in a queue (could be as simple as a "watch a folder / poll an API"
  worker, or a proper message queue if volume demands it), get chunked +
  embedded + upserted asynchronously. Large PDFs (hundreds of pages) need
  to chunk and embed in batches without holding the whole document in
  memory -- `pymupdf4llm` streaming page-by-page rather than
  whole-document parsing.
- **OCR fallback.** `pymupdf4llm` handles text-layer PDFs well but banks
  have plenty of scanned/faxed documents with no text layer at all. Needs
  a detection step (does this page have an extractable text layer? if not,
  route to OCR) rather than assuming every PDF is text-native.
- **Chunking tuned for real documents.** `CHUNK_SIZE=1024` tokens was
  picked for a 2-page demo; needs revisiting against actual bank document
  structure (long policy documents, dense financial statements) --
  probably needs section-aware chunking (respect headings/page boundaries)
  more strictly than the current approach.

## 4. Images at scale

**Current state:** `phase2_multimodal/ingest_images.py` is a manual script
you run by hand after indexing; every extracted image gets a VLM
description regardless of whether it's a meaningful chart or a company
logo.

**Proposed changes:**
- **Fold into the ingestion pipeline** above rather than a separate manual
  step -- image extraction happens automatically as part of processing
  each document, not as an afterthought.
- **Pre-filter before the (expensive) VLM call.** Cheap heuristics first:
  image size/aspect ratio (logos are small and square-ish; charts are
  usually larger), position on page, maybe a lightweight classifier --
  only VLM-describe images likely to actually be charts/tables/diagrams.
  At bank scale, VLM-describing every embedded image (including letterhead
  logos on every page of every document) is real, avoidable cost.
- **Access control applies to images too** -- an extracted chart node
  needs the same `access_groups` metadata as its source document, inherited
  at extraction time, not set separately.

## 5. Confluence connector

Structurally different from the PDF path -- not a file, an API-backed,
frequently-edited page.

- **Ingestion**: Confluence REST API (`/rest/api/content`) → page body
  (HTML) → convert to the same chunk schema as PDF ingestion, so
  downstream retrieval/rerank code doesn't need to know the source type.
- **Change detection**: Confluence pages have a `version.when` timestamp --
  poll for pages modified since last sync (or use Confluence webhooks if
  available) and feed changed pages into the same incremental-ingestion
  queue from Section 3, rather than a separate one-off "connector script."
- **Access control**: Confluence has its own page-level permissions --
  ideally mirror those into `access_groups` at sync time rather than
  granting broader access than the source system intended. This needs a
  mapping from Confluence's permission model to whatever `access_groups`
  scheme is chosen in Section 2.
- **Space-level scoping**: a bank's Confluence likely has per-department
  spaces already (Compliance space, Retail Banking space, etc.) -- that's
  a natural, probably-already-correct mapping to department access
  groups, worth using rather than re-deriving permissions from scratch.

## 6. Audit logging (not asked for, but a bank will require it)

Every query needs a durable record: who asked, what was retrieved (which
chunks/documents, from which access groups), what the LLM answered, when.
This is a compliance requirement, not an engineering nice-to-have, in a
regulated industry. Shape:
- Log at the retrieval layer (what was actually returned, post-access-
  control-filter) and at the answer layer (what the LLM said).
- Needs to be tamper-evident/durable (a proper log store, not stdout) --
  out of scope to design in detail here, but flagging it now so it's not
  bolted on after the fact once access control (Section 2) exists.

## 7. Infrastructure changes implied by all of the above

- **Production Qdrant**, not a local Docker container: auth (API keys),
  TLS, likely a managed/clustered deployment for availability.
- **Reranker as a served model**, not loaded in-process per request. The
  current pattern (`lru_cache`-wrapped `HuggingFaceCrossEncoder` loaded
  once per Python process) works for a CLI/single-user tool; at
  multi-department concurrent load it needs to be a shared inference
  service so N concurrent requests don't each load a ~500M-param model
  into their own process.
- **Rate limiting / cost governance** per department, so one department's
  usage spike doesn't exhaust shared LLM/embedding budget.

## 8. Suggested build order

Given how much Section 2 (access control) reshapes the retrieval function
signatures, doing it **first** avoids rework -- everything else
(ingestion, Confluence, images) should be built emitting the
`access_groups` metadata from day one rather than bolting it on after.

1. Access control: `access_groups` metadata + mandatory filter at
   retrieval time, threaded through `rag_search`/`step_c` -- do this even
   before scaling ingestion, so nothing gets built on the wrong retrieval
   contract.
2. Incremental ingestion pipeline (replaces the batch recreate-everything
   model) -- foundation for both huge files and Confluence.
3. Confluence connector, built on top of (2).
4. Image handling folded into (2), with the pre-filter cost control.
5. Audit logging, threaded through from the start of whichever of the
   above ships first -- not deferred to the end.

## Open questions for you

- Filter-based single collection vs. per-department collections (Section
  2) -- what's the actual sensitivity spread across departments you have
  in mind?
- What identity system would `access_groups` need to integrate with
  (SSO/LDAP/AD/something else)?
- Confluence: real instance to design against, or should this stay
  generic/hypothetical for now?
- Is this meant to stay a portfolio/demo project, or is there a real
  deployment target that should shape how much of this actually gets
  built vs. documented?
