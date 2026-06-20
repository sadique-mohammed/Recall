# Project Plan — RAG Evaluation Pipeline

Retrieval-augmented question answering over ESG / compliance documents, built around a
**from-scratch retrieval evaluation harness** (recall@k, hit-rate@k, MRR) rather than a
black-box "load PDF, ask question" demo. The evaluation layer is the centrepiece: the goal
is to show that retrieval quality is *measured*, not assumed.

---

## 1. What this demonstrates

- A complete RAG pipeline: document ingestion → chunking → embedding → vector search → grounded generation.
- A **vector database** (Qdrant) used deliberately, with payload filtering and a path to hybrid search.
- A **retrieval evaluation harness implemented by hand** — the metrics are written from scratch and unit-checkable, not imported from a library.
- A thin **FastAPI** serving layer over a shared retriever core (offline eval and online serving reuse the same retriever).
- **AWS** integration kept honest and free-tier-bounded: documents in **S3**, generation via **Amazon Bedrock** (`eu-central-1`).

## 2. Architecture

```
                          ┌──────────────┐
                          │  S3 bucket   │   ESG / compliance PDFs
                          └──────┬───────┘
                                 │
                          ┌──────▼───────┐
                          │  ingest.py   │   load → chunk → embed
                          └──────┬───────┘
                                 │  upsert (chunk_id, vector, payload)
                          ┌──────▼───────┐
                          │    Qdrant    │   vector store (Docker local / Cloud free tier)
                          └──────┬───────┘
                                 │
                        ┌────────▼─────────┐
                        │   retriever.py   │   Retriever protocol (dense → hybrid)
                        └───┬──────────┬───┘
              ┌─────────────┘          └─────────────┐
   ┌──────────▼──────────┐              ┌────────────▼─────────┐
   │   eval_harness.py   │  OFFLINE     │     api/main.py      │  ONLINE
   │  recall@k / MRR     │              │   FastAPI  POST /ask │
   └─────────────────────┘              └────────────┬─────────┘
                                                      │  question + retrieved chunks
                                            ┌─────────▼──────────┐
                                            │    generate.py     │ → Amazon Bedrock
                                            │  answer + citations│   (eu-central-1)
                                            └────────────────────┘
```

The retriever is the single shared component. Decoupling it from both the eval harness and
the API means a change of retrieval strategy (dense → hybrid → reranked) is scored on the
*same* labelled set and exposed through the *same* endpoint, with no duplicated logic.

## 3. Tech stack and rationale

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers | Strong small English model; runs on CPU in seconds; free, deterministic, and works offline in CI and the eval harness. |
| Vector store | Qdrant | Rust/HNSW, payload filtering, single-engine hybrid (dense + sparse); runs in local Docker for dev, free managed tier available. |
| Generation | Amazon Bedrock (`eu-central-1`) | Managed LLM, AWS-native; EU region chosen for data residency, which is coherent for an ESG/compliance corpus. Client is abstracted, so the provider is swappable. |
| Serving | FastAPI | Async, Pydantic request/response models, minimal surface. |
| Evaluation | Hand-rolled recall@k, hit-rate@k, MRR | Implementing the metrics directly (not via a library) is the point — it shows the retrieval quality is understood, not delegated. |
| Corpus | Public ESG / sustainability / compliance PDFs in S3 | Realistic document source; domain chosen to mirror real compliance-RAG use cases. |

## 4. AWS scope (deliberately free-tier-bounded)

- **S3** — source of truth for the document corpus; `ingest.py` reads PDFs from a bucket (with a local-directory fallback for offline development). Storage cost is negligible.
- **Amazon Bedrock** — generation only; embeddings stay local to keep cost and reproducibility under control. Token cost for a demo is a few cents and sits inside new-account free-tier credits.
- **Deployment is intentionally local** (`docker-compose` for Qdrant) to stay at zero standing cost. The FastAPI service is containerised and deployable to a scale-to-zero platform if a live URL is wanted later — that is a stretch item, not a requirement.

## 5. Repository layout (target)

```
rag-eval-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── docker-compose.yml          # Qdrant for local dev
├── docs/
│   └── PROJECT_PLAN.md          # this file
├── rag/                         # shared core
│   ├── __init__.py
│   ├── config.py                # bucket, region, model names, collection name
│   ├── ingest.py                # S3 → load → chunk → embed → upsert
│   ├── retriever.py             # QdrantDenseRetriever (Retriever protocol)
│   └── generate.py              # retrieved chunks → Bedrock → answer + citations
├── api/
│   └── main.py                  # FastAPI: GET /health, POST /ask
└── eval/
    ├── eval_harness.py          # recall@k / hit-rate@k / MRR  [done]
    └── eval_set.jsonl           # hand-labelled query → gold chunk ids
```

## 6. Build plan

Each phase has a clear definition of done; nothing moves forward until the previous phase is verified.

- **Phase 0 — Foundation.** `[done]` Repo, license, `.gitignore`, eval harness with a passing self-check against hand-computed values.
- **Phase 1 — Ingestion.** `config.py` + `ingest.py`: read PDFs from S3 (local fallback), chunk with a deliberate strategy (fixed size + overlap; the chosen size is documented), assign stable chunk IDs.
  *Done when:* a dry run prints chunk counts and stable IDs.
- **Phase 2 — Index & retrieve.** Embed chunks, upsert to Qdrant with payload `{chunk_id, text, source}`; move `QdrantDenseRetriever` into `rag/retriever.py` and point the harness at it.
  *Done when:* `retrieve(query, k)` returns ranked chunk IDs and the harness runs end-to-end (baseline numbers, even if poor).
- **Phase 3 — Generate & serve.** `generate.py` builds a grounded prompt from retrieved chunks, calls Bedrock, returns an answer plus source chunk IDs as citations; `api/main.py` exposes `GET /health` and `POST /ask`.
  *Done when:* `POST /ask` returns `{answer, sources}`.
- **Phase 4 — Evaluate for real.** Hand-label 15–25 query → gold-chunk pairs over the corpus; run the harness; record baseline recall@k / hit-rate@k / MRR in the README.
  *Done when:* a results table with real numbers and a one-line read of whether failures are retrieval or ranking.
- **Phase 5 — Stretch (optional).** Hybrid retrieval (dense + sparse) and/or a cross-encoder reranker, reported as before/after deltas on the *same* eval set; optional scale-to-zero deployment.
  *Done when:* a comparison table showing the delta.

## 7. Evaluation methodology

- **recall@k** = |top-k ∩ relevant| / |relevant| — coverage of all relevant chunks.
- **hit-rate@k** = 1 if any relevant chunk is in the top-k — kept separate from recall@k on purpose (they coincide only when each query has exactly one gold chunk).
- **MRR** = mean reciprocal rank of the first relevant chunk — ranking quality.
- Reported across **k = 1, 3, 5, 10** and always against a baseline, so a retrieval problem (low recall@10) is distinguishable from a ranking problem (good recall@10 but low recall@1 / MRR).
- Answer-level evaluation (faithfulness / answer relevance, e.g. via RAGAS) is a possible later addition; retrieval metrics come first because retrieval quality caps everything downstream.

## 8. Out of scope (on purpose)

To keep the project legible and the retrieval-evaluation story sharp, the following are explicitly excluded: infrastructure-as-code (Terraform / CloudFormation), CI/CD pipelines, always-on paid deployment, GraphRAG, agentic frameworks, and model fine-tuning. They add surface area without strengthening the core thesis, and any of them can be a follow-up project.
