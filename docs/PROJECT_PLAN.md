# Project Plan — RAG Evaluation Pipeline

Retrieval-augmented question answering over a PDF document corpus, built around a
**from-scratch retrieval evaluation harness** (recall@k, hit-rate@k, MRR) rather than a
black-box "load PDF, ask question" demo. The evaluation layer is the centrepiece: the goal
is to show that retrieval quality is *measured*, not assumed. The demo corpus is a small
set of NLP research papers, but the pipeline is domain-agnostic — any set of PDFs works.

---

## 1. What this demonstrates

- A complete RAG pipeline: document ingestion → chunking → embedding → vector search → grounded generation.
- A **vector database** (Qdrant) used deliberately, with payloads carrying stable chunk IDs and a path to hybrid search.
- A **retrieval evaluation harness implemented by hand** — the metrics are written from scratch and unit-checkable, not imported from a library.
- A thin **FastAPI** serving layer over a shared retriever core (offline eval and online serving reuse the same retriever).
- **AWS** integration kept honest and free-tier-bounded: documents in **S3**, generation via **Amazon Bedrock** (`eu-central-1`).

## 2. Architecture

```
  AWS S3 (PDFs)
       │
       ▼
   ingest  ──►  chunk  ──►  embed (bge-small)  ──►  Qdrant
                                                      │
                                                  retriever
                                   ┌──────────────────┴──────────────────┐
                                   ▼                                      ▼
                          eval harness (offline)                 FastAPI /ask (online)
                          recall@k · hit@k · MRR                         │
                                                                         ▼
                                                              Amazon Bedrock (Nova)
                                                                answer + source IDs
```

The retriever is the single shared component. Decoupling it from both the eval harness and
the API means a change of retrieval strategy (dense → hybrid → reranked) is scored on the
*same* labelled set and exposed through the *same* endpoint, with no duplicated logic.

## 3. Tech stack and rationale

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers | Strong small English model; runs on CPU in seconds; free, deterministic, and works offline. |
| Vector store | Qdrant (local-file mode) | Production-grade engine (HNSW, payload filtering, a path to hybrid search); local-file mode needs no server for development. |
| Generation | Amazon Bedrock (`eu.amazon.nova-lite-v1:0`, `eu-central-1`) | Managed LLM, AWS-native; EU region chosen for data residency. Client is abstracted, so the provider is swappable. |
| Serving | FastAPI | Async, Pydantic request/response models, minimal surface, auto-generated docs. |
| Evaluation | Hand-rolled recall@k, hit-rate@k, MRR | Implementing the metrics directly (not via a library) is the point — it shows the retrieval quality is understood, not delegated. |
| Document storage | AWS S3, with a local-folder fallback | Realistic document source; the fallback keeps development possible offline. |

## 4. AWS scope (deliberately free-tier-bounded)

- **S3** — source of truth for the document corpus; `ingest.py` reads PDFs from a bucket when `RAG_S3_BUCKET` is set, and from a local `data/` folder otherwise. Storage cost is negligible.
- **Amazon Bedrock** — generation only; embeddings stay local to keep cost and reproducibility under control. Token cost for a demo is a few cents and sits inside new-account free-tier credits.
- **Deployment is intentionally local** to stay at zero standing cost. The FastAPI service is containerisable and deployable to a scale-to-zero platform if a live URL is wanted later — that is a stretch item, not a requirement.

## 5. Repository layout

```
rag-eval-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── docs/
│   └── PROJECT_PLAN.md          # this file
├── rag/                         # shared core
│   ├── ingest.py                # S3 / local → load → chunk → stable IDs
│   ├── build_index.py           # embed → upsert to Qdrant → search
│   ├── retriever.py             # DenseRetriever (returns ranked chunk IDs)
│   ├── generate.py              # retrieved chunks → Bedrock → answer + sources
│   └── label_helper.py          # utility for building the eval set
├── api/
│   └── main.py                  # FastAPI: GET /health, POST /ask
└── eval/
    ├── eval_harness.py          # recall@k / hit-rate@k / MRR
    ├── eval_set.jsonl           # hand-labelled query → relevant chunk IDs
    └── run_eval.py              # score the retriever against the eval set
```

## 6. Build status

- **Foundation** — done. Repo, license, `.gitignore`.
- **Ingestion** — done. Reads PDFs from S3 (local fallback), chunks with overlap, assigns stable chunk IDs.
- **Index & retrieve** — done. Chunks embedded and indexed in Qdrant; a shared `DenseRetriever` returns ranked chunk IDs.
- **Generate & serve** — done. `generate.py` builds a grounded prompt and calls Bedrock; `api/main.py` exposes `GET /health` and `POST /ask` returning an answer plus source chunk IDs.
- **Evaluate** — done. Hand-labelled eval set scored with recall@k / hit-rate@k / MRR. Baseline: recall@10 0.90, MRR 0.71.
- **Stretch (optional, not done):** hybrid retrieval and/or a cross-encoder reranker, reported as before/after deltas on the same eval set; containerisation (Docker) and a scale-to-zero deployment.

## 7. Evaluation methodology

- **recall@k** = |top-k ∩ relevant| / |relevant| — coverage of all relevant chunks.
- **hit-rate@k** = 1 if any relevant chunk is in the top-k — kept separate from recall@k on purpose (they coincide only when each query has exactly one relevant chunk).
- **MRR** = mean reciprocal rank of the first relevant chunk — ranking quality.
- Reported across **k = 1, 3, 5, 10** and against a baseline, so a retrieval problem (low recall@10) is distinguishable from a ranking problem (good recall@10 but low recall@1 / MRR).
- The eval set is small and hand-labelled — a reproducible baseline, not a large benchmark. Answer-level evaluation (faithfulness / answer relevance) is a possible later addition; retrieval metrics come first because retrieval quality caps everything downstream.

## 8. Out of scope (on purpose)

To keep the project legible and the retrieval-evaluation story sharp, the following are explicitly excluded: infrastructure-as-code (Terraform / CloudFormation), CI/CD pipelines, always-on paid deployment, GraphRAG, agentic frameworks, and model fine-tuning. They add surface area without strengthening the core thesis, and any of them can be a follow-up project.