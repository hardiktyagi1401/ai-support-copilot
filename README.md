# AI Support Copilot

Production-grade Retrieval-Augmented Generation (RAG) platform with semantic search, grounded answers, inline citations, retrieval inspection, multi-tenant isolation, and observability tooling.

## Stack

- Next.js 14
- TypeScript
- FastAPI
- PostgreSQL
- ChromaDB
- OpenAI APIs
- Docker Compose

## Features

- PDF ingestion pipeline
- Semantic chunking
- Vector similarity search
- Citation-grounded responses
- Retrieval inspector
- Multi-tenant architecture
- Query observability & logging
- Hallucination refusal handling
- Citation repair loop

## Architecture

Frontend → FastAPI → Retrieval Pipeline → ChromaDB/PostgreSQL → GPT Generation

## Screenshots

See `/docs/screenshots`.

## Status

Phase 1 — Infrastructure  
Phase 2 — Ingestion Pipeline  
Phase 3 — Retrieval + Generation