# Foundry Local RAG project plan

## Approved direction

Build a small, demonstrable local document question-and-answer assistant. The
implementation should remain beginner-readable and use a local Retrieval-
Augmented Generation (RAG) flow rather than production-scale infrastructure.

The planned data flows are:

```text
Ingest -> Chunk -> Embed -> Store

Question -> Embed Query -> Retrieve Relevant Chunks -> Build Grounded Prompt
          -> Local LLM -> Answer
```

Microsoft Foundry Local is the planned local inference runtime, and SQLite is
the planned local persistence mechanism. Once the required models and assets
are available locally, the core question-and-answer path should remain
offline-capable. No cloud API or cloud fallback is part of the approved core
direction.

Later tasks will establish runtime compatibility, configuration, and the
application behavior. This document describes the approved direction; it does
not claim that any later capability has been implemented.

## Task sequence

The project is planned as the following ordered tasks:

- **T00** — Repository and project-source setup.
- **T01** — Python and Foundry Local runtime compatibility research.
- **T02** — Application and configuration foundation.
- **T03–T15** — Subsequent approved implementation, verification, and usability
  tasks, to be specified before each task begins.

T00 establishes only the repository foundation. It does not install Foundry
Local, select models or package versions, create a database, or implement
ingestion, chunking, embeddings, retrieval, prompting, answer orchestration,
or a user interface.
