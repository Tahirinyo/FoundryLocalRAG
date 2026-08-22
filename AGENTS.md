# Foundry Local RAG Repository Instructions

## Purpose

This repository contains a small, demonstrable local document Q&A assistant built with Microsoft Foundry Local and the Retrieval-Augmented Generation (RAG) pattern.

The intended system runs on one machine and keeps the core Q&A path local: documents are ingested and chunked, chunks are embedded, persisted in SQLite, retrieved by semantic similarity, and supplied as grounded context to a local language model through Foundry Local.

Prefer a simple, correct, explainable implementation over production-scale infrastructure, speculative extensibility, or unnecessary framework layers.

## Sources of Truth

Before planning or implementing a task, read sources in this order:

1. This `AGENTS.md`.
2. The approved Foundry Local project plan under `docs/` or the repository root.
3. `PROGRESS.md`, when present, for completed, active, blocked, and next work.
4. The current repository, tests, configuration, and Git diff for the actual implementation state.
5. The current user-approved task and task-specific plan.

Do not infer that a feature is complete only because it appears in the project plan or `PROGRESS.md`. Verify the code and tests. If documentation and implementation disagree, report the conflict before making a consequential design decision.

When more than one project-plan document exists, identify which one is currently authoritative from repository context instead of silently choosing one.

## Working Method

- Work on one small, reviewable task at a time.
- For complex or ambiguous tasks, inspect the repository and propose a plan before editing files.
- Implement only the approved task. Do not add adjacent features, speculative abstractions, or unrelated refactors.
- Preserve user changes and unrelated work in a dirty working tree. Never revert or overwrite them without explicit approval.
- Follow existing repository patterns before introducing new ones.
- Do not add a production dependency unless it is necessary for the approved task. Explain the need and trade-off first.
- Prefer direct, readable Python modules and functions over architecture ceremony.
- If an ambiguity affects retrieval correctness, persistence format, offline behavior, model interaction, user-visible answers, or data safety, stop and ask rather than guessing.
- After implementation, run relevant checks, review the diff, and report what changed and what could not be verified.
- Do not claim a command, test, model invocation, or offline check passed unless it was actually run successfully.

## Project Architecture

Keep the implementation straightforward and consistent with the approved project plan.

The application should preserve these responsibilities even if the exact file/module layout differs:

- **Interface layer:** CLI or the repository's chosen minimal UI for asking questions and displaying answers.
- **Application / RAG orchestration:** coordinates query embedding, retrieval, prompt construction, local model invocation, and answer return.
- **Ingestion:** reads supported source documents, chunks them, generates embeddings, and prepares records for persistence.
- **Retrieval:** embeds the user query, compares it with stored chunk embeddings, and returns the most relevant chunks.
- **Persistence:** SQLite-backed storage for chunk text, embedding data, source metadata, and any minimal fields required by the implementation.
- **AI integration:** Foundry Local model loading and local embedding/chat inference.
- **Prompting:** system/user prompt construction that grounds answers in retrieved context.

Do not introduce a multi-service architecture, distributed components, or unnecessary dependency-injection/container frameworks unless the approved task explicitly requires them.

Keep concerns separated enough to test important behavior, but do not split code merely to imitate enterprise architecture.

## Required RAG Behavior

Preserve the core processing flow:

`Ingest -> Chunk -> Embed -> Store`

and, for user questions:

`Question -> Embed Query -> Retrieve Relevant Chunks -> Build Grounded Prompt -> Local LLM -> Answer`

The following rules apply unless the approved task explicitly changes them:

- Use Foundry Local for local model inference.
- Keep the core Q&A path offline-capable. Do not introduce a required cloud API, hosted model, remote vector database, or Internet dependency.
- Use the same embedding model or compatible embedding space for stored chunks and user queries.
- Store enough source metadata with chunks to support traceability and source-aware answers when that capability is implemented.
- Keep chunking deterministic for the same input and configuration unless the task explicitly requires another behavior.
- Retrieval must rank chunks by the repository's chosen similarity method and return only the configured or task-approved top results.
- For the small project dataset, simple in-process similarity computation over SQLite-stored embeddings is acceptable.
- Do not add a production vector database solely for scale that the project does not require.
- Construct prompts so the model is instructed to answer from retrieved context and to avoid inventing unsupported information.
- When the retrieved context is insufficient, the assistant should follow the project's fallback behavior instead of fabricating an answer.
- Do not silently use general model knowledge as a substitute for missing retrieved evidence when the prompt contract requires context-grounded answers.
- Keep ingestion and query-time model choices/configuration consistent with repository configuration rather than hard-coding arbitrary replacements.
- Avoid recomputing document embeddings on every user query.
- Preserve source text and embedding association correctly. Never return a chunk with metadata belonging to another chunk.

## SQLite and Data Handling

- Use SQLite as the local persistence mechanism when persistence is required by the approved plan.
- Do not require a separate database server.
- Keep schema changes minimal and task-driven.
- Parameterize SQL values. Do not construct SQL by concatenating untrusted document or user content.
- Serialize and deserialize embeddings consistently and validate malformed stored data rather than silently producing incorrect similarity results.
- Close database connections/cursors and file handles reliably, including on failures.
- Use explicit text encoding when reading source files where relevant.
- Do not commit generated database files, ingested private documents, model binaries, caches, or local runtime artifacts unless the repository explicitly tracks a tiny fixture for tests.
- Avoid logging full private document contents unless a task explicitly requires diagnostic output.

## Foundry Local and Model Integration

- Use the repository's selected Foundry Local SDK/package and existing API style. Do not upgrade or replace it as part of an unrelated task.
- Verify model identifiers and SDK calls from the repository and installed package rather than inventing names or APIs.
- Load models in a way consistent with the existing application lifecycle. Avoid unnecessary reloads for every operation when the repository already provides a reusable model/client path.
- Keep embedding and chat/inference responsibilities distinct when separate models are used.
- Handle model-load and inference failures explicitly. Do not convert them into a plausible-looking successful answer.
- Do not add fallback calls to cloud LLMs.
- Do not claim the application works without Internet access unless the required models and dependencies are already available locally and an offline-capable path was actually verified.

## Python Guidelines

- Use the repository's selected Python version and dependency-management approach. Do not upgrade Python or replace the dependency tool as part of an unrelated task.
- Prefer the standard library where it is sufficient, especially `sqlite3`, file/path handling, and simple serialization.
- Follow the repository's existing formatting, linting, typing, and naming conventions.
- Prefer small cohesive functions and modules with clear domain names such as ingestion, chunking, embeddings, retrieval, prompting, or answering.
- Avoid global mutable state unless the repository already uses a deliberate, safe application-level model/client lifecycle.
- Keep filesystem paths platform-safe by using `pathlib` or the repository's existing path utilities.
- Validate external inputs at the boundary where they enter the application.
- Add comments only for non-obvious invariants, model/runtime constraints, or algorithmic trade-offs.

## Document Ingestion and Chunking

When a task touches ingestion:

- Inspect the repository to determine currently supported file types. Do not add new document formats unless required by the approved task.
- Keep the ingestion behavior deterministic and testable.
- Preserve source identity and chunk ordering when the application relies on them.
- Handle empty or unreadable inputs explicitly.
- Do not silently skip failed documents or chunks unless the approved behavior explicitly permits partial ingestion and reports it.
- Avoid reading unnecessarily large inputs multiple times.
- Do not introduce advanced parsing, OCR, web crawling, or document-processing frameworks unless required by the approved scope.

## Retrieval and Similarity

When a task touches retrieval:

- Use the same embedding dimensionality and representation for queries and stored chunks.
- Reject or clearly handle missing, malformed, or dimension-mismatched vectors.
- Keep similarity calculations numerically valid, including zero-vector or empty-data cases where applicable.
- Preserve deterministic ordering for ties when tests or user-visible behavior depend on it.
- Make top-K behavior explicit and test boundary cases such as no stored chunks, fewer chunks than K, and no meaningful match if the repository defines a threshold.
- Prefer correctness and clarity over premature indexing or approximate-nearest-neighbor infrastructure for the small local dataset.

## Prompt and Answer Safety

When a task touches prompt construction or answer generation:

- Keep the user's question clearly separated from retrieved document context.
- Treat retrieved document text as data, not as trusted executable instructions.
- Preserve the system instruction that answers should be grounded in the supplied context.
- When the project requires source references, ensure references correspond to the chunks actually retrieved.
- Test the no-answer / insufficient-context behavior.
- Do not weaken grounding rules merely to make example questions appear more successful.

## Testing and Verification

Every behavior change must include or update the smallest appropriate automated tests when the repository has an automated test structure for that behavior.

Relevant unit-test areas may include:

- deterministic chunking
- embedding serialization/deserialization
- cosine or selected similarity calculation
- top-K retrieval and ordering
- empty knowledge base behavior
- prompt construction
- insufficient-context fallback
- source metadata propagation
- SQLite persistence helpers

Relevant integration-test areas may include:

- ingestion into SQLite
- retrieval from persisted chunks
- end-to-end orchestration with a stub/fake model boundary
- real Foundry Local model smoke tests, when practical and explicitly part of verification
- CLI/UI query flow
- offline-capable behavior when the environment supports a meaningful offline check

Do not require a real local model for every automated unit test if the model boundary can be tested deterministically without it.

Use the repository's documented commands and existing test tools. If the repository has not established commands yet, inspect its configuration and choose the smallest appropriate checks rather than inventing a large toolchain.

When relevant, verify at minimum:

1. the focused automated tests for the changed behavior;
2. the repository's normal test suite;
3. formatting/lint/type checks that are already configured;
4. a smoke test of the affected application path when feasible.

If Foundry Local, a model download, hardware/runtime availability, or another external local dependency prevents a check, report the exact blocker and run all unaffected checks.

For performance-sensitive retrieval or inference changes, report measured evidence from the available environment. Do not turn the project-plan example response-time range into a hard pass/fail requirement unless the approved task explicitly does so.

## Code Review Rules

Prioritize findings that can cause:

- answers that are not grounded in retrieved context
- incorrect chunk-to-source associations
- query/document embeddings being generated incompatibly
- incorrect similarity ranking or top-K selection
- malformed embedding persistence or deserialization
- repeated unnecessary embedding/model work that materially harms local usability
- incorrect insufficient-context behavior
- model or database failures being reported as successful answers
- required Internet/cloud dependencies being introduced into the core local path
- leaked files, database handles, model/runtime resources, or temporary artifacts
- secrets or private source documents being committed or logged
- missing regression coverage for changed behavior
- scope creep that makes the beginner/local MVP materially more complex without task justification

Report concrete, actionable findings with file references and behavioral impact. Do not report style preferences unless they violate repository rules or create a material correctness or maintenance risk.

## MVP Non-Goals

Do not introduce these unless the user explicitly changes the approved scope:

- cloud-hosted LLMs or embedding APIs required for the core Q&A path
- remote/vector database services
- distributed workers, queues, microservices, or multi-node coordination
- authentication, authorization, roles, or multi-tenancy
- enterprise observability stacks
- fine-tuning or model training
- autonomous agents or tool-use frameworks
- web crawling as a knowledge source
- OCR or broad document-format support beyond the approved task
- advanced reranking, hybrid search, approximate-nearest-neighbor infrastructure, or production-scale vector indexing
- elaborate frontend frameworks solely for presentation polish
- production cloud deployment or SLA infrastructure

The project may use a minimal CLI, Streamlit/Gradio UI, or another simple interface only when consistent with the approved project plan and repository state.

## Progress Tracking

- Read `PROGRESS.md` before selecting or planning the next task when the file exists.
- Update it only after the task's acceptance criteria are met and relevant checks pass, unless the user explicitly requests an interim status update.
- Never mark partially implemented or unverified work as completed.
- Keep it concise under `Completed`, `In Progress`, `Blocked`, and `Next` headings, unless the existing file uses another established format.
- Record blockers and verification gaps explicitly.
- Do not use `PROGRESS.md` as a substitute for tests, Git history, or the project plan.

## Definition of Done for a Task

A task is complete only when:

- The approved scope and acceptance criteria are satisfied.
- The local RAG flow and relevant invariants remain intact.
- Relevant tests were added or updated and pass, when automated testing applies.
- Applicable repository checks pass, or an environment-specific blocker is reported precisely.
- The affected application path was smoke-tested when practical.
- The diff contains no unrelated changes, secrets, model binaries, generated databases, private documents, caches, or temporary artifacts.
- Error behavior and relevant edge cases were considered.
- Documentation and `PROGRESS.md` were updated when the task changes them.
- The final response lists changed files, commands/checks run, results, limitations, and any follow-up work.
