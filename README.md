# Foundry Local RAG

A small local document question-and-answer assistant built with Microsoft Foundry Local. It reads `.txt` documents, splits them into deterministic chunks, embeds and stores those chunks in SQLite, retrieves relevant evidence for a question, and asks a local chat model for a grounded answer with source information.

This is a demonstrable local RAG project, not production-scale infrastructure. There is no cloud API or cloud fallback in the core path.

## How it works

Ingestion:

```text
Document (.txt) → Read/Chunk → Foundry Local Embeddings → SQLite
```

Question answering:

```text
Question → Query Embedding → Cosine Retrieval (top 3)
         → Grounded Prompt → Foundry Local Chat
         → Answer + Retrieved Sources
```

The main responsibilities are deliberately small and direct:

- `src/foundry_local_rag/text_processing.py` reads UTF-8 text and chunks paragraphs.
- `embeddings.py` uses the cached Foundry Local embedding model.
- `persistence.py` stores chunks and embeddings in SQLite.
- `retrieval.py` ranks stored chunks with in-process cosine similarity.
- `prompting.py` supplies retrieved text as untrusted reference data and builds the grounding prompt.
- `chat.py` uses the cached Foundry Local chat model.
- `answering.py` coordinates retrieval, prompting, and chat with a fixed top-K of 3.
- `cli.py` provides ingestion, questions, output, and resource cleanup.

## Requirements and setup

The verified target environment is Windows with Python 3.11. The project declares Python `>=3.11,<3.14`; Python 3.14 has not been validated.

Create an environment and install the project from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

The application uses `foundry-local-sdk-winml==1.2.4`.

### Foundry Local models

The application expects these exact cached model variants:

- Embeddings: `qwen3-embedding-0.6b-generic-cpu:1`
- Chat: `qwen2.5-0.5b-instruct-generic-cpu:4`

On a new machine, make these model assets available through the Foundry Local SDK/catalog before running the application. Initial model acquisition may require network access. The application uses the repository-local `model-cache\` directory by default, checks that each model is already cached, and does not download missing models during normal execution. If a required model is unavailable, the command fails instead of silently substituting another model or using a cloud service.

From the repository root, run this one-time acquisition step while network access is available:

```powershell
@'
from pathlib import Path

from foundry_local_sdk import Configuration, FoundryLocalManager

cache_dir = Path("model-cache").resolve()
FoundryLocalManager.initialize(
    Configuration(
        app_name="foundry_local_rag_setup",
        model_cache_dir=str(cache_dir),
    )
)
manager = FoundryLocalManager.instance
if manager is None:
    raise RuntimeError("Foundry Local SDK did not initialize")

model_ids = (
    "qwen3-embedding-0.6b-generic-cpu:1",
    "qwen2.5-0.5b-instruct-generic-cpu:4",
)
for model_id in model_ids:
    model = manager.catalog.get_model_variant(model_id)
    if model is None:
        raise RuntimeError(f"Model variant is unavailable: {model_id}")
    if not model.is_cached:
        print(f"Downloading {model_id}...")
        model.download()
    print(f"Cached: {model_id}")
'@ | .\.venv\Scripts\python.exe -
```

This setup command is the only documented model-download step. After both variants report `Cached`, normal `ingest` and `ask` commands use the local cache and never request a model download. Fully network-isolated execution has not been verified.

See [`docs/foundry-local-runtime.md`](docs/foundry-local-runtime.md) for the verified runtime details and model evidence. The official Foundry Local setup documentation is also listed there.

## Usage

The console command is `foundry-local-rag`.

Ingest a supported UTF-8 text document:

```powershell
foundry-local-rag ingest data/sample/trail-guide.txt
```

This creates the default database at `data/rag.sqlite3` when needed. The database parent directory is created automatically. Re-ingesting the same canonical source path replaces its previous chunks.

Ask a question about the ingested documents:

```powershell
foundry-local-rag ask "How long is the Willow Loop?"
```

The output has this shape; generated wording and source ordering are not guaranteed to be identical on every run:

```text
Answer:
<grounded answer from the local chat model>

Sources:
- <canonical path>\data\sample\trail-guide.txt (chunk 0)
- ...
```

The answer is based on retrieved document evidence. Retrieved text is treated as untrusted data, not as instructions. The prompt prohibits unsupported inference, extrapolation, and general-knowledge additions. Sources are returned separately from the answer.

If an initialized database contains no chunks, the deterministic result is returned without invoking chat:

```text
I don't know based on the retrieved documents.
```

Missing files, unavailable models, invalid databases, and other application failures are reported as CLI errors with a nonzero exit status.

## Sample knowledge base

The repository-owned sample corpus is in [`data/sample/`](data/sample/):

- [`greenhouse.txt`](data/sample/greenhouse.txt)
- [`workshop.txt`](data/sample/workshop.txt)
- [`trail-guide.txt`](data/sample/trail-guide.txt)

The review-friendly evaluation cases are in [`evaluation.md`](data/sample/evaluation.md), with the canonical structured form in [`evaluation.json`](data/sample/evaluation.json). They cover answerable questions, unsupported questions, source attribution, cross-source retrieval, and the empty-knowledge-base fallback.

## Verification

### Automated tests

The full repository suite passes **123 tests**, including **3 focused integration tests**. The suite covers deterministic text processing, persistence, retrieval, prompting, answer orchestration, CLI behavior, model boundaries, and empty-knowledge-base handling with deterministic test doubles.

### Real Foundry Local model smoke

T14 used the production CLI with Python 3.11.9 and the exact cached embedding and chat models listed above. The post-correction smoke ingested `trail-guide.txt`, answered `How long is the Willow Loop?`, returned source metadata, and passed the narrow grounded-answer acceptance check for that case. Models were unloaded and temporary database state was removed.

This is evidence for the tested scenario, not a general guarantee of answer quality or perfect grounding for every question.

### Offline status

**OFFLINE STATUS NOT VERIFIED.**

Initial model acquisition required network access. The documented Windows network-isolation attempt was blocked by authorization before a controlled offline rerun could be completed. The architecture is designed to use local cached assets after setup, but this repository does not claim empirically verified network isolation.

## Limitations

- The chat model is small, so answer quality is not guaranteed for every question.
- Similarity search is brute-force and in-process over SQLite data; it is intended for this small demonstration corpus.
- Only UTF-8 `.txt` input is supported.
- Retrieval returns up to three chunks and has no relevance threshold, so unrelated chunks may appear for unsupported questions.
- There is no cloud fallback by design.
- Grounding instructions and the tested post-correction case provide safeguards, but they do not establish perfect grounding for all model outputs.
- Offline operation remains unverified as described above.

## Further reading

- [`docs/foundry-local-runtime.md`](docs/foundry-local-runtime.md) — detailed runtime and T14 evidence.
- [`docs/project-plan.md`](docs/project-plan.md) — approved project direction.
- [`PROGRESS.md`](PROGRESS.md) — project task status.
