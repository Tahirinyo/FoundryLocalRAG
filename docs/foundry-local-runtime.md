# Foundry Local runtime compatibility evidence

Verified on 2026-08-22 for T01. This records runtime compatibility and two
minimal local inference proofs only; it does not establish application or RAG
behavior.

## Target environment

- Operating-system evidence: `Get-ComputerInfo` and the
  `Win32_OperatingSystem` CIM source report `Microsoft Windows 11 Education`,
  64-bit, display version `25H2`, version `10.0.26200` (build `26200`). The
  `HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion` `ProductName` value
  separately reports `Windows 10 Education`, while its `DisplayVersion` and
  build values report `25H2` and `26200`. The build exceeds Microsoft's
  documented Windows 11 24H2/build 26100 minimum for the Windows Foundry Local
  route; T01 does not infer a stronger product-name conclusion from the legacy
  registry label.
- Memory: 16,515,100 KiB total visible memory (approximately 15.75 GiB).
- Storage at verification: 193,602,830,336 bytes free on `C:` (approximately
  180.30 GiB) after model acquisition.
- Started display devices: NVIDIA GeForce RTX 3050 Ti Laptop GPU and Intel UHD
  Graphics. `nvidia-smi` reported NVIDIA driver `610.62` and 4,096 MiB GPU
  memory.
- The installed SDK catalog selected CPU variants using
  `CPUExecutionProvider`. CUDA, WebGPU, OpenVINO, and NvTensorRTRTX execution
  providers were discoverable but not registered. T01 did not download or
  register optional execution providers.

## Python and Foundry Local SDK

- Selected interpreter: Python 3.11.9, invoked initially with `py -3.11` and
  subsequently through `.venv\Scripts\python.exe`.
- The machine-default interpreter is Python 3.14.2. T01 deliberately selected
  Python 3.11.9 because the installed Foundry Local SDK metadata advertises
  support through Python 3.13; Python 3.14 compatibility was not established
  during T01 and must not be treated as a validated repository alternative
  unless a later task explicitly verifies it.
- Environment: ignored repository-local `.venv/`.
- Revalidation confirmed that both `py -3.11` and the existing `.venv`
  interpreter run as Python 3.11.9 with normal user access. An earlier
  validation attempt ran under a restricted sandbox that could not access the
  Microsoft Store base interpreter beneath `WindowsApps`; the virtual
  environment itself did not require recreation or package reinstallation.
- Selected package: `foundry-local-sdk-winml==1.2.4`. The WinML variant was
  selected because this is a supported Windows build with started physical
  NVIDIA and Intel display devices. The standard SDK variant was not installed.
- Verified package/runtime versions:

  - `foundry-local-sdk-winml` 1.2.4
  - `foundry-local-core-winml` 1.2.4
  - `onnxruntime-core` 1.26.0
  - `onnxruntime-genai-core` 0.14.1

- `import foundry_local_sdk` succeeded. The SDK package declares `openai` as a
  required dependency, so `openai` 3.3.1 was installed transitively. T01 did
  not configure an API key, cloud endpoint, or cloud inference fallback and
  used only the SDK's native Foundry Local clients.
- The unrelated third-party `foundry-local` package and the mutually exclusive
  standard `foundry-local-sdk` package are not installed.

## Catalog-confirmed models

The installed SDK initialized successfully and returned 47 models from its
supported catalog. The documented candidate aliases were both present, so no
fallback selection was needed.

| Role | Alias | Catalog model/variant ID | Task | Capability | Provider | Reported size |
| --- | --- | --- | --- | --- | --- | ---: |
| Embedding | `qwen3-embedding-0.6b` | `qwen3-embedding-0.6b-generic-cpu:1` | `embeddings` | `embedding` | `CPUExecutionProvider` | 495 MB |
| Chat | `qwen2.5-0.5b` | `qwen2.5-0.5b-instruct-generic-cpu:4` | `chat-completion` | `tool-calling` | `CPUExecutionProvider` | 822 MB |

Neither model was cached before T01. Only these two variants were downloaded
into the ignored repository-local `model-cache/`; both were reported cached
afterward. Live revalidation reused these assets without calling the SDK's
download operation and again reported exactly these two cached models.

## Local inference proofs

### Embedding

- Input: `Foundry Local compatibility smoke test.`
- Model: `qwen3-embedding-0.6b-generic-cpu:1`
- Result: one embedding record with dimension 1,024.
- Structural checks: vector was non-empty and every value was numeric and
  finite.
- Lifecycle: model loaded successfully and was unloaded after inference.
- Revalidation reran the same input through the live SDK and reproduced one
  finite 1,024-dimensional vector.

### Chat

- Prompt: `Reply with the single token FOUNDRY_LOCAL_OK.`
- Model: `qwen2.5-0.5b-instruct-generic-cpu:4`
- Result: non-empty local completion `FOUNDRY_LOCAL_OK`.
- Lifecycle: model loaded successfully and was unloaded after inference.
- Revalidation reran the same prompt through the live SDK and reproduced the
  non-empty completion `FOUNDRY_LOCAL_OK`.

The SDK reported no loaded models after both proofs.

## Network and limitations

Network access was required to install the SDK and dependencies, query the
catalog, and acquire both model assets. Inference then ran through native
Foundry Local clients against the repository-local cached models. T01 did not
perform a no-network rerun, so offline operation is not claimed as verified.

Optional accelerated execution providers were not registered; the successful
proofs used CPU variants. This is not an inference blocker, but later work may
separately evaluate acceleration if required. No Foundry Local CLI was present
on `PATH`, and the Python SDK did not require it for these proofs.

## T14 real-RAG smoke attempt

Attempted on 2026-08-25 with the repository's supported Python 3.11.9 virtual
environment and the installed `foundry-local-sdk-winml==1.2.4` runtime. The
live SDK catalog reported exactly the configured embedding and chat variants
as cached and initially unloaded:

- `qwen3-embedding-0.6b-generic-cpu:1`
- `qwen2.5-0.5b-instruct-generic-cpu:4`

No model download, execution-provider installation, model substitution, or
cloud fallback was used. The installed `foundry-local-rag` console entry point
ingested `data/sample/greenhouse.txt` into an isolated temporary SQLite
database and answered the T12 greenhouse-hours question through the real
embedding, retrieval, prompting, and chat boundaries. Both CLI commands exited
successfully. Database inspection found three `greenhouse.txt` chunks with
1,024-dimensional embeddings, and the answer returned all three greenhouse
chunks as source metadata.

The generated answer was:

> The Cedar Grove greenhouse opens to visitors at 9:00 a.m. and closes at
> 4:30 p.m. from Tuesday through Saturday.

This conveyed the opening days and hours but omitted the required fact that the
greenhouse is closed on Sunday and Monday. The T14 real-answer acceptance
criterion therefore was not met, even though the real application path and
both configured models executed successfully.

A controlled offline rerun was attempted by requesting temporary outbound
Windows Firewall rules scoped to the repository virtual-environment Python and
CLI executables. Windows rejected rule creation with system error 5 (`Access
is denied`) before either rule was created. Cleanup confirmed that no T14
firewall rule or temporary smoke database remained. Because the successful
smoke ran while network access remained available, offline operation is not
verified.

Verification results:

- T13 focused integration suite: 3 tests passed.
- Full repository suite: 122 tests passed.
- Temporary SQLite state: removed after inspection.
- Final status: **REAL LOCAL MODEL SMOKE EXECUTED — ANSWER ACCEPTANCE NOT MET —
  OFFLINE STATUS NOT VERIFIED**.

### Focused trail-distance re-smoke

One additional real-model smoke was run on 2026-08-25 with the canonical T12
`answerable-trail-distance` case. The production CLI ingested only
`data/sample/trail-guide.txt` into
`C:\Users\cimeny\AppData\Local\Temp\FoundryLocalRAG-T14-trail-3b5c2590d9424b87bbdc32e655a7d96e\smoke.sqlite3`
and asked `How long is the Willow Loop?`. The exact configured cached models
and Python 3.11.9 environment were reused without download or substitution.

Ingest and ask both exited successfully. Inspection found three persisted
`trail-guide.txt` chunks, each with a 1,024-dimensional embedding. The returned
source order was chunk 0, chunk 2, then chunk 1, all from `trail-guide.txt`.
Chunk 0 contained the required evidence that the Willow Loop is a fictional
three-kilometer trail and that a complete loop usually takes about one hour at
an easy pace.

The generated answer was:

> The Willow Loop is a fictional three-kilometer walking trail that begins
> beside the old stone bridge and ends at a point beyond the meadow sign. It's
> considered a short distance for most people to walk, typically lasting
> around 1-2 hours depending on factors such as weather conditions and personal
> fitness level.

The answer clearly conveyed the required three-kilometer fact, so that narrow
semantic fact check passed. The answer was not fully grounded, however: the
documents do not state that the trail ends beyond the meadow sign, and they say
the loop usually takes about one hour rather than one to two hours based on
weather or fitness. The overall grounded-answer acceptance criterion therefore
remains unmet. With correct retrieval and intact prompt evidence, this second
result strengthens the classification as real small-model output-quality
behavior rather than an ingestion, persistence, retrieval, prompt,
orchestration, or adapter defect.

Postflight reported both configured models cached and unloaded, no normal
application database, no remaining smoke directory, and no generated SQLite
state. A cache snapshot found no added or removed asset paths or size changes;
the existing `foundry.modelinfo.json` modification timestamp changed during
the SDK lifecycle. Offline status remains not verified, and no further network
isolation was attempted.

Focused re-smoke status: **T14 REMAINS BLOCKED — REAL-MODEL ANSWER ACCEPTANCE
NOT MET — OFFLINE STATUS NOT VERIFIED**.

### Post-correction trail-distance smoke

The single authorized post-correction smoke ran on 2026-08-25 after tightening
the grounded system prompt and setting the chat client's temperature to `0.0`.
Preflight confirmed that the exact configured embedding and chat variants were
cached and initially unloaded. No model download, substitution, or cloud
fallback was used.

The production CLI ingested only `data/sample/trail-guide.txt` into
`C:\Users\cimeny\AppData\Local\Temp\FoundryLocalRAG-T14-post-correction-de2fe12d0ac447f1a3ca81e33397e373\smoke.sqlite3`.
Ingestion exited successfully and persisted the canonical source as three
chunks with 1,024-dimensional embeddings. The production CLI then performed
exactly one completion for `How long is the Willow Loop?`. Retrieved source
order was chunk 0, chunk 2, then chunk 1, all from `trail-guide.txt`.

The generated answer was:

> The Willow Loop is a fictional three-kilometer walking trail that begins
> beside the old stone bridge. It usually takes about one hour to walk.

The answer included the required three-kilometer fact. Its additional factual
claims—the fictional trail description, its start beside the old stone bridge,
and the usual approximately one-hour duration—were all explicitly supported by
retrieved chunk 0. The returned source metadata included `trail-guide.txt`, and
no unsupported inference, estimate, extrapolation, or general-knowledge detail
was present. Grounded-answer acceptance therefore passed.

Both configured models were cached and unloaded after command completion. The
temporary database and directory were removed, and `data/rag.sqlite3` was not
created. The cache contained the same 23 paths and file sizes before and after
the smoke; only the existing `foundry.modelinfo.json` modification timestamp
changed during SDK lifecycle use. No network isolation was attempted, so the
final T14 offline status remains **OFFLINE STATUS NOT VERIFIED**.

Post-correction status: **T14 REAL-MODEL GROUNDED-ANSWER ACCEPTANCE PASSED —
OFFLINE STATUS NOT VERIFIED**.

## Authoritative sources

- [Get started with Foundry Local on Windows](https://learn.microsoft.com/en-us/windows/ai/foundry-local/get-started)
- [Foundry Local SDK reference](https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-sdk-current)
- [Generate text embeddings with Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/how-to/how-to-generate-embeddings)
