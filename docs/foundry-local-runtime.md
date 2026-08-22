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

## Authoritative sources

- [Get started with Foundry Local on Windows](https://learn.microsoft.com/en-us/windows/ai/foundry-local/get-started)
- [Foundry Local SDK reference](https://learn.microsoft.com/en-us/azure/foundry-local/reference/reference-sdk-current)
- [Generate text embeddings with Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/how-to/how-to-generate-embeddings)
