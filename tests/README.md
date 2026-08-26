# Tests

The automated test suite is in this directory and exercises deterministic text
processing, SQLite persistence, retrieval, prompting, orchestration, CLI
behavior, and Foundry Local adapter boundaries with test doubles.

Run it from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

The tests do not download models or require real Foundry Local inference.
