# Sample evaluation set

These cases use the three fictional `.txt` documents in this directory. They
are part of the completed automated integration and real-model validation
workflow.

Expected behavior describes required facts and source files rather than exact
answer wording. Runtime source identifiers are canonical absolute paths; the
filenames below are the repository-relative expectations.

`expected_sources` lists required supporting-evidence source filenames. Match
these as an unordered subset of the returned source metadata; additional
retrieved sources are allowed because current top-K retrieval has no relevance
threshold. For a populated knowledge base, an empty `expected_sources` list
means that no source contains supporting evidence; it does not require the
runtime source list to be empty. The empty-knowledge-base case is the explicit
case for the deterministic empty-source fallback.

For an unanswerable question against an ingested sample database, the answer
must be evidence-limited and must not invent facts. The current application
has no relevance threshold, so these cases do not require the deterministic
fallback when unrelated chunks are retrieved. The exact fallback,
`I don't know based on the retrieved documents.`, is required by the empty
knowledge-base edge case.

| ID | Category | Question | Expected behavior | Source(s) | Fallback |
| --- | --- | --- | --- | --- | --- |
| `answerable-greenhouse-hours` | Answerable | When is the Cedar Grove greenhouse open to visitors? | Tuesday–Saturday, 9:00 a.m.–4:30 p.m.; closed Sunday and Monday. | `greenhouse.txt` | No |
| `answerable-workshop-equipment` | Answerable | Which tools does the Pine Street workshop provide for registered participants? | Hand saws, clamps, and measuring tapes; not safety glasses. | `workshop.txt` | No |
| `answerable-trail-distance` | Answerable | How long is the Willow Loop? | Three kilometers. | `trail-guide.txt` | No |
| `answerable-trail-bicycle-rule` | Answerable | Where are bicycles allowed on the Willow Loop? | Only on the yellow-marked section. | `trail-guide.txt` | No |
| `unanswerable-greenhouse-admission` | Unanswerable | How much does admission to the Cedar Grove greenhouse cost? | No admission price is provided; do not invent one. | None | No |
| `unanswerable-workshop-instructor` | Unanswerable | Who teaches the Pine Street woodworking session? | No instructor is identified. | None | No |
| `unanswerable-trail-weather` | Unanswerable | What will the weather be like on the Willow Loop tomorrow? | No forecast is provided; do not use outside information. | None | No |
| `edge-cross-source-safety` | Edge | What visitor conduct rules apply at the greenhouse, and what participant safety rules apply at the workshop? | Use both sources and keep their rules distinct. | `greenhouse.txt`, `workshop.txt` | No |
| `edge-source-attribution` | Edge | What rule applies to dogs on the Willow Loop? | State the leash rule and require `trail-guide.txt` in returned source metadata; the answer need not name the filename. | `trail-guide.txt` | No |
| `edge-empty-knowledge-base` | Edge | What are the greenhouse visitor hours? | With no persisted chunks, return the exact fallback without invoking chat. | None | Yes |

`evaluation.json` is the canonical structured representation; this table is
the review-friendly equivalent.
