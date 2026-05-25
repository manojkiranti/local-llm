# Document Group Indexes & Verification

Group your reference documents under a named index (e.g. `loan_process`, `kyc`, `compliance`) and verify a new document against just that group — without scanning the whole database.

---

## Why

Before this change, every embedded document lived in one Qdrant collection and `/ask` / `/search` always retrieved from the entire database. There was no way to say *"use only my loan-processing reference docs"*.

A **group** gives you:

- A **namespace** for related reference documents
- A **description** stored once and injected into the LLM prompt so the model understands the domain
- **Scoped retrieval** — both regular Q&A and the new `/verify` endpoint can be filtered to a group
- A **structured verification report** comparing a newly uploaded document against the group's reference material

---

## Do I need to run a migration?

**Yes, on environments that already have data in PostgreSQL.** The app uses SQLAlchemy `create_all`, which creates **new** tables but does **not** alter existing ones. So:

- The new `document_groups` table is created automatically on next startup ✓
- The new `group_name` column on the existing `embedded_files` table is **not** added automatically ✗

Run this SQL once against your Postgres database:

```sql
ALTER TABLE embedded_files ADD COLUMN IF NOT EXISTS group_name VARCHAR;
ALTER TABLE embedded_files
    ADD CONSTRAINT embedded_files_group_name_fkey
    FOREIGN KEY (group_name) REFERENCES document_groups(name) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS ix_embedded_files_group_name
    ON embedded_files (group_name);
```

If your database is empty (e.g. fresh dev box, or you're happy to drop and recreate), you can skip the ALTER — `create_all` will pick everything up. To wipe and restart cleanly:

```bash
# DESTRUCTIVE — removes all embedded documents and notices
docker compose down -v
docker compose up -d
```

Qdrant payload index is created idempotently on startup — nothing to do there.

---

## Workflow

### 1. Create a group

```bash
curl -X POST http://localhost:8000/groups \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "loan_process",
    "description": "Australian owner-occupied home loan checklist. Required items: borrower identification (passport or driver licence), 3 most recent payslips, signed loan offer, property valuation, serviceability calculator, and consent to credit check."
  }'
```

**Naming rules:** `^[a-zA-Z0-9_\-]+$`, 1–64 characters. The `description` is what the LLM sees during verification — make it concrete.

### 2. Upload reference documents into the group

```bash
curl -X POST http://localhost:8000/documents/files/upload \
  -F "files=@policy_v1.pdf" \
  -F "files=@checklist.docx" \
  -F "files=@servicing_rules.txt" \
  -F "group_name=loan_process"
```

The upload step copies files to the docs folder and validates the group exists. It does **not** embed yet.

### 3. Embed them

```bash
curl -X POST http://localhost:8000/documents/embed \
  -H 'Content-Type: application/json' \
  -d '{
    "filepaths": [],
    "group_name": "loan_process"
  }'
```

Passing `filepaths: []` embeds every new file in `DOCS_DIR`. Pass explicit paths to limit it.

Returns a `task_id`. Poll progress with:

```bash
curl http://localhost:8000/documents/embed/status/<task_id>
```

Every chunk produced by this run is tagged with `group_name=loan_process` in its Qdrant payload, and each file is recorded in `embedded_files` with the same group.

### 4. Verify a new document against the group

```bash
curl -X POST http://localhost:8000/groups/loan_process/verify \
  -F "file=@new_loan_application.pdf" \
  -F "instruction=Check that borrower ID, all three payslips, property valuation, and a signed loan offer are present. Flag missing income evidence." \
  -F "top_k=10"
```

Response:

```json
{
  "group_name": "loan_process",
  "instruction": "Check that borrower ID, all three payslips, ...",
  "filename": "new_loan_application.pdf",
  "satisfied": [
    {"requirement": "Borrower ID provided", "evidence": "Passport number AU1234567 issued 2022..."},
    {"requirement": "Signed loan offer attached", "evidence": "...signed on 2026-05-12 by John Smith..."}
  ],
  "missing": [
    {"requirement": "Three most recent payslips", "evidence": null},
    {"requirement": "Property valuation report", "evidence": null}
  ],
  "unclear": [
    {"requirement": "Serviceability calculator output", "evidence": "Form referenced but values not shown"}
  ],
  "summary": "Application is missing payslips and a valuation report. ID and signed offer are present.",
  "reference_files": ["policy_v1.pdf", "checklist.docx"]
}
```

How it works under the hood:

1. The uploaded file is extracted to text (reusing `services/embedding.extract_text` + `normalize_text`).
2. A retrieval query is built from `instruction` + the first ~1500 chars of the target document — this captures both rule-driven matches (from the instruction) and content-driven matches (from the target).
3. Qdrant returns top-k chunks **filtered by `group_name`**.
4. The LLM gets: group description, instruction, retrieved reference chunks, full target text — and returns the structured JSON above.

### 5. Scoped Q&A (optional)

Regular `/ask` and `/search` accept an optional `group_name`:

```bash
# Ask a question scoped to a single group
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What documents are required for a self-employed applicant?",
    "group_name": "loan_process",
    "top_k": 5
  }'

# Semantic search scoped to a group
curl "http://localhost:8000/search?query=valuation+report&group_name=loan_process"
```

Leaving `group_name` out keeps the original whole-database behaviour.

### 6. Manage groups

| Endpoint | Method | Purpose |
|---|---|---|
| `/groups` | `POST` | Create group `{name, description}` |
| `/groups` | `GET` | List groups + document counts |
| `/groups/{name}` | `GET` | Group details + document count |
| `/groups/{name}` | `PATCH` | Update description `{description}` |
| `/groups/{name}` | `DELETE` | Delete group, its `embedded_files` rows, and Qdrant points |

`DELETE /groups/{name}` is **destructive** — it removes:
- The Qdrant points with `payload.group_name == name`
- The `embedded_files` rows with that `group_name`
- The `document_groups` row itself

The physical files in `DOCS_DIR` are kept. Delete them with `DELETE /documents/files/{filename}` if you also want to remove them from disk.

---

## Tips

- **Use a tight description.** The LLM uses it to decide what counts as a requirement. "Loan documents" is too vague; "Australian owner-occupied home loan — must include borrower ID, 3 payslips, valuation, signed offer" is much better.
- **Re-use the instruction field per call.** The description is fixed; the per-call `instruction` is where you ask for specific checks (e.g. *"check that payslips are no older than 90 days"*).
- **Bump `top_k`** for large reference corpora — default is 10. The verification prompt fits comfortably with 20–30 chunks for `bge-m3` chunk sizes.
- **Re-embedding** a file with the same path overwrites its old vectors (chunk IDs are deterministic on the file path). To move a file between groups, re-embed it with the new `group_name`.
- **No background queue** for verification — it runs synchronously like `/ask`. Long target documents may exceed Ollama's default context; trim or pre-summarise if needed.
