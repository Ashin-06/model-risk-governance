-- Every signed governance certificate, as issued (JSON). certificate_id is derived from the
-- root hash, so the same certificate can't be stored twice.
CREATE TABLE certificates (
    certificate_id TEXT PRIMARY KEY,
    issued_at TEXT NOT NULL,
    issued_by TEXT NOT NULL,
    model_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    disposition TEXT NOT NULL,
    public_key TEXT NOT NULL,
    doc TEXT NOT NULL
);
CREATE INDEX certificates_model ON certificates (model_id, issued_at);
