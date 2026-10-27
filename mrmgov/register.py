"""Issued certificates, kept in the database (see db.py) so they can be looked up and
re-verified later. Each row records who asked for it and which public key signs it."""
from __future__ import annotations

import json

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from .report import GovernanceCertificate


class Register:
    def __init__(self, engine: Engine):
        self.db = engine

    def add(self, cert: GovernanceCertificate, user: str, public_key_hex: str) -> None:
        try:
            with self.db.begin() as c:
                c.execute(text("INSERT INTO certificates (certificate_id, issued_at, issued_by, model_id, model_version, "
                               "disposition, public_key, doc) VALUES (:id, :t, :u, :m, :v, :d, :k, :doc)"),
                          {"id": cert.certificate_id, "t": cert.issued_at_utc, "u": user, "m": cert.model_id,
                           "v": cert.model_version, "d": cert.overall_disposition, "k": public_key_hex,
                           "doc": json.dumps(cert.to_dict())})
        except IntegrityError:  # the same root hash: this exact certificate is already stored
            pass

    def get(self, certificate_id: str) -> dict | None:
        with self.db.connect() as c:
            row = c.execute(text("SELECT issued_by, public_key, doc FROM certificates WHERE certificate_id = :id"),
                            {"id": certificate_id}).first()
        return {"issued_by": row.issued_by, "public_key": row.public_key, "certificate": json.loads(row.doc)} if row else None

    def recent(self, model_id: str | None = None, limit: int = 50) -> list[dict]:
        sql = ("SELECT certificate_id, issued_at, issued_by, model_id, model_version, disposition FROM certificates "
               "WHERE model_id = :m ORDER BY issued_at DESC LIMIT :n") if model_id else (
               "SELECT certificate_id, issued_at, issued_by, model_id, model_version, disposition FROM certificates "
               "ORDER BY issued_at DESC LIMIT :n")
        with self.db.connect() as c:
            return [dict(r._mapping) for r in c.execute(text(sql), {"m": model_id, "n": limit})]
