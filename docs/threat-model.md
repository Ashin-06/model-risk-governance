# Threat model

What the checks service protects, from whom, and what is still open. Written for the 1.0.0 code;
it has not been reviewed by anyone independent.

## What it protects

| Asset | Where it lives |
|---|---|
| Applicant-level data sent to `/audit`, `/conformal/predict`, `/fairness/repair` (features, outcomes, group labels such as sex) | request memory only; not stored or logged |
| Signing key (Ed25519 seed) | `MRMGOV_SIGNING_KEY` in the environment / secret store |
| Issued certificates (summaries, dispositions, who asked, signing public key) | `certificates` table |
| API keys | callers hold the key; the service holds only SHA-256 hashes |

## Trust boundaries

```
validator / analyst / auditor ──(TLS at your ingress)──► checks service ──► database (certificates)
anyone ──► GET /public-key (pin it once, from a trusted channel)
```

1. **Caller → service.** Callers are authenticated by API key; the role decides who may issue
   certificates (`validator`), who may only run the tools (`analyst`) and who may read the register
   (`auditor`).
2. **Service → database.** Holds certificates (aggregate summaries), not applicant rows.
3. **Certificate → its reader.** A certificate is evidence only to someone who holds the signing
   public key from a trusted source, never from the certificate or a response next to it.

## Threats and what stops them

| Threat | Mitigation | Residual risk |
|---|---|---|
| Someone forges or edits a certificate | Ed25519 signature over a root hash that covers the checklist, the disposition and every summary block; verification recomputes the chain | Only as good as the key handling below |
| Signing key stolen | Key comes from the secret store, required in production, never in the code; rotation with `_PREVIOUS` | Whoever holds it can sign anything. There is no revocation list or trusted timestamp, and `issued_at` is the server clock. Keep it in an HSM/KMS. |
| A certificate is mistaken for a model approval | The disposition names only automated checks; README and model card say it is not an approval | People may still read it that way; governance process has to say otherwise |
| A caller feeds chosen data to get a "passed" certificate | Certificates record who asked (`issued_by`) and are kept in the register | **The service certifies whatever data it is given.** It can't tell that the reference and production rows are the real ones. Validators should be people independent of the model owner. |
| Certificate register edited | Each stored certificate still carries its signature, so edits to the stored JSON fail verification | Deleting rows is not detectable from the register alone; export it to write-once storage |
| Applicant data leaks | Not stored or logged; validation errors don't echo input; unhandled errors return only a request ID | Data is in memory while a request runs; TLS is the ingress's job |
| Stolen API key | Keys per client, revocable; rate limit per key; certificates record the user | A stolen `validator` key can issue certificates until removed |
| Brute-forcing keys | 256-bit random keys; rate limit per address for unknown keys | Rate limit is per process |
| CPU exhaustion | 16 MB body cap, 10,000 rows × 50 features, 50,000 points per list, rate limit; `/bench` admin only (and cached) | A maximum-size `/audit` takes about 2.3 s; several in parallel load the CPU |
| Browser attacks on the console | Security headers, CSP, `frame-ancestors 'none'`, no CORS unless origins are listed | The console's CSP allows its own inline script; don't serve it on a shared origin |
| Misconfigured production start | Production refuses to start without hashed keys, the signing key (exact length, base64) and a database; no public bind without keys | Production mode can't check that keys are *good*, only that they exist and have the right shape |

## Not covered

- TLS. Terminate it at your ingress or service mesh; the service speaks plain HTTP.
- Whether the checks are statistically adequate for a given model (see `docs/model-card.md`).
- An insider with the signing key.
