---
evidence_status: synthetic
data_status: synthetic
is_verified: false
usable_for_public_answers: false
---

# Synthetic Transportation Dataset

This dataset exists exclusively for development and testing of the Transport AI
RAG pipeline. It does not describe real-world transport services in Lafia or
anywhere else.

All routes, stops, fares, operators, schedules, operating days, and journey
durations in these files are fictional. Names beginning with `TEST`, `Demo`, or
`Synthetic` are artificial test data, not real locations or service providers.

Synthetic information must never be mixed with verified or historical
transport information. No synthetic record may be presented to a real
passenger as a transport fact or used for real passenger guidance.

## Dataset files

- `synthetic_keke_routes.md`: ten fictional Keke route records.
- `synthetic_transport_services.md`: five fictional intercity and five
  fictional shared-taxi service records.

Do not ingest these files into a production knowledge base. Use only in an
isolated test environment with synthetic data clearly separated from
real-world information.
