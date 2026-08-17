# Self-Service Feishu CSV Intake Design

## Goal

One Feishu CSV upload must automatically enter the controlled analysis pipeline without a pre-created SHA binding. Aily must accept one or many scenario parameter changes in one request.

## Intake Contract

- Existing registered demo CSV files keep their reviewed registrations.
- An unseen file may enter through a self-describing CSV layout. The file must contain stable cell metadata (`cell_id`, `chemistry`, `nominal_capacity_ah`, `reference_capacity_ah`, and `cell_format`) plus unambiguous cycle columns and units.
- The server maps reviewed aliases to the canonical `CycleRecord` schema, verifies one cell identity, computes source and canonical SHA-256 values, and chooses the highest supported cutoff not exceeding the complete observed cycles (`20`, `50`, `100`, or `150`).
- The server creates an immutable registration and a project-scoped frozen batch binding automatically. Re-uploading identical bytes resolves the same identity.

## Analysis Gate

- Missing, ambiguous, non-finite, or inconsistent required data stops model execution and returns explicit validation reasons.
- Complete data proceeds only when dataset semantics, feature version, cutoff route, and supported domain match an activated model route.
- A complete but unsupported file remains registered and auditable, but RUL/SOH returns an explicit evidence-insufficient result. It must never be relabelled as MATR merely to force inference.
- Business values continue to come only from valid `ToolResult` objects.

## Aily Scenario Updates

- Aily may change any subset of temperature, charge/discharge rate, SOC bounds, DoD, annual EFC, rest duration, horizon, EOL threshold, and cell format in one message.
- Unchanged values are inherited from the last accepted context.
- The full candidate context is validated atomically. If one changed value is invalid, no partial context is stored.
- Every accepted update creates a new immutable scenario context and analysis task; prior contexts remain traceable.
- The existing comparison request continues to support up to eight comparison scenarios.

## Delivery Scope

The implementation reuses existing parsers, Pydantic contracts, project context, model routes, support gates, cards, reports, and Bitable delivery. It does not retrain models, weaken provenance, or make unsupported external cells produce numerical predictions.
