# Training / evaluation data — applicability model

Provenance only. **Nothing here is loaded at runtime.** The runtime standards
knowledge base is `ai-engine/data/bis_full_knowledge_base.json`.

## `synthetic_requirements_2000_hard.csv`

Synthetic training set for the applicability classifier
(`standiq_applicability_model_v2.joblib`).

| | |
|---|---|
| Rows | 2,000 |
| Columns | 22 |
| Distinct requirements | 642 (`requirement_id` — the CV grouping key) |
| Label | `label` — 1 = APPLICABLE, 0 = NOT_APPLICABLE |

14 of the 22 columns are the model's features and must stay identical, in
order, to `FEATURE_COLUMNS` in `src/ml/applicability_features.py`. The
remaining columns are identifiers (`requirement_id`, `standard_id`, `source`),
the label, and features carried in the dataset but deliberately excluded from
the model (`ics_code_match`, `standard_status_current`, `standard_age_years`,
`rrf_score_rank`) — currentness is handled deterministically by
`src/currentness.py`, not learned.

Metrics and the exact preprocessing/validation setup are recorded in
`standiq_applicability_model_v2_metadata.json` at the repository root.
