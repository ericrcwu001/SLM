# Teacher Prompt Quality Gates

Status: Accepted.

Teacher-generated instruction data will include both structured prompt tags and
natural-language prompts. Deterministic color-behavior checks are authoritative
for measurable tag claims, while an LLM/VLM judge is used as a language and
semantic quality gate for concision, unsupported local claims, content leakage,
and tag-prompt consistency.

Prompt generation and L8 judging require `configs/model_clients.yaml` profiles
named `teacher_primary` and `judge_primary`. Each profile must pin provider,
concrete `model_id` or deployment id, endpoint/base-url env var, API-key env
var, prompt version, and batch id. Model aliases such as `latest` are not
allowed. Secret values are never stored in data rows or manifests.
