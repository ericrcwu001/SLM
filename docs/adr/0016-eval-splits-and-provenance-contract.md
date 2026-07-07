# Eval Splits And Provenance Contract

Status: Accepted.

The provenance registry must contain the fields needed by its own split,
selection, and eval rules. In addition to source and license metadata, rows must
record canonical LUT domain metadata, tokenizer/codebook/decoder hashes,
representability status, support-map identifiers, prompt template family,
teacher model/version/batch, paired input image identity, behavior-vector
version, quality-filter version, usage-prior bucket, headline eligibility, and
split unit ids.

Headline eval uses only `headline_eligible = true` rows with
`representability_tier = gold` and canonical-domain v1 metadata. Procedural
filler rows are train-only by default; if evaluated, they are diagnostic-only and
excluded from overall pass, supported pass, baseline deltas, and ship gates.

Required eval slices include usage-weighted headline, coverage macro,
subtle-control, style-discriminability, expert holdout, cross-source expert,
unseen-family, unsupported, mixed unsupported, boundary pairs, procedural
diagnostic, and qualitative demo.

Expert holdouts must keep held-out PPR10K/FiveK expert ids absent from active
SFT, with source images/groups also disjoint. The eval report must include
per-expert and macro-average results.

This ADR supersedes the older assumption that generic source-family holdouts are
sufficient by themselves.
