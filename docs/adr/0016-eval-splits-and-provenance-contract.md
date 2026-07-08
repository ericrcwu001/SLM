# Eval Splits And Provenance Contract

Status: Accepted.

The provenance registry must contain the fields needed by its own split,
selection, and eval rules. In addition to source metadata, rows must record
canonical LUT domain metadata, `vq_codebook_sha256`,
`vq_decoder_sha256`, tokenizer metadata,
representability status, support-map identifiers, prompt template family,
teacher model/version/batch, paired input image identity, behavior-vector
version, quality-filter version, usage-prior bucket, headline eligibility, and
split unit ids.

Headline eval uses only `headline_eligible = true` rows with
`representability_tier = gold` and canonical-domain v1 metadata. Procedural
filler rows are train-only by default; if evaluated, they are diagnostic-only and
excluded from overall pass, supported pass, baseline deltas, and ship gates.

Required eval slices include usage-weighted headline, coverage macro,
image-sensitivity, real-world CLI inputs, subtle-control,
style-discriminability, expert holdout, cross-source expert, unseen-family,
unsupported, mixed unsupported, boundary pairs, procedural diagnostic, and
qualitative demo.

Ship-gated slices must declare min_N/min_paired_N, MDE, CI method, and power
status before final eval freeze. Underpowered slices are diagnostic only and
cannot satisfy a ship gate.

Tokenizer and warmup data are train-only relative to final eval. No
`used_for_tokenizer` or `used_for_warmup` row may share exact or near-neighbor
LUT, image, source-pair, support-map, prompt-template, or split identity with
any final eval, diagnostic, or qualitative row.

Expert holdouts must keep held-out PPR10K/FiveK expert ids absent from active
SFT, with source images/groups also disjoint. The eval report must include
per-expert and macro-average results.

This ADR supersedes the older assumption that generic source-family holdouts are
sufficient by themselves.

If a source family is removed, the removal manifest must state whether tokenizer,
warmup, active SFT, eval, and downstream model/eval artifacts are invalidated.
Rows used in the tokenizer force tokenizer retraining and target retokenization;
rows used in eval force a new `eval_set_version`.
