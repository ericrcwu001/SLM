# Captioning-For-Diversity Data Policy And data/ Write Governance

Status: Accepted. Amends ADR 0013 (source-balanced instruction sampling). Related: ADR 0005
(instruction corpus scale, already superseded by ADR 0015).

Linguistic diversity is currently narrow and, worse, the pipeline discards the semantic metadata it
already has: LUT titles held in `RawArtifact.extra` are dropped by `to_registry_row`, and
scraped-web/pack tags are blanked before selection
(`docs/AUDIT_claude_codex_prompt_to_lut.md` F7). Meanwhile the generator trains on the terse
`instruction` field only, ignoring the richer `instruction_natural`.

Decision:

- **Diversity comes from captioning, not scraping.** The teacher is repointed to produce **many
  diverse captions per real corpus LUT** (literal, metaphor, mood, concept, slang), each mapped to
  that LUT's measured `AttributeSpec`. Every caption is therefore grounded in a renderable LUT — so
  the input language is unbounded while the target stays backable (ADR 0021). Recover the discarded
  LUT titles as caption seeds.
- **Which concepts are supported is defined empirically** by which corpus LUTs materialize within
  the admission gate; out-of-gamut concepts fall to `refuse` (ADR 0023).
- **`data/` write governance:** the LUT/image corpus, the frozen tokenizer, and `luts/` are
  immutable. Every artifact this migration produces — captioning corpus, `AttributeSpec` rows,
  `behavior_v2` vectors, portable unsupported staging — is a NEW, VERSIONED artifact (bumped
  `active_set_version` / behavior version), never an in-place mutation of the frozen corpus. Paths
  and versions are recorded in the manifests.

Consequences: diversity is decoupled from acquisition; the "corpus is immutable" rule
(`AGENTS.md:6`) is preserved because instructions/captions/specs are a regenerable derived layer
written to new versioned files.
