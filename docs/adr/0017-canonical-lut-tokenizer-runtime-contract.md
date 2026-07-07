# Canonical LUT, Tokenizer, And Runtime Contract

Status: Accepted.

V1 uses one canonical LUT domain: display-referred IEC 61966-2-1 sRGB,
transfer-encoded RGB values in [0,1], D65, 17x17x17 grid, full-range RGB, and
trilinear interpolation. Accepted source LUTs are converted into this domain
before hashing, residual conversion, tokenizer encoding, prompt tagging, export,
or evaluation.

The tokenizer contract is 17x17x17 canonical residual LUTs to a 4x4x4 latent
grid, 64 token positions, and a 256-entry single-stage VQ codebook. The frozen
tokenizer manifest records axis order, `.cube` table order, latent flatten
order, token suffix to codebook-index mapping, `vq_codebook_sha256`,
`vq_decoder_sha256`, and the encoder/decoder geometry.

Canonical `.cube` serialization uses full canonical absolute RGB values,
`LUT_3D_SIZE 17`, `DOMAIN_MIN 0 0 0`, `DOMAIN_MAX 1 1 1`, RGB axis convention
with R changing fastest, fixed 10-decimal float formatting, LF line endings,
UTF-8, and no timestamps. Source images and LUTs with embedded or known
wide-gamut profiles are converted to canonical sRGB using the pinned ICC
conversion config before hashing, export, or evaluation.

The default v1 decision is to keep single-stage VQ and metric-order the exported
code ids after training. RVQ is a fallback only if single-stage VQ fails the
mean, tail, per-family, and per-target gates after EMA, augmentation, dead-code
revival, and targeted data filtering. Switching to RVQ changes the token grammar
and requires a new ADR.

Runtime/CLI decoding uses a token-id grammar mask/FSM. Free-generation eval
measures learned syntax validity separately. CLI and eval artifacts include a
version manifest that binds model, adapter, tokenizer ids, codebook, decoder,
flatten order, canonical domain, `.cube` serialization, ICC conversion config,
parser/FSM versions, safety thresholds, eval config, active/eval set versions,
and library versions. Startup fails on manifest mismatch, including codebook
value-hash mismatch.
