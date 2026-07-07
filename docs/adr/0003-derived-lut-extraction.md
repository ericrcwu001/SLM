# Derived LUT Extraction

Status: Amended.

PPR10K-derived expert LUTs are created first by parsing expert XMP metadata,
rejecting local/non-LUT tools, applying accepted edits to an identity color grid,
and converting the result into the v1 canonical LUT domain. Pair fitting is used
only as validation or fallback, and must not override an XMP local-tool
rejection.

FiveK-derived expert LUTs are created by fitting a global LUT from each source
image to its expert-retouched target after ICC-aware conversion into the
canonical domain, because FiveK is represented as before/after expert image
pairs rather than directly reusable edit presets.

All pair-fitted LUTs must pass held-out pixel checks, spatial residual checks,
and per-cell support-map gates before they can be used for tokenizer training,
SFT, or headline eval. Accepted artifacts are canonical display-referred encoded
sRGB 17x17x17 LUTs with trilinear interpolation metadata and representability
tier recorded.
