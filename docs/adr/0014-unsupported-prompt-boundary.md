# Unsupported Prompt Boundary

Status: Amended by current behavior and eval docs.

V1 refuses prompts that require local region edits, semantic object recoloring,
content generation/removal/replacement, geometry/detail changes, relighting,
reference-image style transfer, or selective preservation that cannot be
represented by one global LUT.

Mixed prompts are unsupported when any required component is unsupported. For
example, "make it warmer and remove the background" must produce
`<unsupported>`, not a partial warm LUT.

Vague but global style prompts such as cinematic, filmic, matte, or natural
remain supported as style bundles only when they can be mapped to calibrated,
measurable global color attributes and pass style-discriminability checks.
