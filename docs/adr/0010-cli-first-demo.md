# CLI First Demo

Status: Amended.

The first runnable demo is a CLI that takes an image and instruction, generates
or refuses through grammar-constrained token-id decoding, decodes a canonical
global LUT when supported, applies it to the image, and writes the resulting
`.cube`, preview artifacts, `metrics.json`, and `version_manifest.json`.

The CLI must read embedded ICC profiles, convert inputs to canonical sRGB before
LUT application, use the same in-memory canonical LUT for `output.cube` and
`graded.png`, and validate the version manifest before inference. A Gradio or
application layer may be added after the core tokenizer, model, decoder, and
evaluation pipeline are working.
