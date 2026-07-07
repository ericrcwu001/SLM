# Clean Implementation Over AceTone Fork

We will build the v1 prompt-to-LUT system as a clean implementation rather than forking AceTone. AceTone remains the architectural reference, but a clean codebase lets the project optimize for a smaller instruction-guided-only scope, 17^3 residual LUTs, project-specific evaluation, and one-week deliverability without inheriting unused reference-transfer, large-scale dataset, and infrastructure assumptions.
