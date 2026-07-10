# Source-Balanced Instruction Sampling

Status: Superseded by ADR 0015. Related: ADR 0026 (captioning-for-diversity data policy).

The older decision used a 50,000-example corpus with 30% PPR10K-derived LUTs and
25% FiveK-derived LUTs. That source mix has been superseded.

Current v1 caps PPR10K-derived and FiveK-derived examples at 15%-20% targets with
25% hard caps, and uses usage-prior buckets plus coverage-aware selection rather
than source balance alone.
