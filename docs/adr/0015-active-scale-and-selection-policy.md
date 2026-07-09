# Active Scale And Selection Policy

Status: Accepted.

V1 uses an active instruction SFT set of 10k-15k examples, with 12k as the
default planning target. The older 50k/100k instruction corpus sizes are
scale-up milestones, not v1 requirements.

The project also adds a 30k-100k generative LUT-token warmup stage produced
after active/eval freeze from train-only accepted canonical LUT and image
identities. This warmup adapts the model to the new LUT-token distribution
before instruction SFT and is not pretraining from scratch.

Source mix targets for the active supported set:

- PPR10K-derived: 15%-20%, 25% hard cap.
- FiveK-derived: 15%-20%, 25% hard cap.
- Fresh LUTs: 15%-20%.
- G'MIC / RawTherapee: 20%-25%.
- Smaller public packs: 10%-15%, 20% hard cap.
- Scraped web LUTs: uncapped in selection (a primary diversity/scale source), bounded only by
  the global 50% no-dominance gate; behavior-only, no authored tags. Add a per-family cap only
  if it begins to dominate the mix.
- Controlled/procedural fillers: 0%-10%, train-only by default.

Per-family caps bind against the *realized* active size (`min(target_size, pool)`), so they
still hold below the target scale. The combined ppr10k+fivek expert-source ceiling (50%) is a
soft target for v1: it is reported but waived when marginally exceeded, since the per-family
caps already prevent single-source dominance.

Selection blends usage prior and coverage. The active set is allocated by rough
usage-prior buckets, then selected inside buckets with facility-location/MMR and
source/style/scene quotas. A bounded coverage-tail budget keeps rare styles and
edge cases without letting outliers dominate. HDBSCAN noise is not used for
seeding unless manually approved.

This ADR supersedes ADR 0005 and ADR 0013, and partially supersedes ADR 0009.
