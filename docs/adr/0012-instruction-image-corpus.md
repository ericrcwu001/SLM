# Instruction Image Corpus

The v1 instruction-pair image corpus uses a mixed image corpus weighted toward
photographic quality, to avoid overfitting the prompt-to-LUT model to portraits
or curated photos while keeping the task on photographic global color grading.

Per data_collection_plan.md "Input Image Mix":

- broad photo images (Unsplash-like sources): 60%-70%;
- COCO/OpenImages-style diverse scenes: 20%-30%;
- PPR10K/FiveK source photos as model inputs: <=10%-15% (capped).

PPR10K and FiveK stay the primary sources for deriving expert LUT targets, but
their source photos are capped as instruction inputs so the input distribution
stays broad and photographic rather than curated or portrait-heavy.
