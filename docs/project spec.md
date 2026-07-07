# Train Your Own Small Learning Model - Project Spec

> Status: External/course brief.
>
> This document states the original assignment constraints. It is not the
> authoritative prompt-to-LUT methodology. Use `docs/master_plan.md` and the
> linked implementation docs for current project decisions.

## Objective

Build a small fine-tuned learning model in one week by researching a target behavior, generating and filtering training data, fine-tuning a small open base model, and proving the fine-tune improves the target behavior.

The project is not intended to outperform a frontier model on general capability. The required win is reliable, constrained behavior in a small, cheap, local model.

## Core Principle

The dataset is the primary deliverable. Model quality is expected to come mainly from the generated and filtered training data. Training is a downstream step that turns the dataset into runnable behavior.

Training means supervised fine-tuning with QLoRA on a small open base model. Pretraining from scratch is out of scope.

## Target Behavior

The selected behavior is open, but it must pass the prompt test:

A well-prompted base model cannot already perform the behavior reliably.

Fine-tuning is justified only when reliability is the hard part. The model must perform the selected behavior consistently, without drifting, across relevant inputs.

## Behavior Spec

Before data generation or training begins, produce a falsifiable behavior spec.

The behavior spec must be one or two sentences that a stranger can use to mark any model output as pass or fail. It must define the target behavior clearly enough to serve as:

- The data-generation rubric
- The evaluation criterion
- The project thesis

Vague descriptions such as "a model that does X" are not acceptable.

## Required Work

### Data Generation

Generate hundreds to a few thousand examples (minimum) that embody the behavior spec. Distill examples from a frontier teacher model, then filter aggressively for quality.

The generation prompt and quality gate matter more than raw volume.

### Evaluation

Build the evaluation before training.

The minimum evaluation bar is:

- An LLM-as-judge scoring model outputs against the behavior spec
- A behavioral check for the specific failure the behavior spec forbids
- A base-vs-tuned comparison that makes the fine-tune effect visible in numbers

The project is not complete unless the tuned model measurably beats the base model on the target behavior.

## Success Criteria

The project succeeds when the tuned model demonstrates reliable target behavior that is difficult to obtain from prompting the base model alone.

The project must be framed as behavior from data, not as general intelligence or raw capability improvement.

Capability benchmarks unrelated to the target behavior are not success criteria.

## One-Week Plan

| Day | Focus | Actions | Checkpoint |
| --- | --- | --- | --- |
| 1 | Setup, research, and Brainlift | Get inference working. Research the behavior. Complete the Brainlift. | The base model runs and responds. Target behavior is known. Spiky POVs match the target behavior. |
| 2 | Spec, eval, and smoke test | Write the behavior spec. Build the eval harness and data-generation pipeline. Run 50 junk examples. | Full loop runs end to end: generate, train, evaluate. |
| 3 | v1 dataset and real numbers | Generate and filter real data. Run the first real training pass. Run the first base-vs-tuned evaluation. | Midweek base-vs-tuned numbers are on the board. |
| 4 | v2 dataset iteration | Diagnose failure modes. Fix failures in the data. Retrain and report improvements. | One specific failure mode is resolved through data iteration. |
| 5 | Ship and defend | Run final eval and error analysis. Ship inference demo. Write Brainlift and record demo. | Final submission package is ready. |

## Final Submission Package

The final submission must include:

1. Published dataset
2. Model published on Hugging Face Hub
3. Running inference demo
4. Eval harness and results table comparing base vs. tuned using the behavior metric
5. Brainlift covering the behavior thesis, whether data-to-behavior held, and supporting evidence
6. Three-to-five-minute demo video showing the tuned model performing the target behavior that the base model does not perform reliably

## Rules

- Pick a behavior that fails the prompt test.
- Do not train before the evaluation exists.
- Avoid broad domains. Use one target behavior in one context.
- Fix data problems with data iteration, not hyperparameter tuning.
- Do not chase capability benchmarks unrelated to the target behavior.

## Evaluation Rubric

Score each base and tuned model output with an LLM-as-judge. Report the delta.

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Spec adherence | Violates the target behavior | Partially follows | Fully embodies the spec |
| Robustness | Breaks on messy or adversarial input | Wobbles | Holds the behavior under pressure |
| Task quality | Output is wrong or useless | Acceptable | Genuinely good at the job |
| Consistency | Behaves differently across similar inputs | Mostly stable | Reliable every time |

Required reported outputs:

- Mean score per dimension for base and tuned models on the same held-out scenarios
- A short error-analysis paragraph describing where the tuned model still fails and whether the failure appears to be a data problem

A tuned model that beats the base model on spec adherence and robustness satisfies the core evaluation target.
