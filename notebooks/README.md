# notebooks/

Thin session drivers for remote runs - Kaggle, Colab, and other
notebook-shaped providers.

Rules:

- Notebooks only wrap `src/flygpt/` entry points (ingest, corpus,
  training, evals) plus checkpoint-resume logic tuned to each
  provider's session cap. Resume-from-checkpoint is the normal path,
  not a recovery path.
- No pipeline logic lives in notebooks. If a step cannot be expressed
  as a call into `src/flygpt/`, the step belongs in `src/flygpt/`
  first.
- The same code path must run on Kaggle, Vultr, GCP, or Modal - the
  notebook is a launcher, not an implementation.
