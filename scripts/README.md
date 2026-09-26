# scripts/

This directory is the GPL-tolerant zone.

Data-prep tools that depend on GPL-3.0 packages - natverse, bancr,
fafbseg, navis, and friends - may live here, and only here. Rules:

- Nothing in `src/flygpt/` may import anything in `scripts/`.
- Nothing in `pyproject.toml` may declare a GPL dependency.
- Anything in this directory may carry GPL-3.0 contamination; it ships
  separately from the Apache-2.0 package and is not part of the
  `flygpt` distribution.
- Prefer reimplementing needed queries on neuprint-python + pyarrow
  (Apache-friendly) over adding a new GPL tool.
