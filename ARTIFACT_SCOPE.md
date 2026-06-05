# Public Artifact Scope

This directory is generated from the broader local research workspace by:

```bash
python papers/h1_activation_transfer/scripts/stage_github_paper_repo.py
```

The goal is a self-contained public code-and-data artifact for the arXiv paper,
without the full upstream research project and without the manuscript sources.

## Included

- experiment runner and supporting package code
- deterministic clean-eval builder and clean evaluation data
- final result JSON files and derived summaries
- strict matched shuffled controls
- metric, validation, comparison, and figure scripts

## Excluded

- manuscript sources and workshop artifacts (the paper is on arXiv; sources stay
  in the private repository)
- model cache directories
- `.venv`, Python caches, pytest/mypy caches
- PyTorch checkpoints and activation tensors
- raw WikiText corpus data
- local browser HTML previews with browser-local absolute image paths
- historical planning logs outside the paper workspace
- generated source zip archives, which can be rebuilt from source

## Known Residuals

- The Python import package remains `rosetta` for compatibility with the
  experiment code.
- Some raw model predictions contain arbitrary generated URLs, email-like
  strings, names, and GitHub-looking links. These are model outputs, not
  repository, author, or contact metadata.
